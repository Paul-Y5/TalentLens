"""
Multi-Object Tracker
====================

Tracks multiple players across frames using ByteTrack/BoT-SORT.
"""
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class Track:
    """A single object track."""
    track_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    
    # Kalman filter state
    state: Optional[np.ndarray] = None
    
    # Track lifecycle
    hits: int = 0
    time_since_update: int = 0
    age: int = 0
    
    # Velocity estimates
    velocity: Tuple[float, float] = (0, 0)
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get bounding box center."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )
    
    @property
    def area(self) -> float:
        """Get bounding box area."""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


@dataclass
class TrackingConfig:
    """Configuration for tracker."""
    # Detection thresholds
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    
    # Matching thresholds
    match_thresh: float = 0.8
    
    # Track lifecycle
    track_buffer: int = 30  # Frames to keep lost tracks
    min_hits: int = 3  # Min detections before confirmed
    
    # Motion model
    use_kalman: bool = True


class KalmanFilter:
    """
    Simple Kalman Filter for 2D tracking.
    
    State: [x, y, vx, vy, w, h]
    """
    
    def __init__(self):
        # State transition matrix
        self.F = np.array([
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Observation matrix
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Process noise
        self.Q = np.eye(6, dtype=np.float32) * 0.1
        
        # Measurement noise
        self.R = np.eye(4, dtype=np.float32) * 1.0
        
        # State covariance
        self.P = np.eye(6, dtype=np.float32) * 10.0
    
    def init_state(self, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """Initialize state from bounding box."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        return np.array([cx, cy, 0, 0, w, h], dtype=np.float32)
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict next state."""
        return self.F @ state
    
    def update(self, state: np.ndarray, measurement: np.ndarray) -> np.ndarray:
        """Update state with measurement."""
        # Predicted measurement
        z_pred = self.H @ state
        
        # Innovation
        y = measurement - z_pred
        
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        state = state + K @ y
        
        # Update covariance
        self.P = (np.eye(6) - K @ self.H) @ self.P
        
        return state
    
    def state_to_bbox(self, state: np.ndarray) -> Tuple[float, float, float, float]:
        """Convert state to bounding box."""
        cx, cy, vx, vy, w, h = state
        return (cx - w/2, cy - h/2, cx + w/2, cy + h/2)


class MultiObjectTracker:
    """
    Multi-object tracker using ByteTrack algorithm.
    
    Tracks players, ball, and referees across video frames.
    
    Features:
    - Handles occlusions
    - Re-identifies players after leaving frame
    - Estimates velocity
    - Predicts positions during missed detections
    
    Example:
        tracker = MultiObjectTracker()
        
        for frame in video:
            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            
            for track in tracks:
                draw_bbox(frame, track.bbox, track.track_id)
    """
    
    def __init__(self, config: Optional[TrackingConfig] = None):
        """
        Initialize tracker.
        
        Args:
            config: Tracking configuration
        """
        self.config = config or TrackingConfig()
        
        self.tracks: Dict[int, Track] = {}
        self.lost_tracks: Dict[int, Track] = {}
        self.removed_tracks: List[Track] = []
        
        self.frame_count = 0
        self._next_id = 1
        
        self.kalman = KalmanFilter() if self.config.use_kalman else None
    
    def update(
        self,
        detections: List[Dict],
        frame_id: Optional[int] = None
    ) -> List[Track]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detection dicts with keys:
                - bbox: (x1, y1, x2, y2)
                - confidence: float
                - class_id: int
            frame_id: Optional frame number
            
        Returns:
            List of active tracks
        """
        self.frame_count = frame_id if frame_id is not None else self.frame_count + 1
        
        # No detections - predict all tracks
        if not detections:
            return self._predict_all()
        
        # Split detections by confidence
        high_dets = [d for d in detections if d["confidence"] >= self.config.track_high_thresh]
        low_dets = [d for d in detections if self.config.track_low_thresh <= d["confidence"] < self.config.track_high_thresh]
        
        # First association with high confidence detections
        matched, unmatched_tracks, unmatched_dets = self._associate(
            list(self.tracks.values()),
            high_dets
        )
        
        # Update matched tracks
        for track_id, det_idx in matched:
            self._update_track(self.tracks[track_id], high_dets[det_idx])
        
        # Second association with low confidence detections
        remaining_tracks = [self.tracks[tid] for tid in unmatched_tracks]
        matched_low, still_unmatched, _ = self._associate(remaining_tracks, low_dets)
        
        for i, (track_id, det_idx) in enumerate(matched_low):
            # Use original track IDs
            original_track = remaining_tracks[track_id]
            self._update_track(original_track, low_dets[det_idx])
            unmatched_tracks.remove(original_track.track_id)
        
        # Mark unmatched tracks as lost
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            track.time_since_update += 1
            
            if track.time_since_update > self.config.track_buffer:
                self._remove_track(track_id)
            elif self.kalman:
                # Predict position
                track.state = self.kalman.predict(track.state)
                track.bbox = self.kalman.state_to_bbox(track.state)
        
        # Create new tracks for unmatched high confidence detections
        for det_idx in unmatched_dets:
            det = high_dets[det_idx]
            if det["confidence"] >= self.config.new_track_thresh:
                self._create_track(det)
        
        # Return confirmed tracks
        return [t for t in self.tracks.values() if t.hits >= self.config.min_hits]
    
    def _associate(
        self,
        tracks: List[Track],
        detections: List[Dict]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate tracks with detections using IoU.
        
        Returns:
            - matched: List of (track_id, det_idx) pairs
            - unmatched_tracks: List of track IDs
            - unmatched_dets: List of detection indices
        """
        if not tracks or not detections:
            return [], [t.track_id for t in tracks], list(range(len(detections)))
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._iou(track.bbox, det["bbox"])
        
        # Hungarian algorithm for optimal assignment
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        except ImportError:
            # Greedy fallback
            row_ind, col_ind = self._greedy_assignment(iou_matrix)
        
        matched = []
        unmatched_tracks = set(t.track_id for t in tracks)
        unmatched_dets = set(range(len(detections)))
        
        for i, j in zip(row_ind, col_ind):
            if iou_matrix[i, j] >= self.config.match_thresh:
                matched.append((tracks[i].track_id, j))
                unmatched_tracks.discard(tracks[i].track_id)
                unmatched_dets.discard(j)
        
        return matched, list(unmatched_tracks), list(unmatched_dets)
    
    def _iou(
        self,
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float]
    ) -> float:
        """Compute IoU between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _greedy_assignment(self, cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Greedy assignment as fallback when scipy not available."""
        rows, cols = [], []
        used_rows = set()
        used_cols = set()
        
        # Sort all pairs by cost (highest first for IoU)
        indices = np.argsort(cost_matrix.ravel())[::-1]
        
        for idx in indices:
            i, j = divmod(idx, cost_matrix.shape[1])
            if i not in used_rows and j not in used_cols:
                rows.append(i)
                cols.append(j)
                used_rows.add(i)
                used_cols.add(j)
        
        return np.array(rows), np.array(cols)
    
    def _update_track(self, track: Track, detection: Dict) -> None:
        """Update a track with a new detection."""
        track.bbox = detection["bbox"]
        track.confidence = detection["confidence"]
        track.hits += 1
        track.time_since_update = 0
        track.age += 1
        
        if self.kalman and track.state is not None:
            # Extract measurement
            x1, y1, x2, y2 = detection["bbox"]
            measurement = np.array([
                (x1 + x2) / 2,  # cx
                (y1 + y2) / 2,  # cy
                x2 - x1,        # w
                y2 - y1         # h
            ], dtype=np.float32)
            
            # Update Kalman filter
            track.state = self.kalman.update(track.state, measurement)
            
            # Extract velocity
            track.velocity = (track.state[2], track.state[3])
    
    def _create_track(self, detection: Dict) -> Track:
        """Create a new track."""
        track = Track(
            track_id=self._next_id,
            bbox=detection["bbox"],
            confidence=detection["confidence"],
            class_id=detection.get("class_id", 0),
            hits=1,
            time_since_update=0,
            age=1
        )
        
        if self.kalman:
            track.state = self.kalman.init_state(detection["bbox"])
        
        self.tracks[track.track_id] = track
        self._next_id += 1
        
        return track
    
    def _remove_track(self, track_id: int) -> None:
        """Remove a track."""
        if track_id in self.tracks:
            track = self.tracks.pop(track_id)
            self.removed_tracks.append(track)
    
    def _predict_all(self) -> List[Track]:
        """Predict all tracks when no detections."""
        for track in self.tracks.values():
            track.time_since_update += 1
            
            if self.kalman and track.state is not None:
                track.state = self.kalman.predict(track.state)
                track.bbox = self.kalman.state_to_bbox(track.state)
            
            if track.time_since_update > self.config.track_buffer:
                self._remove_track(track.track_id)
        
        return [t for t in self.tracks.values() if t.hits >= self.config.min_hits]
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.tracks.clear()
        self.lost_tracks.clear()
        self.removed_tracks.clear()
        self.frame_count = 0
        self._next_id = 1
