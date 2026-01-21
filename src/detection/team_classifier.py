"""
Team Classifier
===============

Classifies players into teams based on jersey color using K-Means clustering
and optional CNN-based classification.
"""
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import KMeans
from collections import Counter
from loguru import logger

from src.detection.detector import Detection


@dataclass
class TeamAssignment:
    """Team assignment for a player."""
    player_id: int
    team: str  # "home", "away", "referee", "unknown"
    confidence: float
    dominant_color: Tuple[int, int, int]  # BGR


class TeamClassifier:
    """
    Classifies players into teams based on jersey colors.
    
    Uses K-Means clustering to identify dominant colors in player
    bounding boxes, then assigns teams based on color similarity.
    
    Example:
        classifier = TeamClassifier()
        classifier.fit(frame, detections)  # Learn team colors
        assignments = classifier.classify(frame, detections)
    """
    
    def __init__(
        self,
        n_clusters: int = 3,  # 2 teams + 1 for referee/goalkeeper
        color_space: str = "hsv",  # "rgb", "hsv", "lab"
        exclude_ratio: float = 0.3  # Exclude top/bottom of bbox
    ):
        """
        Initialize the classifier.
        
        Args:
            n_clusters: Number of color clusters (usually 3: home, away, ref)
            color_space: Color space for clustering
            exclude_ratio: Ratio of bbox to exclude (for face/shorts)
        """
        self.n_clusters = n_clusters
        self.color_space = color_space
        self.exclude_ratio = exclude_ratio
        
        self.kmeans: Optional[KMeans] = None
        self.team_colors: Dict[str, np.ndarray] = {}
        self.fitted = False
    
    def _extract_jersey_region(
        self,
        frame: np.ndarray,
        detection: Detection
    ) -> np.ndarray:
        """Extract the jersey region from a player detection."""
        x1, y1, x2, y2 = map(int, detection.bbox)
        
        # Ensure within frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Get player crop
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            return np.array([])
        
        # Exclude top (head) and bottom (shorts/legs)
        crop_h = crop.shape[0]
        top_exclude = int(crop_h * self.exclude_ratio)
        bottom_exclude = int(crop_h * (1 - self.exclude_ratio))
        
        jersey_region = crop[top_exclude:bottom_exclude, :]
        
        return jersey_region
    
    def _get_dominant_color(self, region: np.ndarray) -> np.ndarray:
        """Get dominant color from a region using K-Means."""
        if region.size == 0:
            return np.array([0, 0, 0])
        
        # Convert color space if needed
        if self.color_space == "hsv":
            region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        elif self.color_space == "lab":
            region = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
        
        # Reshape to list of pixels
        pixels = region.reshape(-1, 3).astype(np.float32)
        
        if len(pixels) < 10:
            return np.array([0, 0, 0])
        
        # Use K-Means to find dominant color
        kmeans = KMeans(n_clusters=1, n_init=10, random_state=42)
        kmeans.fit(pixels)
        
        dominant = kmeans.cluster_centers_[0]
        
        return dominant.astype(np.uint8)
    
    def fit(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        home_hint: Optional[Tuple[int, int, int]] = None,
        away_hint: Optional[Tuple[int, int, int]] = None
    ) -> None:
        """
        Learn team colors from frame and detections.
        
        Args:
            frame: BGR image
            detections: List of player detections
            home_hint: Optional hint for home team color (BGR)
            away_hint: Optional hint for away team color (BGR)
        """
        colors = []
        
        for det in detections:
            if det.class_name != "player":
                continue
                
            region = self._extract_jersey_region(frame, det)
            if region.size > 0:
                color = self._get_dominant_color(region)
                colors.append(color)
        
        if len(colors) < self.n_clusters:
            logger.warning(f"Not enough players to fit ({len(colors)} < {self.n_clusters})")
            return
        
        colors = np.array(colors)
        
        # Fit K-Means
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        self.kmeans.fit(colors)
        
        # Assign cluster IDs to teams
        centers = self.kmeans.cluster_centers_
        
        if home_hint is not None and away_hint is not None:
            # Use hints to assign teams
            home_hint = np.array(home_hint)
            away_hint = np.array(away_hint)
            
            home_idx = np.argmin([np.linalg.norm(c - home_hint) for c in centers])
            away_idx = np.argmin([np.linalg.norm(c - away_hint) for c in centers])
            
            self.team_colors = {
                "home": centers[home_idx],
                "away": centers[away_idx]
            }
        else:
            # Assume two largest clusters are teams
            labels = self.kmeans.labels_
            label_counts = Counter(labels)
            sorted_labels = [l for l, _ in label_counts.most_common()]
            
            if len(sorted_labels) >= 2:
                self.team_colors = {
                    "home": centers[sorted_labels[0]],
                    "away": centers[sorted_labels[1]]
                }
                if len(sorted_labels) >= 3:
                    self.team_colors["referee"] = centers[sorted_labels[2]]
        
        self.fitted = True
        logger.info(f"Team classifier fitted with {len(self.team_colors)} teams")
    
    def classify(
        self,
        frame: np.ndarray,
        detections: List[Detection]
    ) -> List[TeamAssignment]:
        """
        Classify players into teams.
        
        Args:
            frame: BGR image
            detections: List of player detections
            
        Returns:
            List of TeamAssignment objects
        """
        if not self.fitted:
            logger.warning("Classifier not fitted, returning unknown for all")
            return [
                TeamAssignment(
                    player_id=i,
                    team="unknown",
                    confidence=0.0,
                    dominant_color=(0, 0, 0)
                )
                for i in range(len(detections))
            ]
        
        assignments = []
        
        for i, det in enumerate(detections):
            if det.class_name == "referee":
                assignments.append(TeamAssignment(
                    player_id=det.track_id or i,
                    team="referee",
                    confidence=1.0,
                    dominant_color=(0, 0, 0)
                ))
                continue
            
            region = self._extract_jersey_region(frame, det)
            if region.size == 0:
                assignments.append(TeamAssignment(
                    player_id=det.track_id or i,
                    team="unknown",
                    confidence=0.0,
                    dominant_color=(0, 0, 0)
                ))
                continue
            
            color = self._get_dominant_color(region)
            
            # Find closest team color
            min_dist = float("inf")
            assigned_team = "unknown"
            
            for team, team_color in self.team_colors.items():
                dist = np.linalg.norm(color - team_color)
                if dist < min_dist:
                    min_dist = dist
                    assigned_team = team
            
            # Calculate confidence (inverse of distance, normalized)
            max_dist = 255 * np.sqrt(3)  # Max possible distance in RGB
            confidence = 1.0 - (min_dist / max_dist)
            
            assignments.append(TeamAssignment(
                player_id=det.track_id or i,
                team=assigned_team,
                confidence=confidence,
                dominant_color=tuple(map(int, color))
            ))
        
        return assignments
    
    def classify_single(
        self,
        frame: np.ndarray,
        detection: Detection
    ) -> TeamAssignment:
        """Classify a single detection."""
        results = self.classify(frame, [detection])
        return results[0] if results else TeamAssignment(
            player_id=0,
            team="unknown",
            confidence=0.0,
            dominant_color=(0, 0, 0)
        )
