"""
Match Analyzer
==============

Main orchestrator for analyzing football matches and extracting player metrics.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Generator
from dataclasses import dataclass, field
from loguru import logger
from tqdm import tqdm
import json
import uuid

from src.detection import FootballDetector, TeamClassifier, JerseyOCR, Detection, FrameDetections
from src.scout.player_profile import PlayerProfile, PlayerMetrics, Position, Action


@dataclass
class MatchInfo:
    """Match metadata."""
    match_id: str
    home_team: str
    away_team: str
    scout_team: str  # Team being scouted
    competition: str = ""
    date: str = ""
    video_path: str = ""
    fps: float = 25.0
    total_frames: int = 0
    duration_seconds: float = 0.0


@dataclass
class MatchAnalysis:
    """Complete match analysis results."""
    match_info: MatchInfo
    players: Dict[int, PlayerProfile] = field(default_factory=dict)  # track_id -> profile
    
    def get_top_players(
        self,
        limit: int = 5,
        team: Optional[str] = None
    ) -> List[PlayerProfile]:
        """Get top players by scout score."""
        players = list(self.players.values())
        
        if team:
            players = [p for p in players if p.team == team]
        
        # Calculate scores if not done
        for p in players:
            if p.scout_score == 0:
                p.calculate_scout_score()
        
        # Sort by score
        players.sort(key=lambda p: p.scout_score, reverse=True)
        
        return players[:limit]
    
    def get_player(self, jersey_number: int) -> Optional[PlayerProfile]:
        """Get player by jersey number."""
        for player in self.players.values():
            if player.jersey_number == jersey_number:
                return player
        return None
    
    def get_player_by_track(self, track_id: int) -> Optional[PlayerProfile]:
        """Get player by track ID."""
        return self.players.get(track_id)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "matchInfo": {
                "matchId": self.match_info.match_id,
                "homeTeam": self.match_info.home_team,
                "awayTeam": self.match_info.away_team,
                "scoutTeam": self.match_info.scout_team,
                "competition": self.match_info.competition,
                "date": self.match_info.date,
                "duration": self.match_info.duration_seconds
            },
            "players": [p.to_dict() for p in self.players.values()]
        }
    
    def save(self, path: str) -> None:
        """Save analysis to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class MatchAnalyzer:
    """
    Main class for analyzing football matches.
    
    Orchestrates detection, tracking, team classification, jersey OCR,
    and metrics computation to produce complete player profiles.
    
    Example:
        analyzer = MatchAnalyzer()
        analysis = analyzer.analyze(
            video_path="match.mp4",
            home_team="FC Porto",
            away_team="SL Benfica",
            scout_team="FC Porto"
        )
        
        for player in analysis.get_top_players(5):
            print(f"#{player.jersey_number}: {player.scout_score:.1f}")
    """
    
    def __init__(
        self,
        model_path: str = "models/yolo26_football.pt",
        enable_gpu: bool = True,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the analyzer.
        
        Args:
            model_path: Path to YOLO model
            enable_gpu: Whether to use GPU
            confidence_threshold: Detection confidence threshold
        """
        self.model_path = model_path
        self.device = "cuda" if enable_gpu else "cpu"
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        self.detector = FootballDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=self.device
        )
        self.team_classifier = TeamClassifier()
        self.jersey_ocr = JerseyOCR(use_gpu=enable_gpu)
        
        # Tracking state
        self.players: Dict[int, PlayerProfile] = {}
        self.frame_positions: Dict[int, List[Tuple[float, float]]] = {}  # track_id -> positions
        
    def analyze(
        self,
        video_path: str,
        home_team: str,
        away_team: str,
        scout_team: str,
        competition: str = "",
        date: str = "",
        progress_callback: Optional[callable] = None
    ) -> MatchAnalysis:
        """
        Analyze a complete match video.
        
        Args:
            video_path: Path to match video
            home_team: Name of home team
            away_team: Name of away team  
            scout_team: Team to analyze ("home" or team name)
            competition: Competition name
            date: Match date
            progress_callback: Optional callback for progress updates
            
        Returns:
            MatchAnalysis with all player profiles
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Create match info
        match_info = MatchInfo(
            match_id=str(uuid.uuid4())[:8],
            home_team=home_team,
            away_team=away_team,
            scout_team=scout_team,
            competition=competition,
            date=date,
            video_path=str(video_path),
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration
        )
        
        logger.info(f"Analyzing match: {home_team} vs {away_team}")
        logger.info(f"Video: {total_frames} frames @ {fps} FPS ({duration/60:.1f} min)")
        
        # Reset state
        self.players = {}
        self.frame_positions = {}
        
        # Process video
        frame_id = 0
        team_classifier_fitted = False
        
        with tqdm(total=total_frames, desc="Analyzing") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip some frames for speed (process every Nth frame)
                if frame_id % 2 != 0:  # Process every 2nd frame
                    frame_id += 1
                    pbar.update(1)
                    continue
                
                # Run detection with tracking
                detections = self.detector.detect_with_tracking(frame, frame_id)
                
                # Fit team classifier on first good frame
                if not team_classifier_fitted and len(detections.players) >= 6:
                    self.team_classifier.fit(frame, detections.players)
                    team_classifier_fitted = True
                
                # Process detections
                self._process_frame(
                    frame=frame,
                    frame_id=frame_id,
                    detections=detections,
                    fps=fps,
                    scout_team=scout_team
                )
                
                frame_id += 1
                pbar.update(1)
                
                if progress_callback:
                    progress_callback(frame_id / total_frames)
        
        cap.release()
        
        # Finalize player profiles
        self._finalize_profiles(match_info)
        
        # Create analysis result
        analysis = MatchAnalysis(
            match_info=match_info,
            players=self.players
        )
        
        logger.info(f"Analysis complete: {len(self.players)} players detected")
        
        return analysis
    
    def _process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        detections: FrameDetections,
        fps: float,
        scout_team: str
    ) -> None:
        """Process a single frame."""
        # Classify teams
        if self.team_classifier.fitted:
            team_assignments = self.team_classifier.classify(frame, detections.players)
        else:
            team_assignments = []
        
        # Process each player detection
        for i, det in enumerate(detections.players):
            if det.track_id is None:
                continue
            
            track_id = det.track_id
            
            # Get or create player profile
            if track_id not in self.players:
                self.players[track_id] = PlayerProfile(
                    player_id=f"player_{track_id}",
                    track_id=track_id
                )
                self.frame_positions[track_id] = []
            
            player = self.players[track_id]
            
            # Update team assignment
            if team_assignments and i < len(team_assignments):
                assignment = team_assignments[i]
                player.team = assignment.team
            
            # Try to detect jersey number
            if player.jersey_number is None:
                jersey = self.jersey_ocr.detect(frame, det)
                if jersey:
                    player.jersey_number = jersey.number
            
            # Record position
            center = det.center
            self.frame_positions[track_id].append((center[0], center[1], frame_id))
            player.position_history.append(center)
    
    def _finalize_profiles(self, match_info: MatchInfo) -> None:
        """Finalize all player profiles after processing."""
        fps = match_info.fps
        
        for track_id, player in self.players.items():
            positions = self.frame_positions.get(track_id, [])
            
            if len(positions) < 10:
                continue  # Skip players with too few detections
            
            # Calculate minutes played (approximate)
            if positions:
                first_frame = positions[0][2]
                last_frame = positions[-1][2]
                player.minutes_played = (last_frame - first_frame) / fps / 60
            
            # Calculate physical metrics
            self._calculate_physical_metrics(player, positions, fps)
            
            # Detect position from heatmap
            self._detect_position(player)
            
            # Calculate scout score
            player.calculate_scout_score()
            
            # Analyze strengths/weaknesses
            player.analyze_strengths_weaknesses()
            
            # Set match ID
            player.match_id = match_info.match_id
    
    def _calculate_physical_metrics(
        self,
        player: PlayerProfile,
        positions: List[Tuple[float, float, int]],
        fps: float
    ) -> None:
        """Calculate physical metrics from position history."""
        if len(positions) < 2:
            return
        
        physical = player.metrics.physical
        
        speeds = []
        total_distance = 0.0
        sprint_count = 0
        high_intensity_count = 0
        
        # Assume pitch is 105m x 68m, frame is ~1920x1080
        # This is a rough conversion - in production, use homography
        pixels_per_meter_x = 1920 / 105
        pixels_per_meter_y = 1080 / 68
        
        prev_pos = None
        prev_frame = None
        
        for x, y, frame_id in positions:
            if prev_pos is not None and prev_frame is not None:
                # Calculate displacement
                dx = (x - prev_pos[0]) / pixels_per_meter_x
                dy = (y - prev_pos[1]) / pixels_per_meter_y
                distance = np.sqrt(dx**2 + dy**2)
                
                # Calculate time delta
                dt = (frame_id - prev_frame) / fps
                
                if dt > 0:
                    # Speed in m/s, convert to km/h
                    speed = (distance / dt) * 3.6
                    
                    # Filter unrealistic speeds (noise)
                    if speed < 40:  # Max realistic speed
                        speeds.append(speed)
                        total_distance += distance
                        
                        if speed > 25:
                            sprint_count += 1
                        elif speed > 21:
                            high_intensity_count += 1
            
            prev_pos = (x, y)
            prev_frame = frame_id
        
        if speeds:
            physical.top_speed = max(speeds)
            physical.avg_speed = np.mean(speeds)
            physical.total_distance = total_distance / 1000  # Convert to km
            physical.sprint_count = sprint_count
            physical.high_intensity_runs = high_intensity_count
    
    def _detect_position(self, player: PlayerProfile) -> None:
        """Detect player's position from movement patterns."""
        if not player.position_history:
            return
        
        positions = np.array(player.position_history)
        
        # Calculate average position
        avg_x = np.mean(positions[:, 0])
        avg_y = np.mean(positions[:, 1])
        
        # Normalize to 0-1 (assuming 1920x1080 frame)
        norm_x = avg_x / 1920
        norm_y = avg_y / 1080
        
        player.metrics.intelligence.avg_position = (norm_x, norm_y)
        player.metrics.intelligence.position_variance = np.var(positions[:, 0]) + np.var(positions[:, 1])
        
        # Heuristic position detection based on average position
        # This is simplified - in production, use ML classifier
        
        # Y position determines line (defense/midfield/attack)
        # X position determines side (left/center/right)
        
        if norm_y < 0.25:  # Defensive third
            if norm_x < 0.35:
                player.detected_position = Position.LB
            elif norm_x > 0.65:
                player.detected_position = Position.RB
            else:
                player.detected_position = Position.CB
        elif norm_y < 0.5:  # Defensive midfield
            if norm_x < 0.35:
                player.detected_position = Position.LM
            elif norm_x > 0.65:
                player.detected_position = Position.RM
            else:
                player.detected_position = Position.CDM
        elif norm_y < 0.75:  # Attacking midfield
            if norm_x < 0.35:
                player.detected_position = Position.LW
            elif norm_x > 0.65:
                player.detected_position = Position.RW
            else:
                player.detected_position = Position.CAM
        else:  # Attacking third
            player.detected_position = Position.ST
        
        player.position_confidence = 0.7  # Default confidence
    
    def analyze_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0
    ) -> FrameDetections:
        """
        Analyze a single frame (for real-time use).
        
        Args:
            frame: BGR image
            frame_id: Frame number
            
        Returns:
            FrameDetections with all detections
        """
        return self.detector.detect_with_tracking(frame, frame_id)
