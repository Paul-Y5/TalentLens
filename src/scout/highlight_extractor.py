"""
Highlight Extractor
==================

Extracts highlight clips from match video based on player actions.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import uuid
from loguru import logger

from src.scout.player_profile import PlayerProfile, Highlight, Action


@dataclass
class ClipConfig:
    """Configuration for clip extraction."""
    pre_action_seconds: float = 3.0
    post_action_seconds: float = 2.0
    min_duration: float = 3.0
    max_duration: float = 15.0
    output_fps: int = 30
    output_resolution: Tuple[int, int] = (1280, 720)
    output_format: str = "mp4"
    codec: str = "mp4v"


class HighlightExtractor:
    """
    Extracts highlight clips from match video.
    
    Features:
    - Extract clips for specific actions (goals, assists, dribbles, etc.)
    - Create player compilation videos
    - Add stats overlay to clips
    - Generate thumbnails
    
    Example:
        extractor = HighlightExtractor(analysis)
        
        # Get highlights for a player
        highlights = extractor.get_player_highlights(
            jersey_number=10,
            actions=["goal", "assist", "dribble"]
        )
        
        # Create compilation
        extractor.create_compilation(
            player_jersey=10,
            output_path="highlights/player_10.mp4"
        )
    """
    
    # Action importance scores for ranking
    ACTION_IMPORTANCE = {
        "goal": 10.0,
        "assist": 9.0,
        "shot_on_target": 7.0,
        "shot": 5.0,
        "key_pass": 7.0,
        "dribble": 6.0,
        "tackle": 5.0,
        "interception": 5.0,
        "save": 8.0,
        "clearance": 4.0,
        "cross": 5.0,
        "header": 5.0,
        "foul_won": 4.0
    }
    
    def __init__(
        self,
        analysis: "MatchAnalysis",  # Forward reference
        config: Optional[ClipConfig] = None
    ):
        """
        Initialize the extractor.
        
        Args:
            analysis: MatchAnalysis object with player data
            config: Clip extraction configuration
        """
        self.analysis = analysis
        self.config = config or ClipConfig()
        self.video_path = Path(analysis.match_info.video_path)
        self.fps = analysis.match_info.fps
    
    def get_player_highlights(
        self,
        jersey_number: Optional[int] = None,
        track_id: Optional[int] = None,
        actions: Optional[List[str]] = None,
        max_clips: int = 10,
        min_importance: float = 0.0
    ) -> List[Highlight]:
        """
        Get highlights for a specific player.
        
        Args:
            jersey_number: Player jersey number
            track_id: Player track ID
            actions: List of action types to include (None = all)
            max_clips: Maximum number of clips to return
            min_importance: Minimum importance score
            
        Returns:
            List of Highlight objects
        """
        # Find player
        player = None
        if jersey_number is not None:
            player = self.analysis.get_player(jersey_number)
        elif track_id is not None:
            player = self.analysis.get_player_by_track(track_id)
        
        if player is None:
            logger.warning(f"Player not found: jersey={jersey_number}, track={track_id}")
            return []
        
        # Filter actions
        player_actions = player.actions
        if actions:
            player_actions = [a for a in player_actions if a.action_type in actions]
        
        # Convert to highlights with importance scores
        highlights = []
        for action in player_actions:
            importance = self.ACTION_IMPORTANCE.get(action.action_type, 5.0)
            
            if importance < min_importance:
                continue
            
            # Adjust importance based on success
            if action.success:
                importance *= 1.2
            
            # Calculate clip timestamps
            timestamp_start = max(0, action.timestamp - self.config.pre_action_seconds)
            timestamp_end = action.timestamp + self.config.post_action_seconds
            
            # Ensure within duration limits
            duration = timestamp_end - timestamp_start
            if duration < self.config.min_duration:
                timestamp_end = timestamp_start + self.config.min_duration
            elif duration > self.config.max_duration:
                timestamp_end = timestamp_start + self.config.max_duration
            
            highlight = Highlight(
                clip_id=str(uuid.uuid4())[:8],
                action_type=action.action_type,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
                frame_start=int(timestamp_start * self.fps),
                frame_end=int(timestamp_end * self.fps),
                importance_score=importance
            )
            highlights.append(highlight)
        
        # Sort by importance and limit
        highlights.sort(key=lambda h: h.importance_score, reverse=True)
        return highlights[:max_clips]
    
    def extract_clip(
        self,
        highlight: Highlight,
        output_path: str,
        include_stats: bool = False,
        player: Optional[PlayerProfile] = None
    ) -> bool:
        """
        Extract a single highlight clip.
        
        Args:
            highlight: Highlight to extract
            output_path: Output file path
            include_stats: Whether to add stats overlay
            player: Player profile for stats overlay
            
        Returns:
            True if successful
        """
        if not self.video_path.exists():
            logger.error(f"Video not found: {self.video_path}")
            return False
        
        # Open video
        cap = cv2.VideoCapture(str(self.video_path))
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, highlight.frame_start)
        
        # Setup output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*self.config.codec)
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.config.output_fps,
            self.config.output_resolution
        )
        
        # Extract frames
        frame_count = 0
        total_frames = highlight.frame_end - highlight.frame_start
        
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize to output resolution
            frame = cv2.resize(frame, self.config.output_resolution)
            
            # Add stats overlay if requested
            if include_stats and player:
                frame = self._add_stats_overlay(frame, player, highlight)
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        # Update highlight with URL
        highlight.url = str(output_path)
        
        logger.info(f"Extracted clip: {output_path} ({frame_count} frames)")
        return True
    
    def create_compilation(
        self,
        player_jersey: Optional[int] = None,
        player_track: Optional[int] = None,
        output_path: str = "compilation.mp4",
        actions: Optional[List[str]] = None,
        max_clips: int = 10,
        include_stats_overlay: bool = True,
        include_transitions: bool = True
    ) -> bool:
        """
        Create a highlight compilation video.
        
        Args:
            player_jersey: Player jersey number
            player_track: Player track ID
            output_path: Output file path
            actions: Action types to include
            max_clips: Maximum clips
            include_stats_overlay: Add stats to video
            include_transitions: Add transitions between clips
            
        Returns:
            True if successful
        """
        # Get player
        player = None
        if player_jersey is not None:
            player = self.analysis.get_player(player_jersey)
        elif player_track is not None:
            player = self.analysis.get_player_by_track(player_track)
        
        if player is None:
            logger.error("Player not found")
            return False
        
        # Get highlights
        highlights = self.get_player_highlights(
            jersey_number=player_jersey,
            track_id=player_track,
            actions=actions,
            max_clips=max_clips
        )
        
        if not highlights:
            logger.warning("No highlights found for player")
            return False
        
        # Open video
        cap = cv2.VideoCapture(str(self.video_path))
        
        # Setup output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*self.config.codec)
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.config.output_fps,
            self.config.output_resolution
        )
        
        # Process each highlight
        for i, highlight in enumerate(highlights):
            logger.info(f"Processing highlight {i+1}/{len(highlights)}: {highlight.action_type}")
            
            # Add title card
            if include_transitions:
                title_card = self._create_title_card(
                    f"#{player.jersey_number or '?'} - {highlight.action_type.upper()}",
                    self.config.output_resolution
                )
                for _ in range(int(self.config.output_fps * 1.5)):  # 1.5 seconds
                    out.write(title_card)
            
            # Seek to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, highlight.frame_start)
            
            # Extract frames
            frame_count = 0
            total_frames = highlight.frame_end - highlight.frame_start
            
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.resize(frame, self.config.output_resolution)
                
                if include_stats_overlay:
                    frame = self._add_stats_overlay(frame, player, highlight)
                
                out.write(frame)
                frame_count += 1
        
        cap.release()
        out.release()
        
        logger.info(f"Compilation created: {output_path}")
        return True
    
    def generate_thumbnail(
        self,
        highlight: Highlight,
        output_path: str
    ) -> bool:
        """Generate thumbnail for a highlight."""
        if not self.video_path.exists():
            return False
        
        cap = cv2.VideoCapture(str(self.video_path))
        
        # Seek to middle of highlight
        middle_frame = (highlight.frame_start + highlight.frame_end) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return False
        
        # Resize and save
        frame = cv2.resize(frame, (320, 180))
        cv2.imwrite(output_path, frame)
        
        highlight.thumbnail_url = output_path
        return True
    
    def _add_stats_overlay(
        self,
        frame: np.ndarray,
        player: PlayerProfile,
        highlight: Highlight
    ) -> np.ndarray:
        """Add stats overlay to frame."""
        h, w = frame.shape[:2]
        
        # Semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 80), (300, h - 10), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Player info
        jersey_text = f"#{player.jersey_number}" if player.jersey_number else "Player"
        cv2.putText(
            frame, jersey_text,
            (20, h - 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        # Action type
        cv2.putText(
            frame, highlight.action_type.upper(),
            (20, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        
        # Scout score
        score_text = f"Scout: {player.scout_score:.1f}/10"
        cv2.putText(
            frame, score_text,
            (150, h - 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
        )
        
        return frame
    
    def _create_title_card(
        self,
        text: str,
        resolution: Tuple[int, int]
    ) -> np.ndarray:
        """Create a title card frame."""
        w, h = resolution
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Add gradient background
        for i in range(h):
            intensity = int(30 + (i / h) * 20)
            frame[i, :] = [intensity, intensity // 2, 0]
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        
        cv2.putText(
            frame, text,
            (text_x, text_y),
            font, font_scale, (255, 255, 255), thickness
        )
        
        return frame
