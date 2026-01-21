"""
YOLO26 Football Detector
========================

Main detection module using YOLO26 for player, ball, and referee detection.
"""
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import cv2
from loguru import logger

try:
    from ultralytics import YOLO
except ImportError:
    logger.warning("ultralytics not installed. Run: pip install ultralytics")
    YOLO = None


@dataclass
class Detection:
    """Single detection result."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def width(self) -> float:
        """Get width of bounding box."""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        """Get height of bounding box."""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> float:
        """Get area of bounding box."""
        return self.width * self.height


@dataclass
class FrameDetections:
    """All detections for a single frame."""
    frame_id: int
    players: List[Detection]
    ball: Optional[Detection]
    referees: List[Detection]
    goalkeepers: List[Detection]
    
    @property
    def all_detections(self) -> List[Detection]:
        """Get all detections as a flat list."""
        detections = self.players + self.referees + self.goalkeepers
        if self.ball:
            detections.append(self.ball)
        return detections


class FootballDetector:
    """
    YOLO26-based detector for football matches.
    
    Detects:
    - Players (field players)
    - Ball
    - Referees
    - Goalkeepers
    
    Example:
        detector = FootballDetector("models/yolo26_football.pt")
        detections = detector.detect(frame)
        for player in detections.players:
            print(f"Player at {player.center} with confidence {player.confidence}")
    """
    
    CLASS_NAMES = {
        0: "player",
        1: "ball", 
        2: "referee",
        3: "goalkeeper"
    }
    
    def __init__(
        self,
        model_path: str = "models/yolo26_football.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        img_size: int = 1280
    ):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to YOLO26 model weights
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ("cuda" or "cpu")
            img_size: Input image size for model
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.img_size = img_size
        
        self._load_model()
        
    def _load_model(self) -> None:
        """Load the YOLO model."""
        if YOLO is None:
            raise ImportError("ultralytics package not installed")
            
        if not self.model_path.exists():
            logger.warning(f"Model not found at {self.model_path}. Using pretrained model.")
            # Fall back to pretrained model for demo
            self.model = YOLO("yolov8l.pt")
        else:
            self.model = YOLO(str(self.model_path))
            
        # Move to device
        if self.device == "cuda" and torch.cuda.is_available():
            self.model.to("cuda")
            logger.info("Model loaded on GPU")
        else:
            self.model.to("cpu")
            logger.info("Model loaded on CPU")
    
    def detect(
        self,
        frame: np.ndarray,
        frame_id: int = 0
    ) -> FrameDetections:
        """
        Run detection on a single frame.
        
        Args:
            frame: BGR image as numpy array
            frame_id: Frame number for tracking
            
        Returns:
            FrameDetections object containing all detections
        """
        # Run inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            verbose=False
        )[0]
        
        # Parse results
        players = []
        referees = []
        goalkeepers = []
        ball = None
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            # Get track IDs if available
            track_ids = None
            if results.boxes.id is not None:
                track_ids = results.boxes.id.cpu().numpy().astype(int)
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                track_id = track_ids[i] if track_ids is not None else None
                
                # Map class IDs (adjust based on your model's classes)
                # For demo with pretrained model, class 0 is "person"
                if cls_id == 0:  # person -> treat as player
                    class_name = "player"
                    cls_id = 0
                elif cls_id == 32:  # sports ball
                    class_name = "ball"
                    cls_id = 1
                else:
                    continue  # Skip other classes
                
                detection = Detection(
                    bbox=tuple(box),
                    confidence=float(conf),
                    class_id=cls_id,
                    class_name=class_name,
                    track_id=track_id
                )
                
                if class_name == "player":
                    players.append(detection)
                elif class_name == "ball":
                    if ball is None or conf > ball.confidence:
                        ball = detection
                elif class_name == "referee":
                    referees.append(detection)
                elif class_name == "goalkeeper":
                    goalkeepers.append(detection)
        
        return FrameDetections(
            frame_id=frame_id,
            players=players,
            ball=ball,
            referees=referees,
            goalkeepers=goalkeepers
        )
    
    def detect_with_tracking(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        persist: bool = True
    ) -> FrameDetections:
        """
        Run detection with built-in tracking.
        
        Args:
            frame: BGR image as numpy array
            frame_id: Frame number
            persist: Whether to persist tracks between frames
            
        Returns:
            FrameDetections with track IDs assigned
        """
        # Run inference with tracking
        results = self.model.track(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            persist=persist,
            verbose=False
        )[0]
        
        # Parse results (same as detect but with track IDs)
        players = []
        referees = []
        goalkeepers = []
        ball = None
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            track_ids = None
            if results.boxes.id is not None:
                track_ids = results.boxes.id.cpu().numpy().astype(int)
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                track_id = int(track_ids[i]) if track_ids is not None else None
                
                # Map classes
                if cls_id == 0:
                    class_name = "player"
                    cls_id = 0
                elif cls_id == 32:
                    class_name = "ball"
                    cls_id = 1
                else:
                    continue
                
                detection = Detection(
                    bbox=tuple(box),
                    confidence=float(conf),
                    class_id=cls_id,
                    class_name=class_name,
                    track_id=track_id
                )
                
                if class_name == "player":
                    players.append(detection)
                elif class_name == "ball":
                    if ball is None or conf > ball.confidence:
                        ball = detection
                elif class_name == "referee":
                    referees.append(detection)
                elif class_name == "goalkeeper":
                    goalkeepers.append(detection)
        
        return FrameDetections(
            frame_id=frame_id,
            players=players,
            ball=ball,
            referees=referees,
            goalkeepers=goalkeepers
        )
    
    def batch_detect(
        self,
        frames: List[np.ndarray],
        start_frame_id: int = 0
    ) -> List[FrameDetections]:
        """
        Run detection on a batch of frames.
        
        Args:
            frames: List of BGR images
            start_frame_id: Starting frame ID
            
        Returns:
            List of FrameDetections
        """
        results = []
        for i, frame in enumerate(frames):
            detection = self.detect(frame, frame_id=start_frame_id + i)
            results.append(detection)
        return results


class BallDetector:
    """
    Specialized detector for football/ball detection.
    
    Uses a dedicated model optimized for small ball detection,
    which is typically more challenging than player detection.
    """
    
    def __init__(
        self,
        model_path: str = "models/ball_detector.pt",
        confidence_threshold: float = 0.3,
        device: str = "cuda"
    ):
        """Initialize ball detector."""
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Ball tracking history for interpolation
        self.ball_history: List[Optional[Detection]] = []
        self.max_history = 10
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the ball detection model."""
        if YOLO is None:
            raise ImportError("ultralytics package not installed")
            
        if self.model_path.exists():
            self.model = YOLO(str(self.model_path))
        else:
            # Use general model as fallback
            logger.warning("Ball detector model not found, using general detector")
            self.model = YOLO("yolov8l.pt")
        
        if self.device == "cuda" and torch.cuda.is_available():
            self.model.to("cuda")
        else:
            self.model.to("cpu")
    
    def detect(self, frame: np.ndarray) -> Optional[Detection]:
        """
        Detect the ball in a frame.
        
        Args:
            frame: BGR image
            
        Returns:
            Ball Detection or None if not found
        """
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            verbose=False
        )[0]
        
        ball = None
        
        if results.boxes is not None:
            for box, conf, cls_id in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.conf.cpu().numpy(),
                results.boxes.cls.cpu().numpy()
            ):
                # Class 32 is sports ball in COCO
                if int(cls_id) == 32:
                    if ball is None or conf > ball.confidence:
                        ball = Detection(
                            bbox=tuple(box),
                            confidence=float(conf),
                            class_id=1,
                            class_name="ball"
                        )
        
        # Update history
        self.ball_history.append(ball)
        if len(self.ball_history) > self.max_history:
            self.ball_history.pop(0)
        
        return ball
    
    def interpolate_position(self) -> Optional[Tuple[float, float]]:
        """
        Interpolate ball position when detection fails.
        
        Uses history to estimate current position.
        """
        # Find last N valid detections
        valid_positions = [
            d.center for d in self.ball_history
            if d is not None
        ]
        
        if len(valid_positions) < 2:
            return None
        
        # Simple linear interpolation
        last = valid_positions[-1]
        prev = valid_positions[-2]
        
        dx = last[0] - prev[0]
        dy = last[1] - prev[1]
        
        return (last[0] + dx, last[1] + dy)
