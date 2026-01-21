"""
Jersey Number OCR
=================

Detects and recognizes jersey numbers from player bounding boxes
using PaddleOCR or custom trained model.
"""
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
from loguru import logger

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    logger.warning("PaddleOCR not installed. Run: pip install paddleocr")

from src.detection.detector import Detection


@dataclass
class JerseyNumber:
    """Detected jersey number."""
    number: int
    confidence: float
    bbox: Tuple[float, float, float, float]  # Within player crop
    raw_text: str


class JerseyOCR:
    """
    Jersey number recognition using OCR.
    
    Extracts jersey numbers from player bounding boxes using
    PaddleOCR or a custom trained model.
    
    Example:
        ocr = JerseyOCR()
        jersey = ocr.detect(frame, player_detection)
        if jersey:
            print(f"Player #{jersey.number}")
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        confidence_threshold: float = 0.6,
        model_path: Optional[str] = None
    ):
        """
        Initialize the OCR system.
        
        Args:
            use_gpu: Whether to use GPU for OCR
            confidence_threshold: Minimum confidence for detections
            model_path: Path to custom trained model (optional)
        """
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        
        self._init_ocr()
        
        # Jersey number cache (track_id -> number)
        self.number_cache: dict = {}
        self.cache_hits: dict = {}
        self.cache_threshold = 3  # Require N consistent detections
    
    def _init_ocr(self) -> None:
        """Initialize the OCR engine."""
        if not PADDLE_AVAILABLE:
            self.ocr = None
            return
        
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=self.use_gpu,
            show_log=False,
            det_model_dir=self.model_path if self.model_path else None
        )
        logger.info("PaddleOCR initialized")
    
    def _preprocess_image(
        self,
        frame: np.ndarray,
        detection: Detection
    ) -> np.ndarray:
        """
        Preprocess player crop for OCR.
        
        Focuses on the back/front of jersey where number is located.
        """
        x1, y1, x2, y2 = map(int, detection.bbox)
        
        # Ensure within frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            return crop
        
        # Focus on jersey region (exclude head and legs)
        crop_h, crop_w = crop.shape[:2]
        top = int(crop_h * 0.15)  # Exclude head
        bottom = int(crop_h * 0.65)  # Exclude shorts/legs
        
        jersey_crop = crop[top:bottom, :]
        
        if jersey_crop.size == 0:
            return jersey_crop
        
        # Enhance contrast for better OCR
        lab = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Resize for better OCR (if too small)
        if enhanced.shape[0] < 64 or enhanced.shape[1] < 32:
            scale = max(64 / enhanced.shape[0], 32 / enhanced.shape[1])
            enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale)
        
        return enhanced
    
    def _parse_number(self, text: str) -> Optional[int]:
        """Parse OCR text to extract jersey number."""
        # Remove non-numeric characters
        digits = ''.join(c for c in text if c.isdigit())
        
        if not digits:
            return None
        
        try:
            number = int(digits)
            # Valid jersey numbers are typically 1-99
            if 1 <= number <= 99:
                return number
        except ValueError:
            pass
        
        return None
    
    def detect(
        self,
        frame: np.ndarray,
        detection: Detection
    ) -> Optional[JerseyNumber]:
        """
        Detect jersey number from a player detection.
        
        Args:
            frame: BGR image
            detection: Player detection with bounding box
            
        Returns:
            JerseyNumber if found, None otherwise
        """
        if self.ocr is None:
            return None
        
        # Check cache first
        if detection.track_id is not None and detection.track_id in self.number_cache:
            cached = self.number_cache[detection.track_id]
            return JerseyNumber(
                number=cached,
                confidence=1.0,  # High confidence for cached
                bbox=detection.bbox,
                raw_text=str(cached)
            )
        
        # Preprocess image
        crop = self._preprocess_image(frame, detection)
        
        if crop.size == 0:
            return None
        
        # Run OCR
        try:
            results = self.ocr.ocr(crop, cls=True)
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return None
        
        if not results or not results[0]:
            return None
        
        # Find best number detection
        best_number = None
        best_confidence = 0.0
        best_text = ""
        best_bbox = None
        
        for line in results[0]:
            if line is None or len(line) < 2:
                continue
                
            bbox, (text, confidence) = line[0], line[1]
            
            if confidence < self.confidence_threshold:
                continue
            
            number = self._parse_number(text)
            
            if number is not None and confidence > best_confidence:
                best_number = number
                best_confidence = confidence
                best_text = text
                best_bbox = bbox
        
        if best_number is None:
            return None
        
        # Update cache with voting
        if detection.track_id is not None:
            track_id = detection.track_id
            
            if track_id not in self.cache_hits:
                self.cache_hits[track_id] = {}
            
            if best_number not in self.cache_hits[track_id]:
                self.cache_hits[track_id][best_number] = 0
            
            self.cache_hits[track_id][best_number] += 1
            
            # Check if we have enough consistent detections
            if self.cache_hits[track_id][best_number] >= self.cache_threshold:
                self.number_cache[track_id] = best_number
        
        return JerseyNumber(
            number=best_number,
            confidence=best_confidence,
            bbox=tuple(best_bbox[0]) if best_bbox else detection.bbox,
            raw_text=best_text
        )
    
    def detect_batch(
        self,
        frame: np.ndarray,
        detections: List[Detection]
    ) -> List[Optional[JerseyNumber]]:
        """
        Detect jersey numbers for multiple players.
        
        Args:
            frame: BGR image
            detections: List of player detections
            
        Returns:
            List of JerseyNumber (or None for each)
        """
        results = []
        for det in detections:
            if det.class_name in ["player", "goalkeeper"]:
                result = self.detect(frame, det)
            else:
                result = None
            results.append(result)
        return results
    
    def get_cached_number(self, track_id: int) -> Optional[int]:
        """Get cached jersey number for a track ID."""
        return self.number_cache.get(track_id)
    
    def clear_cache(self) -> None:
        """Clear the number cache."""
        self.number_cache.clear()
        self.cache_hits.clear()
