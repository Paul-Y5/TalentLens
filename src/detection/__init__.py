"""
Detection module for player, ball, and referee detection.
"""
from src.detection.detector import FootballDetector, BallDetector, Detection, FrameDetections
from src.detection.team_classifier import TeamClassifier
from src.detection.jersey_ocr import JerseyOCR

__all__ = [
    "FootballDetector",
    "BallDetector", 
    "Detection",
    "FrameDetections",
    "TeamClassifier",
    "JerseyOCR"
]
