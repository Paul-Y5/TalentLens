"""
Football Intelligent Scout System
=================================

AI-Powered Player Scouting System using YOLO26 for detection,
tracking for player identification, and advanced metrics computation.

Main modules:
- detection: Player and ball detection using YOLO26
- tracking: Multi-object tracking with Re-ID
- actions: Action recognition (pass, shot, dribble, tackle)
- metrics: Physical, technical, and defensive metrics
- scout: Report generation and highlight extraction
- visualization: Heatmaps, radar charts, and annotations
"""

__version__ = "1.0.0"
__author__ = "Paulo"

from src.scout.match_analyzer import MatchAnalyzer
from src.scout.player_profile import PlayerProfile
from src.scout.report_generator import ScoutReport
from src.scout.highlight_extractor import HighlightExtractor

__all__ = [
    "MatchAnalyzer",
    "PlayerProfile", 
    "ScoutReport",
    "HighlightExtractor"
]
