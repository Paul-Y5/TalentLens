"""
Configuration settings for the Scout System.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import yaml


@dataclass
class DetectionConfig:
    """Detection model configuration."""
    model_path: str = "models/yolo26_football.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "cuda"  # "cuda" or "cpu"
    img_size: int = 1280
    classes: List[str] = field(default_factory=lambda: ["player", "ball", "referee", "goalkeeper"])


@dataclass
class TrackingConfig:
    """Tracking configuration."""
    tracker_type: str = "bytetrack"  # "bytetrack", "botsort", "native"
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 0.8
    
    # Re-ID settings
    enable_reid: bool = True
    reid_model_path: str = "models/player_reid.pt"
    reid_threshold: float = 0.7


@dataclass
class OCRConfig:
    """Jersey number OCR configuration."""
    model_path: str = "models/jersey_ocr.pt"
    confidence_threshold: float = 0.6
    use_gpu: bool = True
    lang: str = "en"


@dataclass  
class ActionConfig:
    """Action recognition configuration."""
    model_path: str = "models/action_classifier.pt"
    actions: List[str] = field(default_factory=lambda: [
        "pass", "shot", "dribble", "tackle", "interception",
        "header", "cross", "clearance", "block", "foul"
    ])
    window_size: int = 30  # frames
    confidence_threshold: float = 0.6


@dataclass
class MetricsConfig:
    """Metrics computation configuration."""
    # Speed thresholds (km/h)
    sprint_threshold: float = 25.0
    high_intensity_threshold: float = 21.0
    jogging_threshold: float = 14.0
    
    # Distance zones
    zones: List[str] = field(default_factory=lambda: [
        "defensive_third", "middle_third", "attacking_third"
    ])
    
    # Pitch dimensions (meters)
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    
    # Frame rate for calculations
    fps: int = 25


@dataclass
class HighlightConfig:
    """Highlight extraction configuration."""
    pre_action_seconds: float = 3.0
    post_action_seconds: float = 2.0
    min_clip_duration: float = 3.0
    max_clip_duration: float = 15.0
    output_format: str = "mp4"
    output_fps: int = 30
    output_resolution: tuple = (1280, 720)


@dataclass
class ReportConfig:
    """Report generation configuration."""
    template_dir: str = "src/scout/templates"
    output_dir: str = "data/reports"
    include_highlights: bool = True
    include_heatmap: bool = True
    include_radar: bool = True
    logo_path: Optional[str] = None


@dataclass
class Config:
    """Main configuration class."""
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    highlight: HighlightConfig = field(default_factory=HighlightConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    
    # Paths
    models_dir: str = "models"
    data_dir: str = "data"
    matches_dir: str = "data/matches"
    highlights_dir: str = "data/highlights"
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        if "detection" in data:
            config.detection = DetectionConfig(**data["detection"])
        if "tracking" in data:
            config.tracking = TrackingConfig(**data["tracking"])
        if "ocr" in data:
            config.ocr = OCRConfig(**data["ocr"])
        if "action" in data:
            config.action = ActionConfig(**data["action"])
        if "metrics" in data:
            config.metrics = MetricsConfig(**data["metrics"])
        if "highlight" in data:
            config.highlight = HighlightConfig(**data["highlight"])
        if "report" in data:
            config.report = ReportConfig(**data["report"])
            
        return config
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import dataclasses
        
        def to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj
        
        with open(path, "w") as f:
            yaml.dump(to_dict(self), f, default_flow_style=False)


# Default configuration instance
default_config = Config()
