"""
Player Re-Identification
========================

Re-identifies players across camera cuts using appearance features.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from loguru import logger


@dataclass
class PlayerAppearance:
    """Player appearance features for re-identification."""
    player_id: str
    features: np.ndarray
    jersey_number: Optional[int] = None
    team: Optional[str] = None
    
    def similarity(self, other: "PlayerAppearance") -> float:
        """Compute cosine similarity with another appearance."""
        norm1 = np.linalg.norm(self.features)
        norm2 = np.linalg.norm(other.features)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(self.features, other.features) / (norm1 * norm2))


class PlayerReID:
    """
    Player Re-Identification module.
    
    Uses deep learning features to re-identify players after:
    - Camera cuts
    - Occlusions
    - Leaving and re-entering frame
    
    Features:
    - OSNet/FastReID feature extraction
    - Gallery management
    - Cross-camera matching
    
    Example:
        reid = PlayerReID()
        
        # Extract features for a player
        features = reid.extract_features(player_crop)
        
        # Match against gallery
        matches = reid.match(features, gallery)
    """
    
    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        model_path: Optional[str] = None,
        use_gpu: bool = True,
        feature_dim: int = 512,
        match_threshold: float = 0.7
    ):
        """
        Initialize ReID module.
        
        Args:
            model_name: Name of ReID model
            model_path: Path to model weights
            use_gpu: Whether to use GPU
            feature_dim: Feature dimension
            match_threshold: Minimum similarity for match
        """
        self.model_name = model_name
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.feature_dim = feature_dim
        self.match_threshold = match_threshold
        
        # Gallery of known players
        self.gallery: Dict[str, PlayerAppearance] = {}
        
        # Feature extractor
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the ReID model."""
        try:
            import torch
            import torchvision.transforms as T
            
            self.device = torch.device(
                "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
            )
            
            # Try to load torchreid
            try:
                import torchreid
                
                self.model = torchreid.models.build_model(
                    name=self.model_name,
                    num_classes=1000,
                    pretrained=True
                )
                self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"Loaded ReID model: {self.model_name}")
                
            except ImportError:
                logger.warning("torchreid not installed, using simple feature extractor")
                self._use_simple_extractor()
            
            # Preprocessing
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        except ImportError:
            logger.warning("PyTorch not available, using color histogram features")
            self.model = None
    
    def _use_simple_extractor(self) -> None:
        """Use simple ResNet as fallback."""
        import torch
        import torchvision.models as models
        
        resnet = models.resnet18(pretrained=True)
        # Remove final classification layer
        self.model = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.model.to(self.device)
        self.model.eval()
        
        self.feature_dim = 512
        logger.info("Using ResNet18 for feature extraction")
    
    def extract_features(
        self,
        image: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Extract appearance features from player crop.
        
        Args:
            image: BGR player crop
            normalize: Whether to L2 normalize features
            
        Returns:
            Feature vector
        """
        if self.model is None:
            # Fallback to color histogram
            return self._extract_color_features(image)
        
        import torch
        
        # Convert BGR to RGB
        image_rgb = image[:, :, ::-1].copy()
        
        # Transform
        input_tensor = self.transform(image_rgb)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(input_batch)
            features = features.squeeze().cpu().numpy()
        
        # Normalize
        if normalize:
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
        
        return features
    
    def _extract_color_features(self, image: np.ndarray) -> np.ndarray:
        """Extract color histogram features as fallback."""
        import cv2
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Compute histogram
        h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        # Concatenate and normalize
        features = np.concatenate([
            h_hist.flatten(),
            s_hist.flatten(),
            v_hist.flatten()
        ])
        
        features = features / (np.linalg.norm(features) + 1e-8)
        
        # Pad to feature_dim
        if len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)))
        
        return features[:self.feature_dim]
    
    def add_to_gallery(
        self,
        player_id: str,
        features: np.ndarray,
        jersey_number: Optional[int] = None,
        team: Optional[str] = None
    ) -> None:
        """
        Add a player to the gallery.
        
        Args:
            player_id: Unique player identifier
            features: Feature vector
            jersey_number: Optional jersey number
            team: Optional team
        """
        appearance = PlayerAppearance(
            player_id=player_id,
            features=features,
            jersey_number=jersey_number,
            team=team
        )
        
        if player_id in self.gallery:
            # Update with running average
            old = self.gallery[player_id]
            appearance.features = 0.7 * old.features + 0.3 * features
            appearance.features /= np.linalg.norm(appearance.features)
        
        self.gallery[player_id] = appearance
    
    def match(
        self,
        features: np.ndarray,
        top_k: int = 1,
        team_filter: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Match features against gallery.
        
        Args:
            features: Query feature vector
            top_k: Number of top matches to return
            team_filter: Only match within team
            
        Returns:
            List of (player_id, similarity) tuples
        """
        if not self.gallery:
            return []
        
        query = PlayerAppearance(player_id="query", features=features)
        
        matches = []
        for player_id, appearance in self.gallery.items():
            # Filter by team
            if team_filter and appearance.team != team_filter:
                continue
            
            similarity = query.similarity(appearance)
            
            if similarity >= self.match_threshold:
                matches.append((player_id, similarity))
        
        # Sort by similarity
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches[:top_k]
    
    def find_best_match(
        self,
        features: np.ndarray,
        team_filter: Optional[str] = None
    ) -> Optional[Tuple[str, float]]:
        """
        Find the best matching player.
        
        Args:
            features: Query features
            team_filter: Only match within team
            
        Returns:
            (player_id, similarity) or None if no match
        """
        matches = self.match(features, top_k=1, team_filter=team_filter)
        return matches[0] if matches else None
    
    def update_gallery_from_tracks(
        self,
        tracks: List["Track"],
        frame: np.ndarray,
        team_assignments: Optional[Dict[int, str]] = None
    ) -> None:
        """
        Update gallery from current tracks.
        
        Args:
            tracks: List of active tracks
            frame: Current frame
            team_assignments: Map of track_id -> team
        """
        for track in tracks:
            # Extract player crop
            x1, y1, x2, y2 = map(int, track.bbox)
            
            # Ensure valid crop
            h, w = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            # Extract features
            features = self.extract_features(crop)
            
            # Get team
            team = team_assignments.get(track.track_id) if team_assignments else None
            
            # Add to gallery
            self.add_to_gallery(
                player_id=f"track_{track.track_id}",
                features=features,
                team=team
            )
    
    def clear_gallery(self) -> None:
        """Clear the gallery."""
        self.gallery.clear()
    
    def save_gallery(self, path: str) -> None:
        """Save gallery to file."""
        data = {
            player_id: {
                "features": appearance.features.tolist(),
                "jersey_number": appearance.jersey_number,
                "team": appearance.team
            }
            for player_id, appearance in self.gallery.items()
        }
        
        import json
        with open(path, "w") as f:
            json.dump(data, f)
        
        logger.info(f"Saved gallery with {len(self.gallery)} players to {path}")
    
    def load_gallery(self, path: str) -> None:
        """Load gallery from file."""
        import json
        
        with open(path) as f:
            data = json.load(f)
        
        for player_id, info in data.items():
            self.gallery[player_id] = PlayerAppearance(
                player_id=player_id,
                features=np.array(info["features"]),
                jersey_number=info.get("jersey_number"),
                team=info.get("team")
            )
        
        logger.info(f"Loaded gallery with {len(self.gallery)} players from {path}")
