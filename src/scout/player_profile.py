"""
Player Profile
==============

Data structures for player profiles and metrics.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class Position(Enum):
    """Player positions."""
    GK = "Goalkeeper"
    CB = "Center Back"
    LB = "Left Back"
    RB = "Right Back"
    CDM = "Defensive Midfielder"
    CM = "Central Midfielder"
    CAM = "Attacking Midfielder"
    LW = "Left Winger"
    RW = "Right Winger"
    LM = "Left Midfielder"
    RM = "Right Midfielder"
    ST = "Striker"
    CF = "Center Forward"
    UNKNOWN = "Unknown"


@dataclass
class PhysicalMetrics:
    """Physical performance metrics."""
    top_speed: float = 0.0  # km/h
    avg_speed: float = 0.0  # km/h
    total_distance: float = 0.0  # km
    sprint_distance: float = 0.0  # km (>25 km/h)
    high_intensity_distance: float = 0.0  # km (>21 km/h)
    sprint_count: int = 0
    high_intensity_runs: int = 0
    peak_acceleration: float = 0.0  # m/s²
    
    # Zone distances
    defensive_third_distance: float = 0.0
    middle_third_distance: float = 0.0
    attacking_third_distance: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "topSpeed": self.top_speed,
            "avgSpeed": self.avg_speed,
            "totalDistance": self.total_distance,
            "sprintDistance": self.sprint_distance,
            "highIntensityDistance": self.high_intensity_distance,
            "sprintCount": self.sprint_count,
            "highIntensityRuns": self.high_intensity_runs,
            "peakAcceleration": self.peak_acceleration,
            "zoneDistances": {
                "defensive": self.defensive_third_distance,
                "middle": self.middle_third_distance,
                "attacking": self.attacking_third_distance
            }
        }


@dataclass
class TechnicalMetrics:
    """Technical performance metrics."""
    # Passing
    passes_attempted: int = 0
    passes_completed: int = 0
    pass_accuracy: float = 0.0
    key_passes: int = 0
    assists: int = 0
    progressive_passes: int = 0
    long_passes_completed: int = 0
    through_balls: int = 0
    
    # Dribbling
    dribbles_attempted: int = 0
    dribbles_completed: int = 0
    dribble_success_rate: float = 0.0
    progressive_carries: int = 0
    
    # Shooting
    shots_total: int = 0
    shots_on_target: int = 0
    shot_accuracy: float = 0.0
    goals: int = 0
    xg: float = 0.0  # Expected goals
    
    # Ball control
    first_touch_score: float = 0.0  # 0-100
    ball_retention_time: float = 0.0  # seconds
    touches: int = 0
    touches_in_box: int = 0
    
    # Crossing
    crosses_attempted: int = 0
    crosses_completed: int = 0
    cross_accuracy: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passing": {
                "attempted": self.passes_attempted,
                "completed": self.passes_completed,
                "accuracy": self.pass_accuracy,
                "keyPasses": self.key_passes,
                "assists": self.assists,
                "progressive": self.progressive_passes
            },
            "dribbling": {
                "attempted": self.dribbles_attempted,
                "completed": self.dribbles_completed,
                "successRate": self.dribble_success_rate,
                "progressiveCarries": self.progressive_carries
            },
            "shooting": {
                "total": self.shots_total,
                "onTarget": self.shots_on_target,
                "accuracy": self.shot_accuracy,
                "goals": self.goals,
                "xG": self.xg
            },
            "ballControl": {
                "firstTouchScore": self.first_touch_score,
                "retentionTime": self.ball_retention_time,
                "touches": self.touches,
                "touchesInBox": self.touches_in_box
            }
        }


@dataclass
class DefensiveMetrics:
    """Defensive performance metrics."""
    tackles_attempted: int = 0
    tackles_won: int = 0
    tackle_success_rate: float = 0.0
    
    interceptions: int = 0
    blocks: int = 0
    clearances: int = 0
    
    aerial_duels_attempted: int = 0
    aerial_duels_won: int = 0
    aerial_duel_success_rate: float = 0.0
    
    pressures: int = 0
    pressure_success_rate: float = 0.0
    
    recoveries: int = 0
    fouls_committed: int = 0
    fouls_won: int = 0
    
    # Normalized per 90 minutes
    defensive_actions_per_90: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tackles": {
                "attempted": self.tackles_attempted,
                "won": self.tackles_won,
                "successRate": self.tackle_success_rate
            },
            "interceptions": self.interceptions,
            "blocks": self.blocks,
            "clearances": self.clearances,
            "aerialDuels": {
                "attempted": self.aerial_duels_attempted,
                "won": self.aerial_duels_won,
                "successRate": self.aerial_duel_success_rate
            },
            "pressures": {
                "total": self.pressures,
                "successRate": self.pressure_success_rate
            },
            "recoveries": self.recoveries,
            "fouls": {
                "committed": self.fouls_committed,
                "won": self.fouls_won
            },
            "defensiveActionsPer90": self.defensive_actions_per_90
        }


@dataclass
class IntelligenceMetrics:
    """Tactical intelligence metrics (AI-analyzed)."""
    positioning_score: float = 0.0  # 0-100
    off_ball_movement_score: float = 0.0  # 0-100
    space_creation_runs: int = 0
    pressing_triggers: int = 0
    defensive_awareness_score: float = 0.0  # 0-100
    decision_making_score: float = 0.0  # 0-100
    
    # Heatmap data
    avg_position: tuple = (0.0, 0.0)  # (x, y) normalized
    position_variance: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "positioningScore": self.positioning_score,
            "offBallMovement": self.off_ball_movement_score,
            "spaceCreation": self.space_creation_runs,
            "pressingTriggers": self.pressing_triggers,
            "defensiveAwareness": self.defensive_awareness_score,
            "decisionMaking": self.decision_making_score,
            "avgPosition": {
                "x": self.avg_position[0],
                "y": self.avg_position[1]
            }
        }


@dataclass
class PlayerMetrics:
    """Complete player metrics."""
    physical: PhysicalMetrics = field(default_factory=PhysicalMetrics)
    technical: TechnicalMetrics = field(default_factory=TechnicalMetrics)
    defensive: DefensiveMetrics = field(default_factory=DefensiveMetrics)
    intelligence: IntelligenceMetrics = field(default_factory=IntelligenceMetrics)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "physical": self.physical.to_dict(),
            "technical": self.technical.to_dict(),
            "defensive": self.defensive.to_dict(),
            "intelligence": self.intelligence.to_dict()
        }


@dataclass
class Action:
    """A detected action/event."""
    action_type: str  # pass, shot, dribble, tackle, etc.
    timestamp: float  # seconds from start
    frame_id: int
    success: bool
    confidence: float
    start_position: tuple  # (x, y) on pitch
    end_position: Optional[tuple] = None  # For passes, shots
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Highlight:
    """A highlight clip."""
    clip_id: str
    action_type: str
    timestamp_start: float
    timestamp_end: float
    frame_start: int
    frame_end: int
    importance_score: float  # 0-10
    url: Optional[str] = None
    thumbnail_url: Optional[str] = None


@dataclass
class PlayerProfile:
    """Complete player profile from match analysis."""
    # Identification
    player_id: str
    track_id: int
    jersey_number: Optional[int] = None
    team: str = "unknown"  # "home", "away"
    
    # Match info
    match_id: str = ""
    minutes_played: float = 0.0
    
    # Position (detected)
    detected_position: Position = Position.UNKNOWN
    position_confidence: float = 0.0
    
    # Metrics
    metrics: PlayerMetrics = field(default_factory=PlayerMetrics)
    
    # Scout score (0-10)
    scout_score: float = 0.0
    
    # Analysis
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    
    # Actions & Highlights
    actions: List[Action] = field(default_factory=list)
    highlights: List[Highlight] = field(default_factory=list)
    
    # Heatmap data (positions over time)
    position_history: List[tuple] = field(default_factory=list)
    
    def calculate_scout_score(self, position_weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate overall scout score based on metrics.
        
        Args:
            position_weights: Custom weights for position
            
        Returns:
            Scout score from 0 to 10
        """
        # Default weights
        if position_weights is None:
            position_weights = {
                "physical": 0.20,
                "technical": 0.35,
                "defensive": 0.20,
                "intelligence": 0.25
            }
        
        # Normalize each category to 0-10
        physical_score = self._normalize_physical()
        technical_score = self._normalize_technical()
        defensive_score = self._normalize_defensive()
        intelligence_score = self._normalize_intelligence()
        
        # Weighted sum
        self.scout_score = (
            physical_score * position_weights["physical"] +
            technical_score * position_weights["technical"] +
            defensive_score * position_weights["defensive"] +
            intelligence_score * position_weights["intelligence"]
        )
        
        return self.scout_score
    
    def _normalize_physical(self) -> float:
        """Normalize physical metrics to 0-10 scale."""
        p = self.metrics.physical
        
        # Speed score (based on typical pro ranges)
        speed_score = min(10, (p.top_speed / 35) * 10)  # 35 km/h = 10
        
        # Distance score (per 90 min, ~10-13 km is good)
        distance_score = min(10, (p.total_distance / 13) * 10)
        
        # Sprint score
        sprint_score = min(10, (p.sprint_count / 40) * 10)  # 40 sprints = 10
        
        return (speed_score + distance_score + sprint_score) / 3
    
    def _normalize_technical(self) -> float:
        """Normalize technical metrics to 0-10 scale."""
        t = self.metrics.technical
        
        scores = []
        
        # Pass accuracy (weight heavily)
        if t.passes_attempted > 0:
            pass_score = (t.pass_accuracy / 100) * 10
            scores.append(pass_score * 1.5)
        
        # Dribble success
        if t.dribbles_attempted > 0:
            dribble_score = (t.dribble_success_rate / 100) * 10
            scores.append(dribble_score)
        
        # Key passes
        key_pass_score = min(10, t.key_passes * 2)  # 5 key passes = 10
        scores.append(key_pass_score)
        
        # Goals/assists bonus
        scores.append(min(10, (t.goals + t.assists) * 3))
        
        return sum(scores) / len(scores) if scores else 5.0
    
    def _normalize_defensive(self) -> float:
        """Normalize defensive metrics to 0-10 scale."""
        d = self.metrics.defensive
        
        scores = []
        
        # Tackle success
        if d.tackles_attempted > 0:
            tackle_score = (d.tackle_success_rate / 100) * 10
            scores.append(tackle_score)
        
        # Interceptions
        intercept_score = min(10, d.interceptions * 2)
        scores.append(intercept_score)
        
        # Recoveries
        recovery_score = min(10, d.recoveries)
        scores.append(recovery_score)
        
        # Aerial duels
        if d.aerial_duels_attempted > 0:
            aerial_score = (d.aerial_duel_success_rate / 100) * 10
            scores.append(aerial_score)
        
        return sum(scores) / len(scores) if scores else 5.0
    
    def _normalize_intelligence(self) -> float:
        """Normalize intelligence metrics to 0-10 scale."""
        i = self.metrics.intelligence
        
        return (
            i.positioning_score / 10 +
            i.off_ball_movement_score / 10 +
            i.defensive_awareness_score / 10 +
            i.decision_making_score / 10
        ) / 4
    
    def analyze_strengths_weaknesses(self) -> None:
        """Analyze and set strengths/weaknesses based on metrics."""
        self.strengths = []
        self.weaknesses = []
        
        p = self.metrics.physical
        t = self.metrics.technical
        d = self.metrics.defensive
        i = self.metrics.intelligence
        
        # Physical analysis
        if p.top_speed > 32:
            self.strengths.append("Pace")
        elif p.top_speed < 28:
            self.weaknesses.append("Pace")
        
        if p.sprint_count > 30:
            self.strengths.append("Work Rate")
        
        # Technical analysis
        if t.pass_accuracy > 85:
            self.strengths.append("Passing")
        elif t.pass_accuracy < 70:
            self.weaknesses.append("Passing")
        
        if t.dribble_success_rate > 65:
            self.strengths.append("Dribbling")
        elif t.dribble_success_rate < 40:
            self.weaknesses.append("Ball Control")
        
        if t.key_passes > 3:
            self.strengths.append("Vision")
        
        if t.shot_accuracy > 50:
            self.strengths.append("Finishing")
        elif t.shots_total > 2 and t.shot_accuracy < 30:
            self.weaknesses.append("Finishing")
        
        # Defensive analysis
        if d.tackle_success_rate > 70:
            self.strengths.append("Tackling")
        elif d.tackles_attempted > 3 and d.tackle_success_rate < 40:
            self.weaknesses.append("Tackling")
        
        if d.aerial_duel_success_rate > 60:
            self.strengths.append("Aerial Duels")
        elif d.aerial_duels_attempted > 3 and d.aerial_duel_success_rate < 35:
            self.weaknesses.append("Aerial Duels")
        
        if d.interceptions > 4:
            self.strengths.append("Reading the Game")
        
        # Intelligence analysis
        if i.positioning_score > 75:
            self.strengths.append("Positioning")
        elif i.positioning_score < 50:
            self.weaknesses.append("Positioning")
        
        if i.off_ball_movement_score > 75:
            self.strengths.append("Off-ball Movement")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "playerId": self.player_id,
            "trackId": self.track_id,
            "jerseyNumber": self.jersey_number,
            "team": self.team,
            "matchId": self.match_id,
            "minutesPlayed": self.minutes_played,
            "position": {
                "detected": self.detected_position.value,
                "confidence": self.position_confidence
            },
            "metrics": self.metrics.to_dict(),
            "scoutScore": self.scout_score,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "highlightsCount": len(self.highlights),
            "actionsCount": len(self.actions)
        }
