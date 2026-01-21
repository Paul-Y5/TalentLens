"""
Player Ranker
=============

Ranks and filters players based on various criteria.
"""
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from src.scout.player_profile import PlayerProfile, Position


class RankingCriteria(Enum):
    """Available ranking criteria."""
    SCOUT_SCORE = "scout_score"
    PHYSICAL = "physical"
    TECHNICAL = "technical"
    DEFENSIVE = "defensive"
    INTELLIGENCE = "intelligence"
    SPEED = "speed"
    DISTANCE = "distance"
    PASS_ACCURACY = "pass_accuracy"
    GOALS = "goals"
    ASSISTS = "assists"


@dataclass
class RankingFilter:
    """Filter for player ranking."""
    min_minutes: float = 0
    max_minutes: float = 120
    positions: Optional[List[Position]] = None
    teams: Optional[List[str]] = None
    min_score: float = 0


class PlayerRanker:
    """
    Ranks players based on various criteria.
    
    Example:
        ranker = PlayerRanker(analysis.players.values())
        
        # Get top scorers
        top = ranker.rank(RankingCriteria.SCOUT_SCORE, limit=5)
        
        # Get fastest players
        fastest = ranker.rank(RankingCriteria.SPEED, limit=5)
        
        # Filter by position
        midfielders = ranker.rank(
            RankingCriteria.TECHNICAL,
            filters=RankingFilter(positions=[Position.CM, Position.CAM])
        )
    """
    
    def __init__(self, players: List[PlayerProfile]):
        """
        Initialize ranker with players.
        
        Args:
            players: List of PlayerProfile objects
        """
        self.players = list(players)
    
    def rank(
        self,
        criteria: RankingCriteria,
        limit: int = 10,
        filters: Optional[RankingFilter] = None,
        ascending: bool = False
    ) -> List[PlayerProfile]:
        """
        Rank players by criteria.
        
        Args:
            criteria: Ranking criteria
            limit: Maximum players to return
            filters: Optional filters
            ascending: Sort ascending (False = highest first)
            
        Returns:
            Ranked list of players
        """
        players = self._apply_filters(self.players, filters)
        
        # Get sort key function
        key_func = self._get_sort_key(criteria)
        
        # Sort
        players.sort(key=key_func, reverse=not ascending)
        
        return players[:limit]
    
    def rank_by_position(
        self,
        position: Position,
        criteria: RankingCriteria = RankingCriteria.SCOUT_SCORE,
        limit: int = 5
    ) -> List[PlayerProfile]:
        """Rank players filtered by position."""
        filters = RankingFilter(positions=[position])
        return self.rank(criteria, limit, filters)
    
    def compare(
        self,
        player1: PlayerProfile,
        player2: PlayerProfile
    ) -> Dict[str, Dict]:
        """
        Compare two players across all metrics.
        
        Returns dict with winner for each category.
        """
        comparison = {}
        
        # Scout score
        comparison["scout_score"] = {
            "player1": player1.scout_score,
            "player2": player2.scout_score,
            "winner": 1 if player1.scout_score > player2.scout_score else 2
        }
        
        # Physical
        p1_speed = player1.metrics.physical.top_speed
        p2_speed = player2.metrics.physical.top_speed
        comparison["speed"] = {
            "player1": p1_speed,
            "player2": p2_speed,
            "winner": 1 if p1_speed > p2_speed else 2
        }
        
        # Technical
        p1_pass = player1.metrics.technical.pass_accuracy
        p2_pass = player2.metrics.technical.pass_accuracy
        comparison["passing"] = {
            "player1": p1_pass,
            "player2": p2_pass,
            "winner": 1 if p1_pass > p2_pass else 2
        }
        
        # Defensive
        p1_tackles = player1.metrics.defensive.tackles_won
        p2_tackles = player2.metrics.defensive.tackles_won
        comparison["tackles"] = {
            "player1": p1_tackles,
            "player2": p2_tackles,
            "winner": 1 if p1_tackles > p2_tackles else 2
        }
        
        return comparison
    
    def find_similar(
        self,
        player: PlayerProfile,
        limit: int = 5
    ) -> List[PlayerProfile]:
        """
        Find players with similar profiles.
        
        Args:
            player: Reference player
            limit: Number of similar players to find
            
        Returns:
            List of similar players
        """
        def similarity_score(other: PlayerProfile) -> float:
            if other.player_id == player.player_id:
                return float('inf')  # Exclude self
            
            # Compare key metrics
            score = 0
            
            # Position match
            if other.detected_position == player.detected_position:
                score += 10
            
            # Speed similarity
            speed_diff = abs(
                player.metrics.physical.top_speed - 
                other.metrics.physical.top_speed
            )
            score += max(0, 10 - speed_diff)
            
            # Technical similarity
            pass_diff = abs(
                player.metrics.technical.pass_accuracy -
                other.metrics.technical.pass_accuracy
            )
            score += max(0, 10 - pass_diff / 10)
            
            # Scout score similarity
            scout_diff = abs(player.scout_score - other.scout_score)
            score += max(0, 10 - scout_diff)
            
            return score
        
        # Rank by similarity
        others = [p for p in self.players if p.player_id != player.player_id]
        others.sort(key=similarity_score, reverse=True)
        
        return others[:limit]
    
    def _apply_filters(
        self,
        players: List[PlayerProfile],
        filters: Optional[RankingFilter]
    ) -> List[PlayerProfile]:
        """Apply filters to player list."""
        if filters is None:
            return players
        
        filtered = []
        for p in players:
            # Minutes filter
            if not (filters.min_minutes <= p.minutes_played <= filters.max_minutes):
                continue
            
            # Position filter
            if filters.positions and p.detected_position not in filters.positions:
                continue
            
            # Team filter
            if filters.teams and p.team not in filters.teams:
                continue
            
            # Score filter
            if p.scout_score < filters.min_score:
                continue
            
            filtered.append(p)
        
        return filtered
    
    def _get_sort_key(self, criteria: RankingCriteria) -> Callable:
        """Get sort key function for criteria."""
        if criteria == RankingCriteria.SCOUT_SCORE:
            return lambda p: p.scout_score
        elif criteria == RankingCriteria.PHYSICAL:
            return lambda p: (
                p.metrics.physical.top_speed +
                p.metrics.physical.total_distance * 3 +
                p.metrics.physical.sprint_count
            ) / 3
        elif criteria == RankingCriteria.TECHNICAL:
            return lambda p: (
                p.metrics.technical.pass_accuracy +
                p.metrics.technical.dribble_success_rate
            ) / 2
        elif criteria == RankingCriteria.DEFENSIVE:
            return lambda p: (
                p.metrics.defensive.tackles_won * 2 +
                p.metrics.defensive.interceptions * 2 +
                p.metrics.defensive.recoveries
            )
        elif criteria == RankingCriteria.INTELLIGENCE:
            return lambda p: p.metrics.intelligence.positioning_score
        elif criteria == RankingCriteria.SPEED:
            return lambda p: p.metrics.physical.top_speed
        elif criteria == RankingCriteria.DISTANCE:
            return lambda p: p.metrics.physical.total_distance
        elif criteria == RankingCriteria.PASS_ACCURACY:
            return lambda p: p.metrics.technical.pass_accuracy
        elif criteria == RankingCriteria.GOALS:
            return lambda p: p.metrics.technical.goals
        elif criteria == RankingCriteria.ASSISTS:
            return lambda p: p.metrics.technical.assists
        else:
            return lambda p: p.scout_score
