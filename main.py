"""
Intelligent Player Scout System
===============================

AI-powered football match analysis with YOLO26 for player scouting.

Features:
- Match video analysis
- Player detection & tracking
- Team classification
- Jersey number recognition
- Performance metrics (physical, technical, defensive)
- Scout reports & player rankings
- Highlight extraction

Usage:
    python main.py analyze --video match.mp4 --home "FC Porto" --away "SL Benfica"
    python main.py report --player 10 --output report.pdf
    python main.py highlights --player 10 --output player_10.mp4
"""
import argparse
import sys
from pathlib import Path
from typing import Optional
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO"
)


def setup_argparser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="🔍 Intelligent Player Scout System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a match
  python main.py analyze --video match.mp4 --home "FC Porto" --away "Benfica" --scout "FC Porto"
  
  # Generate report for player #10
  python main.py report --analysis analysis.json --player 10 --output report.pdf
  
  # Extract highlights for a player
  python main.py highlights --analysis analysis.json --player 10 --output highlights/
  
  # List top players from analysis
  python main.py top --analysis analysis.json --limit 5
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a match video")
    analyze_parser.add_argument("--video", "-v", required=True, help="Path to match video")
    analyze_parser.add_argument("--home", required=True, help="Home team name")
    analyze_parser.add_argument("--away", required=True, help="Away team name")
    analyze_parser.add_argument("--scout", required=True, help="Team to scout (home/away name)")
    analyze_parser.add_argument("--competition", default="", help="Competition name")
    analyze_parser.add_argument("--date", default="", help="Match date")
    analyze_parser.add_argument("--output", "-o", default="data/analysis.json", help="Output analysis file")
    analyze_parser.add_argument("--model", default="models/yolo26_football.pt", help="YOLO model path")
    analyze_parser.add_argument("--gpu", action="store_true", default=True, help="Use GPU")
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate scout report")
    report_parser.add_argument("--analysis", "-a", required=True, help="Analysis JSON file")
    report_parser.add_argument("--player", "-p", type=int, required=True, help="Player jersey number")
    report_parser.add_argument("--output", "-o", required=True, help="Output report file (.pdf, .html, .json)")
    report_parser.add_argument("--type", choices=["quick", "full"], default="full", help="Report type")
    
    # Highlights command
    highlights_parser = subparsers.add_parser("highlights", help="Extract player highlights")
    highlights_parser.add_argument("--analysis", "-a", required=True, help="Analysis JSON file")
    highlights_parser.add_argument("--player", "-p", type=int, required=True, help="Player jersey number")
    highlights_parser.add_argument("--output", "-o", required=True, help="Output video file or directory")
    highlights_parser.add_argument("--max-clips", type=int, default=10, help="Maximum clips to extract")
    highlights_parser.add_argument("--actions", nargs="+", help="Action types to include")
    
    # Top players command
    top_parser = subparsers.add_parser("top", help="List top players")
    top_parser.add_argument("--analysis", "-a", required=True, help="Analysis JSON file")
    top_parser.add_argument("--limit", "-n", type=int, default=5, help="Number of players to show")
    top_parser.add_argument("--team", help="Filter by team")
    top_parser.add_argument("--criteria", choices=["score", "speed", "passing", "defensive"], 
                           default="score", help="Ranking criteria")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two players")
    compare_parser.add_argument("--analysis", "-a", required=True, help="Analysis JSON file")
    compare_parser.add_argument("--player1", type=int, required=True, help="First player jersey")
    compare_parser.add_argument("--player2", type=int, required=True, help="Second player jersey")
    
    return parser


def cmd_analyze(args) -> int:
    """Run match analysis."""
    from src.scout import MatchAnalyzer
    
    logger.info(f"🎬 Analyzing: {args.home} vs {args.away}")
    logger.info(f"📹 Video: {args.video}")
    logger.info(f"🔍 Scouting: {args.scout}")
    
    # Check video exists
    if not Path(args.video).exists():
        logger.error(f"Video not found: {args.video}")
        return 1
    
    # Create analyzer
    analyzer = MatchAnalyzer(
        model_path=args.model,
        enable_gpu=args.gpu
    )
    
    # Run analysis
    try:
        analysis = analyzer.analyze(
            video_path=args.video,
            home_team=args.home,
            away_team=args.away,
            scout_team=args.scout,
            competition=args.competition,
            date=args.date
        )
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        analysis.save(str(output_path))
        
        logger.info(f"✅ Analysis saved to: {output_path}")
        
        # Print summary
        top_players = analysis.get_top_players(3)
        logger.info("\n🏆 Top 3 Players:")
        for i, player in enumerate(top_players, 1):
            jersey = player.jersey_number or "?"
            logger.info(f"  {i}. #{jersey} - Score: {player.scout_score:.1f}/10")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


def cmd_report(args) -> int:
    """Generate scout report."""
    import json
    from src.scout import ScoutReport, PlayerProfile
    
    # Load analysis
    try:
        with open(args.analysis) as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Analysis file not found: {args.analysis}")
        return 1
    
    # Find player
    player_data = None
    for p in data.get("players", []):
        if p.get("jerseyNumber") == args.player:
            player_data = p
            break
    
    if not player_data:
        logger.error(f"Player #{args.player} not found in analysis")
        return 1
    
    # Create player profile (simplified reconstruction)
    player = PlayerProfile(
        player_id=player_data.get("playerId", ""),
        track_id=player_data.get("trackId", 0),
        jersey_number=player_data.get("jerseyNumber"),
        team=player_data.get("team", ""),
        scout_score=player_data.get("scoutScore", 0)
    )
    
    # Generate report
    report = ScoutReport(player)
    success = report.generate(
        output_path=args.output,
        report_type=args.type
    )
    
    if success:
        logger.info(f"✅ Report generated: {args.output}")
        return 0
    else:
        logger.error("Failed to generate report")
        return 1


def cmd_highlights(args) -> int:
    """Extract player highlights."""
    import json
    from src.scout import HighlightExtractor, PlayerProfile
    
    logger.info(f"🎬 Extracting highlights for player #{args.player}")
    
    # Load analysis
    try:
        with open(args.analysis) as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Analysis file not found: {args.analysis}")
        return 1
    
    # Find player and create mock analysis for extractor
    # In production, we'd deserialize the full MatchAnalysis
    logger.warning("Highlight extraction requires full analysis object")
    logger.info("Creating compilation from video...")
    
    # For demo, just show what would happen
    logger.info(f"Would extract {args.max_clips} clips for player #{args.player}")
    if args.actions:
        logger.info(f"Filtering by actions: {', '.join(args.actions)}")
    
    return 0


def cmd_top(args) -> int:
    """List top players."""
    import json
    
    # Load analysis
    try:
        with open(args.analysis) as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Analysis file not found: {args.analysis}")
        return 1
    
    players = data.get("players", [])
    
    # Filter by team
    if args.team:
        players = [p for p in players if p.get("team") == args.team]
    
    # Sort by criteria
    if args.criteria == "score":
        players.sort(key=lambda p: p.get("scoutScore", 0), reverse=True)
    elif args.criteria == "speed":
        players.sort(
            key=lambda p: p.get("metrics", {}).get("physical", {}).get("topSpeed", 0),
            reverse=True
        )
    elif args.criteria == "passing":
        players.sort(
            key=lambda p: p.get("metrics", {}).get("technical", {}).get("passing", {}).get("accuracy", 0),
            reverse=True
        )
    elif args.criteria == "defensive":
        players.sort(
            key=lambda p: p.get("metrics", {}).get("defensive", {}).get("interceptions", 0),
            reverse=True
        )
    
    # Print results
    match_info = data.get("matchInfo", {})
    logger.info(f"\n🏆 Top {args.limit} Players ({args.criteria.upper()})")
    logger.info(f"Match: {match_info.get('homeTeam', '?')} vs {match_info.get('awayTeam', '?')}\n")
    
    for i, player in enumerate(players[:args.limit], 1):
        jersey = player.get("jerseyNumber") or "?"
        score = player.get("scoutScore", 0)
        team = player.get("team", "?")
        position = player.get("position", {}).get("detected", "?")
        
        print(f"  {i}. #{jersey:<3} | {team:<6} | {position:<18} | Score: {score:.1f}/10")
    
    return 0


def cmd_compare(args) -> int:
    """Compare two players."""
    import json
    
    # Load analysis
    try:
        with open(args.analysis) as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Analysis file not found: {args.analysis}")
        return 1
    
    # Find players
    players = {p.get("jerseyNumber"): p for p in data.get("players", [])}
    
    p1 = players.get(args.player1)
    p2 = players.get(args.player2)
    
    if not p1:
        logger.error(f"Player #{args.player1} not found")
        return 1
    if not p2:
        logger.error(f"Player #{args.player2} not found")
        return 1
    
    # Compare
    logger.info(f"\n⚔️  Player Comparison: #{args.player1} vs #{args.player2}\n")
    
    metrics = [
        ("Scout Score", "scoutScore", None),
        ("Top Speed", "metrics.physical.topSpeed", "km/h"),
        ("Distance", "metrics.physical.totalDistance", "km"),
        ("Pass Accuracy", "metrics.technical.passing.accuracy", "%"),
        ("Tackles", "metrics.defensive.tackles.won", None),
        ("Interceptions", "metrics.defensive.interceptions", None),
    ]
    
    def get_nested(d, path):
        for key in path.split("."):
            d = d.get(key, {}) if isinstance(d, dict) else 0
        return d if not isinstance(d, dict) else 0
    
    for name, path, unit in metrics:
        v1 = get_nested(p1, path)
        v2 = get_nested(p2, path)
        
        winner = "←" if v1 > v2 else ("→" if v2 > v1 else "=")
        unit_str = f" {unit}" if unit else ""
        
        print(f"  {name:<15} | {v1:>8.1f}{unit_str:<4} {winner:^3} {v2:>8.1f}{unit_str}")
    
    return 0


def main() -> int:
    """Main entry point."""
    parser = setup_argparser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Route to command handler
    commands = {
        "analyze": cmd_analyze,
        "report": cmd_report,
        "highlights": cmd_highlights,
        "top": cmd_top,
        "compare": cmd_compare
    }
    
    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
