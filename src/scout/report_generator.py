"""
Scout Report Generator
======================

Generates professional scouting reports in PDF and HTML formats.
"""
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import json
from datetime import datetime
from loguru import logger

from src.scout.player_profile import PlayerProfile, PlayerMetrics


@dataclass
class ReportConfig:
    """Report generation configuration."""
    template_dir: str = "src/scout/templates"
    output_dir: str = "data/reports"
    logo_path: Optional[str] = None
    include_highlights: bool = True
    include_heatmap: bool = True
    include_radar: bool = True
    company_name: str = "Football Intelligence Scout"


class ScoutReport:
    """
    Generates professional scouting reports.
    
    Features:
    - Quick overview reports
    - Full detailed analysis
    - Player comparison reports
    - PDF and HTML export
    - Customizable templates
    
    Example:
        report = ScoutReport(player)
        report.generate(
            output_path="reports/player_10.pdf",
            report_type="full"
        )
    """
    
    def __init__(
        self,
        player: PlayerProfile,
        config: Optional[ReportConfig] = None
    ):
        """
        Initialize report generator.
        
        Args:
            player: PlayerProfile to generate report for
            config: Report configuration
        """
        self.player = player
        self.config = config or ReportConfig()
    
    def generate(
        self,
        output_path: str,
        report_type: str = "full",
        include_highlights: bool = True,
        include_heatmap: bool = True,
        include_radar: bool = True,
        comparison_players: Optional[List[PlayerProfile]] = None
    ) -> bool:
        """
        Generate a scouting report.
        
        Args:
            output_path: Output file path (.pdf or .html)
            report_type: "quick", "full", or "comparison"
            include_highlights: Include highlight clips section
            include_heatmap: Include position heatmap
            include_radar: Include radar chart
            comparison_players: Players to compare (for comparison report)
            
        Returns:
            True if successful
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate report data
        if report_type == "quick":
            report_data = self._generate_quick_report()
        elif report_type == "comparison" and comparison_players:
            report_data = self._generate_comparison_report(comparison_players)
        else:
            report_data = self._generate_full_report()
        
        # Add optional sections
        report_data["include_highlights"] = include_highlights
        report_data["include_heatmap"] = include_heatmap
        report_data["include_radar"] = include_radar
        
        # Generate output based on extension
        if output_path.suffix == ".pdf":
            return self._generate_pdf(report_data, output_path)
        elif output_path.suffix == ".html":
            return self._generate_html(report_data, output_path)
        elif output_path.suffix == ".json":
            return self._generate_json(report_data, output_path)
        else:
            logger.error(f"Unsupported format: {output_path.suffix}")
            return False
    
    def _generate_quick_report(self) -> Dict[str, Any]:
        """Generate quick overview report data."""
        p = self.player
        m = p.metrics
        
        return {
            "report_type": "quick",
            "generated_at": datetime.now().isoformat(),
            "company": self.config.company_name,
            "player": {
                "jersey_number": p.jersey_number,
                "team": p.team,
                "position": p.detected_position.value,
                "minutes_played": p.minutes_played,
                "scout_score": round(p.scout_score, 1)
            },
            "match": {
                "match_id": p.match_id
            },
            "summary": {
                "physical": {
                    "speed": f"{m.physical.top_speed:.1f} km/h",
                    "distance": f"{m.physical.total_distance:.2f} km",
                    "sprints": m.physical.sprint_count
                },
                "technical": {
                    "pass_accuracy": f"{m.technical.pass_accuracy:.1f}%",
                    "dribble_success": f"{m.technical.dribble_success_rate:.1f}%",
                    "key_passes": m.technical.key_passes
                },
                "defensive": {
                    "tackles": m.defensive.tackles_won,
                    "interceptions": m.defensive.interceptions,
                    "pressures": m.defensive.pressures
                }
            },
            "strengths": p.strengths[:5],
            "weaknesses": p.weaknesses[:3],
            "highlights_count": len(p.highlights)
        }
    
    def _generate_full_report(self) -> Dict[str, Any]:
        """Generate full detailed report data."""
        quick = self._generate_quick_report()
        p = self.player
        m = p.metrics
        
        # Extend with detailed metrics
        quick["report_type"] = "full"
        quick["detailed_metrics"] = m.to_dict()
        
        # Add radar chart data
        quick["radar_data"] = self._calculate_radar_data()
        
        # Add action breakdown
        action_counts = {}
        for action in p.actions:
            action_type = action.action_type
            if action_type not in action_counts:
                action_counts[action_type] = {"total": 0, "successful": 0}
            action_counts[action_type]["total"] += 1
            if action.success:
                action_counts[action_type]["successful"] += 1
        
        quick["action_breakdown"] = action_counts
        
        # Add scout notes
        quick["scout_notes"] = self._generate_scout_notes()
        
        return quick
    
    def _generate_comparison_report(
        self,
        comparison_players: List[PlayerProfile]
    ) -> Dict[str, Any]:
        """Generate comparison report data."""
        report = self._generate_full_report()
        report["report_type"] = "comparison"
        
        comparisons = []
        for other in comparison_players:
            comparisons.append({
                "jersey_number": other.jersey_number,
                "team": other.team,
                "position": other.detected_position.value,
                "scout_score": round(other.scout_score, 1),
                "metrics": other.metrics.to_dict(),
                "radar_data": self._calculate_radar_data_for(other)
            })
        
        report["comparisons"] = comparisons
        return report
    
    def _calculate_radar_data(self) -> Dict[str, float]:
        """Calculate radar chart data for the player."""
        return self._calculate_radar_data_for(self.player)
    
    def _calculate_radar_data_for(self, player: PlayerProfile) -> Dict[str, float]:
        """Calculate radar chart data for any player."""
        m = player.metrics
        
        # Normalize to 0-100 scale
        pace = min(100, (m.physical.top_speed / 35) * 100)
        shooting = min(100, m.technical.shot_accuracy)
        passing = min(100, m.technical.pass_accuracy)
        dribbling = min(100, m.technical.dribble_success_rate)
        defending = min(100, m.defensive.tackle_success_rate)
        physical = min(100, (m.physical.total_distance / 13) * 100)
        vision = min(100, m.intelligence.positioning_score)
        
        return {
            "pace": round(pace, 1),
            "shooting": round(shooting, 1),
            "passing": round(passing, 1),
            "dribbling": round(dribbling, 1),
            "defending": round(defending, 1),
            "physical": round(physical, 1),
            "vision": round(vision, 1)
        }
    
    def _generate_scout_notes(self) -> str:
        """Generate automated scout notes."""
        p = self.player
        m = p.metrics
        notes = []
        
        # Position-based analysis
        pos = p.detected_position.value
        notes.append(f"Analyzed as a {pos}.")
        
        # Strengths
        if p.strengths:
            strengths_str = ", ".join(p.strengths[:3])
            notes.append(f"Key strengths include {strengths_str}.")
        
        # Technical analysis
        if m.technical.pass_accuracy > 85:
            notes.append("Excellent passing accuracy, reliable in possession.")
        elif m.technical.pass_accuracy < 70:
            notes.append("Passing needs improvement, loses ball frequently.")
        
        if m.technical.key_passes > 3:
            notes.append("Creates chances consistently, good vision.")
        
        # Physical analysis
        if m.physical.top_speed > 32:
            notes.append("Electric pace, can beat defenders with speed.")
        
        if m.physical.sprint_count > 30:
            notes.append("High work rate, covers ground well.")
        
        # Defensive analysis
        if m.defensive.interceptions > 4:
            notes.append("Reads the game well, anticipates passes.")
        
        # Weaknesses
        if p.weaknesses:
            weak_str = p.weaknesses[0].lower()
            notes.append(f"Area to develop: {weak_str}.")
        
        return " ".join(notes)
    
    def _generate_pdf(self, data: Dict, output_path: Path) -> bool:
        """Generate PDF report."""
        try:
            # Simple PDF generation using reportlab
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib import colors
        except ImportError:
            logger.warning("reportlab not installed, generating HTML instead")
            return self._generate_html(data, output_path.with_suffix(".html"))
        
        doc = SimpleDocTemplate(str(output_path), pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []
        
        # Title
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Title'],
            fontSize=24,
            textColor=colors.darkblue
        )
        elements.append(Paragraph(f"SCOUT REPORT", title_style))
        elements.append(Spacer(1, 0.5*cm))
        
        # Player info
        player = data["player"]
        info_text = f"Player #{player['jersey_number']} | {player['position']} | Score: {player['scout_score']}/10"
        elements.append(Paragraph(info_text, styles['Heading2']))
        elements.append(Spacer(1, 0.5*cm))
        
        # Summary table
        summary = data["summary"]
        table_data = [
            ["Category", "Metric", "Value"],
            ["Physical", "Top Speed", summary["physical"]["speed"]],
            ["Physical", "Distance", summary["physical"]["distance"]],
            ["Physical", "Sprints", str(summary["physical"]["sprints"])],
            ["Technical", "Pass Accuracy", summary["technical"]["pass_accuracy"]],
            ["Technical", "Dribble Success", summary["technical"]["dribble_success"]],
            ["Technical", "Key Passes", str(summary["technical"]["key_passes"])],
            ["Defensive", "Tackles", str(summary["defensive"]["tackles"])],
            ["Defensive", "Interceptions", str(summary["defensive"]["interceptions"])],
        ]
        
        table = Table(table_data, colWidths=[4*cm, 5*cm, 4*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        elements.append(Spacer(1, 1*cm))
        
        # Strengths & Weaknesses
        elements.append(Paragraph("STRENGTHS", styles['Heading3']))
        for s in data["strengths"]:
            elements.append(Paragraph(f"✓ {s}", styles['Normal']))
        
        elements.append(Spacer(1, 0.5*cm))
        elements.append(Paragraph("AREAS TO IMPROVE", styles['Heading3']))
        for w in data["weaknesses"]:
            elements.append(Paragraph(f"• {w}", styles['Normal']))
        
        # Scout notes
        if "scout_notes" in data:
            elements.append(Spacer(1, 1*cm))
            elements.append(Paragraph("SCOUT NOTES", styles['Heading3']))
            elements.append(Paragraph(data["scout_notes"], styles['Normal']))
        
        # Build PDF
        doc.build(elements)
        logger.info(f"PDF report generated: {output_path}")
        return True
    
    def _generate_html(self, data: Dict, output_path: Path) -> bool:
        """Generate HTML report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Scout Report - Player #{data['player']['jersey_number']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a365d; border-bottom: 3px solid #2c5282; padding-bottom: 10px; }}
        h2 {{ color: #2c5282; }}
        .score {{ font-size: 48px; font-weight: bold; color: #38a169; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #edf2f7; padding: 15px; border-radius: 8px; }}
        .metric-card h3 {{ margin: 0 0 10px 0; color: #4a5568; font-size: 14px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2d3748; }}
        .strengths {{ color: #38a169; }}
        .weaknesses {{ color: #e53e3e; }}
        .notes {{ background: #fffbeb; padding: 15px; border-left: 4px solid #d69e2e; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 SCOUT REPORT</h1>
        
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2>Player #{data['player']['jersey_number']} - {data['player']['position']}</h2>
                <p>Team: {data['player']['team']} | Minutes: {data['player']['minutes_played']:.0f}</p>
            </div>
            <div class="score">{data['player']['scout_score']}/10</div>
        </div>
        
        <h2>📊 Performance Summary</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>⚡ TOP SPEED</h3>
                <div class="metric-value">{data['summary']['physical']['speed']}</div>
            </div>
            <div class="metric-card">
                <h3>🏃 DISTANCE</h3>
                <div class="metric-value">{data['summary']['physical']['distance']}</div>
            </div>
            <div class="metric-card">
                <h3>💨 SPRINTS</h3>
                <div class="metric-value">{data['summary']['physical']['sprints']}</div>
            </div>
            <div class="metric-card">
                <h3>🎯 PASS ACCURACY</h3>
                <div class="metric-value">{data['summary']['technical']['pass_accuracy']}</div>
            </div>
            <div class="metric-card">
                <h3>⚽ DRIBBLE SUCCESS</h3>
                <div class="metric-value">{data['summary']['technical']['dribble_success']}</div>
            </div>
            <div class="metric-card">
                <h3>🔑 KEY PASSES</h3>
                <div class="metric-value">{data['summary']['technical']['key_passes']}</div>
            </div>
        </div>
        
        <h2 class="strengths">✅ Strengths</h2>
        <ul>
            {''.join(f'<li>{s}</li>' for s in data['strengths'])}
        </ul>
        
        <h2 class="weaknesses">❌ Areas to Improve</h2>
        <ul>
            {''.join(f'<li>{w}</li>' for w in data['weaknesses'])}
        </ul>
        
        {f'<div class="notes"><strong>📝 Scout Notes:</strong><br>{data.get("scout_notes", "")}</div>' if data.get("scout_notes") else ''}
        
        <hr>
        <p style="color: #718096; font-size: 12px;">
            Generated by {self.config.company_name} | {data['generated_at'][:10]}
        </p>
    </div>
</body>
</html>
"""
        
        with open(output_path, "w") as f:
            f.write(html)
        
        logger.info(f"HTML report generated: {output_path}")
        return True
    
    def _generate_json(self, data: Dict, output_path: Path) -> bool:
        """Generate JSON report."""
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"JSON report generated: {output_path}")
        return True
