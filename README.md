# TalentLens

<div align="center">

![TalentLens](https://img.shields.io/badge/TalentLens-AI%20Scout-red?style=for-the-badge)
![YOLO](https://img.shields.io/badge/YOLO-Detection-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python)
![TypeScript](https://img.shields.io/badge/TypeScript-API-3178C6?style=for-the-badge)

**AI-Powered Player Scouting System — Analyze matches, identify talents, generate player reports**

[Demo](#demo) · [Features](#features) · [Installation](#installation) · [API](#api-reference) · [Reports](#scout-reports)

</div>

---

## Overview

**TalentLens** is an end-to-end AI solution that transforms raw match footage into actionable scouting intelligence. Upload a match video, select the team to analyze, and get:

- **Highlight clips** of standout players automatically extracted
- **Player identification** using jersey number detection + Re-ID
- **Complete player profiles** with detailed performance metrics
- **Scouting reports** ready for recruitment decisions

### Target Users

| User | Use Case |
|------|----------|
| **Scout Departments** | Discover talents from lower leagues |
| **Technical Directors** | Evaluate potential signings |
| **Agents** | Build player portfolios |
| **Analysts** | Deep-dive into opponent players |
| **Academies** | Track youth player development |

---

## How It Works

```
                         TALENTLENS PIPELINE
+------------------------------------------------------------------------+
|                                                                         |
|  MATCH VIDEO                                                            |
|       |                                                                 |
|       v                                                                 |
|  +-----------+    +-----------+    +-----------+                        |
|  |   YOLO    |--->|  TRACKER  |--->|   TEAM    |                        |
|  | Detection |    | ByteTrack |    | Classifier|                        |
|  +-----------+    +-----------+    +-----------+                        |
|       |                |                |                               |
|       v                v                v                               |
|  +--------------------------------------------------+                   |
|  |           PLAYER IDENTIFICATION                  |                   |
|  |  - Jersey Number OCR                             |                   |
|  |  - Player Re-Identification                      |                   |
|  |  - Team Assignment                               |                   |
|  +--------------------------------------------------+                   |
|                        |                                                |
|                        v                                                |
|  +--------------------------------------------------+                   |
|  |           ACTION RECOGNITION                     |                   |
|  |  - Pass / Shot / Dribble / Tackle                |                   |
|  |  - Sprint / Press / Recovery                     |                   |
|  +--------------------------------------------------+                   |
|                        |                                                |
|                        v                                                |
|  +--------------------------------------------------+                   |
|  |           METRICS COMPUTATION                    |                   |
|  |  - Speed & Acceleration                          |                   |
|  |  - Technical Actions Success Rate                |                   |
|  |  - Defensive Contributions                       |                   |
|  +--------------------------------------------------+                   |
|                        |                                                |
|                        v                                                |
|  +------------+  +------------+  +------------+                         |
|  | HIGHLIGHTS |  |   SCOUT    |  |  PLAYER    |                         |
|  |   CLIPS    |  |  REPORTS   |  |  RANKINGS  |                         |
|  +------------+  +------------+  +------------+                         |
|                                                                         |
+------------------------------------------------------------------------+
```

---

## Features

### Player Detection & Identification

| Feature | Description | Technology |
|---------|-------------|------------|
| **Player Detection** | Real-time detection of all players | YOLO |
| **Jersey Number OCR** | Read jersey numbers for identification | PaddleOCR |
| **Player Re-ID** | Track same player across camera cuts | OSNet + FastReID |
| **Team Classification** | Separate teams by jersey color | K-Means + CNN |
| **Ball Possession** | Detect which player has the ball | Proximity analysis |

### Performance Metrics

#### Physical Metrics
| Metric | Description | Unit |
|--------|-------------|------|
| **Top Speed** | Maximum speed reached | km/h |
| **Avg Speed** | Average running speed | km/h |
| **Sprint Count** | Number of sprints (>25 km/h) | count |
| **Distance Covered** | Total distance by zone | km |
| **Acceleration** | Peak acceleration | m/s2 |

#### Technical Metrics
| Metric | Description | Calculation |
|--------|-------------|-------------|
| **Pass Accuracy** | Successful passes / Total passes | % |
| **Dribble Success** | Successful dribbles / Attempts | % |
| **Shot Accuracy** | Shots on target / Total shots | % |
| **First Touch** | Quality of ball control | AI score 0-100 |
| **Progressive Carries** | Carries advancing >10m | count |

#### Defensive Metrics
| Metric | Description |
|--------|-------------|
| **Tackles Won** | Successful tackles |
| **Interceptions** | Passes intercepted |
| **Blocks** | Shots/passes blocked |
| **Aerial Duels** | Headers won % |
| **Pressures** | Pressing actions |
| **Recoveries** | Ball recoveries |

### Highlight Generation

| Feature | Description |
|---------|-------------|
| **Auto-Clip Extraction** | Extract best moments per player |
| **Action-Based Clips** | Goals, assists, tackles, dribbles |
| **Compilation Generator** | Create player highlight reels |
| **Export Formats** | MP4, GIF, WebM |

### Scout Reports

| Report Type | Contents |
|-------------|----------|
| **Quick Overview** | Key stats, strengths, weaknesses |
| **Full Analysis** | Detailed breakdown by category |
| **Comparison Report** | Compare with similar players |
| **PDF Export** | Professional scout report |

---

## Architecture

```
TalentLens/
|-- src/
|   |-- detection/              # Player & Ball Detection
|   |   |-- detector.py         # YOLO main detector
|   |   |-- ball_detector.py    # Specialized ball detection
|   |   |-- jersey_ocr.py       # Jersey number recognition
|   |   +-- team_classifier.py  # Team classification
|   |
|   |-- tracking/               # Multi-Object Tracking
|   |   |-- tracker.py          # ByteTrack
|   |   |-- player_reid.py      # Player re-identification
|   |   +-- identity_manager.py # Consistent ID assignment
|   |
|   |-- scout/                  # Scouting Module
|   |   |-- player_profile.py   # Player profile builder
|   |   |-- report_generator.py # Scout report generation
|   |   |-- highlight_extractor.py
|   |   +-- player_ranker.py    # Rank players by position
|   |
|   +-- utils/
|       |-- video.py            # Video processing
|       |-- geometry.py         # Homography & transforms
|       +-- config.py           # Configuration
|
|-- models/                     # AI Models
|   |-- yolo_football.pt        # Player/ball detection
|   |-- jersey_ocr.pt           # Jersey number OCR
|   +-- player_reid.pt          # Re-identification
|
|-- data/
|   |-- matches/                # Uploaded matches
|   |-- highlights/             # Extracted clips
|   +-- reports/                # Generated reports
|
|-- tests/
|-- requirements.txt
+-- README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.0+ (recommended for GPU)
- FFmpeg (for video processing)

### Quick Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/TalentLens.git
cd TalentLens

# Python environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Download AI models
python scripts/download_models.py

# Run
python main.py
```

---

## Usage

### 1. Analyze Match

```python
from src.scout import MatchAnalyzer

analyzer = MatchAnalyzer(
    model_path="models/yolo_football.pt",
    enable_gpu=True
)

analysis = analyzer.analyze(
    video_path="match.mp4",
    home_team="FC Porto",
    away_team="SL Benfica",
    scout_team="FC Porto"
)

# Get top performers
for player in analysis.get_top_players(limit=5):
    print(f"#{player.jersey} - Score: {player.scout_score:.1f}")
```

### 2. Player Profile

```python
player = analysis.get_player(jersey_number=10)

print(f"Player #{player.jersey}")
print(f"Position: {player.detected_position}")
print(f"Top Speed: {player.metrics.physical.top_speed:.1f} km/h")
print(f"Pass Accuracy: {player.metrics.technical.pass_accuracy:.1f}%")
```

### 3. Generate Report

```python
from src.scout import ScoutReport

report = ScoutReport(player)
report.generate(
    output_path="reports/player_10.pdf",
    include_highlights=True,
    include_heatmap=True
)
```

---

## API Reference

### Match Endpoints

```
POST /api/matches/upload      Upload match video
GET  /api/matches/:id         Get match details
GET  /api/matches/:id/players Get detected players
```

### Player Endpoints

```
GET  /api/players/:id/profile    Full player profile
GET  /api/players/:id/metrics    Performance metrics
GET  /api/players/:id/highlights Player highlights
```

### Report Endpoints

```
POST /api/reports/generate    Generate scout report
GET  /api/reports/:id         Download report
```

---

## Scout Score

The **Scout Score** (0-10) is calculated using weighted metrics:

```
scout_score = (
    physical_score * 0.20 +      # 20%
    technical_score * 0.35 +     # 35%
    defensive_score * 0.20 +     # 20%
    intelligence_score * 0.25    # 25%
)
```

### Position-Specific Weights

| Position | Physical | Technical | Defensive | Intelligence |
|----------|----------|-----------|-----------|--------------|
| GK | 15% | 20% | 40% | 25% |
| CB | 25% | 15% | 40% | 20% |
| FB | 30% | 20% | 25% | 25% |
| CM | 20% | 35% | 20% | 25% |
| CAM | 15% | 40% | 10% | 35% |
| WNG | 35% | 35% | 10% | 20% |
| ST | 25% | 40% | 5% | 30% |

---

## Testing

```bash
pytest tests/ -v
pytest --cov=src --cov-report=html
```

---

## Origin

This project evolved from the [Football-Stuff](https://github.com/YOUR_USERNAME/Football-Stuff) repository, where initial experiments with football data analysis and simulations inspired the development of a full AI-powered scouting system.

---

## License

MIT License

---

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO
- [SoccerNet](https://www.soccer-net.org/) - Dataset & Benchmarks
- [FastReID](https://github.com/JDAI-CV/fast-reid) - Re-identification

---

<div align="center">

**TalentLens** - *Discover talents through AI*

</div>
