# 🔥 Football Intelligent Scout System

<div align="center">

![Scout System](https://img.shields.io/badge/AI%20Scout-Player%20Analysis-red?style=for-the-badge)
![YOLO26](https://img.shields.io/badge/YOLO26-Detection-orange?style=for-the-badge)
![TypeScript](https://img.shields.io/badge/TypeScript-API-3178C6?style=for-the-badge)
![React](https://img.shields.io/badge/React-Dashboard-61DAFB?style=for-the-badge)

**AI-Powered Player Scouting System — Analyze matches, identify talents, generate player reports**

[Demo](#-demo) • [Features](#-features) • [Installation](#-installation) • [API](#-api-reference) • [Scout Reports](#-scout-reports)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Scout Reports](#-scout-reports)
- [Metrics & KPIs](#-metrics--kpis)
- [Roadmap](#-roadmap)

---

## 🎯 Overview

**Football Intelligent Scout System** is an end-to-end AI solution that transforms raw match footage into actionable scouting intelligence. Upload a match video, select the team to analyze, and get:

- 🎬 **Highlight clips** of standout players automatically extracted
- 🆔 **Player identification** using jersey number detection + Re-ID
- 📊 **Complete player profiles** with detailed performance metrics
- 📈 **Scouting reports** ready for recruitment decisions

### Target Users

| User | Use Case |
|------|----------|
| **Scout Departments** | Discover talents from lower leagues |
| **Technical Directors** | Evaluate potential signings |
| **Agents** | Build player portfolios |
| **Analysts** | Deep-dive into opponent players |
| **Academies** | Track youth player development |

---

## 🎬 How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INTELLIGENT SCOUT PIPELINE                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  📹 MATCH VIDEO                                                          │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │   YOLO26    │───▶│   TRACKER   │───▶│    TEAM     │                  │
│  │  Detection  │    │  ByteTrack  │    │ Classifier  │                  │
│  └─────────────┘    └─────────────┘    └─────────────┘                  │
│       │                   │                   │                          │
│       ▼                   ▼                   ▼                          │
│  ┌─────────────────────────────────────────────────┐                    │
│  │              PLAYER IDENTIFICATION              │                    │
│  │  • Jersey Number OCR                            │                    │
│  │  • Player Re-Identification                     │                    │
│  │  • Team Assignment                              │                    │
│  └─────────────────────────────────────────────────┘                    │
│                          │                                               │
│                          ▼                                               │
│  ┌─────────────────────────────────────────────────┐                    │
│  │              ACTION RECOGNITION                  │                    │
│  │  • Pass / Shot / Dribble / Tackle               │                    │
│  │  • Sprint / Press / Recovery                    │                    │
│  │  • Header / Interception / Block                │                    │
│  └─────────────────────────────────────────────────┘                    │
│                          │                                               │
│                          ▼                                               │
│  ┌─────────────────────────────────────────────────┐                    │
│  │              METRICS COMPUTATION                 │                    │
│  │  • Speed & Acceleration                         │                    │
│  │  • Technical Actions Success Rate               │                    │
│  │  • Defensive Contributions                      │                    │
│  │  • Positioning & Movement                       │                    │
│  └─────────────────────────────────────────────────┘                    │
│                          │                                               │
│                          ▼                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │   HIGHLIGHT  │  │    SCOUT     │  │   PLAYER     │                   │
│  │    CLIPS     │  │   REPORTS    │  │   RANKINGS   │                   │
│  └──────────────┘  └──────────────┘  └──────────────┘                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 🎯 Player Detection & Identification

| Feature | Description | Technology |
|---------|-------------|------------|
| **Player Detection** | Real-time detection of all players | YOLO26 |
| **Jersey Number OCR** | Read jersey numbers for identification | PaddleOCR |
| **Player Re-ID** | Track same player across camera cuts | OSNet + Fastreid |
| **Team Classification** | Separate teams by jersey color | K-Means + CNN |
| **Ball Possession** | Detect which player has the ball | Proximity analysis |

### 📊 Performance Metrics

#### ⚡ Physical Metrics
| Metric | Description | Unit |
|--------|-------------|------|
| **Top Speed** | Maximum speed reached | km/h |
| **Avg Speed** | Average running speed | km/h |
| **Sprint Count** | Number of sprints (>25 km/h) | count |
| **Distance Covered** | Total distance by zone | km |
| **Acceleration** | Peak acceleration | m/s² |
| **High Intensity Runs** | Runs >21 km/h | count |

#### ⚽ Technical Metrics
| Metric | Description | Calculation |
|--------|-------------|-------------|
| **Pass Accuracy** | Successful passes / Total passes | % |
| **Dribble Success** | Successful dribbles / Attempts | % |
| **Shot Accuracy** | Shots on target / Total shots | % |
| **First Touch** | Quality of ball control | AI score 0-100 |
| **Ball Retention** | Time maintaining possession | seconds |
| **Progressive Carries** | Carries advancing >10m | count |

#### 🛡️ Defensive Metrics
| Metric | Description | Calculation |
|--------|-------------|-------------|
| **Tackles Won** | Successful tackles | count |
| **Interceptions** | Passes intercepted | count |
| **Blocks** | Shots/passes blocked | count |
| **Aerial Duels** | Headers won | % |
| **Pressures** | Pressing actions | count |
| **Recoveries** | Ball recoveries | count |
| **Defensive Actions/90** | Normalized defensive output | per 90 min |

#### 🧠 Intelligence Metrics
| Metric | Description | AI Analysis |
|--------|-------------|-------------|
| **Positioning Score** | Quality of off-ball positioning | 0-100 |
| **Space Creation** | Runs that create space | count |
| **Pressing Trigger** | Initiates team press | count |
| **Defensive Awareness** | Covers dangerous spaces | 0-100 |
| **Decision Making** | Correct option chosen | % |

### 🎬 Highlight Generation

| Feature | Description |
|---------|-------------|
| **Auto-Clip Extraction** | Extract best moments per player |
| **Action-Based Clips** | Goals, assists, tackles, dribbles |
| **Compilation Generator** | Create player highlight reels |
| **Timestamp Markers** | Navigate to specific actions |
| **Export Formats** | MP4, GIF, WebM |

### 📋 Scout Reports

| Report Type | Contents |
|-------------|----------|
| **Quick Overview** | Key stats, strengths, weaknesses |
| **Full Analysis** | Detailed breakdown by category |
| **Comparison Report** | Compare with similar players |
| **Development Report** | Track progress over time |
| **PDF Export** | Professional scout report |

---

## 🏗️ Architecture

```
GameAnalytics/
├── 📁 src/
│   ├── 📁 detection/              # Player & Ball Detection
│   │   ├── detector.py            # YOLO26 main detector
│   │   ├── ball_detector.py       # Specialized ball detection
│   │   ├── jersey_ocr.py          # Jersey number recognition
│   │   └── team_classifier.py     # Team classification by color
│   │
│   ├── 📁 tracking/               # Multi-Object Tracking
│   │   ├── tracker.py             # YOLO26 native + ByteTrack
│   │   ├── player_reid.py         # Player re-identification
│   │   ├── trajectory.py          # Trajectory smoothing
│   │   └── identity_manager.py    # Consistent ID assignment
│   │
│   ├── 📁 actions/                # Action Recognition
│   │   ├── action_classifier.py   # Main action classifier
│   │   ├── events/
│   │   │   ├── pass_detector.py   # Pass detection
│   │   │   ├── shot_detector.py   # Shot detection
│   │   │   ├── dribble_detector.py
│   │   │   ├── tackle_detector.py
│   │   │   └── aerial_detector.py
│   │   └── context_analyzer.py    # Game context analysis
│   │
│   ├── 📁 metrics/                # Performance Metrics
│   │   ├── physical/
│   │   │   ├── speed.py           # Speed & acceleration
│   │   │   ├── distance.py        # Distance covered
│   │   │   └── stamina.py         # Fatigue analysis
│   │   ├── technical/
│   │   │   ├── passing.py         # Pass metrics
│   │   │   ├── dribbling.py       # Dribble metrics
│   │   │   ├── shooting.py        # Shot metrics
│   │   │   └── ball_control.py    # First touch quality
│   │   ├── defensive/
│   │   │   ├── tackles.py         # Tackle analysis
│   │   │   ├── interceptions.py   # Interception analysis
│   │   │   └── positioning.py     # Defensive positioning
│   │   └── intelligence/
│   │       ├── positioning.py     # Off-ball movement
│   │       ├── decision.py        # Decision making
│   │       └── awareness.py       # Spatial awareness
│   │
│   ├── 📁 scout/                  # Scouting Module
│   │   ├── player_profile.py      # Player profile builder
│   │   ├── report_generator.py    # Scout report generation
│   │   ├── highlight_extractor.py # Clip extraction
│   │   ├── player_ranker.py       # Rank players by position
│   │   ├── comparison.py          # Player comparison
│   │   └── templates/             # Report templates
│   │       ├── quick_report.html
│   │       ├── full_report.html
│   │       └── pdf_template.html
│   │
│   ├── 📁 visualization/          # Visual Rendering
│   │   ├── annotator.py           # Video annotations
│   │   ├── heatmap.py             # Position heatmaps
│   │   ├── pitch.py               # Pitch drawing
│   │   ├── radar_chart.py         # Player radar charts
│   │   └── action_timeline.py     # Action timeline
│   │
│   └── 📁 utils/
│       ├── video.py               # Video processing
│       ├── geometry.py            # Homography & transforms
│       ├── config.py              # Configuration
│       └── database.py            # Player database
│
├── 📁 api/                        # REST API (TypeScript)
│   ├── 📁 src/
│   │   ├── index.ts
│   │   ├── 📁 routes/
│   │   │   ├── matches.ts         # Match management
│   │   │   ├── players.ts         # Player endpoints
│   │   │   ├── analysis.ts        # Analysis endpoints
│   │   │   ├── highlights.ts      # Clip endpoints
│   │   │   └── reports.ts         # Report endpoints
│   │   ├── 📁 services/
│   │   │   ├── matchService.ts
│   │   │   ├── playerService.ts
│   │   │   ├── analysisService.ts
│   │   │   ├── highlightService.ts
│   │   │   └── reportService.ts
│   │   └── 📁 types/
│   │       ├── player.ts
│   │       ├── match.ts
│   │       ├── metrics.ts
│   │       └── report.ts
│   ├── package.json
│   └── tsconfig.json
│
├── 📁 web/                        # React Dashboard
│   ├── 📁 src/
│   │   ├── 📁 components/
│   │   │   ├── MatchUploader.tsx
│   │   │   ├── TeamSelector.tsx
│   │   │   ├── PlayerCard.tsx
│   │   │   ├── PlayerProfile.tsx
│   │   │   ├── RadarChart.tsx
│   │   │   ├── Heatmap.tsx
│   │   │   ├── ActionTimeline.tsx
│   │   │   ├── HighlightPlayer.tsx
│   │   │   └── ComparisonView.tsx
│   │   ├── 📁 pages/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── MatchAnalysis.tsx
│   │   │   ├── PlayerScout.tsx
│   │   │   ├── Highlights.tsx
│   │   │   └── Reports.tsx
│   │   └── 📁 hooks/
│   │       ├── useAnalysis.ts
│   │       ├── usePlayer.ts
│   │       └── useHighlights.ts
│   ├── package.json
│   └── vite.config.ts
│
├── 📁 models/                     # AI Models
│   ├── yolo26_football.pt         # Player/ball detection
│   ├── jersey_ocr.pt              # Jersey number OCR
│   ├── action_classifier.pt       # Action recognition
│   ├── team_classifier.pt         # Team classification
│   └── player_reid.pt             # Re-identification
│
├── 📁 data/
│   ├── 📁 matches/                # Uploaded matches
│   ├── 📁 players/                # Player database
│   ├── 📁 highlights/             # Extracted clips
│   └── 📁 reports/                # Generated reports
│
├── 📁 notebooks/
│   ├── 01_detection_demo.ipynb
│   ├── 02_player_identification.ipynb
│   ├── 03_action_recognition.ipynb
│   ├── 04_metrics_analysis.ipynb
│   └── 05_scout_report_demo.ipynb
│
├── 📁 tests/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### Prerequisites

- Python 3.12+
- Node.js 20+
- CUDA 12.0+ (recommended for GPU)
- FFmpeg (for video processing)

### Quick Setup

```bash
# Clone repository
git clone https://github.com/username/Football-Stuff.git
cd Football-Stuff/Python/GameAnalytics

# Python environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download AI models
python scripts/download_models.py

# API & Dashboard
cd api && npm install
cd ../web && npm install

# Start all services
docker-compose up -d
```

---

## 💻 Usage

### 1. Upload & Analyze Match

```python
from src.scout import MatchAnalyzer, ScoutReport

# Initialize analyzer
analyzer = MatchAnalyzer(
    model_path="models/yolo26_football.pt",
    enable_gpu=True
)

# Analyze match - select team to scout
analysis = analyzer.analyze(
    video_path="match_porto_vs_benfica.mp4",
    home_team="FC Porto",
    away_team="SL Benfica",
    scout_team="FC Porto"  # Team to analyze
)

# Get top performers
top_players = analysis.get_top_players(limit=5)
for player in top_players:
    print(f"#{player.jersey} - Score: {player.scout_score:.1f}")
```

### 2. Generate Player Profile

```python
# Get detailed player profile
player = analysis.get_player(jersey_number=10)

print(f"Player #{player.jersey}")
print(f"Position: {player.detected_position}")
print(f"\n⚡ Physical:")
print(f"  Top Speed: {player.metrics.physical.top_speed:.1f} km/h")
print(f"  Distance: {player.metrics.physical.distance:.2f} km")
print(f"  Sprints: {player.metrics.physical.sprint_count}")
print(f"\n⚽ Technical:")
print(f"  Pass Accuracy: {player.metrics.technical.pass_accuracy:.1f}%")
print(f"  Dribble Success: {player.metrics.technical.dribble_success:.1f}%")
print(f"\n🛡️ Defensive:")
print(f"  Tackles Won: {player.metrics.defensive.tackles_won}")
print(f"  Interceptions: {player.metrics.defensive.interceptions}")
```

### 3. Extract Highlights

```python
from src.scout import HighlightExtractor

extractor = HighlightExtractor(analysis)

# Get all highlights for a player
highlights = extractor.get_player_highlights(
    jersey_number=10,
    actions=["goal", "assist", "dribble", "key_pass"],
    max_clips=10
)

# Export compilation
extractor.create_compilation(
    player_jersey=10,
    output_path="highlights/player_10_compilation.mp4",
    include_stats_overlay=True
)
```

### 4. Generate Scout Report

```python
from src.scout import ScoutReport

report = ScoutReport(player)

# Generate full report
report.generate(
    output_path="reports/player_10_scout_report.pdf",
    include_highlights=True,
    include_heatmap=True,
    include_radar=True,
    comparison_players=["similar_player_1", "similar_player_2"]
)
```

---

## 🔌 API Reference

### Match Endpoints

#### `POST /api/matches/upload`
Upload match video for analysis.

```typescript
// Request
POST /api/matches/upload
Content-Type: multipart/form-data

{
  video: File,
  homeTeam: "FC Porto",
  awayTeam: "SL Benfica",
  scoutTeam: "FC Porto",
  competition: "Liga Portugal",
  date: "2026-01-15"
}

// Response
{
  matchId: "match_abc123",
  status: "processing",
  estimatedTime: 300
}
```

#### `GET /api/matches/:matchId/players`
Get all detected players.

```typescript
// Response
{
  matchId: "match_abc123",
  scoutTeam: "FC Porto",
  players: [
    {
      id: "player_1",
      jerseyNumber: 10,
      detectedPosition: "CAM",
      minutesPlayed: 90,
      scoutScore: 8.7,
      highlights: 12
    },
    // ...
  ]
}
```

### Player Endpoints

#### `GET /api/players/:playerId/profile`
Get complete player profile.

```typescript
// Response
{
  player: {
    id: "player_1",
    jerseyNumber: 10,
    detectedPosition: "CAM",
    metrics: {
      physical: {
        topSpeed: 32.4,
        avgSpeed: 7.2,
        distance: 11.3,
        sprints: 24,
        highIntensityRuns: 48
      },
      technical: {
        passAccuracy: 87.5,
        passesCompleted: 42,
        dribbleSuccess: 71.4,
        shotsOnTarget: 3,
        keyPasses: 4
      },
      defensive: {
        tacklesWon: 3,
        interceptions: 2,
        blocks: 1,
        pressures: 18,
        recoveries: 5
      },
      intelligence: {
        positioningScore: 82,
        decisionMaking: 78,
        spaceCreation: 6
      }
    },
    scoutScore: 8.7,
    strengths: ["Passing", "Vision", "Dribbling"],
    weaknesses: ["Aerial Duels", "Defensive Work"]
  }
}
```

### Highlight Endpoints

#### `GET /api/highlights/:playerId`
Get player highlights.

```typescript
// Response
{
  playerId: "player_1",
  highlights: [
    {
      id: "clip_1",
      action: "goal",
      timestamp: "34:12",
      duration: 15,
      url: "/highlights/clip_1.mp4",
      thumbnail: "/highlights/clip_1_thumb.jpg"
    },
    // ...
  ]
}
```

#### `POST /api/highlights/:playerId/compilation`
Generate highlight compilation.

```typescript
// Request
{
  actions: ["goal", "assist", "dribble"],
  maxClips: 10,
  includeStats: true
}

// Response
{
  compilationId: "comp_xyz",
  status: "generating",
  estimatedTime: 60
}
```

### Report Endpoints

#### `POST /api/reports/generate`
Generate scout report.

```typescript
// Request
{
  playerId: "player_1",
  reportType: "full",  // "quick" | "full" | "comparison"
  format: "pdf",
  includeHighlights: true,
  comparisonPlayers: ["player_2", "player_3"]
}

// Response
{
  reportId: "report_123",
  status: "generating",
  downloadUrl: null  // Available when complete
}
```

---

## 📋 Scout Reports

### Quick Report Example

```
┌─────────────────────────────────────────────────────────────┐
│                    SCOUT REPORT - QUICK                      │
├─────────────────────────────────────────────────────────────┤
│  Player: #10                    Position: CAM                │
│  Match: FC Porto vs SL Benfica  Date: 2026-01-15            │
│  Minutes: 90                    Scout Score: 8.7/10         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ⚡ PHYSICAL          ⚽ TECHNICAL        🛡️ DEFENSIVE       │
│  ─────────────       ─────────────      ─────────────       │
│  Speed: 32.4 km/h    Pass Acc: 87.5%   Tackles: 3           │
│  Distance: 11.3 km   Dribbles: 71.4%   Interceptions: 2     │
│  Sprints: 24         Key Passes: 4     Pressures: 18        │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│  ✅ STRENGTHS                  ❌ AREAS TO IMPROVE           │
│  • Excellent passing range     • Aerial presence             │
│  • Creative vision             • Defensive contribution      │
│  • Dribbling in tight spaces   • Work rate without ball      │
├─────────────────────────────────────────────────────────────┤
│  📝 SCOUT NOTES                                              │
│  Technically gifted playmaker with excellent vision.         │
│  Creates chances consistently. Would benefit from            │
│  improving defensive work rate for pressing systems.         │
└─────────────────────────────────────────────────────────────┘
```

### Radar Chart Comparison

```
                    Pace
                     100
                      │
                     80
                      │
         Defending   60        Shooting
              ╲      │       ╱
               ╲    40      ╱
                ╲   │     ╱
                 ╲ 20    ╱
                  ╲│   ╱
    Physical ──────┼──────── Dribbling
                  ╱│╲
                 ╱ │ ╲
                ╱  │  ╲
               ╱   │   ╲
              ╱    │    ╲
         Passing   │    Vision
                      
    ─── Player #10  ─── League Average
```

---

## 📈 Metrics & KPIs

### Scout Score Calculation

The **Scout Score** (0-10) is calculated using weighted metrics:

```python
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
| CDM | 20% | 25% | 35% | 20% |
| CM | 20% | 35% | 20% | 25% |
| CAM | 15% | 40% | 10% | 35% |
| WNG | 35% | 35% | 10% | 20% |
| ST | 25% | 40% | 5% | 30% |

---

## 🗺️ Roadmap

### v1.0 (Current)
- [x] Player detection with YOLO26
- [x] Jersey number OCR
- [x] Team classification
- [x] Basic metrics calculation
- [x] Highlight extraction

### v1.1 (Q1 2026)
- [ ] Action recognition (pass, shot, dribble, tackle)
- [ ] Advanced metrics computation
- [ ] Scout report generation
- [ ] Player comparison tool

### v1.2 (Q2 2026)
- [ ] Player database & history
- [ ] Multi-match aggregation
- [ ] Similar player finder
- [ ] Market value estimation

### v2.0 (Q3 2026)
- [ ] Real-time analysis (live matches)
- [ ] Integration with external data (Transfermarkt, WhoScored)
- [ ] Mobile app for scouts
- [ ] API for clubs

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Test specific modules
pytest tests/test_detection.py -v
pytest tests/test_metrics.py -v
pytest tests/test_scout.py -v

# Coverage report
pytest --cov=src --cov-report=html
```

---

## 📄 License

MIT License - see [LICENSE](../LICENSE) for details.

---

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO26
- [SoccerNet](https://www.soccer-net.org/) — Dataset & Benchmarks
- [Roboflow](https://roboflow.com/) — Annotation tools
- [FastReID](https://github.com/JDAI-CV/fast-reid) — Re-identification

---

<div align="center">

**⭐ Star this repo if it helped you!**

*Built for scouts, by football lovers* ⚽

</div>
