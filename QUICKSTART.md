# GameAnalytics - Football Intelligent Scout System

AI-powered player scouting system using computer vision and machine learning.

## Features

- Player detection and tracking (YOLO)
- Jersey number recognition
- Player re-identification across frames
- Performance metrics calculation
- Scout report generation
- Heat maps and action zones

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/GameAnalytics.git
cd GameAnalytics

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run
python main.py
```

## Project Structure

```
GameAnalytics/
    main.py              # Entry point
    requirements.txt     # Dependencies
    src/
        detection/       # YOLO player detection
        tracking/        # Multi-object tracking
        scout/           # Scout report generation
        utils/           # Utility functions
```

## Documentation

See [README.md](README.md) for full documentation.

## License

MIT License
