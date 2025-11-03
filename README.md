# MLB Career Prediction Project

Predicting long-term MLB career success from early career performance using Statcast data.

## Project Overview
This project analyzes whether a player's first 500 plate appearances can predict their career outcomes (WAR, career length, etc.).

## Setup
```bash
# Clone repository
git clone <repo-url>
cd mlb_project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# 1. Download data
python scripts/download_data.py

# 2. Run full pipeline
python scripts/run_pipeline.py

# 3. Train model
python scripts/train_model.py
```

## Project Structure
See directory tree above.

## Configuration
Edit `config/config.yaml` to modify:
- Number of plate appearances to analyze
- Minimum career PA threshold
- Model hyperparameters

## Contributors
