# Pitch-Predict AI ⚽🤖

Pitch-Predict AI is a highly sophisticated, machine-learning-driven match outcome engine. By synthesizing historical databases, advanced head-to-head metrics, and real-time team form, Pitch-Predict AI produces accurate probabilities for football (soccer) match outcomes (Home Win, Draw, Away Win).

## Project Overview

Modern football prediction requires more than just analyzing point standings. Pitch-Predict AI engineers momentum factors, long-term defensive and offensive metrics, and real-world historical archives to power an XGBoost machine-learning core. The system analyzes the user's input of recent team form and matches it against extensive historical records to ensure maximum predictive accuracy while robustly avoiding data leakage.

## Architecture

At a high level, the architecture is split into a robust **Data Ingestion** layer and an advanced **Match Prediction** core.

1. **Data Ingestion (`src/data_ingestion/`)**: Connects to the primary historical SQLite database. Fetches past head-to-head results, goal differences, match outcomes, and broader historical patterns.
2. **Match Prediction Engine (`src/match_prediction/`)**:
   - **Feature Pipeline & Merging**: Transforms raw metrics into time-sealed, rolling-window historical features, ready for Machine Learning ingestion.
   - **Predictor Model**: A tuned XGBoost classifier trained on the prepared features.
   - **Model Evaluation**: Generates precision, recall, accuracy, and detailed feature importance weighting.
   - **User Input Interface**: An interactive Command-Line Interface (CLI) that dynamically parses recent forms into game-ready prediction probabilities.

## Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd pitch-predict-ai
   ```

2. **Setup the Environment**
   Using the built-in Makefile, install all core dependencies, linters, and testing frameworks:
   ```bash
   make install
   ```

3. **Ensure Data Availability**
   Place your database in `data/raw/european_database.sqlite`.

## Usage Guidelines

To start the AI Predictor Engine CLI:

```bash
python3 main_app.py
```

The system will:
1. Connect to the historical database.
2. Engineer features and train the core ML Model.
3. Provide an AI evaluation report.
4. Launch an interactive prompt where you can provide Home/Away teams and their recent forms (in `GF-GC` format) to receive an instant probability outcome!

## Development & Code Quality

This project is built to a professional standard. Code formatting, linting, and tests can be executed seamlessly via the Makefile.

```bash
make test     # Runs the pytest suite
make lint     # Runs the Ruff linter
make format   # Formats the codebase with Black and Ruff
make clean    # Cleans up pycache and pyc files
```