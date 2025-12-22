# FPL Predictor 

Machine Learning system and weighted formula for predicting Fantasy Premier League player performance for the upcoming gameweek.

Course: Datascience and Advanced Programming 2025-2026
Author: Evan Dejean 
Student ID: 21427794
Email: [evan.dejean@unil.ch]


## Research Question

Which Fantasy Premier League players will score the most points in the next gameweek, based on historical performance data, recent form, and upcoming fixture difficulty? And which ones have the best points-to-price ratio?


## Setup

```bash
# Clone the repository:

git clone https://github.com/2gens/fpl_project.git
cd fpl_project


# Create and activate a virtual environment:

python -m venv .venv
.\.venv\Scripts\Activate.ps1


# Install dependencies:

pip install -r requirements.txt
```

## Usage

```bash
Run the complete pipeline: python main.py
```
This executes:
1. Data collection from the FPL API
2. Data preprocessing and feature engineering
3. Model training (5 algorithms)
4. Exploratory data analysis (7 visualizations)
5. Point predictions for next gameweek
6. Results saved to data/predictions/   -> predictions_formula.csv (for the weighted formula)
                                        -> predictions_ml.csv (for the Machine Learning system)

```bash 
Run the interactive dashboard: streamlit app.py 
```                                        


## Project Structure

fpl_project/                                

    data/                                    
        predictions/                        -> Models output (.csv)
        processed/                          -> Cleaned data (.csv)
        raw/                                -> FPL API data (.jason)

    examples/
        test_collection.py                  -> Run data_collections.py
        test_eda.py                         -> Run eda.py
        test_models.py                      -> Run models.py
        test_prediction_comparaison.py      -> Run predictor.py
        test_processing.py                  -> Run data_preprocessing.py

    results/ 
        figures/                            -> 7 EDA plots
        models/                             -> Training models + metrics (.pkl)

    src/ 
        data_collections.py                 -> Collecting FPL API data 
        data_preprocessing.py               -> Cleaning data
        eda.py                              -> Exploratory Data Analysis
        models.py                           -> Ml models training (3 mains)
        predictor.py                        -> Prediction (ML + Formula)
    
    tests/                                  -> Unit tests

    app.py                                  -> Web interface
    main.py                                 -> Entry point (run all the project)
    PROPOSAL.md                             -> Proposal 
    README.md                               -> Readme
    requirements.txt                        -> Python dependencies



## Data

Source: Official Fantasy Premier League API
Endpoint: https://fantasy.premierleague.com/api/bootstrap-static/
Access: Public, no authentication required
Size: 750+ players, 20 teams, fixture information
Update frequency: After each gameweek

The data includes player statistics (goals, assists, minutes played), team information, upcoming fixtures, and various performance metrics.


## Results

Model performance metrics are saved in results/models/results_summary.json.

XGBoost demonstrated the best performance among individual models. The ensemble approach provides more stable predictions than any single model.

Key predictive features (by importance):
1. Form (recent 30-day performance)
2. ICT Index (influence, creativity, threat combined)
3. Expected points per 90 minutes
4. Next fixture difficulty
5. Momentum (form trend)

Predictions for the next gameweek are saved in data/predictions/ with columns for player name, team, position, price, and predicted points. There are two rankings of players with the best potential : one in data/predictions/predictions_formula.csv for the wighted formula system and one in data/predictions/predictions_ml.csv for the machine learning system. 

Detailed analysis and interpretation are provided in the technical report.


## Limitations

- Predictions based on historical data may not capture last-minute team news
- Player rotation and tactical decisions are difficult to predict
- Random match events (red cards, injuries during play) cannot be anticipated
- Model trained on current season data only


## Requirements 
```bash
# Core data manipulation
pandas==2.2.3
numpy==2.1.3

# Machine Learning
scikit-learn==1.6.0
xgboost==2.1.3

# Visualization
matplotlib==3.9.2
seaborn==0.13.2

# Data collection
requests==2.32.3

# Testing
pytest==8.3.3

# Web interface
streamlit==1.40.1

```

## AI Tools Disclosure

AI assistants were used during development:
- Claude : Code debugging, optimization suggestions
- GitHub Copilot: Code completion
- ChatGpt : Answer some problems


## Contact

Author: [Evan Dejean]
Email: [evan.dejean@unil.ch]
GitHub: [2gens]
Institution: HEC Lausanne