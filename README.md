# Traffic Accidents Analysis and Prediction

This project is a Streamlit app for exploring traffic accident patterns and predicting crash severity.

## Features

- Data cleaning and preprocessing
- Exploratory visual analysis (time, weather, road, injury, and causes)
- Crash severity classification with Decision Tree and Random Forest
- Live prediction UI based on encoded feature inputs

## Project Files

- `app.py`: Main Streamlit application
- `traffic_accidents.csv`: Input dataset used by the app
- `requirements.txt`: Python dependencies
- `cleanedData.ipynb`: Notebook with related analysis work

## Setup

1. Create and activate a virtual environment:
   - macOS/Linux:
     - `python3 -m venv .venv`
     - `source .venv/bin/activate`

2. Install dependencies:
   - `pip install -r requirements.txt`

## Run

From the project root, run:

`streamlit run app.py`

Then open the local URL shown in the terminal (usually `http://localhost:8501`).
