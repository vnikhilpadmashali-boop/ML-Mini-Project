# VINIX ML Mini Project – Smart Digital Wellness Analyzer

## Project overview
A Streamlit application that explores smartphone usage patterns and uses a Random Forest machine-learning classifier to estimate a usage category.

## Technologies
- Python
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib
- SQLite

## Features
- CSV upload and dataset validation
- Dataset overview and category charts
- Random Forest classification model
- Train/test accuracy and classification report
- Confusion matrix
- Interactive usage-category prediction
- SQLite prediction history
- Dataset preview and CSV download

## Dataset
The included sample file is based on the smartphone-usage dataset supplied for this project. The CSV must contain a `Usage_Category` target column. ID columns such as `User_ID` are excluded from model inputs.

## Run the application
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Notes
This is an educational demonstration, not a medical or clinical assessment. Model performance depends on dataset size, class balance, and data quality.
