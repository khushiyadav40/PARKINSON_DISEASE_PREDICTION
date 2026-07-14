# Parkinson's Disease Prediction

An internship group project that predicts whether a person has Parkinson's
disease based on voice measurement data, using a machine learning
classification model.

## About the Project

This project uses biomedical voice measurements to classify whether a
person is healthy or has Parkinson's disease. It covers the full pipeline:
data analysis, model training, evaluation, and a simple web interface for
predictions.

## Dataset

**UCI Parkinson's Disease Data Set**
- 195 rows, 22 numeric voice-measurement features
- Target column: `status` (1 = Parkinson's, 0 = Healthy)
- Class distribution: 147 Parkinson's cases, 48 healthy cases (imbalanced)

## Project Structure

```
Parkinson_Disease_Prediction/
│
├── data/
│   └── parkinsons.csv          # Dataset
│
├── notebooks/
│   └── Parkinson_Data_Analysis.ipynb   # EDA & data analysis (Task 1 & 2)
│
├── models/
│   ├── parkinsons_model.pkl    # Trained model (created by train_model.py)
│   └── scaler.pkl              # Feature scaler (created by train_model.py)
│
├── train_model.py              # Model training script (Task 3)
├── app.py                      # Streamlit prediction app (Task 4)
├── requirements.txt            # Required libraries
└── README.md                   # This file
```

## Libraries Used

- pandas, numpy — data handling
- scikit-learn — machine learning models and evaluation
- matplotlib, seaborn — visualization
- streamlit — web interface
- joblib — saving/loading the trained model

