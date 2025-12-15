# US Real Estate Valuation Predictor & Web Analyzer

## Project Overview

This project implements an end-to-end Machine Learning (ML) pipeline designed to predict the fair market value of residential properties using a large US real estate dataset (~1 million entries). The goal is to determine if a public listing (e.g., from Zillow) is currently overvalued or undervalued based on its physical characteristics and location metrics.

This solution is integrated into a Gradio web interface for real-time analysis, demonstrating the full lifecycle from data scraping to prediction deployment.

_This is the SIM DAC (2025-26) Internal Project from Group 6._

## Project Objectives

* High-Accuracy Regression: Train and tune highly effective ensemble models (XGBoost and Gradient Boosting) on structured tabular data.

* Advanced Feature Engineering: Implement a robust Target Encoding strategy to handle high-cardinality categorical features (like City and State) without introducing data leakage or high dimensionality.

* End-to-End Deployment: Create a consumer-facing web application that integrates web scraping, data preprocessing, model prediction, and visualization in a seamless workflow.

## Model & Pipeline Architecture

The core of the project is a comparison between two leading tree-based regression models, managed via a Scikit-learn Pipeline.

1. Feature Preprocessing Highlights

To ensure model accuracy and stability, we use a custom preprocessing flow:

Categorical Features (City/State): We do NOT use simple Label Encoding, which imposes false order. Instead, we use Target Encoding (TE) with K-Fold cross-validation to replace categorical labels with the average price of the target variable for that category. This provides a strong, numeric signal while preventing data leakage.

Numerical Features: All numerical features (including the new Target Encoded columns) are scaled using StandardScaler to ensure the model converges efficiently.

Pipeline: The StandardScaler and the final predictive model (XGBoost/Gradient Boosting) are chained together using a Scikit-learn Pipeline for consistent training and prediction.

2. Predictive Models

We use hyperparameter tuning (GridSearchCV) to find the optimal configuration for two powerful ensemble methods:

XGBoost Regressor (XGBR): Often the state-of-the-art model for structured data competitions.

Gradient Boosting Regressor (GBR): A robust baseline model for comparison.

## Technical Workflow and Data Flow

The final Gradio application follows a precise, sequential logic:

__Input:__ User provides a Zillow URL.

__Scraping:__ Python logic extracts raw features: Price (Actual), City, State, Lot Size, House Size, Beds, Baths.

__Encoding__: The saved Target Encoders (fitted ONLY on training data) are applied to the raw city and state values.

__Scaling & Prediction:__ The data is fed into the best-performing Pipeline (e.g., best_xgboost_pipeline.joblib), which automatically scales the features and outputs $\mathbf{P_{\text{predicted}}}$.

__Valuation Chart:__ A function compares $\mathbf{P_{\text{actual}}}$ to $\mathbf{P_{\text{predicted}}}$, calculates the over/undervaluation percentage, and generates a visual Bullet Chart.

__Setup and Installation__

Prerequisites

Python 3.8+

Access to a Zillow URL for testing (scraping single pages is used for demonstration).

Dependencies

Install the required libraries:

pip install pandas scikit-learn xgboost category-encoders gradio matplotlib plotly joblib


## Key Project Files

__File Name__

* __Purpose__

_final_city_encoder.pkl
final_state_encoder.pkl_

* Saved Python object containing the fitted Target Encoder mappings for the City and State features.

_best_model_pipeline.pkl_

* Saved final model object (includes the scaler and the XGBoost model combined).

Usage Instructions (Once Deployed)

Run the Gradio application (e.g., python app_gradio.py).
