# Crop Yield Prediction Using Machine Learning

## Project Overview

Crop Yield Prediction is a machine learning project that predicts agricultural crop yield based on factors such as crop type, region, season, year, cultivated area, rainfall, temperature, fertilizer usage, and pesticide usage.

The project compares multiple machine learning algorithms and selects the best-performing model based on evaluation metrics.

## Objectives

- Predict crop yield using machine learning.
- Analyze important agricultural factors affecting crop yield.
- Compare different regression algorithms.
- Evaluate model performance using MAE, RMSE, and R² score.
- Develop a reusable trained machine learning model.

## Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## Machine Learning Algorithms

The following regression algorithms were evaluated:

1. Ridge Regression
2. Random Forest Regression
3. Gradient Boosting Regression

## Dataset Features

The dataset contains the following attributes:

| Feature | Description |
|---|---|
| Crop | Type of crop |
| Region | Agricultural region |
| Season | Crop growing season |
| Year | Year of cultivation |
| Area | Cultivated area |
| Rainfall | Rainfall amount |
| Temperature | Temperature |
| Fertilizer | Fertilizer usage |
| Pesticide | Pesticide usage |
| Yield | Crop yield |

## Model Performance

The best-performing model in the current experiment was:

**Gradient Boosting Regression**

Test-set results:

- MAE: 0.1089
- RMSE: 0.1460
- R² Score: 0.9777
- Cross-validation R²: 0.8733

The dataset currently contains 30 records, so these results should be considered a demonstration rather than a definitive measure of real-world performance.

## Sample Prediction

The trained model predicted approximately:

```text
3.45 tonnes/hectare