# 🌾 Crop Yield Prediction Using Machine Learning

## 📌 Project Overview

Crop Yield Prediction is a Machine Learning project developed using Python. The system predicts agricultural crop yield based on factors such as crop type, region, season, year, cultivated area, rainfall, temperature, fertilizer usage, and pesticide usage.

The project compares multiple Machine Learning regression algorithms and selects the best-performing model based on evaluation metrics.

---

## 🎯 Objectives

- Predict crop yield using Machine Learning.
- Analyze agricultural and environmental factors affecting crop production.
- Compare different regression algorithms.
- Identify the best-performing prediction model.
- Provide a data-driven approach for agricultural planning.

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Joblib
- Google Colab
- GitHub

---

## 🤖 Machine Learning Algorithms

The project uses and compares the following algorithms:

1. Ridge Regression
2. Random Forest Regression
3. Gradient Boosting Regression

---

## 📊 Dataset Features

| Feature | Description |
|---|---|
| Crop | Type of crop |
| Region | Agricultural region/state |
| Season | Growing season |
| Year | Crop production year |
| Area | Cultivated area |
| Rainfall | Rainfall amount |
| Temperature | Temperature |
| Fertilizer | Fertilizer usage |
| Pesticide | Pesticide usage |
| Yield | Crop yield (Target Variable) |

---

## 📈 Model Evaluation

The models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- 5-Fold Cross Validation

The model with the best R² performance is selected as the final prediction model.

---

## ⚙️ How to Run the Project

### Using Google Colab

1. Open the `crop_yield_prediction.ipynb` notebook in Google Colab.
2. Upload `crop_yield_dataset.csv`.
3. Run all the cells.
4. Select the CSV dataset option when prompted.
5. The Machine Learning models will be trained.
6. Model performance will be displayed.
7. The best-performing model will be selected.
8. A sample crop yield prediction will be generated.

### Using Python Locally

Install the required libraries:

```bash
pip install -r requirements.txt
