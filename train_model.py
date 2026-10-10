
import os
import pandas as pd
import joblib
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Scikit-learn version:", sklearn.__version__)

# Load the dataset
df = pd.read_csv("crop_yield_dataset.csv")
print("Dataset shape:", df.shape)

categorical = ["Crop", "Region", "Season"]
numeric = [
    "Year", "Area", "Rainfall",
    "Temperature", "Fertilizer", "Pesticide"
]
target = "Yield"

required = categorical + numeric + [target]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

X = df[categorical + numeric].copy()
y = pd.to_numeric(df[target], errors="coerce")

valid = y.notna()
X = X.loc[valid].copy()
y = y.loc[valid]

for col in numeric:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor = ColumnTransformer([
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical),
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))
print("RMSE:", mean_squared_error(y_test, predictions) ** 0.5)
print("R2:", r2_score(y_test, predictions))

# Save a model compatible with this local environment
joblib.dump(model, "crop_yield_model.joblib")

# Verify the saved model
loaded_model = joblib.load("crop_yield_model.joblib")

print("Model saved and loaded successfully!")
print("Model file size:", os.path.getsize("crop_yield_model.joblib"), "bytes")

# Save evaluation results
pd.DataFrame([{
    "Model": "Gradient Boosting",
    "MAE": mean_absolute_error(y_test, predictions),
    "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
    "R2": r2_score(y_test, predictions)
}]).to_csv("model_performance.csv", index=False)

print("Performance results saved successfully!")
