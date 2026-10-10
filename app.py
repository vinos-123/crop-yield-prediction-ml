
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

# -----------------------------
# Load Trained Model
# -----------------------------
MODEL_PATH = Path(__file__).parent / "crop_yield_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the trained model: {e}")
    st.stop()

# -----------------------------
# Application Header
# -----------------------------
st.title("🌾 Crop Yield Prediction System")
st.markdown(
    "### Machine Learning-Based Agricultural Yield Estimation"
)
st.write(
    "Enter crop and environmental details to estimate crop yield "
    "using the trained machine learning model."
)

st.divider()

# -----------------------------
# Input Form
# -----------------------------
st.subheader("Enter Agricultural Details")

with st.form("crop_prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        crop = st.selectbox(
            "Crop Type",
            ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane"]
        )

        region = st.selectbox(
            "Region",
            [
                "Telangana",
                "Punjab",
                "Karnataka",
                "Maharashtra",
                "Uttar Pradesh"
            ]
        )

        season = st.selectbox(
            "Growing Season",
            ["Kharif", "Rabi", "Summer"]
        )

    with col2:
        year = st.number_input(
            "Cultivation Year",
            min_value=2000,
            max_value=2100,
            value=2024,
            step=1
        )

        area = st.number_input(
            "Cultivated Area",
            min_value=0.1,
            value=1200.0,
            step=10.0
        )

        rainfall = st.number_input(
            "Rainfall",
            min_value=0.0,
            value=950.0,
            step=10.0
        )

    with col3:
        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=28.0,
            step=0.5
        )

        fertilizer = st.number_input(
            "Fertilizer Usage",
            min_value=0.0,
            value=150.0,
            step=5.0
        )

        pesticide = st.number_input(
            "Pesticide Usage",
            min_value=0.0,
            value=1.2,
            step=0.1
        )

    submitted = st.form_submit_button(
        "🌱 Predict Crop Yield",
        use_container_width=True
    )

# -----------------------------
# Prediction
# -----------------------------
if submitted:
    input_data = pd.DataFrame([{
        "Crop": crop,
        "Region": region,
        "Season": season,
        "Year": int(year),
        "Area": area,
        "Rainfall": rainfall,
        "Temperature": temperature,
        "Fertilizer": fertilizer,
        "Pesticide": pesticide
    }])

    try:
        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                label="Predicted Crop Yield",
                value=f"{prediction:.2f} tonnes/hectare"
            )

        with result_col2:
            st.info(
                f"Selected crop: **{crop}**\n\n"
                f"Region: **{region}**\n\n"
                f"Season: **{season}**"
            )

        st.caption(
            "This is a machine-learning estimate based on the training "
            "dataset. Actual agricultural yield may differ."
        )

        with st.expander("View submitted input details"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -----------------------------
# Model Information
# -----------------------------
st.divider()
st.subheader("About This Project")

info1, info2, info3 = st.columns(3)

info1.metric("Model", "Gradient Boosting")
info2.metric("Input Features", "9")
info3.metric("Task", "Regression")

st.write(
    "The project compares Ridge Regression, Random Forest Regression, "
    "and Gradient Boosting Regression to estimate crop yield."
)

st.warning(
    "The current dataset contains only 30 records. Predictions are "
    "for educational demonstration and should not be used as reliable "
    "agricultural advice without validation on larger real-world data."
)
