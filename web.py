import pandas as pd
import streamlit as st

import predict
from utils import load_data

st.set_page_config(page_title="Regression Model Comparison", layout="wide")
st.title("Car MPG Prediction - Model Comparison")
st.markdown(
    "Repository https://github.com/tayaee/linear-regression-ridge-lasso-elasticnet"
)

# Load data to get feature names and min/max for inputs
X, _ = load_data()
feature_names = X.columns
stats = X.describe()

with st.sidebar:
    st.header("Input Features")
    inputs = {}
    for feature in feature_names:
        min_val = float(stats[feature]["min"])
        max_val = float(stats[feature]["max"])
        mean_val = float(stats[feature]["mean"])

        inputs[feature] = st.slider(
            label=feature, min_value=min_val, max_value=max_val, value=mean_val
        )

if st.button("Predict"):
    # Create dataframe for input
    input_df = pd.DataFrame([inputs])

    # Load models
    models = predict.load_models()

    if not models:
        st.error("No models found in data/ directory. Please run train.py first.")
    else:
        # 1. Display Coefficients
        st.subheader("Model Coefficients Comparison")
        try:
            coef_df = predict.get_coefficients_df(models, feature_names)

            # Method 1
            # st.dataframe(coef_df)

            # Method 2
            coef_df = coef_df.set_index("Feature")
            st.dataframe(coef_df.style.highlight_max(axis=1))

            # Method 3
            # cols_to_highlight = [col for col in coef_df.columns if col != "Feature"]
            # st.dataframe(coef_df.style.highlight_max(axis=1, subset=cols_to_highlight))

        except Exception as e:
            st.error(f"Error displaying coefficients: {e}")

        # 2. Display Predictions
        st.subheader("Prediction Results")
        predictions = predict.predict_all(models, input_df)

        # Display as metrics or table
        pred_df = pd.DataFrame(
            list(predictions.items()), columns=["Model", "Predicted MPG"]
        )

        # Sort by Predicted MPG in descending order
        # Ensure we only sort if values are comparable (e.g. all numeric), otherwise fallback or handle errors
        # Assuming most come back as floats:
        pred_df = pred_df.sort_values(by="Predicted MPG", ascending=False)

        # Format Predicted MPG
        pred_df["Predicted MPG"] = pred_df["Predicted MPG"].apply(
            lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
        )

        st.table(pred_df)

        # Determine best prediction? (Can't know without actual usage, but can show range)

st.markdown("---")
st.info("Train models offline using `train.py`.")
