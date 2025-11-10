import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import os
from pathlib import Path

# ---------------------------
# Page Config
# ---------------------------
st.markdown('<div class ="header-banner"> 💧 Water Potability Predictor</div>', unsafe_allow_html=True)
# Custom CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom, #e0f7fa, #ffffff);
}
.header-banner {
    background: linear-gradient(90deg, #2196f3, #00bcd4);
    padding: 20px;
    border-radius: 12px;
    color: white;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.card {
    padding: 15px;
    border-radius: 12px;
    background-color: #f0f4f8;
    margin-bottom: 15px;
    box-shadow: 0 3px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)
# ---------------------------
# Header + About Section
# ---------------------------

st.markdown("""
### 👥 Who We Are
**We are NaN Masters** — a curious crew of data explorers, problem solvers, and change-makers.
Our mission? To turn raw data into real-world impact.

We dive deep into numbers, patterns, and possibilities to help communities access **clean, safe, and sustainable water**.
For us, every dataset tells a story — and we use technology to make that story count.

With curiosity as our compass and innovation as our toolkit,
we’re here to prove that even from **NaN (Not a Number)** beginnings, great solutions can flow.
""")

# ---------------------------
# Problem Overview
# ---------------------------
st.markdown("---")
st.subheader("🌍 Why This Matters")
st.markdown("""
Many communities worldwide lack access to **safe drinking water**, and testing is often **slow or unavailable**.
Our model predicts whether water is safe or unsafe to drink using **chemical properties like pH, turbidity, hardness, and more**.

This helps communities and municipalities **take quick, data-driven action** to ensure water safety.
""")

# --- Manual Feature Engineering Logic (replicated from training) ---
def calculate_safety_and_contamination_scores(df):
    """Calculates 'safety_score' and 'contamination_risk' features."""
    temp_df = df.copy() # Work on a copy

    # Ensure required columns exist before calculating scores
    required_cols_safety_score = ['ph', 'Turbidity', 'Trihalomethanes', 'Chloramines', 'Organic_carbon']
    if all(col in temp_df.columns for col in required_cols_safety_score):
        temp_df['safety_score'] = (
            (temp_df['ph'].between(6.5, 8.5).astype(int) * 2) +
            (temp_df['Turbidity'] < 5).astype(int) +
            (temp_df['Trihalomethanes'] < 80).astype(int) +
            (temp_df['Chloramines'] > 2).astype(int) +
            (temp_df['Organic_carbon'] < 15).astype(int)
        )
    else:
        temp_df['safety_score'] = 0 # Default if columns are missing

    required_cols_contamination_risk = ['Organic_carbon', 'Turbidity', 'Trihalomethanes', 'ph']
    if all(col in temp_df.columns for col in required_cols_contamination_risk):
        temp_df['contamination_risk'] = (
            temp_df['Organic_carbon'] * 0.3 +
            temp_df['Turbidity'] * 0.25 +
            (temp_df['Trihalomethanes'] > 80).astype(int) * 20 +
            (temp_df['ph'] < 6.5).astype(int) * 15
        )
    else:
        temp_df['contamination_risk'] = 0 # Default if columns are missing

    # Handle potential NaNs created during score calculation if base columns had NaNs
    for col in ['safety_score', 'contamination_risk']:
         if col in temp_df.columns:
              temp_df[col].fillna(temp_df[col].median() if not temp_df[col].median().isnull() else 0, inplace=True) # Impute with median or 0 if median is NaN


    # Return only the newly created features
    return temp_df[['safety_score', 'contamination_risk']]
# --- End Manual Feature Engineering Logic ---


# ---------------------------
# Prediction Mode
# ---------------------------
@st.cache_resource
def load_model():
    try:
        # Get script directory
        current_dir = Path(__file__).parent

        # Use correct relative path for the model trained on scores only
        # Make sure you save a pipeline trained ONLY on scaled safety_score and contamination_risk
        model_path = current_dir / "rf_pipeline_scores_only.pkl" # Suggested new model file name

        # Debug output
        st.write(f"Loading model from: {model_path}")
        st.write(f"File exists: {os.path.exists(model_path)}")

        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found at: {model_path}")
            st.info("Please ensure a pipeline trained on 'safety_score' and 'contamination_risk' is saved as 'rf_pipeline_scores_only.pkl'")
            return None

        with open(model_path, 'rb') as f:
            # Assuming the loaded pipeline includes the StandardScaler and the model
            loaded_pipeline = joblib.load(f)
            st.success("✅ Model loaded successfully!")
            return loaded_pipeline

    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# Load the model pipeline
model_pipeline = load_model()

st.markdown("---")
st.subheader("🚀 Choose a Prediction Mode")
mode = st.radio(
    "Select how you'd like to predict water safety:",
    ("🔹 Manual Input", "📂 Batch CSV Upload"),
    horizontal=True
)
# ---------------------------
# Manual Input Mode
# ---------------------------
if mode == "🔹 Manual Input":
    st.sidebar.header("Enter Water Quality Features")

    def user_input_features():
        # Collect all necessary original features to calculate scores
        pH = st.sidebar.slider("pH Level", 0.0, 14.0, 7.0)
        Hardness = st.sidebar.slider("Hardness (mg/L)", 0.0, 500.0, 150.0)
        Solids = st.sidebar.number_input("Solids (ppm)", 0.0, 50000.0, 20000.0)
        Chloramines = st.sidebar.slider("Chloramines (mg/L)", 0.0, 10.0, 3.0)
        Sulfate = st.sidebar.number_input("Sulfate (mg/L)", 0.0, 500.0, 200.0)
        Conductivity = st.sidebar.number_input("Conductivity (μS/cm)", 0.0, 1000.0, 400.0)
        Organic_carbon = st.sidebar.slider("Organic Carbon (mg/L)", 0.0, 30.0, 10.0)
        Trihalomethanes = st.sidebar.number_input("Trihalomethanes (µg/L)", 0.0, 150.0, 50.0)
        Turbidity = st.sidebar.slider("Turbidity (NTU)", 0.0, 10.0, 4.0)

        data = {
            'ph': pH,
            'Hardness': Hardness,
            'Solids': Solids,
            'Chloramines': Chloramines,
            'Sulfate': Sulfate,
            'Conductivity': Conductivity,
            'Organic_carbon': Organic_carbon,
            'Trihalomethanes': Trihalomethanes,
            'Turbidity': Turbidity
        }
        # Return original features as DataFrame
        return pd.DataFrame(data, index=[0])

    # Get original input features
    input_df_original = user_input_features()

    st.subheader("🔍 Entered Water Quality Data:")
    st.write(input_df_original)

    # Calculate the engineered features ('safety_score', 'contamination_risk')
    input_df_engineered = calculate_safety_and_contamination_scores(input_df_original)

    st.subheader("⚙️ Engineered Features:")
    st.write(input_df_engineered)

    if st.button("💧 Predict Water Safety"):
        if model_pipeline is not None:
            try:
                # Predict using the loaded pipeline on the engineered features
                # The pipeline is expected to handle scaling internally
                prediction = model_pipeline.predict(input_df_engineered)
                prob = model_pipeline.predict_proba(input_df_engineered)
                prob = prob[0] # Access the first (and only) prediction's probabilities

                color = "green" if prediction[0] == 1 else "red"
                label = "✅ SAFE" if prediction[0] == 1 else "⚠️ UNSAFE"
                st.markdown(f"<h3 style='color:{color}'>{label}</h3>", unsafe_allow_html=True)
                st.write(f"Probability: Unsafe: {round(prob[0]*100, 2)}% | Safe: {round(prob[1]*100, 2)}%")

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.error("Please ensure the loaded model pipeline expects exactly the 'safety_score' and 'contamination_risk' features.")


        else:
            st.error("Model not loaded. Cannot make prediction.")

# ---------------------------
# Batch CSV Upload Mode
# ---------------------------
elif mode == "📂 Batch CSV Upload":
    st.subheader("📁 Upload a CSV File for Batch Predictions")
    st.markdown("""
    Upload a CSV file with these columns:
    `ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity`
    """)

    uploaded_file = st.file_uploader("Upload your water quality dataset", type=["csv"])

    if uploaded_file is not None:
        df_uploaded_original = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data:")
        st.dataframe(df_uploaded_original.head())

        if st.button("🚀 Predict for All Rows"):
            if model_pipeline is not None:
                try:
                    # Calculate the engineered features for the batch data
                    df_uploaded_engineered = calculate_safety_and_contamination_scores(df_uploaded_original)

                    st.write("### Preview of Engineered Features:")
                    st.dataframe(df_uploaded_engineered.head())

                    # Apply the pipeline to the engineered features DataFrame
                    df_uploaded_original['Potability_Prediction'] = model_pipeline.predict(df_uploaded_engineered)

                    st.success("✅ Predictions generated successfully!")
                    st.dataframe(df_uploaded_original.head())

                    csv = df_uploaded_original.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")

                except Exception as e:
                    st.error(f"Batch prediction failed: {str(e)}")
                    st.error("Please ensure the loaded model pipeline expects exactly the 'safety_score' and 'contamination_risk' features and that your CSV has the necessary original columns.")

            else:
                st.error("Model not loaded. Cannot make batch predictions.")
    else:
        st.info("Please upload a CSV file to continue.")

# ---------------------------
# Model Insights Section
# ---------------------------
st.markdown("---")
st.subheader("📊 Model Insights")

# Demo metrics (replace with real model metrics)
accuracy = 0.85
precision = 0.82
recall = 0.80
f1 = 0.81

st.markdown(f"""
**Model Performance (Demo Metrics):**
- Accuracy: {accuracy}
- Precision: {precision}
- Recall: {recall}
- F1 Score: {f1}
""")

# Feature importance using Streamlit only (no matplotlib)
st.markdown("**Feature Importance:**")

# You'll need to get the actual feature importances from your trained model
# If your final model is the one trained on 'safety_score' and 'contamination_risk',
# get the feature importances from that model's underlying estimator.
# For now, using placeholder data for the two engineered features.
feature_data = pd.DataFrame({
    'Feature': ['safety_score', 'contamination_risk'],
    'Importance': [0.6, 0.4] # Placeholder/Example importance
})

# Display as bar chart using Streamlit
st.bar_chart(feature_data.set_index('Feature'))

# Or display as table
st.dataframe(feature_data.sort_values('Importance', ascending=False))

# ---------------------------
# Meet the Team Section
# ---------------------------
st.markdown("---")
st.subheader("👩‍💻 Meet the Team")
st.markdown("""
1. **Snenhlanhla Nsele** - Data Scientist - [LinkedIn](https://www.linkedin.com/in/sinenhlanhla-nsele-126a6a18a)
2. **Nonhlanhla Magagula** - Data Scientist - [LinkedIn](https://www.linkedin.com/in/nonhlanhla-magagula-b741b3207)
3. **Thandiwe Mkhabela** - Data Scientist - [LinkedIn](https://www.linkedin.com/in/thandiwe-m)
4. **Thabiso Seema** - Software Engineer - [LinkedIn](https://www.linkedin.com/in/thabisoseema)
""")

# ---------------------------
# Model Versioning Info
# ---------------------------
st.markdown("---")
st.subheader("🧩 Model Versioning Info")
st.markdown("""
- **Model Version:** 1.0
- **Last Updated:** Nov 2025
- **Training Data:** 3,000+ samples
""")

# ---------------------------
# Footer
# ---------------------------
st.caption("Created with ❤️ by NaN Masters | Powered by Streamlit")
