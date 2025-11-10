import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer


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
# Define the feature engineering function that exactly mirrors the training steps
def apply_feature_engineering(df):
    """Applies feature engineering steps to the DataFrame."""
    temp_df = df.copy()

    # Ensure all necessary original columns exist before engineering
    original_cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    for col in original_cols:
        if col not in temp_df.columns:
            # Handle missing original columns - imputation with median from training data is ideal here
            # For simplicity in this app, we'll fill with a placeholder or a common central value if column is completely missing
            # A robust solution would require saving/loading medians from training data
            temp_df[col] = temp_df[col].fillna(temp_df[col].median() if not temp_df[col].median().isnull().all() else 0) # Impute NaNs if any
            if col not in temp_df.columns: # If column was completely missing
                 temp_df[col] = 0 # Or a more appropriate default/imputed value

    # Handle missing values by filling with the median (replicate training step)
    # In a real app, load medians from training data
    for c in ['ph', 'Sulfate', 'Trihalomethanes']:
        if c in temp_df.columns:
             temp_df[c].fillna(temp_df[c].median() if not temp_df[c][~temp_df[c].isnull()].empty else 0, inplace=True)


    # Cap outliers (replicate training step)
    numeric_cols = temp_df.select_dtypes(include=np.number).columns.tolist()
    # Exclude target if it exists
    if 'Potability' in numeric_cols:
        numeric_cols.remove('Potability')

    for col in numeric_cols:
        if col in temp_df.columns:
            Q1 = temp_df[col].quantile(0.25)
            Q3 = temp_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            temp_df[col] = temp_df[col].clip(lower=lower_bound, upper=upper_bound)


    # Water Safety Score (domain knowledge composite) - replicate training step
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

    # Contamination risk index - replicate training step
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

    # Apply log transformation to highly skewed features if they were log-transformed during training
    # Based on previous notebook cells, only 'Solids', 'Conductivity', 'Trihalomethanes' and potentially the engineered features were checked for skewness
    # Replicate the log transformation ONLY for features that were actually log-transformed in the training notebook
    # Assuming 'Solids', 'Conductivity', 'Trihalomethanes' were candidates but the output showed no highly skewed features after initial steps.
    # Let's check the training notebook again. In cell `VMMHTL_hxXxk`, log transformation was applied to `highly_skewed_features.index`.
    # The output of cell `t6WJb2gpxG1J` (Highly skewed features) was empty.
    # So, no log transformation was actually applied to the original features or the initially engineered features in the training notebook.
    # Therefore, we should NOT apply log transformation here based on the training process shown.

    # If new engineered features were created and then log-transformed, list them here.
    # Based on cell `nP39YkvPmXx4`, 'safety_score' and 'contamination_risk' were created.
    # Based on cell `t6WJb2gpxG1J`, these new features were NOT highly skewed.
    # So, no log transformation was applied to any feature in the training notebook based on the provided cells.
    # If you added more feature engineering steps later that included log transformations, you need to add them here.

    # For robustness, let's ensure the output DataFrame has the same columns in the same order as the training data.
    # We need the list of column names from X_train. Let's assume X_train had the original features plus 'safety_score' and 'contamination_risk'.
    # If X_train had other engineered features, you need to add their creation logic above and include them in this list.
    trained_columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'safety_score', 'contamination_risk'] # Reorder to match training if necessary

    # Select and reindex columns to match the training data features
    # Fill missing columns (if any) with 0, and drop extra columns
    processed_df = temp_df.reindex(columns=trained_columns, fill_value=0)


    return processed_df

# --- End Manual Feature Engineering Logic ---


# ---------------------------
# Prediction Mode
# ---------------------------
@st.cache_resource
def load_model():
    try:
        # Get script directory
        current_dir = Path(__file__).parent

        # Use the correct relative path for the pipeline saved in the notebook
        model_path = current_dir / "Random_pipeline.pkl" # This should match the filename used in joblib.dump

        # Debug output
        st.write(f"Loading model from: {model_path}")
        st.write(f"File exists: {os.path.exists(model_path)}")

        if not os.path.exists(model_path):
            # Try loading from the current directory if the script directory method fails
            model_path = "Random_pipeline.pkl"
            st.write(f"Attempting to load from current directory: {model_path}")
            st.write(f"File exists: {os.path.exists(model_path)}")

            if not os.path.exists(model_path):
                st.error(f"❌ Model file not found at: {current_dir / 'Random_pipeline.pkl'} or {model_path}")
                st.info("Please ensure the trained pipeline is saved as 'Random_pipeline.pkl' in the correct location.")
                return None


        with open(model_path, 'rb') as f:
            # Assuming the loaded pipeline includes the StandardScaler and the model
            loaded_pipeline = joblib.load(f)
            st.success("✅ Model loaded successfully!")
            return loaded_pipeline

    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.exception(e) # Display the full traceback for better debugging
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

    # Apply the full feature engineering pipeline to the original input data
    input_df_processed = apply_feature_engineering(input_df_original)

    st.subheader("⚙️ Processed Features for Prediction:")
    st.dataframe(input_df_processed) # Display processed features

    if st.button("💧 Predict Water Safety"):
        if model_pipeline is not None:
            try:
                # Predict using the loaded pipeline on the *processed* features
                # The pipeline is expected to handle scaling internally as it was trained with StandardScaler
                prediction = model_pipeline.predict(input_df_processed)
                prob = model_pipeline.predict_proba(input_df_processed)
                prob = prob[0] # Access the first (and only) prediction's probabilities

                color = "green" if prediction[0] == 1 else "red"
                label = "✅ SAFE" if prediction[0] == 1 else "⚠️ UNSAFE"
                st.markdown(f"<h3 style='color:{color}'>{label}</h3>", unsafe_allow_html=True)
                st.write(f"Probability: Unsafe: {round(prob[0]*100, 2)}% | Safe: {round(prob[1]*100, 2)}%")

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.error("Please ensure the loaded model pipeline expects the exact features generated by the feature engineering step.")
                st.exception(e) # Display the full traceback


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
    (Other columns, including 'Potability', will be ignored for prediction but kept in the output.)
    """)

    uploaded_file = st.file_uploader("Upload your water quality dataset", type=["csv"])

    if uploaded_file is not None:
        df_uploaded_original = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data:")
        st.dataframe(df_uploaded_original.head())

        if st.button("🚀 Predict for All Rows"):
            if model_pipeline is not None:
                try:
                    # Apply the full feature engineering pipeline to the batch data
                    # We need to apply feature engineering on a copy to not modify the original uploaded df during engineering steps
                    # Then, align the processed features with the training columns before prediction
                    df_uploaded_processed = apply_feature_engineering(df_uploaded_original.copy())

                    st.write("### Preview of Processed Features for Prediction:")
                    st.dataframe(df_uploaded_processed.head())


                    # Apply the pipeline to the processed features DataFrame
                    # The pipeline (including scaler and model) expects the processed features
                    df_uploaded_original['Potability_Prediction'] = model_pipeline.predict(df_uploaded_processed)

                    st.success("✅ Predictions generated successfully!")
                    st.write("### Predictions Added to Data:")
                    st.dataframe(df_uploaded_original.head())

                    csv = df_uploaded_original.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")

                except Exception as e:
                    st.error(f"Batch prediction failed: {str(e)}")
                    st.error("Please ensure the loaded model pipeline expects the exact features generated by the feature engineering step.")
                    st.exception(e) # Display the full traceback


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
# You would ideally get these from evaluating your best model on the test set
accuracy = 0.6555 # Example from Random Forest test results
precision_0 = 0.6593 # Example from Random Forest test results (Class 0)
recall_0 = 0.9000  # Example from Random Forest test results (Class 0)
f1_0 = 0.7611      # Example from Random Forest test results (Class 0)
precision_1 = 0.6364 # Example from Random Forest test results (Class 1)
recall_1 = 0.2734  # Example from Random Forest test results (Class 1)
f1_1 = 0.3825      # Example from Random Forest test results (Class 1)
f1_weighted = 0.6134 # Example from Random Forest test results (weighted avg f1)


st.markdown(f"""
**Model Performance (Based on Test Set Evaluation):**
- Overall Accuracy: **{accuracy:.4f}**
- **Class 0 (Not Potable):**
    - Precision: {precision_0:.4f}
    - Recall: {recall_0:.4f}
    - F1 Score: {f1_0:.4f}
- **Class 1 (Potable):**
    - Precision: {precision_1:.4f}
    - Recall: {recall_1:.4f}
    - F1 Score: {f1_1:.4f}
- **Weighted Average F1 Score:** {f1_weighted:.4f}
""")

# Feature importance using Streamlit only (no matplotlib)
st.markdown("**Feature Importance (from Random Forest):**")

# You'll need to get the actual feature importances from your trained model
# If your final model is the one trained on the full set of engineered features,
# get the feature importances from that model's underlying estimator.
# The loaded pipeline is 'model_pipeline'. The estimator is likely model_pipeline.named_steps['model'].
# Ensure this estimator supports feature_importances_ (e.g., RandomForestClassifier, XGBoost)
if model_pipeline is not None and hasattr(model_pipeline.named_steps['model'], 'feature_importances_'):
    feature_importances = model_pipeline.named_steps['model'].feature_importances_
    # Get feature names from the scaler or processed data
    # Assuming the pipeline was fitted on X_train_scaled which had columns corresponding to the processed features
    # The processed features should match the columns expected by the scaler and model
    # Use the column names from the 'apply_feature_engineering' output
    feature_names = apply_feature_engineering(pd.DataFrame(columns=user_input_features().columns)).columns.tolist()

    feature_data = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)

    # Display as bar chart using Streamlit
    st.bar_chart(feature_data.set_index('Feature'))

    # Or display as table
    st.dataframe(feature_data)

else:
    st.info("Feature importance available only if the loaded model supports it.")


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
