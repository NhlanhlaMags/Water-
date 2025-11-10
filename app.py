import pandas as pd
import joblib
import streamlit as st
from pathlib import Path
import os

# ---------------------------
# Page Config
# ---------------------------
st.markdown('<div class="header-banner">💧 Water Potability Predictor</div>', unsafe_allow_html=True)
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
# About Section
# ---------------------------
st.markdown("""
### 👥 Who We Are
**We are NaN Masters** — a curious crew of data explorers and problem solvers.
Our mission? To turn raw data into real-world impact and help communities access **clean, safe water**.
""")

st.markdown("---")
st.subheader("🌍 Why This Matters")
st.markdown("""
Many communities worldwide lack access to **safe drinking water**, and testing is often **slow or unavailable**.
Our model predicts whether water is safe or unsafe using **chemical properties** like pH, turbidity, hardness, and more.
""")

# ---------------------------
# Model Loader
# ---------------------------
@st.cache_resource
def load_model():
    try:
        current_dir = Path(__file__).parent
        model_path = current_dir / "Random_pipeline.pkl"
        if not os.path.exists(model_path):
            model_path = "Random.pkl"
        if not os.path.exists(model_path):
            st.error("❌ Model file not found. Please ensure 'Random_pipeline.pkl' exists.")
            return None

        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.exception(e)
        return None

model_pipeline = load_model()

# ---------------------------
# Prediction Mode
# ---------------------------
st.markdown("---")
st.subheader("🚀 Choose a Prediction Mode")
mode = st.radio("Select mode:", ("🔹 Manual Input", "📂 Batch CSV Upload"), horizontal=True)

# ---------------------------
# Manual Input
# ---------------------------
if mode == "🔹 Manual Input":
    st.sidebar.header("Enter Water Quality Features")

    def user_input_features():
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
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()
    st.subheader("🔍 Input Data:")
    st.write(input_df)

    if st.button("💧 Predict Water Safety"):
        if model_pipeline is not None:
            try:
                # No preprocessing – pipeline handles scaling internally
                prediction = model_pipeline.predict(input_df)
                prob = model_pipeline.predict_proba(input_df)[0]

                color = "green" if prediction[0] == 1 else "red"
                label = "✅ SAFE" if prediction[0] == 1 else "⚠️ UNSAFE"
                st.markdown(f"<h3 style='color:{color}'>{label}</h3>", unsafe_allow_html=True)
                st.write(f"Probability: Unsafe: {round(prob[0]*100, 2)}% | Safe: {round(prob[1]*100, 2)}%")

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.exception(e)
        else:
            st.error("Model not loaded.")

# ---------------------------
# Batch CSV Upload
# ---------------------------
elif mode == "📂 Batch CSV Upload":
    st.subheader("📁 Upload a CSV File for Batch Predictions")
    st.markdown("Required columns: `ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity`")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data:")
        st.dataframe(df.head())

        if st.button("🚀 Predict for All Rows"):
            if model_pipeline is not None:
                try:
                    # Directly predict using the pipeline (handles scaling internally)
                    preds = model_pipeline.predict(df)
                    probs = model_pipeline.predict_proba(df)[:, 1]
                    df['Potability_Prediction'] = preds
                    df['Safe_Probability'] = probs

                    st.success("✅ Predictions generated successfully!")
                    st.dataframe(df.head())

                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")

                except Exception as e:
                    st.error(f"Batch prediction failed: {str(e)}")
                    st.exception(e)
            else:
                st.error("Model not loaded.")
    else:
        st.info("Please upload a CSV file to continue.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Created with ❤️ by NaN Masters | Powered by Streamlit")
