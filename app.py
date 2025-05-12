import streamlit as st
import joblib
import numpy as np

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('breast_cancer_rf.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

model, scaler = load_model()
feature_names = model.feature_names_in_

st.title("🔬 Breast Cancer Predictor")
st.markdown("Enter the following medical measurements to predict whether the tumor is **Benign** or **Malignant**.")

# Input layout
cols = st.columns(2)
user_input = []

for i, feat in enumerate(feature_names):
    col = cols[i % 2]  # cycle through columns
    val = col.number_input(f"Enter value for {feat}: ", format="%.4f", key=feat)
    user_input.append(val)

# Prediction
if st.button("🧠 Predict"):
    if any(v is None for v in user_input):
        st.warning("Please fill in all fields.")
    else:
        X_new = np.array(user_input).reshape(1, -1)
        X_scaled = scaler.transform(X_new)

        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0, prediction]

        label = "Malignant" if prediction == 0 else "Benign"
        st.success(f"🩺 **Prediction: {label}**\n\nConfidence: {probability:.2%}")