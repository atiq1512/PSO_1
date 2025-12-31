import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# 1. Load model & scaler
# ===============================
model = joblib.load("pso_model.pkl")
scaler = joblib.load("scaler.pkl")

weights = model["weights"]
bias = model["bias"]
feature_names = model["feature_names"]

# ===============================
# 2. Streamlit UI
# ===============================
st.title("Delhi Metro Ridership Prediction (PSO)")

# Collect user input
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"Enter {feature}", value=0.0)

X_input = pd.DataFrame([user_input])

# ===============================
# 3. Scale input
# ===============================
X_scaled = scaler.transform(X_input)

# ===============================
# 4. Predict
# ===============================
y_pred = np.dot(X_scaled, weights) + bias

st.subheader("Predicted Ridership")
st.write(y_pred[0])

