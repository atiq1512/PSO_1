import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("pso_model.pkl")
scaler = joblib.load("scaler.pkl")

weights = model["weights"]
bias = model["bias"]
feature_names = model["feature_names"]

st.title("Delhi Metro Passengers Prediction (PSO)")

# Input numeric features only
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"Enter {feature}", value=0.0)

X_input = pd.DataFrame([user_input])

# Scale input
X_scaled = scaler.transform(X_input)

# Predict
y_pred = np.dot(X_scaled, weights) + bias

st.subheader("Predicted Passengers")
st.write(y_pred[0])

