import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ==================================================
# Page Configuration
# ==================================================
st.set_page_config(
    page_title="Delhi Metro Passenger Prediction",
    page_icon="🚇",
    layout="wide"
)

# ==================================================
# Load PSO Model & Scaler
# ==================================================
model = joblib.load("pso_model.pkl")
scaler = joblib.load("scaler.pkl")

weights = model["weights"]
bias = model["bias"]
feature_names = model["feature_names"]

# ==================================================
# Load Dataset
# ==================================================
data = pd.read_csv("delhi_metro_updated.csv")
data = data[['Distance_km', 'Fare', 'Cost_per_passenger', 'Passengers']]
data = data.dropna()

# ==================================================
# Header Section
# ==================================================
st.markdown(
    """
    <h1 style='text-align: center;'>🚇 Delhi Metro Passenger Prediction</h1>
    <p style='text-align: center; font-size:18px;'>
    PSO-Optimized Regression Model for Transport Demand Forecasting
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ==================================================
# Project Overview
# ==================================================
with st.container():
    st.subheader("📌 Project Overview")
    st.write(
        """
        This application predicts **Delhi Metro passenger demand** using a  
        **Particle Swarm Optimization (PSO) based linear regression model**.

        **Objective:**  
        Minimize **Mean Squared Error (MSE)** between predicted and actual passengers.

        **Optimization Method:**  
        - Particle Swarm Optimization (PSO)  
        - Continuous search for optimal weights and bias
        """
    )

# ==================================================
# User Input & Prediction
# ==================================================
st.markdown("## 🔢 Passenger Prediction")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🧾 Input Parameters")

    distance = st.number_input(
        "🚏 Distance (km)",
        min_value=0.0,
        value=10.0,
        step=0.5
    )

    fare = st.number_input(
        "💰 Fare",
        min_value=0.0,
        value=30.0,
        step=1.0
    )

    cost = st.number_input(
        "📉 Cost per Passenger",
        min_value=0.0,
        value=15.0,
        step=1.0
    )

    X_input = pd.DataFrame([{
        "Distance_km": distance,
        "Fare": fare,
        "Cost_per_passenger": cost
    }])

with col2:
    st.markdown("### 📊 Prediction Output")

    X_scaled = scaler.transform(X_input)
    y_pred = np.dot(X_scaled, weights) + bias

    st.metric(
        label="🚇 Estimated Number of Passengers",
        value=f"{int(y_pred[0]):,}"
    )

    st.info(
        "Prediction is generated using PSO-optimized regression parameters."
    )

# ==================================================
# Feature Contribution Section
# ==================================================
st.markdown("## 📌 Feature Contribution Analysis")

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Weight": weights
}).set_index("Feature")

st.bar_chart(coef_df)

# ==================================================
# Sensitivity Analysis Section (Interactive)
# ==================================================
st.markdown("## 📈 Sensitivity Analysis")
st.caption("Effect of distance on passenger demand (other variables fixed)")

distance_range = np.linspace(1, 50, 50)

sensitivity_df = pd.DataFrame({
    "Distance_km": distance_range,
    "Fare": fare,
    "Cost_per_passenger": cost
})

X_sensitivity_scaled = scaler.transform(sensitivity_df)
predicted_passengers = np.dot(X_sensitivity_scaled, weights) + bias
sensitivity_df["Predicted Passengers"] = predicted_passengers.astype(int)

fig = px.line(
    sensitivity_df,
    x="Distance_km",
    y="Predicted Passengers",
    markers=True,
    title="Passenger Demand Sensitivity to Distance"
)

st.plotly_chart(fig, use_container_width=True)

# ==================================================
# Dataset Preview Section
# ==================================================
st.markdown("## 🗂 Dataset Preview")

with st.expander("Click to view sample data from Delhi Metro dataset"):
    st.dataframe(data.head(10), use_container_width=True)

# ==================================================
# PSO Explanation Section
# ==================================================
st.markdown("## 🧠 Particle Swarm Optimization (PSO)")

st.write(
    """
    - Each **particle** represents a candidate solution (weights + bias)
    - **Fitness function:** Mean Squared Error (MSE)
    - Particles update positions based on:
        - Personal best solution
        - Global best solution
    - Bound constraints are applied to ensure numerical stability
    """
)

# ==================================================
# Conclusion Section
# ==================================================
st.markdown("## ✅ Conclusion")

st.success(
    """
    Particle Swarm Optimization successfully optimized the regression parameters
    for Delhi Metro passenger prediction.

    The model demonstrates reliable predictive performance and is suitable for
    **transport demand forecasting and planning applications**.
    """
)

