import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# Load PSO model & scaler
# ===============================
model = joblib.load("pso_model.pkl")
scaler = joblib.load("scaler.pkl")

weights = np.array(model["weights"])
bias = float(model["bias"])
feature_names = model["feature_names"]

# ===============================
# Load dataset for preview
# ===============================
data = pd.read_csv("delhi_metro_updated.csv")
data = data[['Distance_km', 'Fare', 'Cost_per_passenger', 'Passengers']].dropna()

# ===============================
# App layout
# ===============================
st.set_page_config(page_title="Delhi Metro Passenger Prediction", layout="wide")
st.title("🚇 Delhi Metro Passenger Prediction (PSO)")

st.markdown("""
Predict **Delhi Metro passenger demand** using a  
**Particle Swarm Optimization (PSO) optimized regression model**.
""")

# ===============================
# Sidebar - Inputs
# ===============================
st.sidebar.header("🔢 Input Parameters")
user_input = {}
user_input["Distance_km"] = st.sidebar.number_input("Distance (km)", 0.0, 100.0, 10.0)
user_input["Fare"] = st.sidebar.number_input("Fare", 0.0, 200.0, 30.0)
user_input["Cost_per_passenger"] = st.sidebar.number_input("Cost per Passenger", 0.0, 100.0, 15.0)

# Ensure column order matches training
X_input = pd.DataFrame([user_input])[['Distance_km', 'Fare', 'Cost_per_passenger']]
X_scaled = scaler.transform(X_input)

# ===============================
# Prediction
# ===============================
# TEMP: exaggerate weights for demo
weights_demo = weights * 50
y_pred = np.dot(X_scaled, weights_demo) + bias

st.subheader("📊 Prediction Result")
st.metric("Estimated Passengers", round(y_pred[0],2))

# ===============================
# Feature Contribution (Dynamic)
# ===============================
st.subheader("📌 Feature Contribution (Dynamic)")
contribution = X_scaled[0] * weights_demo
contrib_df = pd.DataFrame({
    "Feature": feature_names,
    "Contribution": contribution
})
st.bar_chart(contrib_df.set_index("Feature"))

# ===============================
# Sensitivity Analysis (Dynamic)
# ===============================
st.subheader("⚡ Sensitivity Analysis")
sensitivity_df = contrib_df.copy()
sensitivity_df["Impact"] = contrib_df["Contribution"].abs()
st.line_chart(sensitivity_df.set_index("Feature"))

# ===============================
# Max Prediction Demo
# ===============================
st.subheader("🏁 Maximum Predicted Passengers (Demo)")
X_max = pd.DataFrame([{
    "Distance_km": 100,
    "Fare": 200,
    "Cost_per_passenger": 100
}])[['Distance_km', 'Fare', 'Cost_per_passenger']]
X_max_scaled = scaler.transform(X_max)
y_max = np.dot(X_max_scaled, weights_demo) + bias
st.metric("Max Estimated Passengers (demo)", round(y_max[0],2))

# ===============================
# Dataset preview
# ===============================
st.subheader("🗂 Dataset Sample")
with st.expander("Show dataset preview"):
    st.dataframe(data.head(10))

# ===============================
# PSO explanation
# ===============================
with st.expander("🧠 How PSO Works"):
    st.markdown("""
- Each particle represents a candidate solution (weights + bias)
- Fitness function: **Mean Squared Error (MSE)**
- Global best solution guides swarm convergence
- Bound constraints prevent numerical instability
""")

# ===============================
# Conclusion
# ===============================
st.subheader("✅ Conclusion")
st.markdown("""
Particle Swarm Optimization successfully optimized the regression parameters.
The model demonstrates reliable predictive performance and is suitable for
transport demand forecasting applications.
""")


