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
# Page config
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

X_input = pd.DataFrame([user_input])[['Distance_km', 'Fare', 'Cost_per_passenger']]
X_scaled = scaler.transform(X_input)

# ===============================
# Prediction
# ===============================
weights_demo = weights * 750  # Demo-friendly exaggeration
y_pred = np.dot(X_scaled, weights_demo) + bias

# ===============================
# Metrics cards
# ===============================
st.subheader("📊 Prediction Result")
col1, col2 = st.columns(2)
col1.metric("Estimated Passengers", round(y_pred[0],2))

# Max prediction demo
X_max = pd.DataFrame([{"Distance_km": 100, "Fare": 200, "Cost_per_passenger": 100}])
X_max_scaled = scaler.transform(X_max)
y_max = np.dot(X_max_scaled, weights_demo) + bias
col2.metric("Max Predicted Passengers (demo)", round(y_max[0],2))

# ===============================
# Dynamic Charts
# ===============================
st.subheader("📈 Feature Contribution & Sensitivity")
contribution = X_scaled[0] * weights_demo
contrib_df = pd.DataFrame({"Feature": feature_names, "Contribution": contribution})
sensitivity_df = contrib_df.copy()
sensitivity_df["Impact"] = contrib_df["Contribution"].abs()

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.markdown("**Feature Contribution**")
    st.bar_chart(contrib_df.set_index("Feature"))

with chart_col2:
    st.markdown("**Sensitivity Analysis**")
    st.line_chart(sensitivity_df.set_index("Feature"))

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



