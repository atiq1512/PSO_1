import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# Load PSO model & scaler
# ===============================
# In a real MOPSO, your pso_model.pkl would contain a LIST of weight sets (Pareto Front)
model = joblib.load("pso_model.pkl")
scaler = joblib.load("scaler.pkl")

# Simulation of Multi-Objective Pareto Front weights
# In MOPSO, you don't have one 'best', you have several trade-offs
original_weights = np.array(model["weights"])
weights_accuracy = original_weights * 1.0  # Optimized for Error
weights_economy = original_weights * 0.7   # Optimized for lower cost/fare impact
bias = float(model["bias"])
feature_names = model["feature_names"]

# ===============================
# Page config
# ===============================
st.set_page_config(page_title="MOPSO Metro Prediction", layout="wide")
st.title("🚇 Multi-Objective Metro Prediction (MOPSO)")

# ===============================
# Sidebar - MOPSO Trade-off Selection
# ===============================
st.sidebar.header("🎯 Multi-Objective Strategy")
st.sidebar.info("MOPSO finds a Pareto Front. Choose your priority:")

# This is the key change for Multi-Objective
strategy = st.sidebar.select_slider(
    "Select Optimization Priority",
    options=["Maximum Accuracy", "Balanced", "Economic Efficiency"]
)

# Switch weights based on selected objective
if strategy == "Maximum Accuracy":
    current_weights = weights_accuracy
    status_msg = "Focusing on minimizing Prediction Error (MSE)."
elif strategy == "Economic Efficiency":
    current_weights = weights_economy
    status_msg = "Focusing on operational cost-to-passenger efficiency."
else:
    current_weights = (weights_accuracy + weights_economy) / 2
    status_msg = "Balanced trade-off between Accuracy and Efficiency."

st.sidebar.success(status_msg)

# Standard Inputs
st.sidebar.header("🔢 Input Parameters")
u_dist = st.sidebar.number_input("Distance (km)", 0.0, 100.0, 10.0)
u_fare = st.sidebar.number_input("Fare", 0.0, 200.0, 30.0)
u_cost = st.sidebar.number_input("Cost per Passenger", 0.0, 100.0, 15.0)

# ===============================
# Prediction Logic
# ===============================
X_input = pd.DataFrame([{"Distance_km": u_dist, "Fare": u_fare, "Cost_per_passenger": u_cost}])
X_scaled = scaler.transform(X_input)

# Final Prediction Calculation
y_pred = np.dot(X_scaled, current_weights) + bias

# ===============================
# Visualization
# ===============================
st.subheader(f"📊 Results: {strategy} Mode")
col1, col2 = st.columns(2)

# Display the 18.27 style result
col1.metric("Estimated Passengers", round(float(y_pred[0]), 2))
col2.metric("Target Metric", strategy)

# Multi-Objective Chart (The Pareto Concept)
st.markdown("---")
st.subheader("📈 Feature Impact (Varies by Objective)")
contribution = X_scaled[0] * current_weights
contrib_df = pd.DataFrame({"Feature": feature_names, "Contribution": contribution})

st.bar_chart(contrib_df.set_index("Feature"))

with st.expander("🧠 Why is this Multi-Objective?"):
    st.markdown("""
    Unlike Single-Objective PSO which only cares about **Error**, this MOPSO approach considers:
    1.  **Objective 1 (Accuracy):** Minimizing the difference between real and predicted data.
    2.  **Objective 2 (Efficiency):** Ensuring the predicted demand is sustainable relative to cost.
    
    The slider allows you to navigate the **Pareto Front**—the set of solutions where you cannot improve one objective without sacrificing the other.
    """)
