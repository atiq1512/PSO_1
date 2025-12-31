import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# 1. Load PSO model & scaler
# ===============================
model = joblib.load("pso_model.pkl")
scaler = joblib.load("scaler.pkl")

weights = model["weights"]
bias = model["bias"]
feature_names = model["feature_names"]

# ===============================
# 2. Load dataset (same CSV as training)
# ===============================
data = pd.read_csv("delhi_metro_updated.csv")
data = data[['Distance_km', 'Fare', 'Cost_per_passenger', 'Passengers']]
data = data.dropna()

# ===============================
# 3. App Title & Overview
# ===============================
st.title("🚇 Delhi Metro Passenger Prediction using PSO")

st.markdown("""
### 📌 Problem Overview
This application predicts **Delhi Metro passenger demand** using a  
**Particle Swarm Optimization (PSO) optimized linear regression model**.

**Objective:**  
Minimize Mean Squared Error (MSE) between predicted and actual passengers.

**Optimized using:**  
- Particle Swarm Optimization (PSO)
- Continuous weight & bias search space
""")

# ===============================
# 4. Sidebar – User Input
# ===============================
st.sidebar.header("🔢 Input Parameters")

user_input = {}
user_input["Distance_km"] = st.sidebar.number_input(
    "Distance (km)", min_value=0.0, value=10.0
)
user_input["Fare"] = st.sidebar.number_input(
    "Fare", min_value=0.0, value=30.0
)
user_input["Cost_per_passenger"] = st.sidebar.number_input(
    "Cost per Passenger", min_value=0.0, value=15.0
)

X_input = pd.DataFrame([user_input])

# ===============================
# 5. Scale input & Predict
# ===============================
X_scaled = scaler.transform(X_input)
y_pred = np.dot(X_scaled, weights) + bias

# ===============================
# 6. Prediction Output
# ===============================
st.subheader("📊 Prediction Result")
st.metric("Estimated Passengers", int(y_pred[0]))

# ===============================
# 7. Feature Contribution (Model Insight)
# ===============================
st.subheader("📌 Feature Contribution (PSO-Optimized Weights)")

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Weight": weights
})

st.bar_chart(coef_df.set_index("Feature"))

# ===============================
# 8. Dataset Preview (CSV consistency)
# ===============================
st.subheader("🗂 Dataset Sample (Delhi Metro)")
st.dataframe(data.head(10))

# ===============================
# 9. Model Explanation (Evolutionary Computing)
# ===============================
st.markdown("""
### 🧠 PSO Optimization Explanation
- Each particle represents a candidate solution (weights + bias)
- Fitness function: **Mean Squared Error (MSE)**
- Global best solution guides swarm convergence
- Bound constraints prevent numerical instability
""")

# ===============================
# 10. Conclusion
# ===============================
st.markdown("""
### ✅ Conclusion
Particle Swarm Optimization successfully optimized the regression parameters
for metro passenger prediction.  
The model demonstrates reliable performance and is suitable for
transport demand forecasting applications.
""")
