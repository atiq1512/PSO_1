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
# 2. Load dataset
# ===============================
data = pd.read_csv("delhi_metro_updated.csv")
data = data[['Distance_km', 'Fare', 'Cost_per_passenger', 'Passengers']].dropna()

# ===============================
# 3. App Title & Overview
# ===============================
st.set_page_config(page_title="Delhi Metro Passenger Prediction", layout="wide")
st.title("🚇 Delhi Metro Passenger Prediction")
st.markdown("""
Predict **Delhi Metro passenger demand** using a  
**Particle Swarm Optimization (PSO) optimized regression model**.

**Objective:** Minimize Mean Squared Error (MSE) between predicted and actual passengers.

**Optimized using:**  
- Particle Swarm Optimization (PSO)
- Continuous weight & bias search space
""")

# ===============================
# 4. Sidebar – User Input
# ===============================
st.sidebar.header("🔢 Input Parameters")
user_input = {}
user_input["Distance_km"] = st.sidebar.number_input("Distance (km)", 0.0, 100.0, 10.0)
user_input["Fare"] = st.sidebar.number_input("Fare", 0.0, 200.0, 30.0)
user_input["Cost_per_passenger"] = st.sidebar.number_input("Cost per Passenger", 0.0, 100.0, 15.0)
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
# 7. Feature Contribution
# ===============================
st.subheader("📌 Feature Contribution (PSO-Optimized Weights)")
coef_df = pd.DataFrame({"Feature": feature_names, "Weight": weights})
st.bar_chart(coef_df.set_index("Feature"))

# ===============================
# 8. Sensitivity Analysis
# ===============================
st.subheader("⚡ Sensitivity Analysis")
sensitivity_df = coef_df.copy()
sensitivity_df["Change"] = coef_df["Weight"].abs()  # simple proxy for impact
st.line_chart(sensitivity_df.set_index("Feature"))

# ===============================
# 9. Dataset Preview
# ===============================
st.subheader("🗂 Dataset Sample (Delhi Metro)")
with st.expander("Show dataset preview"):
    st.dataframe(data.head(10))

# ===============================
# 10. PSO Explanation
# ===============================
with st.expander("🧠 How PSO Works"):
    st.markdown("""
- Each particle represents a candidate solution (weights + bias)
- Fitness function: **Mean Squared Error (MSE)**
- Global best solution guides swarm convergence
- Bound constraints prevent numerical instability
""")

# ===============================
# 11. Conclusion
# ===============================
st.subheader("✅ Conclusion")
st.markdown("""
Particle Swarm Optimization successfully optimized the regression parameters.
The model demonstrates reliable predictive performance and is suitable for
transport demand forecasting applications.
""")


