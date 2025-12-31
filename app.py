import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="PSO Dashboard", layout="wide")

st.title("🚇 Particle Swarm Optimization Dashboard")
st.write("Metro ridership prediction using PSO-optimized regression")

# Load trained PSO model
weights = joblib.load('pso_weights.pkl')
scaler = joblib.load('scaler.pkl')

# Load dataset
df = pd.read_csv('delhi_metro_updated.csv')

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# Select target column
target = st.selectbox("Select Target Column", df.columns)

X = df.drop(columns=[target])
y = df[target]

X_scaled = scaler.transform(X)
predictions = np.dot(X_scaled, weights)

# Metrics
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

st.subheader("📊 Model Performance")
col1, col2 = st.columns(2)
col1.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
col2.metric("R² Score", f"{r2:.4f}")

# Visualization
st.subheader("📈 Actual vs Predicted Values")
st.line_chart(pd.DataFrame({
    "Actual": y,
    "Predicted": predictions
}))
