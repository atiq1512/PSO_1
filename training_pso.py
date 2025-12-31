import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pyswarms as ps

# ===============================
# 1. Load Dataset
# ===============================
data = pd.read_csv("delhi_metro_updated.csv")

# Adjust target column if needed
X = data.drop(columns=["ridership"])
y = data["ridership"]

feature_names = X.columns.tolist()

# ===============================
# 2. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 3. Scaling
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 4. Vectorized Fitness Function
# ===============================
def fitness_function(weights):
    # weights shape: (n_particles, n_dimensions)
    w = weights[:, :-1]              # all particle weights
    b = weights[:, -1].reshape(-1,1) # all particle biases
    y_pred = X_train_scaled @ w.T    # shape: (n_samples, n_particles)
    mse = np.mean((y_train.values.reshape(-1,1) - y_pred)**2, axis=0)
    return mse

# ===============================
# 5. PSO Optimization
# ===============================
dimensions = X_train_scaled.shape[1] + 1  # weights + bias
options = {"c1": 1.5, "c2": 1.5, "w": 0.7}

optimizer = ps.single.GlobalBestPSO(
    n_particles=15,  # faster
    dimensions=dimensions,
    options=options
)

best_cost, best_position = optimizer.optimize(fitness_function, iters=50)  # faster

# ===============================
# 6. Extract Best Parameters
# ===============================
best_weights = best_position[:-1]
best_bias = best_position[-1]

# ===============================
# 7. Evaluate Model
# ===============================
y_test_pred = np.dot(X_test_scaled, best_weights) + best_bias
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE:", test_mse)

# ===============================
# 8. Save PSO Model (.pkl)
# ===============================
pso_model = {
    "model_type": "PSO-Optimized Linear Regression",
    "weights": best_weights,
    "bias": best_bias,
    "feature_names": feature_names,
    "fitness_function": "Mean Squared Error",
    "optimizer": "Particle Swarm Optimization (pyswarms)"
}
joblib.dump(pso_model, "pso_model.pkl")

# ===============================
# 9. Save Scaler (.pkl)
# ===============================
joblib.dump(scaler, "scaler.pkl")
print("✅ pso_model.pkl and scaler.pkl saved successfully")

