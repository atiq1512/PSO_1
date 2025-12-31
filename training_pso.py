import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pyswarms as ps

# ===============================
# 1. Load dataset
# ===============================
data = pd.read_csv("delhi_metro_updated.csv")

# ===============================
# OPTIONAL: Use only first 5000 rows to speed up training
# ===============================
data = data.head(5000)

# ===============================
# 2. Select numeric features only
# ===============================
X = data[['Distance_km', 'Fare', 'Cost_per_passenger']]
y = data['Passengers']
feature_names = X.columns.tolist()

# ===============================
# 3. Train-test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 4. Scale features
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 5. Vectorized fitness function
# ===============================
def fitness_function(weights):
    """
    weights: shape (n_particles, n_features + 1)
    returns: MSE per particle
    """
    if weights.size == 0:
        return np.array([np.inf])
    
    w = weights[:, :-1]                  # (n_particles, n_features)
    b = weights[:, -1].reshape(1, -1)   # (1, n_particles)
    
    y_pred = X_train_scaled @ w.T        # (n_samples, n_particles)
    y_pred += b                          # broadcast bias across samples
    
    mse = np.mean((y_train.values.reshape(-1,1) - y_pred)**2, axis=0)
    return mse

# ===============================
# 6. PSO optimization
# ===============================
dimensions = X_train_scaled.shape[1] + 1  # features + bias
options = {"c1": 1.5, "c2": 1.5, "w": 0.7}

optimizer = ps.single.GlobalBestPSO(
    n_particles=15,       # smaller for faster Colab run
    dimensions=dimensions,
    options=options
)

best_cost, best_position = optimizer.optimize(fitness_function, iters=50)

# ===============================
# 7. Extract best parameters
# ===============================
best_weights = best_position[:-1]
best_bias = best_position[-1]

# ===============================
# 8. Evaluate model
# ===============================
y_test_pred = np.dot(X_test_scaled, best_weights) + best_bias
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE:", test_mse)

# ===============================
# 9. Save PSO model
# ===============================
pso_model = {
    "model_type": "PSO-Optimized Linear Regression",
    "weights": best_weights,
    "bias": best_bias,
    "feature_names": feature_names
}
joblib.dump(pso_model, "pso_model.pkl")

# ===============================
# 10. Save scaler
# ===============================
joblib.dump(scaler, "scaler.pkl")
print("✅ pso_model.pkl and scaler.pkl saved successfully")

