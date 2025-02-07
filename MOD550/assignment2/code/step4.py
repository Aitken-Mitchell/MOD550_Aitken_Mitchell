import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Load the generated data
df_clean = pd.read_csv("step2_oscillatory_data_clean.csv")
df_noisy = pd.read_csv("step2_oscillatory_data_noisy.csv")

# Extract time and displacement values
t_train = df_noisy["t"].values.reshape(-1, 1)
u_train = df_noisy["u"].values.reshape(-1, 1)
t_test = df_clean["t"].values.reshape(-1, 1)
u_test = df_clean["u"].values.reshape(-1, 1)

# ---- Polynomial Regression ----
# Define the polynomial degree and regularization strength
degree = 30
alpha = 1.0  # Regularization parameter for Ridge Regression

# Create a pipeline that scales the data, generates polynomial features, and applies Ridge Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=degree)),
    ('ridge', Ridge(alpha=alpha))
])

# Train the model and record the time taken
lr_start = time.time()
pipeline.fit(t_train, u_train.ravel())
lr_end = time.time()
lr_time = lr_end - lr_start

# Predict on the test set
u_pred_lr = pipeline.predict(t_test)

# Optionally, compute and print the test MSE for evaluation
mse_test = mean_squared_error(u_test, u_pred_lr)

print("Task completed for LR (plot to follow).")

# ---- Neural Network (Reverting to Tanh) ----
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.fc(x)

# Convert data to PyTorch tensors
t_tensor = torch.tensor(t_train, dtype=torch.float32)
u_tensor = torch.tensor(u_train, dtype=torch.float32)
t_test_tensor = torch.tensor(t_test, dtype=torch.float32)

# Initialize and train the NN model
nn_model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)  # Lower learning rate for smoother training

nn_start = time.time()
for epoch in range(20001):
    optimizer.zero_grad()
    output = nn_model(t_tensor)
    loss_nn = criterion(output, u_tensor)
    loss_nn.backward()
    optimizer.step()

nn_end = time.time()
nn_time = nn_end - nn_start

u_pred_nn = nn_model(t_test_tensor).detach().numpy()

print("Task completed for NN (plot to follow).")

# ---- PINN Method ----
class FCN(nn.Module):
    def __init__(self, in_dim, out_dim, width, depth):
        super(FCN, self).__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, out_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

# --------------------------
# Initialize PINN
# --------------------------
pinn = FCN(1, 1, 32, 3)

# --------------------------
# Define Training Data
# --------------------------
t_physics = torch.linspace(0,1,30).view(-1,1).requires_grad_(True)

# Given Physics Constants
d, w0 = 0.5, 3  # Damping coefficient and natural frequency
mut, kt = 2 * d, w0**2  # True values

# Initialize Learnable Parameters
mui = torch.rand(1, requires_grad=True)
ki = torch.rand(1, requires_grad=True)

mu = torch.nn.Parameter(mui)
k = torch.nn.Parameter(ki)

# Store Training History
pinn_history = {}

# --------------------------
# Define Optimizer
# --------------------------
optimiser = torch.optim.Adam(list(pinn.parameters()) + [mu, k], lr=0.005)

# Convert numpy arrays to PyTorch tensors
t_train_tensor = torch.tensor(t_train, dtype=torch.float32)
u_train_tensor = torch.tensor(u_train, dtype=torch.float32)

# --------------------------
# PINN Training Loop
# --------------------------
num_iterations = 25001
lambda1 = 5000

pinn_start = time.time() #start timing

for i in range(num_iterations):
    optimiser.zero_grad()

    # ---- Compute Physics Loss ----
    u = pinn(t_physics)
    dudt = torch.autograd.grad(u, t_physics, torch.ones_like(u), create_graph=True)[0]
    d2udt2 = torch.autograd.grad(dudt, t_physics, torch.ones_like(dudt), create_graph=True)[0]
    loss1 = torch.mean((d2udt2 + mu * dudt + 100 * k * u) ** 2)

    # ---- Compute Data Loss (Fixed) ----
    u_pred_train = pinn(t_train_tensor)  # Now using a tensor
    loss2 = torch.mean((u_pred_train - u_train_tensor) ** 2)

    # ---- Compute Total Loss ----
    loss_pinn = loss1 + lambda1 * loss2
    loss_pinn.backward()
    optimiser.step()

    # Store test predictions every 5000 iterations
    if i % 5000 == 0: 
        with torch.no_grad():
            pinn_history[i] = pinn(t_test_tensor).detach().numpy()

pinn_end = time.time()
pinn_time = pinn_end - pinn_start

print("Task completed for PINN (plot to follow).")

# --------------------------
# Generate Predictions
# --------------------------
u_pred_pinn = pinn(t_test_tensor).detach().numpy()

# ---- Plot Results ----
def plot_results(t_test, u_test, u_pred, method, exec_time, filename):
    plt.figure(figsize=(8, 5))
    plt.scatter(t_train, u_train, label="Noisy Data", color='red', s=5, alpha=0.5)
    plt.plot(t_test, u_test, label="Ground Truth", color='black', linestyle="dashed")
    plt.plot(t_test, u_pred, label=f"{method} Prediction", color='blue')
    plt.xlabel("Time (t)")
    plt.ylabel("Displacement (u)")
    plt.legend()
    plt.title(f"{method} Regression (Time: {exec_time:.2f} sec)")
    plt.savefig(filename)
    plt.show()


plot_results(t_test, u_test, u_pred_lr, "Polynomial Regression", lr_time, "step4_LR.png")
plot_results(t_test, u_test, u_pred_nn, "Neural Network", nn_time, "step4_NN.png")
plot_results(t_test, u_test, u_pred_pinn, "PINN", pinn_time, "step4_PINN.png")

