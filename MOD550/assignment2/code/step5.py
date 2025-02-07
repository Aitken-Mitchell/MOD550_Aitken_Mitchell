import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time

# Load the generated data
df_clean = pd.read_csv("step2_oscillatory_data_clean.csv")
df_noisy = pd.read_csv("step2_oscillatory_data_noisy.csv")

# Extract time and displacement values
t_train = df_noisy["t"].values.reshape(-1, 1)
u_train = df_noisy["u"].values.reshape(-1, 1)
t_test = df_clean["t"].values.reshape(-1, 1)
u_test = df_clean["u"].values.reshape(-1, 1)

# ---- Neural Network (Tracking Convergence) ----
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

nn_model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.01)

t_tensor = torch.tensor(t_train, dtype=torch.float32)
u_tensor = torch.tensor(u_train, dtype=torch.float32)
t_test_tensor = torch.tensor(t_test, dtype=torch.float32)

nn_start = time.time()
nn_history = {}
for epoch in range(20001):
    optimizer.zero_grad()
    output = nn_model(t_tensor)
    loss_nn = criterion(output, u_tensor)
    loss_nn.backward()
    optimizer.step()
    if epoch % 4000 == 0:
        nn_history[epoch] = nn_model(t_test_tensor).detach().numpy()
nn_end = time.time()
nn_time = nn_end - nn_start

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
d, w0 = 0.5, 3
mut, kt = 2*d, w0**2  # True values

# Initialize Learnable Parameters
mui = torch.rand(1, requires_grad=True)
ki = torch.rand(1, requires_grad=True)

mu = torch.nn.Parameter(mui)
k = torch.nn.Parameter(ki)

# Store Training History
pinn_history = {}
mus = []
ks = []

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

    # Store Training Values
    mus.append(mu.item())
    ks.append(100 * k.item())  # Match lecturer’s scaling

    # Store test predictions every 5000 iterations
    if i % 5000 == 0: 
        with torch.no_grad():
            pinn_history[i] = pinn(t_test_tensor).detach().numpy()


# ---- Plot Subplots for NN & PINN Convergence ----
def plot_subplots(history, method, steps, filename):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, step in enumerate(steps):
        if step in history:
            axes[idx].plot(t_test, history[step], label=f"Iteration {step}")
            axes[idx].plot(t_test, u_test, 'k--', label="Ground Truth")
            axes[idx].set_title(f"{method} at {step} iterations")
            axes[idx].legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

plot_subplots(nn_history, "NN", [0, 4000, 8000, 12000, 16000, 20000], "step5_NN_sub.png")
plot_subplots(pinn_history, "PINN", [0, 5000, 10000, 15000, 20000, 25000], "step5_PINN_sub.png")

# --------------------------
# Plot Mu & K Evolution
# --------------------------
plt.figure()
plt.title(r"$\mu$")
plt.plot(mus, 'r--', label="PINN estimate")
plt.hlines(2*d, 0, len(mus), colors="tab:green", label="True value")
plt.legend()
plt.xlabel("Training step")
plt.savefig("step5_pinn_mu_evolution.png")
plt.show()

plt.figure()
plt.title(r"$k$")
plt.plot(ks, 'b--', label="PINN estimate")
plt.hlines(kt, 0, len(ks), colors="tab:green", label="True value")
plt.legend()
plt.xlabel("Training step")
plt.savefig("step5_pinn_k_evolution.png")
plt.show()

