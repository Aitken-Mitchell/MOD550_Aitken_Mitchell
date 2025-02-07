import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time

# Load the generated data
df_clean = pd.read_csv("step2_oscillatory_data_clean.csv")
df_noisy = pd.read_csv("step2_oscillatory_data_noisy.csv")

# Extract time and displacement values & convert to PyTorch tensors
t_train = torch.tensor(df_noisy["t"].values.reshape(-1, 1), dtype=torch.float32)
u_train = torch.tensor(df_noisy["u"].values.reshape(-1, 1), dtype=torch.float32)
t_test = torch.tensor(df_clean["t"].values.reshape(-1, 1), dtype=torch.float32)
u_test = torch.tensor(df_clean["u"].values.reshape(-1, 1), dtype=torch.float32)

# ---- Polynomial Feature Transformation ----
degree = 30  # High-degree polynomial for best performance
poly = PolynomialFeatures(degree=degree)
scaler = StandardScaler()

# Convert tensors to NumPy before transformation
t_train_poly = poly.fit_transform(t_train.numpy())
t_train_poly = scaler.fit_transform(t_train_poly)
t_test_poly = poly.transform(t_test.numpy())
t_test_poly = scaler.transform(t_test_poly)

# ---- Initialize SGDRegressor ----
sgd_model = SGDRegressor(max_iter=1, warm_start=True, penalty=None, learning_rate='constant', eta0=0.001, random_state=42)

# Convert target to 1D NumPy array for SGDRegressor
u_train_1d = u_train.numpy().ravel()

# Lists to store error metrics over iterations
lr_epoch_history = []
lr_mse_history = []
lr_mae_history = []
lr_rmse_history = []

# ---- Iterative Training and Error Logging ----
n_iterations = 101  # Total number of iterations
lr_start = time.time()

for i in range(n_iterations):
    sgd_model.partial_fit(t_train_poly, u_train_1d)
    
    if i % 10 == 0:
        u_pred_lr = sgd_model.predict(t_test_poly)
        
        mse = mean_squared_error(u_test.numpy(), u_pred_lr)
        mae = np.mean(np.abs(u_test.numpy().ravel() - u_pred_lr))
        rmse = np.sqrt(mse)

        lr_epoch_history.append(i)
        lr_mse_history.append(mse)
        lr_mae_history.append(mae)
        lr_rmse_history.append(rmse)

lr_end = time.time()
lr_time = lr_end - lr_start

print("Task complete for LR.")

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

def to_tensor(data):
    """Ensure the data is a PyTorch tensor."""
    if isinstance(data, torch.Tensor):
        return data.float()  # Ensure correct dtype
    return torch.from_numpy(data).float()

# Convert only if the data is a NumPy array
t_tensor = to_tensor(t_train)
u_tensor = to_tensor(u_train)
t_test_tensor = to_tensor(t_test)
u_test_tensor = to_tensor(u_test)

# Initialize and train the NN model
nn_model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.01)  # Lower learning rate for smoother training

# Lists to record training metrics and epochs
NN_mse_history = []
NN_mae_history = []
NN_rmse_history = []
NN_epoch_history = []

nn_start = time.time()
for epoch in range(3001):
    optimizer.zero_grad()
    output = nn_model(t_tensor)
    loss_nn = criterion(output, u_tensor)
    loss_nn.backward()
    optimizer.step()
    
    # Record the error metrics every 1000 iterations
    if epoch % 100 == 0:
        NN_mse_val = loss_nn.item()  # MSE
        NN_mae_val = torch.mean(torch.abs(output - u_tensor)).item()  # MAE
        NN_rmse_val = np.sqrt(NN_mse_val)  # RMSE

        NN_mse_history.append(NN_mse_val)
        NN_mae_history.append(NN_mae_val)
        NN_rmse_history.append(NN_rmse_val)
        NN_epoch_history.append(epoch)

nn_end = time.time()
nn_time = nn_end - nn_start

print("Task completed for NN.")

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

# Initialize PINN
pinn = FCN(1, 1, 32, 3)

# Define Training Data
t_physics = torch.linspace(0, 1, 30).view(-1, 1).requires_grad_(True)

# Given Physics Constants
d, w0 = 0.5, 3
mut, kt = 2*d, w0**2

mui = torch.rand(1, requires_grad=True)
ki = torch.rand(1, requires_grad=True)

mu = torch.nn.Parameter(mui)
k = torch.nn.Parameter(ki)

pinn_mse_history, pinn_mae_history, pinn_rmse_history, pinn_epoch_history = [], [], [], []

optimiser = torch.optim.Adam(list(pinn.parameters()) + [mu, k], lr=0.005)
criterion = nn.MSELoss()

# Start PINN Training
pinn_start = time.time()
for i in range(3001):
    optimiser.zero_grad()

    # Compute Physics Loss
    u_physics = pinn(t_physics)
    dudt = torch.autograd.grad(u_physics, t_physics, torch.ones_like(u_physics), create_graph=True)[0]
    d2udt2 = torch.autograd.grad(dudt, t_physics, torch.ones_like(dudt), create_graph=True)[0]
    loss1 = torch.mean((d2udt2 + mu * dudt + 100 * k * u_physics) ** 2)

    # Compute Data Loss (Fix: Ensure t_train is a tensor)
    u_pred_train = pinn(t_train)
    loss2 = criterion(u_pred_train, u_train)

    # Compute Total Loss
    loss_pinn = loss1 + 5000 * loss2
    loss_pinn.backward()
    optimiser.step()

    # Store test predictions every 5000 iterations
    if i % 100 == 0:
        u_pred_test = pinn(t_test).detach()
        mse_test = mean_squared_error(u_test.numpy(), u_pred_test.numpy())
        mae_test = np.mean(np.abs(u_test.numpy().ravel() - u_pred_test.numpy().ravel()))
        rmse_test = np.sqrt(mse_test)

        pinn_mse_history.append(mse_test)
        pinn_mae_history.append(mae_test)
        pinn_rmse_history.append(rmse_test)
        pinn_epoch_history.append(i)

pinn_end = time.time()
pinn_time = pinn_end - pinn_start

print("Task completed for PINN.")

# ---- Plot Error Metrics ----
def plot_error_metrics(epoch_history, mse_history, mae_history, rmse_history, method, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_history, mse_history, label="MSE", color='blue')
    plt.plot(epoch_history, mae_history, label="MAE", color='red')
    plt.plot(epoch_history, rmse_history, label="RMSE", color='green')
    plt.xlabel("Epochs" if method=="Neural Network" else "Iterations")
    plt.ylabel("Error")
    plt.title(f"{method} Error Metrics")
    plt.legend()
    plt.savefig(filename)
    plt.show()

plot_error_metrics(lr_epoch_history, lr_mse_history, lr_mae_history, lr_rmse_history, "Polynomial Regression", "step6_LR_loss.png")
plot_error_metrics(NN_epoch_history, NN_mse_history, NN_mae_history, NN_rmse_history, "Neural Network", "step6_NN_loss.png")
plot_error_metrics(pinn_epoch_history, pinn_mse_history, pinn_mae_history, pinn_rmse_history, "PINN", "step6_pinn_loss.png")

# Obtain final predictions on test data
u_pred_nn = nn_model(t_test_tensor).detach().numpy()
u_pred_pinn = pinn(t_test_tensor).detach().numpy()
u_pred_final = sgd_model.predict(t_test_poly)
