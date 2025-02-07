import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

#Define function to generate oscilatory data
def generate_oscillatory_data(t, d, w0, noise_level=0):
    """
    Generates a 1D damped oscillatory function u(t) = e^(-d*t) * cos(w0*t)
    with an option to add Gaussian noise.
    
    Parameters:
    - t (numpy array): Time points at which to evaluate the function.
    - d (float): Damping coefficient.
    - w0 (float): Natural frequency.
    - noise_level (float): Standard deviation of Gaussian noise to add.
    
    Returns:
    - u (numpy array): The computed oscillatory function values.
    """
    u = np.exp(-d * t) * np.cos(w0 * t)
    
    if noise_level > 0:
        u += np.random.normal(0, noise_level, size=t.shape)
    
    return u

#Generate the data and save it in two separate .csv files
def save_data():
    """
    Generates and saves oscillatory data with and without noise.
    """
    # Define parameters
    d, w0 = 0.5, 3  # Example damping coefficient and frequency
    t = np.linspace(0, 10, 500)  # Time vector from 0 to 10 seconds, 500 points
    n_points = len(t)  # Number of points in dataset
    
    # Generate data
    u_clean = generate_oscillatory_data(t, d, w0, noise_level=0)
    u_noisy = generate_oscillatory_data(t, d, w0, noise_level=0.03)
    
    # Save as CSV files
    df_clean = pd.DataFrame({"t": t, "u": u_clean})
    df_noisy = pd.DataFrame({"t": t, "u": u_noisy})
    
    df_clean.to_csv("step2_oscillatory_data_clean.csv", index=False)
    df_noisy.to_csv("step2_oscillatory_data_noisy.csv", index=False)

    # Plot the generated data
    plt.figure(figsize=(8, 4))
    plt.plot(t, u_clean, label="Truth", linestyle='--', color='blue')
    plt.scatter(t, u_noisy, label="Noisy Data", color='red', marker='o', s=5, alpha=0.6)
    plt.scatter(t, u_clean, label="Clean Data", color='black', marker='x', s=10, alpha=0.6)
    plt.xlabel("Time (t)")
    plt.ylabel("Displacement (u)")
    plt.legend()
    plt.title("Generated 1D Oscillatory Data")
    plt.savefig('step2_oscilatory_data')
    plt.show()

    # Print summary information
    print(f'Data generated: {n_points} points, Time range: {t[0]} to {t[-1]} seconds, Damping: {d}, Frequency: {w0}')

if __name__ == "__main__":
    save_data()
