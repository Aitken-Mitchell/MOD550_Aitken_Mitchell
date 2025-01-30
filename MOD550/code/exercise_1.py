import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import textwrap
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import json

os.environ["LOKY_MAX_CPU_COUNT"] = "1"  # Prevents joblib from checking physical cores

# Define the script directory
script_dir = Path(__file__).parent  # Gets the folder where the script is located

#### Steps 1 - 4 of Assignment 1
#Define class for DataGenerator
class DataGenerator:
    #Define data class and attributes
    def __init__(self, description='Data generator', n_points=500, noise=75, range=[0, 50], distribution='uniform'):
        self.description = description
        self.n_points = n_points
        self.noise = noise
        self.range = range
        self.distribution = distribution
        self.data = None

    #Define the method for generating random data as well as defining different distributions
    def generate_data(self):
        low, high = self.range
        noise = self.noise
        
        # Generate random x and y data for exponential distribution
        if self.distribution == 'uniform':
            x = np.random.uniform(low, high, self.n_points) #Use the low / high to have the exponential curve within the range
            #Add noise to the y data
            y = np.random.uniform(low, high, self.n_points) * self.noise
        
        #Generate random x and y data for parabolic distribution
        elif self.distribution == 'parabolic':
            a = 3 #Define the coefficient for the parabolic curve
            h = (low + high) / 2 #Use low / high to center the parabola in the middle of the range
            k = 0 #Define the y-intercept for the parabolic curve
            x = np.random.uniform(low, high, self.n_points)
            y = a * (x - h)**2 + k
            #Add noise to the y data
            y += np.random.uniform(low, high, self.n_points) * self.noise
        
        # If no valid distribution is provided
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

        self.data = (x, y)  # Store the generated data

    #Define the method to combine the different datasets (method is appending the datasets)
    def __add__(self, other):
        if self.data and other.data:
            x_combined = np.append(self.data[0], other.data[0])
            y_combined = np.append(self.data[1], other.data[1])
            return x_combined, y_combined
        else: #If no data is generated for both classes then display and error
            raise ValueError("Both classes must have generated data before combining.")

    #Define the method for plotting the data
    def plot_data(self, color=None):
        if self.data:
            x, y = self.data
            plt.scatter(x, y, label=self.description, color=color)
        else:
            print(f"No data generated yet for {self.description}")

# Create multiple data classes with distinct parameters including distribution type
data_classes = [
    DataGenerator(description="Class 1", n_points=412, noise=16, range=[0, 50], distribution='uniform'),
    DataGenerator(description="Class 2", n_points=506, noise=6, range=[0, 30], distribution='parabolic')
]

# Generate and plot individual datasets
plt.figure(figsize=(12, 8))
all_data = []
colors = ["red", "blue"]

for data_class, color in zip(data_classes, colors):
    data_class.generate_data()
    x, y = data_class.data
    all_data.append(pd.DataFrame({"x": x, "y": y, "class": data_class.description}))
    data_class.plot_data(color=color)

# Combine all datasets into one DataFrame
combined_data = pd.concat(all_data, ignore_index=True)

# Save combined data to a CSV file
csv_file = "steps1_to_4.csv"
combined_data.to_csv(csv_file, index=False)
print(f"Data saved to {csv_file}")

# Combine data for visualization
combined_x, combined_y = data_classes[0].data
for data_class in data_classes[1:]:
    combined_x, combined_y = np.append(combined_x, data_class.data[0]), np.append(combined_y, data_class.data[1])

# Plot combined data
plt.scatter(combined_x, combined_y, label="Combined Data", color="black", alpha=0.6, marker="x")
plt.title("Steps 1 to 4 of Assignment 1")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid()

# Save the chart as an image
image_file = "steps1_to_4_plot.png"
plt.savefig(image_file)
print(f"Chart saved to {image_file}")

#####Step 5 of Assignment 1 - Open MetaData file and display the content
# Define file paths
input_file_md = Path("assignment1_metadata.md")
output_file_md = Path("step5_assignment1_metadata.md")

# Check if the input file exists
if input_file_md.exists():
    # Open and read the file
    with input_file_md.open("r") as file:
        content = file.read()

    # Write the content to a new file
    with output_file_md.open("w") as file:
        file.write("=== File Contents ===\n")
        file.write(content)

    print(f"Content successfully written to {output_file_md}")

else:
    print(f"Error: {input_file_md} not found.")

#####Step 6-7 of Assignment 1 - review another student's data and metadata and provide assumption of the truth of the source data and make regression of the imported data
# Import the data from the CSV file
df = pd.read_csv('assignment_1_anders_data.csv')

# Assign x and y
x = df['x']
y = df['y']

# Define the approximate sine wave formula from the last step
amplitude = 11
period = 6.3
y_expected = amplitude * np.sin((2 * np.pi / period) * x)

# Calculate residuals
residuals = y - y_expected

# Set a threshold for separating sine wave data
threshold = 4.5  # Adjusted based on what looked right on the chart
sine_wave_data = df[np.abs(residuals) <= threshold]
random_data = df[np.abs(residuals) > threshold]

#Clear previous plot
plt.clf()  # Clears the current figure
plt.figure(figsize=(12, 8))  # Create a new figure

# Plot the results
plt.scatter(sine_wave_data['x'], sine_wave_data['y'], color='blue', label='Sine wave data')
plt.scatter(random_data['x'], random_data['y'], color='orange', label='Random data')
plt.plot(x, y_expected, color='red', label='Expected sine wave', linewidth=2)

# Customize the plot
plt.xlabel('x')
plt.ylabel('y')
plt.title('Step 6-7 of Assignment 1 Assumed Truth of Source Data')
plt.legend()
# **Add a text box**
text_str = "For this plot I saw visually that the data resembled a sine wave and through iteration landed on an amplitude of 11 and period of 6.3. I then used residuals with a threshold of 4.5 to separate the sine data from the uniform data."
# Wrap the text into multiple lines
wrapped_text = "\n".join(textwrap.wrap(text_str, width=60))  # Adjust width as needed
plt.text(
    0.05, 0.95, wrapped_text, transform=plt.gca().transAxes, fontsize=12,
    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black')
)

# Save the chart as an image
image_file_1 = "step6-7_plot.png"
plt.savefig(image_file_1)
print(f"Chart saved to {image_file_1}")

######Step 7a of Assignment 1 - Make a regression of the imported data
# Given Metadata (Directly Included in the Script)
metadata = {
    "filename": "assignment_1_anders_data.csv",
    "length": 350,
    "x_min": 0.0,
    "x_max": 10.0,
    "x_mean": 4.696139785038783,
    "y_min": -20.61551606735788,
    "y_max": 25.8732511302362,
    "y_mean": 0.41362633372123975
}

# Load the dataset
df = pd.read_csv(metadata["filename"])

# Verify that dataset matches metadata (Ensures correctness)
assert len(df) == metadata["length"], "Error: Data length mismatch!"
assert np.isclose(df["x"].min(), metadata["x_min"], atol=0.05), "Error: x_min mismatch!"
assert np.isclose(df["x"].max(), metadata["x_max"], atol=0.05), "Error: x_max mismatch!"
assert np.isclose(df["x"].mean(), metadata["x_mean"], atol=0.05), "Error: x_mean mismatch!"
assert np.isclose(df["y"].min(), metadata["y_min"], atol=0.05), "Error: y_min mismatch!"
assert np.isclose(df["y"].max(), metadata["y_max"], atol=0.05), "Error: y_max mismatch!"
assert np.isclose(df["y"].mean(), metadata["y_mean"], atol=0.05), "Error: y_mean mismatch!"

print("Dataset matches metadata!")

# Define a sine wave function for fitting
def sine_function(x, A, B, C, D):
    return A * np.sin(B * x + C) + D  # A = Amplitude, B = Frequency, C = Phase, D = Offset

# Extract x and y values
x_data = df["x"].values
y_data = df["y"].values

# Initial guess for sine wave parameters
initial_guess = [10, 2 * np.pi / 6.3, 0, 0]  # (Amplitude, Frequency, Phase, Offset)

# Fit the sine wave model to the data
params, covariance = curve_fit(sine_function, x_data, y_data, p0=initial_guess)

# Extract fitted sine wave parameters
A_fit, B_fit, C_fit, D_fit = params
print(f"🔹 Best-Fit Sine Wave Parameters: Amplitude={A_fit:.4f}, Frequency={B_fit:.4f}, Phase={C_fit:.4f}, Offset={D_fit:.4f}")

# Generate predicted sine wave using the fitted parameters
df["y_pred"] = sine_function(x_data, A_fit, B_fit, C_fit, D_fit)

# Calculate residuals (distance between actual and predicted values)
df["residuals"] = np.abs(df["y"] - df["y_pred"])

# Use KMeans to classify points into two clusters: Sine Wave and Random Noise
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(df[["residuals"]])

# Identify which cluster corresponds to sine wave points
sine_cluster = df.groupby("cluster")["residuals"].mean().idxmin()  # Lower residuals → Sine Wave

# Separate sine wave and random data
sine_wave_data = df[df["cluster"] == sine_cluster]
random_data = df[df["cluster"] != sine_cluster]

# Compute metadata for each dataset
def compute_metadata(data, filename):
    return {
        "filename": filename,
        "length": len(data),
        "x_min": float(data["x"].min()),
        "x_max": float(data["x"].max()),
        "x_mean": float(data["x"].mean()),
        "y_min": float(data["y"].min()),
        "y_max": float(data["y"].max()),
        "y_mean": float(data["y"].mean()),
    }

sine_wave_metadata = compute_metadata(sine_wave_data, "sine_wave_data.csv")
random_metadata = compute_metadata(random_data, "random_data.csv")

# Save datasets to CSV
sine_wave_data.to_csv("sine_wave_data.csv", index=False)
random_data.to_csv("random_data.csv", index=False)

# Save metadata to JSON
with open("sine_wave_metadata.json", "w") as f:
    json.dump(sine_wave_metadata, f, indent=4)
with open("random_data_metadata.json", "w") as f:
    json.dump(random_metadata, f, indent=4)

# Print metadata
print("Sine Wave Metadata:", sine_wave_metadata)
print("Random Data Metadata:", random_metadata)

#Clear previous plot
plt.clf()  # Clears the current figure
plt.figure(figsize=(12, 8))  # Create a new figure

# 🔹 Scatter plot: Sine wave data vs. Random noise
plt.scatter(sine_wave_data["x"], sine_wave_data["y"], color='blue', label='Sine Wave Data')
plt.scatter(random_data["x"], random_data["y"], color='orange', label='Random Data')

# 🔹 Plot the fitted sine wave
x_fit = np.linspace(metadata["x_min"], metadata["x_max"], 1000)
y_fit = sine_function(x_fit, A_fit, B_fit, C_fit, D_fit)
plt.plot(x_fit, y_fit, color='red', linewidth=2, label='Fitted Sine Wave')

# Customize the plot
plt.xlabel('x')
plt.ylabel('y')
plt.title('Separated Sine Wave and Random Data')
plt.legend()
plt.grid()
# **Add a text box**
text_str = "For this plot Scikit-Learn was used and provided with the metadata, it splits the data into two sets using Kmean clustering, fits a sine wave to one dataset and plots the datasets."
# Wrap the text into multiple lines
wrapped_text = "\n".join(textwrap.wrap(text_str, width=60))  # Adjust width as needed
plt.text(
    0.05, 0.95, wrapped_text, transform=plt.gca().transAxes, fontsize=12,
    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black')
)

# Save the chart as an image
image_file_2 = "step7a_plot.png"
plt.savefig(image_file_2)
print(f"Chart saved to {image_file_2}")

######Step 8 of Assignment 1 Select 3 github/gitlab repository and make an assesment on the coding standards they use. Open .txt file
# Define file paths
input_file_txt = Path("community_standards_review.txt")
output_file_txt = Path("step8_3_githubs.txt")

# Check if the input file exists
if input_file_txt.exists():
    # Open and read the file
    with input_file_txt.open("r") as file:
        content = file.read()

    # Write the content to a new file
    with output_file_txt.open("w") as file:
        file.write("=== File Contents ===\n")
        file.write(content)

    print(f"Content successfully written to {output_file_txt}")

else:
    print(f"Error: {input_file_txt} not found.")


#Open all files at the same time

# Define file paths
image_file = "steps1_to_4_plot.png"
md_file = "step5_assignment1_metadata.md"
image_file_1 = "step6-7_plot.png"
image_file_2 = "step7a_plot.png"
text_file = "step8_3_githubs.txt"


# Open files based on OS
if sys.platform == "win32":  # Windows
    os.startfile(image_file)  # Open image
    os.startfile(md_file)   # Open md file
    os.startfile(image_file_1)  # Open image
    os.startfile(image_file_2)  # Open image
    os.startfile(text_file)   # Open text file


elif sys.platform == "darwin":  # macOS
    subprocess.Popen(["open", image_file])  # Open image
    subprocess.Popen(["open", md_file])   # Open md file
    subprocess.Popen(["open", image_file_1])  # Open image
    subprocess.Popen(["open", image_file_2])  # Open image
    subprocess.Popen(["open", text_file])   # Open text file
    

else:  # Linux
    subprocess.Popen(["xdg-open", image_file])  # Open image
    subprocess.Popen(["xdg-open", md_file])   # Open md file
    subprocess.Popen(["xdg-open", image_file_1])  # Open image
    subprocess.Popen(["xdg-open", image_file_2])  # Open image
    subprocess.Popen(["xdg-open", text_file])   # Open text file
    

