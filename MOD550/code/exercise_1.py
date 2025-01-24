import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
csv_file = "combined_data.csv"
combined_data.to_csv(csv_file, index=False)
print(f"Data saved to {csv_file}")

# Combine data for visualization
combined_x, combined_y = data_classes[0].data
for data_class in data_classes[1:]:
    combined_x, combined_y = np.append(combined_x, data_class.data[0]), np.append(combined_y, data_class.data[1])

# Plot combined data
plt.scatter(combined_x, combined_y, label="Combined Data", color="black", alpha=0.6, marker="x")
plt.title("Combined Data from Multiple Classes")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid()

# Save the chart as an image
image_file = "combined_chart.png"
plt.savefig(image_file)
print(f"Chart saved to {image_file}")

plt.show()