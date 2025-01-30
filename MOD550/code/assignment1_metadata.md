=== File Contents ===

# Metadata File

## Project Information
- **Assignment**: Assignment 1 of MOD550
- **Author**: Aitken Mitchell
- **Email**: a.mitchell@stud.uis.no
- **Date Created**: 2025-01-24
- **Description**: This project generates and visualizes synthetic datasets based on user-defined distributions. It saves the combined dataset as a CSV file and the visualization as a PNG image. The purpose is to explore distributions in Python.

## Data Information
- **Data Sources**: Synthetic data generated programmatically using Python.
- **Distributions Used**:
  - Class 1: Uniform distribution
    - Points: 412
    - Noise: 16
    - Range: [0, 50]
  - Class 2: Parabolic distribution
    - Points: 506
    - Noise: 6
    - Range: [0, 30]
- **Output File**: `combined_data.csv`
  - Columns:
    - `x`: Generated x-coordinates.
    - `y`: Generated y-coordinates.
    - `class`: Class label ("Class 1" or "Class 2").
  - Description: Contains combined x, y, and class labels for both datasets.

## Script Information
- **Script Name**: `exercise_1.py`
- **Description**: 
  - Generates synthetic datasets based on specified distributions.
  - Combines datasets from multiple classes.
  - Visualizes the data.
  - Saves the combined data and visualization for further use.
- **Dependencies**:
  - Python 3.9+
  - Libraries:
    - `numpy`
    - `matplotlib`
    - `pandas`
- **Execution Instructions**:
  1. Install dependencies using `pip install numpy matplotlib pandas`.
  2. Run the script: `python exercise_1.py`.

## Outputs
- **CSV File**: `combined_data.csv`
  - Contains combined data from all generated datasets.
- **Chart**: `combined_chart.png`
  - Visualizes the combined data with distinct colors for each class.

## Methods
- **Uniform Distribution**:
  - Formula: `y = np.random.uniform(low, high) * noise`
  - Noise added using `np.random.uniform`.
- **Parabolic Distribution**:
  - Formula: `y = a * (x - h)**2 + k + noise`
  - Parameters:
    - `a`: 3
    - `h`: Midpoint of the range.
    - `k`: 0 (y-intercept).
  - Noise added using `np.random.uniform`.

## Licensing and Ethics
- **License**: MIT License.
  - **License Text**:
    ```
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    ```
- **Ethical Considerations**: Synthetic data generated for educational purposes. No real-world data or personal information is used.

## Contact Information
- **Author**: Aitken Mitchell
- **Email**: a.mitchell@stud.uis.no
- **University**: University of Stavanger
