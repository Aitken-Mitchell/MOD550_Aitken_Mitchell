# README

## Overview
This program automates the execution of multiple Python scripts in a sequential manner. It ensures that each script runs successfully before proceeding to the next one. The main controller script, `exercise_2.py`, calls eight other scripts in the following order:

1. `step1.py`
    1.1 `mse_numpy.py`
    1.2 `mse_vanilla.py`
2. `step2.py`
3. `step3.py`
4. `step4.py`
5. `step5.py`
6. `step6.py`
7. `step7.py`
8. `step8.py`

## Prerequisites
Before running the program, ensure you have the required dependencies installed. These dependencies are listed in the `requirements.txt` file.

### Install Dependencies
To install the necessary Python packages, run:
```bash
pip install -r requirements.txt
```

## Usage
### Running the Program
Execute the main script to run all steps in sequence:
```bash
python exercise_2.py
```

If any script encounters an error, the execution will stop, and an error message will be displayed.

## Dependencies
The program requires the following Python libraries:
- `scikit-learn`
- `numpy`
- `pandas`
- `matplotlib`
- `torch`
- `ipython`

These dependencies are listed in `requirements.txt` and can be installed as described above.

## Error Handling
If any script fails, the program will stop execution and display an error message. Ensure that all required files and dependencies are available before running the script.

## Notes
- Ensure that all Python scripts (`step1.py` to `step8.py` as well as `mse_numpy.py` & `mse_vanilla.py`) are located in the same directory as `exercise_2.py`.
- The scripts should be executable and correctly formatted.


