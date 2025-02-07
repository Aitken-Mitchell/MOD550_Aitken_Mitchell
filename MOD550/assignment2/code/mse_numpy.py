import numpy as np

def mean_squared_error(observed, predicted):
    observed = np.array(observed)
    predicted = np.array(predicted)
    
    mse = np.mean((observed - predicted)**2)
    return mse