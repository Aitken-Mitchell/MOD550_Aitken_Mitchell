def mean_squared_error(observed, predicted):
    if len(observed) != len(predicted):
        raise ValueError("The lengths of input lists are" + f"not equal {len(observed)} != {len(predicted)}")
    
    # Initialise the sum of squared errors
    sum_squared_error = 0

    #Loop through all observations
    for obs, pred in zip(observed, predicted):
        # Calculate the square difference and add it to the sum
        sum_squared_error += (obs - pred)**2
        
    # Calculate the mean squared error
    mse = sum_squared_error / len(observed)

    return mse