from mse_vanilla import mean_squared_error as vanilla_mse
from mse_numpy import mean_squared_error as numpy_mse
import sklearn.metrics as sk
import timeit as it

observed = [2, 4, 6, 8]
predicted = [2.5, 3.5, 5.5, 7.5]

# Define argument mappings for different functions
vanilla_karg = {'observed': observed, 'predicted': predicted}
sklearn_karg = {'y_true': observed, 'y_pred': predicted}  # Sklearn expects y_true and y_pred

# Store function and its corresponding argument dictionary
factory = {
    'mse_vanilla': (vanilla_mse, vanilla_karg),
    'mse_numpy': (numpy_mse, vanilla_karg),
    'sk_mse': (sk.mean_squared_error, sklearn_karg)  # Ensure sklearn gets correct args
}

# Iterate through each function
for talker, (worker, karg) in factory.items():
    exec_time = it.timeit(lambda: worker(**karg), number=100) / 100
    mse = worker(**karg)
    print(f"Mean Squared Error, {talker} :", mse, f"Average execution time: {exec_time:.10f} seconds")