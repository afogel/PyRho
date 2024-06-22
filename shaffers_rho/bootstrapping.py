import numpy as np
from typing import List

def get_boot_p_value(distribution: List[float], result: float) -> float:
    """
    Gets the p-value of the result in the distribution.

    Parameters:
        distribution (List[float]): The distribution that the result falls into.
        result (float): The value that you are evaluating where in the distribution it falls.

    Returns:
        float: The p-value from the result into the distribution.
    """
    distribution_array = np.array(distribution)
    N = len(distribution_array)

    if result < np.mean(distribution_array):
        # If the result is less than the mean of the distribution, then the p-value is 1
        return 1.0
    else:
        # Return the number of times that the distribution is greater than or equal to the result
        return np.sum(distribution_array >= result) / N
