import numpy as np

def any_equal(target, current, tolerance=np.sqrt(np.finfo(float).eps)) -> bool:
    """
    Checks if any value in the target list is approximately equal to the current value within a given tolerance.

    Parameters:
        target (list): List of target values.
        current (float): Current value to compare.
        tolerance (float, optional): Tolerance for comparison. Defaults to the square root of machine epsilon.

    Returns:
        bool: True if any value in the target list is approximately equal to the current value, False otherwise.
    """
    return any(abs(current - targ) < tolerance for targ in target)

def all_equal(target, current, tolerance=np.sqrt(np.finfo(float).eps)) -> bool:
    """
    Checks if all values in the target list are approximately equal to the current value within a given tolerance.

    Parameters:
        target (list): List of target values.
        current (float): Current value to compare.
        tolerance (float, optional): Tolerance for comparison. Defaults to the square root of machine epsilon.

    Returns:
        bool: True if all values in the target list are approximately equal to the current value, False otherwise.
    """
    return all(abs(current - targ) < tolerance for targ in target)

def as_contingency_table(x: np.ndarray) -> np.ndarray:
    """
    Converts a codeset to a contingency table.

    Parameters:
        x (np.ndarray): The codeset to convert.

    Returns:
        np.ndarray: A 2x2 contingency table.
    """
    if x.shape[0] > 2 and x.shape[1] == 2:
        tp = np.sum((x[:, 0] == 1) & (x[:, 1] == 1))
        tn = np.sum((x[:, 0] == 0) & (x[:, 1] == 0))
        fp = np.sum((x[:, 0] == 0) & (x[:, 1] == 1))
        fn = np.sum((x[:, 0] == 1) & (x[:, 1] == 0))
        
        y = np.array([[tp, fp], [fn, tn]])
    else:
        y = x
    
    return y

def as_code_set(x: np.ndarray) -> np.ndarray:
    """
    Converts a contingency table to a codeset.

    Parameters:
        x (np.ndarray): A 2x2 contingency table.

    Returns:
        np.ndarray: A 2-column matrix representation of the contingency table.
    """
    if x.shape == (2, 2):
        y = np.vstack([
            np.tile([1, 1], x[0, 0]),  # TP
            np.tile([1, 0], x[0, 1]),  # FN
            np.tile([0, 1], x[1, 0]),  # FP
            np.tile([0, 0], x[1, 1])   # TN
        ])
    else:
        y = x
    
    return y

# Helper functions to add special values to a rating set

def get_rating_set_attribute(x: np.ndarray, attr_name: str):
    """
    Helper function to return special values on a rating set.

    Parameters:
        x (np.ndarray): Set or contingency table.
        attr_name (str): Attribute name to search for.

    Returns:
        Any: The attribute value if found, None otherwise.
    """
    attributes = ['baserate', 'kappa', 'agreement']
    if attr_name in attributes:
        return getattr(x, attr_name, None)
    return None

def dollar_names_rating_set(x: np.ndarray, pattern: str = ""):
    """
    Provides the names of attributes for a rating set.

    Parameters:
        x (np.ndarray): The rating set.
        pattern (str, optional): Pattern to match names. Defaults to "".

    Returns:
        list: List of attribute names.
    """
    attributes = ['baserate', 'kappa', 'agreement']
    cols = x.dtype.names if x.dtype.names is not None else []
    return attributes + list(cols)

def print_rating_set(x: np.ndarray):
    """
    Prints a rating set excluding additional attributes.

    Parameters:
        x (np.ndarray): The rating set.
    """
    additional_attributes = ['baserate', 'kappa', 'agreement']
    y = x.copy()
    for attr in additional_attributes:
        if hasattr(y, attr):
            delattr(y, attr)
    print(y)
