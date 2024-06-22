import numpy as np
from .utils import as_contingency_table

def rating_set_recall(x: np.ndarray) -> float:
    """
    Calculates the recall for a rating set.

    Parameters:
        x (np.ndarray): The rating set or contingency table.

    Returns:
        float: The recall value.
    """
    if x.dtype.names is None or 'agreement' not in x.dtype.names:
        x = as_contingency_table(x)
    
    true_positive = x['agreement'][0]['true_positive']
    false_negative = x['agreement'][0]['false_negative']
    
    return true_positive / (true_positive + false_negative)

def rating_set_precision(x: np.ndarray) -> float:
    """
    Calculates the precision for a rating set.

    Parameters:
        x (np.ndarray): The rating set or contingency table.

    Returns:
        float: The precision value.
    """
    if x.dtype.names is None or 'agreement' not in x.dtype.names:
        x = as_contingency_table(x)
    
    true_positive = x['agreement'][0]['true_positive']
    false_positive = x['agreement'][0]['false_positive']
    
    return true_positive / (true_positive + false_positive)
