from typing import List, Dict, Optional, Union
from enum import Enum
import numpy as np
from scipy.stats import norm

class DistributionType(Enum):
    FLAT = 'FLAT'
    BELL = 'BELL'

def generate_kps(
    num_needed: int, baserate: float, 
    kappa_min: float, kappa_max: float, 
    precision_min: float, precision_max: float, 
    distribution_type: Union[DistributionType, str] = 'FLAT', 
    distribution_length: int = 100000
) -> List[Dict[str, float]]:
    """
    Generates a list of kappa-precision (KP) combinations based on specified parameters.

    Parameters:
        num_needed (int): Number of KP combinations needed.
        baserate (float): Base rate used for generating KP combinations.
        kappa_min (float): Minimum value of kappa for the distribution.
        kappa_max (float): Maximum value of kappa for the distribution.
        precision_min (float): Minimum value of precision for the distribution.
        precision_max (float): Maximum value of precision for the distribution.
        distribution_type (Union[DistributionType, str], optional): Type of distribution to use ('FLAT' or 'BELL'). Defaults to 'FLAT'.
        distribution_length (int, optional): Length of the kappa distribution. Defaults to 100000.

    Returns:
        List[Dict[str, float]]: A list of dictionaries containing kappa and precision values.
    """
    kappa_distribution = np.linspace(kappa_min, kappa_max, distribution_length)
    
    if distribution_type == DistributionType.FLAT.value:
        kappa_probability = None
    elif distribution_type == DistributionType.BELL.value:
        kappa_probability = norm.pdf(kappa_distribution, loc=0.9, scale=0.1)
    else:
        raise ValueError("Invalid distribution type specified.")
    
    precision_distribution = np.linspace(precision_min, precision_max, 10000)
    precision_probability = None

    KPs = []
    try:
        for _ in range(num_needed):
            kp = gen_pk_combo(kappa_distribution, kappa_probability, precision_distribution, precision_probability, baserate)
            KPs.append(kp)
    except Exception as e:
        raise ValueError("Could not generate a valid set given ranges of kappa and precision") from e

    return KPs

def check_br_pk_combo(baserate: float, precision: float, kappa: float) -> bool:
    """
    Checks to ensure the combination of baserate, precision, and kappa will create a valid set.

    Parameters:
        baserate (float): The supplied baserate.
        precision (float): The supplied precision.
        kappa (float): The supplied kappa.

    Returns:
        bool: True if the combination is valid, False otherwise.
    """
    right = (2 * baserate * kappa - 2 * baserate - kappa) / (kappa - 2)
    return precision > right

def gen_pk_combo(
    kappa_distribution: np.ndarray, 
    kappa_probability: Optional[np.ndarray], 
    precision_distribution: np.ndarray, 
    precision_probability: Optional[np.ndarray], 
    baserate: float
) -> Dict[str, float]:
    """
    Generates a precision-kappa (PK) combination based on specified parameters.

    Parameters:
        kappa_distribution (np.ndarray): The distribution of kappa values.
        kappa_probability (Optional[np.ndarray]): The probabilities associated with the kappa distribution. Can be None.
        precision_distribution (np.ndarray): The distribution of precision values.
        precision_probability (Optional[np.ndarray]): The probabilities associated with the precision distribution. Can be None.
        baserate (float): Base rate used for generating PK combinations.

    Returns:
        Dict[str, float]: A dictionary containing the precision and kappa values.
    """
    curr_kappa = np.random.choice(kappa_distribution, p=kappa_probability)
    curr_precision = np.random.choice(precision_distribution, p=precision_probability)

    if not check_br_pk_combo(baserate, curr_precision, curr_kappa):
        precision_min = (2 * baserate * curr_kappa - 2 * baserate - curr_kappa) / (curr_kappa - 2)
        indices = np.where(precision_distribution > precision_min)[0]

        if len(indices) == 0:
            return gen_pk_combo(kappa_distribution, kappa_probability, precision_distribution, precision_probability, baserate)

        precision_distribution = precision_distribution[indices]
        precision_probability = precision_probability[indices] if precision_probability is not None else None
        curr_precision = np.random.choice(precision_distribution, p=precision_probability)
        return {'precision': curr_precision, 'kappa': curr_kappa}
    else:
        return {'precision': curr_precision, 'kappa': curr_kappa}
