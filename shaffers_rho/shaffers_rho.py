import numpy as np
from dataclasses import dataclass, field
from typing import Union, Optional, Dict, Any
from .kappa_precision import generate_kps
from .handsets import HandSet
from .bootstrapping import get_boot_p_value
from .calculations import rating_set_recall, rating_set_precision

@dataclass
class ShaffersRho:
    """
    ShaffersRho calculates the rho value for interrater reliability statistics.

    Parameters:
        x (Union[float, np.ndarray]): The observed kappa value, a codeset, or a 2x2 contingency table.
        observed_code_baserate (Optional[float], optional): Base rate used for calculating rho. Defaults to None.
        test_set_length (Optional[int], optional): Length of the test set. Defaults to None.
        testset_baserate_inflation (float, optional): Inflation factor for base rate in case of a test_set_length being used. Defaults to 0.0.
        observed_code_length (int, optional): Length of the overall set. Defaults to 10000.
        replicates (int, optional): Number of replicates for Monte Carlo simulation. Defaults to 800.
        simulated_code_kappa_threshold (float, optional): Threshold for rho calculation based on kappa values. Defaults to 0.9.
        simulated_code_min_kappa (float, optional): Minimum threshold value for the calculated kappa. Defaults to 0.40.
        simulated_code_min_precision (float, optional): Minimum precision threshold for the calculated rho. Defaults to 0.6.
        simulated_code_max_precision (float, optional): Maximum precision threshold for the calculated rho. Defaults to 1.0.
        method (str, optional): Calculation method ("standard" or "c"). Defaults to "standard".

    Attributes:
        result (Dict[str, Any]): The computed rho value based on the parameters provided and calculation method selected.
    """

    x: Union[float, np.ndarray]
    observed_code_baserate: Optional[float] = None
    test_set_length: Optional[int] = None
    testset_baserate_inflation: float = 0.0
    observed_code_length: int = 10000
    replicates: int = 800
    simulated_code_kappa_threshold: float = 0.9
    simulated_code_min_kappa: float = 0.40
    simulated_code_min_precision: float = 0.6
    simulated_code_max_precision: float = 1.0
    method: str = "standard"
    
    result: Dict[str, Any] = field(init=False, default=None)
    
    def __post_init__(self):
        """
        Post-initialization method to validate inputs and calculate the rho value.
        """
        self.validate_inputs()
        self.result = self.calculate_rho()

    def validate_inputs(self):
        """
        Validates the input parameters.

        Raises:
            ValueError: If the inputs do not meet specified conditions.
        """
        if isinstance(self.x, (int, float)):
            if self.observed_code_baserate is None or self.test_set_length is None:
                raise ValueError("When x is a kappa value, both 'observed_code_baserate' and 'test_set_length' must be provided.")
        elif isinstance(self.x, np.ndarray):
            if self.x.ndim != 2 or (self.x.shape[0] != 2 and self.x.shape[1] != 2):
                raise ValueError("When x is an array, it must be either a code set or a 2x2 contingency table.")
        else:
            raise ValueError("Invalid input type for x. Must be a numeric value or a 2D numpy array.")

    def calculate_rho(self) -> Dict[str, Any]:
        """
        Calculates rho based on the selected method and input type.

        Returns:
            Dict[str, Any]: A dictionary containing rho, kappa, recall, and precision.
        """
        if isinstance(self.x, (int, float)):
            return self.calculate_rho_kappa()
        elif self.x.shape[0] != 2:
            return self.calculate_rho_set()
        else:
            return self.calculate_rho_contingency_table()

    def calculate_rho_kappa(self) -> Dict[str, Any]:
        """
        Calculates rho given a kappa value.

        Returns:
            Dict[str, Any]: A dictionary containing rho, kappa, recall, and precision.
        """
        observed_kappa = self.x
        rho_value = self._calculate_rho(observed_kappa)

        return {
            'rho': rho_value,
            'kappa': observed_kappa,
            'recall': None,  # Recall is not applicable for a single kappa value
            'precision': None  # Precision is not applicable for a single kappa value
        }

    def calculate_rho_set(self) -> Dict[str, Any]:
        """
        Calculates rho given a code set.

        Returns:
            Dict[str, Any]: A dictionary containing rho, kappa, recall, and precision.
        """
        kappa = HandSet.calc_kappa(self.x)
        row_count = self.x.shape[0]

        if self.observed_code_baserate is None:
            self.observed_code_baserate = np.sum(self.x[:, 0] == 1) / row_count

        rho_value = self._calculate_rho(observed_kappa=kappa)

        return {
            'rho': rho_value,
            'kappa': kappa,
            'recall': rating_set_recall(self.x),
            'precision': rating_set_precision(self.x)
        }

    def calculate_rho_contingency_table(self) -> Dict[str, Any]:
        """
        Calculates rho given a contingency table.

        Returns:
            Dict[str, Any]: A dictionary containing rho, kappa, recall, and precision.
        """
        if np.any(self.x < 0):
            raise ValueError("Values in Contingency Table must be positive")

        kappa = HandSet.calc_kappa(self.x)

        if self.observed_code_baserate is None:
            self.observed_code_baserate = np.sum(self.x[0, :]) / np.sum(self.x)

        rho_value = self._calculate_rho(observed_kappa=kappa)

        return {
            'rho': rho_value,
            'kappa': kappa,
            'recall': rating_set_recall(self.x),
            'precision': rating_set_precision(self.x)
        }

    def _calculate_rho(self, observed_kappa=None) -> float:
        """
        Calculates rho for an observed kappa value with associated set parameters.

        Parameters:
            observed_kappa (Optional[float], optional): The observed kappa value. Defaults to None.

        Returns:
            float: The bootstrapped p-value corresponding to the observed kappa value and distribution of kappas generated from Monte Carlo simulations.
        """
        if observed_kappa is None:
            observed_kappa = self.x

        # Generate a distribution of random kappa sets using the given parameters
        kappas = generate_kps(self.replicates, self.observed_code_baserate, self.simulated_code_min_kappa, 
                                 self.simulated_code_kappa_threshold, self.simulated_code_min_precision, 
                                 self.simulated_code_max_precision)
        
        if len(kappas) < self.replicates:
            self.replicates = len(kappas)
        
        saved_kappas = np.zeros(self.replicates)
        
        # Loop through the generated KP sets to compute kappa values for each handset
        for i in range(self.replicates):
            KP = kappas[i]
            handset = HandSet(KP['precision'], KP['recall'], self.test_set_length, self.testset_baserate_inflation)
            kappa = handset.get_handset()
            # Add the computed kappa to the distribution of saved Kappas
            saved_kappas[i] = kappa
        
        # Compute and return the bootstrapped p-value comparing observed kappa with distribution
        return get_boot_p_value(saved_kappas, observed_kappa)
    
    def _calculate_rho_c(self):
        """
        Calculates rho using the C++ implemented method.

        Returns:
            float: The computed rho value using the C++ implemented method.
        """
        return rhoR.calcRho_c(
            self.x, self.observed_code_baserate, self.test_set_length,
            self.testset_baserate_inflation, self.observed_code_length, self.replicates,
            self.simulated_code_kappa_threshold, self.simulated_code_min_kappa,
            self.simulated_code_min_precision, self.simulated_code_max_precision
        )