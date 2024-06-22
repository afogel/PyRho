import numpy as np
from typing import Union, Optional, Dict, Any
from .utils import as_code_set

class HandSet:
    def __init__(self, set_array: np.ndarray, handset_length: int, handset_baserate: float):
        """
        Initializes the HandSet class with the provided parameters.

        Parameters:
            set_array (np.ndarray): The set to take a handset of.
            handset_length (int): The length of the handset to take.
            handset_baserate (float): The minimum baserate to inflate the handset to.
        """
        self.set_array = set_array
        self.handset_length = handset_length
        self.handset_baserate = handset_baserate

    def get_handset(self, return_set: bool = False) -> Union[np.ndarray, float]:
        """
        Gets a handset of a set and calculates the kappa.

        Parameters:
            return_set (bool, optional): If True, then return the handset. If False, return the kappa of the handset. Defaults to False.

        Returns:
            Union[np.ndarray, float]: The handset if return_set is True or the kappa of the handset if not.
        """
        positives = int(np.ceil(self.handset_length * self.handset_baserate))
        pos_ind = np.where(self.set_array[:, 0] == 1)[0]

        if positives > len(pos_ind):
            raise ValueError("Not enough positives in first rater to inflate to this level")

        if positives > 0:
            positive_indices = np.random.choice(pos_ind, size=positives, replace=False)
            others = self.set_array[~np.isin(np.arange(len(self.set_array)), positive_indices)]
            other_indices = np.random.choice(np.arange(len(others)), size=(self.handset_length - positives), replace=False)
            this_set = np.vstack((self.set_array[positive_indices], others[other_indices]))
        else:
            these_indices = np.random.choice(np.arange(len(self.set_array)), size=self.handset_length, replace=False)
            this_set = self.set_array[these_indices]

        if return_set:
            this_set = as_code_set(this_set)
            return this_set

        handset_kappa = self.calc_kappa(this_set)
        return handset_kappa

    @staticmethod
    def calc_kappa(set_array: np.ndarray, is_set: bool = True, kappa_threshold: Optional[float] = None) -> Union[float, Dict[str, Any]]:
        """
        Calculates kappa given a set.

        Parameters:
            set_array (np.ndarray): The set to calculate kappa from.
            is_set (bool, optional): True if set, False if contingency table. Defaults to True.
            kappa_threshold (Optional[float], optional): If None, return kappa, otherwise return a dictionary with kappa and whether it is above the threshold.

        Returns:
            Union[float, Dict[str, Any]]: The kappa of the set or a dictionary with kappa and whether it is above the threshold.
        """
        if not is_set:
            set_array = HandSet.contingency_to_set(set_array[0, 0], set_array[1, 0], set_array[0, 1], set_array[1, 1])

        if np.all(set_array[:, 0] == set_array[:, 1]):
            return 1.0

        # Calculates kappa by calculating the agreement and the baserates and then creating the adjacency matrix
        agreement = np.mean(set_array[:, 0] == set_array[:, 1])
        baserate1 = np.mean(set_array[:, 0] == 1)
        baserate2 = np.mean(set_array[:, 1] == 1)
        random_agreement = baserate1 * baserate2 + (1 - baserate1) * (1 - baserate2)
        kappa = (agreement - random_agreement) / (1 - random_agreement)

        if kappa_threshold is None:
            return kappa
        else:
            return {'kappa': kappa, 'above': kappa > kappa_threshold}

    @staticmethod
    def contingency_to_set(TP: int, FP: int, FN: int, TN: int) -> np.ndarray:
        """
        Converts a contingency table into a set of actual and predicted class labels.

        Parameters:
            TP (int): True Positives - Number of instances where both the actual and predicted classes are positive.
            FP (int): False Positives - Number of instances where the actual class is negative, but the predicted class is positive.
            FN (int): False Negatives - Number of instances where the actual class is positive, but the predicted class is negative.
            TN (int): True Negatives - Number of instances where both the actual and predicted classes are negative.

        Returns:
            np.ndarray: A 2D numpy array where each row contains a pair of actual (gold) and predicted (silver) values.
                        - The first column (gold) represents the actual class labels.
                        - The second column (silver) represents the predicted class labels.

        Example:
            Given a contingency table:
            - TP = 3
            - FP = 2
            - FN = 1
            - TN = 4

            The function will produce the following set:
            array([[1, 1],
                   [1, 1],
                   [1, 1],
                   [1, 0],
                   [0, 1],
                   [0, 1],
                   [0, 0],
                   [0, 0],
                   [0, 0],
                   [0, 0]])
        """
        set_length = TP + FP + FN + TN

        gold_1s = TP + FN
        gold_0s = set_length - gold_1s

        gold = np.array([1] * gold_1s + [0] * gold_0s)
        silver = np.array([1] * TP + [0] * (gold_1s - TP) + [1] * FP + [0] * (gold_0s - FP))

        return np.column_stack((gold, silver))
