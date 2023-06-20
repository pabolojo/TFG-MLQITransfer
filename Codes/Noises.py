import numpy as np
import warnings
from typing import Optional

class Noise:
    """
    Noise class for adding noise to the pulses.
    """

    def __init__(self, noise_type: str, percentage: float, seed: Optional[int] = None):
        """
        Initialize the Noise class.

        Parameters
        ----------
        noise_type : str
            Type of noise. Can be "random" or "gaussian".

        percentage : float
            Percentage of the maximum pulse amplitude to be used as the maximum noise amplitude or the standard deviation of the noise.

        seed : int, optional [default = None]
            Seed for the random number generator.
        """

        if noise_type not in ["none", "random", "gaussian"]:
            raise ValueError("The noise type must be 'random' or 'gaussian'.")
        if percentage < 0:
            raise ValueError("The percentage must be positive.")
        elif percentage > 1:
            warnings.warn("The percentage is greater than 1.")
        
        self.noise_type = noise_type
        self.percentage = percentage
        self.seed = seed

        np.random.seed(self.seed)

    def add_noise(self, pulses: np.ndarray) -> np.ndarray:
        """
        Add noise to the pulses.

        Parameters
        ----------
        pulses : np.ndarray
            Array of shape (n_steps, N-1) with the pulses.
        
        Returns
        -------
        pulses : np.ndarray
            Array of shape (n_steps, N-1) with the pulses with noise.
        """

        if self.noise_type == "random":
            maximum = np.max(np.abs(pulses)) * self.percentage
            minimum = -maximum
            return pulses + np.random.uniform(minimum, maximum, size=pulses.shape)
        
        elif self.noise_type == "gaussian":
            maximum = np.max(np.abs(pulses)) * self.percentage
            return pulses + np.random.normal(0, maximum, size=pulses.shape)
        
        else:
            return pulses