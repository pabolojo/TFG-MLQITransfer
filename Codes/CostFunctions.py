import numpy as np
from typing import List, Optional

from QEnvs.QEnv import QEnv

class CostFunction:

    def __init__(self, proportional_weights: List[float], integral_weights: Optional[List[float]] = None, derivative_weights: Optional[List[float]] = None, end_episode_threshold: float = np.inf):
        """
        Punish the middle populations of the system.

        Parameters
        ----------
        proportional_weights : List[float]
            Weights for the each of the populations.

        integral_weights : List[float], optional [default = None]
            Weights for the each of the integrals of the populations.

        derivative_weights : List[float], optional [default = None]
            Weights for the each of the derivatives of the populations.

        end_episode_threshold : float, optional [default = np.inf]
            Threshold for stopping the episode.

        """
        self.proportional_weights = proportional_weights

        if integral_weights is None:
            integral_weights = []
        self.integral_weights = integral_weights

        if derivative_weights is None:
            derivative_weights = []
        self.derivative_weights = derivative_weights

        self.end_episode_threshold = end_episode_threshold


    def costFunction(self, env: QEnv):
        """
        Cost function for the QEnv.

        Parameters
        ----------
        env : QEnv
            Environment for the cost function.
        
        Returns
        -------
        cost : float
            Cost of the current state of the environment.

        end_episode : bool
            Flag for ending the episode.
        
        """
        
        if len(self.proportional_weights) > env.num_qubits or len(self.integral_weights) > env.num_qubits:
            raise ValueError("The length of the lists of weights is greater than the number of qubits.")
        
        cost = 0.
        for i, proportional in enumerate(self.proportional_weights):
            cost += proportional * env.populations[i][env.current_step]

        for i, integral in enumerate(self.integral_weights):
            cost += integral * env.populations_integrals[i][env.current_step]

        for i, derivative in enumerate(self.derivative_weights):
            cost += derivative * env.populations_derivatives[i][env.current_step]

        current_expectation = env.expectations[env.current_step]
        
        return env.reward_gain * cost, current_expectation >= self.end_episode_threshold
