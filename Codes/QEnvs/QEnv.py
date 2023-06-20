from __future__ import annotations
from abc import abstractmethod

import numpy as np
from tf_agents.environments import py_environment
import qutip as qt
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec
from typing import Optional, List, Callable, Tuple
from Noises import Noise

class QEnv(py_environment.PyEnvironment):

    """
    General environment for the agents to evaluate quantum transfer actions.
    """

    def __init__(
            self,
            num_qubits: int,
            time_max: float,
            num_steps: int,
            cost_function: Callable[[QEnv], Tuple[float, bool]],
            initial_state: Optional[qt.Qobj] = None,
            target_state: Optional[qt.Qobj] = None,
            reward_gain: Optional[float] = 1.0,
            omega_min: Optional[float] = 0.0,
            omega_max: Optional[float] = 1.0,
            noise: Optional[Noise] = None,
            deltas: Optional[np.ndarray] = None,
            saveBestEpisodeDir: Optional[str] = None,
            name: Optional[str] = "env") -> None:
        """
        Initialize the environment.

        Parameters
        ----------
        num_qubits : int
            Number of qubits.

        time_max : float
            Maximum time.

        num_steps : int
            Number of steps.

        cost_function : function
            Cost function.

        initial_state : qt.Qobj, optional [default = qt.basis(N, 0)]
            Initial state.

        target_state : qt.Qobj, optional [default = qt.basis(N, N-1)]
            Target state.

        reward_gain : float, optional [default = 1.0]
            Gain of the reward.

        c_ops : List, optional [default = []]
            List of collapse operators.

        omega_min : float, optional [default = 0.0]
            Minimum amplitude of the pulse.
        
        omega_max : float, optional [default = 1.0]
            Maximum amplitude of the pulse.

        noise : Noise, optional [default = None]
            Noise to be added to the pulses.

        deltas : np.ndarray, optional [default = np.zeros((N, num_steps))]
            Array of shape (N, num_steps) with the qubit relative time dependent energies.

        saveBestEpisodeDir : str, optional [default = None]
            Directory for saving the best episode.
        
        name : str, optional [default = "env"]
            Name of the environment.

        """

        if initial_state is None:
            initial_state = qt.basis(num_qubits, 0)
        if target_state is None:
            target_state = qt.basis(num_qubits, num_qubits-1)
        if deltas is None:
            deltas = np.zeros(num_qubits, dtype=complex)

        self.num_qubits = num_qubits
        self.time_max = time_max
        self.num_steps = num_steps
        self.cost_function = cost_function
        self.psi0 = initial_state

        self.initial_state = initial_state
        self.target_state = target_state
        
        self.reward_gain = reward_gain
        
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.noise = noise
        self.deltas = deltas

        self.name = name
        if saveBestEpisodeDir is None:
            self.saveBestEpisode = False
        else:
            self.saveBestEpisode = True
            self.saveBestEpisodeDir = saveBestEpisodeDir + "/" + self.name + "_best_episode.npy"

        self.current_step = 0
        self._episode_ended = False
        self.solve_opts = qt.Options(atol=1e-11, rtol=1e-9, nsteps=int(1e6))
        self.solve_opts.normalize_output = False

        self.expectations = np.zeros(self.num_steps + 1)
        self.pulses = np.zeros((num_qubits - 1, self.num_steps))
        self.populations = np.zeros((num_qubits, self.num_steps + 1))
        self.populations_integrals = np.zeros((num_qubits, self.num_steps + 1))
        self.populations_derivatives = np.zeros((num_qubits, self.num_steps + 1))
        self.costs = np.zeros(self.num_steps + 1)

        self.best_reward = -np.inf
        self.best_pulses = np.zeros((num_qubits - 1, self.num_steps))
        self.best_populations = np.zeros((num_qubits, self.num_steps + 1))
        self.best_costs = np.zeros(self.num_steps + 1)

        self.update()

    def update(self) -> None:
        """
        Update the environment.
        """

        self.times = np.linspace(0, self.time_max, self.num_steps + 1)
        self.Δt = self.times[1] - self.times[0]

    def _reset(self) -> ts.TimeStep:
        """
        Reset the environment.
        
        Returns
        -------
        ts.TimeStep
            Time step with the initial observation.
        """

        self.current_step = 0
        self.current_qstate = self.psi0
        self._state = self._quantum2rlstate(self.initial_state)
        self._episode_ended = False

        self.pulses = np.zeros((self.num_qubits - 1, self.num_steps))
        self.populations = np.zeros((self.num_qubits, self.num_steps + 1))
        self.populations_integrals = np.zeros((self.num_qubits, self.num_steps + 1))
        self.populations_derivatives = np.zeros((self.num_qubits, self.num_steps + 1))
        self.costs = np.zeros(self.num_steps + 1)

        return ts.restart(self._state)

    def _step(self, action: np.ndarray) -> ts.TimeStep:
        """
        Take a step in the environment.

        Parameters
        ----------
        action : np.ndarray
            Array of shape (N, 2) with the pulse amplitudes.

        Returns
        -------
        ts.TimeStep
            Time step with the next observation.
        """

        if self._episode_ended:
            return self.reset()

        if self.current_step < self.num_steps:
            self.current_qstate = self._solvestep(action, self.current_qstate)
            next_state = self._quantum2rlstate(self.current_qstate)
            self._saveStep(action, self.current_qstate)
            reward, end_episode = self.cost_function(self)
            self.costs[self.current_step] = reward
            terminal = False

            if self.current_step == self.num_steps - 1 or end_episode:
                terminal = True
        else:
            terminal = True
            reward = -np.inf
            next_state = 0
        
        self.current_step += 1

        if terminal:
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_pulses = self.pulses
                self.best_populations = self.populations
                self.best_costs = self.costs
                if self.saveBestEpisode:
                    self._saveEpisode(self.pulses, self.populations, self.costs)

            self._episode_ended = True
            return ts.termination(next_state, reward)
        else:
            return ts.transition(next_state, reward)

    def action_spec(self) -> array_spec.BoundedArraySpec:
        """
        Return the action spec.
        """

        return array_spec.BoundedArraySpec(
            shape=(self.num_qubits - 1,),
            dtype=np.float32,
            name="pulses",
            minimum=self.omega_min,
            maximum=self.omega_max,
        )

    @abstractmethod
    def observation_spec(self) -> array_spec.BoundedArraySpec:
        """
        Returns the observation spec.
        """
        raise NotImplementedError

    @abstractmethod
    def _quantum2rlstate(self, qstate) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _saveStep(self, actions, qstate):
        raise NotImplementedError
    
    def _saveEpisode(self, pulses, populations, costs):
        if self.saveBestEpisode:     
            labels = ["Time", "Pulses", "Populations", "Costs"]
            save = np.array([np.array(labels), self.times, pulses, populations, costs])
            np.save(self.saveBestEpisodeDir, save)

    @abstractmethod
    def _solvestep(self, actions, qstate) -> np.ndarray:
        raise NotImplementedError

    def run_qevolution(self, actions: np.ndarray) -> np.ndarray:
        """
        Run the evolution of the system with the given pulse amplitudes.

        Parameters
        ----------
        Ωs : np.ndarray
            Array of shape (num_steps, N - 1) with the pulse amplitudes.
        
        Returns
        -------
        states : list
            List of quantum states.
        """

        states = [self.initial_state]

        for i in range(self.num_steps):
            states.append(self._solvestep(actions[i], states[-1]))

        return np.array(states)
