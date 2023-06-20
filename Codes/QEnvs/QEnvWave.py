from QEnvs.QEnv import QEnv
from tf_agents.specs import array_spec
from typing import Optional, Callable, Tuple
from Noises import Noise

import numpy as np
import QTransferLib as qtr
import qutip as qt

class QEnvWave(QEnv):

    """
    Environment for simple systems.
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

        super().__init__(
            num_qubits=num_qubits,
            time_max=time_max,
            num_steps=num_steps,
            cost_function=cost_function,
            initial_state=initial_state,
            target_state=target_state,
            reward_gain=reward_gain,
            omega_min=omega_min,
            omega_max=omega_max,
            noise=noise,
            deltas=deltas,
            saveBestEpisodeDir=saveBestEpisodeDir,
            name=name)
    
        self.current_qstate = self.initial_state
        self._state = self._quantum2rlstate(self.initial_state)

    def observation_spec(self) -> array_spec.BoundedArraySpec:
        return array_spec.BoundedArraySpec(
            shape=(self.num_qubits, ),
            dtype=np.float32,
            name="mesolve_result",
            minimum=0.,
            maximum=1.,
        )
    
    def _quantum2rlstate(self, qstate) -> np.ndarray:
        return np.abs(np.array(qstate)).astype(np.float32).flatten()**2

    def _saveStep(self, actions, qstate):
        target = qt.ket2dm(self.target_state)
        current = qt.ket2dm(qstate)
        populations = np.abs(qstate.full())**2
        
        self.expectations[self.current_step] = np.abs(qt.expect(target, current))
        self.pulses[:, self.current_step] = actions
        self.populations[:, self.current_step] = populations.flatten()
        if self.current_step == 0:
            self.populations_integrals[:, self.current_step] = self.populations[:, self.current_step]
            self.populations_derivatives[:, self.current_step] = 0
        else:
            self.populations_integrals[:, self.current_step] = self.populations_integrals[:, self.current_step - 1] + self.populations[:, self.current_step]
            self.populations_derivatives[:, self.current_step] = self.populations[:, self.current_step] - self.populations[:, self.current_step - 1]

    def _solvestep(self, actions, qstate) -> np.ndarray:
        """
        Quantum step.
        """
        times = self.times[self.current_step : self.current_step + 2]

        if len(self.deltas.shape) == 1:
            H = qt.Qobj(np.diag(self.deltas))
        else:
            H = qt.Qobj(np.diag(self.deltas[:, self.current_step]))

        if self.noise is not None:
            actions = self.noise.add_noise(actions)
        
        for i, action in enumerate(actions):
            H += action * qtr.hamiltonian_coupling_factory(self.num_qubits, i)

        result = qt.mesolve(
            H, qstate, times, options=self.solve_opts
        )
        return result.states[-1]