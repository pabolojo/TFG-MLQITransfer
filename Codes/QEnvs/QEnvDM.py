import warnings

import numpy as np
import QTransferLib as qtr
import qutip as qt

from QEnvs.QEnv import QEnv
from tf_agents.specs import array_spec
from typing import Optional, List, Callable, Tuple
from tf_agents.trajectories import time_step as ts
from Noises import Noise

class QEnvDM(QEnv):

    """
    Environment for systems with collape operators.
    """

    def __init__(
        self,
        num_qubits: int,
        time_max: float,
        num_steps: int,
        c_ops: List,
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

        c_ops : List
            List of collapse operators.

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

        self.c_ops = c_ops
        if len(c_ops) == 0:
            raise ValueError("c_ops must have at least one element.")
        warnings.warn("c_ops is not None. Imaginary deltas may cause populations greater than 1.")
        
        self.initial_state = qt.ket2dm(self.initial_state)
        self.target_state =  qt.ket2dm(self.target_state)
        self.current_qstate = self.initial_state
        self._state = self._quantum2rlstate(self.initial_state)

    def _reset(self) -> ts.TimeStep:
        """
        Reset the environment.
        
        Returns
        -------
        ts.TimeStep
            Time step with the initial observation.
        """

        timestep = super()._reset()
        self.current_qstate = qt.ket2dm(self.psi0)

        return timestep
    
    def observation_spec(self) -> array_spec.BoundedArraySpec:
        num_diag_elements = self.num_qubits
        num_offdiag_elements = int(self.num_qubits * (self.num_qubits - 1) / 2)
        num_elements = num_diag_elements + 2 * num_offdiag_elements
        return array_spec.BoundedArraySpec(
            shape=(num_elements, ),
            dtype=np.float32,
            name="mesolve_result",
            minimum=np.append(
                np.zeros(num_diag_elements, dtype=np.float32), -1 * np.ones(2 * num_offdiag_elements, dtype=np.float32)
            ),
            maximum=np.ones(num_elements, dtype=np.float32),
        )

    def _quantum2rlstate(self, dm) -> np.ndarray:
        """
        Density Matrix to State.
        State : [diag, offdiag.real, offdiag.imag]
        """
        dm_offdiag = np.array(dm)
        mask = np.triu(np.ones((self.num_qubits, self.num_qubits), dtype=bool), k=1)
        dm_offdiag = dm_offdiag[mask]

        return np.concatenate(
            (dm.diag(),
            dm_offdiag.real,
            dm_offdiag.imag)
            ).astype(np.float32)

    def _saveStep(self, actions, qstate):
        target = self.target_state
        current = qstate
        populations = np.diag(qstate.full())
        
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
            H, qstate, times, c_ops=[self.c_ops], options=self.solve_opts
        )
        return result.states[-1]