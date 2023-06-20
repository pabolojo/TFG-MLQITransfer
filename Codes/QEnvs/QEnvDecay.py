import warnings

import numpy as np
import QTransferLib as qtr
import qutip as qt

from QEnvs.QEnv import QEnv
from QEnvs.QEnvDM import QEnvDM
from tf_agents.specs import array_spec
from typing import Optional, List, Callable, Tuple
from tf_agents.trajectories import time_step as ts
from Noises import Noise

class QEnvDecay(QEnvDM):

    """
    Environment for systems with decay.
    """

    def __init__(
        self,
        num_qubits: int,
        time_max: float,
        num_steps: int,
        cost_function: Callable[[QEnv], Tuple[float, bool]],
        decay_factors: List,
        c_ops: Optional[List] = None,
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

        decay_factors : List
            List of decay factors.

        c_ops : List, optional [default = None]
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
        """

        if c_ops is None:
            c_ops = []

        if len(decay_factors) != num_qubits:
            raise ValueError("Number of decay factors must be equal to the number of number of qubits.")

        if initial_state is not None and initial_state.shape[0] == num_qubits:
            initial_state = qt.Qobj(np.append(np.array(initial_state), 0))

        if target_state is not None and target_state.shape[0] == num_qubits:
            target_state = qt.Qobj(np.append(np.array(target_state), 0))
        
        if deltas is not None and len(deltas) == num_qubits:
            deltas = np.append(deltas, 0)
        
        for index, decay in enumerate(decay_factors):
            decay_op = np.zeros((num_qubits + 1, num_qubits + 1))
            decay_op[-1, index] = 1
            c_ops.append(np.sqrt(decay) * qt.Qobj(decay_op))

        super().__init__(
            num_qubits + 1,
            time_max,
            num_steps,
            c_ops,
            cost_function,
            initial_state,
            target_state,
            reward_gain,
            omega_min,
            omega_max,
            noise,
            deltas,
            saveBestEpisodeDir,
            name)
        
    def action_spec(self) -> array_spec.BoundedArraySpec:
        """
        Return the action spec.
        """

        return array_spec.BoundedArraySpec(
            shape=(self.num_qubits - 2,),
            dtype=np.float32,
            name="pulses",
            minimum=self.omega_min,
            maximum=self.omega_max,
        )

    def observation_spec(self) -> array_spec.BoundedArraySpec:
        num_diag_elements = self.num_qubits - 1
        num_offdiag_elements = int(num_diag_elements * (num_diag_elements - 1) / 2)
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
        mask[:,-1] = np.zeros(self.num_qubits, dtype=bool)
        dm_offdiag = dm_offdiag[mask]

        return np.concatenate(
            (dm.diag()[:-1],
            dm_offdiag.real,
            dm_offdiag.imag)
            ).astype(np.float32)

    def _saveStep(self, actions, qstate):
        target = self.target_state
        current = qstate
        populations = np.diag(qstate.full())
        # Adding dummy action for the last qubit
        actions = np.append(actions, self.omega_min)
        
        self.expectations[self.current_step] = np.abs(qt.expect(target, current))
        self.pulses[:, self.current_step] = actions
        self.populations[:, self.current_step] = populations.flatten()
        if self.current_step == 0:
            self.populations_integrals[:, self.current_step] = self.populations[:, self.current_step]
            self.populations_derivatives[:, self.current_step] = 0
        else:
            self.populations_integrals[:, self.current_step] = self.populations_integrals[:, self.current_step - 1] + self.populations[:, self.current_step]
            self.populations_derivatives[:, self.current_step] = self.populations[:, self.current_step] - self.populations[:, self.current_step - 1]
