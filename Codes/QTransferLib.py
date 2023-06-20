import numpy as np
from qutip import Qobj, mesolve, ket2dm, Options
from typing import Optional, Tuple, List

def hamiltonian_coupling_factory(N: int, coupling: int) -> Qobj:
    """
    Creates the amiltonian with the couplingv between qubit N and N + 1 set to -1.
    
    Parameters
    ----------
    N : int
        Number of qubits.
    
    coupling : int
        Index of the qubit to be coupled to the next one.
    
    Returns
    -------
    H : Qobj
        Hamiltonian with the coupling set to -1.
    
    """
    
    if coupling < 0 or coupling > N - 1:
        raise ValueError("Interaction must be positive and smaller than N - 1")

    H = np.zeros((N, N), dtype=complex)
    
    H[coupling, coupling + 1] = -1
    
    H += H.T.conjugate() - np.diag(np.diag(H))
    return Qobj(H)

def dephase_factory(N: int, gamma: float) -> List[Qobj]:
    """
    Creates the dephasing operators for a given number of qubits and dephasing rate.
    
    Parameters
    ----------
    N : int
        Number of qubits.
        
    gamma : float
        Dephasing rate.
    
    Returns
    -------
    c_ops : list[Qobj]
        List of dephasing operators.
    
    """

    if gamma < 0:
        raise ValueError("Gamma must be positive")

    c_ops = []
    for i in range(N):
        op = np.zeros(N)
        op[i] = np.sqrt(gamma)
        c_ops.append(op)
    
    return [Qobj(np.diag(op)) for op in c_ops]

def constant_delta_factory(deltas: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Creates a constant energy difference between the qubits.
    
    Parameters
    ----------
    deltas : np.ndarray
        Array of shape (N,) with the qubit relative energies.

    n_steps : int
        Number of time steps.
    
    Returns
    -------
    deltas : np.ndarray
        Array of shape (N, n_steps) with the qubit relative time dependent energies.
    
    """

    return np.array([deltas] * n_steps, dtype=complex).T

def solve_dynamics(N: int, psi0: Qobj, time_max: float, pulses: np.ndarray, deltas: Optional[np.ndarray] = None, dephase_ops: Optional[List[Qobj]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the dynamics of an N qubit system under given pulses and operators.
    
    Parameters
    ----------
    N : int
        Number of qubits.
    
    psi0 : Qobj
        Initial state.
    
    time_max : float
        Maximum time.
        
    pulses : np.ndarray
        Array of shape (n_steps, N-1) with the pulses.

    deltas : np.ndarray, optional [default = np.zeros((N, n_steps))]
        Array of shape (N, n_steps) with the qubit relative time dependent energies.
    
    dephase_ops : list[Qobj], optional [default = None]
        List of dephasing operators.
    
    Returns
    -------
    times : np.ndarray
        Array of shape (n_steps,) with the times.

    states : np.ndarray
        Array of shape (n_steps, N, N) with the density matrices.
    
    """

    pulses = pulses.T

    if time_max < 0:
        raise ValueError("time_max must be positive")
    elif len(pulses) != N - 1:
        raise ValueError("Pulses must be a list of length N - 1")
    
    n_steps = len(pulses[0])

    if dephase_ops is None:
        dephase_ops = []
    if deltas is None:
        deltas = np.zeros(N, dtype=complex)

    for pulse in pulses:
        if len(pulse) != n_steps:
            raise ValueError("Pulses must be a list of lists of equal length")
    
    times = np.linspace(0, time_max, n_steps)
    
    if time_max == 0:
        return np.array(0), np.array(ket2dm(psi0)).reshape((1, N, N))

    H_total = []
    if len(deltas.shape) == 1:
        H_total.append(Qobj(np.diag(deltas)))
    else:
        for i, Δ in enumerate(deltas):
            H = np.zeros((N, N), dtype=complex)
            H[i, i] = 1
            H_total.append([Qobj(H), Δ])

    for i in range(N - 1):
        H_total.append([hamiltonian_coupling_factory(N, i), pulses[i]])

    opts = Options(atol=1e-11, rtol=1e-9, nsteps=int(1e6))
    # Allow for non pure evolutions
    opts.normalize_output = False

    rho_t = mesolve(H_total, psi0, times, c_ops=dephase_ops, options=opts).states
    
    return times, np.array(rho_t)

def compute_populations(N: int, psi0: Qobj, time_max: float, pulses: np.ndarray, deltas: Optional[np.ndarray] = None, dephase_ops: Optional[List[Qobj]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the populations of the N qubits under given pulses and operators.

    Parameters
    ----------
    N : int
        Number of qubits.
    
    psi0 : Qobj
        Initial state.
    
    time_max : float
        Maximum time.

    pulses : np.ndarray
        Array of shape (n_steps, N-1) with the pulses.

    deltas : np.ndarray, optional [default = np.zeros((N, n_steps))]
        Array of shape (N, n_steps) with the qubit relative time dependent energies.

    dephase_ops : list[Qobj], optional [default = None]
        List of dephasing operators.
    
    Returns
    -------
    times : np.ndarray
        Array of shape (n_steps,) with the times.
    
    populations : np.ndarray
        Array of shape (n_steps, N) with the populations.
    
    """
    
    times, rho_t = solve_dynamics(N, psi0, time_max, pulses, deltas, dephase_ops)
    if dephase_ops is None:
        n_steps = len(pulses)
        populations = (np.abs(rho_t) ** 2).reshape((n_steps, N))
    else:
        populations = np.diagonal(rho_t, axis1=1, axis2=2).real
    return times, populations