import numpy as np
import warnings
from typing import Optional, List

def CTAP_pulses(N: int, time_max: float, n_steps: int, tau: float, sigma: float, omega0: float, omegaS: Optional[float] = 0.) -> np.ndarray:
    """
    CTAP pulses in a 1-N trajectory.

    Parameters
    ----------
    N : int
        Number of qubits.
    
    time_max : float
        Maximum time of the pulses.
    
    n_steps : int
        Number of steps in the time discretization.

    tau : float
        Pulses shift parameter.

    sigma : float
        Width of the pulses.

    omega0 : float
        Pulses amplitude.

    omegaS : float, optional [default = 0]
        Straddling pulses amplitude.

    Returns
    -------
    pulses : np.ndarray
        Array of shape (n_steps, N-1) with the pulses.
    
    """

    if time_max <= 0:
        return np.zeros((n_steps, N-1))
    
    if N > 3 and omegaS == 0:
        warnings.warn("The number of qubits is greater than 3 and omegaS is 0.")

    times = np.linspace(0, time_max, n_steps)

    n = (N - 1)/2

    timesp = times - times[-1]/2

    omega12 = np.sqrt(n)*_gaussian(timesp, tau, sigma/np.sqrt(2), omega0)
    omega23 = np.sqrt(n)*_gaussian(timesp, -tau, sigma/np.sqrt(2), omega0)
    omegasp = _gaussian(timesp, 0, sigma, omegaS)

    return np.array([omega12] + [omegasp]*(N-3) + [omega23]).T

def _gaussian(times, tau, sigma, amplitude):
    return amplitude * np.exp(-(times - tau)**2 / (2*sigma**2))

def STA_pulses(N: int, time_max: float, n_steps: int, alpha_0: float, omegaS: Optional[float] = 0., sigma: Optional[float] = 1., chi_coefficients: Optional[List] = None) -> np.ndarray:
    """
    STA pulses based on IE and a Gutman 1-N trajectory.

    Parameters
    ----------
    N : int
        Number of qubits.
    
    time_max : float
        Maximum time of the pulses.
    
    n_steps : int
        Number of steps in the time discretization.

    alpha_0 : float
        Pulse maximum amplitude.

    omegaS : float, optional [default = 0]
        Straddling pulse amplitude.
    
    sigma : float, optional [default = 1]
        Width of the straddling pulses.

    chi_coefficients : list, optional [default = [1/2, -1/3, 1/24]]
        Coefficients that parametrize the Chi series.

    Returns
    -------
    pulses : np.ndarray
        Array of shape (n_steps, N-1) with the pulses.
    
    """

    if time_max == 0:
        return np.zeros((n_steps, N-1))

    if N > 3 and omegaS == 0:
        warnings.warn("The number of qubits is greater than 3 and omegaS is 0.")

    if chi_coefficients is None:
        chi_coefficients = [1/2, -1/3, 1/24]
    
    times = np.linspace(0, time_max, n_steps)
    chi = _auxiliar_chi(times, chi_coefficients)
    d_chi = np.gradient(chi, times)
    
    eta = _auxiliar_eta(d_chi, alpha_0)
    d_eta = np.gradient(eta, times)
    
    Omega_12 = np.cos(chi) * d_eta + np.sin(chi) * d_chi / (np.tan(eta) + 1e-15)
    Omega_23 = -np.sin(chi) * d_eta + np.cos(chi) * d_chi / (np.tan(eta) + 1e-15)

    timesp = times - times[-1]/2

    omegasp = _gaussian(timesp, 0, sigma, omegaS)
    n = (N - 1)/2
    
    return np.array([np.sqrt(n)*Omega_12] + [10*omegasp]*(N-3) + [np.sqrt(n)*Omega_23]).T

def _auxiliar_chi(times, chi_coefficients):
    tf = times[-1]
    t_prime = times / tf
    
    result = 0
    for i, coeff in enumerate(chi_coefficients):
        if i == 0:
            result += coeff * np.pi * t_prime
        else:
            result += coeff * np.sin(2 * i * np.pi * t_prime)
    
    return result

def _auxiliar_eta(d_chi, alpha_0):
    return np.arctan(d_chi / alpha_0)