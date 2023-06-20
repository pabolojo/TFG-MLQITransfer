import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from tf_agents.trajectories import trajectory

LINE_STYLES = ["-", "--", ":", "-."]

def plot_pulses(times: np.ndarray, pulses: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot the pulses.

    Parameters
    ----------
    times : np.ndarray
        Array of times.
    
    pulses : np.ndarray
        Array of shape (n_steps, N-1) with the pulses.

    ax: plt.Axes (optional) default: None
        Axes to plot on.
    
    Returns
    -------
    ax : plt.Axes
        Plot.
    
    """

    if ax is None:
        fig, ax = plt.subplots()
    
    for i, pop in enumerate(pulses.T):
        #if i == 0 or i == len(pulses.T) - 1:
        ax.plot(times, pop, label=r'$\Omega_{%s}$' % (str(i + 1) + str(i+2)), linestyle=LINE_STYLES[i % len(LINE_STYLES)])
        #else:
        #    ax.plot(times, pop, label=r'$\Omega_{s}$', linestyle=LINE_STYLES[i % len(LINE_STYLES)])

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\Omega$')
    ax.legend()

    return ax

def plot_populations(times: np.ndarray, populations: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot the populations.

    Parameters
    ----------
    times : np.ndarray
        Array of times.
    
    populations : np.ndarray
        Array of shape (n_steps, N) with the populations.
    
    Returns
    -------
    ax : plt.Axes
        Plot.
    
    """
    
    if ax is None:
        fig, ax = plt.subplots()
    
    for i, pop in enumerate(populations.T):
        ax.plot(times, pop, label=r'$|%i\rangle$' % (i + 1), linestyle=LINE_STYLES[i % len(LINE_STYLES)])

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\rho$')
    ax.legend()

    return ax

def plot_training_returns(returns: List, threshold: Optional[float] = None, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot the training returns.

    Parameters
    ----------
    returns : List
        List of returns.

    threshold : float (optional) default: None
        Threshold to plot.

    ax: plt.Axes (optional) default: None
        Axes to plot on.
    
    Returns
    -------
    ax : plt.Axes
        Plot.
    
    """
    
    if ax is None:
        fig, ax = plt.subplots()
    
    if threshold is not None:
        ax.plot([0, len(returns)], [threshold, threshold], label=str(threshold), linestyle="--", color="black")
    
    iterations = np.linspace(0, len(returns), len(returns))
    ax.plot(iterations, returns)
    ax.set_ylabel("Return")
    ax.set_xlabel("Iteration")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()

    return ax

def plot_episode(times: np.ndarray, episode: trajectory.Trajectory, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot the episode.

    Parameters
    ----------
    times : np.ndarray
        Array of times.

    episode : trajectory.Trajectory
        Episode to plot.
    
    ax: plt.Axes (optional) default: None
        Axes to plot on.

    Returns
    -------
    ax : plt.Axes
        Plot.
    
    """
    
    pulses = episode.action.numpy()[0, :, :].T

    if ax is None:
        fig, ax = plt.subplots(2, 1, sharex=True)

    ax[0].set_ylabel("Pulses")
    ax[1].set_ylabel("Populations")
    ax[1].set_xlabel("Time")

    for i, pulse in enumerate(pulses):
        ax[0].plot(times, pulse, label=r"$\Omega_{%s}$" % str(i + 1))

    for i in range(len(pulses) + 1):
        ax[1].plot(times, episode.observation.numpy()[0, :, i], label=r'$\rho_{%s}$' % (str(i + 1) + str(i + 1)))
        

    ax[0].legend()
    ax[1].legend()

    return ax

def plot_best_episode(episode: np.ndarray, ax: Optional[plt.Axes] = None):
    """
    Plot the best episode.

    Parameters
    ----------
    episode : np.ndarray
        Episode to plot.

    ax: plt.Axes (optional) default: None
        Axes to plot on.
    
    Returns
    -------
    ax : plt.Axes
        Plot.
    
    """

    times = episode[1][:-1]
    pulses = episode[2]
    populations = episode[3][:, :-1]

    if ax is None:
        if len(episode) == 5:
            fig, ax = plt.subplots(3, 1, sharex=True)
        else:
            fig, ax = plt.subplots(2, 1, sharex=True)
    
    if len(episode) == 5:
        cost = episode[4][:-1]
        ax[2].set_ylabel("Cost")
        ax[2].plot(times, cost)
    
    ax[0].set_ylabel("Pulses")
    ax[1].set_ylabel("Populations")
    ax[1].set_xlabel("Time")

    for i, pulse in enumerate(pulses):
        ax[0].plot(times, pulse, label=r"$\Omega_{%s}$" % str(i + 1))

    for i in range(len(pulses) + 1):
        ax[1].plot(times, populations[i, :], label=r'$\rho_{%s}$' % (str(i + 1) + str(i + 1)))        

    ax[0].legend()
    ax[1].legend()

    return ax

def plot_rewards(labels: np.ndarray, times: np.ndarray, rewards: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot the reward.

    Parameters
    ----------
    labels : np.ndarray
        Array of labels.
    
    times : np.ndarray
        Array of times.

    rewards : np.ndarray
        Array of rewards.

    ax: plt.Axes (optional) default: None
        Axes to plot on.
    
    Returns
    -------
    ax : plt.Axes
        Plot.
    
    """

    if ax is None:
        fig, ax = plt.subplots()

    for i, rew in enumerate(rewards):
        ax.plot(times, rew, label=labels[i], linestyle=LINE_STYLES[i % len(LINE_STYLES)])
    
    ax.set_xlabel("t")
    ax.set_ylabel("Reward")
    ax.legend()
    
    return ax

def plot_noise_effect(noise_percentages: np.ndarray, noise_results: np.ndarray, ax: Optional[plt.Axes] = None, show_standard_dev: Optional[bool] = True) -> plt.Axes:
    """
    Plot the noise effect.

    Parameters
    ----------
    noise_percentages : np.ndarray
        Array of noise percentages.

    noise_results : np.ndarray
        Array with mean and std of the fidelity for each noise percentage.

    ax: plt.Axes (optional) default: None
        Axes to plot on.

    show_standard_dev: bool (optional) default: True
        Show standard deviation.
    
    Returns
    -------
    ax : plt.Axes
        Plot.
    
    """

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(noise_percentages*100, noise_results[:, 0])

    upper = noise_results[:, 0] + noise_results[:, 1]
    if np.any(upper > 1.0):
        upper[upper > 1.0] = 1.0
    
    if show_standard_dev:
        ax.fill_between(noise_percentages*100, noise_results[:, 0] - noise_results[:, 1], upper, alpha=0.1)

    ax.set_xlabel("Noise percentage")
    ax.set_ylabel("Fidelity")

    return ax

def get_min(array):
    min_array = []
    for i in range(len(array)):
        min_array.append(np.min(array[i]))
    return np.min(min_array)

def get_max(array):
    max_array = []
    for i in range(len(array)):
        max_array.append(np.max(array[i]))
    return np.max(max_array)

def get_axis_lims(data: np.ndarray, margin_percentage: List[float] = []) -> Tuple[float, float]:
    """
    Get the axis limits.

    Parameters
    ----------
    data : np.ndarray
        Axis data.
    
    margin_percentage : List[float] (optional) default: None
        Margin percentage to add to the axis limits.

    Returns
    -------
    min : float
        Minimum value.

    max : float
        Maximum value.
    
    """

    min = get_min(data)
    max = get_max(data)
    if len(margin_percentage) == 0:
        margin_percentage = [0.05, 0.05]
    margins = [margin_percentage[0] * (max - min), margin_percentage[1] * (max - min)]

    return min - margins[0], max + margins[1]