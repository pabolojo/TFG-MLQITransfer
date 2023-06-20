import qutip as qt
import numpy as np
from Noises import Noise
from CostFunctions import CostFunction
from QEnvs.QEnvDecay import QEnvDecay

class Hyperparameters:
    # Environment Parameters

    num_qubits = 3
    time_max = 20
    num_steps = 50
    end_episode_threshold = 1.
    initial_state = qt.basis(num_qubits, 0)
    target_state = qt.basis(num_qubits, num_qubits-1)
    reward_gain = 1.0
    c_ops = []
    omega_min = 0
    omega_max = 1
    noise_type = 'gaussian'
    noise_percentage = 0.
    noise_seed = None
    deltas = np.zeros(num_qubits, dtype=complex)
    decay_factors = [0., 0.21, 0.]

    noise = Noise(noise_type, percentage=noise_percentage, seed=noise_seed)

    # Cost Function

    populations_proportional_weights = [-2, -1., 1.] # [gamma, beta, alpha] gamma < beta < alpha
    populations_integral_weights = [0, -0.03, 0]
    
    costClass = CostFunction(
        proportional_weights=populations_proportional_weights,
        integral_weights=populations_integral_weights)
    
    costFunction = costClass.costFunction

    def __init__(self, root_dir):
        self.root_dir = root_dir
        # Environments
        self.env_collect_py = QEnvDecay(name = "train",
                                num_qubits=self.num_qubits,
                                time_max=self.time_max,
                                num_steps=self.num_steps,
                                cost_function=self.costFunction,
                                decay_factors=self.decay_factors,
                                initial_state=self.initial_state,
                                target_state=self.target_state,
                                reward_gain=self.reward_gain,
                                omega_min=self.omega_min,
                                omega_max=self.omega_max,
                                noise=self.noise,
                                deltas=self.deltas,
                                saveBestEpisodeDir=root_dir)

        self.env_eval_py = QEnvDecay(name = "eval",
                                num_qubits=self.num_qubits,
                                time_max=self.time_max,
                                num_steps=self.num_steps,
                                cost_function=self.costFunction,
                                decay_factors=self.decay_factors,
                                initial_state=self.initial_state,
                                target_state=self.target_state,
                                reward_gain=self.reward_gain,
                                omega_min=self.omega_min,
                                omega_max=self.omega_max,
                                noise=self.noise,
                                deltas=self.deltas,
                                saveBestEpisodeDir=root_dir)

    # Model Hyperparameters

    num_iterations = 100

    actor_fc_layers = (1024, 1024, 512, 512, 256, 256)
    critic_joint_fc_layers = (1024, 1024, 512, 512, 256, 256)
    replay_buffer_capacity = 10*num_steps#1000*num_steps
    initial_collect_steps = np.ceil(replay_buffer_capacity/2)
    eval_interval = np.ceil(num_iterations/100)
    summary_interval = np.ceil(num_iterations/100)
    collect_steps_per_iteration = 1
    batch_size = 256
    actor_learning_rate = 3e-4
    critic_learning_rate = 3e-4
    alpha_learning_rate = 3e-4
    gamma = 0.99
    num_eval_episodes = 10
    target_entropy = -(num_qubits-1)/2 # -dim(Actions)/2
    train_checkpoint_interval = None
    policy_checkpoint_interval = None
    rb_checkpoint_interval = None