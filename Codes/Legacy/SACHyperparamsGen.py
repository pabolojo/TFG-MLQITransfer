import time
import os
import qutip as qt
import numpy as np
from sorcery import dict_of

from SaveLoad import save_params

if __name__ == '__main__':
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    save_best_pulses = True

    # Environment Parameters

    num_qubits = 3
    omega_min = 0
    omega_max = 1
    time_max = 5 * 2*np.pi/omega_max
    num_steps = 30
    initial_state = qt.basis(num_qubits, 0)
    target_state = qt.basis(num_qubits, num_qubits-1)
    reward_gain = 1.0
    c_ops = []
    noise_type = 'gaussian'
    noise_percentage = 0.00
    noise_seed = 0
    deltas = np.zeros(num_qubits, dtype=complex)
    deltas[1] = 0 - 0j

    # Model Hyperparameters

    num_iterations = 100000 # @param {type:"integer"}

    initial_collect_steps = 500*num_steps # @param {type:"integer"}
    collect_steps_per_iteration = 1 # @param {type:"integer"}
    replay_buffer_capacity = 500*num_steps # @param {type:"integer"}

    batch_size = 256 # @param {type:"integer"}

    critic_learning_rate = 3e-4 # @param {type:"number"}
    actor_learning_rate = 3e-4 # @param {type:"number"}
    alpha_learning_rate = 3e-4 # @param {type:"number"}
    target_update_tau = 0.005 # @param {type:"number"}
    target_update_period = 1 # @param {type:"number"}
    gamma = 0.99 # @param {type:"number"}
    reward_scale_factor = 1.0 # @param {type:"number"}
    summaries_flush_secs = 10 # @param {type:"integer"}

    actor_fc_layer_params = (512, 512, 512, 256, 256)
    critic_joint_fc_layer_params = (512, 512, 512, 256, 256)

    log_interval = 500 # @param {type:"integer"}
    summary_interval = 50 # @param {type:"integer"}
    train_checkpoint_interval = 500
    policy_checkpoint_interval = 500
    rb_checkpoint_interval = 500

    num_eval_episodes = 10 # @param {type:"integer"}
    eval_interval = 50 # @param {type:"integer"}

    use_tf_functions = True # @param {type:"boolean"}
    train_steps_per_iteration = num_steps

    root_dir = 'logs/'
    root_dir = os.path.expanduser(root_dir)

    params = dict_of(num_qubits,
                     time_max,
                     num_steps,
                     initial_state,
                     target_state,
                     reward_gain,
                     c_ops,
                     omega_min,
                     omega_max,
                     noise_type,
                     noise_percentage,
                     noise_seed,
                     deltas,
                     num_iterations,
                     initial_collect_steps,
                     collect_steps_per_iteration,
                     replay_buffer_capacity,
                     batch_size,
                     critic_learning_rate,
                     actor_learning_rate,
                     alpha_learning_rate,
                     target_update_tau,
                     target_update_period,
                     gamma,
                     reward_scale_factor,
                     summaries_flush_secs,
                     actor_fc_layer_params,
                     critic_joint_fc_layer_params,
                     log_interval,
                     summary_interval,
                     train_checkpoint_interval,
                     policy_checkpoint_interval,
                     rb_checkpoint_interval,
                     num_eval_episodes,
                     eval_interval,
                     use_tf_functions,
                     train_steps_per_iteration,
                     root_dir)

    hyper_dir = 'hyperparams/'
    hyper_dir = os.path.expanduser(hyper_dir)
    save_params(os.path.join(hyper_dir, "hyperparams_" + timestamp), params, show=True, add_time=False)