import time
import os
import qutip as qt
import numpy as np
from sorcery import dict_of

from SaveLoad import save_params

if __name__ == '__main__':
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    save_best_pulses = True

    N = 3
    Ωmax = 1
    n_steps = 50
    c_ops = []
    t_max = 12 * Ωmax / (2 * np.pi )
    initial_state = qt.basis(N, 0)
    target_state = qt.basis(N, N-1)
    noise_type = "gaussian"
    noise_percentage = 0.05
    noise_seed = None

    deltas = np.zeros(N, dtype=complex)
    deltas[1] = 0 - 5j

    num_iterations = 100
    reward_gain = 1.0
    summaries_flush_secs = 1
    num_eval_episodes = 10

    fc_layer_params=(200, 100, 50)
    optimizer_learning_rate=1e-3
    replay_buffer_episodes_capacity=50
    initial_collect_episodes=20
    collect_episodes_per_iteration=1
    use_tf_functions=True
    batch_size=256
    train_steps_per_iteration=1

    eval_interval=10
    summary_interval=10

    train_checkpoint_interval=50
    policy_checkpoint_interval=50
    rb_checkpoint_interval=50
    log_interval=50

    replay_buffer_capacity=replay_buffer_episodes_capacity * n_steps

    root_dir = 'logs/'
    root_dir = os.path.expanduser(root_dir)

    params = dict_of(N,
                    num_iterations,
                    fc_layer_params,
                    replay_buffer_episodes_capacity,
                    Ωmax,
                    n_steps,
                    t_max,
                    deltas,
                    initial_state,
                    target_state,
                    c_ops,
                    noise_type,
                    noise_percentage,
                    noise_seed,
                    initial_collect_episodes,
                    optimizer_learning_rate,
                    eval_interval,
                    summary_interval,
                    train_checkpoint_interval,
                    policy_checkpoint_interval,
                    rb_checkpoint_interval,
                    log_interval,
                    reward_gain,
                    summaries_flush_secs,
                    num_eval_episodes,
                    collect_episodes_per_iteration,
                    use_tf_functions,
                    batch_size,
                    train_steps_per_iteration,
                    root_dir,
                    save_best_pulses,
                    )

    hyper_dir = 'hyperparams/'
    hyper_dir = os.path.expanduser(hyper_dir)
    save_params(os.path.join(hyper_dir, "hyperparams_" + timestamp), params, show=True, add_time=False)