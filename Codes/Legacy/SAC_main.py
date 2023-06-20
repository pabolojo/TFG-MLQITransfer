import argparse
import os
import time
import pickle

import tensorflow as tf
import qutip as qt
import numpy as np
import time

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.networks import actor_distribution_network
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.environments import tf_py_environment
from tf_agents.drivers import dynamic_step_driver
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.utils import common
from tf_agents.eval import metric_utils
from absl import logging

from SaveLoad import load_params
from MLQTransfer.Codes.QEnvs.QEnv import QTransferEnv
from Noises import Noise

def parseArguments(global_vars: dict):
    """
    Parse arguments from the command line.
    
    Parameters
    ----------
    global_vars : dict
        Dictionary of global variables.
    
    Returns
    -------
    params : dict
        Dictionary of parameters.
    
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='MLQTransfer')
    parser.add_argument('-lhf', '--load_hyperparameters_file', type=str, required=True, help='File containing the hiperparameters')
    parser.add_argument('-ts', '--timestamp', type=str, required=True, help='Timestamp of the run')
    args = parser.parse_args()
    params = load_params(path=args.load_hyperparameters_file, show=True, global_vars=global_vars)
    global_vars['timestamp'] = args.timestamp
    global_vars['train_dir'] = os.path.join(root_dir, 'train' + timestamp)
    global_vars['eval_dir'] = os.path.join(root_dir, 'eval' + timestamp)
    return params

def createAndTrainAgent():
    """
    Create and train the agent.

    Returns
    -------
    Best Evaluation Episode.

    """

    # Configure GPU
    use_gpu = True
    
    if use_gpu:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)
    
    # Compute some parameters
    noise = Noise(noise_type, percentage=noise_percentage, seed=noise_seed)

    # Create the environments
    env_collect_py = QTransferEnv(num_qubits=num_qubits,
                            t_max=time_max,
                            n_steps=num_steps,
                            initial_state=initial_state,
                            target_state=target_state,
                            reward_gain=reward_gain,
                            c_ops=c_ops,
                            omega_min=omega_min,
                            omega_max=omega_max,
                            noise=noise,
                            deltas=deltas)

    env_eval_py = QTransferEnv(num_qubits=num_qubits,
                            t_max=time_max,
                            n_steps=num_steps,
                            initial_state=initial_state,
                            target_state=target_state,
                            reward_gain=reward_gain,
                            c_ops=c_ops,
                            omega_min=omega_min,
                            omega_max=omega_max,
                            noise=noise,
                            deltas=deltas)

    collect_env = tf_py_environment.TFPyEnvironment(env_collect_py)
    eval_env = tf_py_environment.TFPyEnvironment(env_eval_py)

    # Create summaries
    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        train_dir, flush_millis=summaries_flush_secs * 1000)
    train_summary_writer.set_as_default()

    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        eval_dir, flush_millis=summaries_flush_secs * 1000)

    # Collect Environment Specs
    observation_spec, action_spec, time_step_spec = (
      spec_utils.get_tensor_specs(collect_env))
    
    # Actor Network
    with strategy.scope():
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec,
            action_spec,
            fc_layer_params=actor_fc_layer_params,
            continuous_projection_net=(
                tanh_normal_projection_network.TanhNormalProjectionNetwork))
        
    # Critic Network
    with strategy.scope():
        critic_net = critic_network.CriticNetwork(
                (observation_spec, action_spec),
                observation_fc_layer_params=None,
                action_fc_layer_params=None,
                joint_fc_layer_params=critic_joint_fc_layer_params,
                kernel_initializer='glorot_uniform',
                last_kernel_initializer='glorot_uniform')

    # Create the agent
    with strategy.scope():
        global_step = tf.compat.v1.train.get_or_create_global_step()

        tf_agent = sac_agent.SacAgent(
                time_step_spec,
                action_spec,
                actor_network=actor_net,
                critic_network=critic_net,
                actor_optimizer=tf.keras.optimizers.Adam(
                    learning_rate=actor_learning_rate),
                critic_optimizer=tf.keras.optimizers.Adam(
                    learning_rate=critic_learning_rate),
                alpha_optimizer=tf.keras.optimizers.Adam(
                    learning_rate=alpha_learning_rate),
                target_update_tau=target_update_tau,
                target_update_period=target_update_period,
                td_errors_loss_fn=tf.math.squared_difference,
                gamma=gamma,
                reward_scale_factor=reward_scale_factor,
                train_step_counter=global_step)

        tf_agent.initialize()
    
    # Create the replay buffers
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=collect_env.batch_size,
        max_length=replay_buffer_capacity)

    eval_replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=eval_env.batch_size,
        max_length=num_steps + 1)

    # Train Metrics
    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(
            buffer_size=num_eval_episodes, batch_size=collect_env.batch_size),
        tf_metrics.AverageEpisodeLengthMetric(
            buffer_size=num_eval_episodes, batch_size=collect_env.batch_size),
    ]

    # Eval Metrics
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
    ]

    # Policies
    eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
    initial_collect_policy = random_tf_policy.RandomTFPolicy(
        time_step_spec, action_spec)
    collect_policy = tf_agent.collect_policy

    # Checkpointers
    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        policy=eval_policy,
        global_step=global_step)
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=replay_buffer)

    train_checkpointer.initialize_or_restore()
    rb_checkpointer.initialize_or_restore()

    # Initialize the Drivers
    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        collect_env,
        initial_collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=initial_collect_steps)

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        collect_env,
        collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=collect_steps_per_iteration)

    eval_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        eval_env,
        eval_policy,
        observers=[eval_replay_buffer.add_batch] + eval_metrics,
        num_episodes=num_eval_episodes)

    # Set function
    if use_tf_functions:
        initial_collect_driver.run = common.function(initial_collect_driver.run)
        collect_driver.run = common.function(collect_driver.run)
        eval_driver.run = common.function(eval_driver.run)
        tf_agent.train = common.function(tf_agent.train)

    # Collect initial replay data
    if replay_buffer.num_frames() == 0:
        # Collect initial replay data.
        initial_collect_driver.run()

    # Log initial metrics
    eval_driver.run()
    metric_utils.log_metrics(eval_metrics)

    # Initialize times
    time_step = None
    policy_state = collect_policy.get_initial_state(collect_env.batch_size)

    timed_at_step = global_step.numpy()
    time_acc = 0

    # Prepare replay buffer as dataset with invalid transitions filtered.
    def _filter_invalid_transition(trajectories, unused_arg1):
        return ~trajectories.is_boundary()[0]
    dataset = replay_buffer.as_dataset(
        sample_batch_size=batch_size,
        num_steps=2).unbatch().filter(
            _filter_invalid_transition).batch(batch_size).prefetch(num_steps)
    # Dataset generates trajectories with shape [Bx2x...]
    iterator = iter(dataset)

    # Override train_step function
    def train_step():
        experience, _ = next(iterator)
        return tf_agent.train(experience)

    if use_tf_functions:
        train_step = common.function(train_step)

    global_step_val = global_step.numpy()
    best_population = -1
    best_episode = None

    # Train
    with tf.compat.v2.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):
        while global_step_val < num_iterations:
            start_time = time.time()
            time_step, policy_state = collect_driver.run(
                time_step=time_step,
                policy_state=policy_state,
            )
            for _ in range(train_steps_per_iteration):
                train_loss = train_step()
            time_acc += time.time() - start_time

            global_step_val = global_step.numpy()

            if log_interval and global_step_val % log_interval == 0:
                logging.info('step = %d, loss = %f', global_step_val,
                            train_loss.loss)
                steps_per_sec = (global_step_val - timed_at_step) / time_acc
                logging.info('%.3f steps/sec', steps_per_sec)
                tf.compat.v2.summary.scalar(
                    name='global_steps_per_sec', data=steps_per_sec, step=global_step)
                timed_at_step = global_step_val
                time_acc = 0

            for train_metric in train_metrics:
                train_metric.tf_summaries(
                    train_step=global_step, step_metrics=train_metrics[:2])

            if eval_interval and global_step_val % eval_interval == 0:
                eval_metrics[0].reset()
                eval_driver.run()
                if best_population < eval_metrics[0].result().numpy():
                    best_population = eval_metrics[0].result().numpy()
                    best_episode = eval_replay_buffer.gather_all()
                with eval_summary_writer.as_default():
                    metric_utils.log_metrics(eval_metrics)
                    tf.compat.v2.summary.scalar(
                        name='best_population', data=best_population, step=global_step)
                # Stop training if the population is perfect
                if eval_metrics[0].result().numpy() > 0.999:
                    return best_episode

            if train_checkpoint_interval and global_step_val % train_checkpoint_interval == 0:
                train_checkpointer.save(global_step=global_step_val)

            if policy_checkpoint_interval and global_step_val % policy_checkpoint_interval == 0:
                policy_checkpointer.save(global_step=global_step_val)

            if rb_checkpoint_interval and global_step_val % rb_checkpoint_interval == 0:
                rb_checkpointer.save(global_step=global_step_val)

    return best_episode

def saveResults(best_episode):
    """
    Save the results of the training
    
    Parameters
    ----------
    best_episode : tf_agents.trajectories.trajectory.Trajectory
        The best episode of the training
    
    """
    print("Best population: ", best_episode.observation.numpy()[0, -1, num_qubits - 1])
    # Save the best episode
    with open(os.path.join(eval_dir, "best_episode_" + timestamp + ".pkl"), 'wb') as f:
        pickle.dump(best_episode, f)
    np.save(os.path.join(eval_dir, "best_pulses_" + timestamp + ".npy"), best_episode.action.numpy()[0, :, :].T)

if __name__ == '__main__':
    # Parse arguments
    parseArguments(global_vars=globals())
    # Create and Train the agent
    best_episode = createAndTrainAgent()
    # Save results
    saveResults(best_episode)