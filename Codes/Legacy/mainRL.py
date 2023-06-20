import argparse
import tensorflow as tf
import os
import time
import pickle

from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.policies import random_tf_policy
from tf_agents.utils import common
from tf_agents.eval import metric_utils
from tf_agents.drivers import dynamic_episode_driver
from absl import logging

import numpy as np

from SaveLoad import load_params
from QTransferEnv import QTransferEnv
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

    # Compute some parameters

    noise = Noise(noise_type, percentage=noise_percentage, seed=noise_seed)
    replay_buffer_capacity=replay_buffer_episodes_capacity * n_steps

    # Configure GPU
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # Create the environments
    env_train_py = QTransferEnv(num_qubits=N,
                            t_max=t_max,
                            n_steps=n_steps,
                            initial_state=initial_state,
                            target_state=target_state,
                            reward_gain=reward_gain,
                            c_ops=c_ops,
                            omega_min=0,
                            omega_max=Ωmax,
                            noise=noise,
                            deltas=deltas)

    env_eval_py = QTransferEnv(num_qubits=N,
                            t_max=t_max,
                            n_steps=n_steps,
                            initial_state=initial_state,
                            target_state=target_state,
                            reward_gain=reward_gain,
                            c_ops=c_ops,
                            omega_min=0,
                            omega_max=Ωmax,
                            noise=noise,
                            deltas=deltas)

    # Create summaries
    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        train_dir, flush_millis=summaries_flush_secs * 1000)
    train_summary_writer.set_as_default()

    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        eval_dir, flush_millis=summaries_flush_secs * 1000)

    # Initialize global step
    global_step = tf.compat.v1.train.get_or_create_global_step()

    # Create and Train the Agent
    with tf.compat.v2.summary.record_if(lambda: tf.math.equal(global_step % summary_interval, 0)):
        tf_env = tf_py_environment.TFPyEnvironment(env_train_py)
        eval_tf_env = tf_py_environment.TFPyEnvironment(env_eval_py)

        time_step_spec = tf_env.time_step_spec()
        observation_spec = time_step_spec.observation
        action_spec = tf_env.action_spec()

        # Actor Network
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec,
            action_spec,
            fc_layer_params=fc_layer_params,
        )

        # Agent
        tf_agent = reinforce_agent.ReinforceAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_net,
            optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=optimizer_learning_rate),
            normalize_returns=True,
            train_step_counter=global_step,
        )

        tf_agent.initialize()

        # Replay Buffers
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=tf_agent.collect_data_spec,
            batch_size=tf_env.batch_size,
            max_length=replay_buffer_capacity,
        )
        replay_observer = [replay_buffer.add_batch]

        eval_replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=tf_agent.collect_data_spec,
            batch_size=eval_tf_env.batch_size,
            max_length=n_steps + 1,
        )
        eval_replay_observer = [eval_replay_buffer.add_batch]

        # Train Metrics
        train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(
                buffer_size=num_eval_episodes, batch_size=tf_env.batch_size),
            tf_metrics.AverageEpisodeLengthMetric(
                buffer_size=num_eval_episodes, batch_size=tf_env.batch_size),
        ]

        # Eval Metrics
        eval_metrics = [
            tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
        ]

        # Policies
        eval_policy = tf_agent.policy
        initial_collect_policy = random_tf_policy.RandomTFPolicy(time_step_spec, action_spec)
        collect_policy = tf_agent.collect_policy

        # Checkpoints
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

        # Drivers
        initial_collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            tf_env,
            initial_collect_policy,
            observers=replay_observer + train_metrics,
            num_episodes=initial_collect_episodes)

        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            tf_env,
            collect_policy,
            observers=replay_observer + train_metrics,
            num_episodes=collect_episodes_per_iteration)

        eval_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            eval_tf_env,
            eval_policy,
            eval_replay_observer + eval_metrics,
            num_episodes=1)

        if use_tf_functions:
            initial_collect_driver.run = common.function(initial_collect_driver.run)
            collect_driver.run = common.function(collect_driver.run)
            eval_driver.run = common.function(eval_driver.run)
            tf_agent.train = common.function(tf_agent.train)

        if replay_buffer.num_frames() == 0:
            # Collect initial replay data.
            initial_collect_driver.run()

        # Save initial eval metrics
        eval_driver.run()
        metric_utils.log_metrics(eval_metrics)

        time_step = None
        policy_state = collect_policy.get_initial_state(tf_env.batch_size)

        timed_at_step = global_step.numpy()
        time_acc = 0

        def train_step():
            experience = replay_buffer.gather_all()
            return tf_agent.train(experience)

        if use_tf_functions:
            train_step = common.function(train_step)

        global_step_val = global_step.numpy()
        
        best_population = -1
        best_episode = None

        # Training Loop
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

            # Logging each log_interval
            if global_step_val % log_interval == 0:
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

            # Evaluate each eval_interval
            if global_step_val % eval_interval == 0:
                eval_metrics[0].reset()
                eval_driver.run()
                if best_population < eval_metrics[0].result().numpy():
                    best_population = eval_metrics[0].result().numpy()
                    best_episode = eval_replay_buffer.gather_all()
                metric_utils.log_metrics(eval_metrics)
                with eval_summary_writer.as_default():
                    tf.compat.v2.summary.scalar(
                        name='best_population', data=best_population, step=global_step)

            # Train checkpoint save each train_checkpoint_interval
            if global_step_val % train_checkpoint_interval == 0:
                train_checkpointer.save(global_step=global_step_val)

            # Policy checkpoint save each policy_checkpoint_interval
            if global_step_val % policy_checkpoint_interval == 0:
                    policy_checkpointer.save(global_step=global_step_val)

            # Replay Buffer save each rb_checkpoint_interval
            if global_step_val % rb_checkpoint_interval == 0:
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
    print("Best population: ", best_episode.observation.numpy()[0, -1, N - 1])
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