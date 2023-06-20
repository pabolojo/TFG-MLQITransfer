import os

import tensorflow as tf
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import spec_utils
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.utils import common
from tf_agents.eval import metric_utils
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment


def trainSAC(
        env_collect_py,
        env_eval_py,
        root_dir,
        num_iterations,
        actor_fc_layers,
        critic_joint_fc_layers,
        replay_buffer_capacity,
        initial_collect_steps,
        eval_interval,
        summary_interval,
        critic_obs_fc_layers=None,
        critic_action_fc_layers=None,
        collect_steps_per_iteration=1,
        target_update_tau=0.005,
        target_update_period=1,
        # Params for train
        train_steps_per_iteration=1,
        batch_size=256,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=0.99,
        reward_scale_factor=1.0,
        use_tf_functions=True,
        # Params for eval
        num_eval_episodes=10,
        target_entropy=None,
        # Params for summaries and logging
        train_checkpoint_interval=None,
        policy_checkpoint_interval=None,
        rb_checkpoint_interval=None,
        summaries_flush_secs=10,
        eval_metrics_callback=None,
        use_gpu=True):

    # Set up the GPU
    if use_gpu:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)

    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')

    # Create the environments
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
            fc_layer_params=actor_fc_layers,
            continuous_projection_net=(
                tanh_normal_projection_network.TanhNormalProjectionNetwork))

    # Critic Network
    with strategy.scope():
        critic_net = critic_network.CriticNetwork(
            (observation_spec, action_spec),
            observation_fc_layer_params=critic_obs_fc_layers,
            action_fc_layer_params=critic_action_fc_layers,
            joint_fc_layer_params=critic_joint_fc_layers,
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
            td_errors_loss_fn=td_errors_loss_fn,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            train_step_counter=global_step,
            target_entropy=target_entropy)

        tf_agent.initialize()

    # Create the replay buffers
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=collect_env.batch_size,
        max_length=replay_buffer_capacity)

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
    policy_checkpointer.initialize_or_restore()
    rb_checkpointer.initialize_or_restore()

    # Initialize the Drivers
    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        collect_env,
        initial_collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=initial_collect_steps)

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        collect_env,
        collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=collect_steps_per_iteration)

    # Set function
    if use_tf_functions:
        initial_collect_driver.run = common.function(
            initial_collect_driver.run)
        collect_driver.run = common.function(collect_driver.run)
        tf_agent.train = common.function(tf_agent.train)

    # Collect initial replay data
    if replay_buffer.num_frames() == 0:
        # Collect initial replay data.
        print("Filling Replay Buffer...")
        initial_collect_driver.run()

    # Log initial metrics
    results = metric_utils.eager_compute(
        eval_metrics,
        eval_env,
        eval_policy,
        num_episodes=num_eval_episodes,
        train_step=global_step,
        summary_writer=eval_summary_writer,
        summary_prefix='Metrics',
    )
    if eval_metrics_callback is not None:
        eval_metrics_callback(results, global_step.numpy())
    metric_utils.log_metrics(eval_metrics)

    # Initialize variables
    time_step = None
    policy_state = collect_policy.get_initial_state(collect_env.batch_size)
    train_loss = None

    # Prepare replay buffer as dataset with invalid transitions filtered.
    def _filter_invalid_transition(trajectories, unused_arg1):
        return ~trajectories.is_boundary()[0]
    dataset = replay_buffer.as_dataset(
        sample_batch_size=batch_size,
        num_steps=2).unbatch().filter(
            _filter_invalid_transition).batch(batch_size).prefetch(5)
    # Dataset generates trajectories with shape [Bx2x...]
    iterator = iter(dataset)

    # Override train_step function
    def train_step():
        experience, _ = next(iterator)
        return tf_agent.train(experience)

    if use_tf_functions:
        train_step = common.function(train_step)

    global_step_val = global_step.numpy()

    print("Training...")
    # Train
    with tf.compat.v2.summary.record_if(
            lambda: tf.math.equal(global_step % summary_interval, 0)):
        while global_step_val < num_iterations:
            time_step, policy_state = collect_driver.run(
                time_step=time_step,
                policy_state=policy_state,
            )
            for _ in range(train_steps_per_iteration):
                train_loss = train_step()

            global_step_val = global_step.numpy()

            for train_metric in train_metrics:
                train_metric.tf_summaries(
                    train_step=global_step, step_metrics=train_metrics[:2])

            if eval_interval and global_step_val % eval_interval == 0:
                results = metric_utils.eager_compute(
                    eval_metrics,
                    eval_env,
                    eval_policy,
                    num_episodes=num_eval_episodes,
                    train_step=global_step,
                    summary_writer=eval_summary_writer,
                    summary_prefix='Metrics',
                )
                if eval_metrics_callback is not None:
                    eval_metrics_callback(results, global_step_val)
                metric_utils.log_metrics(eval_metrics)

            if train_checkpoint_interval and global_step_val % train_checkpoint_interval == 0:
                train_checkpointer.save(global_step=global_step_val)

            if policy_checkpoint_interval and global_step_val % policy_checkpoint_interval == 0:
                policy_checkpointer.save(global_step=global_step_val)

            if rb_checkpoint_interval and global_step_val % rb_checkpoint_interval == 0:
                rb_checkpointer.save(global_step=global_step_val)

    return train_loss
