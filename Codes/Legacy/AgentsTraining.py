from tqdm.auto import trange
from typing import Optional, List, Tuple
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import dynamic_episode_driver

def run_training(
    agent: reinforce_agent.ReinforceAgent,
    train_driver: dynamic_episode_driver.DynamicEpisodeDriver,
    replay_buffer: tf_uniform_replay_buffer.TFUniformReplayBuffer,
    eval_driver: dynamic_episode_driver.DynamicEpisodeDriver,
    eval_replay_buffer: tf_uniform_replay_buffer.TFUniformReplayBuffer,
    avg_return: tf_metrics.AverageReturnMetric,
    num_iterations: int,
    eval_interval: Optional[int] = 1,
    save_episodes: Optional[bool] = False,
    clear_buffer: Optional[bool] = False
    ) -> Tuple[List, List, List]:
    
    """
    Run the training loop.
    
    Parameters
    ----------
    agent : tf_agent.TFAgent
        Agent.
    
    train_driver : driver.Driver
        Driver for the training.
        
    replay_buffer : tf_uniform_replay_buffer.TFUniformReplayBuffer
        Replay buffer for the training.
        
    eval_driver : driver.Driver
        Driver for the evaluation.
        
    eval_replay_buffer : tf_uniform_replay_buffer.TFUniformReplayBuffer
        Replay buffer for the evaluation.
    
    avg_return : tf_metrics.AverageReturnMetric
        Metric for the average return.
    
    num_iterations : int
        Number of iterations.
        
    eval_interval : int, optional [default = 1]
        Interval for the evaluation.
        
    save_episodes : bool, optional [default = False]
        Whether to save the episodes.
        
    clear_buffer : bool, optional [default = False]
        Whether to clear the replay buffer.
    
    Returns
    -------
    return_list : List
        List of the average returns.
    
    episode_list : List
        List of the episodes.
        
    iteration_list : List
        List of the iterations.
    """

    return_list = []
    episode_list = []
    iteration_list = []
    with trange(num_iterations, dynamic_ncols=False) as t:
        for i in t:
            # t.set_description(f'episode {i}')

            #if clear_buffer:
            #    replay_buffer.clear()

            _ = train_driver.run()
            
            experience = replay_buffer.gather_all()
            _ = agent.train(experience)

            if i % eval_interval == 0 or i == num_iterations - 1:
                avg_return.reset()
                _ = eval_driver.run()

                iteration_list.append(agent.train_step_counter.numpy())
                return_list.append(avg_return.result().numpy())

                t.set_postfix({"return": return_list[-1]})

                if save_episodes:
                    pass
                    # episode_list.append(eval_replay_buffer.gather_all())

    return return_list, episode_list, iteration_list