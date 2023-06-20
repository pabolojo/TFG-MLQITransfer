import pickle
import time
import argparse
import os
import errno
import numpy as np

from SAC import trainSAC
from SACHyperparameters import Hyperparameters

def my_import(name):
    components = name.split('/')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def parseArguments():
    """
    Parse arguments from the command line.
    
    Returns
    -------
    root_dir : str
        Root directory for saving the results.
    
    hp : Hyperparameters
        Hyperparameters for the SAC algorithm.
    
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='MLQTransfer')
    parser.add_argument('-r', '--root_dir', type=str, required=True, help='Root directory for saving the results')
    parser.add_argument('-hp', '--hyperparameters', type=str, help='File containing the hiperparameters')
    parser.add_argument('-shp', '--save_hyperparameters', action='store_true', help='Save the hyperparameters')
    args = parser.parse_args()

    if args.hyperparameters is None:
        hp = Hyperparameters(root_dir=args.root_dir)
    else:
        print("---------------------------------------")
        print("Loading hyperparameters from: ", args.hyperparameters)
        print("---------------------------------------")
        hp = my_import(args.hyperparameters).Hyperparameters(root_dir=args.root_dir)
    
    root_dir = args.root_dir
    saveHyperparameters = args.save_hyperparameters

    if saveHyperparameters:
        print("Saving hyperparameters")
        filepath = root_dir + "/hyperparameters_" + time.strftime("%Y%m%d-%H%M%S") + ".pkl"
        try:
            os.makedirs(root_dir)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(root_dir):
                pass
            else: raise
        
        with open(filepath, 'wb') as f:
            pickle.dump(hp, f)

    return root_dir, hp

if __name__ == '__main__':
    # Parse arguments
    root_dir, hp = parseArguments()

    # Create and Train the agent
    trainSAC(
        env_collect_py=hp.env_collect_py,
        env_eval_py=hp.env_eval_py,
        root_dir=root_dir,
        num_iterations=hp.num_iterations,
        actor_fc_layers=hp.actor_fc_layers,
        critic_joint_fc_layers=hp.critic_joint_fc_layers,
        replay_buffer_capacity=hp.replay_buffer_capacity,
        initial_collect_steps=hp.initial_collect_steps,
        eval_interval=hp.eval_interval,
        summary_interval=hp.summary_interval,
        collect_steps_per_iteration=hp.collect_steps_per_iteration,
        batch_size=hp.batch_size,
        actor_learning_rate=hp.actor_learning_rate,
        critic_learning_rate=hp.critic_learning_rate,
        alpha_learning_rate=hp.alpha_learning_rate,
        gamma=hp.gamma,
        num_eval_episodes=hp.num_eval_episodes,
        target_entropy=hp.target_entropy,
        train_checkpoint_interval=hp.train_checkpoint_interval,
        policy_checkpoint_interval=hp.policy_checkpoint_interval,
        rb_checkpoint_interval=hp.rb_checkpoint_interval
        )
    
    max_target_population = np.max(hp.env_eval_py.best_populations[:, :-1][-1, :])
    max_intermediate_population = np.max(hp.env_eval_py.best_populations[:, :-1][1:-1, :])

    print("---------------------------------------")
    print("Training finished")
    print("---------------------------------------")
    print("Best Reward: ", hp.env_eval_py.best_reward)
    print("Final Target Population: ", hp.env_eval_py.best_populations[:, :-1][-1, -1])
    print("Max Target Population: ", max_target_population)
    print("Max intermidiate population: ", max_intermediate_population)
    print("---------------------------------------")
