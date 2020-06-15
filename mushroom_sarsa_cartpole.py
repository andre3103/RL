import numpy as np
from joblib import Parallel, delayed

from mushroom_rl.algorithms.value import TrueOnlineSARSALambda
from mushroom_rl.core import Core
from mushroom_rl.environments import Gym
from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import compute_J, episodes_length
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.utils.spaces import Box

"""
This script aims to replicate the experiments on the Mountain Car MDP as
presented in:
"True Online TD(lambda)". Seijen H. V. et al.. 2014.

"""


def experiment(alpha):
    np.random.seed()

    # MDP
    mdp = Gym(name='CartPole-v0', horizon=np.inf, gamma=1.)

    # Policy
    epsilon = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    n_tilings = 10
    tilings = Tiles.generate(n_tilings, [10, 10, 10, 10],
                             mdp.info.observation_space.low,
                             mdp.info.observation_space.high)
    features = Features(tilings=tilings)

    learning_rate = Parameter(alpha / n_tilings)

    approximator_params = dict(input_shape=(features.size,),
                               output_shape=(mdp.info.action_space.n,),
                               n_actions=mdp.info.action_space.n)
    algorithm_params = {'learning_rate': learning_rate,
                        'lambda_coeff': .9}

    agent = TrueOnlineSARSALambda(mdp.info, pi,
                                  approximator_params=approximator_params,
                                  features=features, **algorithm_params)

    #shape = agent.approximator.Q
    #print(agent.Q)

    # Algorithm
    core = Core(agent, mdp)

    # Train
    core.learn(n_episodes=100, n_steps_per_fit=1, render=True)
    dataset = core.evaluate(n_episodes=10, render=False)
    #print(dataset)
    print(episodes_length(dataset))

    return np.mean(compute_J(dataset, .96))


if __name__ == '__main__':
    n_experiment = 1

    alpha = .1
    Js = Parallel(
        n_jobs=-1)(delayed(experiment)(alpha) for _ in range(n_experiment))

    print((np.mean(Js)))