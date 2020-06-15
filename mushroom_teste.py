import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from mushroom_rl.algorithms.value import QLearning
from mushroom_rl.core import Core
from mushroom_rl.environments import Gym
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import Parameter, ExponentialParameter

mdp = Gym(name='FrozenLake-v0', horizon=np.inf, gamma=0.96)
# Policy
epsilon = Parameter(value=0.9)
pi = EpsGreedy(epsilon=epsilon)

# Approximator
learning_rate = Parameter(value=0.81)
approximator_params = dict(input_shape=mdp.info.observation_space.shape,
                           n_actions=mdp.info.action_space.n,
                           n_estimators=50,
                           min_samples_split=5,
                           min_samples_leaf=2)

# Agent
agent = QLearning(mdp.info, pi, learning_rate)

core = Core(agent, mdp)
#train
mdp.reset
core.learn(n_steps_per_fit=100, n_episodes=1000,  render=True)