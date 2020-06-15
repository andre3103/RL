from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.environments import GridWorld
from mushroom_rl.algorithms.value import QLearning
from mushroom_rl.core.core import Core
import numpy as np
from mushroom_rl.environments import Gym

mdp = Gym(name='FrozenLake-v0', horizon=np.inf, gamma=1.)

epsilon = Parameter(value=1.)
policy = EpsGreedy(epsilon=epsilon)

learning_rate = Parameter(value=.6)
agent = QLearning(mdp.info, policy, learning_rate)

core = Core(agent, mdp)
core.learn(n_steps=10000, n_steps_per_fit=1)

shape = agent.approximator.shape
q = np.zeros(shape)
print(q)
for i in range(shape[0]):
    for j in range(shape[1]):
        state = np.array([i])
        action = np.array([j])
        q[i, j] = agent.approximator.predict(state, action)
print(q)
