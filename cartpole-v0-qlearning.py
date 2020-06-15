import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import math
import gym

#inicializa o ambiente cartpole
env = gym.make('CartPole-v0')
env.reset()

temp_divisor=25
num_states = (1,1,6,3)

def discretize(obs):
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
    ratios = [(obs[i] - lower_bounds[i]) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((num_states[i] - 1) * ratios[i])) for i in range(len(obs))]
    new_obs = [min(num_states[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
    return tuple(new_obs)
    
def update_q(state_adj, action, reward, state2_adj, alpha, gamma):
    Q[state_adj][action] += alpha * (reward + gamma * np.max(Q[state2_adj]) - Q[state_adj][action])
    return Q

def choose_action(state, epsilon):
    return env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(Q[state])
  
def get_epsilon(t, min_epsilon):
    return max(min_epsilon, min(1, 1.0 - math.log10((t + 1) / temp_divisor)))

def get_alpha(t, min_alpha):
    return max(min_alpha, min(1.0, 1.0 - math.log10((t + 1) / temp_divisor)))
  
def QLearning(env, learning, discount, exploration, min_eps, episodes):        
    reward_list = deque(maxlen=100)
    reward_history = []

    for i in range(episodes):
        done = False
        cummulative_reward = 0
        
        alpha = get_alpha(i, learning)        
        epsilon = get_epsilon(i, exploration)
        current_state = discretize(env.reset())

        while (not done):
            
            action = choose_action(current_state, epsilon)
            temp_state, reward, done, info = env.step(action) 
            #env.render()
            new_state = discretize(temp_state)

            Q = update_q(current_state, action, reward, new_state, alpha, discount)          
            cummulative_reward += reward
            current_state = new_state
        reward_list.append(cummulative_reward)
        reward_history.append(cummulative_reward)
        
        if (i+1) % 100 == 0:
            print('Episode {} Average Reward: {}'.format(i+1, np.mean(reward_list)))
            #print(Q)

        if np.mean(reward_list) >= 200:
            print("Solution achieved at episode {}". format(i+1 - 100))
            return reward_history

Q = np.zeros(num_states + (env.action_space.n,))
#print(Q)
reward_history = QLearning(env, 0.1, 1.0, 0.1, 0, 5000)