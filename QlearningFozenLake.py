#impotando as bibliotecas, numpy para armazenar a qtable, e pickle para salvar a qtable em formato pkl
import gym
import numpy as np
import time, pickle, os

#inicializa o ambiente frozenlake
env = gym.make('FrozenLake-v0')

#inicializa as variáveis
##abordagem epsilon-greedy 
epsilon = 0.9
total_episodes = 10 #10000
max_steps = 100

lr_rate = 0.81
##fator de desconto
gamma = 0.96


print(env.observation_space.n, env.action_space.n)

#inicializa a q-table com tamanho de 16x4, preenchida por zeros
##env.observation_space.n = numero total de stados
##env.action_space.n = numero total de ações
Q = np.zeros((env.observation_space.n, env.action_space.n))

#escolhe a ação a ser tomada de acordod com a abordagem epsilon -greedy    
def choose_action(state):
    action=0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action

#função que vai atualizar a q-table
def learn(state, state2, reward, action):
    predict = Q[state, action]
    target = reward + gamma * np.max(Q[state2, :])
    Q[state, action] = Q[state, action] + lr_rate * (target - predict)

# Inicializa o episódio
for episode in range(total_episodes):
    #inicializa o estado com 0
    state = env.reset()
    #inicializa time steps(time steps)
    t = 0
    print('inicializa time step:', t)
    
    while t < max_steps:
        print('time step (passo):', t) 
        #renderiza ambiente
        env.render()

        #action recebe a ação escolhida na função choose_action
        action = choose_action(state) 

        #dados gerados pelo ambiente na execução do passo(step)
        state2, reward, done, info = env.step(action) 

        #a q-table ṕe atualizada de acordo com os resultados do ambiente
        learn(state, state2, reward, action)
        print('estado: ', state, '. estado 2:',state2)
        state = state2

        t += 1
       
        if done:
            break

        time.sleep(0.1)

print(Q)

with open("frozenLake_qTable.pkl", 'wb') as f:
    pickle.dump(Q, f)
