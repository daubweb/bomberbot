import gym
import matplotlib.pyplot as plt
env = gym.make("Taxi-v3").env
observation = env.reset()

alpha = 0.1
gamma = 0.6
epsilon = 0.1

import random
from IPython.display import clear_output
import numpy as np
q_table = np.zeros([env.observation_space.n, env.action_space.n])

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

averageRewards = []
for i in range(0, 100000):
    state = env.reset()
    epochs, penalties, reward, = 0, 0, 0
    done = False


    #train agent for this episode
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1-alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1
        
        state = next_state
        epochs += 1


    if i > 0:
        state = env.reset()
        totalReward = 0
        numberSteps = 0
        done = False
        while not done and numberSteps < 100:
            action = np.argmax(q_table[state])
            next_state, reward, done, info = env.step(action)
            #if reward == -10:
            totalReward += reward
            #print("Penalty!")
            numberSteps += 1
            state = next_state
        averageReward = totalReward / numberSteps
        averageRewards.append(averageReward)
        
        clear_output(wait=True)
        #print(f"Episode: {i}")
    if i % 99999 == 0 and i > 0:
        plt.plot(moving_average(averageRewards, 1000))
        plt.show()
