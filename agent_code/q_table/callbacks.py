import os
import pickle
import random

import numpy as np
e = 0.1

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
lastSavedEpoch = 0

def setup(self):
    #                       5 features, taken Action, reward
    self.q_table = np.zeros((3, 3, 3, 3, 3, 50, 50, 6))
    #self.q_table = np.load("./q_table.npy")
    print(self.q_table)


def act(self, game_state: dict) -> str:
    global e
    global lastSavedEpoch
    #get maximum probability
    #sometimes choose random action
    if game_state is not None and game_state["round"] % 100 == 0:
        e = e / 1.3
        if game_state["round"] != lastSavedEpoch:
            lastSavedEpoch = game_state["round"]
            #np.save("./q_table", self.q_table)
            #print("Saved Q-TABLE to file!")
    if np.random.rand(1) < e:
        #take random action
        return np.random.choice(ACTIONS)
    else:
        #take currently best action
        features = state_to_features(game_state)
        features_tuple = tuple(features)
        best_choice = ACTIONS[np.argmax(self.q_table[features_tuple])]
        """print("Taking best option:")
        print(ACTIONS)
        print(self.q_table[features_tuple])"""
        return best_choice

def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return None
    name, score, bomb_possible, own_position = game_state["self"]
    x, y = own_position
    leftOfMe = game_state["field"][x-1, y] + 1
    rightOfMe = game_state["field"][x+1, y] + 1
    onTopOfMe = game_state["field"][x, y-1] + 1
    belowMe = game_state["field"][x, y+1] + 1
    atMyPosition = game_state["field"][x, y] + 1
    allCoins = game_state["coins"]
    minDistance = 10000
    nearestCoinX = None
    nearestCoinY = None
    for coin in allCoins:
        coinX, coinY = coin
        distToCoin = (x - coinX)**2 + (y - coinY)**2
        if distToCoin < minDistance:
            minDistance = distToCoin
            nearestCoinX = coinX
            nearestCoinY = coinY
    nearestCoinRelativeX = x - nearestCoinX + 25
    nearestCoinRelativeY = y - nearestCoinY + 25
    return np.array([leftOfMe, onTopOfMe, rightOfMe, belowMe, atMyPosition, nearestCoinRelativeX, nearestCoinRelativeY])
