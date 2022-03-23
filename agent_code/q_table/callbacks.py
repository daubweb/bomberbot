import os
import pickle
import random

import numpy as np

#from .train import epsilon
epsilon = 0.1

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
lastSavedEpoch = 0
load_from_file = False

def setup(self):
    #                       5 features, taken Action, reward
    self.q_lookup_table = np.zeros((3, 3, 3, 3, 3, 50, 50, 8, 8))
    counter = 1
    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                for l in range(0, 3):
                    for m in range(0, 3):
                        for n in range(0, 50):
                            for o in range(0, 50):
                                for p in range(0, 8):
                                    for q in range(0, 8):
                                        self.q_lookup_table[i, j, k, l, m, n, o, p, q] = counter
                                        counter += 1
    if load_from_file:
        self.q_table = np.load("q_table_integers.npy")
    else:
        self.q_table = np.zeros([counter, 6])
    print(self.q_lookup_table)
    print(self.q_table)


def act(self, game_state: dict) -> str:
    state = state_to_features(self, game_state)
    if random.uniform(0, 1) < epsilon and hasattr(self, 'is_training'):
        action = np.random.choice(ACTIONS)
    else:
        action = ACTIONS[np.argmax(self.q_table[state])]
    return action

def state_to_features(self, game_state: dict) -> np.array:
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
    nearestCoinX = 0
    nearestCoinY = 0
    for coin in allCoins:
        coinX, coinY = coin
        distToCoin = (x - coinX)**2 + (y - coinY)**2
        if distToCoin < minDistance:
            minDistance = distToCoin
            nearestCoinX = coinX
            nearestCoinY = coinY
    nearestCoinRelativeX = x - nearestCoinX + 25
    nearestCoinRelativeY = y - nearestCoinY + 25
    bombs = game_state["bombs"]
    minDistance = 1000
    minBombX = 7
    minBombY = 7
    for position, countdown in bombs:
        bombX, bombY = position
        distance = (x-bombX)**2 + (y-bombY)**2
        if distance < minDistance:
            minBombX = bombX
            minBombY = bombY
            minDistance = distance
    if np.abs(minBombX) > 3:
        minBombX = 3 * (minBombX / np.abs(minBombX))
    if np.abs(minBombY) > 3:
        minBombY = 3 * (minBombY / np.abs(minBombY))
    minBombX = 3 + int(minBombX)
    minBombY = 3 + int(minBombY)
    """if np.abs(nearestCoinRelativeX) > 10:
        nearestCoinRelativeX = int(10 * nearestCoinRelativeX / np.abs(nearestCoinRelativeX))
    if np.abs(nearestCoinRelativeY) > 10:
        nearestCoinRelativeY = int(10 * nearestCoinRelativeY / np.abs(nearestCoinRelativeY))
    """
    state_integer = self.q_lookup_table[leftOfMe, onTopOfMe, rightOfMe, belowMe, atMyPosition, nearestCoinRelativeX, nearestCoinRelativeY, minBombX, minBombY]
    return int(state_integer)
