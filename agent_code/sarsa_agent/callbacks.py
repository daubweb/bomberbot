import os
import pickle
import random

import numpy as np

#from .train import epsilon
epsilon = 0.1

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
lastSavedEpoch = 0
load_from_file = True
FILENAME = "q_table_walls"

def trim_num(num, max):
    if int(num) > max:
        return max
    elif int(num) < -max:
        return -max
    return int(num)

def setup(self):
    #                       5 features, taken Action, reward
    self.q_lookup_table = np.zeros((4, 4, 4, 4, 4, 50, 50 ))
    counter = 1
    for i in range(0, 4):
        for j in range(0, 4):
            for k in range(0, 4):
                for l in range(0, 4):
                    for m in range(0, 4):
                        for n in range(0, 50):
                            for o in range(0, 50):
                                """for p in range(0, 4):
                                    for q in range(0, 4):
                                        for r in range(0, 4):
                                            for s in range(0, 4):"""
                                                #for r in range(0, 1):
                                self.q_lookup_table[i, j, k, l, m, n, o] = counter
                                counter += 1
    if load_from_file:
        self.q_table = np.load(FILENAME + ".npy")
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
    explosion_map = game_state["explosion_map"]
    for bombPosition, _ in game_state["bombs"]:
        bombX, bombY = bombPosition
        for curr_x in range(bombX - 5, bombX + 5):
            if curr_x < 17:
                explosion_map[curr_x, bombY] = 1
        for curr_y in range(bombY - 5, bombY + 5):
            if curr_y < 17:
                explosion_map[bombX, curr_y] = 1
    leftOfMe = (game_state["field"][x-1, y] + 1) if explosion_map[x-1, y] == 0 else 3
    #twoLeftOfMe = (game_state["field"][x-2, y] + 1) if explosion_map[x-1, y] == 0 else 3
    rightOfMe = (game_state["field"][x+1, y] + 1) if explosion_map[x+1, y] == 0 else 3
    #twoRightOfMe = (game_state["field"][x+2, y] + 1) if explosion_map[x+1, y] == 0 else 3
    onTopOfMe = (game_state["field"][x, y-1] + 1) if explosion_map[x, y-1] == 0 else 3
    #twoOnTopOfMe = (game_state["field"][x, y-2] + 1) if explosion_map[x, y-1] == 0 else 3
    belowMe = (game_state["field"][x, y+1] + 1) if explosion_map[x, y+1] == 0 else 3
    #twoBelowMe = (game_state["field"][x, y+2] + 1) if explosion_map[x, y+1] == 0 else 3
    atMyPosition = (game_state["field"][x, y] + 1) if explosion_map[x, y] == 0 else 3
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
    """minBombX = trim_num(minBombX, 3)+3
    minBombY = trim_num(minBombY, 3)+3
    nearestCoinRelativeX = trim_num(nearestCoinRelativeX, 10)+10
    nearestCoinRelativeY = trim_num(nearestCoinRelativeY, 10)+10
    hasBomb = 0
    if minDistance < 4:
        hasBomb = 1
    minBombX = 6 + trim_num(minBombX, 6)
    minBombY = 6 + trim_num(minBombX, 6)"""
    """if np.abs(nearestCoinRelativeX) > 10:
        nearestCoinRelativeX = int(10 * nearestCoinRelativeX / np.abs(nearestCoinRelativeX))
    if np.abs(nearestCoinRelativeY) > 10:
        nearestCoinRelativeY = int(10 * nearestCoinRelativeY / np.abs(nearestCoinRelativeY))
    """
    #state_integer = self.q_lookup_table[leftOfMe, onTopOfMe, rightOfMe, belowMe, atMyPosition, nearestCoinRelativeX, nearestCoinRelativeY, twoLeftOfMe, twoOnTopOfMe, twoRightOfMe, twoBelowMe]
    state_integer = self.q_lookup_table[leftOfMe, onTopOfMe, rightOfMe, belowMe, atMyPosition, nearestCoinRelativeX, nearestCoinRelativeY]
    return int(state_integer)