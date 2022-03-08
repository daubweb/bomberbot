import os
import pickle
import random

import numpy as np
e = 0.1

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    #                       5 features, taken Action, reward
    self.q_table = np.zeros((3, 3, 3, 3, 3, 6))


def act(self, game_state: dict) -> str:
    #get maximum probability
    #sometimes choose random action
    if np.random.rand(1) > e:
        #take random action
        return np.random.choice(ACTIONS)
    else:
        #take currently best action
        features = state_to_features(game_state)
        features_tuple = tuple(features)
        best_choice = ACTIONS[np.argmax(self.q_table[features_tuple])]
        return best_choice


def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return None
    name, score, bomb_possible, own_position = game_state["self"]
    x, y = own_position
    leftOfMe = game_state["field"][x-1, y] + 1
    rightOfMe = game_state["field"][y, x+1,] + 1
    onTopOfMe = game_state["field"][y-1, x] + 1
    belowMe = game_state["field"][y+1, x] + 1
    atMyPosition = game_state["field"][y, x] + 1
    return np.array([leftOfMe, onTopOfMe, rightOfMe, belowMe, atMyPosition])
