import os
import pickle
import random

import numpy as np
e = 0.1

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    


def act(self, game_state: dict) -> str:
    //get maximum probability
    //sometimes choose random action
    if np.random.rand(1) > e:
        #take random action
        return np.random.choice(ACTION)
    else:
        #take currently best action



def state_to_features(game_state: dict) -> np.array:
    name, score, bomb_possible, own_position = game_state["self"]
    x, y = own_position
    leftOfMe = game_state["field"][x-1, y]
    rightOfMe = game_state["field"][y, x+1,]
    onTopOfMe = game_state["field"][y-1, x]
    belowMe = game_state["field"][y+1, x]
    atMyPosition = game_state["field"][y, x]
    return np.array([leftOfMe, onTopOfMe, rightOfMe, belowMe, atMyPosition])
