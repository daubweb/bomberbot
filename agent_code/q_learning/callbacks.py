import os
import pickle
import random

import numpy as np
from tensorflow import keras

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):


    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

    self.model = keras.models.load_model("my_model")




def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    features = state_to_features(game_state)
    epsilon = 0.9

    if self.train:
        if np.rand(1)[0] > epsilon:
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        else:

    # print(features)
    # todo Exploration vs exploitation

    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    else:
        test_input = np.random.random((1, 32))

    self.logger.debug("Querying model for action.")

    # return ACTIONS[np.argmax(self.model.predict(test_input).flatten())]


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends

    if game_state is None:
        return None

    coin_array = np.zeros(game_state["field"].shape)

    expl_map = game_state["explosion_map"]
    coins = np.array(game_state["coins"])
    others = game_state["others"]

    for coin in coins:
        coin_array[coin[0]][coin[1]] = 1
    # For example, you could construct several channels of equal shape, ...
    channels = [
        game_state["field"].flatten(),
        np.array(game_state["explosion_map"]).flatten(),
        expl_map.flatten(),
        coin_array.flatten(),
    ]
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)

    # and return them as a vector
    stacked_vector = stacked_channels.reshape(-1)
    stacked_vector = np.append(stacked_vector, np.array(game_state["self"][3]))

    for other in others:
        stacked_vector = np.append(stacked_vector, np.array(other[3]))
    return stacked_vector
