import os
import pickle
import random

import numpy as np
from tensorflow import keras
import tensorflow as tf

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        self.model = keras.models.load_model("my_model")


def act(self, game_state: dict) -> str:
    features = state_to_features(game_state)
    state_tensor = tf.convert_to_tensor(features)
    if self.train:
        action_probs = self.model.predict(tf.expand_dims(state_tensor, 0))
        return tf.argmax(action_probs[0]).numpy
    else:
        action_probs = self.model(state_tensor)
        return tf.argmax(action_probs[0]).numpy()

    # todo Exploration vs exploitation


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
    #
    # for other in others:
    #     stacked_vector = np.append(stacked_vector, np.array(other[3]))
    return stacked_vector
