import os
import pickle
import numpy as np
import random
import doctest
from turtle import shape
import da_bomb_module.hyperparameter as hp 
import da_bomb_module.functions as ft
np.set_printoptions(threshold=np.inf)


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    >>> os.name
    'posix'
    >>> os.getcwd()
    '/Users/cici/bomberbot/agent_code/slayer'
    >>> os.listdir()
    ['functions.py', '__pycache__', 'hyperparameter.py', 'callbacks.py', 'train.py', 'x']
    >>> os.path.isfile("functions.py")
    True
    >>> x = [4,4,4,6,3,3,3]
    >>> filename = 'x'
    >>> outfile = open(filename,'wb')
    >>> pickle.dump(x,outfile)
    >>> outfile.close()
    >>> os.listdir()
    ['functions.py', '__pycache__', 'hyperparameter.py', 'callbacks.py', 'train.py', 'x']
    >>> infile = open(filename,'rb')
    >>> new_dict = pickle.load(infile)
    >>> infile.close()
    >>> new_dict
    [4, 4, 4, 6, 3, 3, 3]
    >>> new_dict==x
    True
    >>> type(new_dict)
    <class 'list'> 
    """
    if os.path.isfile("q_table"):
        self.q_table = ft.load_q_table(hp.FILENAME)
        self.logger.info("Q_table was loaded.")
    else:
        print("Initialize Q_table")
        self.q_table = ft.initialize_q_table(hp.FEATUREARRAY)
        self.logger.info("Q_table was initialized.")

def act(self, game_state):
    """

    """
    # todo Exploration vs exploitation
    # if gamma is under epsilon 
    if np.random.rand(1) < hp.EPSILON:
        return np.random.choice(hp.ACTIONS)
    else:
        # take the currently best action
        return hp.ACTIONS[np.argmax(ft.get_q_values_for_state(self.q_table, game_state))]

doctest.testmod()