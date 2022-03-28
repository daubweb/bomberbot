import numpy as np
# all hyperparameter

# possible Actions
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Features: left, right, up, down, center, nearestcoinx, nearestcoiny
FEATUREARRAY = np.array([3, 3, 3, 3, 3, 24, 24])

# Filename, where q_table is saved
FILENAME = "q_table"

TILE = {
    'Wall' : 0,
    'Empty' : 1,
    'Crate' : 2,
    'Coin' : 3,
    'Slayer' : 4,
}

# Wall(-1), Empty(0), Crate(1), Coins(), Self()
TILESNUMBER = len(TILE)

# Epsilon: This is needed for exploration in the Q-learning algorithm
EPSILON = 0.1

# Gamma: Learning factor
GAMMA = 0.6

#
ALPHA = 0.1

#######################################
# Hyperparameter for Testing pruposes #
#######################################

# hypothetical Field
TESTXWIDTH = 15
TESTYWIDTH = 15
TESTFIELD = np.zeros((TESTXWIDTH,TESTYWIDTH))
TESTFIELD[:,-1] = -1
TESTFIELD[0,:] = -1
TESTFIELD[-1,:] = -1
TESTFIELD[:,0] = -1
TESTFIELD[:-1:2,0:-1:2] = -1

# Teststate old
TESTGAMESTATE = {
    'round': 3,
    'step': 14,
    'field': np.array(TESTFIELD, dtype = np.int8),
    'self': ('slayer', 6, False, (3,4)),
    'others': [],
    'bombs': [],
    'coins': [(3,3), (4,5), (3,4)],
    'user_input': None,
    'explosion_map' : [],
}

# Teststate new
NEWTESTGAMESTATEN = {
    'round': 3,
    'step': 15,
    'field': np.array(TESTFIELD, dtype = np.int8),
    'self': ('slayer', 6, False, (3,3)),
    'others': [],
    'bombs': [],
    'coins': [(4,5), (3,4)],
    'user_input': None,
    'explosion_map' : [],
}

# Teststate new
COINTESTGAMESTATE = {
    'round': 3,
    'step': 15,
    'field': np.array(TESTFIELD, dtype = np.int8),
    'self': ('slayer', 6, False, (3,3)),
    'others': [],
    'bombs': [],
    'coins': [(8,8), (9,9)],
    'user_input': None,
    'explosion_map' : [],
}

TESTFEATURES = np.array([1, 1, 1, 1, 1, 5, 5])