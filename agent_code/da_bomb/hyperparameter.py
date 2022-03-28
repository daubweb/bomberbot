import numpy as np
# all hyperparameter

#human inputs for learning purposes
LEARNWITHHUMAN = False
PEACEFUL = False

# possible Actions
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
PEACEFULACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']

# Features: left, right, up, down, center, bomb, risk, priorityx, priorityy
FEATUREARRAY = np.array([4, 4, 4, 4, 2, 2, 31, 31])

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
EPSILON = 0.0

# Gamma: learning factor
GAMMA = 0.8
#
ALPHA = 0.5

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
EXPLOSIONFIELD = np.zeros((TESTXWIDTH,TESTYWIDTH))
EXPLOSIONTESTFIELD = np.zeros((TESTXWIDTH,TESTYWIDTH))
EXPLOSIONTESTFIELD[6,7] = 1


# Teststate old
TESTGAMESTATE = {
    'round': 3,
    'step': 14,
    'field': np.array(TESTFIELD, dtype = np.int8),
    'self': ('slayer', 6, False, (3,4)),
    'others': [],
    'bombs': [((3,5), 1), ((3,9), 1), ((10,5), 1), ((4,5), 1)],
    'coins': [(3,3), (4,5), (3,4)],
    'user_input': None,
    'explosion_map' : [],
}

# Teststate new
NEWTESTGAMESTATE = {
    'round': 3,
    'step': 15,
    'field': np.array(TESTFIELD, dtype = np.int8),
    'self': ('slayer', 6, True, (3,3)),
    'others': [],
    'bombs': [((3,7), 1), ((3,9), 1), ((10,5), 1), ((4,5), 1)],
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


TESTFEATURES = np.array([2, 2, 2, 2, 1, 1, 30, 30])


POSITIONSGAMESTATE = {
    'round': 3,
    'step': 14,
    'field': np.array(TESTFIELD, dtype = np.int8),
    'self': ('slayer', 6, False, (6,6)),
    'others': [('dingdong', 4, True, (6,7)), ('blablabla', 3, True, (9,9)), ('blub', 10, True, (10,10))],
    'bombs': [((6,5), 1), ((3,9), 1), ((10,5), 1), ((4,5), 1)],
    'coins': [(7,6), (4,5), (3,4)],
    'user_input': None,
    'explosion_map' : np.array(EXPLOSIONFIELD, dtype = np.int8),
}

POSITIONSEXPLOSIONGAMESTATE = {
    'round': 3,
    'step': 14,
    'field': np.array(TESTFIELD, dtype = np.int8),
    'self': ('slayer', 6, False, (6,6)),
    'others': [('dingdong', 4, True, (11,11)), ('blablabla', 3, True, (9,9)), ('blub', 10, True, (10,10))],
    'bombs': [((6,5), 1), ((3,9), 1), ((10,5), 1), ((4,5), 1)],
    'coins': [(7,6), (4,5), (3,4)],
    'user_input': None,
    'explosion_map' : np.array(EXPLOSIONTESTFIELD, dtype = np.int8),
}