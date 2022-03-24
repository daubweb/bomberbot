from pyexpat import features
import numpy as np
import pickle
#import slayer_module.hyperparameter as hp
import hyperparameter as hp
import doctest



def game_state_to_features(game_state):
    """
    >>> game_state_to_features(hp.COINTESTGAMESTATE)
    array([ 1,  1,  1,  1,  1, 17, 17])
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    left, right, up, down, center = positions(game_state)
    cx, cy = nearest_coin(game_state)
    features = np.array([left, right, up, down, center, cx, cy])
    return features


def positions(game_state):
    """
    >>> hp.TESTGAMESTATE["field"]
    array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
           [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
           [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
           [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
           [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
           [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
           [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
           [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
           [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
           [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
           [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
           [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
           [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
           [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
           [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]],
          dtype=int8)
    >>> hp.TESTGAMESTATE["self"]
    ('slayer', 6, False, (3, 4))
    >>> hp.TESTGAMESTATE["self"][3]
    (3, 4)
    >>> x, y = hp.TESTGAMESTATE["self"][3]
    >>> x
    3
    >>> y
    4
    >>> hp.TESTGAMESTATE["field"][x-1, y] + 1
    0
    >>> hp.TESTGAMESTATE["field"][x+1, y] + 1
    0
    >>> hp.TESTGAMESTATE["field"][x, y-1] + 1
    1
    >>> hp.TESTGAMESTATE["field"][x, y+1] + 1
    1
    >>> hp.TESTGAMESTATE["field"][x, y] + 1
    1
    >>> positions(hp.TESTGAMESTATE)
    (0, 0, 1, 1, 1)
    """
    _,_,_, own_position = game_state["self"]
    x, y = own_position
    leftOfMe = game_state["field"][x-1, y] + 1
    rightOfMe = game_state["field"][x+1, y] + 1
    onTopOfMe = game_state["field"][x, y-1] + 1
    belowMe = game_state["field"][x, y+1] + 1
    atMyPosition = game_state["field"][x, y] + 1
    #print("5-state-field:", leftOfMe, rightOfMe, onTopOfMe, belowMe, atMyPosition)
    return leftOfMe, rightOfMe, onTopOfMe, belowMe, atMyPosition

def nearest_coin(game_state):
    """
    >>> hp.COINTESTGAMESTATE["self"][3]
    (3, 3)
    >>> hp.COINTESTGAMESTATE["coins"]
    [(8, 8), (9, 9)]
    >>> nearest_coin(hp.COINTESTGAMESTATE)
    (17, 17)
    """
    _,_,_, own_position = game_state["self"]
    x, y = own_position
    allCoins = game_state["coins"]
    minDistance = 100
    nearestCoinX = 0
    nearestCoinY = 0
    for coin in allCoins:
        coinX, coinY = coin
        distToCoin = (x - coinX)**2 + (y - coinY)**2
        if distToCoin < minDistance:
            minDistance = distToCoin
            nearestCoinX = coinX
            nearestCoinY = coinY
    nearestCoinRelativeX = nearestCoinX - x + 12
    nearestCoinRelativeY = nearestCoinY - y + 12
    #print("Nearest Coin:", nearestCoinRelativeX, nearestCoinRelativeY)
    return nearestCoinRelativeX, nearestCoinRelativeY

def bomb():
    pass



def initialize_q_table(feature_array):
    """
    >>> initialize_q_table([2,2,2])
    array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]])
    >>> initialize_q_table([1,2])
    array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]])
    >>> len(hp.ACTIONS)
    5
    >>> hp.FEATUREARRAY
    array([ 3,  3,  3,  3,  3, 24, 24])
    >>> 3*3*3*3*3*12*12
    34992
    >>> initialize_q_table(hp.FEATUREARRAY)
    array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           ...,
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]])
    """
    q_state_number = 1
    for i in range(len(feature_array)):
        q_state_number *= feature_array[i]
    q_table = np.zeros((q_state_number, len(hp.ACTIONS)))
    return q_table

    

def feature_to_q_table_indice(features, feature_array):
    """
    >>> feature_to_q_table_indice([0, 0, 0], [2, 2, 2])
    0
    >>> feature_to_q_table_indice([1, 1, 1], [2, 2, 2])
    7
    >>> feature_to_q_table_indice([1, 0, 1], [2, 2, 2])
    5
    >>> hp.FEATUREARRAY
    array([ 3,  3,  3,  3,  3, 24, 24])
    >>> hp.TESTFEATURES
    array([1, 1, 1, 1, 1, 5, 5])
    >>> 1*1 + 1*3 + 1*3*3 + 1*3*3*3 + 1*3*3*3*3 + 5*3*3*3*3*3 + 5*3*3*3*3*3*24
    30496
    >>> feature_to_q_table_indice(hp.TESTFEATURES, hp.FEATUREARRAY)
    30496
    """
    
    q_state_number = 0
    counter = 1
    #print(features, feature_array)
    for i in range(len(feature_array)):
        #print(i, features[i], counter)
        q_state_number += features[i] * counter
        counter *= feature_array[i]
    #print("q_state_number:", q_state_number)
    return q_state_number


def get_q_values_for_state(q_table, state):
    """
    Testing was done in update_q_table(q_table, old_state, new_state, reward, old_action, new_action)
    """
    # This is the dict before the game begins and after it ends
    if state is None:
        return None
    #print("State:", state["field"], state["coins"])
    features = game_state_to_features(state)
    #print("Features:", features)
    indice = feature_to_q_table_indice(features, hp.FEATUREARRAY)
    #print("indices:", indice)
    return q_table[indice, :]


def set_q_value_for_state_and_action(q_table, state, action, new_q_value):
    features = game_state_to_features(state)
    indicex = feature_to_q_table_indice(features, hp.FEATUREARRAY)
    indicey = action_to_indices(action)
    q_table[indicex, indicey] = new_q_value


def action_to_indices(action):
    """
    """
    for indice in range(len(hp.ACTIONS)):
        #print("action:", action, hp.ACTIONS, indice)
        if hp.ACTIONS[indice] == action:
            return indice

    

def update_q_table(q_table, old_state, new_state, reward, action):
    """
    >>> hp.FEATUREARRAY
    array([ 3,  3,  3,  3,  3, 24, 24])
    >>> q_table = initialize_q_table(hp.FEATUREARRAY)
    >>> old_state, new_state, action = hp.TESTGAMESTATE, hp.NEWTESTGAMESTATEN, 'DOWN'
    >>> reward = 100
    >>> action
    'DOWN'
    >>> indice_old_state = feature_to_q_table_indice(game_state_to_features(old_state), hp.FEATUREARRAY)
    >>> indice_new_state = feature_to_q_table_indice(game_state_to_features(new_state), hp.FEATUREARRAY)
    >>> indice_action = action_to_indices(action)
    >>> indice_old_state, indice_new_state, indice_action
    (73017, 78853, 2)


    >>> q_table[indice_old_state, :]
    array([0., 0., 0., 0., 0.])
    >>> q_table[indice_old_state, indice_action]
    0.0
    >>> q_table[indice_new_state, :]
    array([0., 0., 0., 0., 0.])
    >>> q_table[indice_new_state, indice_action]
    0.0
    >>> (1-hp.ALPHA)* q_table[indice_old_state, indice_action] + hp.ALPHA*(reward+hp.GAMMA+q_table[indice_new_state, indice_action])
    10.06
    
    
    >>> update_q_table(q_table, old_state, new_state, reward, action)


    >>> q_table[indice_old_state, :]
    array([ 0.  ,  0.  , 10.06,  0.  ,  0.  ])
    >>> q_table[indice_old_state, indice_action]
    10.06
    """

    indice_old_state = feature_to_q_table_indice(game_state_to_features(old_state), hp.FEATUREARRAY)
    indice_new_state = feature_to_q_table_indice(game_state_to_features(new_state), hp.FEATUREARRAY)
    indice_action = action_to_indices(action)

    q_table[indice_old_state, indice_action] = (1-hp.ALPHA)* q_table[indice_old_state, indice_action] + hp.ALPHA*(reward+hp.GAMMA+q_table[indice_new_state, indice_action])

    print("Q_values", indice_old_state, indice_action,q_table[indice_old_state,:])



def load_q_table(filename):
    """
    >>> q_table = np.array([5,4,3,2,2])
    >>> filename = "blabla"
    >>> save_q_table(q_table, filename)
    >>> q_table_new = load_q_table(filename)
    >>> q_table == q_table_new
    array([ True,  True,  True,  True,  True])
    """
    infile = open(filename,'rb')
    q_table = pickle.load(infile)
    infile.close()
    return q_table

def save_q_table(q_table, filename):
    outfile = open(filename,'wb')
    pickle.dump(q_table,outfile)
    outfile.close()
    

doctest.testmod()