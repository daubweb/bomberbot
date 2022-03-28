from pyexpat import features
from turtle import distance, position
import numpy as np
import random
import pickle
#import da_bomb_module.hyperparameter as hp
from .hyperparameter import *
import doctest
import math



def game_state_to_features(game_state):
    """
    >>> game_state_to_features(hp.POSITIONSGAMESTATE)
    array([1, 5, 3, 4, 1, 0, 3, 3])
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    left, right, up, down = get_positions(game_state)       # Empty, Crate, Bomb, other Agent, Coin, Wall, Explosion, Danger, Priority
    center = get_center(game_state)                         # Empty, Bomb(agent placed a bomb) and Danger(at least one bomb threatens your current position)              # 
    bomb = get_bomb_availability(game_state)                # Agent can place bomb or not
    #risk = get_risk_score(game_state)                       # Riskscore, probably rounded to 5 values
    priority_x, priority_y = get_priority_tile(game_state)
    features = np.array([left, right, up, down, center, bomb, priority_x, priority_y])
    #print(features)
    return features


# get objects on position: { 0: wall, 1: empty, 2: crate, 0: bomb, 0: other, 1: coins, 0: explosion, 3: danger}
def get_positions(game_state):
    """
    >>> game_state1 = hp.POSITIONSGAMESTATE
    >>> game_state2 = hp.POSITIONSEXPLOSIONGAMESTATE
    >>> get_positions(game_state1)
    [1, 5, 3, 4]
    >>> get_positions(game_state2)
    [1, 5, 3, 6]
    """
    # get wall, empty and crate
    positions = [0,0,0,0]
    _,_,_, own_position = game_state["self"]
    x, y = own_position
    locations = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    positions[0] = game_state["field"][x-1, y] + 1
    positions[1] = game_state["field"][x+1, y] + 1
    positions[2] = game_state["field"][x, y-1] + 1
    positions[3] = game_state["field"][x, y+1] + 1

    #for position in positions:
    #    if position == 2:
    #        position = 1

    
    #look for explosion
    if game_state["explosion_map"][x-1, y] == 1: positions[0] = 0
    if game_state["explosion_map"][x+1, y] == 1: positions[1] = 0
    if game_state["explosion_map"][x, y-1] == 1: positions[2] = 0
    if game_state["explosion_map"][x, y+1] == 1: positions[3] = 0

    
    # look for bombs
    bombs = game_state['bombs']
    bomb_locations = [xy for (xy, t) in bombs]
    for bomb in bomb_locations:
        for i in range(len(locations)):
            if locations[i] == bomb:
                positions[i] = 0
    
    #look for other agents
    others = game_state['others']
    others_locations = [location for (n, score, b, location) in others]
    for other in others_locations:
        for i in range(len(locations)):
            if locations[i] == other:
                positions[i] = 0
    
    

    #look for coins
    coins = game_state['coins']
    for coin in coins:
        for i in range(len(locations)):
            if locations[i] == coin:
                positions[i] = 1
    
    #look for danger
   
    for bomb in bomb_locations:
        for i in range(len(locations)):
            if get_danger_of_tile(game_state, locations[i]) and locations[i]==0:
                positions[i] = 3
    
    
    return positions


# return 0: empty, 1: danger, 2: bomb
def get_center(game_state):
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
    >>> hp.TESTGAMESTATE["bombs"]
    [((3, 5), 1), ((3, 9), 1), ((10, 5), 1), ((4, 5), 1)]
    >>> hp.TESTGAMESTATE["self"][3]
    (3, 4)
    >>> get_center(hp.TESTGAMESTATE)
    1
    >>> get_center(hp.NEWTESTGAMESTATE)
    0
    """
    center_value = 0
    _,_,_,center = game_state["self"]
    if get_danger_of_tile(game_state, center):
        center_value = 1
    #bombs = game_state['bombs']
    #bomb_locations = [xy for (xy, t) in bombs]
    #for bomb in bomb_locations:
    #    if center == bomb:
    #        center_value = 2
    return center_value


def get_distance(p, q):
    """ 
    Return euclidean distance between points p and q
    assuming both to have the same number of dimensions
    >>> get_distance((3,3),(3,4))
    1.0
    >>> get_distance((3,1),(14,-13))
    17.804493814764857
    >>> get_distance((3,4,3),(7,1,6))
    5.830951894845301
    """
    # sum of squared difference between coordinates
    s_sq_difference = 0
    for p_i,q_i in zip(p,q):
        s_sq_difference += (p_i - q_i)**2
    
    # take sq root of sum of squared difference
    distance = s_sq_difference**0.5
    return distance


# returning indices for direction with most potential reward measured in distance:
def get_priority_tile(game_state):
    """
    >>> game_state = { 'self': ('da_bomb', 6, True, (7,1)), 'coins': [(9,7), (7,9), (8,7)], 'others': [('dingdong', 4, True, (7,8)), ('blablabla', 3, True, (7,7)), ('blub', 10, True, (8,8))]}
    >>> get_priority_tile(game_state)
    3
    >>> game_state = { 'self': ('da_bomb', 6, True, (7,15)), 'coins': [(9,7), (7,9), (8,7)], 'others': [('dingdong', 4, True, (7,8)), ('blablabla', 3, True, (7,7)), ('blub', 10, True, (8,8))]}
    >>> get_priority_tile(game_state)
    2
    >>> game_state = { 'self': ('da_bomb', 6, True, (1,7)), 'coins': [(9,7), (7,9), (8,7)], 'others': [('dingdong', 4, True, (7,8)), ('blablabla', 3, True, (7,7)), ('blub', 10, True, (8,8))]}
    >>> get_priority_tile(game_state)
    1
    >>> game_state = { 'self': ('da_bomb', 6, True, (15,7)), 'coins': [(9,7), (7,9), (8,7)], 'others': [('dingdong', 4, True, (7,8)), ('blablabla', 3, True, (7,7)), ('blub', 10, True, (8,8))]}
    >>> get_priority_tile(game_state)
    0
    """
    # initialize positions
    _,_,_,center = game_state["self"]
    priority_location = center
    priority_minimal_distance = 30

    # get minimal distanced coin:
    coins_locations = game_state['coins']
    minimal_distanced_coin = 30
    for coin in coins_locations:
        if get_distance(center, coin) < priority_minimal_distance:
            minimal_distanced_coin = get_distance(center, coin)
            priority_minimal_distance = get_distance(center, coin)
            priority_location = coin
    
    #get minimal distanced crate:
    crates = np.where(game_state["field"] == 1)
    crates_locations = list(zip(crates[0], crates[1]))
    minimal_distanced_crate = 30
    minimal_distanced_crate_location = center
    for crate_location in crates_locations:
        if get_distance(center, crate_location) < minimal_distanced_crate:
            minimal_distanced_crate = get_distance(center, crate_location)
            minimal_distanced_crate_location = crate_location

    
    #get minimal distanced opponent:
    others = game_state['others']
    opponents = [location for (n, score, b, location) in others]
    minimal_distanced_opponnent = 30
    minimal_distanced_opponnent_location = center
    for opponent in opponents:
        if get_distance(center, opponent) < minimal_distanced_opponnent:
            minimal_distanced_opponnent = get_distance(center, opponent)
            minimal_distanced_opponnent_location = opponent
    
    if minimal_distanced_opponnent < minimal_distanced_coin + 2:
        priority_location = minimal_distanced_opponnent_location
    if minimal_distanced_crate < minimal_distanced_coin - 4:
        priority_location = minimal_distanced_crate_location



  
    relative_priority_location = tuple(map(lambda i, j: i - j + 15, center, priority_location))
    return relative_priority_location



# Check if agent is in at least one bomb radius
def get_danger_of_tile(game_state, position):
    """
    >>> _,_,_,position1 = hp.TESTGAMESTATE["self"]
    >>> _,_,_,position2 = hp.NEWTESTGAMESTATE["self"]
    >>> get_danger_of_tile(hp.TESTGAMESTATE, position1)
    True
    >>> get_danger_of_tile(hp.NEWTESTGAMESTATE, position2)
    False
    """
    # Get the location of the bombs and your agent:
    bombs = game_state['bombs']
    bomb_locations = [xy for (xy, t) in bombs]
    
    # Check if the distance between bomb and agent x-values is smaller then 4, while y values are the same
    # and vice-versa. If the answer is yes return True.
    for bomb in bomb_locations:
        if bomb[0] == position[0] and 3 >= abs(bomb[1]-position[1]):
            return True
        elif bomb[1] == position[1] and 3 >= abs(bomb[0]-position[0]):
            return True
    # Or return False because no bomb radii is threatening your agent.
    return False
    

# Returns false if the agent cannot place bomb and True if he can place a bomb
def get_bomb_availability(game_state):
    """
    >>> get_bomb_availability(hp.TESTGAMESTATE)
    False
    >>> get_bomb_availability(hp.NEWTESTGAMESTATE)
    True
    """
    _,_,bomb,_ = game_state["self"]
    if bomb:
        return 1
    else:
        return 0
    

# Returns score 1(no risk), 2(less risk), 3 (neutral), 4(more risk) and 5(high risk) 
def get_risk_score(game_state):
    if game_state['others'] == [] or None:
        return 0
    # get scores of each bomberman and steps until the game end
    step = game_state["step"]
    step_to_end = 400 - step
    _, da_bomb_score, _, _ = game_state["self"]
    others = game_state['others']
    others_scores = [score for (n, score, b, (x, y)) in others]
    score_difference_to_first = da_bomb_score - max(others_scores)
    risk_score = 0

    if step_to_end >= 330 or score_difference_to_first == 0:
        risk_score = 1

    return risk_score





def initialize_q_table(feature_array):
    """
    >>> initialize_q_table([2,2,2])
    array([[0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.]])
    >>> initialize_q_table([1,2])
    array([[0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.]])
    >>> len(hp.ACTIONS)
    6
    >>> hp.FEATUREARRAY
    array([8, 8, 8, 8, 3, 2, 5, 4])
    >>> 8*8*8*8*3*2*5*4
    491520
    >>> initialize_q_table(hp.FEATUREARRAY)
    array([[0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           ...,
           [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.]])
    """
    #print("Featurearray:", feature_array)

    q_state_number = 1
    for i in range(len(feature_array)):
        q_state_number *= feature_array[i]
    q_table = np.zeros((q_state_number, len(ACTIONS)))
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
    array([ 6,  6,  6,  6,  3,  2,  2, 31, 31])
    >>> hp.TESTFEATURES
    array([1, 5, 3, 4, 1, 0, 3, 3])
    >>> 1*1 + 5*8 + 3*8*8 + 4*8*8*8 + 1*8*8*8*8 + 0*8*8*8*8*3 + 3*8*8*8*8*3*2 + 3*8*8*8*8*3*2*5
    448745
    >>> feature_to_q_table_indice(hp.TESTFEATURES, hp.FEATUREARRAY)
    14945471
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
    features = game_state_to_features(state)
    indice = feature_to_q_table_indice(features, FEATUREARRAY)
    return q_table[indice, :]


def set_q_value_for_state_and_action(q_table, state, action, new_q_value):
    features = game_state_to_features(state)
    indicex = feature_to_q_table_indice(features, FEATUREARRAY)
    indicey = action_to_indices(action)
    q_table[indicex, indicey] = new_q_value


def action_to_indices(action):
    """
    """
    for indice in range(len(ACTIONS)):
        #print("action:", action, hp.ACTIONS, indice)
        if ACTIONS[indice] == action:
            return indice

    

def update_q_table(q_table, old_state, new_state, reward, action):
    """
    >>> hp.FEATUREARRAY
    array([8, 8, 8, 8, 3, 2, 5, 4])
    >>> q_table = initialize_q_table(hp.FEATUREARRAY)
    >>> old_state, new_state, action = hp.POSITIONSEXPLOSIONGAMESTATE, hp.POSITIONSGAMESTATE, 'DOWN'
    >>> reward = 100
    >>> action
    'DOWN'
    >>> indice_old_state = feature_to_q_table_indice(game_state_to_features(old_state), hp.FEATUREARRAY)
    >>> indice_new_state = feature_to_q_table_indice(game_state_to_features(new_state), hp.FEATUREARRAY)
    >>> indice_action = action_to_indices(action)
    >>> indice_old_state, indice_new_state, indice_action
    (204009, 448745, 2)


    >>> q_table[indice_old_state, :]
    array([0., 0., 0., 0., 0., 0.])
    >>> q_table[indice_old_state, indice_action]
    0.0
    >>> q_table[indice_new_state, :]
    array([0., 0., 0., 0., 0., 0.])
    >>> q_table[indice_new_state, indice_action]
    0.0
    >>> (1-hp.ALPHA)* q_table[indice_old_state, indice_action] + hp.ALPHA*(reward+hp.GAMMA+q_table[indice_new_state, indice_action])
    10.06
    
    
    >>> update_q_table(q_table, old_state, new_state, reward, action)


    >>> q_table[indice_old_state, :]
    array([ 0.  ,  0.  , 10.06,  0.  ,  0.  ,  0.  ])
    >>> q_table[indice_old_state, indice_action]
    10.06
    """

    
    indice_old_state = feature_to_q_table_indice(game_state_to_features(old_state), FEATUREARRAY)
    indice_new_state = feature_to_q_table_indice(game_state_to_features(new_state), FEATUREARRAY)
    indice_action = action_to_indices(action)
    #print(indice_old_state, indice_new_state, indice_action,q_table[indice_old_state, indice_action],q_table[indice_old_state, indice_action] + hp.ALPHA*(reward+hp.GAMMA*np.max(q_table[indice_new_state, :])-q_table[indice_old_state, indice_action]))
    q_table[indice_old_state, indice_action] = q_table[indice_old_state, indice_action] + ALPHA*(reward+GAMMA*np.max(q_table[indice_new_state, :])-q_table[indice_old_state, indice_action])

    



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