import os
from random import shuffle
from collections import deque
import pickle
import numpy as np
import random
import doctest
from turtle import shape
from .hyperparameter import *
from .functions import *
import settings as s
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
        self.q_table = load_q_table(FILENAME)
        self.logger.info("Q_table was loaded.")
        #print(self.q_table.shape)
    else:
        self.q_table = initialize_q_table(FEATUREARRAY)
        self.logger.info("Q_table was initialized.")
        #print(self.q_table.shape)
    #self.last_four_moves = []
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0
    self.is_training = False
    

def act(self, game_state):
    """

    """
    # todo Exploration vs exploitation
    # if gamma is under epsilon
    if LEARNWITHHUMAN:
        return game_state['user_input']
    elif PEACEFUL == True and np.random.rand(1) < EPSILON:
        actions = PEACEFULACTIONS
        return np.random.choice(actions)
    elif PEACEFUL == False and np.random.rand(1) < EPSILON:
        actions = ACTIONS
        return np.random.choice(actions)
    else:
        # take the currently best action
        q_values = get_q_values_for_state(self.q_table, game_state)
        #if len(self.last_four_moves) == 4 and self.last_four_moves[0] != self.last_four_moves[1] and self.last_four_moves[0] == self.last_four_moves[2] and self.last_four_moves[1] == self.last_four_moves[3] and self.is_training:
        #    actions = hp.PEACEFULACTIONS
        #    return np.random.choice(repeat_stopper(self, actions))
        #q_values[5] = -20000
        #print("Q_values:", q_values ,"\n Best Q_value:", hp.ACTIONS[np.argmax(q_values)])
        #if self.is_training:
        return acting(self, game_state)
        #else:
        #return ACTIONS[np.argmax(q_values)]


def repeat_stopper(self, list):
    if len(self.last_four_moves) == 4 and self.last_four_moves[0] != self.last_four_moves[1] and self.last_four_moves[0] == self.last_four_moves[2] and self.last_four_moves[1] == self.last_four_moves[3] and self.is_training:
        #print("Last four moves:", self.last_four_moves)
        if self.last_four_moves[0] == 'MOVED_DOWN':
            return ['UP', 'RIGHT', 'LEFT', 'WAIT']
        elif self.last_four_moves[0] == 'MOVED_UP':
            return ['RIGHT', 'DOWN', 'LEFT', 'WAIT']
        elif self.last_four_moves[0] == 'MOVED_LEFT':
            return ['UP', 'RIGHT', 'DOWN', 'WAIT']
        elif self.last_four_moves[0] == 'MOVED_RIGHT':
            return ['UP', 'DOWN', 'LEFT', 'WAIT']

doctest.testmod()


def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]




def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0


def acting(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    self.logger.info('Picking action according to rule set')
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x, y), targets, self.logger)
    if d == (x, y - 1): action_ideas.append('UP')
    if d == (x, y + 1): action_ideas.append('DOWN')
    if d == (x - 1, y): action_ideas.append('LEFT')
    if d == (x + 1, y): action_ideas.append('RIGHT')
    if d is None:
        self.logger.debug('All targets gone, nothing to do anymore')
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    if (x, y) in dead_ends:
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            # Keep track of chosen action for cycle detection
            if a == 'BOMB':
                self.bomb_history.append((x, y))
            return a
