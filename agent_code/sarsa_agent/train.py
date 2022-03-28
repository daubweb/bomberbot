from typing import List
import matplotlib.pyplot as plt
import numpy as np

import events as e
from .callbacks import state_to_features, ACTIONS, FILENAME

alpha = 0.1
gamma = 0.6
epsilon = 0.1

GAMMA = 0.1

def setup_training(self):
    self.counter = 1
    self.totalReward = 0
    self.previousRewards = []
    self.is_training = True
    self.all_coins_collected = []

def reward_for_bomb_reaction(self, old_game_state: dict, new_game_state: dict):
    if old_game_state == None:
        return 0
    _, _, _, old_position = old_game_state["self"]
    _, _, _, new_posiion = new_game_state["self"]

    old_x, old_y = old_position
    new_x, new_y = new_posiion

    old_bombs = old_game_state["bombs"]
    new_bombs = new_game_state["bombs"]

    new_exlosion_map = old_game_state["explosion_map"]

    old_bomb_x = 1000
    old_bomb_y = 1000

    new_bomb_x = 1000
    new_bomb_y = 1000

    min_old_bomb_distance = 1000
    min_new_bomb_distance = 1000

    for position, _ in old_bombs:
        bombX, bombY = position
        distance = (old_x-bombX)**2 + (old_y-bombY)**2
        if distance < min_old_bomb_distance:
            min_old_bomb_distance = distance
            old_bomb_x = bombX
            old_bomb_y = bombX
    
    for position, _ in new_bombs:
        bombX, bombY = position
        distance = (new_x-bombX)**2 + (new_y-bombY)**2
        if distance < min_new_bomb_distance:
            min_new_bomb_distance = distance
            new_bomb_x = bombX
            new_bomb_y = bombY

    if min_old_bomb_distance < 5:
        if old_bomb_x == new_bomb_x and old_bomb_y == new_bomb_y:
            if new_exlosion_map[old_x, old_y] > 0 and new_exlosion_map[new_x, new_y] < 0.001:
                print("escaped bomb")
                return 10
            if new_exlosion_map[old_x, old_y] < 0.001 and new_exlosion_map[new_x, new_y] > 0:
                print("went to bomb")
                return -10
        if min_old_bomb_distance < min_new_bomb_distance:
            return +5
        elif min_old_bomb_distance > min_new_bomb_distance:
            return -5

    return 0

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    global alpha
    state = state_to_features(self, old_game_state)
    next_state = state_to_features(self, new_game_state)

    action = ACTIONS.index(self_action)

    reward = reward_from_events(self, events, old_game_state)

    reward += reward_for_bomb_reaction(self, old_game_state, new_game_state)

    name, score, bomb_possible, own_position = new_game_state["self"]
    x, y = own_position
    for bombPosition, _ in new_game_state["bombs"]:
        bombX, bombY = bombPosition
        if x in range(bombX - 3, bombX + 3) or y in range(bombY - 3, bombY + 3):
            reward -= 5
            break

    old_value = self.q_table[state, ACTIONS.index(self_action)]
    next_max = np.max(self.q_table[next_state])

    new_value = (1-alpha) * old_value + alpha * (reward + gamma * next_max)
    self.q_table[state, action] = new_value
    #alpha = 1 / (self.counter**(1/5))
    self.totalReward += reward

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.previousRewards.append(self.totalReward)
    self.totalReward = 0
    self.counter += 1
    self.all_coins_collected.append(100 * ( 1- (len(last_game_state["coins"]) / 50)))

    if (self.counter % 1000 == 0) and self.counter > 100:
        #plot, axs = plt.subplots(2)
        plt.plot(np.convolve(np.array(self.previousRewards), np.ones(100)/100, mode='valid'), color="blue")
        #axs[0].set_title("Average Reward per Round")
        #axs[1].plot(np.convolve(np.array(self.all_coins_collected), np.ones(100)/100, mode='valid'))
        #axs[1].set_title("Percentage of All Coins Picked Up")
        plt.draw()
        plt.pause(0.001)
        np.save(FILENAME, self.q_table)

def reward_from_events(self, events: List[str], before_state) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 50,
        e.INVALID_ACTION: -2,
        e.CRATE_DESTROYED: 10,
        e.KILLED_SELF: -300,
        e.SURVIVED_ROUND: 200,
        #e.WAITED: -1,
        #e.MOVED_LEFT: -1,
        #e.MOVED_RIGHT: -1,
        #e.MOVED_UP: -1,
        #e.MOVED_DOWN: -1,
        e.BOMB_DROPPED: -1,
        e.GOT_KILLED: -100
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    if e.WAITED in events and before_state != None and len(before_state["coins"]) > 0:
        reward_sum -= 1
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
