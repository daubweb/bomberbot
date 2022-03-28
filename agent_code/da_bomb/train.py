from collections import namedtuple, deque
import doctest
import random
import matplotlib.pyplot as plt
import numpy as np
import pickle
from typing import List
import events as e
from .functions import *
from .hyperparameter import *
np.set_printoptions(threshold=np.inf)





def setup_training(self):
    self.counter = 0
    self.totalReward = 0
    self.previousRewards = []
    self.is_training = True
    self.all_coins_collected = []
    print("Setup training")


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    reward = reward_from_events(self, events, old_game_state)
    #print("Between steps reward:", reward)
    if old_game_state!= None:
        #print("update q table:", reward)
        update_q_table(self.q_table, old_game_state, new_game_state, reward, self_action)
    self.totalReward += reward
    
    


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    reward = reward_from_events(self, events, last_game_state)
    self.totalReward += reward
    indice_last_state = feature_to_q_table_indice(game_state_to_features(last_game_state), FEATUREARRAY)
    indice_last_action = action_to_indices(last_action)
    self.q_table[indice_last_state, indice_last_action]= reward



    self.previousRewards.append(self.totalReward)
    self.totalReward = 0
    self.counter += 1
    self.all_coins_collected.append(100 * ( 1- (len(last_game_state["coins"]) / 50)))

   
    #regular saving of q_table
    if (self.counter % 1000 == 0):
        save_q_table(self.q_table, FILENAME)
    
    if (self.counter % 50 == 0):
        #plot, axs = plt.subplots(2)
        plt.plot(np.convolve(np.array(self.previousRewards), np.ones(100)/100, mode='valid'))
        #axs[0].set_title("Average Reward per Round")
        #axs[1].plot(np.convolve(np.array(self.all_coins_collected), np.ones(100)/100, mode='valid'))
        #axs[1].set_title("Percentage of All Coins Picked Up")
        plt.draw()
        plt.pause(0.01)



def reward_from_events(self, events: List[str], before_state) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    
    game_rewards = {

        e.MOVED_LEFT: 0,
        e.MOVED_RIGHT: 0,
        e.MOVED_UP: 0,
        e.MOVED_DOWN: 0,
        e.WAITED: -10,
        e.INVALID_ACTION: -10,

        e.BOMB_DROPPED: 0,
        #e.BOMB_EXPLODED:,

        e.CRATE_DESTROYED: 10,
        e.COIN_FOUND: 15,
        e.COIN_COLLECTED: 20,

        e.KILLED_OPPONENT:100,
        e.KILLED_SELF: -100,

        e.GOT_KILLED: -100,
        #e.OPPONENT_ELIMINATED:,
        e.SURVIVED_ROUND: 1000,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    """
    if e.MOVED_LEFT in events:
        self.last_four_moves.pop(0)
        self.last_four_moves.append(e.MOVED_LEFT)
        ft.save_q_table(self.last_four_moves, "last_four_moves")
    if e.MOVED_RIGHT in events:
        self.last_four_moves.pop(0)
        self.last_four_moves.append(e.MOVED_RIGHT)
        ft.save_q_table(self.last_four_moves, "last_four_moves")
    if e.MOVED_UP in events:
        self.last_four_moves.pop(0)
        self.last_four_moves.append(e.MOVED_UP)
        ft.save_q_table(self.last_four_moves, "last_four_moves")
    if e.MOVED_DOWN in events:
        self.last_four_moves.pop(0)
        self.last_four_moves.append(e.MOVED_DOWN)
        ft.save_q_table(self.last_four_moves, "last_four_moves")
    if len(self.last_four_moves) >= 5:
        self.last_four_moves.pop(0)
    """
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
doctest.testmod()