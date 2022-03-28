from collections import namedtuple, deque
import doctest
import random
import matplotlib.pyplot as plt
import numpy as np
import pickle
from typing import List
import events as e
import slayer_module.functions as ft
import slayer_module.hyperparameter as hp
np.set_printoptions(threshold=np.inf)


alpha = 0.1
gamma = 0.6
epsilon = 0.1

GAMMA = 0.1





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
        #print("update q table")
        ft.update_q_table(self.q_table, old_game_state, new_game_state, reward, self_action)
    self.totalReward += reward
    


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    print("End of round")
    self.previousRewards.append(self.totalReward)
    self.totalReward = 0
    self.counter += 1
    self.all_coins_collected.append(100 * ( 1- (len(last_game_state["coins"]) / 50)))
    #print("Status:", self.counter)
    #print(self.q_table)
    
    
    if (self.counter % 200 == 0):
        plot, axs = plt.subplots(2)
        axs[0].plot(np.convolve(np.array(self.previousRewards), np.ones(100)/100, mode='valid'))
        axs[0].set_title("Average Reward per Round")
        axs[1].plot(np.convolve(np.array(self.all_coins_collected), np.ones(100)/100, mode='valid'))
        axs[1].set_title("Percentage of All Coins Picked Up")
        plt.show()
        ft.save_q_table(self.q_table, hp.FILENAME)
        print(self.q_table)
        
        print("save q_table")



def reward_from_events(self, events: List[str], before_state) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 50,
        e.INVALID_ACTION: -2,
        e.KILLED_SELF: -300,
        e.SURVIVED_ROUND: 200,
        e.WAITED: -1,
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
doctest.testmod()