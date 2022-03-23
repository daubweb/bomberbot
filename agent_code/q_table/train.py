from typing import List
import matplotlib.pyplot as plt
import numpy as np

import events as e
from .callbacks import state_to_features, ACTIONS

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

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    state = state_to_features(self, old_game_state)
    next_state = state_to_features(self, new_game_state)

    action = ACTIONS.index(self_action)

    reward = reward_from_events(self, events, old_game_state)

    old_value = self.q_table[state, ACTIONS.index(self_action)]
    next_max = np.max(self.q_table[next_state])

    new_value = (1-alpha) * old_value + alpha * (reward + gamma * next_max)
    self.q_table[state, action] = new_value

    self.totalReward += reward

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.previousRewards.append(self.totalReward)
    self.totalReward = 0
    self.counter += 1
    self.all_coins_collected.append(100 * ( 1- (len(last_game_state["coins"]) / 50)))

    if (self.counter % 2000 == 0):
        plot, axs = plt.subplots(2)
        axs[0].plot(np.convolve(np.array(self.previousRewards), np.ones(100)/100, mode='valid'))
        axs[0].set_title("Average Reward per Round")
        axs[1].plot(np.convolve(np.array(self.all_coins_collected), np.ones(100)/100, mode='valid'))
        axs[1].set_title("Percentage of All Coins Picked Up")
        plt.show()
        np.save("q_table_integers", self.q_table)

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
