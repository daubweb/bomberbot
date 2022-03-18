import pickle
from collections import namedtuple
from typing import List

import keras
import numpy as np
from keras import layers
from tensorflow import keras
import tensorflow as tf

import events as e
from .callbacks import state_to_features

# Experience replay buffers
frame_count = 0
episode_count = 0
num_actions = 6


def create_q_model(self):
    inputs = layers.Input(shape=(1450, 1))
    layer_1 = layers.Dense(250, activation="relu")(inputs)
    layer_2 = layers.Dense(10, activation="relu")(layer_1)
    action = layers.Dense(num_actions, activation="linear")(layer_2)
    return keras.Model(inputs=inputs, outputs=action)


def setup_training(self):
    self.seed = 42
    self.gamma = 0.99  # Discount factor for past rewards
    self.epsilon = 1.0  # Epsilon greedy parameter
    self.epsilon_min = 0.1  # Minimum epsilon greedy parameter
    self.epsilon_max = 1.0  # Maximum epsilon greedy parameter
    self.epsilon_interval = (
            self.epsilon_max - self.epsilon_min
    )  # Rate at which to reduce chance of random action being taken
    self.batch_size = 32  # Size of batch taken from replay buffer
    self.max_steps_per_episode = 1000

    self.optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
    self.action_history = []
    self.state_history = []
    self.state_next_history = []
    self.rewards_history = []
    self.done_history = []
    self.episode_reward_history = []
    self.running_reward = 0
    self.epsilon_random_frames = 50000
    self.epsilon_greedy_frames = 1000000.0
    self.max_memory_length = 100000
    self.update_after_actions = 4
    self.update_target_network = 10000
    self.loss_function = keras.losses.Huber()

    self.frame_count = 0
    self.episode_count = 0
    self.episode_reward = 0

    self.model_target = create_q_model()
    self.model = create_q_model()


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.frame_count += 1
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if self.frame_count < self.epsilon_random_frames or self.epsilon > np.random.rand(1)[0]:
        action = np.random.choice(num_actions)
    else:
        state_tensor = tf.convert_to_tensor(state_to_features(old_game_state))
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = self.model(state_tensor, training=False)
        action = tf.argmax(action_probs[0]).numpy()

    self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
    self.epsilon = max(self.epsilon, self.epsilon_min)

    state_next = state_to_features(new_game_state)
    state_next = np.array(state_next)

    reward = reward_from_events(self, events)
    self.episode_reward += reward

    self.action_history.append(action)
    self.state_history.append(state_to_features(old_game_state))
    self.state_next_history.append(state_to_features(new_game_state))
    done = 1 if new_game_state["step"] > 399 else 0
    self.done_history.append(done)
    self.rewards_history.append(reward)

    if self.frame_count % self.update_after_actions == 0 and len(self.done_history) > self.batch_size:
        indices = np.random.choice(range(len(self.done_history)), size=self.batch_size)
        state_sample = np.array([self.state_hidzotx[i] for i in indices])
        state_next_sample = np.array([self.state_next_history[i] for i in indices])
        rewards_sample = np.array([self.rewards_history[i] for i in indices])
        action_sample = np.array([self.action_history[i] for i in indices])
        done_sample = tf.convert_to_tensor(
            [float(self.done_history[i]) for i in indices]
        )

        future_rewards = self.model_target.predict(state_next_sample)

        updated_q_values = rewards_sample * self.gamma * tf.reduce_max(future_rewards, axis=1)
        updated_q_values = updated_q_values * (1 - done_sample) - done_sample

        masks = tf.one_hot(action_sample, num_actions)

        with tf.GradientTape() as tape:
            q_values = self.model(state_sample)
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = self.loss_function(updated_q_values)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        if self.frame_count % self.update_target_network == 0:
            self.model.target.set_weights(self.model.get_weights())
            message = "running reward {:.2f} at episode {}, frame_count {}]"
            # print(message.format(running_reward, episode_count, self.frame_count))

        if len(self.rewards_history) > self.max_memory_length:
            del self.rewards_history[:-1]
            del self.state_history[:-1]
            del self.state_next_history[:-1]
            del self.action_history[:-1]
            del self.dopne_history[:-1]


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    self.episode_reward_history.append(self.episode_reward)
    if len(self.episode_reward_history) > 100:
        del self.episode_reward_history[:1]
    running_reward = np.mean(self.episode_reward_history)
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)
    self.episode_count += 1


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 50,
        e.INVALID_ACTION: -2,
        e.MOVED_LEFT: -1,
        e.MOVED_UP: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_DOWN: -1,
        e.CRATE_DESTROYED: 2,
        e.KILLED_SELF: -100,
        e.GOT_KILLED: -100
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
