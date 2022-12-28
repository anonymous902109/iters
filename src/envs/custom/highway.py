from highway_env import utils

import numpy as np
import copy
from highway_env.envs import highway_env
from highway_env.vehicle.controller import ControlledVehicle

from src.feedback.feedback_processing import encode_trajectory


class CustomHighwayEnv(highway_env.HighwayEnvFast):

    def __init__(self, shaping=False, time_window=5):
        super().__init__()
        self.shaping = shaping
        self.time_window = time_window

        self.episode = []

        self.state_len = 5
        self.lows = np.zeros((self.state_len, ))
        self.highs = np.ones((self.state_len, ))

        # speed is in [-1, 1]
        self.lows[[3, 4]] = -1
        self.action_dtype = 'int'
        self.state_dtype = 'cont'

        self.lane = 0
        self.lmbda = 0.2

        self.lane_changed = []

        # presence features are immutable
        self.immutable_features = [0]

        self.discrete_features = [0]
        self.cont_features = [f for f in range(self.state_len) if f not in self.discrete_features]

        self.max_changed_lanes = 3

    def step(self, action):
        self.episode.append((self.state, action))

        curr_lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) else self.vehicle.lane_index[2]
        self.state, rew, done, info = super().step(action)

        info['true_rew'] = rew

        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        self.lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) else self.vehicle.lane_index[2]

        self.lane_changed.append(self.lane != curr_lane)

        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])

        coll_rew = self.config["collision_reward"] * self.vehicle.crashed
        right_lane_rew = self.config["right_lane_reward"] * self.lane / max(len(neighbours) - 1, 1)
        speed_rew = self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)

        lane_change = sum(self.lane_changed[-self.time_window:]) >= self.max_changed_lanes
        true_reward = self.calculate_true_reward(rew, lane_change)

        aug_rew = 0
        if self.shaping:
            aug_rew = self.augment_reward(action, self.state)

        rew += aug_rew
        rew += lane_change * self.config['lane_change_reward']

        info['rewards'] = {'collision_rew': coll_rew,
                           'right_lane_rew': right_lane_rew,
                           'speed_rew': speed_rew,
                           'lane_change_rew': aug_rew,
                           'lane_changed': lane_change,
                           'true_reward': true_reward}

        return self.state, rew, done, info

    def calculate_true_reward(self, rew, lane_change):
        true_rew = rew + self.true_rewards['lane_change_reward'] * lane_change

        return true_rew

    def reset(self):
        self.episode = []
        self.lane_changed = []
        self.state = super().reset()

        self.lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) else self.vehicle.lane_index[2]

        return self.state

    def close(self):
        pass

    def render(self):
        super().render(mode='human')

    def render_state(self, state):
        print('State = {}'.format(state.flatten()[0:5]))

    def augment_reward(self, action, state):
        running_rew = 0
        past = copy.copy(self.episode)
        curr = 1

        for j in range(len(past)-1, -1, -1):  # go backwards in the past
            state_enc = encode_trajectory(past[j:], state, curr, self.time_window, self)

            rew = self.reward_model.predict(state_enc)
            running_rew += self.lmbda * rew.item()

            if curr >= self.time_window:
                break

            curr += 1

        return running_rew

    def set_reward_model(self, rm):
        self.reward_model = rm

    def set_shaping(self, boolean):
        self.shaping = boolean

    def set_true_reward(self, rewards):
        self.true_rewards = rewards

    def random_state(self):
        return np.random.uniform(self.lows, self.highs, (self.state_len, ))

    def encode_state(self, state):
        return state[0].flatten()

    def get_feedback(self, best_traj, expl_type):
        feedback_list = []

        for traj in best_traj:
            lanes = [s.flatten()[2] for s, a in traj]
            changed_lanes = [abs(lanes[i] - lanes[i-1]) > 0.1 if i >= 1 else False for i, _ in enumerate(lanes)]

            start = 0
            end = start + 2

            while end < len(changed_lanes):
                while (end - start) <= self.time_window:
                    if end >= len(changed_lanes):
                        break

                    changed = sum(changed_lanes[(start+1):end]) >= self.max_changed_lanes

                    if changed and changed_lanes[start+1]:
                        feedback_list.append(('s', traj[start:end], -1, [2 + (i*self.state_len) for i in range(0, end-start)], end-start))
                        start = end
                        end = start + 2
                        if expl_type == 'expl':
                            break
                    else:
                        end += 1

                start += 1
                end = start + 2

        print('Feedback: {}'.format(feedback_list))
        return feedback_list, True

    def set_lambda(self, l):
        self.lmbda = l



