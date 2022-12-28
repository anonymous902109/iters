import copy
import random

import gym
import numpy as np

from src.feedback.feedback_processing import encode_trajectory


class Gridworld(gym.Env):

    def __init__(self, time_window, shaping=False):
        self.world_dim = 5
        self.time_window = time_window

        self.lows = np.array([0, 0, 0, 0, 0])
        self.highs = np.array([self.world_dim, self.world_dim, self.world_dim, self.world_dim, 4])
        self.observation_space = gym.spaces.Box(self.lows, self.highs, shape=(5, ))
        self.action_space = gym.spaces.Discrete(2)
        self.action_dtype = 'int'

        self.state = np.zeros((5, ))
        self.state_len = 5
        self.state_dtype = 'int'

        self.step_pen = -1
        self.turn_pen = 0
        self.goal_rew = 1
        self.shaping = shaping

        self.max_steps = 50
        self.steps = 0
        self.lmbda = 0.2

        # keep record of the last episode
        self.episode = []

        self.config = {
            "goal_rew": 1,
            "step_pen": -1,
            "turn_pen": -1
        }

        self.immutable_features = []
        self.discrete_features = [0, 1, 2, 3, 4]
        self.cont_features = []

        self.feature_names = ['agent_x', 'agent_y', 'goal_x', 'goal_y', 'orient']
        self.feature_names = [fn + '_{}'.format(i) for i in range(self.time_window) for fn in self.feature_names] + \
                             ['action_{}'.format(i) for i in range(self.time_window-1)]

    def step(self, action):
        self.episode.append((self.state, action))

        agent_x, agent_y, goal_x, goal_y, orient = self.state
        if action == 0:
            if orient == 0:
                if agent_x + 1 < self.world_dim:
                    agent_x += 1
            elif orient == 1:
                if agent_y + 1 < self.world_dim:
                    agent_y += 1
            elif orient == 2:
                if agent_x - 1 >= 0:
                    agent_x -= 1
            elif orient == 3:
                if agent_y - 1 >= 0:
                    agent_y -= 1

        if action == 1:
            orient = (orient + 1) % 4

        new_state = np.array([agent_x, agent_y, goal_x, goal_y, orient])

        done = self.check_if_done(new_state)
        rew = self.calculate_reward(action, new_state)

        self.state = new_state
        self.steps += 1

        reached_goal = (self.state[0] == self.state[2]) and (self.state[1] == self.state[3])
        info = {}

        true_rew = self.calculate_true_reward(action, new_state)
        info['rewards'] = {'goal': int(reached_goal), 'true_reward': true_rew, 'train_rew': rew}

        return new_state.flatten(), rew, done, info

    def check_if_done(self, state):
        if self.steps >= self.max_steps:
            return True

        agent_x, agent_y, goal_x, goal_y, orient = state

        if (agent_x == goal_x) and (agent_y == goal_y):
            return True

        return False

    def calculate_reward(self, action, state):
        agent_x, agent_y, goal_x, goal_y, orient = state

        rew = 0.0

        if (agent_x == goal_x) and (agent_y == goal_y):
            rew = self.goal_rew
        elif action == 0:
            rew = self.step_pen
        elif action == 1:
            rew = self.turn_pen

        if self.shaping:
            rew += self.augment_reward(action, state)

        return rew

    def calculate_true_reward(self, action, state):
        true_step_pen = self.true_rewards['step_pen']
        true_turn_pen = self.true_rewards['turn_pen']
        true_goal_rew = self.true_rewards['goal_rew']

        agent_x, agent_y, goal_x, goal_y, orient = state

        rew = 0.0

        if (agent_x == goal_x) and (agent_y == goal_y):
            rew = true_goal_rew
        elif action == 0:
            rew = true_step_pen
        elif action == 1:
            rew = true_turn_pen

        return rew

    def augment_reward(self, action, state):
        running_rew = 0
        past = self.episode
        curr = 1
        for j in range(len(past)-1, -1, -1):  # go backwards in the past
            state_enc = encode_trajectory(past[j:], state, curr, self.time_window, self)

            rew = self.lmbda * self.reward_model.predict(state_enc).item()
            running_rew += rew

            # if rew.item() < -0.2:
            # print('{} {}'.format(state_enc, rew.item()))

            if curr >= self.time_window:
                break

            curr += 1

        return running_rew

    def reset(self):
        goal_x = random.randint(0, self.world_dim - 1)
        goal_y = random.randint(0, self.world_dim - 1)

        agent_x = random.randint(0, self.world_dim - 1)
        agent_y = random.randint(0, self.world_dim - 1)

        while (abs(agent_x - goal_x) < 2) or (abs(agent_y - goal_y) < 2):
            agent_x = random.randint(0, self.world_dim - 1)
            agent_y = random.randint(0, self.world_dim - 1)

        orient = random.randint(0, 3)

        self.state = np.array([agent_x, agent_y, goal_x, goal_y, orient])
        self.steps = 0
        self.episode = []

        return self.state.flatten()

    def close(self):
        pass

    def render(self):
        self.render_state(self.state)

    def render_state(self, state):
        agent_x, agent_y, goal_x, goal_y, orient = state
        rendering = '---------------\n'
        rendering += 'State = {}\n'.format(state)

        for j in range(self.world_dim):
            row = ''
            for i in range(self.world_dim):
                if agent_x == i and agent_y == j:
                    if orient == 0:
                        row += ' > '
                    elif orient == 1:
                        row += ' v '
                    elif orient == 2:
                        row += ' < '
                    elif orient == 3:
                        row += ' ^ '
                elif goal_x == i and goal_y == j:
                    row += ' G '
                else:
                    row += ' - '
            rendering += row + '\n'

        rendering += '---------------'
        print(rendering)

    def set_state(self, s):
        self.state = copy.copy(s)

    def set_reward_model(self, rm):
        self.reward_model = rm

    def set_shaping(self, boolean):
        self.shaping = boolean

    def configure(self, rewards):
        self.goal_rew = rewards['goal_rew']
        self.step_pen = rewards['step_pen']
        self.turn_pen = rewards['turn_pen']

        self.config.update(rewards)

    def set_true_reward(self, rewards):
        self.true_rewards = rewards

    def random_state(self):
        return np.random.randint(self.lows, self.highs, (self.state_len,))

    def encode_state(self, state):
        return state

    def get_feedback(self, best_traj, expl_type):
        solved = False
        for t in best_traj:
            if len(t) < 50:
                solved = True
                break

        feedback = []
        if not solved:
            found = False
            for t in best_traj:
                actions = [a for s, a in t]
                start = 0
                end = start + 4
                while not found and end < len(t):
                    if sum(actions[start:end]) == 4:
                        if expl_type == 'expl':
                            found = True
                        feedback_traj = t[start:end]
                        feedback.append(('a', feedback_traj, -1, [], 4))

                    start += 1
                    end += 1

                if found:
                    break

            return feedback, True
        else:
            return [], True

    def set_lambda(self, l):
        self.lmbda = l


