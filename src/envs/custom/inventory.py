import numpy as np

from src.envs.original.gym_inventory.inventory_env import InventoryEnv
from src.feedback.feedback_processing import encode_trajectory


class Inventory(InventoryEnv):

    def __init__(self, time_window, shaping=False):
        super().__init__()
        self.shaping = shaping
        self.time_window = time_window

        self.episode = []

        self.state_len = 1
        self.lows = np.zeros((1, 0))
        self.highs = np.ones((1, 0))
        self.highs.fill(self.n)

        self.lmbda = 0.2

        self.immutable_features = []
        self.discrete_features = [0, 1, 2, 3, 4]
        self.cont_features = []

        self.state_dtype = 'int'
        self.action_dtype = 'int'

        self.lows = [0]
        self.highs = [self.n]

        self.feature_names = ['stock', 'action']
        self.feature_names = [fn + '_{}'.format(i) for fn in self.feature_names for i in
                              range(self.time_window - 1)] + self.feature_names[:-1]

    def step(self, action):
        self.episode.append((self.state.flatten(), action))

        self.state, rew, done, info = super().step(action)

        self.state = np.array([self.state]).flatten()

        if self.shaping:
            shaped_rew = self.augment_reward(action, self.state.flatten())
            rew += shaped_rew

        true_rew = info['rewards']['true_rew']
        orders = [a for s, a in self.episode]
        freq_orders = sum(np.array(orders[-self.time_window:]) > 0) > self.max_orders
        true_rew += self.true_rewards['delivery_cost'] * freq_orders
        info['rewards']['true_rew'] = true_rew
        info['rewards']['freq_orders'] = freq_orders

        rew += self.config['delivery_cost'] * freq_orders

        return self.state, rew, done, info

    def reset(self):
        self.episode = []
        self.state = np.array([super().reset()]).flatten()
        return self.state

    def close(self):
        pass

    def render(self):
        print('Obs: {}'.format(self.obs))

    def augment_reward(self, action, state):
        running_rew = 0
        past = self.episode
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

    def render_state(self, state):
        print('Inventory: {}'.format(state))

    def configure(self, rewards):
        super().configure(rewards)

    def set_true_reward(self, rewards):
        super().set_true_reward(rewards)

    def random_state(self):
        return np.random.randint(self.lows, self.highs, (self.state_len,))

    def encode_state(self, state):
        return state

    def get_feedback(self, best_traj, expl_type):
        start = 0
        end = start + self.time_window

        feedback_list = []

        for i, t in enumerate(best_traj):
            actions = [a for s, a in t]

            while end < len(t):

                orders = sum(np.array(actions[start:end]) > 0)
                if orders > self.max_orders:
                    print('Trajectory id = {} Start = {} End = {}'.format(i, start, end))
                    feedback = ('a', t[start:end], -1, ['count(a>0)>{}'.format(self.max_orders)], self.time_window)
                    feedback_list.append(feedback)

                    if expl_type == 'expl':
                        return feedback_list, True

                start = start + 1
                end = start + self.time_window

        return feedback_list, True


    def set_lambda(self, l):
        self.lmbda = l