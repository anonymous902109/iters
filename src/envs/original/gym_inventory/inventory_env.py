import gym
from gym import spaces
from gym import utils
from gym.utils import seeding
import numpy as np

import logging
logger = logging.getLogger(__name__)


class InventoryEnv(gym.Env, utils.EzPickle):
    """Inventory control with lost sales environment

    TO BE EDITED

    This environment corresponds to the version of the inventory control
    with lost sales problem described in Example 1.1 in Algorithms for
    Reinforcement Learning by Csaba Szepesvari (2010).
    https://sites.ualberta.ca/~szepesva/RLBook.html
    """

    def __init__(self, n=50, lam=10):
        self.n = n
        self.action_space = spaces.Discrete(int(n/10))
        self.observation_space = spaces.Box(0, n + 1, (1, ), dtype=np.int)
        self.max = n
        self.state = n
        self.lam = lam

        # Set seed
        self.seed()

        # Start the first round
        self.reset()

        self.max_timesteps = 14
        self.max_orders = 5

        self.config = {
            "item_cost": -1,
            "item_sale": 2,
            "hold_cost": 0,
            "loss_cost": -1,
            "delivery_cost": 0
          }

    def demand(self):
        return np.random.poisson(self.lam)

    def transition(self, x, a, d):
        m = self.max
        x = x.item()
        return max(min(x + a, m) - d, 0)

    def reward(self, x, a, y):
        x = x.item()
        m = self.max

        new_x = min(x + a, m)

        item_cost = a
        profit = min(y, new_x)
        demand_loss = max(y - new_x, 0)
        delivery_loss = (a > 0)
        hold_cost = max(new_x - y, 0)

        r = item_cost * self.config["item_cost"]\
            + profit * self.config["item_sale"]\
            + demand_loss * self.config["loss_cost"]\
            + hold_cost * self.config["hold_cost"]\

        true_reward = item_cost * self.true_rewards["item_cost"]\
            + profit * self.true_rewards["item_sale"]\
            + demand_loss * self.true_rewards["loss_cost"] \
            + hold_cost * self.true_rewards["hold_cost"]\

        return r, item_cost*self.config["item_cost"], \
               profit*self.config["item_sale"],\
               demand_loss*self.config["loss_cost"], \
               hold_cost*self.config["hold_cost"],\
               delivery_loss*self.config["delivery_cost"], \
               true_reward

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        action = action*10

        obs = self.state[0]
        demand = self.demand()

        obs2 = self.transition(obs, action, demand)
        self.state = obs2

        reward, item_cost, profit, demand_loss, hold_cost, delivery_loss, true_reward = self.reward(obs, action, demand)

        done = self.steps >= self.max_timesteps
        self.steps += 1

        info = {}

        info['rewards'] = {'item_cost': item_cost,
                           'profit': profit,
                           'demand_loss': demand_loss,
                           'delivery_loss': delivery_loss,
                           'true_rew': true_reward}

        return obs2, reward, done, info

    def reset(self):
        self.steps = 0
        return self.state

    def configure(self, rewards):
        self.config.update(rewards)

    def set_true_reward(self, rewards):
        self.true_rewards = rewards
