import copy
from os.path import exists

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from stable_baselines3 import DQN

from src.feedback.feedback_processing import present_successful_traj
from src.visualization.visualization import visualize_rewards, visualize_feature


class Evaluator:

    def __init__(self, expert_model=None,  feedback_freq=10000, env=None):
        self.feedback_freq = feedback_freq
        self.env = env
        self.reward_dict = None
        self.similarities = []

        self.expert_model = expert_model

    def evaluate(self, model, env, feedback_size=0, path=None, seed=None, lmbda=0.2, write=False):
        # Evaluate multiple objectives
        rew_values = self.evaluate_MO(model, env, n_episodes=100)
        if self.reward_dict is None:
            self.reward_dict = rew_values
            self.reward_dict['feedback'] = [feedback_size]
        else:
            new_feedback = self.reward_dict['feedback'] + [feedback_size]
            self.reward_dict = {rn: self.reward_dict[rn] + rew_values[rn] for rn in rew_values.keys()}

            self.reward_dict['feedback'] = new_feedback

        if write:
            self.write_csv(self.reward_dict, path, seed, lmbda)

        print('Rewards: {}'.format(self.reward_dict))

    def visualize(self, iteration):
        # visualize the effect of shaping on objectives
        xs = np.arange(0, self.feedback_freq * iteration, step=self.feedback_freq)
        visualize_rewards(self.reward_dict, title='Average reward objectives with reward shaping', xticks=xs)

        plt.show()

    def evaluate_MO(self, model, env, n_episodes=10):
        # estimate number of objectives
        env.reset()
        _, _, _, info = env.step(env.action_space.sample())

        objectives = info['rewards']
        reward_names = [obj_n for obj_n, obj_val in objectives.items()]
        num_objectives = len(info['rewards'])
        ep_average = {rn: 0.0 for rn in reward_names}

        for ep in range(n_episodes):
            rewards = {rn: 0.0 for rn in reward_names}

            done = False
            obs = env.reset()
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, info = env.step(action)

                step_rewards = info['rewards']
                rewards = {rn: rewards[rn] + step_rewards[rn] for rn in rewards.keys()}

            ep_average = {rn: ep_average[rn] + rewards[rn] for rn in ep_average.keys()}

        ep_average = {rn: [ep_average[rn] / n_episodes] for rn in ep_average.keys()}

        return ep_average

    def evaluate_similarity(self, model_A, model_B, env):
        actions_A = []
        actions_B = []

        for i in range(10):
            done = False
            obs = env.reset()
            while not done:
                action_A, _ = model_A.predict(obs, deterministic=True)
                action_B, _ = model_B.predict(obs, deterministic=True)

                obs, rew, done, _ = env.step(action_A)

                actions_A.append(action_A)
                actions_B.append(action_B)

        sim = sum(np.array(actions_A) == np.array(actions_B)) / len(actions_A)

        return sim

    def get_rewards_dict(self):
        return self.reward_dict

    def reset_reward_dict(self):
        self.reward_dict = None
        self.similarities = []

    def write_csv(self, rew_dict, path, seed, lmbda):
        df = pd.DataFrame.from_dict(rew_dict)
        df['seed'] = seed
        df['lmbda'] = lmbda
        df['iter'] = np.arange(1, len(df) + 1, step=1)
        header = not exists(path)
        df.to_csv(path, mode='a', header=header)

