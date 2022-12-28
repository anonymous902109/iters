import copy
import os

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.data import TensorDataset
from tqdm import tqdm

from src.evaluation.evaluator import Evaluator
from src.feedback.feedback_processing import encode_trajectory, present_successful_traj
from src.visualization.visualization import visualize_feature


def check_dtype(env):
    state_dtype = env.state_dtype
    action_dtype = env.action_dtype

    return state_dtype, action_dtype


def init_replay_buffer(env, model, time_window, n_episodes=1000, expl_type='expl'):
    print('Initializing replay buffer with env reward...')
    D = []
    n_episodes = n_episodes if expl_type == 'expl' else int(n_episodes / 10)
    for i in tqdm(range(n_episodes)):
        done = False
        obs = env.reset()
        while not done:
            if model is None:
                action = np.random.randint(0, env.action_space.n, size=(1,)).item()
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, rew, done, _ = env.step(action)

            past = env.episode
            curr = 1
            for j in range(len(past)-1, -1, -1):
                enc = encode_trajectory(past[j:], obs, curr, time_window, env)

                D.append(enc)

                if curr >= time_window:
                    break

                curr += 1

    D = torch.tensor(np.vstack(D))
    D = torch.unique(D, dim=0)  # remove duplicates

    y_D = np.zeros((len(D), ))
    y_D = torch.tensor(y_D)

    dataset = TensorDataset(D, y_D)
    print('Generated {} env samples for dataset'.format(len(D)))

    return dataset


def train_expert_model(env, env_config, model_config, expert_path, eval_path, feedback_freq, max_iter, timesteps=int(1e5)):
    orig_config = copy.copy(env.config)
    rewards = env_config['true_reward_func']
    env.configure(rewards)

    try:
        model = DQN.load(expert_path, seed=1, env=env)
        print('Loaded expert.csv')
    except FileNotFoundError:
        print('Training expert.csv...')
        expert_eval_path = os.path.join(eval_path, 'expert.csv')
        callback = CustomEvalCallback(feedback_freq, env, expert_eval_path)

        model = DQN('MlpPolicy', env, **model_config)
        model.learn(total_timesteps=feedback_freq*max_iter, callback=callback)

        model.save(expert_path)

    best_traj = present_successful_traj(model, env, n_traj=10)
    visualize_feature(best_traj, 0, plot_actions=True, title='Model trained on the true reward')

    # reset original config
    env.configure(orig_config)
    return model


def check_is_unique(unique_feedback, feedback_traj, timesteps, time_window, env, important_features,expl_type):
    if expl_type == 'no_expl':
        return True

    unique = True
    threshold = 0.05

    for f, imp_f, ts in unique_feedback:
        if (imp_f == important_features) and (len(f) == len(feedback_traj)):

            enc_1 = encode_trajectory(feedback_traj, None, timesteps, time_window, env)
            enc_2 = encode_trajectory(f, None, ts, time_window, env)

            distance = np.mean(abs(enc_1[imp_f] - enc_2[important_features]))
            if distance < threshold:
                unique = False
                break

    return unique


def train_model(env, model_config, path, eval_path, feedback_freq, max_iter):
    try:
        model = DQN.load(path, seed=1, env=env)
        print('Loaded initial model')
    except FileNotFoundError:
        expert_eval_path = os.path.join(eval_path, 'model_env.csv')
        callback = CustomEvalCallback(feedback_freq, env, expert_eval_path)

        print('Training initial model...')
        model = DQN('MlpPolicy', env, **model_config)
        model.learn(total_timesteps=feedback_freq*max_iter, callback=callback)

        model.save(path)

    evaluator = Evaluator()
    avg_mo = evaluator.evaluate(model, env, path=os.path.join(eval_path, 'model_env.csv'), seed=0, write=True)
    print('Mean reward for objectives = {} for initial model = {}'.format(env.config, avg_mo))

    best_traj = present_successful_traj(model, env, n_traj=10)
    visualize_feature(best_traj, 0, plot_actions=True, title='Model trained with environment reward')

    return model


class CustomEvalCallback(BaseCallback):

    def __init__(self, timesteps, env, eval_path):
        super(CustomEvalCallback, self).__init__(0)

        self.eval_freq = timesteps
        self.env = env
        self.eval_path = eval_path

        self.evaluator = Evaluator()

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        if self.num_timesteps % self.eval_freq == 0:
            avg_mo = self.evaluator.evaluate(self.model, self.env, path=self.eval_path, seed=0, write=False)
        return True

    def _on_training_end(self) -> None:
        self.evaluator.evaluate(self.model, self.env, path=self.eval_path, seed=0, write=True)
