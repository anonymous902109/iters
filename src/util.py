import json
import os
import random

import numpy as np
import torch


def play_episode(model, env, verbose=0):
    done = False
    obs = env.reset()
    if verbose:
        env.render()

    total_rew = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, _ = env.step(action)
        if verbose:
            env.render()
        total_rew += rew

    return total_rew


def evaluate_policy(model, env, verbose=False, n_ep=100):
    rews = []
    for i in range(n_ep):
        ep_rew = play_episode(model, env, verbose)
        rews.append(ep_rew)

    return np.mean(rews)


def load_config(config_path):
    with open(config_path) as f:
        data = json.loads(f.read())

    return data


def seed_everything(seed):
    seed_value = seed
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    # tf.random.set_seed(seed_value)
    g = torch.Generator()
    g.manual_seed(seed_value)

