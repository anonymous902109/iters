import numpy as np
import torch
from dtw import dtw
import re
from torch.utils.data import TensorDataset


def present_successful_traj(model, env, summary_type='best_summary', n_traj=10):
    # gather trajectories
    print('Gathering successful trajectories for partially trained model...')
    traj_buffer, rews = gather_trajectories(model, env, 50)

    # filter trajectories
    if summary_type == 'best_summary':
        top_indices = np.argsort(rews)[-n_traj:]
        filtered_traj = [traj_buffer[i] for i in top_indices]
        top_rews = [rews[i] for i in top_indices]
    elif summary_type == 'rand_summary':
        indices = np.random.choice(len(traj_buffer), n_traj)
        filtered_traj = [traj_buffer[i] for i in indices]

    # # play filtered trajectories
    # for j, t in enumerate(filtered_traj):
    #     print('------------------\n Trajectory {} \n------------------\n'.format(j))
    #     print('Trajectory reward = {}'.format(top_rews[j]))
    #     play_trajectory(env, t)

    return filtered_traj


def play_trajectory(env, traj):
    for i, (s, a) in enumerate(traj):
        print('------------------\n Timestep = {}'.format(i))
        env.render_state(s)
        print('Action = {}\n------------------\n '.format(a))


def gather_trajectories(model, env, n_traj):
    traj_buffer = []
    rews = []

    for i in range(n_traj):
        traj, rew = get_ep_traj(model, env)
        traj_buffer.append(traj)
        rews.append(rew)

    return traj_buffer, rews


def get_ep_traj(model, env):
    done = False
    obs = env.reset()
    traj = []

    total_rew = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        traj.append((obs, action))

        obs, rew, done, _ = env.step(action)
        total_rew += rew

    return traj, total_rew


def get_input(best_traj):
    print('Gathering user feedback')
    done = False
    feedback_list = []

    while not done:
        print('Input feedback type (s, a, none or done)')
        feedback_type = input()
        if feedback_type == 'done':
            return [], False

        if feedback_type == 'none':
            return [], True

        print('Input trajectory number:')
        traj_id = int(input())
        print('Enter starting timestep:')
        start_timestep = int(input())
        print('Enter ending timestep:')
        end_timestep = int(input())
        print('Enter feedback:')
        feedback_signal = int(input())

        f = best_traj[traj_id][start_timestep:(end_timestep + 1)]  # for inclusivity + 1

        timesteps = end_timestep - start_timestep + 1

        print('Enter ids of important features separated by space:')
        important_features = input()
        try:
            important_features = [int(x) for x in important_features.split(' ')]
        except ValueError:
            important_features = []

        feedback = (feedback_type, f, feedback_signal, important_features, timesteps)

        feedback_list.append(feedback)

        print('Enter another trajectory (y/n?)')
        cont = input()
        if cont == 'y':
            done = False
        else:
            done = True

    return feedback_list, done


def gather_feedback(best_traj, time_window, env, disruptive=False, noisy=False, prob=0, expl_type='expl', auto=False):
    if auto:
        feedback_list, cont = env.get_feedback(best_traj, expl_type=expl_type)
    else:
        feedback_list, cont = get_input(best_traj)

    if noisy:
        feedback_list = noise(feedback_list, best_traj, env, time_window, prob)
    elif disruptive:
        disrupted_feedback_list = []
        for f in feedback_list:
            disrupted_f = disrupt(f, prob)
            disrupted_feedback_list.append(disrupted_f)

            return disrupted_feedback_list, True

    return feedback_list, cont


def noise(feedback_list, best_traj, env, time_window, prob):
    add_noisy_sample = np.random.choice([0, 1], p=[1-prob, prob])

    if add_noisy_sample:
        state_features_len = env.state_len

        rand_traj = np.random.randint(0, len(best_traj))
        f_rand_traj = best_traj[rand_traj]

        rand_start = np.random.randint(0, len(f_rand_traj))
        rand_len = np.random.randint(1, time_window)

        f_rand_traj = f_rand_traj[rand_start: (rand_start + rand_len)]

        rand_f_type = np.random.randint(0, 2)
        rand_f_type = 's' if rand_f_type else 'a'

        rand_f_signal = np.random.choice((-1, +1))

        timesteps = rand_len

        important_features = []
        if rand_f_type == 's':
            important_features = [np.random.randint(0, state_features_len)]

        feedback_list.append((rand_f_type, f_rand_traj, rand_f_signal, important_features, timesteps))

    return feedback_list


def disrupt(feedback, prob):
    feedback_type, f, feedback_signal, important_features, timesteps = feedback

    disrupt_sample = np.random.choice([0, 1], p=[1-prob, prob])
    if disrupt_sample:
        feedback_signal = -feedback_signal
        return (feedback_type, f, feedback_signal, important_features, timesteps)
    else:
        return feedback


def augment_feedback_diff(traj, signal, important_features, rules, timesteps, env, time_window, actions, datatype, expl_type='expl', length=100):
    print('Augmenting feedback...')
    if expl_type == 'expl':
        state_dtype, action_dtype = datatype
        state_len = env.state_len

        traj_len = len(traj)
        traj_enc = encode_trajectory(traj, state=None, timesteps=timesteps, time_window=time_window, env=env)
        enc_len = traj_enc.shape[0]

        immutable_features = [im_f + (state_len * i) for i in range(traj_len) for im_f in env.immutable_features]

        important_features += immutable_features

        # generate mask to preserve important features
        random_mask = np.ones((length, enc_len))
        random_mask[:, important_features] = 0
        inverse_random_mask = 1 - random_mask

        D = np.tile(traj_enc, (length, 1))

        # add noise to important features if they are continuous
        if state_dtype != 'int':
            # adding noise for continuous state features
            D[:, env.cont_features] = D[:, env.cont_features] + np.random.normal(0, 0.001, (length, len(env.cont_features)))

        # observation limits
        lows = list(np.tile(env.lows, (time_window + 1, 1)).flatten())
        highs = list(np.tile(env.highs, (time_window + 1, 1)).flatten())

        # action limits
        lows += [0] * time_window
        highs += [env.action_space.n] * time_window

        # timesteps limits
        lows += [1]
        highs += [time_window]

        # generate matrix of random values within allowed ranges
        if state_dtype == 'int' or (actions and action_dtype == 'int'):
            rand_D = np.random.randint(lows, highs, size=(length, enc_len))
        else:
            rand_D = np.random.uniform(lows, highs, size=(length, enc_len))

        if actions and action_dtype == 'cont':
            rand_D[:, (time_window*state_len+1):-1] = augment_actions(traj, length)

        if not actions:
            # randomize actions
            rand_D[:, -(time_window+1):] = np.random.randint(lows[-(time_window+1):], highs[-(time_window+1):], (length, time_window+1))

        D = np.multiply(rand_D, random_mask) + np.multiply(inverse_random_mask, D)

        for rule in rules:
            D, _ = satisfy(D, rule, time_window)

    else:
        traj_enc = encode_trajectory(traj, state=None, timesteps=timesteps, time_window=time_window, env=env)
        D = np.tile(traj_enc, (1, 1))

    # reward for feedback the signal
    D = torch.tensor(D)
    D = torch.unique(D, dim=0)

    y = np.zeros((len(D),))
    y.fill(signal)
    y = torch.tensor(y)

    dataset = TensorDataset(D, y)

    print('Generated {} augmented samples'.format(len(dataset)))
    return dataset


def satisfy(D, r, time_window):
    if r['quant'] == 'a':
        start = -(time_window+1)

        if r['limit_sign'] == '>':
            satisfies = np.sum(D[:, start:-1] > r['filter_num'], axis=1) > r['limit']
        elif r['limit_sign'] == '<':
            satisfies = np.sum(D[:, start:-1] > r['filter_num'], axis=1) < r['limit']
        elif r['limit_sign'] == '>=':
            satisfies = np.sum(D[:, start:-1] > r['filter_num'], axis=1) >= r['limit']
        elif r['limit_sign'] == '<=':
            satisfies = np.sum(D[:, start:-1] > r['filter_num'], axis=1) <= r['limit']

        max_index = torch.argmax(torch.tensor(satisfies, dtype=int), dim=-1).item()
        max_value = satisfies[max_index]

        if max_value > 0:
            return D[satisfies], [max_index]
        else:
            return D[satisfies], []


def decode_rule(rule):
    ''' Decode the rule from string to a dict form '''
    agg = rule.split('(')[0]
    term = rule.split('(')[1].split(')')[0]

    filter = re.findall("<=|>=|<|>|=", term)[0]
    filter_num = int(term.split(filter)[1])
    quant = term.split(filter)[0]

    limit_sign = re.findall("<=|>=|<|>|=", rule)[1]
    limit = int(rule.split(limit_sign)[-1])

    decoded = {
        'agg': agg,
        'quant': quant,
        'filter': filter,
        'filter_num': filter_num,
        'limit_sign': limit_sign,
        'limit': limit
    }

    return decoded

def augment_actions(feedback_traj, length=2000):
    actions = [a for (s, a) in feedback_traj]
    traj_len = len(actions)

    # generate neighbourhood of sequences of actions
    random_traj = np.tile(actions, (length*100, 1))

    random_traj = np.round(random_traj + np.random.normal(0, 5, size=(length*100, traj_len)))
    random_traj[random_traj<0] = 0

    # find similarities to the original trajectory actions using dynamic time warping
    sims = [dtw(actions, traj, keep_internals=True).normalizedDistance for traj in random_traj]
    top_indices = np.argsort(sims)[:length]

    # filter out only the most similar trajectories
    filtered_traj = random_traj[top_indices]

    D = torch.tensor(filtered_traj)

    return D


def encode_trajectory(traj, state, timesteps, time_window, env):
    states = []
    actions = []

    assert len(traj) <= time_window

    curr = 1
    for s, a in traj:
        states += list(env.encode_state(s))
        actions.append(a)

        curr += 1

    if state is None:
        states += list(env.random_state())
    else:
        states += list(env.encode_state(state))

    # add the last state to fill up the trajectory encoding
    # this will be randomized so it does not matter
    while curr <= time_window:
        states += list(env.random_state())
        actions.append(env.action_space.sample())
        curr += 1

    enc = states + actions + [timesteps]
    enc = np.array(enc)

    return enc


def generate_important_features(important_features, state_len, feedback_type, time_window, feedback_traj):
    actions = feedback_type == 'a'

    imf = []
    rules = []

    for i in important_features:
        if isinstance(i, int):
            imf.append(i)

        if isinstance(i, str):
            rules.append(decode_rule(i))

    traj_len = len(feedback_traj)
    imf += [(time_window + 1) * state_len + time_window]  # add timesteps as important

    if actions and len(rules) == 0:
        imf += list(np.arange((time_window + 1) * state_len, (time_window + 1) * state_len + traj_len))

    print('Rules = {}'.format(rules))
    return imf, actions, rules
