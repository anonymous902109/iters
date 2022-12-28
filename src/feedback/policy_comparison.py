import copy
import torch


def get_state_importance(state, env, policyA, policyB):
    importance_A = get_action_certainty(policyA, state)
    importance_B = get_action_certainty(policyB, state)

    return max([importance_A, importance_B])


def get_traj_score(Q_A, Q_B, Q_A_s, Q_B_s, state_importance):
    traj_score = 1 - abs(Q_A - Q_B)
    state_disagreement = 1 - abs(Q_A_s - Q_B_s)

    return (traj_score > 0.8) and (state_importance > 0.5) and (state_disagreement > 0.8)


def unroll_policy(env, model, obs, action, k=10):
    env = copy.deepcopy(env)
    traj = [obs]

    env.set_state(obs)

    obs, reward, done, _ = env.step(action)

    count = 1

    while count < k and not done:
        count += 1

        action, _ = model.predict(obs, deterministic=True)
        traj.append(obs)
        obs, reward, done, _ = env.step(action)

    return traj


def predict_value(policy, x):
    tensor_obs = to_torch(x)
    state_val = max(policy.policy.q_net(tensor_obs).squeeze())

    return state_val.detach().numpy().item()


def get_Q_values(policy, state):
    tensor_obs = to_torch(state)
    Q_vals = policy.policy.q_net(tensor_obs).squeeze()

    return Q_vals


def get_action_certainty(policy, state):
    tensor_obs = to_torch(state)

    Q_vals = get_Q_values(policy, state)
    certainty = 1.0*max(torch.softmax(Q_vals, dim=-1))

    # baseline = [1.0/len(Q_vals)] * len(Q_vals)
    # sm = torch.softmax(Q_vals, dim=-1).cpu().detach().numpy()
    # certainty = 1 - entropy(sm, qk=baseline)
    return certainty


def to_torch(x):
    tensor_x = torch.Tensor(x)
    if len(tensor_x.shape) == 1:
        tensor_x = tensor_x.unsqueeze(0)

    return tensor_x


def get_simulated_Q_vals(policy, env, n_ep=1000):
    Q_vals = []
    env = copy.deepcopy(env)

    for i_ep in range(n_ep):
        obs = env.reset()
        done = False
        while not done:
            action, _ = policy.predict(obs)
            Q_val = max(get_Q_values(policy, obs))
            obs, rew, done, _ = env.step(action)

            Q_vals.append(Q_val.detach().numpy().item())

    return Q_vals
