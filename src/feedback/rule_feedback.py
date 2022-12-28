import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from src.feedback.feedback_processing import encode_trajectory
from src.feedback.policy_comparison import predict_value, get_state_importance, get_traj_score, unroll_policy, \
    get_simulated_Q_vals


def give_rule_feedback(model_A, model_B, env):
    dataset = []
    n_ep = 1000

    scaler_A = MinMaxScaler(feature_range=[0, 1])
    scaler_B = MinMaxScaler(feature_range=[0, 1])

    Q_A_simulated = get_simulated_Q_vals(model_A, env)
    Q_B_simulated = get_simulated_Q_vals(model_B, env)

    scaler_A.fit([[min(Q_A_simulated)], [max(Q_A_simulated)]])
    scaler_B.fit([[min(Q_B_simulated)], [max(Q_B_simulated)]])

    for i in range(n_ep):
        obs = env.reset()

        done = False
        while not done:
            action_A, _ = model_A.predict(obs, deterministic=True)
            action_B, _ = model_B.predict(obs, deterministic=True)

            episode_len = len(env.episode[-(env.time_window):])
            t = encode_trajectory(env.episode[-(env.time_window):], obs, episode_len, env.time_window, env)

            diff = 0
            if action_A != action_B:
                t_A = unroll_policy(env, model_A, obs, action_A, k=5)
                t_B = unroll_policy(env, model_B, obs, action_B, k=5)

                outcome_A = t_A[-1]
                outcome_B = t_B[-1]

                Q_A = predict_value(model_A, outcome_A)
                Q_B = predict_value(model_B, outcome_B)

                Q_A_s = predict_value(model_A, obs)
                Q_B_s = predict_value(model_B, obs)

                Q_A = scaler_A.transform([[Q_A]]).item()
                Q_B = scaler_B.transform([[Q_B]]).item()

                Q_A_s = scaler_A.transform([[Q_A_s]]).item()
                Q_B_s = scaler_B.transform([[Q_B_s]]).item()

                state_importance = get_state_importance(obs, env, model_A, model_B)

                score = get_traj_score(Q_A, Q_B, Q_A_s, Q_B_s, state_importance)

                # if score:
                diff = 1

            record = list(t[-(env.time_window * (env.state_len+1) + 1):-2]) + [episode_len] + [diff]

            dataset.append(record)

            obs, rew, done, _ = env.step(action_A)

    df = pd.DataFrame(dataset, columns=env.feature_names + ['ep_len', 'diff'])

    print('Training decision tree on {} samples'.format(len(df)))

    clf = DecisionTreeClassifier(max_depth=10, min_samples_leaf=1000)

    # Train Decision Tree Classifer
    train_cols = [c for c in df.columns if (c != 'diff' and c.startswith('action'))] + ['ep_len']
    clf = clf.fit(df[train_cols], df['diff'])

    text_representation = tree.export_text(clf, feature_names=train_cols)
    print(text_representation)


