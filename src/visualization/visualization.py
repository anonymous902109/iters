import os
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def visualize_experiments(task_name, eval_path):
    expert_path = os.path.join(eval_path, 'expert.csv')
    model_env_path = os.path.join(eval_path, 'model_env.csv')

    expert_df = pd.read_csv(expert_path)
    model_env_df = pd.read_csv(model_env_path)

    dfs = []
    experiment_names = []

    for file_name in os.listdir(eval_path):
        file = os.path.join(eval_path, file_name)
        df = pd.read_csv(file)

        dfs.append(df)
        experiment_names.append(file_name.split('.csv')[0])

        col_names = df.columns

    for metric in col_names:
        for i, df in enumerate(dfs):
            sns.lineplot(data=df, x="iter", y=metric, label=experiment_names[i])

        sns.lineplot(data=expert_df, x='iter', y=metric)
        sns.lineplot(data=model_env_df, x='iter', y=metric)

        plt.title(task_name)
        plt.legend()
        plt.show()

def visualize_feature(traj, feature_id, plot_actions=False, title=''):
    feature_vals = []
    for t in traj:
        ep_vals = [p[0].flatten()[feature_id] for p in t]
        feature_vals.append(ep_vals)

    for f_vals in feature_vals:
        plt.plot(f_vals)

    plt.title(title)
    plt.xlabel('Time step')
    plt.ylabel('Agent\'s lane')

    plt.show()

    if plot_actions:
        actions = []
        for t in traj:
            ep_actions = [p[1] for p in t]
            actions.append(ep_actions)

        for a_vals in actions:
            plt.plot(a_vals)

        plt.title('Action distribution through an episode across successful trajectories')
        plt.xlabel('Time step')
        plt.ylabel('Action')
        plt.show()


def visualize_rewards(rew_dict, title='', xticks=None):
    for rew_name, rew_values in rew_dict.items():
        plt.plot(rew_values)

        plt.title(rew_name)

        if xticks is not None:
            plt.xticks = xticks

        plt.xlabel('Time steps')
        plt.ylabel('Average reward')

        plt.show()

def visualize_best_experiment(path, expert_path, model_env_path, task_name, title):
    df = pd.read_csv(path, header=0)

    expert_df = pd.read_csv(expert_path)
    model_env_df = pd.read_csv(model_env_path)

    expert_end_vals = expert_df.iloc[-1:]
    baseline_end_vals = model_env_df.iloc[-1:]

    pal = sns.color_palette('Set2')

    for i, metric in enumerate(expert_df.columns):
        expert_df[metric] = expert_end_vals[metric].values[0]
        model_env_df[metric] = baseline_end_vals[metric].values[0]

        y_label = r'$R_{true}$' if metric == 'True reward' else metric

        sns.lineplot(df, x="Iteration", y=metric, hue="lmbda", palette=pal)
        sns.lineplot(data=expert_df, x='Iteration', y=metric, label=r'$M_{true}$')
        sns.lineplot(data=model_env_df, x='Iteration', y=metric, label=r'$M_{env}$')

        plt.title(title)
        plt.ylabel(y_label)
        plt.legend(loc='upper left')
        plt.show()

def get_cummulative_feedback(feedback):
    cm_feedback = []
    for i, f in enumerate(feedback):
        cm_feedback.append(sum(feedback[0:(i+1)]))

    return cm_feedback


def visualize_best_vs_rand_summary(best_summary_path, rand_summary_path, lmbdas, task_name, title):
    df_best = pd.read_csv(best_summary_path, header=0)
    df_rand = pd.read_csv(rand_summary_path, header=0)

    dfs = [df_best, df_rand]
    seeds = df_best['seed'].unique()
    lmbdas = df_best['lmbda'].unique()

    for i, df in enumerate(dfs):
        partials = []
        print('-----------------------------------------')
        for s in seeds:
            for l in lmbdas:
                partial_df = df[(df['seed'] == s) & (df['lmbda'] == l)]
                feedback = partial_df['feedback'].values
                partial_df = partial_df.assign(cummulative_feedback=get_cummulative_feedback(feedback))
                partials.append(partial_df)

                print('Seed = {} Lambda = {} Total feedback = {}'.format(s, l, sum(partial_df['feedback'])))

        dfs[i] = pd.concat(partials)

    df_best, df_rand = dfs

    for l in lmbdas:
        # for metric in df_best.columns:
        #     y_label = r'$R_{true}$' if metric == 'True reward' else metric
        #
        #     sns.lineplot(data=df_best[df_best['lmbda'] == l], x='Iteration', y=metric, label='Best summary')
        #     sns.lineplot(data=df_rand[df_rand['lmbda'] == l], x='Iteration', y=metric, label='Random summary')
        #
        #     title_tmp = title + ', \u03BB = {}'.format(l)
        #
        #     plt.title(title_tmp)
        #     plt.ylabel(y_label)
        #     plt.legend(loc='upper left')
        #     plt.show()

        print('')

        sns.lineplot(data=df_best[df_best['lmbda'] == l],
                     x='Iteration',
                     y='cummulative_feedback',
                     label='Best summary')
        sns.lineplot(data=df_rand[df_rand['lmbda'] == l],
                     x='Iteration',
                     y='cummulative_feedback',
                     label='Random summary')

        title = 'Feedback for different summaries lambda = {}'.format(l)

        plt.title(title)
        plt.ylabel('Feedback')
        plt.legend()
        plt.show()

def visualize_feedback(expl_path, no_expl_path, lmbdas, task_name, title):
    df_expl = pd.read_csv(expl_path, header=0)
    df_no_expl = pd.read_csv(no_expl_path, header=0)

    for l in lmbdas:
        for metric in df_expl.columns():
            sns.lineplot(data=df_expl[df_expl['lmbda'] == l], x='iter', y='cummulative_feedback', label='best summary')
            sns.lineplot(data=df_no_expl[df_no_expl['lmbda'] == l], x='iter', y='cummulative_feedback', label='random summary')

            title += ' lambda = {}'.format(l)

            plt.title(task_name)
            plt.legend()
            plt.show()