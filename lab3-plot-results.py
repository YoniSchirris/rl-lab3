import pickle
import seaborn as sns; sns.set()
import time

import matplotlib.pyplot as plt

import pandas as pd


def run_fig_1():
    results = pickle.load(open('results_for_figs.pkl', 'rb'))
    test_df = pd.DataFrame(results)
    new_df = test_df[['episode', 'replay_steps1_repeats10','no-replay_steps1_repeats10' ]].copy()
    new_df.columns= ['episode', 'DQN + Experience Replay, no Target Network', 'DQN without Experience Replay, no Target Network']
    plt.figure()
    lineplot = sns.lineplot(x='episode', y='value', hue='variable', 
                data=pd.melt(new_df, ['episode']), ci=95)
    lineplot.set(ylabel='Return', xlabel='Episode')
    handles, labels = lineplot.get_legend_handles_labels()
    lineplot.legend(handles=handles[1:], labels=labels[1:], loc='upper left')
    fig = lineplot.get_figure()
    fig.savefig("yoni-fig-1-{}.png".format(int(time.time())))


def run_fig_2():
    results = pickle.load(open('results_for_figs.pkl', 'rb'))
    test_df = pd.DataFrame(results)
    new_df = test_df[['episode', 'replay_steps1_repeats10','replay_steps50_repeats10', 'replay_steps200_repeats10' ]].copy()
    new_df.columns= ['episode', 'DQN + Experience Replay, no Target Network', 'DQN + Experience Replay, Target Network 50', 'DQN + Experience Replay, Target Network 200']
    plt.figure()
    lineplot = sns.lineplot(x='episode', y='value', hue='variable', 
                data=pd.melt(new_df, ['episode']), ci=95)
    lineplot.set(ylabel='Return', xlabel='Episode')
    handles, labels = lineplot.get_legend_handles_labels()
    lineplot.legend(handles=handles[1:], labels=labels[1:], loc='upper left')
    fig = lineplot.get_figure()
    fig.savefig("yoni-fig-2-{}.png".format(int(time.time())))


def run_fig_3():
    results = pickle.load(open('results_for_figs.pkl', 'rb'))
    test_df = pd.DataFrame(results)
    new_df = test_df[['episode', 'no-replay_steps1_repeats10','no-replay_steps50_repeats10', 'no-replay_steps200_repeats10' ]].copy()
    new_df.columns= ['episode', 'DQN without Experience Replay, no Target Network', 'DQN without Experience Replay, Target Network 50', 'DQN without Experience Replay, Target Network 200']
    plt.figure()
    lineplot = sns.lineplot(x='episode', y='value', hue='variable', 
                data=pd.melt(new_df, ['episode']), ci=95)
    lineplot.set(ylabel='Return', xlabel='Episode')
    handles, labels = lineplot.get_legend_handles_labels()
    lineplot.legend(handles=handles[1:], labels=labels[1:], loc='upper left')
    fig = lineplot.get_figure()
    fig.savefig("yoni-fig-3-{}.png".format(int(time.time())))

def run_ommitted_fig_of_non_convergence():
    results = pickle.load(open('results_for_dqn_wo_replay_not_converging_over_500_episodes.pkl', 'rb'))
    test_df = pd.DataFrame(results)
    new_df = test_df[['episode', 'no-replay_steps50_repeats10']].copy()
    new_df.columns= ['episode', 'DQN without Experience Replay, Target Network 50']
    plt.figure()
    lineplot = sns.lineplot(x='episode', y='value', hue='variable', 
                data=pd.melt(new_df, ['episode']), ci=95)
    lineplot.set(ylabel='Return', xlabel='Episode')
    handles, labels = lineplot.get_legend_handles_labels()
    lineplot.legend(handles=handles[1:], labels=labels[1:], loc='upper left')
    fig = lineplot.get_figure()
    fig.savefig("yoni-fig-omit-{}.png".format(int(time.time())))

def run_q_values():
    results = pickle.load(open('qresults_1571420474.pkl', 'rb'))
    test_df = pd.DataFrame(results)
    #new_df = test_df[['episode', 'max_q_replay-replay_steps50_repeats3']].copy()
    #new_df.columns= ['episode', 'DQN without Experience Replay, Target Network 50']
    a4_dims = (11.7, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    # plt.figure()
    lineplot = sns.lineplot(ax=ax, x='episode', y='value', hue='variable', 
                data=pd.melt(test_df, ['episode']), ci=95)
    lineplot.set(ylabel='Return', xlabel='Episode')
    handles, labels = lineplot.get_legend_handles_labels()
    lineplot.legend(handles=handles[1:], labels=labels[1:], loc='upper left')
    fig = lineplot.get_figure()
    fig.savefig("q-values-fig-omit-{}.png".format(int(time.time())))


# run_fig_1()
# run_fig_2()
# run_fig_3()
# run_ommitted_fig_of_non_convergence()
run_q_values()
