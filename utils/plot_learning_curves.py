"""
Run an evaluation script on the saved models to get average performance
"""

import os
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import scipy

sns.set(style="darkgrid")
sns.set_context("paper")

cols_deep = sns.color_palette("deep", 10)
cols_bright = sns.color_palette("bright", 10)
cols_dark = sns.color_palette("dark", 10)


def moving_average(array, num_points, only_past=False):

    if not only_past:
        ma = []
        for i in range(num_points, array.shape[0]):
            ma.append(np.mean(array[i - num_points:i]))
        ma = np.array(ma)
    else:
        ma = []
        for i in range(1, array.shape[0]+1):
            ma.append(np.mean(array[max(0, i-num_points):i]))
        ma = np.array(ma)

    return ma


def get_array_from_event(event_path, tag, m):
    arr = []
    steps = []
    try:
        for event in summary_iterator(event_path):
            if hasattr(event.summary, 'value') and len(event.summary.value) > 0:
                if event.summary.value[0].tag == tag:
                    arr.append(event.summary.value[0].simple_value)
                    steps.append(event.step)
    except:
        pass

    steps = np.array(steps)
    arr = moving_average(np.array(arr), m, only_past=True)
    return arr, steps


def get_array_from_event_multi_episode(event_path, tag, rollout_indices, m):

    num_rollouts = len(rollout_indices)
    r1 = [[] for _ in range(num_rollouts)]
    steps = []

    try:
        for event in summary_iterator(event_path):
            if hasattr(event.summary, 'value') and len(event.summary.value) > 0:
                for i, n in enumerate(rollout_indices):
                    if event.summary.value[0].tag == tag + '{}'.format(n):
                        r1[i].append(event.summary.value[0].simple_value)
                        if i == 0:
                            steps.append(event.step)
    except:
        pass

    if len(np.unique([len(r) for r in r1])) > 1:
        print('warning: different lengths found')
    min_len = min([len(r) for r in r1])
    arr = np.array([np.array(r)[:min_len] for r in r1]).sum(axis=0)  # sum over all rollouts
    steps = np.array(steps)

    arr = moving_average(arr, m, only_past=True)

    return arr, steps


###################################################################################


def plot_learning_curve(x, y, label, mode='std', **kwargs):
    """
    Takes as input an x-value (number of frames)
    and a matrix of y-values (rows: runs, columns: results)
    """

    y = y[:, :len(x)]

    # get the mean (only where we have data) and compute moving average
    mean = np.sum(y, axis=0) / (np.sum(y != 0, axis=0) + 1e-6)

    y_std = np.std(y, axis=0)
    y_se = scipy.stats.sem(y, axis=0)
    y_cfi = scipy.stats.norm.interval(0.95, loc=mean, scale=y_se)

    p = plt.plot(x, mean, linewidth=2, label=label,
                 c=kwargs['color'] if 'color' in kwargs else cols_deep[0])
    plt.gca().fill_between(x, y_cfi[0], y_cfi[1], facecolor=p[0].get_color(), alpha=0.2)
    # if mode == 'std':
    #     # compute standard deviation
    #     std = np.std(y, axis=0)
    #     # compute confidence intervals
    #     cis = [mean - std, mean + std]

    #     plt.gca().fill_between(x, cis[0], cis[1], facecolor=p[0].get_color(), alpha=0.5)
    # elif mode == 'all':
        # plt.plot(x, y.T, linewidth=2, alpha=0.3, c=p[0].get_color())
    # else:
    #     raise NotImplementedError


def plot_tb_results(env_name, exp_name, tag, m, **kwargs):
    """

    :param env_name:            name of the environment
    :param exp_name:            in env_name folder, which experiment
    :param m:                   parameter for temporally smoothing the curve
    :return:
    """

    # results_directory = os.path.join(os.getcwd(), '../logs/{}'.format(env_name)) #pycharm
    results_directory = os.path.join(os.getcwd(), 'logs_result/{}'.format(env_name)) #vscode
    exp_ids = [folder for folder in os.listdir(results_directory) if
                   folder.startswith(exp_name + '__')]

    arrays = []
    for exp_id in exp_ids:
        exp_dir = os.path.join(results_directory, exp_id)
        tf_event = [event for event in os.listdir(exp_dir) if event.startswith('event')][0]

        if kwargs['multi_episode'] == True:
            arr, steps = get_array_from_event_multi_episode(os.path.join(exp_dir, tf_event), tag=tag,
                                                            rollout_indices=kwargs['rollout_indices'], m=m)
        else:
            arr, steps = get_array_from_event(os.path.join(exp_dir, tf_event), tag=tag,m=m)

        arrays.append(arr)

    arr_lens = np.array([len(array) for array in arrays])
    if len(np.unique(arr_lens)) > 1:
        min_len = min(arr_lens)
        arrays = [array[:min_len] for array in arrays]
        steps = steps[:min_len]
    arrays = np.vstack(arrays)

    # plot_learning_curve(steps, arrays, label=kwargs['label'] if 'label' in kwargs else exp_name,mode='std',
    #                     color=kwargs['color'])
    x=steps
    y=arrays
    y = y[:, :len(x)]

    # get the mean (only where we have data) and compute moving average
    mean = np.sum(y, axis=0) / (np.sum(y != 0, axis=0) + 1e-6)
    # print("_____",type(y))
    # print(x)
    # y_std = np.std(y, axis=0)
    # y_se = scipy.stats.sem(y, axis=0)
    # y_cfi = scipy.stats.norm.interval(0.95, loc=mean, scale=y_se)
    # print(y_se)
    # print(y_cfi)
    print(y)
    std = np.std(y, axis=1)
    cis = [mean - std, mean + std]
    print(std)
    # print(cis)
    # plt.gca().fill_between(x, cis[0], cis[1], facecolor=p[0].get_color(), alpha=0.5)

    p = plt.plot(x, mean, linewidth=2, label=kwargs['label'] if 'label' in kwargs else exp_name,
                 c=kwargs['color'] if 'color' in kwargs else cols_deep[0])
    # plt.gca().fill_between(x, y_cfi[0], y_cfi[1], facecolor=p[0].get_color(), alpha=0.2)                    
    plt.fill_between(x, cis[0], cis[1], facecolor=p[0].get_color(), alpha=0.5)

def compare(env_names, exp_names, tags, m, ylabel, save_path=None, **kwargs):
    for i in range(len(env_names)):
        plot_tb_results(env_names[i], exp_names[i], tags[i], m,
                        multi_episode=kwargs['multi_episode'][i],
                        rollout_indices=kwargs['rollout_indices'][i],
                        label=kwargs['labels'][i], mode='std',color=kwargs['colors'][i])

    plt.ylabel(ylabel, fontsize=20)
    plt.xlabel('Frames', fontsize=15)
    # plt.xlabel('Train Steps', fontsize=15)
    # plt.title('Gridworld', fontsize=20)
    plt.title(env_names[0], fontsize=20)
    from matplotlib.ticker import ScalarFormatter
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    tx = plt.gca().xaxis.get_offset_text()
    tx.set_fontsize(15)

    if 'truncate_at' in kwargs:
        plt.xlim([0., kwargs['truncate_at']])


    # plt.legend(fontsize=18, loc='lower right', prop={'size': 16})
    plt.legend(fontsize=18, loc='lower right', prop={'size': 15})
    plt.tight_layout()
    plt.gca().tick_params(axis='both', which='major', labelsize=15)
    plt.gca().tick_params(axis='both', which='minor', labelsize=15)
    plt.subplots_adjust(left=0.16) 
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == '__main__':
    # compare(env_names=['PointRobotSparse-v0', 'SparsePointEnv-v0'], exp_names=['sac', 'ppo_latest'],
    #         tags=['returns_multi_episode/sum_eval', 'return_avg_per_frame/episode_'], m=10,
    #         ylabel='Average return', multi_episode=[False, True],
    #         rollout_indices=[[], [1, 2]], labels=['Ours', 'VariBAD'],
    #         colors=[cols_dark[0], cols_dark[1]])

    # plot_tb_results('HalfCheetahVel-v0','sac','returns_multi_episode/sum_eval',m=10)
    # plt.show()

    # compare(env_names=['HalfCheetahVel-v0'], exp_names=['sac'],
    #         tags=['returns_multi_episode/sum_eval'], m=10,
    #         ylabel='Average return', save_path="./",multi_episode=[False],
    #         rollout_indices=[[]], labels=['with state decoder',],
    #         colors=[cols_dark[0]])


    # compare(env_names=['AntSemiCircleSparse-v0-bisimu','AntSemiCircleSparse-v0-wo-state','AntSemiCircleSparse-v0-w-state'], exp_names=['sac','sac','sac'],
    #         tags=['returns_multi_episode/sum_eval','returns_multi_episode/sum_eval','returns_multi_episode/sum_eval'], m=10,
    #         ylabel='Average return', save_path="./result/SemiCircleSparse",multi_episode=[False,False,False],
    #         rollout_indices=[[],[],[]], labels=['bisimulation erro',"without state decoder",'with state decoder'],
    #         colors=[cols_dark[0], cols_dark[1], cols_dark[2]])
    
    # bisimulation 3 tasks
    # compare(env_names=['AntSemiCircleSparse-v0-bisimu','AntSemiCircleSparse-v0-w-state'], exp_names=['sac','sac'],
    #         tags=['returns_multi_episode/sum','returns_multi_episode/sum'], m=10,
    #         ylabel='Average return', save_path="./result/SemiCircleSparse_bisimu-3",multi_episode=[False,False],
    #         rollout_indices=[[],[]], labels=['bisimulation erro','with state decoder'],
    #         colors=[cols_dark[0], cols_dark[1]])
    
    # bisimulation 5 tasks
    # compare(env_names=['AntSemiCircleSparse-v0-bisimu-5','AntSemiCircleSparse-v0-wo-state'], exp_names=['sac','sac'],
    #         tags=['returns_multi_episode/sum','returns_multi_episode/sum'], m=10,
    #         ylabel='Average return', save_path="./result/SemiCircleSparse_bisimu-5-and-wo",multi_episode=[False,False],
    #         rollout_indices=[[],[]], labels=['bisimulation erro','with state decoder'],
    #         colors=[cols_dark[0], cols_dark[1]])

    # bisimulation 5 tasks
    # compare(env_names=['HalfCheetahVel-v0-bisimu-5','HalfCheetahVel-v0-wo-statedecoder','HalfCheetahVel-v0-w-statedecoder'], exp_names=['sac','sac','sac'],
    #         tags=['returns_multi_episode/sum','returns_multi_episode/sum','returns_multi_episode/sum'], m=10,
    #         ylabel='Average return', save_path="./result/HalfCheetahVel_3",multi_episode=[False,False,False],
    #         rollout_indices=[[],[],[]], labels=['bisimulation erro','without state decoder','with state decoder'],
    #         colors=[cols_dark[0], cols_dark[1],cols_dark[2]])


    # compare(env_names=['GridNavi-v2'], exp_names=['dqn'],
    #         tags=['returns_multi_episode/sum'], m=10,
    #         ylabel='Average return', save_path="./result/gridnavi",multi_episode=[False],
    #         rollout_indices=[[]], labels=['GridNavi'],
    #         colors=[cols_dark[0]])

    compare(env_names=['PointRobotSparse-v0'], exp_names=['sac'],
            tags=['returns_multi_episode/sum'], m=10,
            ylabel='Average return', save_path="./result/PointRobotSparse-v0",multi_episode=[False],
            rollout_indices=[[]], labels=['PointRobotSparse'],
            colors=[cols_dark[0]])








