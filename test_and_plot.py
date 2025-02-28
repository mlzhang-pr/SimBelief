import os
import glob
import torch
import argparse
import numpy as np
from torchkit.pytorch_utils import set_gpu_mode
# from metalearner import MetaLearner
from evaluate_plot import MetaLearner
from online_config import args_gridworld, args_point_robot, args_point_robot_sparse, \
    args_cheetah_vel, args_ant_semicircle, args_ant_semicircle_sparse
from utils import evaluation as utl_eval

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env-type', default='ant_semicircle_sparse')
    parser.add_argument('--env-type', default='ant_semicircle')
    # parser.add_argument('--env-type', default='point_robot_sparse')
    # parser.add_argument('--env-type', default='gridworld')
    args, rest_args = parser.parse_known_args()
    env = args.env_type

    # --- GridWorld ---
    if env == 'gridworld':
        args = args_gridworld.get_args(rest_args)
    # --- PointRobot ---
    elif env == 'point_robot':
        args = args_point_robot.get_args(rest_args)
    elif env == 'point_robot_sparse':
        args = args_point_robot_sparse.get_args(rest_args)
    # --- Mujoco ---
    elif env == 'cheetah_vel':
        args = args_cheetah_vel.get_args(rest_args)
    elif env == 'ant_semicircle':
        args = args_ant_semicircle.get_args(rest_args)
    elif env == 'ant_semicircle_sparse':
        args = args_ant_semicircle_sparse.get_args(rest_args)

    # make sure we have log directories
    try:
        os.makedirs(args.agent_log_dir)
    except OSError:
        files = glob.glob(os.path.join(args.agent_log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
    eval_log_dir = args.agent_log_dir + "_eval"
    try:
        os.makedirs(eval_log_dir)
    except OSError:
        files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

    # set gpu
    set_gpu_mode(torch.cuda.is_available() and args.use_gpu)

    # start training
    learner = MetaLearner(args)
    print('-------------------------')
    print(learner.goals)

    

    env=learner.env

    train_tasks=learner.train_tasks
    eval_tasks=learner.eval_tasks
    print(train_tasks.shape)
    print(eval_tasks.shape)

    goals=[learner.goals[i]['goal'][0] for i in train_tasks]

    evaluate_goal_num=20
    goals=goals[:evaluate_goal_num]

    print(goals)
    print(len(goals))
    # goals=train_tasks['goals'][0]
    # print(goals)

    # returns_train, success_rate_train, log_probs, observations, rewards_train, reward_preds_train, \
    # task_samples, task_means, task_logvars = learner.evaluate(train_tasks[:50])

    # repeat_task_samples,repeat_reward_preds_train=[],[]
    repeat=40

    goals=goals*repeat
    for i in range(repeat):
        returns_train, success_rate_train, log_probs, observations, rewards_train, reward_preds_train, \
        task_samples, task_means, task_logvars = learner.evaluate(train_tasks[:evaluate_goal_num])

        

        if i ==0:
            repeat_task_samples=task_means
            # repeat_task_samples=task_samples
            repeat_reward_preds_train=reward_preds_train
        else:

            repeat_task_samples=np.vstack((repeat_task_samples,task_means))
            # repeat_task_samples=np.vstack((repeat_task_samples,task_samples))
            repeat_reward_preds_train=np.vstack((repeat_reward_preds_train,reward_preds_train))

    # repeat_task_samples=np.array(repeat_task_samples).reshape(repeat_task_samples.shap(0)*)  # shape [20*50,401,5]
    # repeat_reward_preds_train=np.array(repeat_reward_preds_train) # shape [20*50,400]

    print(repeat_task_samples.shape)
    print(repeat_reward_preds_train.shape)

    # env=learner.env
    # returns_eval, success_rate_eval, _, observations_eval, rewards_eval, reward_preds_eval, \
    # _, _, _ = learner.evaluate(eval_tasks)
    # print(observations.shape)
    # print(observations[49])
    # for i, task in enumerate(train_tasks[:1]):
    #     env.reset(task)

    #     utl_eval.plot_rollouts(observations[i], env)

    # env.reset(train_tasks[16])
    # print(task_samples.shape)
    # print(reward_preds_train.shape)
    # utl_eval.plot_rollouts(observations[16], env)
    # utl_eval.visualize_bahavior(observations[44], env)
    # utl_eval.plot_visited_states(observations[44], env)



    # ------plot t-sne-------
    task_samples2=repeat_task_samples.transpose(1,0,2)
    reward_preds_train=repeat_reward_preds_train.transpose(1,0)
    # utl_eval.visualize_latent_space_tsne(task_samples[400],reward_preds_train[399])
    utl_eval.visualize_latent_space_tsne(task_samples2[400],goals)

if __name__ == '__main__':
    main()
#python online_training.py --env-type cheetah_vel