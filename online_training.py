import os
import glob
import torch
import argparse

from torchkit.pytorch_utils import set_gpu_mode

# from metalearner import MetaLearner
# from metalearner_raw1 import MetaLearner
# from metalearner_datasample import MetaLearner
# from metalearner_localrec import MetaLearner
# from metalearner_local_bisim import MetaLearner
# from metalearner_local_bisim_flip import MetaLearner
# from metalearner_beliefbisim import MetaLearner
# from metalearner_beliefbisim2 import MetaLearner
from metalearner_sacvae import MetaLearner
from online_config import (
    args_gridworld,
    args_point_robot,
    args_point_robot_sparse,
    args_cheetah_vel,
    args_cheetah_vel_sparse,
    args_ant_semicircle,
    args_ant_semicircle_sparse,
    args_fourrooms,
    args_panda_reach,
    args_panda_push,
    args_panda_pick_and_place,
)

"""
task dim permute
"""


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env-type', default='ant_semicircle')
    # parser.add_argument("--env-type", default="point_robot_sparse")
    # parser.add_argument('--env-type', default='cheetah_vel')
    parser.add_argument('--env-type', default='cheetah_vel_sparse')
    # parser.add_argument('--env-type', default='panda_reach')
    # parser.add_argument('--env-type', default='panda_push')
    # parser.add_argument('--env-type', default='panda_pick_and_place')

    args, rest_args = parser.parse_known_args()
    env = args.env_type

    # --- GridWorld ---
    if env == "gridworld":
        args = args_gridworld.get_args(rest_args)
    # --- PointRobot ---
    elif env == "point_robot":
        args = args_point_robot.get_args(rest_args)
    elif env == "point_robot_sparse":
        args = args_point_robot_sparse.get_args(rest_args)
    # --- Mujoco ---
    elif env == "cheetah_vel":
        args = args_cheetah_vel.get_args(rest_args)
    elif env == "cheetah_vel_sparse":
        args = args_cheetah_vel_sparse.get_args(rest_args)
    elif env == "ant_semicircle":
        args = args_ant_semicircle.get_args(rest_args)
    elif env == "ant_semicircle_sparse":
        args = args_ant_semicircle_sparse.get_args(rest_args)
    # --- Minigrid ---
    elif env == "fourrooms":
        args = args_fourrooms.get_args(rest_args)

    elif env == "panda_reach":
        args = args_panda_reach.get_args(rest_args)
    elif env == "panda_push":
        args = args_panda_push.get_args(rest_args)
    elif env == "panda_pick_and_place":
        args = args_panda_pick_and_place.get_args(rest_args)

    # make sure we have log directories
    try:
        os.makedirs(args.agent_log_dir)
    except OSError:
        files = glob.glob(os.path.join(args.agent_log_dir, "*.monitor.csv"))
        for f in files:
            os.remove(f)
    eval_log_dir = args.agent_log_dir + "_eval"
    try:
        os.makedirs(eval_log_dir)
    except OSError:
        files = glob.glob(os.path.join(eval_log_dir, "*.monitor.csv"))
        for f in files:
            os.remove(f)

    # set gpu
    # set_gpu_mode(torch.cuda.is_available() and args.use_gpu)
    set_gpu_mode(torch.cuda.is_available() and args.use_gpu, gpu_id=0)

    # print("gpu:",torch.cuda.current_device())
    # start training
    learner = MetaLearner(args)

    learner.train()


if __name__ == "__main__":
    main()


# python online_training.py --vae-batch-num-elbo-terms 50  --decode-only-past True

# python online_training.py   --decode-only-past True
