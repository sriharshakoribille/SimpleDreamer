import os

os.environ["MUJOCO_GL"] = "egl"

import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from dreamer.algorithms.dreamer import Dreamer
from dreamer.algorithms.plan2explore import Plan2Explore
from dreamer.utils.utils import load_config, get_base_directory
from dreamer.envs.envs import make_dmc_env, make_atari_env, get_env_infos, make_gymnasium_env
import shutil

def main(config_file, args):
    config, config_path = load_config(config_file)
    # config.operation.device = args.device

    if config.environment.benchmark == "atari":
        env = make_atari_env(
            task_name=config.environment.task_name,
            seed=config.environment.seed,
            height=config.environment.height,
            width=config.environment.width,
            skip_frame=config.environment.frame_skip,
            pixel_norm=config.environment.pixel_norm,
        )
    elif config.environment.benchmark == "dmc":
        env = make_dmc_env(
            domain_name=config.environment.domain_name,
            task_name=config.environment.task_name,
            seed=config.environment.seed,
            visualize_reward=config.environment.visualize_reward,
            from_pixels=config.environment.from_pixels,
            height=config.environment.height,
            width=config.environment.width,
            frame_skip=config.environment.frame_skip,
            pixel_norm=config.environment.pixel_norm,
        )
    elif config.environment.benchmark == "gymnasium":
        env = make_gymnasium_env(
            task_name=config.environment.task_name,
            width=config.environment.width,
            height=config.environment.height,
        )
    obs_shape, discrete_action_bool, action_size = get_env_infos(env)

    log_dir = (
        get_base_directory()
        + "/runs_new/"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + "_"
        + config.operation.log_dir
        + "_"
        + args.custom_msg
    )
    writer = SummaryWriter(log_dir)
    shutil.copy(config_path, log_dir+"/config.yml")

    device = config.operation.device
    print("Config: ", config)
    print("Initializing agent")
    if config.algorithm == "dreamer-v1":
        agent = Dreamer(
            obs_shape, discrete_action_bool, action_size, writer, device, config
        )
    elif config.algorithm == "plan2explore":
        agent = Plan2Explore(
            obs_shape, discrete_action_bool, action_size, writer, device, config
        )
    print("Training agent")
    agent.train(env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="dmc-cartpole-balance.yml",
        help="config file to run(default: dmc-cartpole-balance.yml)",
    )
    parser.add_argument(
        "--custom_msg",
        type=str,
        default="",
        help="custom message to append to log directory name(default: Vanilla Dreamer)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the code on(default: cuda)",
    )
    args = parser.parse_args()
    main(args.config, args)
