import os

os.environ["MUJOCO_GL"] = "egl"

import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from dreamer.algorithms.dreamer import Dreamer
from dreamer.algorithms.plan2explore import Plan2Explore
from dreamer.utils.utils import load_config_dir, get_base_directory
from dreamer.envs.envs import make_dmc_env, make_atari_env, get_env_infos, make_gym_env
import shutil

def main(config_dir, ckpt, custom_msg):
    config, config_path = load_config_dir(config_dir)

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
    elif config.environment.benchmark == "gym":
        env = make_gym_env(
            task_name=config.environment.task_name,
            frame_skip=config.environment.frame_skip,
            from_pixels=config.environment.from_pixels,
            width=config.environment.width,
            height=config.environment.height,
        )
    obs_shape, discrete_action_bool, action_size = get_env_infos(env)
    checkpoint_path = os.path.join(config_path, "ckpt_%s.pt" % ckpt)
    log_dir = (
        get_base_directory()
        + "/evals/"
        + config.operation.log_dir + "/"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + "_"
        + config.operation.log_dir
        + "_"
        + custom_msg
    )

    writer = SummaryWriter(log_dir)
    # shutil.copyfile(config_path, log_dir+"/config.yml")
    device = config.operation.device
    print("Config: ", config)
    print("Initializing agent")
    if config.algorithm == "dreamer-v1":
        agent = Dreamer(
            obs_shape, discrete_action_bool, action_size, writer, device, config,
            log_dir = log_dir
        )
    elif config.algorithm == "plan2explore":
        agent = Plan2Explore(
            obs_shape, discrete_action_bool, action_size, writer, device, config
        )
    print("Loading checkpoint")
    agent.load_agent(checkpoint_path)
    print("Evaluating agent")
    agent.evaluate_agent(env, num_episodes=10)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_dir",
        type=str,
        default="",
        help="Directory of configs and checkpoints",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Checkpoint to load",
    )
    parser.add_argument(
        "--custom_msg",
        type=str,
        default="",
        help="custom message to append to log directory name(default: Vanilla Dreamer)",
    )
    args = parser.parse_args()
    main(args.load_dir, args.checkpoint, args.custom_msg)
