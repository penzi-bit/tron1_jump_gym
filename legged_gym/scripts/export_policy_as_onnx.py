from legged_gym import LEGGED_GYM_ROOT_DIR
import os, sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger, get_load_path, class_to_dict
from legged_gym.algorithm.mlp_encoder import MLP_Encoder
from legged_gym.algorithm.actor_critic import ActorCritic

import numpy as np
import torch
import copy

def export_policy_as_onnx(args, robot_type):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', args.task, robot_type)
    resume_path = get_load_path(log_root, load_run=args.load_run, checkpoint=args.checkpoint)
    loaded_dict = torch.load(resume_path)
    export_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', args.task, train_cfg.runner.experiment_name, 'exported', 'policies')
    os.makedirs(export_path, exist_ok=True)
    # encoder
    encoder_class = eval(train_cfg.runner.encoder_class_name)
    encoder = encoder_class(**class_to_dict(train_cfg)[train_cfg.runner.encoder_class_name]).to(args.rl_device)
    encoder.load_state_dict(loaded_dict['encoder_state_dict'])
    encoder_path = os.path.join(export_path, "encoder.onnx")
    encoder_model = copy.deepcopy(encoder.encoder).to("cpu")
    encoder_model.eval()
    dummy_input = torch.randn(encoder.num_input_dim)
    input_names = ["nn_input"]
    output_names = ["nn_output"]
    
    torch.onnx.export(
        encoder_model,
        dummy_input,
        encoder_path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=13,
    )
    print("Exported encoder as onnx script to: ", encoder_path)

    # actor_critic
    actor_critic_class = eval(train_cfg.runner.policy_class_name)

    actor_input_dim = env_cfg.env.num_observations + encoder.num_output_dim + env_cfg.commands.num_commands
    critic_input_dim = env_cfg.env.num_critic_observations + env_cfg.commands.num_commands + encoder.num_output_dim
    actor_critic = actor_critic_class(
        actor_input_dim, critic_input_dim, env_cfg.env.num_actions, **class_to_dict(train_cfg.policy)
    ).to(args.rl_device)
    actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    print()
    # export policy as an onnx file
    policy_path = os.path.join(export_path, "policy.onnx")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    model.eval()

    dummy_input = torch.randn(actor_input_dim)
    input_names = ["nn_input"]
    output_names = ["nn_output"]

    torch.onnx.export(
        model,
        dummy_input,
        policy_path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=13,
    )
    print("Exported policy as onnx script to: ", policy_path)


if __name__ == '__main__':
    
    # check ROBOT_TYPE validity
    robot_type = os.getenv("ROBOT_TYPE")
    if not robot_type:
        print("\033[1m\033[31mError: Please set the ROBOT_TYPE using 'export ROBOT_TYPE=<robot_type>'.\033[0m")
        sys.exit(1)

    if not robot_type in ["PF_TRON1A", "PF_P441A", "PF_P441B", "PF_P441C", "PF_P441C2", "SF_TRON1A", "WF_TRON1A"]:
        print("\033[1m\033[31mError: Input ROBOT_TYPE={}".format(robot_type), 
        "is not among valid robot types WF_TRON1A, SF_TRON1A, PF_TRON1A, PF_P441A, PF_P441B, PF_P441C, PF_P441C2.\033[0m")
        sys.exit(1)
    args = get_args()
    export_policy_as_onnx(args, robot_type)
