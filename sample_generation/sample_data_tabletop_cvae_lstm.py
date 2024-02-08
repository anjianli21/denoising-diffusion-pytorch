import os
import glob
import sys
import re
import time

import sys
import os

sys.path.append('/home/anjian/Desktop/project/generative_trajectory_optimization')

from models import *


from denoising_diffusion_pytorch.classifier_free_guidance_cond_1d_constraint_car import Unet1D, GaussianDiffusion1D, Trainer1D, \
    Dataset1D
from denoising_diffusion_pytorch.constraint_violation_function_tabletop import get_constraint_violation_tabletop


import copy
import numpy as np
import pickle
import yaml
import torch

import importlib.util

CVAE_PARENT_DIR = "results/from_autodl/cvae_lstm/tabletop/cvae_seed_0"
RNN_PARENT_DIR = "results/from_autodl/cvae_lstm/tabletop/lstm_seed_0"

def main():

    data_type_list = ["cvae_lstm_seed_0"]

    TIME_MIN = 3.67867
    TIME_MAX = 6.0
    CONTROL_MIN = - 1.0005
    CONTROL_MAX = 1.0005
    OBS_POS_MIN = 1.0
    OBS_POS_MAX = 9.0
    OBS_RADIUS_MIN = 0.2
    OBS_RADIUS_MAX = 0.5
    GOAL_POS_MIN = 1.0
    GOAL_POS_MAX = 9.0

    diffusion_w = 5.0
    device = "cuda:0"

    sample_num = 10
    condition_seed_num = 500

    condition_seed_list = [5000 + i for i in range(condition_seed_num)]

    constraint_violation_list = []
    for i in range(len(data_type_list)):
        data_type = data_type_list[i]
        current_prediction_data_list = []

        obs_goal_condition_input_list = []
        for j in range(len(condition_seed_list)):
            condition_seed = condition_seed_list[j]

            # Sample obs
            rng_condition = np.random.RandomState(seed=condition_seed)
            # obs sample
            is_condition_reasonable = False
            while not is_condition_reasonable:
                print("sample obs again")
                car_start_pos = np.array([[5.0, 5.0], [0.0, 0.0], [10.0, 10.0]])
                car_goal_pos = np.array(
                    [[[1.0, 1.0], [1.0, 9.0], [9.0, 1.0], [9.0, 9.0]][rng_condition.randint(low=0, high=4)]])
                obs_radius = rng_condition.rand(4) * (OBS_RADIUS_MAX - OBS_RADIUS_MIN) + OBS_RADIUS_MIN

                pos_x_min = np.minimum(car_start_pos[0][0], car_goal_pos[0][0])
                pos_y_min = np.minimum(car_start_pos[0][1], car_goal_pos[0][1])
                pos_x_max = np.maximum(car_start_pos[0][0], car_goal_pos[0][0])
                pos_y_max = np.maximum(car_start_pos[0][1], car_goal_pos[0][1])
                obs_pos_x = rng_condition.rand(4) * (pos_x_max - pos_x_min) + pos_x_min
                obs_pos_y = rng_condition.rand(4) * (pos_y_max - pos_y_min) + pos_y_min

                obs_pos = np.hstack((obs_pos_x.reshape(-1, 1), obs_pos_y.reshape(-1, 1)))

                parameters = {}
                parameters["obs_radius"] = obs_radius
                parameters["obs_pos"] = obs_pos
                parameters["car_goal_pos"] = car_goal_pos
                is_condition_reasonable = check_condition(parameters=parameters)

            obs_radius = obs_radius.reshape(1, 4)
            obs_radius = (obs_radius - OBS_RADIUS_MIN) / (OBS_RADIUS_MAX - OBS_RADIUS_MIN)

            obs_pos = obs_pos.reshape(1, 8)
            obs_pos = (obs_pos - OBS_POS_MIN) / (OBS_POS_MAX - OBS_POS_MIN)

            car_goal_pos = car_goal_pos.reshape(1, 2)
            car_goal_pos = (car_goal_pos - GOAL_POS_MIN) / (GOAL_POS_MAX - GOAL_POS_MIN)

            obs_goal_condition_input = np.hstack([obs_pos, obs_radius, car_goal_pos])

            # Repeat the same obs input as the sample num
            obs_goal_condition_input = np.tile(obs_goal_condition_input, (sample_num, 1))
            obs_goal_condition_input = torch.tensor(obs_goal_condition_input).float().cuda().to(device)
            obs_goal_condition_input_list.append(obs_goal_condition_input)

        obs_goal_condition_input_list = torch.vstack(obs_goal_condition_input_list)

        # First sample t_final using CVAE
        t_final_samples = get_sample_from_vanilla_cvae(condition_input=obs_goal_condition_input_list,
                                                               sample_num=sample_num * condition_seed_num)
        t_final_samples = t_final_samples.to(obs_goal_condition_input_list.device)
        obs_goal_t_final_samples = torch.hstack([obs_goal_condition_input_list, t_final_samples])

        control_samples = get_sample_from_rnn(conditional_input=obs_goal_t_final_samples)

        obs_goal_condition_input_list = obs_goal_condition_input_list.detach().cpu().numpy()
        t_final_samples = t_final_samples.detach().cpu().numpy()

        obs_goal_t_final_control_samples = np.hstack([obs_goal_condition_input_list, t_final_samples, control_samples])


        current_prediction_data_list.append(copy.copy(obs_goal_t_final_control_samples))

        #############################################################################
        # check constraint violation
        current_prediction_data_list = np.vstack(current_prediction_data_list)
        current_prediction_data_tensor = torch.tensor(current_prediction_data_list)
        current_violation = get_constraint_violation_tabletop(x = current_prediction_data_tensor[:, 14:],
                                                         c = current_prediction_data_tensor[:, :14],
                                                         scale=torch.tensor(1.0),
                                                         device=current_prediction_data_tensor.device)
        print(f"data type is {data_type}, violation is {current_violation}")

        constraint_violation_list.append(current_violation)

        # Save as a whole file
        sample_data_parent_path = f"/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/tabletop/{data_type}"
        if not os.path.exists(sample_data_parent_path):
            os.makedirs(sample_data_parent_path, exist_ok=True)
        sample_data_path = f"{sample_data_parent_path}/{data_type}_num_{condition_seed_num * sample_num}.pkl"
        with open(sample_data_path, 'wb') as f:
            pickle.dump(obs_goal_t_final_control_samples, f)
        print(f"{sample_data_path} is saved")

        # # Save as several file ###################################3
        # total_num = sample_num * condition_seed_num
        # for num in range(total_num):
        #     curr_conditional_seed = 5000 + num // 10
        #     curr_initial_guess_seed = num % 10
        #
        #     warmstart_data_parent_path = f"/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/warmstart_data/tabletop/{data_type}"
        #     if not os.path.exists(warmstart_data_parent_path):
        #         os.makedirs(warmstart_data_parent_path, exist_ok=True)
        #     warmstart_data_path = f"{warmstart_data_parent_path}/{data_type}_condition_seed_{curr_conditional_seed}_initial_guess_seed_{curr_initial_guess_seed}.pkl"
        #     with open(warmstart_data_path, 'wb') as f:
        #         pickle.dump(obs_goal_t_final_control_samples[num, :], f)
        #     print(f"{warmstart_data_path} is saved")

    for i in range(len(constraint_violation_list)):
        print(f"{data_type_list[i]}, constraint violation {constraint_violation_list[i]}")

    return True

def check_condition(parameters, to_print=False):
    car_num = 1
    car_radius = 0.2
    obs_num = 4
    obs_radius = parameters["obs_radius"]
    obs_pos = parameters["obs_pos"]

    car_start_pos = np.array([[5.0, 5.0]])
    car_goal_pos = parameters["car_goal_pos"]

    # Check if car start and goal positions are far enough from each other
    for i in range(car_num):
        for j in range(car_num):
            if i != j:
                if np.linalg.norm(car_start_pos[i, :] - car_start_pos[j, :]) < 4 * car_radius:
                    if to_print:
                        print(f"car {i} and car {j} start pos is too close")
                    return False
                if np.linalg.norm(car_goal_pos[i, :] - car_goal_pos[j, :]) < 4 * car_radius:
                    if to_print:
                        print(f"car {i} and car {j} goal pos is too close")
                    return False

    # Check if car start and goal positions are far enough from obstacles
    for i in range(car_num):
        for j in range(obs_num):
            if np.linalg.norm(car_start_pos[i, :] - obs_pos[j, :]) < 4 * car_radius + obs_radius[j]:
                if to_print:
                    print(f"car {i} start pos and obs {j} pos is too close")
                return False
            if np.linalg.norm(car_goal_pos[i, :] - obs_pos[j, :]) < 4 * car_radius + obs_radius[j]:
                if to_print:
                    print(f"car {i} goal pos and obs {j} pos is too close")
                return False

    for i in range(obs_num):
        for j in range(obs_num):
            if i != j:
                if np.linalg.norm(obs_pos[i, :] - obs_pos[j, :]) < obs_radius[i] + obs_radius[j]:
                    if to_print:
                        print(f"obs {i} and obs {j} start pos is too close")
                    return False

    return True

def get_sample_from_vanilla_cvae(sample_num, condition_input, seed=0):
    parent_dir = CVAE_PARENT_DIR
    cvae_config_path = parent_dir + "/version_0/config.yaml"
    cvae_ckpt_path = parent_dir + "/version_0/training_stage_3/checkpoints/last.ckpt"

    with open(cvae_config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    model = generative_models[config['model_params']['model_name']](**config['model_params'],
                                                                    **config['data_params'])

    # https://pytorch-lightning.readthedocs.io/en/stable/deploy/production_intermediate.html
    checkpoint = torch.load(cvae_ckpt_path, map_location="cpu")
    model_weights = checkpoint["state_dict"]
    for key in list(model_weights):
        model_weights[key.replace("model.", "")] = model_weights.pop(key)
    model.load_state_dict(model_weights)
    model.eval()

    curr_device = "cpu"

    # Sample ###################################################################
    # Sample x for alpha
    sample_x_list = []

    output_sample = model.sample(num_samples=sample_num, alpha=torch.tensor(condition_input),
                                      current_device=curr_device, seed=seed)
    # output_sample = output_sample.cpu().data.numpy()
    #
    # sample_data = np.squeeze(np.asarray(output_sample))
    # np.random.shuffle(sample_data)

    return output_sample.to("cuda:0")


def get_sample_from_rnn(conditional_input):
    rnn_parent_dir = RNN_PARENT_DIR
    rnn_config_path = f"{rnn_parent_dir}/version_0/config.yaml"
    rnn_ckpt_path = f"{rnn_parent_dir}/version_0/training_stage_3/checkpoints/last.ckpt"

    with open(rnn_config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    model = generative_models[config['model_params']['model_name']](**config['model_params'],
                                                                    **config['data_params'])

    # https://pytorch-lightning.readthedocs.io/en/stable/deploy/production_intermediate.html
    checkpoint = torch.load(rnn_ckpt_path, map_location="cpu")
    model_weights = checkpoint["state_dict"]
    for key in list(model_weights):
        model_weights[key.replace("model.", "")] = model_weights.pop(key)
    model.load_state_dict(model_weights)
    model.eval()

    # curr_device = "cuda:0"
    curr_device = "cpu"

    # sample #########################################################################################################

    [_, control] = model(input=torch.tensor(0.0).to(curr_device), alpha=conditional_input.to(curr_device), control_label=torch.tensor(0.0).to(curr_device))

    return control.detach().cpu().numpy()

if __name__ == "__main__":
    main()
