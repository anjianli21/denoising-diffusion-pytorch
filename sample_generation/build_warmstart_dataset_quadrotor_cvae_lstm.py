import glob

import sys
import os

sys.path.append('/home/anjian/Desktop/project/generative_trajectory_optimization')

from models import *

from denoising_diffusion_pytorch.constraint_violation_function_improved_quadrotor import \
    get_constraint_violation_quadrotor

import copy
import numpy as np
import pickle
import yaml
import torch

CVAE_PARENT_DIR = "results/iclr25/quadrotor/results/cvae_seed_2"
RNN_PARENT_DIR = "results/iclr25/quadrotor/results/lstm_seed_2"

def main():


    data_type_list = ["cvae_lstm_seed_2"]

    # Setup v2
    TIME_MIN = 4.20
    TIME_MAX = 4.57
    CONTROL_U1_MIN = - np.pi / 9 - .001
    CONTROL_U1_MAX = np.pi / 9 + .001
    CONTROL_U2_MIN = - np.pi / 9 - .001
    CONTROL_U2_MAX = np.pi / 9 + .001
    CONTROL_U3_MIN = 0. - .001
    CONTROL_U3_MAX = 1.5 * 9.81 + .001

    OBS_POS_X_MIN = -6.0
    OBS_POS_X_MAX = 6.0
    OBS_POS_Y_MIN = -3.0
    OBS_POS_Y_MAX = 3.0
    OBS_POS_Z_MIN = -3.0
    OBS_POS_Z_MAX = 3.0

    OBS_RADIUS_MIN = 1.5
    OBS_RADIUS_MAX = 3.5

    GOAL_PERTURBATION_RANGE = 2.

    obs_num = 4

    device = "cuda:0"

    sample_num = 10
    condition_seed_num = 20

    condition_seed_list = [5000 + i for i in range(condition_seed_num)]

    constraint_violation_list = []
    for i in range(len(data_type_list)):
        data_type = data_type_list[i]
        current_prediction_data_list = []

        obs_goalperturb_condition_input_list = []
        for j in range(len(condition_seed_list)):
            condition_seed = condition_seed_list[j]

            # Sample obs
            rng_condition = np.random.RandomState(seed=condition_seed)
            is_condition_reasonable = False

            # Sample condition
            num_of_sample_until_feasible = 0
            while not is_condition_reasonable:
                # print("sample obs again")

                agent_start_pos = np.array([[-12.0, 0.0, 0.0]])
                # TODO: add random perturbation to the goal pos
                agent_goal_pos_perturbation = rng_condition.rand(
                    1) * 2 * GOAL_PERTURBATION_RANGE - GOAL_PERTURBATION_RANGE
                agent_goal_pos = np.array([[12., 0., 0.]]) + agent_goal_pos_perturbation

                obs_radius = rng_condition.rand(obs_num) * (OBS_RADIUS_MAX - OBS_RADIUS_MIN) + OBS_RADIUS_MIN

                obs_pos_x = rng_condition.rand(obs_num - 1) * (OBS_POS_X_MAX - OBS_POS_X_MIN) + OBS_POS_X_MIN
                obs_pos_y = rng_condition.rand(obs_num - 1) * (OBS_POS_Y_MAX - OBS_POS_Y_MIN) + OBS_POS_Y_MIN
                obs_pos_z = rng_condition.rand(obs_num - 1) * (OBS_POS_Z_MAX - OBS_POS_Z_MIN) + OBS_POS_Z_MIN

                # random sample from 0.3 to 1.0
                center_obs_pos_x = (rng_condition.rand(1) * 0.4 + 0.3) * (OBS_POS_X_MAX - OBS_POS_X_MIN) + OBS_POS_X_MIN
                gradient = (agent_goal_pos[0][1] - agent_start_pos[0][1]) / (
                        agent_goal_pos[0][0] - agent_start_pos[0][0])
                center_obs_pos_y = agent_start_pos[0][1] + (
                        center_obs_pos_x - agent_start_pos[0][0]) * gradient
                center_obs_pos_z = agent_start_pos[0][2] + (
                        center_obs_pos_x - agent_start_pos[0][0]) * gradient

                all_obs_pos_x = np.hstack([obs_pos_x, center_obs_pos_x])
                all_obs_pos_y = np.hstack([obs_pos_y, center_obs_pos_y])
                all_obs_pos_z = np.hstack([obs_pos_z, center_obs_pos_z])

                obs_pos = np.hstack(
                    (all_obs_pos_x.reshape(-1, 1), all_obs_pos_y.reshape(-1, 1), all_obs_pos_z.reshape(-1, 1)))

                parameters = {}
                parameters["obs_radius"] = obs_radius
                parameters["obs_pos"] = obs_pos
                parameters["agent_goal_pos"] = agent_goal_pos
                parameters["agent_start_pos"] = agent_start_pos
                is_condition_reasonable = check_condition(parameters=parameters)

                num_of_sample_until_feasible += 1

            print(f"number of samples until feasible {num_of_sample_until_feasible}")
            obs_radius = obs_radius.reshape(1, 4)
            obs_radius = (obs_radius - OBS_RADIUS_MIN) / (OBS_RADIUS_MAX - OBS_RADIUS_MIN)

            obs_pos_x = obs_pos[:, 0]
            obs_pos_y = obs_pos[:, 1]
            obs_pos_z = obs_pos[:, 2]
            obs_pos_x = (obs_pos_x - OBS_POS_X_MIN) / (OBS_POS_X_MAX - OBS_POS_X_MIN)
            obs_pos_y = (obs_pos_y - OBS_POS_Y_MIN) / (OBS_POS_Y_MAX - OBS_POS_Y_MIN)
            obs_pos_z = (obs_pos_z - OBS_POS_Z_MIN) / (OBS_POS_Z_MAX - OBS_POS_Z_MIN)
            obs_pos = np.hstack([obs_pos_x.reshape(1, -1), obs_pos_y.reshape(1, -1), obs_pos_z.reshape(1, -1)])

            obs_condition_input = np.hstack([obs_pos, obs_radius])
            # Repeat the same obs input as the sample num
            obs_condition_input = np.tile(obs_condition_input, (sample_num, 1))

            # TODO: also normalize the goal pos perturbation
            agent_goal_pos_perturbation = (agent_goal_pos_perturbation + GOAL_PERTURBATION_RANGE) / (
                    2 * GOAL_PERTURBATION_RANGE)
            agent_goal_pos_perturbation = agent_goal_pos_perturbation.reshape(1, -1)
            agent_goal_pos_perturbation = np.tile(agent_goal_pos_perturbation, (sample_num, 1))

            # TODO: combine obs condition and goalperturb
            obs_goalperturb_condition_input = np.hstack([obs_condition_input, agent_goal_pos_perturbation])
            obs_goalperturb_condition_input = torch.tensor(obs_goalperturb_condition_input).float().cuda()
            obs_goalperturb_condition_input_list.append(obs_goalperturb_condition_input)

        obs_goalperturb_condition_input_list = torch.vstack(obs_goalperturb_condition_input_list)

        # First sample t_final using CVAE
        t_final_samples = get_sample_from_vanilla_cvae(condition_input=obs_goalperturb_condition_input_list,
                                                               sample_num=sample_num * condition_seed_num)
        t_final_samples = t_final_samples.to(obs_goalperturb_condition_input_list.device)

        obs_goalperturb_t_final_samples = torch.hstack([obs_goalperturb_condition_input_list, t_final_samples])

        control_samples = get_sample_from_rnn(conditional_input=obs_goalperturb_t_final_samples)

        obs_goalperturb_condition_input_list = obs_goalperturb_condition_input_list.detach().cpu().numpy()
        t_final_samples = t_final_samples.detach().cpu().numpy()

        obs_goalperturb_t_final_control_samples = np.hstack([obs_goalperturb_condition_input_list, t_final_samples, control_samples])

        current_prediction_data_list.append(copy.copy(obs_goalperturb_t_final_control_samples))

        #############################################################################
        # check constraint violation
        current_prediction_data_list = np.vstack(current_prediction_data_list)
        current_prediction_data_tensor = torch.tensor(current_prediction_data_list)
        current_violation = get_constraint_violation_quadrotor(x=current_prediction_data_tensor[:, 17:],
                                                               c=current_prediction_data_tensor[:, :17],
                                                               scale=torch.tensor(1.0),
                                                               device=current_prediction_data_tensor.device)
        print(f"data type is {data_type}, violation is {current_violation}")

        constraint_violation_list.append(current_violation)

        # Data preparation #######################################################################################################
        # obs_pos,
        obs_goalperturb_t_final_control_samples[:, :4] = obs_goalperturb_t_final_control_samples[:, :4] * (
                OBS_POS_X_MAX - OBS_POS_X_MIN) + OBS_POS_X_MIN
        obs_goalperturb_t_final_control_samples[:, 4:8] = obs_goalperturb_t_final_control_samples[:, 4:8] * (
                OBS_POS_Y_MAX - OBS_POS_Y_MIN) + OBS_POS_Y_MIN
        obs_goalperturb_t_final_control_samples[:, 8:12] = obs_goalperturb_t_final_control_samples[:, 8:12] * (
                OBS_POS_Z_MAX - OBS_POS_Z_MIN) + OBS_POS_Z_MIN

        # obs_radius, original range [0.5, 1.5]
        obs_goalperturb_t_final_control_samples[:, 12:16] = obs_goalperturb_t_final_control_samples[:,
                                                            12:16] * (
                                                                    OBS_RADIUS_MAX - OBS_RADIUS_MIN) + OBS_RADIUS_MIN

        obs_goalperturb_t_final_control_samples[:, 16] = obs_goalperturb_t_final_control_samples[:,
                                                         16] * 2 * GOAL_PERTURBATION_RANGE - GOAL_PERTURBATION_RANGE

        # t_final, original range [TIME_MIN, TIME_MAX]
        obs_goalperturb_t_final_control_samples[:, 17] = obs_goalperturb_t_final_control_samples[:, 17] * (
                TIME_MAX - TIME_MIN) + TIME_MIN
        # Control, original range u1, u2, u3
        obs_goalperturb_t_final_control_samples[:, 18:18 + 80] = obs_goalperturb_t_final_control_samples[:,
                                                                 18:18 + 80] * (
                                                                         CONTROL_U1_MAX - CONTROL_U1_MIN) + CONTROL_U1_MIN
        obs_goalperturb_t_final_control_samples[:, 18 + 80:18 + 160] = obs_goalperturb_t_final_control_samples[
                                                                       :, 18 + 80:18 + 160] * (
                                                                               CONTROL_U2_MAX - CONTROL_U2_MIN) + CONTROL_U2_MIN
        obs_goalperturb_t_final_control_samples[:, 18 + 160:18 + 240] = obs_goalperturb_t_final_control_samples[
                                                                        :, 18 + 160:18 + 240] * (
                                                                                CONTROL_U3_MAX - CONTROL_U3_MIN) + CONTROL_U3_MIN
        print("data normalization is done")

        # Save as several file ###################################3
        total_num = sample_num * condition_seed_num
        for num in range(total_num):
            curr_conditional_seed = 5000 + num // 10
            curr_initial_guess_seed = num % 10

            warmstart_data_parent_path = f"/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/warmstart_data/quadrotor/{data_type}"
            if not os.path.exists(warmstart_data_parent_path):
                os.makedirs(warmstart_data_parent_path, exist_ok=True)
            warmstart_data_path = f"{warmstart_data_parent_path}/{data_type}_condition_seed_{curr_conditional_seed}_initial_guess_seed_{curr_initial_guess_seed}.pkl"
            with open(warmstart_data_path, 'wb') as f:
                pickle.dump(obs_goalperturb_t_final_control_samples[num, :], f)
            print(f"{warmstart_data_path} is saved")

    for i in range(len(constraint_violation_list)):
        print(f"{data_type_list[i]}, constraint violation {constraint_violation_list[i]}")

    return True

def check_condition(parameters, to_print=False):
    to_print = False

    agent_num = 1
    agent_radius = 0.5
    obs_num = 4
    obs_radius = parameters["obs_radius"]
    obs_pos = parameters["obs_pos"]

    agent_start_pos = parameters["agent_start_pos"]
    agent_goal_pos = parameters["agent_goal_pos"]

    # Check if agent start and goal positions are far enough from each other
    for i in range(agent_num):
        for j in range(agent_num):
            if i != j:
                if np.linalg.norm(agent_start_pos[i, :] - agent_start_pos[j, :]) < 4 * agent_radius:
                    if to_print:
                        print(f"agent {i} and agent {j} start pos is too close")
                    return False
                if np.linalg.norm(agent_goal_pos[i, :] - agent_goal_pos[j, :]) < 4 * agent_radius:
                    if to_print:
                        print(f"agent {i} and agent {j} goal pos is too close")
                    return False

    # Check if agent start and goal positions are far enough from obstacles
    for i in range(agent_num):
        for j in range(obs_num):
            if np.linalg.norm(agent_start_pos[i, :] - obs_pos[j, :]) < 4 * agent_radius + obs_radius[j]:
                if to_print:
                    print(f"agent {i} start pos and obs {j} pos is too close")
                return False
            if np.linalg.norm(agent_goal_pos[i, :] - obs_pos[j, :]) < 4 * agent_radius + obs_radius[j]:
                if to_print:
                    print(f"agent {i} goal pos and obs {j} pos is too close")
                return False

    # Check if obstacles are away from each other
    for i in range(obs_num):
        for j in range(obs_num):
            if i != j:
                if np.linalg.norm(obs_pos[i, :] - obs_pos[j, :]) < obs_radius[i] + obs_radius[j] + 3 * agent_radius:
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
