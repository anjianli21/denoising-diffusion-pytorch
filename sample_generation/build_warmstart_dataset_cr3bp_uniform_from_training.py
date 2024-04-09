import os
import glob
import sys
import re
import time

from denoising_diffusion_pytorch.classifier_free_guidance_cond_1d import Unet1D, GaussianDiffusion1D, Trainer1D, \
    Dataset1D

import copy
import numpy as np
import pickle
import yaml
import torch

import importlib.util

MAX_MASS = 438.0  #
MINIMUM_SHOOTING_TIME = 0.0
MAXIMUM_SHOOTING_TIME = 40.0  # 50.0
CONTROL_SEGMENT = int(20)  # 60


def main():

    # sample_type_list = ["full_sample"]
    # data_type = f"diffusion_full_sample_300k_thrust_0.15_seed_2"
    # input_alpha_output_time_mass_control_parent_path = f"results/from_autodl/diffusion/cr3bp/results/cond_alpha_data_time_mass_control_300k_seed_2"

    # uniform data
    sample_type_list = ["uniform"]
    data_type = f"uniform_from_training_thrust_0.15_seed_2"
    seed = 2

    training_data_num_list = ["300k"]
    alpha_list = [0.15]
    sample_num = 2000

    training_data_path = "data/CR3BP/cr3bp_alpha_time_mass_control.pkl"
    with open(training_data_path, "rb") as f:
        training_data = pickle.load(f)

    rng_intial_guess = np.random.RandomState(seed=seed)
    sample_index = rng_intial_guess.randint(low=0, high=training_data.shape[0], size=sample_num * 1)
    training_data_to_use = training_data[sample_index, :][:, 1:]

    for i in range(len(training_data_num_list)):
        training_data_num = training_data_num_list[i]

        for j in range(len(sample_type_list)):
            sample_type = sample_type_list[j]

            for k in range(len(alpha_list)):
                thrust = alpha_list[k]

                if sample_type == "uniform":
                    # data_time_mass_normalized = get_sample_from_uniform_time_mass(sample_num=sample_num, seed=seed)
                    # normalized_alpha = (thrust - 0.1) / (1.0 - 0.1)
                    # normalized_alpha = normalized_alpha * np.ones((data_time_mass_normalized.shape[0], 1))
                    # # data_time_mass_alpha_normalized = np.hstack((data_time_mass_normalized, normalized_alpha))
                    # print(f"thrust = {thrust}, time and mass generated from uniform samples")
                    #
                    # number_of_segments = 20
                    # random_state = np.random.RandomState(seed=seed)
                    # theta = random_state.uniform(0, 2 * np.pi, number_of_segments * sample_num)
                    # psi = random_state.uniform(0, 2 * np.pi, number_of_segments * sample_num)
                    # r = random_state.uniform(0, 1, number_of_segments * sample_num)
                    #
                    # data_control_radius = []
                    # for i in range(sample_num):
                    #     data_control_radius_tmp = []
                    #     for j in range(number_of_segments):
                    #         data_control_radius_tmp.append(theta[i * number_of_segments + j])
                    #         data_control_radius_tmp.append(psi[i * number_of_segments + j])
                    #         data_control_radius_tmp.append(r[i * number_of_segments + j])
                    #
                    #     data_control_radius.append(data_control_radius_tmp)
                    # data_control_radius = np.asarray(data_control_radius)
                    # print(f"thrust = {thrust}, control generated from uniform sample")

                    data_time_mass_normalized = training_data_to_use[:, :4]
                    data_control_normalized = training_data_to_use[:, 4:]

                    # Data normalization
                    full_solution = np.hstack((np.hstack((data_time_mass_normalized[:, :3], data_control_normalized)),
                                               data_time_mass_normalized[:, -1].reshape(-1, 1)))
                    full_solution[:, 0] = full_solution[:, 0] * (
                            MAXIMUM_SHOOTING_TIME - MINIMUM_SHOOTING_TIME) + MINIMUM_SHOOTING_TIME
                    full_solution[:, 1] = full_solution[:, 1] * 15.0
                    full_solution[:, 2] = full_solution[:, 2] * 15.0
                    full_solution[:, -1] = full_solution[:, -1] * (MAX_MASS - 415.0) + 415.0
                    full_solution[:, 3:-1] = full_solution[:, 3:-1] * 2.0 - 1.0

                    # data_in_spherical = copy.copy(full_solution)
                    data_in_spherical = revert_converted_u_data(full_solution)

                for num in range(sample_num):
                    warmstart_data_parent_path = f"/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/icml_data/warmstart_data/{data_type}"
                    if not os.path.exists(warmstart_data_parent_path):
                        os.makedirs(warmstart_data_parent_path, exist_ok=True)
                    warmstart_data_path = f"{warmstart_data_parent_path}/{data_type}_seed_{num}.pkl"
                    with open(warmstart_data_path, 'wb') as f:
                        pickle.dump(data_in_spherical[num, :], f)
                    print(f"{warmstart_data_path} is saved")


def convert_to_spherical(ux, uy, uz):
    u = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
    theta = np.zeros_like(u)
    mask_non_zero = u != 0
    theta[mask_non_zero] = np.arcsin(uz[mask_non_zero] / u[mask_non_zero])
    alpha = np.arctan2(uy, ux)
    alpha = np.where(alpha >= 0, alpha, 2 * np.pi + alpha)

    # Make sure theta is in [0, 2*pi]
    theta = np.where(theta >= 0, theta, 2 * np.pi + theta)

    return alpha, theta, u


def revert_converted_u_data(converted_u_data):
    # TODO: this function does 2 things:
    #  1) Convert ux uy uz from [-1, 1] to theta, psi,
    #  2) convert the second half order from correct order to reverse

    n = converted_u_data.shape[0]  # number of data points
    time_data = converted_u_data[:, :3]
    mass_data = converted_u_data[:, -1].reshape(n, 1)
    control_data = converted_u_data[:, 3:-1].reshape(n, CONTROL_SEGMENT, 3)

    # Convert Cartesian coordinates to spherical coordinates
    for i in range(control_data.shape[1]):
        alpha, theta, u = convert_to_spherical(control_data[:, i, 0], control_data[:, i, 1], control_data[:, i, 2])
        control_data[:, i, 0] = alpha
        control_data[:, i, 1] = theta
        control_data[:, i, 2] = u

    # Revert the order of the second half of the segments
    control_data[:, int(CONTROL_SEGMENT / 2):] = np.flip(control_data[:, int(CONTROL_SEGMENT / 2):], axis=1)

    # Flatten the control data and stack with time_data and mass_data
    control_data = control_data.reshape(n, -1)
    data_total = np.hstack((time_data, control_data, mass_data))

    return data_total


def get_sample_from_uniform_time_mass(sample_num, seed):
    random_state = np.random.RandomState(seed=seed)
    t_shooting = random_state.uniform(MINIMUM_SHOOTING_TIME, MAXIMUM_SHOOTING_TIME, sample_num)
    t_init = random_state.uniform(0, 15.0, sample_num)
    t_final = random_state.uniform(0, 15.0, sample_num)
    mass = random_state.uniform(415.0, MAX_MASS, sample_num)
    sample_data = []
    for i in range(sample_num):
        data = []
        data.append(t_shooting[i])
        data.append(t_init[i])
        data.append(t_final[i])
        data.append(mass[i])
        data = np.asarray(data)

        sample_data.append(data)

    sample_data = np.asarray(sample_data)

    normalized_sample_data = np.zeros_like(sample_data)
    normalized_sample_data[:, 0] = (sample_data[:, 0] - MINIMUM_SHOOTING_TIME) / (MAXIMUM_SHOOTING_TIME - MINIMUM_SHOOTING_TIME)
    normalized_sample_data[:, 1] = sample_data[:, 1] / 15.0
    normalized_sample_data[:, 2] = sample_data[:, 2] / 15.0
    normalized_sample_data[:, -1] = (sample_data[:, -1] - 415.0) / (
                MAX_MASS - 415.0)

    normalized_sample_data = normalized_sample_data.astype(np.float32)

    return normalized_sample_data

if __name__ == "__main__":
    main()
