import os
import glob
import sys
import re
sys.path.append('/home/anjian/Desktop/project/generative_trajectory_optimization')
sys.path.append('/home/anjian/Desktop/project/denoising-diffusion-pytorch')
from models import * # TODO, import CVAE models and lstm models, from '/home/anjian/Desktop/project/generative_trajectory_optimization'
from denoising_diffusion_pytorch.classifier_free_guidance_cond_1d import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D


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

    # thrust = 0.85
    # thrust = 0.15
    # thrust = 1.0

    # Time and mass sample ########################################################################################
    # time_mass_type = "cvae"
    # time_mass_type = "gt"
    # time_mass_type = "uniform"
    # time_mass_type = "vanilla_cvae"

    # Control sample ################################################################################################
    # # TODO: if control type is diffusion
    # control_type = "diffusion_attention"
    # # Configure diffusion type, top 1 pred_V ##########################################
    # unet_dim = 128
    # unet_dim_mults = (4,8,8)
    # embed_class_layers_dims = (256, 512)
    # timesteps = 500
    # objective = "pred_v"
    # batch_size = 512
    # # Configure diffusion type, top 2 pred_V ##########################################
    # unet_dim = 128
    # unet_dim_mults = (4,4,8)
    # embed_class_layers_dims = (64, 128)
    # timesteps = 1000
    # objective = "pred_v"
    # batch_size = 512
    # # Configure diffusion type, top 1 pred_noise ##########################################
    # unet_dim = 128
    # unet_dim_mults = (4, 8, 8)
    # embed_class_layers_dims = (256, 512)
    # timesteps = 500
    # objective = "pred_noise"
    # batch_size = 512
    # # Configure diffusion type, top 2 pred_noise ##########################################
    # unet_dim = 128
    # unet_dim_mults = (4, 4, 8)
    # embed_class_layers_dims = (64, 128)
    # timesteps = 500
    # objective = "pred_noise"
    # batch_size = 512

    #
    # diffusion_type = f"unet_{unet_dim}_mults_{unet_dim_mults}_embed_class_{embed_class_layers_dims}_timesteps_{timesteps}_objective_{objective}_batch_size_{batch_size}"
    # diffusion_w_list = [1.0, 5.0, 10.0, 20.0]

    # TODO: if control type is others
    # control_type = "uniform"
    # control_type = "lstm"
    # control_type = "gt"
    # control_type = "vanilla_cvae"

    # all
    # time_mass_type_list = ["uniform", "cvae", "uniform", "vanilla_cvae", "cvae", "gt", "uniform", "cvae", "uniform", "vanilla_cvae", "cvae", "gt"]
    # control_type_list = ["uniform", "uniform", "lstm", "vanilla_cvae", "lstm", "gt", "uniform", "uniform", "lstm", "vanilla_cvae", "lstm", "gt"]
    # thrust_list = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]

    # time_mass_type_list = ["vanilla_cvae", "vanilla_cvae"]
    # control_type_list = ["vanilla_cvae", "vanilla_cvae"]
    # thrust_list = [0.15, 0.85]

    time_mass_type_list = ["cvae"] * 1
    control_type_list = ["lstm"] * 1
    thrust_list = [0.15]

    diffusion_w_list = [1.0] * len(time_mass_type_list)

    sample_num = 5000
    seed = 0
    to_save_uxyz_control_correct_order_data = False
    to_save_radius_control_revert_order_data = False
    for_icml = False

    for i in range(len(time_mass_type_list)):
        diffusion_w = diffusion_w_list[i]
        time_mass_type = time_mass_type_list[i]
        control_type = control_type_list[i]
        thrust = thrust_list[i]


        # Sample time and mass ############################################################################################
        data_time_mass_alpha_normalized = None
        if time_mass_type == "gt":
            # TODO: gt data is generated from data_processing/pre_process/fixed_and_analyze_solution_solved.py
            #  time, mass are in original scale [0, 40], [0, 15], [0, 15], [415, ]
            #  Then, data_time_mass_normalized are in nomarlized scale [0, 1]
            #  controls are in ux, uy, uz, second half correct order, range from original [-1, 1]
            if thrust in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                groundtruth_path = f"/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/paper_data/from_cluster_fixed/part_4/thrust_{thrust}_part_4_fixed_uxyz_mass_415.0.pickle"
                with open(groundtruth_path, "rb") as f:  # load pickle
                    data_loaded = pickle.load(f)
            elif thrust in [0.85]:
                groundtruth_path = f"/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/paper_data/from_cluster_fixed/test/thrust_{thrust}_part_1_fixed_uxyz_mass_415.0.pickle"
                with open(groundtruth_path, "rb") as f:  # load pickle
                    data_loaded = pickle.load(f)
            elif thrust in [0.15]:
                groundtruth_path = f"/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/paper_data/from_cluster_fixed/test/thrust_0.15_test_fixed_uxyz_mass_415.0.pickle"
                with open(groundtruth_path, "rb") as f:  # load pickle
                    data_loaded = pickle.load(f)
            data_time = data_loaded[:sample_num, :3]
            data_mass = data_loaded[:sample_num, -1].reshape(-1, 1)
            data_time_mass = np.hstack((data_time, data_mass))
            data_time_mass_normalized = copy.copy(data_time_mass)
            data_time_mass_normalized[:, 0] = data_time_mass[:, 0] / 40.0
            data_time_mass_normalized[:, 1] = data_time_mass[:, 1] / 15.0
            data_time_mass_normalized[:, 2] = data_time_mass[:, 2] / 15.0
            data_time_mass_normalized[:, 3] = (data_time_mass[:, 3] - 415.0) / (438.0 - 415.0)
            normalized_alpha = (thrust - 0.1) / (1.0 - 0.1)
            normalized_alpha = normalized_alpha * np.ones((data_time_mass_normalized.shape[0], 1))
            data_time_mass_alpha_normalized = np.hstack((data_time_mass_normalized, normalized_alpha))
            print(f"thrust = {thrust}, time and mass generated from groundtruth")
        elif time_mass_type == "cvae":
            # TODO: CVAE data
            #  time, mass are in normalized scale, [0, 1],
            data_time_mass_normalized = get_sample_from_cvae(sample_num=sample_num, seed=seed, alpha=thrust)
            normalized_alpha = (thrust - 0.1) / (1.0 - 0.1)
            normalized_alpha = normalized_alpha * np.ones((data_time_mass_normalized.shape[0], 1))
            data_time_mass_alpha_normalized = np.hstack((data_time_mass_normalized, normalized_alpha))
            print(f"thrust = {thrust}, time and mass generated from cvae")
        elif time_mass_type == "uniform":
            # TODO: uniform data
            #  time, mass are in normalized scale, [0, 1]
            data_time_mass_normalized = get_sample_from_uniform_time_mass(sample_num=sample_num, seed=seed)
            normalized_alpha = (thrust - 0.1) / (1.0 - 0.1)
            normalized_alpha = normalized_alpha * np.ones((data_time_mass_normalized.shape[0], 1))
            data_time_mass_alpha_normalized = np.hstack((data_time_mass_normalized, normalized_alpha))
            print(f"thrust = {thrust}, time and mass generated from uniform samples")

        # Sample control ##################################################################################################
        if control_type == "gt":
            # TODO: gt data is generated from data_processing/pre_process/fixed_and_analyze_solution_solved.py
            #  time, mass are in original scale [0, 40], [0, 15], [0, 15], [415, ]
            #  Then, data_time_mass_normalized are in nomarlized scale [0, 1]
            #  controls are in ux, uy, uz, second half correct order, range from original [-1, 1]
            if thrust in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                groundtruth_path = f"/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/paper_data/from_cluster_fixed/part_4/thrust_{thrust}_part_4_fixed_uxyz_mass_415.0.pickle"
                with open(groundtruth_path, "rb") as f:  # load pickle
                    data_loaded = pickle.load(f)
            elif thrust in [0.85]:
                groundtruth_path = f"/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/paper_data/from_cluster_fixed/test/thrust_{thrust}_part_1_fixed_uxyz_mass_415.0.pickle"
                with open(groundtruth_path, "rb") as f:  # load pickle
                    data_loaded = pickle.load(f)
            elif thrust in [0.15]:
                groundtruth_path = f"/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/paper_data/from_cluster_fixed/test/thrust_0.15_test_fixed_uxyz_mass_415.0.pickle"
                with open(groundtruth_path, "rb") as f:  # load pickle
                    data_loaded = pickle.load(f)

            data_control_normalized = data_loaded[:sample_num, 3:-1]
            print(f"thrust = {thrust}, control generated from groundtruth")
        elif control_type == "diffusion_attention":
            data_control_normalized = get_sample_from_diffusion_attention(time_mass_alpha_samples=data_time_mass_alpha_normalized,
                                                                          diffusion_type=diffusion_type,
                                                                          diffusion_w=diffusion_w,
                                                                          unet_dim=unet_dim, unet_dim_mults=unet_dim_mults,
                                                                          embed_class_layers_dims=embed_class_layers_dims,
                                                                          timesteps=timesteps, objective=objective,
                                                                          batch_size=batch_size)
            print(f"thrust = {thrust}, diffusion model with w {diffusion_w}, control generated from diffusion attention")
        elif control_type == "uniform":
            # TODO: uniform data
            #  Then, data_time_mass_normalized are in nomarlized scale [0, 1]
            #  control are in spherical, theta, psi, r, no order to say
            number_of_segments = 20
            random_state = np.random.RandomState(seed=seed)
            theta = random_state.uniform(0, 2 * np.pi, number_of_segments * sample_num)
            psi = random_state.uniform(0, 2 * np.pi, number_of_segments * sample_num)
            r = random_state.uniform(0, 1, number_of_segments * sample_num)

            data_control_radius = []
            for i in range(sample_num):
                data_control_radius_tmp = []
                for j in range(number_of_segments):
                    data_control_radius_tmp.append(theta[i * number_of_segments + j])
                    data_control_radius_tmp.append(psi[i * number_of_segments + j])
                    data_control_radius_tmp.append(r[i * number_of_segments + j])

                data_control_radius.append(data_control_radius_tmp)
            data_control_radius = np.asarray(data_control_radius)
            print(
                f"thrust = {thrust}, control generated from uniform sample")
        elif control_type == "lstm":
            # TODO: lstm data
            #  controls are in ux, uy, uz, second half correct order, normalized to [0, 1]
            data_control_normalized = get_sample_from_rnn(time_mass_samples=data_time_mass_normalized, alpha=thrust)
            print(f"thrust = {thrust}, control generated from lstm")

        # TODO: vanilla vae data,
        #  time, mass are in normalized scale, [0, 1],
        #  controls are in ux, uy, uz, second half correct order, normalized to [0, 1]
        if time_mass_type == "vanilla_cvae" and control_type == "vanilla_cvae":
            data_time_mass_control_normalized = get_sample_from_vanilla_cvae(sample_num=sample_num, seed=seed, alpha=thrust)
            data_time_mass_normalized = data_time_mass_control_normalized[:, :4]
            data_control_normalized = data_time_mass_control_normalized[:, 4:]
            print(f"thrust = {thrust}, time mass control all generated from vanila cvae")

        ##################################################################################################################
        ##################################################################################################################
        # Convert the variable  ###################################################################################################
        if control_type == "uniform":
            # TODO: uniform data
            #  time, mass are in normalized scale, [0, 1],
            #  control are in spherical, theta, psi, r, no order to say
            #  convert: only need to scale time, mass
            full_solution = np.hstack((np.hstack((data_time_mass_normalized[:, :3], data_control_radius)),
                                       data_time_mass_normalized[:, -1].reshape(-1, 1)))
            full_solution[:, 0] = full_solution[:, 0] * (
                        MAXIMUM_SHOOTING_TIME - MINIMUM_SHOOTING_TIME) + MINIMUM_SHOOTING_TIME
            full_solution[:, 1] = full_solution[:, 1] * 15.0
            full_solution[:, 2] = full_solution[:, 2] * 15.0
            full_solution[:, -1] = full_solution[:, -1] * (MAX_MASS - 415.0) + 415.0
            data_in_spherical = copy.copy(full_solution)

            if to_save_uxyz_control_correct_order_data:
                converted_u_data = np.zeros_like(full_solution)
                converted_u_data[:, :3] = full_solution[:, :3]
                converted_u_data[:, -1] = full_solution[:, -1]
                for i in range(3, 3 + 20 * 3, 3):
                    converted_u_data[:, i] = full_solution[:, i + 2] * np.cos(full_solution[:, i + 1]) * np.cos(full_solution[:, i])
                    converted_u_data[:, i + 1] = full_solution[:, i + 2] * np.cos(full_solution[:, i + 1]) * np.sin(full_solution[:, i])
                    converted_u_data[:, i + 2] = full_solution[:, i + 2] * np.sin(full_solution[:, i + 1])

                parent_path = "result/generated_initializations/jgcd_data"
                cr3bp_time_mass_control_path = f"{parent_path}/cr3bp_thrust_{thrust}_time_mass_{time_mass_type}_control_{control_type}_uxyz_correct_order_num_{sample_num}.pkl"
                with open(cr3bp_time_mass_control_path, "wb") as fp:  # write pickle
                    pickle.dump(converted_u_data, fp)
                    print(f"{cr3bp_time_mass_control_path} is saved!")

        elif control_type == "gt":
            # TODO: gt data is generated from data_processing/pre_process/fixed_and_analyze_solution_solved.py
            #  Then, data_time_mass_normalized are in nomarlized scale [0, 1]
            #  controls are in ux, uy, uz, second half correct order, range from original [-1, 1]
            #  convert: scale time, mass, convert ux uy uz to spherical, convert the second half to revert order by revert_converted_u_data
            full_solution = np.hstack((np.hstack((data_time_mass_normalized[:, :3], data_control_normalized)),
                                       data_time_mass_normalized[:, -1].reshape(-1, 1)))

            full_solution[:, 0] = full_solution[:, 0] * (
                        MAXIMUM_SHOOTING_TIME - MINIMUM_SHOOTING_TIME) + MINIMUM_SHOOTING_TIME
            full_solution[:, 1] = full_solution[:, 1] * 15.0
            full_solution[:, 2] = full_solution[:, 2] * 15.0
            full_solution[:, -1] = full_solution[:, -1] * (MAX_MASS - 415.0) + 415.0
            if to_save_uxyz_control_correct_order_data:
                parent_path = "result/generated_initializations/jgcd_data"
                cr3bp_time_mass_control_path = f"{parent_path}/cr3bp_thrust_{thrust}_time_mass_{time_mass_type}_control_{control_type}_uxyz_correct_order_num_{sample_num}.pkl"
                with open(cr3bp_time_mass_control_path, "wb") as fp:  # write pickle
                    pickle.dump(full_solution, fp)
                    print(f"{cr3bp_time_mass_control_path} is saved!")
            data_in_spherical = revert_converted_u_data(full_solution)
        elif control_type == "lstm" or control_type == "vanilla_cvae":
            # TODO: cvae + lstm, vanilla_vae data
            #  time, mass are in normalized scale, [0, 1],
            #  controls are in ux, uy, uz, second half correct order, normalized to [0, 1]
            #  convert, scale time, mass, first convert ux uy uz to [-1, 1],
            #  then convert ux uy uz to spherical, convert the second half to revert order by revert_converted_u_data
            full_solution = np.hstack((np.hstack((data_time_mass_normalized[:, :3], data_control_normalized)),
                                   data_time_mass_normalized[:, -1].reshape(-1, 1)))

            # TODO: for icml
            if for_icml:
                parent_path = "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/cr3bp/cvae_lstm"
                cr3bp_time_mass_alpha_control_path = f"{parent_path}/cr3bp_thrust_{thrust}_time_mass_{time_mass_type}_control_{control_type}_num_{sample_num}.pkl"
                with open(cr3bp_time_mass_alpha_control_path, "wb") as fp:  # write pickle
                    pickle.dump(full_solution, fp)
                    print(f"{cr3bp_time_mass_alpha_control_path} is saved!")

            full_solution[:, 0] = full_solution[:, 0] * (MAXIMUM_SHOOTING_TIME - MINIMUM_SHOOTING_TIME) + MINIMUM_SHOOTING_TIME
            full_solution[:, 1] = full_solution[:, 1] * 15.0
            full_solution[:, 2] = full_solution[:, 2] * 15.0
            full_solution[:, 3:-1] = full_solution[:, 3:-1] * 2.0 - 1.0
            full_solution[:, -1] = full_solution[:, -1] * (MAX_MASS - 415.0) + 415.0
            # TODO Save fixed control data, but in ux uy uz
            if to_save_uxyz_control_correct_order_data:
                parent_path = "result/generated_initializations/jgcd_data"
                cr3bp_time_mass_control_path = f"{parent_path}/cr3bp_thrust_{thrust}_time_mass_{time_mass_type}_control_{control_type}_uxyz_correct_order_num_{sample_num}.pkl"
                with open(cr3bp_time_mass_control_path, "wb") as fp:  # write pickle
                    pickle.dump(full_solution, fp)
                    print(f"{cr3bp_time_mass_control_path} is saved!")
            data_in_spherical = revert_converted_u_data(full_solution)

        # save #######################################################################################################
        if to_save_radius_control_revert_order_data:
            parent_path = "result/generated_initializations/jgcd_data"
            if control_type == "diffusion_attention":
                cr3bp_time_mass_alpha_control_path = f"{parent_path}/cr3bp_thrust_{thrust}_time_mass_{time_mass_type}_control_{control_type}_diffusion_type_{diffusion_type}_w_{diffusion_w}_num_{sample_num}.pkl"
            else:
                cr3bp_time_mass_alpha_control_path = f"{parent_path}/cr3bp_thrust_{thrust}_time_mass_{time_mass_type}_control_{control_type}_num_{sample_num}.pkl"
            with open(cr3bp_time_mass_alpha_control_path, "wb") as fp:  # write pickle
                pickle.dump(data_in_spherical, fp)
                print(f"{cr3bp_time_mass_alpha_control_path} is saved!")

def get_sample_from_cvae(sample_num, seed, alpha):
    parent_dir = "/home/anjian/Desktop/project/generative_trajectory_optimization/logs/from_cluster"
    cvae_config_path = parent_dir + "/GMM/data_part4/v1/version_0/config.yaml"
    cvae_ckpt_path = parent_dir + "/GMM/data_part4/v1/version_0/training_stage_6/checkpoints/last.ckpt"

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
    normalized_alpha = (alpha - 0.1) / (1.0 - 0.1)
    alpha_list = [normalized_alpha]

    # Sample x for alpha
    sample_x_list = []
    for i in range(len(alpha_list)):
        alpha = alpha_list[i]
        output_sample = model.sample(num_samples=sample_num, alpha=torch.tensor(alpha),
                                          current_device=curr_device, seed=seed)
        output_sample = output_sample.cpu().data.numpy()

        sample_x_list.append(output_sample)

    sample_data = np.squeeze(np.asarray(sample_x_list))
    np.random.shuffle(sample_data)

    # check data status
    out_of_bound_num = 0
    for i in range(sample_num):
        if sample_data[i, 0] > 1.0 or sample_data[i, 0] < 0 or sample_data[i, 1] > 1.0 or sample_data[i, 1] < 0 or sample_data[i, 2] > 1.0 or \
                sample_data[i, 2] < 0:
            out_of_bound_num += 1
    print(f"out of bound num is {out_of_bound_num}, out ratio is {out_of_bound_num / sample_num:.2f}")

    return sample_data

def get_sample_from_vanilla_cvae(sample_num, seed, alpha):
    parent_dir = "//home/anjian/Desktop/project/generative_trajectory_optimization/logs/CVAEExperiment/cr3bp/CVAEVanilla"
    cvae_config_path = parent_dir + "/v1/version_0/config.yaml"
    cvae_ckpt_path = parent_dir + "/v1/version_0/training_stage_6/checkpoints/last.ckpt"

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
    normalized_alpha = (alpha - 0.1) / (1.0 - 0.1)
    alpha_list = [normalized_alpha]

    # Sample x for alpha
    sample_x_list = []
    for i in range(len(alpha_list)):
        alpha = alpha_list[i]
        output_sample = model.sample(num_samples=sample_num, alpha=torch.tensor(alpha),
                                          current_device=curr_device, seed=seed)
        output_sample = output_sample.cpu().data.numpy()

        sample_x_list.append(output_sample)

    sample_data = np.squeeze(np.asarray(sample_x_list))
    np.random.shuffle(sample_data)

    return sample_data

def get_sample_from_rnn(time_mass_samples, alpha):

    rnn_config_path = "/home/anjian/Desktop/project/generative_trajectory_optimization/logs/from_cluster/time_mass_to_control/data_part4_lstmv2/v7/version_0/config.yaml"
    rnn_ckpt_path = "/home/anjian/Desktop/project/generative_trajectory_optimization/logs/from_cluster/time_mass_to_control/data_part4_lstmv2/v7/version_0/training_stage_4/checkpoints/last.ckpt"

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

    curr_device = "cpu"

    # sample #########################################################################################################
    normalized_alpha = (alpha - 0.1) / (1.0 - 0.1)

    time_mass = torch.tensor(time_mass_samples, dtype=torch.float32).to(curr_device)
    alpha = torch.ones(time_mass.size()[0], dtype=torch.float32).reshape(-1, 1) * normalized_alpha
    time_mass_alpha = torch.cat((time_mass, alpha), 1)

    [_, control] = model(time_mass_alpha)

    return control.detach().cpu().numpy()

def get_sample_from_diffusion_attention(time_mass_alpha_samples, diffusion_type, diffusion_w, unet_dim, unet_dim_mults,
                                                                      embed_class_layers_dims,
                                                                      timesteps, objective,
                                                                      batch_size):

    class_dim = 5
    model = Unet1D(
        dim=unet_dim,
        channels=3,
        dim_mults=unet_dim_mults,
        embed_class_layers_dims=embed_class_layers_dims,
        class_dim=class_dim,
        cond_drop_prob=0.1
    )

    diffusion = GaussianDiffusion1D(
        model=model,
        seq_length=20,
        timesteps=timesteps,
        objective=objective
    ).cuda()

    # results_folder = "/home/anjian/Desktop/project/denoising-diffusion-pytorch/results/2023-10-22_18-59-35"
    # TODO: here the results folder will contain a folder with the name to be the date of the training, so we have to find the child directory mannually
    results_folder = "/home/anjian/Desktop/project/denoising-diffusion-pytorch/from_della/checkpoint/top_10/" + diffusion_type + "/"
    child_directories = glob.glob(os.path.join(results_folder, "*/"))
    if child_directories:
        final_result_folder = child_directories[0]

    # trainer = Trainer1D(
    #     diffusion_model=diffusion,
    #     dataset=[0, 0, 0],
    #     results_folder=final_result_folder,
    #     train_batch_size=batch_size
    # )
    trainer = Trainer1D(
        diffusion_model=diffusion,
        dataset=[0, 0, 0],
        results_folder=final_result_folder,
    )

    # 2. Get the milestone number from the file
    # Assuming final_result_folder is already defined
    files = os.listdir(final_result_folder)

    # Regular expression to extract the epoch number
    regex = r"model-epoch-(\d{3}).pt"

    # Initialize variables
    max_epoch = -1
    milestone = ""

    for file in files:
        if file.endswith(".pt"):
            match = re.search(regex, file)
            if match:
                epoch_num = int(match.group(1))
                if epoch_num > max_epoch:
                    max_epoch = epoch_num
                    milestone = f"epoch-{epoch_num:03d}"  # Format with leading zeros
    # milestone = "epoch-102"
    trainer.load(milestone)

    time_mass_alpha_samples = torch.tensor(time_mass_alpha_samples, dtype=torch.float32)
    # 3. Use the loaded model for sampling
    sample_results = diffusion.sample(
        classes=time_mass_alpha_samples.cuda(),
        cond_scale=diffusion_w,
    )

    sample_results = sample_results.reshape(-1, 60)

    return sample_results.detach().cpu().numpy()

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