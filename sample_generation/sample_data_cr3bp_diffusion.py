import os
import glob
import sys
import re
import time

sys.path.append('/home/anjian/Desktop/project/generative_trajectory_optimization')
sys.path.append('/home/anjian/Desktop/project/denoising-diffusion-pytorch')
from models import *  # TODO, import CVAE models and lstm models, from '/home/anjian/Desktop/project/generative_trajectory_optimization'
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

    # For icml
    checkpoint_path_list = [
                       f"/home/anjian/Desktop/project/denoising-diffusion-pytorch/results/from_della/checkpoint/cond_alpha_data_time_mass_control_range_0_1_num_300000/unet_128_mults_4_4_8_embed_class_256_512_timesteps_500_objective_pred_noise_batch_size_512_cond_drop_0.1_mask_val_0.0/2024-01-15_12-13-32"
    ]

    milestone_list = ["epoch-131"]

    data_range_list = ["0_1"]
    data_num_list = [300000]

    sample_num = 5000
    seed = 0
    diffusion_w = 5.0
    thrust = 0.15
    diffusion_type = "diffusion_time_mass_control"
    # thrust_list = [0.15, 0.35, 0.45, 0.65, 0.85]


    for_icml = True
    save_warmstart_data = False

    for i in range(len(checkpoint_path_list)):

        data_range = data_range_list[i]
        data_num = data_num_list[i]
        checkpoint_path = checkpoint_path_list[i]
        milestone = milestone_list[i]

        # Sample control ##################################################################################################
        if data_num == 30000:
            unet_dim = 64
            unet_dim_mults = "4,4,8"
            unet_dim_mults = tuple(map(int, unet_dim_mults.split(',')))
            embed_class_layers_dims = "128,256"
            embed_class_layers_dims = tuple(map(int, embed_class_layers_dims.split(',')))
            timesteps = 500
            objective = "pred_noise"
            batch_size = 512

            class_dim = 1
            channel = 1
            seq_length = 64
            cond_drop_prob = 0.1

            # Configure input data
            thrust_normalized = (thrust - 0.1) / (1.0 - 0.1)
            if data_range == "-1_1":
                thrust_normalized = thrust_normalized * 2.0 - 1.0
            alpha_data_normalized = thrust_normalized * torch.ones(size=(sample_num, 1), dtype=torch.float32)

            data_time_mass_control_normalized = get_sample_from_diffusion_attention(sample_num=sample_num,
                                                                                    class_dim=class_dim,
                                                                                    channel=channel,
                                                                                    seq_length=seq_length,
                                                                                    cond_drop_prob=cond_drop_prob,
                                                                                    diffusion_w=diffusion_w,
                                                                                    unet_dim=unet_dim,
                                                                                    unet_dim_mults=unet_dim_mults,
                                                                                    embed_class_layers_dims=embed_class_layers_dims,
                                                                                    timesteps=timesteps,
                                                                                    objective=objective,
                                                                                    batch_size=batch_size,
                                                                                    condition_input_data=alpha_data_normalized,
                                                                                    checkpoint_path=checkpoint_path,
                                                                                    milestone=milestone,
                                                                                    data_range=data_range)


        elif data_num == 300000:
            unet_dim = 128
            unet_dim_mults = "4,4,8"
            unet_dim_mults = tuple(map(int, unet_dim_mults.split(',')))
            embed_class_layers_dims = "256,512"
            embed_class_layers_dims = tuple(map(int, embed_class_layers_dims.split(',')))
            timesteps = 500
            objective = "pred_noise"
            batch_size = 512

            class_dim = 1
            channel = 1
            seq_length = 64
            cond_drop_prob = 0.1

            # Configure input data
            thrust_normalized = (thrust - 0.1) / (1.0 - 0.1)
            if data_range == "-1_1":
                thrust_normalized = thrust_normalized * 2.0 - 1.0
            alpha_data_normalized = thrust_normalized * torch.ones(size=(sample_num, 1), dtype=torch.float32)

            data_time_mass_control_normalized = get_sample_from_diffusion_attention(sample_num=sample_num,
                                                                                    class_dim=class_dim,
                                                                                    channel=channel,
                                                                                    seq_length=seq_length,
                                                                                    cond_drop_prob=cond_drop_prob,
                                                                                    diffusion_w=diffusion_w,
                                                                                    unet_dim=unet_dim,
                                                                                    unet_dim_mults=unet_dim_mults,
                                                                                    embed_class_layers_dims=embed_class_layers_dims,
                                                                                    timesteps=timesteps,
                                                                                    objective=objective,
                                                                                    batch_size=batch_size,
                                                                                    condition_input_data=alpha_data_normalized,
                                                                                    checkpoint_path=checkpoint_path,
                                                                                    milestone=milestone,
                                                                                    data_range=data_range)

        # Data preparation #######################################################################################################

        data_time_mass_normalized = data_time_mass_control_normalized[:, :4]
        data_control_normalized = data_time_mass_control_normalized[:, 4:]

        full_solution = np.hstack((np.hstack((data_time_mass_normalized[:, :3], data_control_normalized)),
                                   data_time_mass_normalized[:, -1].reshape(-1, 1)))

        if for_icml:
            parent_path = "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/cr3bp/diffusion"
            cr3bp_time_mass_alpha_control_path = f"{parent_path}/cr3bp_thrust_{thrust}_{diffusion_type}_w_{diffusion_w}_training_num_{data_num}_data_range_{data_range}_num_{sample_num}.pkl"
            with open(cr3bp_time_mass_alpha_control_path, "wb") as fp:  # write pickle
                pickle.dump(full_solution, fp)
                print(f"{cr3bp_time_mass_alpha_control_path} is saved!")

        full_solution[:, 0] = full_solution[:, 0] * (
                    MAXIMUM_SHOOTING_TIME - MINIMUM_SHOOTING_TIME) + MINIMUM_SHOOTING_TIME
        full_solution[:, 1] = full_solution[:, 1] * 15.0
        full_solution[:, 2] = full_solution[:, 2] * 15.0
        full_solution[:, 3:-1] = full_solution[:, 3:-1] * 2.0 - 1.0
        full_solution[:, -1] = full_solution[:, -1] * (MAX_MASS - 415.0) + 415.0
        data_in_spherical = revert_converted_u_data(full_solution)

        if save_warmstart_data:
            parent_path = "result/diffusion/generated_initializations"
            cr3bp_time_mass_alpha_control_path = f"{parent_path}/cr3bp_thrust_{thrust}_{diffusion_type}_w_{diffusion_w}_training_num_{data_num}_data_range_{data_range}_num_{sample_num}.pkl"
            with open(cr3bp_time_mass_alpha_control_path, "wb") as fp:  # write pickle
                pickle.dump(data_in_spherical, fp)
                print(f"{cr3bp_time_mass_alpha_control_path} is saved!")


def get_sample_from_diffusion_attention(sample_num,
                                        class_dim,
                                        channel,
                                        seq_length,
                                        cond_drop_prob,
                                        diffusion_w,
                                        unet_dim,
                                        unet_dim_mults,
                                        embed_class_layers_dims,
                                        timesteps,
                                        objective,
                                        batch_size,
                                        condition_input_data,
                                        checkpoint_path,
                                        milestone,
                                        data_range,
                                        ):
    model = Unet1D(
        seq_length=seq_length,
        dim=unet_dim,
        channels=channel,
        dim_mults=unet_dim_mults,
        embed_class_layers_dims=embed_class_layers_dims,
        class_dim=class_dim,
        cond_drop_prob=cond_drop_prob
    )

    diffusion = GaussianDiffusion1D(
        model=model,
        seq_length=seq_length,
        timesteps=timesteps,
        objective=objective
    ).cuda()

    trainer = Trainer1D(
        diffusion_model=diffusion,
        dataset=[0, 0, 0],
        results_folder=checkpoint_path,
    )

    # milestone = "epoch-102"
    trainer.load(milestone)


    # 3. Use the loaded model for sampling
    start_time = time.time()
    sample_results = diffusion.sample(
        classes=condition_input_data.cuda(),
        cond_scale=diffusion_w,
    )
    end_time = time.time()
    print(f"{checkpoint_path}, {sample_num} data, takes {end_time - start_time} seconds")

    sample_results = sample_results.reshape(sample_num, -1)

    if data_range == "-1_1":
        sample_results = (sample_results + 1.0) / 2.0

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


if __name__ == "__main__":
    main()
