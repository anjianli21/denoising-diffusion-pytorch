import sys
import time

#sys.path.append('/home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/python_scripts')

#from models import *  # TODO, import CVAE models and lstm models, from '/home/anjian/Desktop/project/generative_trajectory_optimization'
from classifier_free_guidance_cond_1d_improved_constrained_diffusion import Unet1D, GaussianDiffusion1D, Trainer1D

import numpy as np
import pickle
import torch
import argparse
import re
import os
from datetime import datetime


def main(timesteps,data_num,sample_num,mask_val):
    
    unet_dim = 128 #20 #128
    unet_dim_mults = "4,4,8"
    unet_dim_mults = tuple(map(int, unet_dim_mults.split(',')))
    unet_dim_mults_in_str = "_".join(map(str, unet_dim_mults))
    embed_class_layers_dims = "256,512" #"40,80"
    embed_class_layers_dims = tuple(map(int, embed_class_layers_dims.split(',')))
    embed_class_layers_dims_in_str = "_".join(map(str, embed_class_layers_dims))
    checkpoint_path = f"/scratch/gpfs/jg3607/Diffusion_model/boundary/results/cr3bp_vanilla_diffusion_seed_0/unet_{unet_dim}_mults_{unet_dim_mults_in_str}_embed_class_{embed_class_layers_dims_in_str}_timesteps_{timesteps}_batch_size_2000_cond_drop_0.1_mask_val_{mask_val}/"

    folder_name = get_latest_file(checkpoint_path)
    checkpoint_path = checkpoint_path + folder_name
    milestone = get_milestone_string(checkpoint_path)

    diffusion_w = 5.0
    thrust = 1.0
    diffusion_type = "diffusion_boundary"

    save_warmstart_data = True

    objective = "pred_noise"
    class_dim = 1
    channel = 1
    seq_length = 66
    cond_drop_prob = 0.1

    # Randomly sample halo energy
    alpha_data_normalized = torch.rand(size=(sample_num, 1), dtype=torch.float32)

    full_solution = get_sample_from_diffusion_attention(sample_num=sample_num,
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
                                                                            condition_input_data=alpha_data_normalized,
                                                                            checkpoint_path=checkpoint_path,
                                                                            milestone=milestone,
                                                                            mask_val=mask_val)

    # Data preparation #######################################################################################################
    min_shooting_time = 0
    max_shooting_time = 40
    min_coast_time = 0
    max_coast_time = 15
    min_halo_energy = 0.008
    max_halo_energy = 0.095
    min_final_fuel_mass = 408 #700-292 => cut off value at 90%
    max_final_fuel_mass = 470
    min_manifold_length = 5
    max_manifold_length = 11


    # Unnormalize times
    full_solution[:, 0] = full_solution[:, 0] * (max_shooting_time - min_shooting_time) + min_shooting_time
    full_solution[:, 1] = full_solution[:, 1] * (max_coast_time - min_coast_time) + min_coast_time
    full_solution[:, 2] = full_solution[:, 2] * (max_coast_time - min_coast_time) + min_coast_time
    #Convert cartesian control back to correct range, NO CONVERSION TO POLAR
    full_solution[:, 3:-3] = full_solution[:, 3:-3] * 2 * thrust - thrust
    ux = full_solution[:,3:-3:3]
    uy = full_solution[:,4:-3:3]
    uz = full_solution[:,5:-3:3]
    alpha, beta, r = convert_to_spherical(ux, uy, uz)
    full_solution[:,3:-3:3] = alpha
    full_solution[:,4:-3:3] = beta
    full_solution[:,5:-3:3] = r 
    # Unnormalize fuel mass and manifold parameters, HALO PERIOD IS NOT UNNORMALIZED, NEEDS TO BE DONE IN THE ACTUAL RUN
    full_solution[:, -3] = full_solution[:, -3] * (max_final_fuel_mass - min_final_fuel_mass) + min_final_fuel_mass
    full_solution[:, -1] = full_solution[:, -1] * (max_manifold_length - min_manifold_length) + min_manifold_length
    # Unnormalize halo energy
    halo_energies = alpha_data_normalized.detach().cpu().numpy() * (max_halo_energy - min_halo_energy) + min_halo_energy
    full_solution = np.hstack((halo_energies, full_solution))

    if save_warmstart_data:
        parent_path = f"/home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/results/generated_initializations/boundary/unet_{unet_dim}_mults_{unet_dim_mults_in_str}_embed_class_{embed_class_layers_dims_in_str}_timesteps_{timesteps}_batch_size_2000_cond_drop_0.1_mask_val_{mask_val}"
        os.makedirs(parent_path, exist_ok=True)
        cr3bp_time_mass_alpha_control_path = f"{parent_path}/cr3bp_{diffusion_type}_w_{diffusion_w}_training_num_{data_num}_num_{sample_num}.pkl"
        with open(cr3bp_time_mass_alpha_control_path, "wb") as fp:  # write pickle
            pickle.dump(full_solution, fp)
            print(f"{cr3bp_time_mass_alpha_control_path} is saved!")

def get_milestone_string(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Regular expression to match the epoch number in the filenames
    epoch_regex = re.compile(r'model-epoch-(\d+)\.pt')
    
    # Extract epoch numbers
    epoch_numbers = []
    for file in files:
        match = epoch_regex.match(file)
        if match:
            epoch_numbers.append(int(match.group(1)))
    
    # Find the highest epoch number
    if epoch_numbers:
        highest_epoch = max(epoch_numbers)
        milestone_string = f"epoch-{highest_epoch}"
        return milestone_string
    else:
        return None

def get_latest_file(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Date format in the filenames
    date_format = "%Y-%m-%d_%H-%M-%S"
    
    latest_time = None
    latest_file = None
    
    for file in files:
        try:
            # Extract the date and time from the filename
            file_time = datetime.strptime(file, date_format)
            # Check if this file is the latest one
            if latest_time is None or file_time > latest_time:
                latest_time = file_time
                latest_file = file
        except ValueError:
            # Skip files that do not match the date format
            continue
    
    return latest_file

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
                                        condition_input_data,
                                        checkpoint_path,
                                        milestone,
                                        mask_val):
    model = Unet1D(
        seq_length=seq_length,
        dim=unet_dim,
        channels=channel,
        dim_mults=unet_dim_mults,
        embed_class_layers_dims=embed_class_layers_dims,
        class_dim=class_dim,
        cond_drop_prob=cond_drop_prob,
        mask_val=mask_val,
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
        results_folder=checkpoint_path, # Do not need to set batch size => automatically set through dimension of class variable
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
    # Make sure u is not larger than 1
    u[u>1] = 1
    return alpha, theta, u

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for diffusion models")
    parser.add_argument('--mask_val',
                        type=str,
                        default=-1.0,
                        help='Mask value for unconditional data generation')
    parser.add_argument('--timesteps',
                        type=int,
                        default="500",
                        help='Nmber of Diffusion timesteps')
    parser.add_argument('--data_num',
                        type=int,
                        default="100000",
                        help='Number of Training Data')
    parser.add_argument('--sample_num',
                        type=int,
                        default="10000",
                        help='Number of initial guesses to be sampled')
    
    args = parser.parse_args()

    timesteps = int(args.timesteps)
    data_num = int(args.data_num)
    sample_num = int(args.sample_num)
    mask_val = float(args.mask_val)

    main(timesteps,data_num,sample_num,mask_val)
