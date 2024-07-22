import time

#sys.path.append('/home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/python_scripts')

#from models import *  # TODO, import CVAE models and lstm models, from '/home/anjian/Desktop/project/generative_trajectory_optimization'
from classifier_free_guidance_cond_1d_improved_constrained_diffusion import Unet1D, GaussianDiffusion1D, Trainer1D

import numpy as np
import pickle
import torch
import os
import argparse
import re
from datetime import datetime

def main(unet_dim,embed_class_layers_dims,timesteps,data_num,sample_num,thrust,diffusion_w,batch_size):

    # For icml
    unet_dim_mults = "4,4,8"
    unet_dim_mults = tuple(map(int, unet_dim_mults.split(',')))
    unet_dim_mults_in_str = "_".join(map(str, unet_dim_mults))
    embed_class_layers_dims = tuple(map(int, embed_class_layers_dims.split(',')))
    embed_class_layers_dims_in_str = "_".join(map(str, embed_class_layers_dims))
    checkpoint_path = f"/scratch/gpfs/jg3607/Diffusion_model/indirect/results/cr3bp_vanilla_diffusion_seed_0/unet_{unet_dim}_mults_{unet_dim_mults_in_str}_embed_class_{embed_class_layers_dims_in_str}_timesteps_{timesteps}_batch_size_{batch_size}_cond_drop_0.1_mask_val_0.0_train_data_{data_num}/"

    folder_name = get_latest_file(checkpoint_path)
    checkpoint_path = checkpoint_path + folder_name
    milestone = get_milestone_string(checkpoint_path)
    diffusion_type = "diffusion_indirect"

    save_warmstart_data = True

    objective = "pred_noise"
    mask_val = 0

    class_dim = 1
    channel = 1
    seq_length = 6
    cond_drop_prob = 0.1

    # Configure input data
    thrust_normalized = thrust

    alpha_data_normalized = thrust_normalized * torch.ones(size=(sample_num, 1), dtype=torch.float32)

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
    min_shooting_time = 38.0
    max_shooting_time = 900.0  # 50.0

    min_coast_time = 0
    max_coast_time = 4.105528 # Terminal orbit period => value can not be bigger since no optimization is happening
    
    min_costates = np.array([-0.5431, -0.0034, -0.0030, -0.3903]) # rounded down to 4 digits after the point
    max_costates = np.array([-0.2608, 0.0026 ,0.0047, -0.1903]) # rounded up to 4 digits after the point

    # Unnormalize times and costates
    full_solution[:, 0] = full_solution[:, 0] * (max_shooting_time - min_shooting_time) + min_shooting_time
    full_solution[:, 1] = full_solution[:, 1] * (max_coast_time - min_coast_time) + min_coast_time
    full_solution[:, 2:6] = full_solution[:, 2:6] * (max_costates - min_costates) + min_costates
    # Add 0 initial coast
    full_solution = np.insert(full_solution,1,np.zeros(sample_num),axis=1)
    # Add 0 z costates
    full_solution = np.insert(full_solution,5,np.zeros(sample_num),axis=1)
    full_solution = np.insert(full_solution,8,np.zeros(sample_num),axis=1)
    # Add mass costate -1   
    full_solution = np.insert(full_solution,9,-np.ones(sample_num),axis=1)

    if save_warmstart_data:
        parent_path = f"/home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/results/generated_initializations/indirect/unet_{unet_dim}_mults_{unet_dim_mults_in_str}_embed_class_{embed_class_layers_dims_in_str}_timesteps_{timesteps}_batch_size_{batch_size}_cond_drop_0.1_mask_val_0.0"
        os.makedirs(parent_path, exist_ok=True)
        cr3bp_time_mass_alpha_control_path = f"{parent_path}/cr3bp_thrust_{thrust}_{diffusion_type}_w_{diffusion_w}_training_num_{data_num}_num_{sample_num}.pkl"
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for diffusion models")
    parser.add_argument('--unet_dim',
                        type=int,
                        default=64,
                        help='Dimension of the first layer of Unet')
    parser.add_argument('--embed_class_layers_dims',
                        type=str,
                        default="128,256",
                        help='List of dimension for embedding class layers')
    parser.add_argument('--timesteps',
                        type=str,
                        default="500",
                        help='Nmber of Diffusion timesteps')
    parser.add_argument('--data_num',
                        type=str,
                        default="270000",
                        help='Number of Training Data')
    parser.add_argument('--sample_num',
                        type=str,
                        default="10000",
                        help='Number of Diffusion timesteps')
    parser.add_argument('--thrust_level',
                        type=str,
                        default="0.85",
                        help='Level of the thrust')
    parser.add_argument('--diffusion_w',
                        type=str,
                        default="1.0",
                        help='w parameter for classifier free guidance sampling')
    parser.add_argument('--batch_size',
                        type=str,
                        default="1024",
                        help='batch size that was used for diffusion model training')
    
    args = parser.parse_args()


    unet_dim = int(args.unet_dim)
    embed_class_layers_dims = args.embed_class_layers_dims
    timesteps = int(args.timesteps)
    data_num = int(args.data_num)
    sample_num = int(args.sample_num)
    thrust = float(args.thrust_level)
    diffusion_w = float(args.diffusion_w)
    batch_size = int(args.batch_size)
    
    main(unet_dim,embed_class_layers_dims,timesteps,data_num,sample_num,thrust,diffusion_w,batch_size)
