import sys
import time

#sys.path.append('/home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/python_scripts')

#from models import *  # TODO, import CVAE models and lstm models, from '/home/anjian/Desktop/project/generative_trajectory_optimization'
from classifier_free_guidance_cond_1d_improved_constrained_diffusion import Unet1D, GaussianDiffusion1D, Trainer1D

import numpy as np
import pickle
import torch

def main():

    # For icml
    checkpoint_path_list = [
                       f"/scratch/gpfs/jg3607/Diffusion_model/indirect/results/cr3bp_vanilla_diffusion_seed_0/unet_20_mults_4_4_8_embed_class_40_80_timesteps_100_objective_pred_noise_batch_size_27000_cond_drop_0.1_mask_val_0.0/2024-06-17_19-58-35"
    ]

    milestone_list = ["epoch-194"]

    data_num_list = [270000]

    sample_num = 1000000
    diffusion_w = 5.0
    thrust = 0.85
    diffusion_type = "diffusion_indirect"
    # thrust_list = [0.15, 0.35, 0.45, 0.65, 0.85]

    save_warmstart_data = True

    for i in range(len(checkpoint_path_list)):
        data_num = data_num_list[i]
        checkpoint_path = checkpoint_path_list[i]
        milestone = milestone_list[i]

        unet_dim = 20
        unet_dim_mults = "4,4,8"
        unet_dim_mults = tuple(map(int, unet_dim_mults.split(',')))
        embed_class_layers_dims = "40,80"
        embed_class_layers_dims = tuple(map(int, embed_class_layers_dims.split(',')))
        timesteps = 100
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
            parent_path = "/home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/results/generated_initializations/indirect"
            cr3bp_time_mass_alpha_control_path = f"{parent_path}/cr3bp_thrust_{thrust}_{diffusion_type}_w_{diffusion_w}_training_num_{data_num}_num_{sample_num}.pkl"
            with open(cr3bp_time_mass_alpha_control_path, "wb") as fp:  # write pickle
                pickle.dump(full_solution, fp)
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
    main()
