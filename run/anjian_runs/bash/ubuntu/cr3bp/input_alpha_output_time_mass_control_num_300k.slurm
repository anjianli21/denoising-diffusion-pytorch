#!/bin/bash

# Define the variable
training_data_type="cond_alpha_data_time_mass_control_300k"

conda activate pydylan
export WANDB_DIR="/home/anjian/Desktop/project/denoising-diffusion-pytorch/wandb/cr3bp/wandb/$training_data_type"
export WANDB_MODE=offline
python /home/anjian/Desktop/project/denoising-diffusion-pytorch/run/train_classifier_free_cond_1d.py --machine ubuntu --data_path /home/anjian/Desktop/project/denoising-diffusion-pytorch/data/CR3BP/cr3bp_alpha_time_mass_control.pkl --class_dim 1 --channel_num 1 --seq_length 64 --wandb_project_name diffusion_for_cr3bp_$training_data_type --training_data_type $training_data_type --training_data_num 300000
