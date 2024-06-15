#!/bin/bash

# Define the variable
training_data_type="cond_alpha_data_time_mass_control_300k_seed_0"

export CUDA_VISIBLE_DEVICES=3
export WANDB_DIR="/root/autodl-tmp/project/diffusion/cr3bp/wandb/$training_data_type"
export WANDB_MODE=offline
python /root/project/denoising-diffusion-pytorch/run/train_classifier_free_cond_1d.py --training_random_seed 0 --result_folder /root/autodl-tmp/project/diffusion/cr3bp/results --data_path /root/autodl-tmp/project/diffusion/cr3bp/Data/cr3bp_alpha_time_mass_control.pkl --class_dim 1 --channel_num 1 --seq_length 64 --wandb_project_name diffusion_for_cr3bp_$training_data_type --training_data_type $training_data_type --training_data_num 300000
