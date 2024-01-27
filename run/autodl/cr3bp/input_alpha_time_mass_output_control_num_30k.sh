#!/bin/bash

# Define the variable
training_data_type="cond_alpha_time_mass_data_control_30k"

export CUDA_VISIBLE_DEVICES=4
export WANDB_DIR="/root/autodl-tmp/project/diffusion/cr3bp/wandb/$training_data_type"
export WANDB_MODE=offline
python /root/project/denoising-diffusion-pytorch/run/train_classifier_free_cond_1d.py --machine autodl-cr3bp --data_path /root/autodl-tmp/project/diffusion/cr3bp/Data/cr3bp_alpha_time_mass_control.pkl --class_dim 5 --channel_num 3 --seq_length 20 --wandb_project_name diffusion_for_cr3bp_$training_data_type --training_data_type $training_data_type --training_data_num 30000
