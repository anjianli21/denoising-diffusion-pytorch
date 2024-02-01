#!/bin/bash

# Define the variable
training_data_type="input_obs_output_time_control_obj_12_data_114k_constraint_1.0_v2"

export CUDA_VISIBLE_DEVICES=3
export WANDB_DIR="/root/autodl-tmp/project/diffusion/fixed_car_vary_obs/wandb/$training_data_type"
export WANDB_MODE=offline
python /root/project/denoising-diffusion-pytorch/run/train_classifier_free_cond_1d_constraint_v2.py --constraint_violation_weight=1.0 --training_data_num 114570 --result_folder /root/autodl-tmp/project/diffusion/fixed_car_vary_obs/results --data_path /root/autodl-tmp/project/diffusion/fixed_car_vary_obs/Data/obstacle_time_control_data_obj_12_num_114570.pkl --class_dim 6 --channel_num 1 --seq_length 81 --wandb_project_name diffusion_fixed_car_vary_obs_$training_data_type --training_data_type $training_data_type
