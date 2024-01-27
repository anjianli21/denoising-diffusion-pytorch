#!/bin/bash

# Define the variable
training_data_type="input_obs_output_time_obj_12"

export WANDB_DIR="/root/autodl-tmp/project/diffusion/fixed_car_vary_obs/wandb/$training_data_type"
export WANDB_MODE=offline
python /root/project/denoising-diffusion-pytorch/run/train_classifier_free_cond_1d.py --training_data_num 77938 --machine "autodl-car" --data_path /root/autodl-tmp/project/diffusion/fixed_car_vary_obs/Data/obstacle_time_data_obj_12.pkl --class_dim 6 --channel_num 1 --seq_length 1 --wandb_project_name diffusion_fixed_car_vary_obs_$training_data_type --training_data_type $training_data_type
