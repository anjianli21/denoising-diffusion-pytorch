#!/bin/bash

# Define the variable
training_data_type="input_obs_output_time_control_obj_12"

conda activate pydylan
export WANDB_DIR="/home/anjian/Desktop/project/denoising-diffusion-pytorch/wandb/fixed_car_vary_obs/wandb/$training_data_type"
export WANDB_MODE=offline
python /home/anjian/Desktop/project/denoising-diffusion-pytorch/run/train_classifier_free_cond_1d.py --training_data_num 77938 --machine ubuntu --data_path /home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/obstacle_time_control_data_obj_12.pkl --class_dim 6 --channel_num 1 --seq_length 81 --wandb_project_name diffusion_fixed_car_vary_obs_$training_data_type --training_data_type $training_data_type
