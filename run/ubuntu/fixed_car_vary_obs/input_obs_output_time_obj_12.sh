#!/bin/bash

# Define the variable
training_data_type="input_obs_output_time_obj_12"

export WANDB_DIR="/home/anjian/Desktop/project/denoising-diffusion-pytorch/results/from_della/diffusion/fixed_car_vary_obs/wandb/$training_data_type"
export WANDB_MODE=offline
python /home/anjian/Desktop/project/denoising-diffusion-pytorch/run/train_classifier_free_cond_1d.py --max_epoch 20 --result_folder="/home/anjian/Desktop/project/denoising-diffusion-pytorch/results/from_della/diffusion/fixed_car_vary_obs/results" --training_data_num 77938 --data_path /home/anjian/Desktop/project/denoising-diffusion-pytorch/data/fixed_car_vary_obs/obstacle_time_data_obj_12.pkl --class_dim 6 --channel_num 1 --seq_length 1 --wandb_project_name diffusion_fixed_car_vary_obs_$training_data_type --training_data_type $training_data_type
