#!/bin/bash

# Define the variable
training_data_type="input_obs_output_time_control_obj_12"

export WANDB_DIR="./data/time_control"
export WANDB_MODE=offline
python run/train_classifier_free_cond_1d.py --result_folder 'data/result_time_control/' --training_data_num 77938 --machine "autodl-car" --data_path data/obstacle_time_control_data_obj_12.pkl --class_dim 6 --channel_num 1 --seq_length 81 --wandb_project_name diffusion_fixed_car_vary_obs_$training_data_type --training_data_type $training_data_type
