#!/bin/bash

# Define the variable
training_data_type="input_obs_goal_output_time_control_obj_6_constraint_1.0_v2"

export CUDA_VISIBLE_DEVICES=1
export WANDB_DIR="/root/autodl-tmp/project/diffusion/tabletop/wandb/$training_data_type"
export WANDB_MODE=offline
python /root/project/denoising-diffusion-pytorch/run/train_classifier_free_cond_1d_constraint_tabletop_v2.py --constraint_violation_weight=1.0 --training_data_num 202654 --result_folder /root/autodl-tmp/project/diffusion/tabletop/results --data_path /root/autodl-tmp/project/diffusion/tabletop/Data/obstacle_goal_time_control_data_obj_6_num_202654.pkl --class_dim 14 --channel_num 1 --seq_length 81 --wandb_project_name diffusion_tabletop_$training_data_type --training_data_type $training_data_type