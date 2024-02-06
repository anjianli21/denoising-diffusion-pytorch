#!/bin/bash

# Define the variable
training_data_type="full_data_202k_constraint_weight_0.01_condscale_6_seed_1"

export CUDA_VISIBLE_DEVICES=5
export WANDB_DIR="/root/autodl-tmp/project/diffusion/tabletop/wandb/$training_data_type"
export WANDB_MODE=offline
python /root/project/denoising-diffusion-pytorch/run/train_classifier_free_cond_1d_constraint_tabletop.py --training_random_seed 1 --constraint_violation_weight 0.01 --constraint_condscale 6.0 --training_data_num 202654 --result_folder /root/autodl-tmp/project/diffusion/tabletop/results --data_path /root/autodl-tmp/project/diffusion/tabletop/Data/obstacle_goal_time_control_data_obj_6_num_202654.pkl --class_dim 14 --channel_num 1 --seq_length 81 --wandb_project_name diffusion_tabletop_$training_data_type --training_data_type $training_data_type
