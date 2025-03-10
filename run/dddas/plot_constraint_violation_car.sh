#!/bin/bash

conda activate pydylan

cd /home/anjian/Desktop/project/denoising-diffusion-pytorch
export PYTHONPATH=$PYTHONPATH:/home/anjian/Desktop/project/denoising-diffusion-pytorch

WANDB_MODE=offline python /home/anjian/Desktop/project/denoising-diffusion-pytorch/run/train_classifier_free_cond_1d_improved_constrained_diffusion.py \
    --training_random_seed=0 \
    --training_data_num=114570 \
    --result_folder=results/car/results \
    --data_path=data/fixed_car_vary_obs/obstacle_time_control_data_obj_12_num_114570.pkl \
    --class_dim=6 \
    --channel_num=1 \
    --seq_length=81 \
    --training_data_type=constrained_diffusion_seed_0 \
    --constraint_violation_weight=1 \
    --batch_size=128 \
    --constraint_loss_type=gt_scaled \
    --task_type=car \
    --constraint_loss_scheduling=NA \
    --max_sample_step_with_constraint_loss=500 \
    --normalize_xt_type=direct_clip \

