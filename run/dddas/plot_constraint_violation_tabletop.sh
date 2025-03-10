#!/bin/bash

conda activate pydylan

cd /home/anjian/Desktop/project/denoising-diffusion-pytorch
export PYTHONPATH=$PYTHONPATH:/home/anjian/Desktop/project/denoising-diffusion-pytorch

WANDB_MODE=offline python /home/anjian/Desktop/project/denoising-diffusion-pytorch/run/train_classifier_free_cond_1d_improved_constrained_diffusion.py \
    --training_random_seed=0 \
    --training_data_num=237370 \
    --result_folder=results/tabletop_v2/results \
    --data_path=data/tabletop_v2/tabletop_v2_obs_goal_time_control_num_237370.pkl \
    --class_dim=14 \
    --channel_num=1 \
    --seq_length=161 \
    --wandb_project_name=diffusion_tabletop_v2_diffusion_seed_0 \
    --training_data_type=constrained_diffusion_seed_0 \
    --batch_size=128 \
    --constraint_violation_weight=10.0 \
    --constraint_condscale=6.0 \
    --task_type=tabletop \
    --constraint_loss_type=gt_scaled \
    --constraint_loss_scheduling=NA \
    --max_sample_step_with_constraint_loss=500 \
    --normalize_xt_type=direct_clip \

