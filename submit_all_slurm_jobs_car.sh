#!/bin/bash

sbatch run/della/dddas/car/0429/car_experiment_gt_log_likelihood_schedule_NA_normalize_direct_clip_max_sample_100_weight_1_seed_0.slurm
sbatch run/della/dddas/car/0429/car_experiment_gt_log_likelihood_schedule_one_over_t_normalize_direct_clip_max_sample_500_weight_10_seed_0.slurm
sbatch run/della/dddas/car/0429/car_experiment_gt_log_likelihood_schedule_sqrt_bar_alpha_normalize_direct_clip_max_sample_500_weight_1_seed_0.slurm
sbatch run/della/dddas/car/0429/car_experiment_gt_std_schedule_NA_normalize_direct_clip_max_sample_100_weight_10_seed_0.slurm
sbatch run/della/dddas/car/0429/car_experiment_gt_std_schedule_one_over_t_normalize_direct_clip_max_sample_500_weight_10_seed_0.slurm
sbatch run/della/dddas/car/0429/car_experiment_gt_std_schedule_sqrt_bar_alpha_normalize_direct_clip_max_sample_500_weight_1_seed_0.slurm
sbatch run/della/dddas/car/0429/car_experiment_predict_x0_violation_schedule_sqrt_bar_alpha_normalize_direct_clip_max_sample_500_weight_0001_seed_0.slurm
