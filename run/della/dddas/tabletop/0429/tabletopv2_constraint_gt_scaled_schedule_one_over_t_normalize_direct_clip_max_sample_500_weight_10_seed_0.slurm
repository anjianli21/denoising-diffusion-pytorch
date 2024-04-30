#!/bin/bash
#SBATCH --job-name=diffusion
#SBATCH --output=/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/output/tabletop_v2/dr_v0.%A.%a.out
#SBATCH --error=/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/output/tabletop_v2/dr_v0.%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=25G
#SBATCH --gres=gpu:1
#SBATCH --time=59:59:59
#SBATCH --mail-type=all
#SBATCH --mail-user=anjianl@princeton.edu
#SBATCH --array=1

module purge
module load anaconda3/2021.11
conda activate pydylan

export WANDB_DIR="/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/wandb/tabletop_v2/tabletopv2_constraint_gt_scaled_schedule_one_over_t_normalize_direct_clip_max_sample_500_weight_10_seed_0"
export WANDB_MODE=offline
python /home/al5844/desktop/project/denoising-diffusion-pytorch/run/train_classifier_free_cond_1d_improved_constrained_diffusion.py --constraint_loss_type gt_scaled --constraint_violation_weight 10 --constraint_loss_scheduling=one_over_t --max_sample_step_with_constraint_loss=500 --normalize_xt_type=direct_clip --training_random_seed 0 --wandb_project_name constrained_diffusion_tabletopv2 --task_type tabletop --training_data_num 237370 --result_folder /scratch/gpfs/al5844/project/denoising-diffusion-pytorch/results/tabletop_v2/results --data_path /scratch/gpfs/al5844/project/denoising-diffusion-pytorch/Data/tabletop_v2_obs_goal_time_control_num_237370.pkl --class_dim 14 --channel_num 1 --seq_length 161 --training_data_type $training_data_type
