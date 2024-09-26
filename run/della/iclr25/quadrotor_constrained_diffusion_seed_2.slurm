#!/bin/bash
#SBATCH --job-name=diffusion
#SBATCH --output=/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/output/quadrotor/dr_v0.%A.%a.out
#SBATCH --error=/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/output/quadrotor/dr_v0.%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --time=30:59:59
#SBATCH --mail-type=all
#SBATCH --mail-user=anjianl@princeton.edu
#SBATCH --array=1

module purge
module load anaconda3/2021.11
conda activate pydylan

# Define the variable
training_data_type="quadrotor_constrained_diffusion_seed_2"

export WANDB_DIR="/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/wandb/quadrotor/$training_data_type"
export WANDB_MODE=offline
python /home/al5844/desktop/project/denoising-diffusion-pytorch/run/train_classifier_free_cond_1d_improved_constrained_diffusion.py --training_random_seed 2 --training_data_num 179219 --constraint_loss_scheduling=one_over_t --constraint_loss_type=predict_x0_violation --constraint_violation_weight=0.1 --task_type=quadrotor --result_folder /scratch/gpfs/al5844/project/denoising-diffusion-pytorch/results/quadrotor/results --data_path /scratch/gpfs/al5844/project/denoising-diffusion-pytorch/Data/quadrotor_obs_goalperturb_time_control_num_179219.pkl --class_dim 17 --channel_num 1 --seq_length 241 --wandb_project_name diffusion_quadrotor_$training_data_type --training_data_type $training_data_type
