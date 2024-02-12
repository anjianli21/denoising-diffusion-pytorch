#!/bin/bash
#SBATCH --job-name=diffusion
#SBATCH --output=/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/output/tabletop_v2/dr_v0.%A.%a.out
#SBATCH --error=/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/output/tabletop_v2/dr_v0.%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=25G
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:59
#SBATCH --mail-type=all
#SBATCH --mail-user=anjianl@princeton.edu
#SBATCH --array=1

module purge
module load anaconda3/2021.11
conda activate pydylan

# Define the variable
training_data_type="tabletop_v2_diffusion_seed_1"

export WANDB_DIR="/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/wandb/tabletop_v2/$training_data_type"
export WANDB_MODE=offline
python /home/al5844/desktop/project/denoising-diffusion-pytorch/run/train_classifier_free_cond_1d.py --training_random_seed 1 --training_data_num 237370 --result_folder /scratch/gpfs/al5844/project/denoising-diffusion-pytorch/results/tabletop_v2/results --data_path /scratch/gpfs/al5844/project/denoising-diffusion-pytorch/Data/tabletop_v2_obs_goal_time_control_num_237370.pkl --class_dim 14 --channel_num 1 --seq_length 161 --wandb_project_name diffusion_tabletop_v2_$training_data_type --training_data_type $training_data_type
