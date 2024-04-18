#!/bin/bash
#SBATCH --job-name=diffusion
#SBATCH --output=/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/output/car/dr_v0.%A.%a.out
#SBATCH --error=/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/output/car/dr_v0.%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=25G
#SBATCH --gres=gpu:1
#SBATCH --time=19:59:59
#SBATCH --mail-type=all
#SBATCH --mail-user=anjianl@princeton.edu
#SBATCH --array=1

module purge
module load anaconda3/2021.11
conda activate pydylan

# Define the variable
training_data_type="car_constraint_one_over_t_weight_01_seed_0"

export WANDB_DIR="/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/wandb/car/$training_data_type"
export WANDB_MODE=offline
python /home/al5844/desktop/project/denoising-diffusion-pytorch/run/train_classifier_free_cond_1d_improved_constrained_diffusion.py --constraint_loss_type one_over_t --training_random_seed 0 --constraint_violation_weight 0.1 --wandb_project_name constrained_diffusion_car --task_type car --training_data_num 114570 --result_folder /scratch/gpfs/al5844/project/denoising-diffusion-pytorch/results/car/results --data_path /scratch/gpfs/al5844/project/denoising-diffusion-pytorch/Data/obstacle_time_control_data_obj_12_num_114570.pkl --class_dim 6 --channel_num 1 --seq_length 81 --training_data_type $training_data_type
