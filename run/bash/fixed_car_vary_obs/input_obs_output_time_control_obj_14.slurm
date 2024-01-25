#!/bin/bash
#SBATCH --job-name=diffusion1d
#SBATCH --output=/scratch/gpfs/al5844/project/diffusion/fixed_car_vary_obs/output/input_obs_output_time_control/dr_v0.%A.%a.out
#SBATCH --error=/scratch/gpfs/al5844/project/diffusion/fixed_car_vary_obs/output/input_obs_output_time_control/dr_v0.%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=09:59:59
#SBATCH --mail-type=all
#SBATCH --mail-user=anjianl@princeton.edu
#SBATCH --array=1

# Define the variable
training_data_type="input_obs_output_time_control_obj_14"

echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"

module purge
module load anaconda3/2021.11
conda activate pydylan
export WANDB_DIR="/scratch/gpfs/al5844/project/diffusion/fixed_car_vary_obs/wandb/$training_data_type"
export WANDB_MODE=offline
python /home/al5844/desktop/project/denoising-diffusion-pytorch/run/train_classifier_free_cond_1d.py --training_data_num 103452 --machine della --data_path /scratch/gpfs/al5844/project/diffusion/fixed_car_vary_obs/Data/obstacle_time_control_data_obj_14_num_103452.pkl --class_dim 6 --channel_num 1 --seq_length 81 --wandb_project_name diffusion_fixed_car_vary_obs_$training_data_type --training_data_type $training_data_type
