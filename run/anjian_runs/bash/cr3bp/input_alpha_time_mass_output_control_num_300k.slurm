#!/bin/bash
#SBATCH --job-name=diffusion1d
#SBATCH --output=/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/output/cond_alpha_time_mass_data_control/dr_v0.%A.%a.out
#SBATCH --error=/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/output/cond_alpha_time_mass_data_control/dr_v0.%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=09:59:59
#SBATCH --mail-type=all
#SBATCH --mail-user=anjianl@princeton.edu
#SBATCH --array=1

echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"

# Define the variable
training_data_type="cond_alpha_time_mass_data_control_300k"

module purge
module load anaconda3/2021.11
conda activate pydylan
export WANDB_DIR="/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/wandb/$training_data_type"
export WANDB_MODE=offline
python /home/al5844/desktop/project/denoising-diffusion-pytorch/run/train_classifier_free_cond_1d.py --machine della --data_path /scratch/gpfs/al5844/project/denoising-diffusion-pytorch/Data/cr3bp_alpha_time_mass_control.pkl --class_dim 5 --channel_num 3 --seq_length 20 --wandb_project_name diffusion_for_cr3bp_$training_data_type --training_data_type $training_data_type --training_data_num 300000
