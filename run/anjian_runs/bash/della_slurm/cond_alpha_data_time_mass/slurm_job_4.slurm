#!/bin/bash
#SBATCH --job-name=diffusion1d
#SBATCH --output=/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/output/cond_alpha_data_time_mass/dr_v0.%A.%a.out
#SBATCH --error=/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/output/cond_alpha_data_time_mass/dr_v0.%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=09:59:59
#SBATCH --mail-type=all
#SBATCH --mail-user=anjianl@princeton.edu
#SBATCH --array=4

echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"

module purge
module load anaconda3/2021.11
conda activate pydylan
export WANDB_DIR='/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/wandb/cond_alpha_data_time_mass'
export WANDB_MODE=offline
python /home/al5844/desktop/project/denoising-diffusion-pytorch/run/train_classifier_free_cond_1d.py --unet_dim 128 --unet_dim_mults "4,4,8" --embed_class_layers_dims "256,512" --timesteps 500 --objective pred_noise --batch_size 512 --machine della --data_path /scratch/gpfs/al5844/project/denoising-diffusion-pytorch/Data/cr3bp_alpha_time_mass.pkl --class_dim 1 --channel_num 1 --seq_length 4 --wandb_project_name diffusion_for_cr3bp_cond_alpha_data_time_mass --training_data_type cond_alpha_data_time_mass --cond_drop_prob 0.1
