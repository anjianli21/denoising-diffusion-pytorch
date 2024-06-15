#!/bin/bash
#SBATCH --job-name=diffusion1d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-user=jg3607@princeton.edu
#SBATCH --array=1

echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"

# Define the variable
training_data_type="cond_alpha_data_time_mass_control_30k"

module purge
module load anaconda3/2021.11
conda activate indirect
module load boost/1.73.0

export WANDB_DIR="/home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/wandb"
export WANDB_MODE=offline

python /home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/run/train_classifier_free_cond_1d_improved_constrained_diffusion.py  --training_random_seed 0 --result_folder /home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/results/cr3bp/results/ --data_path /home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/data/CR3BP/cr3bp_alpha_time_mass_control.pkl --max_epoch 10 --class_dim 1 --channel_num 1 --seq_length 64 --batch_size 10 --constraint_loss_type NA --task_type cr3bp --training_data_type cr3bp_vanilla_diffusion_seed_0 --training_data_num 100
