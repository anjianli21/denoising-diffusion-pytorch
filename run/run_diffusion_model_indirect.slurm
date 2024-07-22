#!/bin/bash
#SBATCH --job-name=u64BS512
#SBATCH --output=/scratch/gpfs/jg3607/Diffusion_model/indirect/output/dr_v0.%A.%a.out
#SBATCH --error=/scratch/gpfs/jg3607/Diffusion_model/indirect/output/dr_v0.%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mail-user=jg3607@princeton.edu
#SBATCH --array=1

echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"


module purge
module load anaconda3/2022.5
conda activate torch-env
module load boost/1.73.0 

export WANDB_DIR="/scratch/gpfs/jg3607/Diffusion_model/indirect"
export WANDB_MODE=offline

python /home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/python_scripts/train_classifier_free_cond_1d_improved_constrained_diffusion.py --wandb_project_name "Diffusion indirect" --unet_dim 64 --embed_class_layers_dims "128,256" --timesteps 1000 --training_random_seed 0 --result_folder /scratch/gpfs/jg3607/Diffusion_model/indirect/results --data_path /home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/data/indirect/training_data_indirect_270000.pkl --max_epoch 200 --class_dim 1 --channel_num 1 --seq_length 6 --batch_size 512 --constraint_loss_type NA --task_type cr3bp --training_data_type cr3bp_vanilla_diffusion_seed_0 --training_data_num 270000 
