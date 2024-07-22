#!/bin/bash
#SBATCH --job-name=2M_0.95_w1.0_u64
#SBATCH --output=/home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/run/dr_v0.%A.%a.out
#SBATCH --error=/home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/run/dr_v0.%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-user=jg3607@princeton.edu
#SBATCH --array=1

echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"


module purge
module load anaconda3/2022.5
conda activate torch-env
module load boost/1.73.0 sq

export WANDB_DIR="/scratch/gpfs/jg3607/Diffusion_model/indirect/sampling/"
export WANDB_MODE=offline   

python /home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/python_scripts/sample_data_diffusion_indirect.py --unet_dim 64 --diffusion_w "1.0" --batch_size "1024" --embed_class_layers_dims "128,256" --timesteps 1000 --data_num 270000 --sample_num 1000000 --thrust_level 0.95  
