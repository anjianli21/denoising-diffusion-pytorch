#!/bin/bash
#SBATCH --job-name=diff_bou_sam
#SBATCH --output=/scratch/gpfs/jg3607/Diffusion_model/boundary/output/dr_v0.%A.%a.out
#SBATCH --error=/scratch/gpfs/jg3607/Diffusion_model/boundary/output/dr_v0.%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mail-user=jg3607@princeton.edu
#SBATCH --array=1

echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"


module purge
module load anaconda3/2022.5
conda activate torch-env
module load boost/1.73.0 

export WANDB_DIR="/scratch/gpfs/jg3607/Diffusion_model/boundary/sampling/"
export WANDB_MODE=offline

python /home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/python_scripts/sample_data_diffusion_boundary.py 
