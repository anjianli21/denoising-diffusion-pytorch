#!/bin/bash
#SBATCH --job-name=diffusion1d 	      # create a short name for your job
#SBATCH --output=/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/output/dr_v0.%A.%a.out  # STDOUT file
#SBATCH --error=/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/output/dr_v0.%A.%a.err  # STDERR file
#SBATCH --nodes=1                	# node count
#SBATCH --ntasks-per-node=1       # total number of tasks per node
#SBATCH --cpus-per-task=2       	# cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G        		    # total memory per node
#SBATCH --gres=gpu:1             	# number of gpus per node
#SBATCH --time=00:09:59          	# total run time limit (HH:MM:SS)
#SBATCH --mail-type=all 	      	# send email when job starts, fails and ends
#SBATCH --mail-user=anjianl@princeton.edu
#SBATCH --array=0

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

module purge
module load anaconda3/2021.11
conda activate pydylan
export WANDB_DIR='/scratch/gpfs/al5844/project/denoising-diffusion-pytorch'
export WANDB_MODE=offline
python /home/al5844/desktop/project/denoising-diffusion-pytorch/run/train_classifier_free_cond_1d.py --unet_dim 64 --unet_dim_mults "(1, 2, 4)" --embed_class_layers_dims (256, 256) --timesteps 1000 --objective pred_v --batch_size 1024 --machine della --data_path /scratch/gpfs/al5844/project/denoising-diffusion-pytorch/Data/cr3bp_time_mass_alpha_control_part_4_250k_each.pkl
python /home/al5844/desktop/project/denoising-diffusion-pytorch/run/train_classifier_free_cond_1d.py --unet_dim 64 --unet_dim_mults "1,2,4" --embed_class_layers_dims "256,256" --timesteps 1000 --objective pred_v --batch_size 1024 --machine della --data_path /scratch/gpfs/al5844/project/denoising-diffusion-pytorch/Data/cr3bp_time_mass_alpha_control_part_4_250k_each.pkl
