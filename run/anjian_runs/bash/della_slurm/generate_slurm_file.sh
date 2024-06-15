#!/bin/bash

# Define arrays of hyperparameter options
unet_dims=(32 64 128)
unet_dim_mults=("1,2,4" "1,2,8" "1,4,8" "2,4,8" "4,4,8" "4,8,8")
embed_class_layers_dims=("64,128" "128,256" "256,512" "64,128,256" "128,256,512")

# Counter for job files
job_counter=1

# Iterate over hyperparameters
for unet_dim in "${unet_dims[@]}"; do
  for unet_dim_mult in "${unet_dim_mults[@]}"; do
    for embed_class_layer_dim in "${embed_class_layers_dims[@]}"; do

      # Create a job file for each combination
      job_file="slurm_job_${job_counter}.slurm"
      cat > "$job_file" <<EOF
#!/bin/bash
#SBATCH --job-name=diffusion1d
#SBATCH --output=/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/output/dr_v0.%A.%a.out
#SBATCH --error=/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/output/dr_v0.%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=05:59:59
#SBATCH --mail-type=all
#SBATCH --mail-user=anjianl@princeton.edu
#SBATCH --array=$job_counter

echo "SLURM_JOBID: \$SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: \$SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID: \$SLURM_ARRAY_JOB_ID"

module purge
module load anaconda3/2021.11
conda activate pydylan
export WANDB_DIR='/scratch/gpfs/al5844/project/denoising-diffusion-pytorch'
export WANDB_MODE=offline
python /home/al5844/desktop/project/denoising-diffusion-pytorch/run/train_classifier_free_cond_1d.py --unet_dim $unet_dim --unet_dim_mults "$unet_dim_mult" --embed_class_layers_dims "$embed_class_layer_dim" --timesteps 1000 --objective pred_v --batch_size 1024 --machine della --data_path /scratch/gpfs/al5844/project/denoising-diffusion-pytorch/Data/cr3bp_time_mass_alpha_control_part_4_250k_each.pkl
EOF

      # Increment job file counter
      ((job_counter++))
    done
  done
done

echo "Generated $((job_counter - 1)) Slurm job files."
