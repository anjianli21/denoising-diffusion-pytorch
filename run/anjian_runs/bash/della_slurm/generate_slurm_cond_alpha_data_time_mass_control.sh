#!/bin/bash

# Define arrays of hyperparameter options
unet_dim=128
unet_dim_mults=("4,4,8" "4,8,8")
embed_class_layers_dims=("64,128" "256,512")
cond_drop_prob_list=(0.1 0.2 0.3)
training_data_type="cond_alpha_data_time_mass_control"
wandb_project_name="diffusion_for_cr3bp_cond_alpha_data_time_mass_control"
data_path="/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/Data/cr3bp_alpha_time_mass_control.pkl"
seq_length=64

# Counter for job files
job_counter=1

# Iterate over hyperparameters
for unet_dim_mult in "${unet_dim_mults[@]}"; do
  for embed_class_layer_dim in "${embed_class_layers_dims[@]}"; do
    for cond_drop_prob in "${cond_drop_prob_list[@]}"; do

      # Create a job file for each combination
      job_file="${training_data_type}/slurm_job_${job_counter}.slurm"
      cat > "$job_file" <<EOF
#!/bin/bash
#SBATCH --job-name=diffusion1d
#SBATCH --output=/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/output/$training_data_type/dr_v0.%A.%a.out
#SBATCH --error=/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/output/$training_data_type/dr_v0.%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=09:59:59
#SBATCH --mail-type=all
#SBATCH --mail-user=anjianl@princeton.edu
#SBATCH --array=$job_counter

echo "SLURM_JOBID: \$SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: \$SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID: \$SLURM_ARRAY_JOB_ID"

module purge
module load anaconda3/2021.11
conda activate pydylan
export WANDB_DIR='/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/wandb/$training_data_type'
export WANDB_MODE=offline
python /home/al5844/desktop/project/denoising-diffusion-pytorch/run/train_classifier_free_cond_1d.py --unet_dim $unet_dim --unet_dim_mults "$unet_dim_mult" --embed_class_layers_dims "$embed_class_layer_dim" --timesteps 500 --objective pred_noise --batch_size 512 --machine della --data_path $data_path --class_dim 1 --channel_num 1 --seq_length $seq_length --wandb_project_name $wandb_project_name --training_data_type $training_data_type --cond_drop_prob $cond_drop_prob
EOF

      # Increment job file counter
      ((job_counter++))
    done
  done
done

echo "Generated $((job_counter - 1)) Slurm job files."
