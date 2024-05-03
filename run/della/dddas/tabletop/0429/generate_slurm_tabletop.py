def generate_slurm_files(constraint_loss_types, constraint_violation_weights, constraint_loss_schedulings,
                         max_sample_steps, normalize_xt_types):
    slurm_template = """#!/bin/bash
#SBATCH --job-name=diffusion
#SBATCH --output=/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/output/tabletop_v2/dr_v0.%A.%a.out
#SBATCH --error=/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/output/tabletop_v2/dr_v0.%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=25G
#SBATCH --gres=gpu:1
#SBATCH --time=59:59:59
#SBATCH --mail-type=all
#SBATCH --mail-user=anjianl@princeton.edu
#SBATCH --array=1

module purge
module load anaconda3/2021.11
conda activate pydylan

export WANDB_DIR="/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/wandb/tabletop_v2/{training_data_type}"
export WANDB_MODE=offline
python /home/al5844/desktop/project/denoising-diffusion-pytorch/run/train_classifier_free_cond_1d_improved_constrained_diffusion.py --constraint_loss_type {constraint_loss_type} --constraint_violation_weight {constraint_violation_weight} --constraint_loss_scheduling={constraint_loss_scheduling} --max_sample_step_with_constraint_loss={max_sample_step} --normalize_xt_type={normalize_xt_type} --training_random_seed 0 --wandb_project_name constrained_diffusion_tabletopv2 --task_type tabletop --training_data_num 237370 --result_folder /scratch/gpfs/al5844/project/denoising-diffusion-pytorch/results/tabletop_v2/results --data_path /scratch/gpfs/al5844/project/denoising-diffusion-pytorch/Data/tabletop_v2_obs_goal_time_control_num_237370.pkl --class_dim 14 --channel_num 1 --seq_length 161 --training_data_type {training_data_type}
"""
    num_files = len(constraint_loss_types)
    for i in range(num_files):
        weight_without_dot = ''.join(constraint_violation_weights[i].split('.'))
        training_data_type = f"tabletopv2_experiment_{constraint_loss_types[i]}_schedule_{constraint_loss_schedulings[i]}_normalize_{normalize_xt_types[i]}_max_sample_{max_sample_steps[i]}_weight_{weight_without_dot}_seed_0"

        file_content = slurm_template.format(
            constraint_loss_type=constraint_loss_types[i],
            constraint_violation_weight=constraint_violation_weights[i],
            constraint_loss_scheduling=constraint_loss_schedulings[i],
            max_sample_step=max_sample_steps[i],
            normalize_xt_type=normalize_xt_types[i],
            training_data_type=training_data_type
        )
        file_name = f"{training_data_type}.slurm"
        with open(file_name, 'w') as f:
            f.write(file_content)
        print(f"Created SLURM file: {file_name}")


# Example usage:
constraint_loss_types = ['vanilla', 'vanilla', 'predict_x0_violation', 'gt_std', 'gt_std', 'gt_std', 'gt_log_likelihood', 'gt_log_likelihood', 'gt_log_likelihood', 'predict_x0_violation']
constraint_violation_weights = ['0.1', '1', '0.1', '10', '10', '10', '10', '1', '1', '0.1']
constraint_loss_schedulings = ['sqrt_bar_alpha', 'NA', "NA", "one_over_t", "sqrt_bar_alpha", 'NA', "one_over_t", "sqrt_bar_alpha", 'NA', "sqrt_bar_alpha"]
max_sample_steps = ['500', '100', '100', '500', '500', '100', '500', '500', '100','500']
normalize_xt_types = ['direct_clip', 'direct_clip', 'direct_clip', 'direct_clip', 'direct_clip', 'direct_clip', 'direct_clip', 'direct_clip', 'direct_clip', 'direct_clip']


generate_slurm_files(constraint_loss_types, constraint_violation_weights, constraint_loss_schedulings, max_sample_steps,
                     normalize_xt_types)
