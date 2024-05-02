def generate_slurm_files(constraint_loss_types, constraint_violation_weights, constraint_loss_schedulings,
                         max_sample_steps, normalize_xt_types):
    slurm_template = """#!/bin/bash
#SBATCH --job-name=diffusion
#SBATCH --output=/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/output/car/dr_v0.%A.%a.out
#SBATCH --error=/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/output/car/dr_v0.%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=25G
#SBATCH --gres=gpu:1
#SBATCH --time=19:59:59
#SBATCH --mail-type=all
#SBATCH --mail-user=anjianl@princeton.edu
#SBATCH --array=1

module purge
module load anaconda3/2021.11
conda activate pydylan

export WANDB_DIR="/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/wandb/car/{training_data_type}"
export WANDB_MODE=offline
python /home/al5844/desktop/project/denoising-diffusion-pytorch/run/train_classifier_free_cond_1d_improved_constrained_diffusion.py --constraint_loss_type {constraint_loss_type} --constraint_violation_weight {constraint_violation_weight} --constraint_loss_scheduling={constraint_loss_scheduling} --max_sample_step_with_constraint_loss={max_sample_step} --normalize_xt_type={normalize_xt_type} --training_random_seed 0 --wandb_project_name constrained_diffusion_car --task_type car --training_data_num 114570 --result_folder /scratch/gpfs/al5844/project/denoising-diffusion-pytorch/results/car/results --data_path /scratch/gpfs/al5844/project/denoising-diffusion-pytorch/Data/obstacle_time_control_data_obj_12_num_114570.pkl --class_dim 6 --channel_num 1 --seq_length 81 --training_data_type {training_data_type}
"""
    num_files = len(constraint_loss_types)
    for i in range(num_files):
        weight_without_dot = ''.join(constraint_violation_weights[i].split('.'))
        training_data_type = f"car_experiment_{constraint_loss_types[i]}_schedule_{constraint_loss_schedulings[i]}_normalize_{normalize_xt_types[i]}_max_sample_{max_sample_steps[i]}_weight_{weight_without_dot}_seed_0"

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
# constraint_loss_types = ['vanilla', 'vanilla', 'gt_scaled', 'gt_scaled', 'gt_scaled', 'predict_x0_violation', 'vanilla', 'vanilla', 'vanilla', 'gt_scaled', 'gt_scaled', 'gt_scaled', 'gt_scaled']
# constraint_violation_weights = ['0.001', '0.01', "10", "1", "1", "0.01", '0.1', '0.001', '0.01', '0.1', '10', '1', '1']
# constraint_loss_schedulings = ['sqrt_bar_alpha', 'NA', "one_over_t", "sqrt_bar_alpha", "NA", "NA", "one_over_t", "sqrt_bar_alpha", "NA", "NA", "one_over_t", "sqrt_bar_alpha", "NA"]
# max_sample_steps = ['500', '100', '500', '500', '100', '100', '500', '500', '100', '500', '500', '500', '100']
# normalize_xt_types = ['direct_clip', 'direct_clip', 'direct_clip', 'direct_clip', 'direct_clip', 'direct_clip', "-1_-1_var", "-1_-1_var", "-1_-1_var", "-1_-1_var", "-1_-1_var", "-1_-1_var", "-1_-1_var"]

constraint_loss_types = ['gt_std', 'gt_std', 'gt_std', 'gt_log_likelihood', 'gt_log_likelihood', 'gt_log_likelihood', 'predict_x0_violation']
constraint_violation_weights = ['10', '1', '10', '10', '1', '1', '0.001']
constraint_loss_schedulings = ["one_over_t", "sqrt_bar_alpha", 'NA', "one_over_t", "sqrt_bar_alpha", 'NA', "sqrt_bar_alpha"]
max_sample_steps = ['500', '500', '100', '500', '500', '100','500']
normalize_xt_types = ['direct_clip', 'direct_clip', 'direct_clip', 'direct_clip', 'direct_clip', 'direct_clip', 'direct_clip']


generate_slurm_files(constraint_loss_types, constraint_violation_weights, constraint_loss_schedulings, max_sample_steps,
                     normalize_xt_types)
