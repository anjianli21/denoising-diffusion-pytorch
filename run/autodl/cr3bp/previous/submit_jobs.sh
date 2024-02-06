#!/bin/bash

# Loop to submit job_i.slurm for i in range(0, 14)
sbatch run/bash/cr3bp/input_alpha_output_time_mass_control_num_300k.slurm
sbatch run/bash/cr3bp/input_alpha_output_time_mass_num_300k.slurm
sbatch run/bash/cr3bp/input_alpha_time_mass_output_control_num_300k.slurm

sbatch run/bash/cr3bp/input_alpha_output_time_mass_control_num_30k.slurm
sbatch run/bash/cr3bp/input_alpha_output_time_mass_num_30k.slurm
sbatch run/bash/cr3bp/input_alpha_time_mass_output_control_num_30k.slurm


