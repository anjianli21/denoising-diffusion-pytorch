#!/bin/bash

# Loop to submit job_i.slurm for i in range(0, 14)
sbatch run/bash/fixed_car_vary_obs/input_obs_output_time_control_obj_12.slurm
sbatch run/bash/fixed_car_vary_obs/input_obs_output_time_obj_12.slurm
sbatch run/bash/fixed_car_vary_obs/input_obs_time_output_control_obj_12.slurm

sbatch run/bash/fixed_car_vary_obs/input_obs_output_time_control_obj_14.slurm
sbatch run/bash/fixed_car_vary_obs/input_obs_output_time_obj_14.slurm
sbatch run/bash/fixed_car_vary_obs/input_obs_time_output_control_obj_14.slurm

