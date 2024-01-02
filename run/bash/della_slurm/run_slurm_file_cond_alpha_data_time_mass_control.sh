#!/bin/bash

for i in {1..12}
do
   sbatch "/home/al5844/desktop/project/denoising-diffusion-pytorch/run/bash/della_slurm/cond_alpha_data_time_mass_control_slurm/slurm_job_$i.slurm"
done
