#!/bin/bash

# Loop from 1 to 90 to submit each job
for i in {1..126}
do
   sbatch "/home/al5844/desktop/project/denoising-diffusion-pytorch/run/bash/della_slurm/slurm_job_$i.slurm"
done
