#!/bin/bash

# Loop from 1 to 90 to submit each job
for i in {1..90}
do
   sbatch "slurm_job_$i.sh"
done
