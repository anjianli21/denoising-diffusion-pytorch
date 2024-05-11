import os


def generate_bash_script(directory):
    # List all files in the directory that end with .slurm
    slurm_files = [f for f in os.listdir(directory) if f.endswith('.slurm')]
    slurm_files.sort()  # Sort files to maintain a predictable order

    # Create the Bash script content
    script_content = "#!/bin/bash\n\n"
    for file in slurm_files:
        script_content += f"sbatch {os.path.join(directory, file)}\n"

    # Write the Bash script to a file
    with open('submit_all_slurm_jobs_tabletop.sh', 'w') as script_file:
        script_file.write(script_content)

    print("Bash script generated: submit_all_slurm_jobs_tabletop.sh")


# Change the directory path as needed
directory_path = 'run/della/dddas/tabletop/0511/'
generate_bash_script(directory_path)
