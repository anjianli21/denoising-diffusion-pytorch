import os
import re

def find_folders_with_timesteps_and_unet_dim(base_folder, target_timesteps):
    matching_folders = []

    for root, dirs, files in os.walk(base_folder):
        for dir_name in dirs:
            if dir_name.startswith("offline-run-"):
                log_file_path = os.path.join(root, dir_name, "logs", "debug.log")
                if os.path.exists(log_file_path):
                    with open(log_file_path, 'r') as log_file:
                        timesteps_match = False
                        unet_dim_match = False
                        for line in log_file:
                            timesteps = re.search(r"'timesteps': (\d+)", line)
                            unet_dim = re.search(r"'unet_dim': 64", line)
                            if timesteps and int(timesteps.group(1)) == target_timesteps:
                                timesteps_match = True
                            if unet_dim:
                                unet_dim_match = True
                            if timesteps_match and unet_dim_match:
                                matching_folders.append(dir_name)
                                break

    return matching_folders

def main():
    base_folder ="/scratch/gpfs/jg3607/Diffusion_model/indirect/sampling/wandb/"
    target_timesteps = int(input("Enter the timesteps value to search for: "))
    matching_folders = find_folders_with_timesteps_and_unet_dim(base_folder, target_timesteps)
    
    if matching_folders:
        print("Folders with timesteps = {} and unet_dim = 64: {}".format(target_timesteps, matching_folders))
    else:
        print("No folders found with timesteps = {} and unet_dim = 64".format(target_timesteps))

if __name__ == "__main__":
    main()