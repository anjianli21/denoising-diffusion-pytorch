import pickle 
import numpy as np

file_path = "/home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/results/generated_initializations/indirect/unet_64_mults_4_4_8_embed_class_128_256_timesteps_1000_batch_size_1024_cond_drop_0.1_mask_val_0.0/cr3bp_thrust_0.95_diffusion_indirect_w_1.0_training_num_270000_num_1000000.pkl"
file_path_2 = "/home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/results/generated_initializations/indirect/unet_64_mults_4_4_8_embed_class_128_256_timesteps_1000_batch_size_1024_cond_drop_0.1_mask_val_0.0/cr3bp_thrust_0.95_diffusion_indirect_w_1.0_training_num_270000_num_1000000_2.pkl"

file = open(file_path, 'rb')
data = pickle.load(file)
file.close()

file_2 = open(file_path_2, 'rb')
data_2 = pickle.load(file_2)
file_2.close()

combined_data = np.vstack((data, data_2))
print(combined_data.shape)

new_file_path = "/home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/results/generated_initializations/indirect/unet_64_mults_4_4_8_embed_class_128_256_timesteps_1000_batch_size_1024_cond_drop_0.1_mask_val_0.0/cr3bp_thrust_0.95_diffusion_indirect_w_1.0_training_num_270000_num_2000000.pkl"
with open(new_file_path, 'wb') as f:
    pickle.dump(combined_data, f)
    print(f"{new_file_path} successfully saved!")