import os
import glob
import re
import time

from denoising_diffusion_pytorch.previous_method.classifier_free_guidance_cond_1d import Unet1D, GaussianDiffusion1D, Trainer1D

import copy
import numpy as np
import pickle

MAX_MASS = 438.0  #
MINIMUM_SHOOTING_TIME = 0.0
MAXIMUM_SHOOTING_TIME = 40.0  # 50.0
CONTROL_SEGMENT = int(20)  # 60


def main():

    # sample_type_list = ["full_sample"]
    # data_type = f"diffusion_full_sample_300k_thrust_0.15_seed_2"
    # input_alpha_output_time_mass_control_parent_path = f"results/from_autodl/diffusion/cr3bp/results/cond_alpha_data_time_mass_control_300k_seed_2"

    # uniform data
    sample_type_list = ["uniform"]
    data_type = f"uniform_thrust_0.15_seed_0"
    seed = 0

    training_data_num_list = ["300k"]
    diffusion_w = 5.0
    alpha_list = [0.15]
    sample_num = 2000

    for i in range(len(training_data_num_list)):
        training_data_num = training_data_num_list[i]

        for j in range(len(sample_type_list)):
            sample_type = sample_type_list[j]

            for k in range(len(alpha_list)):
                thrust = alpha_list[k]

                if sample_type == "uniform":
                    data_time_mass_normalized = get_sample_from_uniform_time_mass(sample_num=sample_num, seed=seed)
                    normalized_alpha = (thrust - 0.1) / (1.0 - 0.1)
                    normalized_alpha = normalized_alpha * np.ones((data_time_mass_normalized.shape[0], 1))
                    data_time_mass_alpha_normalized = np.hstack((data_time_mass_normalized, normalized_alpha))
                    print(f"thrust = {thrust}, time and mass generated from uniform samples")

                    number_of_segments = 20
                    random_state = np.random.RandomState(seed=seed)
                    theta = random_state.uniform(0, 2 * np.pi, number_of_segments * sample_num)
                    psi = random_state.uniform(0, 2 * np.pi, number_of_segments * sample_num)
                    r = random_state.uniform(0, 1, number_of_segments * sample_num)

                    data_control_radius = []
                    for i in range(sample_num):
                        data_control_radius_tmp = []
                        for j in range(number_of_segments):
                            data_control_radius_tmp.append(theta[i * number_of_segments + j])
                            data_control_radius_tmp.append(psi[i * number_of_segments + j])
                            data_control_radius_tmp.append(r[i * number_of_segments + j])

                        data_control_radius.append(data_control_radius_tmp)
                    data_control_radius = np.asarray(data_control_radius)
                    print(f"thrust = {thrust}, control generated from uniform sample")

                    # Data normalization
                    full_solution = np.hstack((np.hstack((data_time_mass_normalized[:, :3], data_control_radius)),
                                               data_time_mass_normalized[:, -1].reshape(-1, 1)))
                    full_solution[:, 0] = full_solution[:, 0] * (
                            MAXIMUM_SHOOTING_TIME - MINIMUM_SHOOTING_TIME) + MINIMUM_SHOOTING_TIME
                    full_solution[:, 1] = full_solution[:, 1] * 15.0
                    full_solution[:, 2] = full_solution[:, 2] * 15.0
                    full_solution[:, -1] = full_solution[:, -1] * (MAX_MASS - 415.0) + 415.0
                    data_in_spherical = copy.copy(full_solution)

                for num in range(sample_num):
                    warmstart_data_parent_path = f"/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/icml_data/warmstart_data/{data_type}"
                    if not os.path.exists(warmstart_data_parent_path):
                        os.makedirs(warmstart_data_parent_path, exist_ok=True)
                    warmstart_data_path = f"{warmstart_data_parent_path}/{data_type}_seed_{num}.pkl"
                    with open(warmstart_data_path, 'wb') as f:
                        pickle.dump(data_in_spherical[num, :], f)
                    print(f"{warmstart_data_path} is saved")

def sample_diffusion(condition_input, input_output_type, checkpoint_parent_path, sample_num, diffusion_w):

    # Initialize model ############################################################################
    unet_dim = 128
    unet_dim_mults = "4,4,8"
    unet_dim_mults = tuple(map(int, unet_dim_mults.split(',')))
    embed_class_layers_dims = "256,512"
    embed_class_layers_dims = tuple(map(int, embed_class_layers_dims.split(',')))
    timesteps = 500
    objective = "pred_noise"
    batch_size = 512
    cond_drop_prob = 0.1

    if input_output_type == "input_alpha_output_time_mass":
        class_dim = 1
        channel = 1
        seq_length = 4
    elif input_output_type == "input_alpha_time_mass_output_control":
        class_dim = 5
        channel = 3
        seq_length = 20
    elif input_output_type == "input_alpha_output_time_mass_control":
        class_dim = 1
        channel = 1
        seq_length = 64

    model = Unet1D(
        dim=unet_dim,
        channels=channel,
        dim_mults=unet_dim_mults,
        embed_class_layers_dims=embed_class_layers_dims,
        class_dim=class_dim,
        cond_drop_prob=cond_drop_prob,
        seq_length=seq_length
    )

    diffusion = GaussianDiffusion1D(
        model=model,
        seq_length=seq_length,
        timesteps=timesteps,
        objective=objective
    ).cuda()

    # read checkpoints ############################################################################
    child_directories = glob.glob(os.path.join(checkpoint_parent_path, "*/*/"))
    if child_directories:
        final_result_folder = child_directories[0]

    trainer = Trainer1D(
        diffusion_model=diffusion,
        dataset=[0, 0, 0],
        results_folder=final_result_folder,
    )
    files = os.listdir(final_result_folder)
    regex = r"model-epoch-(\d+).pt"  # Modified regex to match 1 or more digits
    max_epoch = -1
    milestone = ""
    for file in files:
        if file.endswith(".pt"):
            match = re.search(regex, file)
            if match:
                epoch_num = int(match.group(1))
                if epoch_num > max_epoch:
                    max_epoch = epoch_num
                    milestone = f"epoch-{epoch_num}"  # Format with leading zeros
    # milestone = "epoch-102"
    trainer.load(milestone)

    # Sample results ################################################################################
    # 3. Use the loaded model for sampling
    start_time = time.time()
    sample_results = diffusion.sample(
        classes=condition_input.cuda(),
        cond_scale=diffusion_w,
    )
    end_time = time.time()
    print(f"{input_output_type}, {sample_num} data, takes {end_time - start_time} seconds")

    sample_results = sample_results.reshape(sample_num, -1)

    return sample_results


def convert_to_spherical(ux, uy, uz):
    u = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
    theta = np.zeros_like(u)
    mask_non_zero = u != 0
    theta[mask_non_zero] = np.arcsin(uz[mask_non_zero] / u[mask_non_zero])
    alpha = np.arctan2(uy, ux)
    alpha = np.where(alpha >= 0, alpha, 2 * np.pi + alpha)

    # Make sure theta is in [0, 2*pi]
    theta = np.where(theta >= 0, theta, 2 * np.pi + theta)

    return alpha, theta, u


def revert_converted_u_data(converted_u_data):
    # TODO: this function does 2 things:
    #  1) Convert ux uy uz from [-1, 1] to theta, psi,
    #  2) convert the second half order from correct order to reverse

    n = converted_u_data.shape[0]  # number of data points
    time_data = converted_u_data[:, :3]
    mass_data = converted_u_data[:, -1].reshape(n, 1)
    control_data = converted_u_data[:, 3:-1].reshape(n, CONTROL_SEGMENT, 3)

    # Convert Cartesian coordinates to spherical coordinates
    for i in range(control_data.shape[1]):
        alpha, theta, u = convert_to_spherical(control_data[:, i, 0], control_data[:, i, 1], control_data[:, i, 2])
        control_data[:, i, 0] = alpha
        control_data[:, i, 1] = theta
        control_data[:, i, 2] = u

    # Revert the order of the second half of the segments
    control_data[:, int(CONTROL_SEGMENT / 2):] = np.flip(control_data[:, int(CONTROL_SEGMENT / 2):], axis=1)

    # Flatten the control data and stack with time_data and mass_data
    control_data = control_data.reshape(n, -1)
    data_total = np.hstack((time_data, control_data, mass_data))

    return data_total


def get_sample_from_uniform_time_mass(sample_num, seed):
    random_state = np.random.RandomState(seed=seed)
    t_shooting = random_state.uniform(MINIMUM_SHOOTING_TIME, MAXIMUM_SHOOTING_TIME, sample_num)
    t_init = random_state.uniform(0, 15.0, sample_num)
    t_final = random_state.uniform(0, 15.0, sample_num)
    mass = random_state.uniform(415.0, MAX_MASS, sample_num)
    sample_data = []
    for i in range(sample_num):
        data = []
        data.append(t_shooting[i])
        data.append(t_init[i])
        data.append(t_final[i])
        data.append(mass[i])
        data = np.asarray(data)

        sample_data.append(data)

    sample_data = np.asarray(sample_data)

    normalized_sample_data = np.zeros_like(sample_data)
    normalized_sample_data[:, 0] = (sample_data[:, 0] - MINIMUM_SHOOTING_TIME) / (MAXIMUM_SHOOTING_TIME - MINIMUM_SHOOTING_TIME)
    normalized_sample_data[:, 1] = sample_data[:, 1] / 15.0
    normalized_sample_data[:, 2] = sample_data[:, 2] / 15.0
    normalized_sample_data[:, -1] = (sample_data[:, -1] - 415.0) / (
                MAX_MASS - 415.0)

    normalized_sample_data = normalized_sample_data.astype(np.float32)

    return normalized_sample_data

if __name__ == "__main__":
    main()
