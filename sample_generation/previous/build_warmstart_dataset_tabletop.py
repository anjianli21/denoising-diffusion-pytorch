import os
import glob
import re
import time

from denoising_diffusion_pytorch.previous_method.classifier_free_guidance_cond_1d import Unet1D, GaussianDiffusion1D, Trainer1D

import numpy as np
import pickle
import torch


def main():

    TIME_MIN = 3.67867
    TIME_MAX = 6.0
    CONTROL_MIN = - 1.0005
    CONTROL_MAX = 1.0005
    OBS_POS_MIN = 1.0
    OBS_POS_MAX = 9.0
    OBS_RADIUS_MIN = 0.2
    OBS_RADIUS_MAX = 0.5
    GOAL_POS_MIN = 1.0
    GOAL_POS_MAX = 9.0

    # sample_type_list = ["conditional_sample"]
    sample_type = "full_sample"

    # diffusion_w_list = [10.0, 5.0]
    diffusion_w = 5.0

    # constraint_violation_weight_list = [0.01, 0.001]

    sample_num = 10
    condition_seed_num = 20

    condition_seed_list = [5000 + i for i in range(condition_seed_num)]

    data_type_list = [
        f"full_data_202k_constraint_weight_0.0001_condscale_1",
        f"full_data_202k_constraint_weight_0.0001_condscale_6",
        f"full_data_202k_constraint_weight_0.001_condscale_1",
        f"full_data_202k_constraint_weight_0.001_condscale_6",
        f"full_data_202k_constraint_weight_0.01_condscale_1",
        f"full_data_202k_constraint_weight_0.01_condscale_6",
    ]
    
    # Configure path
    parent_path = f"results/from_autodl/diffusion/tabletop/results"
    input_obs_goal_output_time_control_parent_path_list = [
        f"{parent_path}/full_data_202k_constraint_weight_0.0001_condscale_1",
        f"{parent_path}/full_data_202k_constraint_weight_0.0001_condscale_6",
        f"{parent_path}/full_data_202k_constraint_weight_0.001_condscale_1",
        f"{parent_path}/full_data_202k_constraint_weight_0.001_condscale_6",
        f"{parent_path}/full_data_202k_constraint_weight_0.01_condscale_1",
        f"{parent_path}/full_data_202k_constraint_weight_0.01_condscale_6",
    ]

    for i in range(len(data_type_list)):
        data_type = data_type_list[i]
        input_obs_goal_output_time_control_parent_path = input_obs_goal_output_time_control_parent_path_list[i]

        for j in range(len(condition_seed_list)):
            condition_seed = condition_seed_list[j]

            # Sample obs
            rng_condition = np.random.RandomState(seed=condition_seed)
            # obs sample
            is_condition_reasonable = False
            while not is_condition_reasonable:
                print("sample obs again")

                print("sample obs again")
                car_start_pos = np.array([[5.0, 5.0], [0.0, 0.0], [10.0, 10.0]])
                car_goal_pos = np.array(
                    [[[1.0, 1.0], [1.0, 9.0], [9.0, 1.0], [9.0, 9.0]][rng_condition.randint(low=0, high=4)]])
                obs_radius = rng_condition.rand(4) * (OBS_RADIUS_MAX - OBS_RADIUS_MIN) + OBS_RADIUS_MIN

                pos_x_min = np.minimum(car_start_pos[0][0], car_goal_pos[0][0])
                pos_y_min = np.minimum(car_start_pos[0][1], car_goal_pos[0][1])
                pos_x_max = np.maximum(car_start_pos[0][0], car_goal_pos[0][0])
                pos_y_max = np.maximum(car_start_pos[0][1], car_goal_pos[0][1])
                obs_pos_x = rng_condition.rand(4) * (pos_x_max - pos_x_min) + pos_x_min
                obs_pos_y = rng_condition.rand(4) * (pos_y_max - pos_y_min) + pos_y_min

                obs_pos = np.hstack((obs_pos_x.reshape(-1, 1), obs_pos_y.reshape(-1, 1)))

                parameters = {}
                parameters["obs_radius"] = obs_radius
                parameters["obs_pos"] = obs_pos
                parameters["car_goal_pos"] = car_goal_pos
                is_condition_reasonable = check_condition(parameters=parameters)

            obs_radius = obs_radius.reshape(1, 4)
            obs_radius = (obs_radius - OBS_RADIUS_MIN) / (OBS_RADIUS_MAX - OBS_RADIUS_MIN)

            obs_pos = obs_pos.reshape(1, 8)
            obs_pos = (obs_pos - OBS_POS_MIN) / (OBS_POS_MAX - OBS_POS_MIN)

            car_goal_pos = car_goal_pos.reshape(1, 2)
            car_goal_pos = (car_goal_pos - GOAL_POS_MIN) / (GOAL_POS_MAX - GOAL_POS_MIN)

            obs_goal_condition_input = np.hstack([obs_pos, obs_radius, car_goal_pos])

            # Repeat the same obs input as the sample num
            obs_goal_condition_input = np.tile(obs_goal_condition_input, (sample_num, 1))
            obs_goal_condition_input = torch.tensor(obs_goal_condition_input).float().cuda()

            if sample_type == "full_sample":
                t_final_control_samples = sample_diffusion(condition_input=obs_goal_condition_input,
                                                           input_output_type="input_obs_goal_output_t_control",
                                                           checkpoint_parent_path=input_obs_goal_output_time_control_parent_path,
                                                           sample_num=sample_num,
                                                           diffusion_w=diffusion_w)
                obs_goal_condition_input = obs_goal_condition_input.detach().cpu().numpy()
                t_final_control_samples = t_final_control_samples.detach().cpu().numpy()

                obs_goal_t_final_control_samples = np.hstack((obs_goal_condition_input, t_final_control_samples))


            # Data preparation #######################################################################################################
            # obs_pos, original range [2, 8]
            obs_goal_t_final_control_samples[:, :8] = obs_goal_t_final_control_samples[:, :8] * (OBS_POS_MAX - OBS_POS_MIN) + OBS_POS_MIN
            # obs_radius, original range [0.5, 1.5]
            obs_goal_t_final_control_samples[:, 8:12] = obs_goal_t_final_control_samples[:, 8:12] * (OBS_RADIUS_MAX - OBS_RADIUS_MIN) + OBS_RADIUS_MIN
            # car goal pos
            obs_goal_t_final_control_samples[:, 12:14] = obs_goal_t_final_control_samples[:, 12:14] * (GOAL_POS_MAX - GOAL_POS_MIN) + GOAL_POS_MIN
            # t_final, original range [TIME_MIN, TIME_MAX]
            obs_goal_t_final_control_samples[:, 14] = obs_goal_t_final_control_samples[:, 14] * (TIME_MAX - TIME_MIN) + TIME_MIN
            # Control, original range [CONTROL_MIN, CONTROL_MAX]
            obs_goal_t_final_control_samples[:, 15:] = obs_goal_t_final_control_samples[:, 15:] * (CONTROL_MAX - CONTROL_MIN) + CONTROL_MIN
            print("data normalization is done")

            for num in range(sample_num):
                warmstart_data_parent_path = f"/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/warmstart_data/tabletop/{data_type}"
                if not os.path.exists(warmstart_data_parent_path):
                    os.makedirs(warmstart_data_parent_path, exist_ok=True)
                warmstart_data_path = f"{warmstart_data_parent_path }/{data_type}_condition_seed_{condition_seed}_initial_guess_seed_{num}.pkl"
                with open(warmstart_data_path, 'wb') as f:
                    pickle.dump(obs_goal_t_final_control_samples[num, :], f)
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

    if input_output_type == "input_obs_goal_output_t_control":
        class_dim = 14
        channel = 1
        seq_length = 81
    else:
        print("wrong input output type")
        exit()

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

def check_condition(parameters, to_print=False):
    car_num = 1
    car_radius = 0.2
    obs_num = 4
    obs_radius = parameters["obs_radius"]
    obs_pos = parameters["obs_pos"]

    car_start_pos = np.array([[5.0, 5.0]])
    car_goal_pos = parameters["car_goal_pos"]

    # Check if car start and goal positions are far enough from each other
    for i in range(car_num):
        for j in range(car_num):
            if i != j:
                if np.linalg.norm(car_start_pos[i, :] - car_start_pos[j, :]) < 4 * car_radius:
                    if to_print:
                        print(f"car {i} and car {j} start pos is too close")
                    return False
                if np.linalg.norm(car_goal_pos[i, :] - car_goal_pos[j, :]) < 4 * car_radius:
                    if to_print:
                        print(f"car {i} and car {j} goal pos is too close")
                    return False

    # Check if car start and goal positions are far enough from obstacles
    for i in range(car_num):
        for j in range(obs_num):
            if np.linalg.norm(car_start_pos[i, :] - obs_pos[j, :]) < 4 * car_radius + obs_radius[j]:
                if to_print:
                    print(f"car {i} start pos and obs {j} pos is too close")
                return False
            if np.linalg.norm(car_goal_pos[i, :] - obs_pos[j, :]) < 4 * car_radius + obs_radius[j]:
                if to_print:
                    print(f"car {i} goal pos and obs {j} pos is too close")
                return False

    for i in range(obs_num):
        for j in range(obs_num):
            if i != j:
                if np.linalg.norm(obs_pos[i, :] - obs_pos[j, :]) < obs_radius[i] + obs_radius[j]:
                    if to_print:
                        print(f"obs {i} and obs {j} start pos is too close")
                    return False

    return True

if __name__ == "__main__":
    main()