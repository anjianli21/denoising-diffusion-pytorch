import os
import glob
import re
import time

# TODO: select the model script based on the type of constraints,
#  here is previous 1/t constraint
# from denoising_diffusion_pytorch.classifier_free_guidance_cond_1d_constraint_car import Unet1D, GaussianDiffusion1D, Trainer1D, \
#     Dataset1D
# from denoising_diffusion_pytorch.constraint_violation_function_car import get_constraint_violation_car

# TODO: here is improved constraint, based on sampled mean
from denoising_diffusion_pytorch.classifier_free_guidance_cond_1d_improved_constrained_diffusion import Unet1D, GaussianDiffusion1D, Trainer1D
from denoising_diffusion_pytorch.constraint_violation_function_improved_car import get_constraint_violation_car

import copy
import numpy as np
import pickle
import torch


def main():

    TIME_MIN = 7.81728
    TIME_MAX = 12.0
    CONTROL_MIN = - 1.0005
    CONTROL_MAX = 1.0005

    # sample_type_list = ["full_sample", "conditional_sample"]
    # sample_type_list = ["conditional_sample"]
    # sample_type_list = ["full_sample"]
    sample_type = "full_sample"

    # diffusion_w_list = [10.0, 5.0]
    # diffusion_w_list = [5.0]

    diffusion_w = 5.0

    sample_num = 10
    condition_seed_num = 20
    #
    # sample_num = 50
    # condition_seed_num = 1


    condition_seed_list = [5000 + i for i in range(condition_seed_num)]

    # data_type_list = [
    #     # f"full_data_114k_constraint_weight_0.0001_condscale_1",
    #     # f"full_data_114k_constraint_weight_0.0001_condscale_6",
    #     # f"full_data_114k_constraint_weight_0.001_condscale_1",
    #     # f"full_data_114k_constraint_weight_0.001_condscale_6",
    #     # f"full_data_114k_constraint_weight_0.01_condscale_1",
    #     f"full_data_114k_constraint_weight_0.01_condscale_6",
    #     # f"input_obs_output_time_control_obj_12_data_114k"
    #                   ]

    # data_type_list = [
    #     # f"full_data_114k_constraint_weight_0.01_condscale_6_seed_0",
    #     # f"full_data_114k_constraint_weight_0.01_condscale_6_seed_1",
    #     # f"full_data_114k_constraint_weight_0.01_condscale_6_seed_2",
    #     # f"input_obs_output_time_control_obj_12_data_114k_seed_0",
    #     # f"input_obs_output_time_control_obj_12_data_114k_seed_1",
    #     # f"input_obs_output_time_control_obj_12_data_114k_seed_2",
    # ]

    # data_type_list = [
    #     f"car_constrained_improved_seed_0",
    #     f"car_constrained_improved_seed_1",
    #     f"car_constrained_improved_seed_2",
    # ]

    # data_type_list = [
    #     "car_constrained_step_10_type_scaled_weight_10_seed_0",
    #     "car_constrained_step_10_type_threshold_weight_10_seed_0",
    #     "car_constrained_step_500_type_scaled_weight_1_seed_0",
    # ]

    # TODO: DDDAS
    # data_type_list = [
        # "car_constraint_gt_scaled_weight_01_seed_0",
        # "car_constraint_gt_std_absolute_weight_1_seed_0",
        # "car_constraint_gt_std_threshold_weight_1_seed_0",
        # "car_constraint_gt_std_weight_1_seed_0",
        # "car_constraint_one_over_t_weight_01_seed_0"
        # "car_constraint_gt_std_absolute_weight_01_seed_0",
        # "car_constraint_gt_std_threshold_weight_01_seed_0",
        # "car_constraint_gt_std_weight_01_seed_0",
        # "car_constraint_gt_log_likelihood_weight_01_seed_0"
    # ]
    # data_type_list = [
    #     "car_constraint_pred_x0_one_over_t_weight_01_seed_0",
    #     "car_constraint_pred_x0_weight_0001_seed_0"
    # ]
    # data_type_list = [
    #     "car_experiment_gt_scaled_schedule_NA_normalize_direct_clip_max_sample_100_weight_1_seed_0",
    #     "car_experiment_gt_scaled_schedule_one_over_t_normalize_direct_clip_max_sample_500_weight_10_seed_0",
    #     "car_experiment_gt_scaled_schedule_sqrt_bar_alpha_normalize_direct_clip_max_sample_500_weight_1_seed_0",
    #     "car_experiment_predict_x0_violation_schedule_NA_normalize_direct_clip_max_sample_100_weight_001_seed_0",
    #     "car_experiment_vanilla_schedule_NA_normalize_direct_clip_max_sample_100_weight_001_seed_0",
    #     "car_experiment_vanilla_schedule_sqrt_bar_alpha_normalize_direct_clip_max_sample_500_weight_0001_seed_0"
    # ]
    data_type_list = [
        "car_experiment_gt_log_likelihood_schedule_NA_normalize_direct_clip_max_sample_100_weight_1_seed_0",
        "car_experiment_gt_log_likelihood_schedule_one_over_t_normalize_direct_clip_max_sample_500_weight_10_seed_0",
        "car_experiment_gt_log_likelihood_schedule_sqrt_bar_alpha_normalize_direct_clip_max_sample_500_weight_1_seed_0",
        "car_experiment_gt_std_schedule_NA_normalize_direct_clip_max_sample_100_weight_10_seed_0",
        "car_experiment_gt_std_schedule_one_over_t_normalize_direct_clip_max_sample_500_weight_10_seed_0",
        "car_experiment_gt_std_schedule_sqrt_bar_alpha_normalize_direct_clip_max_sample_500_weight_1_seed_0",
        "car_experiment_predict_x0_violation_schedule_sqrt_bar_alpha_normalize_direct_clip_max_sample_500_weight_0001_seed_0"
    ]

    # Configure path
    # parent_path = f"results/from_autodl/diffusion/fixed_car_vary_obs/results"
    # parent_path = f"results/dddas/car"
    # parent_path = f"results/dddas/car/0429"
    parent_path = f"results/dddas/car/0503"


    # input_obs_output_time_control_parent_path_list = [
    #     # f"{parent_path}/full_data_114k_constraint_weight_0.0001_condscale_1",
    #     # f"{parent_path}/full_data_114k_constraint_weight_0.0001_condscale_6",
    #     # f"{parent_path}/full_data_114k_constraint_weight_0.001_condscale_1",
    #     # f"{parent_path}/full_data_114k_constraint_weight_0.001_condscale_6",
    #     # f"{parent_path}/full_data_114k_constraint_weight_0.01_condscale_1",
    #     f"{parent_path}/full_data_114k_constraint_weight_0.01_condscale_6",
    #     # f"{parent_path}/input_obs_output_time_control_obj_12_data_114k"
    # ]

    # input_obs_output_time_control_parent_path_list = [
    #     f"{parent_path}/full_data_114k_constraint_weight_0.01_condscale_6_seed_0",
    #     # f"{parent_path}/full_data_114k_constraint_weight_0.01_condscale_6_seed_1",
    #     # f"{parent_path}/full_data_114k_constraint_weight_0.01_condscale_6_seed_2",
    #     # f"{parent_path}/input_obs_output_time_control_obj_12_data_114k_seed_0",
    #     # f"{parent_path}/input_obs_output_time_control_obj_12_data_114k_seed_1",
    #     # f"{parent_path}/input_obs_output_time_control_obj_12_data_114k_seed_2",
    # ]

    # input_obs_output_time_control_parent_path_list = [
    #     f"{parent_path}/car_constrained_improved_seed_0",
    #     f"{parent_path}/car_constrained_improved_seed_1",
    #     f"{parent_path}/car_constrained_improved_seed_2",
    # ]
    #
    # input_obs_output_time_control_parent_path_list = [
    #     f"{parent_path}/car_constrained_step_10_type_scaled_weight_10_seed_0",
    #     f"{parent_path}/car_constrained_step_10_type_threshold_weight_10_seed_0",
    #     f"{parent_path}/car_constrained_step_500_type_scaled_weight_1_seed_0",
    # ]

    # DDDAS
    input_obs_output_time_control_parent_path_list = [f"{parent_path}/{i}" for i in data_type_list]

    constraint_violation_list = []
    for i in range(len(data_type_list)):
        data_type = data_type_list[i]
        input_obs_output_time_control_parent_path = input_obs_output_time_control_parent_path_list[i]

        current_prediction_data_list = []

        obs_condition_input_list = []
        for j in range(len(condition_seed_list)):
            condition_seed = condition_seed_list[j]

            # Sample obs
            rng_condition = np.random.RandomState(seed=condition_seed)
            # obs sample
            is_condition_reasonable = False
            while not is_condition_reasonable:
                # print("sample obs again")
                obs_radius = rng_condition.rand(2)
                obs_pos = rng_condition.rand(2, 2)

                parameters = {}
                parameters["obs_radius"] = obs_radius + 0.5
                parameters["obs_pos"] = obs_pos * 6.0 + 2.0
                is_condition_reasonable = check_condition(parameters=parameters)

            obs_radius = obs_radius.reshape(1, 2)
            obs_pos = obs_pos.reshape(1, 4)

            obs_condition_input = np.hstack((obs_pos, obs_radius))

            # Repeat the same obs input as the sample num
            obs_condition_input = np.tile(obs_condition_input, (sample_num, 1))
            obs_condition_input = torch.tensor(obs_condition_input).float().cuda()
            obs_condition_input_list.append(obs_condition_input)

        obs_condition_input_list = torch.vstack(obs_condition_input_list)

        if sample_type == "full_sample":
            t_final_control_samples = sample_diffusion(condition_input=obs_condition_input_list,
                                                       input_output_type="input_obs_output_t_control",
                                                       checkpoint_parent_path=input_obs_output_time_control_parent_path,
                                                       sample_num=sample_num * condition_seed_num,
                                                       diffusion_w=diffusion_w)
            obs_condition_input_list = obs_condition_input_list.detach().cpu().numpy()
            t_final_control_samples = t_final_control_samples.detach().cpu().numpy()

            obs_t_final_control_samples = np.hstack((obs_condition_input_list, t_final_control_samples))


        current_prediction_data_list.append(copy.copy(obs_t_final_control_samples))

        #############################################################################
        # check constraint violation
        current_prediction_data_list = np.vstack(current_prediction_data_list)
        current_prediction_data_tensor = torch.tensor(current_prediction_data_list)
        current_violation = get_constraint_violation_car(x = current_prediction_data_tensor[:, 6:],
                                                         c = current_prediction_data_tensor[:, :6],
                                                         scale=torch.tensor(1.0),
                                                         device=current_prediction_data_tensor.device)
        print(f"data type is {data_type}, violation is {current_violation}")

        constraint_violation_list.append(current_violation)

        # Data normalization ##########################################################
        # obs_pos, original range [2, 8]
        obs_t_final_control_samples[:, :4] = obs_t_final_control_samples[:, :4] * 6.0 + 2.0
        # obs_radius, original range [0.5, 1.5]
        obs_t_final_control_samples[:, 4:6] = obs_t_final_control_samples[:, 4:6] + 0.5
        # t_final, original range [TIME_MIN, TIME_MAX]
        obs_t_final_control_samples[:, 6] = obs_t_final_control_samples[:, 6] * (TIME_MAX - TIME_MIN) + TIME_MIN
        # Control, original range [CONTROL_MIN, CONTROL_MAX]
        obs_t_final_control_samples[:, 7:] = obs_t_final_control_samples[:, 7:] * (
                    CONTROL_MAX - CONTROL_MIN) + CONTROL_MIN
        print("data normalization is done")

        # Save ##########################################################################################################
        total_num = sample_num * condition_seed_num
        for num in range(total_num):
            curr_conditional_seed = 5000 + num // sample_num
            curr_initial_guess_seed = num % sample_num

            warmstart_data_parent_path = f"/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/warmstart_data/car/{data_type}"
            if not os.path.exists(warmstart_data_parent_path):
                os.makedirs(warmstart_data_parent_path, exist_ok=True)
            warmstart_data_path = f"{warmstart_data_parent_path}/{data_type}_condition_seed_{curr_conditional_seed}_initial_guess_seed_{curr_initial_guess_seed}.pkl"
            with open(warmstart_data_path, 'wb') as f:
                pickle.dump(obs_t_final_control_samples[num, :], f)
            print(f"{warmstart_data_path} is saved")


    for i in range(len(constraint_violation_list)):
        print(f"{data_type_list[i]}, constraint violation {constraint_violation_list[i]}")

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

    if input_output_type == "input_obs_output_t":
        class_dim = 6
        channel = 1
        seq_length = 1
    elif input_output_type == "input_obs_t_output_control":
        class_dim = 7
        channel = 4
        seq_length = 20
    elif input_output_type == "input_obs_output_t_control":
        class_dim = 6
        channel = 1
        seq_length = 81

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
    car_num = 2
    car_radius = 0.2
    obs_num = 2
    obs_radius = parameters["obs_radius"]
    obs_pos = parameters["obs_pos"]

    car_start_pos = np.array([[0.0, 10.0], [10.0, 10.0], [5.0, 0.0]])
    car_goal_pos = np.array([[10.0, 0.0], [0.0, 0.0], [5.0, 10.0]])

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
