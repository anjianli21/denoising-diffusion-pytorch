import os
import glob
import re
import time

# TODO： Acoording to the cost function, choose model script,
#  here is constraint 1/t
# from denoising_diffusion_pytorch.classifier_free_guidance_cond_1d import Unet1D, GaussianDiffusion1D, Trainer1D, \
#     Dataset1D
# from denoising_diffusion_pytorch.constraint_violation_function_tabletop_setupv2 import get_constraint_violation_tabletop

# TODO: here is improved constraint, sampled average violation
from denoising_diffusion_pytorch.previous_method.classifier_free_guidance_cond_1d_constraint_improved_tabletop import Unet1D, GaussianDiffusion1D, Trainer1D
from denoising_diffusion_pytorch.constraint_violation_function_improved_tabletop_setupv2 import get_constraint_violation_tabletop

import copy
import numpy as np
import pickle
import torch


def main():

    # TIME_MIN = 3.67867
    # TIME_MAX = 6.0
    # CONTROL_MIN = - 1.0005
    # CONTROL_MAX = 1.0005
    # OBS_POS_MIN = 1.0
    # OBS_POS_MAX = 9.0
    # OBS_RADIUS_MIN = 0.2
    # OBS_RADIUS_MAX = 0.5
    # GOAL_POS_MIN = 1.0
    # GOAL_POS_MAX = 9.0

    # Setup v2
    TIME_MIN = 4.64922
    TIME_MAX = 5.4
    CONTROL_MIN = - 1.0005
    CONTROL_MAX = 1.0005
    OBS_POS_MIN = 1.0
    OBS_POS_MAX = 9.0
    OBS_RADIUS_MIN = 0.5
    OBS_RADIUS_MAX = 1.0
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

    # data_type_list = [
    #     "full_data_202k_constraint_weight_0.01_condscale_6_seed_0",
    #     "full_data_202k_constraint_weight_0.01_condscale_6_seed_1",
    #     "full_data_202k_constraint_weight_0.01_condscale_6_seed_2",
    #     "input_obs_goal_output_time_control_obj_6_seed_0",
    #     "input_obs_goal_output_time_control_obj_6_seed_1",
    #     "input_obs_goal_output_time_control_obj_6_seed_2",
    # ]

    # data_type_list = [
    #     # "tabletop_v2_diffusion_seed_0",
    #     # "tabletop_v2_diffusion_seed_1",
    #     # "tabletop_v2_diffusion_seed_2",
    #     # "tabletop_v2_constrained_diffusion_seed_0",
    #     # "tabletop_v2_constrained_diffusion_seed_1",
    #     # "tabletop_v2_constrained_diffusion_seed_2",
    #     "tabletop_v2_constrained_diffusion_weight_01_seed_0",
    #     "tabletop_v2_constrained_diffusion_weight_01_seed_1",
    #     "tabletop_v2_constrained_diffusion_weight_01_seed_2",
    # ]

    data_type_list = [
        "tabletop_v2_constrained_improved_weight_01_diffusion_seed_0",
        "tabletop_v2_constrained_improved_weight_01_diffusion_seed_1",
        "tabletop_v2_constrained_improved_weight_01_diffusion_seed_2",
        "tabletop_v2_constrained_improved_weight_10_diffusion_seed_0",
        "tabletop_v2_constrained_improved_weight_10_diffusion_seed_1",
        "tabletop_v2_constrained_improved_weight_10_diffusion_seed_2",
    ]
    
    # Configure path ##############################################################################################
    parent_path = f"results/from_autodl/diffusion/tabletop_v2/results"

    # input_obs_goal_output_time_control_parent_path_list = [
    #     f"{parent_path}/full_data_202k_constraint_weight_0.01_condscale_6_seed_0",
    #     f"{parent_path}/full_data_202k_constraint_weight_0.01_condscale_6_seed_1",
    #     f"{parent_path}/full_data_202k_constraint_weight_0.01_condscale_6_seed_2",
    #     f"{parent_path}/input_obs_goal_output_time_control_obj_6_seed_0",
    #     f"{parent_path}/input_obs_goal_output_time_control_obj_6_seed_1",
    #     f"{parent_path}/input_obs_goal_output_time_control_obj_6_seed_2",
    # ]

    # input_obs_goal_output_time_control_parent_path_list = [
    #     # f"{parent_path}/tabletop_v2_diffusion_seed_0",
    #     # f"{parent_path}/tabletop_v2_diffusion_seed_1",
    #     # f"{parent_path}/tabletop_v2_diffusion_seed_2",
    #     # f"{parent_path}/tabletop_v2_constrained_diffusion_seed_0",
    #     # f"{parent_path}/tabletop_v2_constrained_diffusion_seed_1",
    #     # f"{parent_path}/tabletop_v2_constrained_diffusion_seed_2",
    #     f"{parent_path}/tabletop_v2_constrained_diffusion_weight_01_seed_0",
    #     f"{parent_path}/tabletop_v2_constrained_diffusion_weight_01_seed_1",
    #     f"{parent_path}/tabletop_v2_constrained_diffusion_weight_01_seed_2",
    # ]

    input_obs_goal_output_time_control_parent_path_list = [
        f"{parent_path}/tabletop_v2_constrained_improved_weight_01_diffusion_seed_0",
        f"{parent_path}/tabletop_v2_constrained_improved_weight_01_diffusion_seed_1",
        f"{parent_path}/tabletop_v2_constrained_improved_weight_01_diffusion_seed_2",
        f"{parent_path}/tabletop_v2_constrained_improved_weight_10_diffusion_seed_0",
        f"{parent_path}/tabletop_v2_constrained_improved_weight_10_diffusion_seed_1",
        f"{parent_path}/tabletop_v2_constrained_improved_weight_10_diffusion_seed_2",
    ]

    constraint_violation_list = []
    for i in range(len(data_type_list)):
        data_type = data_type_list[i]
        input_obs_goal_output_time_control_parent_path = input_obs_goal_output_time_control_parent_path_list[i]

        current_prediction_data_list = []

        obs_goal_condition_input_list = []
        for j in range(len(condition_seed_list)):
            condition_seed = condition_seed_list[j]

            # Sample obs
            rng_condition = np.random.RandomState(seed=condition_seed)
            # obs sample
            is_condition_reasonable = False
            while not is_condition_reasonable:
                print("sample obs again")
                car_start_pos = np.array([[5.0, 5.0], [0.0, 0.0], [10.0, 10.0]])
                car_goal_pos = np.array(
                    [[[1.0, 1.0], [1.0, 9.0], [9.0, 1.0], [9.0, 9.0]][rng_condition.randint(low=0, high=4)]])
                obs_radius = rng_condition.rand(4) * (OBS_RADIUS_MAX - OBS_RADIUS_MIN) + OBS_RADIUS_MIN

                pos_x_min = np.minimum(car_start_pos[0][0], car_goal_pos[0][0])
                pos_y_min = np.minimum(car_start_pos[0][1], car_goal_pos[0][1])
                pos_x_max = np.maximum(car_start_pos[0][0], car_goal_pos[0][0])
                pos_y_max = np.maximum(car_start_pos[0][1], car_goal_pos[0][1])
                obs_pos_x = rng_condition.rand(3) * (pos_x_max - pos_x_min) + pos_x_min
                obs_pos_y = rng_condition.rand(3) * (pos_y_max - pos_y_min) + pos_y_min

                # random sample from 0.3 to 1.0
                center_obs_pos_x = (rng_condition.rand(1) * 0.7 + 0.3) * (pos_x_max - pos_x_min) + pos_x_min
                gradient = (car_goal_pos[0][1] - car_start_pos[0][1]) / (
                            car_goal_pos[0][0] - car_start_pos[0][0])
                center_obs_pos_y = car_start_pos[0][1] + (
                            center_obs_pos_x - car_start_pos[0][0]) * gradient

                all_obs_pos_x = np.hstack([obs_pos_x, center_obs_pos_x])
                all_obs_pos_y = np.hstack([obs_pos_y, center_obs_pos_y])

                obs_pos = np.hstack((all_obs_pos_x.reshape(-1, 1), all_obs_pos_y.reshape(-1, 1)))
                # obs_pos = np.hstack((obs_pos_x.reshape(-1, 1), obs_pos_y.reshape(-1, 1)))

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
            obs_goal_condition_input_list.append(obs_goal_condition_input)

        obs_goal_condition_input_list = torch.vstack(obs_goal_condition_input_list)

        if sample_type == "full_sample":
            t_final_control_samples = sample_diffusion(condition_input=obs_goal_condition_input_list,
                                                       input_output_type="input_obs_goal_output_t_control",
                                                       checkpoint_parent_path=input_obs_goal_output_time_control_parent_path,
                                                       sample_num=sample_num * condition_seed_num,
                                                       diffusion_w=diffusion_w)
            obs_goal_condition_input_list = obs_goal_condition_input_list.detach().cpu().numpy()
            t_final_control_samples = t_final_control_samples.detach().cpu().numpy()

            obs_goal_t_final_control_samples = np.hstack((obs_goal_condition_input_list, t_final_control_samples))

        current_prediction_data_list.append(copy.copy(obs_goal_t_final_control_samples))

        # Data preparation #######################################################################################################
        # obs_pos, original range [2, 8]
        obs_goal_t_final_control_samples[:, :8] = obs_goal_t_final_control_samples[:, :8] * (
                    OBS_POS_MAX - OBS_POS_MIN) + OBS_POS_MIN
        # obs_radius, original range [0.5, 1.5]
        obs_goal_t_final_control_samples[:, 8:12] = obs_goal_t_final_control_samples[:, 8:12] * (
                    OBS_RADIUS_MAX - OBS_RADIUS_MIN) + OBS_RADIUS_MIN
        # car goal pos
        obs_goal_t_final_control_samples[:, 12:14] = obs_goal_t_final_control_samples[:, 12:14] * (
                    GOAL_POS_MAX - GOAL_POS_MIN) + GOAL_POS_MIN
        # t_final, original range [TIME_MIN, TIME_MAX]
        obs_goal_t_final_control_samples[:, 14] = obs_goal_t_final_control_samples[:, 14] * (
                    TIME_MAX - TIME_MIN) + TIME_MIN
        # Control, original range [CONTROL_MIN, CONTROL_MAX]
        obs_goal_t_final_control_samples[:, 15:] = obs_goal_t_final_control_samples[:, 15:] * (
                    CONTROL_MAX - CONTROL_MIN) + CONTROL_MIN
        print("data normalization is done")

        # Save ##########################################################################################################
        total_num = sample_num * condition_seed_num
        for num in range(total_num):
            curr_conditional_seed = 5000 + num // 10
            curr_initial_guess_seed = num % 10

            warmstart_data_parent_path = f"/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/warmstart_data/tabletop_v2/{data_type}"
            if not os.path.exists(warmstart_data_parent_path):
                os.makedirs(warmstart_data_parent_path, exist_ok=True)
            warmstart_data_path = f"{warmstart_data_parent_path}/{data_type}_condition_seed_{curr_conditional_seed}_initial_guess_seed_{curr_initial_guess_seed}.pkl"
            with open(warmstart_data_path, 'wb') as f:
                pickle.dump(obs_goal_t_final_control_samples[num, :], f)
            print(f"{warmstart_data_path} is saved")

        #############################################################################
        # check constraint violation
        current_prediction_data_list = np.vstack(current_prediction_data_list)
        current_prediction_data_tensor = torch.tensor(current_prediction_data_list)
        current_violation = get_constraint_violation_tabletop(x=current_prediction_data_tensor[:, 14:],
                                                         c=current_prediction_data_tensor[:, :14],
                                                         scale=torch.tensor(1.0),
                                                         device=current_prediction_data_tensor.device)
        print(f"data type is {data_type}, violation is {current_violation}")

        constraint_violation_list.append(current_violation)

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

    if input_output_type == "input_obs_goal_output_t_control":
        class_dim = 14
        channel = 1
        seq_length = 161
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
