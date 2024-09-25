import os
import glob
import re
import time

# TODO: here is improved constraint, sampled average violation
from denoising_diffusion_pytorch.classifier_free_guidance_cond_1d_improved_constrained_diffusion import Unet1D, \
    GaussianDiffusion1D, Trainer1D
from denoising_diffusion_pytorch.constraint_violation_function_improved_quadrotor import \
    get_constraint_violation_quadrotor

import copy
import numpy as np
import pickle
import torch


def main():

    # TODO: change time range
    TIME_MIN = 0.
    TIME_MAX = 10.
    CONTROL_U1_MIN = - np.pi / 9 - .001
    CONTROL_U1_MAX = np.pi / 9 + .001
    CONTROL_U2_MIN = - np.pi / 9 - .001
    CONTROL_U2_MAX = np.pi / 9 + .001
    CONTROL_U3_MIN = 0. - .001
    CONTROL_U3_MAX = 1.5 * 9.81 + .001

    OBS_POS_X_MIN = -6.0
    OBS_POS_X_MAX = 6.0
    OBS_POS_Y_MIN = -3.0
    OBS_POS_Y_MAX = 3.0
    OBS_POS_Z_MIN = -3.0
    OBS_POS_Z_MAX = 3.0

    OBS_RADIUS_MIN = 1.5
    OBS_RADIUS_MAX = 3.5

    GOAL_PERTURBATION_RANGE = 2.

    obs_num = 4

    # sample_type_list = ["conditional_sample"]
    sample_type = "full_sample"

    # diffusion_w_list = [10.0, 5.0]
    diffusion_w = 5.0

    sample_num = 10
    condition_seed_num = 200

    condition_seed_list = [5000 + i for i in range(condition_seed_num)]

    # data_type_list = [
    #     "quadrotor_diffusion_seed_0",
    #     "quadrotor_diffusion_seed_1",
    #     "quadrotor_diffusion_seed_2",
    # ]

    data_type_list = [
        "quadrotor_constrained_diffusion_seed_0",
        # "quadrotor_constrained_diffusion_seed_1",
        # "quadrotor_constrained_diffusion_seed_2",
    ]

    # Configure path ##############################################################################################
    parent_path = f"results/iclr25/quadrotor/results"

    # DDDAS
    model_parent_path_list = [f"{parent_path}/{i}" for i in data_type_list]

    constraint_violation_list = []
    for i in range(len(data_type_list)):
        data_type = data_type_list[i]
        model_parent_path = model_parent_path_list[i]

        current_prediction_data_list = []

        obs_condition_goalperturb_input_list = []
        for j in range(len(condition_seed_list)):
            condition_seed = condition_seed_list[j]

            # Sample obs
            rng_condition = np.random.RandomState(seed=condition_seed)
            # obs sample
            is_condition_reasonable = False

            # Sample condition
            num_of_sample_until_feasible = 0
            while not is_condition_reasonable:
                # print("sample obs again")

                agent_start_pos = np.array([[-12.0, 0.0, 0.0]])
                # TODO: add random perturbation to the goal pos
                agent_goal_pos_perturbation = rng_condition.rand(1) * 2 * GOAL_PERTURBATION_RANGE - GOAL_PERTURBATION_RANGE
                agent_goal_pos = np.array([[12., 0., 0.]]) + agent_goal_pos_perturbation

                obs_radius = rng_condition.rand(obs_num) * (OBS_RADIUS_MAX - OBS_RADIUS_MIN) + OBS_RADIUS_MIN

                obs_pos_x = rng_condition.rand(obs_num - 1) * (OBS_POS_X_MAX - OBS_POS_X_MIN) + OBS_POS_X_MIN
                obs_pos_y = rng_condition.rand(obs_num - 1) * (OBS_POS_Y_MAX - OBS_POS_Y_MIN) + OBS_POS_Y_MIN
                obs_pos_z = rng_condition.rand(obs_num - 1) * (OBS_POS_Z_MAX - OBS_POS_Z_MIN) + OBS_POS_Z_MIN

                # random sample from 0.3 to 1.0
                center_obs_pos_x = (rng_condition.rand(1) * 0.4 + 0.3) * (OBS_POS_X_MAX - OBS_POS_X_MIN) + OBS_POS_X_MIN
                gradient = (agent_goal_pos[0][1] - agent_start_pos[0][1]) / (
                        agent_goal_pos[0][0] - agent_start_pos[0][0])
                center_obs_pos_y = agent_start_pos[0][1] + (
                        center_obs_pos_x - agent_start_pos[0][0]) * gradient
                center_obs_pos_z = agent_start_pos[0][2] + (
                        center_obs_pos_x - agent_start_pos[0][0]) * gradient

                all_obs_pos_x = np.hstack([obs_pos_x, center_obs_pos_x])
                all_obs_pos_y = np.hstack([obs_pos_y, center_obs_pos_y])
                all_obs_pos_z = np.hstack([obs_pos_z, center_obs_pos_z])

                obs_pos = np.hstack(
                    (all_obs_pos_x.reshape(-1, 1), all_obs_pos_y.reshape(-1, 1), all_obs_pos_z.reshape(-1, 1)))

                parameters = {}
                parameters["obs_radius"] = obs_radius
                parameters["obs_pos"] = obs_pos
                parameters["agent_goal_pos"] = agent_goal_pos
                parameters["agent_start_pos"] = agent_start_pos
                is_condition_reasonable = check_condition(parameters=parameters)

                num_of_sample_until_feasible += 1

            print(f"number of samples until feasible {num_of_sample_until_feasible}")
            obs_radius = obs_radius.reshape(1, 4)
            obs_radius = (obs_radius - OBS_RADIUS_MIN) / (OBS_RADIUS_MAX - OBS_RADIUS_MIN)

            obs_pos_x = obs_pos[:, 0]
            obs_pos_y = obs_pos[:, 1]
            obs_pos_z = obs_pos[:, 2]
            obs_pos_x = (obs_pos_x - OBS_POS_X_MIN) / (OBS_POS_X_MAX - OBS_POS_X_MIN)
            obs_pos_y = (obs_pos_y - OBS_POS_Y_MIN) / (OBS_POS_Y_MAX - OBS_POS_Y_MIN)
            obs_pos_z = (obs_pos_z - OBS_POS_Z_MIN) / (OBS_POS_Z_MAX - OBS_POS_Z_MIN)
            obs_pos = np.hstack([obs_pos_x.reshape(1, -1), obs_pos_y.reshape(1, -1), obs_pos_z.reshape(1, -1)])

            obs_condition_input = np.hstack([obs_pos, obs_radius])

            # Repeat the same obs input as the sample num
            obs_condition_input = np.tile(obs_condition_input, (sample_num, 1))

            # TODO: also normalize the goal pos perturbation
            agent_goal_pos_perturbation = (agent_goal_pos_perturbation + GOAL_PERTURBATION_RANGE) / (
                        2 * GOAL_PERTURBATION_RANGE)
            agent_goal_pos_perturbation = agent_goal_pos_perturbation.reshape(1, -1)
            agent_goal_pos_perturbation = np.tile(agent_goal_pos_perturbation, (sample_num, 1))

            # TODO: combine obs condition and goalperturb
            obs_condition_goalperturb_input = np.hstack([obs_condition_input, agent_goal_pos_perturbation])
            obs_condition_goalperturb_input = torch.tensor(obs_condition_goalperturb_input).float().cuda()
            obs_condition_goalperturb_input_list.append(obs_condition_goalperturb_input)

        obs_condition_goalperturb_input_list = torch.vstack(obs_condition_goalperturb_input_list)

        if sample_type == "full_sample":
            t_final_control_samples = sample_diffusion(condition_input=obs_condition_goalperturb_input_list,
                                                       input_output_type="input_obs_goalperturb_output_t_control",
                                                       checkpoint_parent_path=model_parent_path,
                                                       sample_num=sample_num * condition_seed_num,
                                                       diffusion_w=diffusion_w)
            obs_condition_goalperturb_input_list = obs_condition_goalperturb_input_list.detach().cpu().numpy()
            t_final_control_samples = t_final_control_samples.detach().cpu().numpy()

            obs_goalperturb_t_final_control_samples = np.hstack((obs_condition_goalperturb_input_list, t_final_control_samples))

        current_prediction_data_list.append(copy.copy(obs_goalperturb_t_final_control_samples))

        #############################################################################
        # check constraint violation
        current_prediction_data_list = np.vstack(current_prediction_data_list)
        current_prediction_data_tensor = torch.tensor(current_prediction_data_list)
        current_violation = get_constraint_violation_quadrotor(x=current_prediction_data_tensor[:, 17:],
                                                               c=current_prediction_data_tensor[:, :17],
                                                               scale=torch.tensor(1.0),
                                                               device=current_prediction_data_tensor.device)
        print(f"data type is {data_type}, violation is {current_violation}")

        constraint_violation_list.append(current_violation)

        # Save ##########################################################################################################
        sample_data_parent_path = f"/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/quadrotor/{data_type}"
        if not os.path.exists(sample_data_parent_path):
            os.makedirs(sample_data_parent_path, exist_ok=True)
        sample_data_path = f"{sample_data_parent_path}/{data_type}_num_{condition_seed_num * sample_num}.pkl"
        with open(sample_data_path, 'wb') as f:
            pickle.dump(obs_goalperturb_t_final_control_samples, f)
        print(f"{sample_data_path} is saved")

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

    if input_output_type == "input_obs_goalperturb_output_t_control":
        class_dim = 17
        channel = 1
        seq_length = 241
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
    to_print = False

    agent_num = 1
    agent_radius = 0.5
    obs_num = 4
    obs_radius = parameters["obs_radius"]
    obs_pos = parameters["obs_pos"]

    agent_start_pos = parameters["agent_start_pos"]
    agent_goal_pos = parameters["agent_goal_pos"]

    # Check if agent start and goal positions are far enough from each other
    for i in range(agent_num):
        for j in range(agent_num):
            if i != j:
                if np.linalg.norm(agent_start_pos[i, :] - agent_start_pos[j, :]) < 4 * agent_radius:
                    if to_print:
                        print(f"agent {i} and agent {j} start pos is too close")
                    return False
                if np.linalg.norm(agent_goal_pos[i, :] - agent_goal_pos[j, :]) < 4 * agent_radius:
                    if to_print:
                        print(f"agent {i} and agent {j} goal pos is too close")
                    return False

    # Check if agent start and goal positions are far enough from obstacles
    for i in range(agent_num):
        for j in range(obs_num):
            if np.linalg.norm(agent_start_pos[i, :] - obs_pos[j, :]) < 4 * agent_radius + obs_radius[j]:
                if to_print:
                    print(f"agent {i} start pos and obs {j} pos is too close")
                return False
            if np.linalg.norm(agent_goal_pos[i, :] - obs_pos[j, :]) < 4 * agent_radius + obs_radius[j]:
                if to_print:
                    print(f"agent {i} goal pos and obs {j} pos is too close")
                return False

    # Check if obstacles are away from each other
    for i in range(obs_num):
        for j in range(obs_num):
            if i != j:
                if np.linalg.norm(obs_pos[i, :] - obs_pos[j, :]) < obs_radius[i] + obs_radius[j] + 3 * agent_radius:
                    if to_print:
                        print(f"obs {i} and obs {j} start pos is too close")
                    return False

    return True


if __name__ == "__main__":
    main()
