import os
import sys

sys.path.append('/home/anjian/Desktop/project/generative_trajectory_optimization')
sys.path.append('/home/anjian/Desktop/project/denoising-diffusion-pytorch')
from models import * # TODO, import CVAE models and lstm models, from '/home/anjian/Desktop/project/generative_trajectory_optimization'

import numpy as np
import pickle
import yaml
import torch

MAX_MASS = 438.0  #
MINIMUM_SHOOTING_TIME = 0.0
MAXIMUM_SHOOTING_TIME = 40.0  # 50.0
CONTROL_SEGMENT = int(20)  # 60

CVAE_PARENT_DIR = "results/from_autodl/cvae_lstm/cr3bp/cvae_seed_77"
RNN_PARENT_DIR = "results/from_autodl/cvae_lstm/cr3bp/lstm_seed_77"

def main():
    time_mass_type_list = ["cvae"] * 1
    control_type_list = ["lstm"] * 1
    thrust_list = [0.15]

    sample_num = 2000
    seed = 0

    data_type = "cvae_lstm_thrust_0.15_seed_77"

    for i in range(len(time_mass_type_list)):
        time_mass_type = time_mass_type_list[i]
        control_type = control_type_list[i]
        thrust = thrust_list[i]


        # Sample time and mass ############################################################################################
        data_time_mass_alpha_normalized = None
        if time_mass_type == "cvae":
            # TODO: CVAE data
            #  time, mass are in normalized scale, [0, 1],
            data_time_mass_normalized = get_sample_from_cvae(sample_num=sample_num, seed=seed, alpha=thrust)
            normalized_alpha = (thrust - 0.1) / (1.0 - 0.1)
            normalized_alpha = normalized_alpha * np.ones((data_time_mass_normalized.shape[0], 1))
            data_time_mass_alpha_normalized = np.hstack((data_time_mass_normalized, normalized_alpha))
            print(f"thrust = {thrust}, time and mass generated from cvae")

        # Sample control ##################################################################################################
        if control_type == "lstm":
            # TODO: lstm data
            #  controls are in ux, uy, uz, second half correct order, normalized to [0, 1]
            data_control_normalized = get_sample_from_rnn(time_mass_samples=data_time_mass_normalized, alpha=thrust)
            print(f"thrust = {thrust}, control generated from lstm")


        ##################################################################################################################
        ##################################################################################################################
        # Convert the variable  ###################################################################################################
        if control_type == "lstm" or control_type == "vanilla_cvae":
            # TODO: cvae + lstm, vanilla_vae data
            #  time, mass are in normalized scale, [0, 1],
            #  controls are in ux, uy, uz, second half correct order, normalized to [0, 1]
            #  convert, scale time, mass, first convert ux uy uz to [-1, 1],
            #  then convert ux uy uz to spherical, convert the second half to revert order by revert_converted_u_data
            full_solution = np.hstack((np.hstack((data_time_mass_normalized[:, :3], data_control_normalized)),
                                   data_time_mass_normalized[:, -1].reshape(-1, 1)))

            # # TODO: for icml
            # if for_icml:
            #     icml_parent_path = ICML_PARENT_PATH
            #     if not os.path.exists(icml_parent_path):
            #         os.makedirs(icml_parent_path, exist_ok=True)
            #     cr3bp_time_mass_alpha_control_path = f"{icml_parent_path}/cr3bp_thrust_{thrust}_time_mass_{time_mass_type}_control_{control_type}_num_{sample_num}.pkl"
            #     with open(cr3bp_time_mass_alpha_control_path, "wb") as fp:  # write pickle
            #         pickle.dump(full_solution, fp)
            #         print(f"{cr3bp_time_mass_alpha_control_path} is saved!")

            full_solution[:, 0] = full_solution[:, 0] * (MAXIMUM_SHOOTING_TIME - MINIMUM_SHOOTING_TIME) + MINIMUM_SHOOTING_TIME
            full_solution[:, 1] = full_solution[:, 1] * 15.0
            full_solution[:, 2] = full_solution[:, 2] * 15.0
            full_solution[:, 3:-1] = full_solution[:, 3:-1] * 2.0 - 1.0
            full_solution[:, -1] = full_solution[:, -1] * (MAX_MASS - 415.0) + 415.0

            print("data normalization is done")

            time_control_mass_samples_spherical = revert_converted_u_data(full_solution)

            for num in range(sample_num):
                warmstart_data_parent_path = f"/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/icml_data/warmstart_data/{data_type}"
                if not os.path.exists(warmstart_data_parent_path):
                    os.makedirs(warmstart_data_parent_path, exist_ok=True)
                warmstart_data_path = f"{warmstart_data_parent_path}/{data_type}_seed_{num}.pkl"
                with open(warmstart_data_path, 'wb') as f:
                    pickle.dump(time_control_mass_samples_spherical[num, :], f)
                print(f"{warmstart_data_path} is saved")


def get_sample_from_cvae(sample_num, seed, alpha):
    parent_dir = CVAE_PARENT_DIR
    cvae_config_path = parent_dir + "/version_0/config.yaml"
    cvae_ckpt_path = parent_dir + "/version_0/training_stage_3/checkpoints/last.ckpt"

    with open(cvae_config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    model = generative_models[config['model_params']['model_name']](**config['model_params'],
                                                                    **config['data_params'])

    # https://pytorch-lightning.readthedocs.io/en/stable/deploy/production_intermediate.html
    checkpoint = torch.load(cvae_ckpt_path, map_location="cpu")
    model_weights = checkpoint["state_dict"]
    for key in list(model_weights):
        model_weights[key.replace("model.", "")] = model_weights.pop(key)
    model.load_state_dict(model_weights)
    model.eval()

    curr_device = "cpu"

    # Sample ###################################################################
    normalized_alpha = (alpha - 0.1) / (1.0 - 0.1)
    alpha_list = [normalized_alpha]

    # Sample x for alpha
    sample_x_list = []
    for i in range(len(alpha_list)):
        alpha = alpha_list[i]
        alpha = np.full((sample_num, 1), alpha, dtype=np.float32)
        output_sample = model.sample(num_samples=sample_num, alpha=torch.tensor(alpha),
                                          current_device=curr_device, seed=seed)
        output_sample = output_sample.cpu().data.numpy()

        sample_x_list.append(output_sample)

    sample_data = np.squeeze(np.asarray(sample_x_list))
    np.random.shuffle(sample_data)

    # check data status
    out_of_bound_num = 0
    for i in range(sample_num):
        if sample_data[i, 0] > 1.0 or sample_data[i, 0] < 0 or sample_data[i, 1] > 1.0 or sample_data[i, 1] < 0 or sample_data[i, 2] > 1.0 or \
                sample_data[i, 2] < 0:
            out_of_bound_num += 1
    print(f"out of bound num is {out_of_bound_num}, out ratio is {out_of_bound_num / sample_num:.2f}")

    return sample_data

def get_sample_from_rnn(time_mass_samples, alpha):

    rnn_parent_dir = RNN_PARENT_DIR
    rnn_config_path = f"{rnn_parent_dir}/version_0/config.yaml"
    rnn_ckpt_path = f"{rnn_parent_dir}/version_0/training_stage_3/checkpoints/last.ckpt"

    with open(rnn_config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    model = generative_models[config['model_params']['model_name']](**config['model_params'],
                                                                    **config['data_params'])

    # https://pytorch-lightning.readthedocs.io/en/stable/deploy/production_intermediate.html
    checkpoint = torch.load(rnn_ckpt_path, map_location="cpu")
    model_weights = checkpoint["state_dict"]
    for key in list(model_weights):
        model_weights[key.replace("model.", "")] = model_weights.pop(key)
    model.load_state_dict(model_weights)
    model.eval()

    curr_device = "cpu"

    # sample #########################################################################################################
    normalized_alpha = (alpha - 0.1) / (1.0 - 0.1)

    time_mass = torch.tensor(time_mass_samples, dtype=torch.float32).to(curr_device)
    alpha = torch.ones(time_mass.size()[0], dtype=torch.float32).reshape(-1, 1) * normalized_alpha

    alpha_time_mass = torch.hstack([alpha, time_mass])
    # time_mass_alpha = torch.cat((time_mass, alpha), 1)
    # [_, control] = model(time_mass_alpha)

    [_, control] = model(input=None, alpha=alpha_time_mass, control_label=None)


    return control.detach().cpu().numpy()

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


if __name__ == "__main__":
    main()