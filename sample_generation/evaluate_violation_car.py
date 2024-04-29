import numpy as np
import copy
import pickle
import glob
import os
import torch
from denoising_diffusion_pytorch.constraint_violation_function_improved_car import get_constraint_violation_car
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_violation_car():
    num = 5000
    # data_parent_path = "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/car"
    #
    # data_type_list = [
    #     "full_data_114k_constraint_weight_0.01_condscale_6",
    #                   "uniform",
    #     "input_obs_output_time_control_obj_12_data_114k",
    #     "cvae_lstm",
    # ]

    # # # TODO: Improved constrained, threshold
    # data_parent_path = "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/car"
    # data_type_list = [
    #     # "car_constrained_improved_seed_0",
    #     "car_constrained_improved_seed_1",
    #     # "car_constrained_improved_seed_2",
    # ]

    # # Constrained
    # data_parent_path = "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/car/full_data_114k_constraint_weight_0.01_condscale_6"
    #
    # data_type_list = [
    #     "full_data_114k_constraint_weight_0.01_condscale_6_seed_0",
    #     "full_data_114k_constraint_weight_0.01_condscale_6_seed_1",
    #     "full_data_114k_constraint_weight_0.01_condscale_6_seed_2",
    # ]
    #
    # # TODO: uniform sample from training
    # data_parent_path = "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/car"
    #
    # data_type_list = [
    #     "uniform_from_training_seed_0",
    #     "uniform_from_training_seed_1",
    #     "uniform_from_training_seed_2",
    # ]

    # TODO: add statistical constraints
    # data_parent_path = "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/car"
    # data_type_list = [
    #     "car_constrained_step_10_type_scaled_weight_10_seed_0",
    #     # "car_constrained_step_10_type_threshold_weight_10_seed_0",
    #     # "car_constrained_step_500_type_scaled_weight_1_seed_0"
    # ]

    # # Diffusion
    # data_parent_path = "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/car/input_obs_output_time_control_obj_12_data_114k"
    #
    # data_type_list = [
    #     "input_obs_output_time_control_obj_12_data_114k_seed_0",
    #     # "input_obs_output_time_control_obj_12_data_114k_seed_1",
    #     # "input_obs_output_time_control_obj_12_data_114k_seed_2",
    # ]

    # # cvae lstm
    # data_parent_path = "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/car/cvae_lstm"
    #
    # data_type_list = [
    #     "cvae_lstm",
    #     "cvae_lstm_seed_0",
    #     "cvae_lstm_seed_1",
    # ]

    # # uniform
    # data_parent_path = "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/car/uniform"
    #
    # data_type_list = [
    #     "uniform",
    #     "uniform_seed_1",
    #     "uniform_seed_2",
    # ]

    # # TODO: DDDAS
    data_parent_path = "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/car"
    data_type_list = [
        # "car_constraint_gt_scaled_weight_01_seed_0",
        # "car_constraint_gt_std_absolute_weight_1_seed_0",
        # "car_constraint_gt_std_threshold_weight_1_seed_0",
        # "car_constraint_gt_std_weight_1_seed_0",
        # "car_constraint_one_over_t_weight_01_seed_0",
        # "car_constraint_gt_std_absolute_weight_01_seed_0",
        # "car_constraint_gt_std_threshold_weight_01_seed_0",
        "car_constraint_gt_std_weight_01_seed_0",
        # "car_constraint_gt_log_likelihood_weight_01_seed_0"
    ]

    for i in range(len(data_type_list)):
        data_type = data_type_list[i]
        data_path = f"{data_parent_path}/{data_type}/{data_type}_num_{num}.pkl"
        # data_path = f"{data_parent_path}/{data_type}_num_{num}.pkl"


        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        if i == 0:
            total_data = copy.copy(data)
        else:
            total_data = np.vstack([total_data, data])
    data = torch.tensor(total_data).cuda()
    c = data[:, :6]
    x = data[:, 6:]

    violation_constraint = get_constraint_violation_car(x=x, c=c, scale=1.0, device=data.device)

    print(violation_constraint)

    violation_constraint = violation_constraint.detach().cpu().numpy().tolist()
    print(violation_constraint)

    # Convert the list into a Pandas DataFrame
    df = pd.DataFrame(violation_constraint, columns=['ViolationConstraint'])

    # Compute basic statistics using pandas\
    print(f"{data_type} =======================================================================")
    stats = df.describe()
    print(stats)

    # Count the number of elements less than 1e-3
    count_less_than_1e3 = (df['ViolationConstraint'] < 1e-3).sum()
    print(f"Number of elements less than 1e-3: {count_less_than_1e3}")

    # Plot histogram using seaborn
    sns.histplot(df['ViolationConstraint'], bins=10, kde=True)
    plt.xlabel('Violation Constraint Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Violation Constraint Values')
    plt.show()



    return True

if __name__ == "__main__":
    evaluate_violation_car()