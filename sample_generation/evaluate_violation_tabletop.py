import pickle
import glob
import os
import torch
from denoising_diffusion_pytorch.constraint_violation_function_tabletop_setupv2_evaluate import get_constraint_violation_tabletop
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import copy


def evaluate_violation_car():
    num = 2000
    # data_parent_path = "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/tabletop"
    #
    # data_type_list = [
    #     "full_data_202k_constraint_weight_0.01_condscale_6",
    #     "full_data_202k_constraint_weight_0.001_condscale_6",
    #     "full_data_202k_constraint_weight_0.0001_condscale_6",
    #     "uniform",
    #     "input_obs_goal_output_time_control_obj_6",
    #     "cvae_lstm"
    # ]

    # # Constrained， weight 0.1
    # data_parent_path_list = [
    #         "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/tabletop_v2/tabletop_v2_constrained_diffusion_weight_01_seed_0",
    #         "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/tabletop_v2/tabletop_v2_constrained_diffusion_weight_01_seed_1",
    #         "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/tabletop_v2/tabletop_v2_constrained_diffusion_weight_01_seed_2",
    # ]
    # data_type_list = [
    #     "tabletop_v2_constrained_diffusion_weight_01_seed_0",
    #     "tabletop_v2_constrained_diffusion_weight_01_seed_1",
    #     "tabletop_v2_constrained_diffusion_weight_01_seed_2",
    # ]

    # # # Constrained
    # data_parent_path_list = [
    #         "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/tabletop_v2/tabletop_v2_constrained_diffusion_seed_0",
    #         "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/tabletop_v2/tabletop_v2_constrained_diffusion_seed_1",
    #         "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/tabletop_v2/tabletop_v2_constrained_diffusion_seed_2",
    # ]
    # data_type_list = [
    #     "tabletop_v2_constrained_diffusion_seed_0",
    #     "tabletop_v2_constrained_diffusion_seed_1",
    #     "tabletop_v2_constrained_diffusion_seed_2",
    # ]

    # # # Diffusion
    # data_parent_path_list = [
    #         "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/tabletop_v2/tabletop_v2_diffusion_seed_0",
    #         "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/tabletop_v2/tabletop_v2_diffusion_seed_1",
    #         "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/tabletop_v2/tabletop_v2_diffusion_seed_2",
    # ]
    # data_type_list = [
    #     "tabletop_v2_diffusion_seed_0",
    #     "tabletop_v2_diffusion_seed_1",
    #     "tabletop_v2_diffusion_seed_2",
    # ]

    # # # cvae lstm
    # data_parent_path_list = [
    #         "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/tabletop_v2/cvae_lstm_seed_0",
    #         "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/tabletop_v2/cvae_lstm_seed_1",
    #         "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/tabletop_v2/cvae_lstm_seed_2",
    # ]
    #
    # data_type_list = [
    #     "cvae_lstm_seed_0",
    #     "cvae_lstm_seed_1",
    #     "cvae_lstm_seed_2",
    # ]

    # uniform
    data_parent_path_list = [
        "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/tabletop_v2/uniform_seed_0",
        "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/tabletop_v2/uniform_seed_1",
        "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/tabletop_v2/uniform_seed_2",
]

    data_type_list = [
        "uniform_seed_0",
        "uniform_seed_1",
        "uniform_seed_2",
    ]

    # # local optimal
    # data_parent_path_list = [
    #     "",
    # ]
    #
    # data_type_list = [
    #     "local_optimal",
    # ]

    for i in range(len(data_type_list)):
        data_type = data_type_list[i]
        data_parent_path = data_parent_path_list[i]
        if data_type == "local_optimal":
            # Local optimal
            data_path = "data/tabletop_v2/tabletop_v2_obs_goal_time_control_num_237370.pkl"
        else:
            data_path = f"{data_parent_path}/{data_type}_num_{num}.pkl"

        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        if i == 0:
            total_data = copy.copy(data)
        else:
            total_data = np.vstack([total_data, data])

    data = torch.tensor(total_data).cuda()
    c = data[:, :14]
    x = data[:, 14:]

    violation_constraint = get_constraint_violation_tabletop(x=x, c=c, scale=1.0, device=data.device)

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