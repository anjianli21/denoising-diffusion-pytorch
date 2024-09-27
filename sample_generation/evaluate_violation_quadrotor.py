import pickle
import glob
import os
import torch
from denoising_diffusion_pytorch.constraint_violation_function_improved_quadrotor import get_constraint_violation_quadrotor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import copy


def evaluate_violation_car():
    num = 2000

    # data_type_list = [
    #     "quadrotor_diffusion_seed_0",
    #     "quadrotor_diffusion_seed_1",
    #     "quadrotor_diffusion_seed_2",
    # ]

    data_type_list = [
        "cvae_lstm_seed_0",
        "cvae_lstm_seed_1",
        "cvae_lstm_seed_2",
    ]

    # data_type_list = [
    #     "quadrotor_constrained_diffusion_seed_0",
    #     "quadrotor_constrained_diffusion_seed_1",
    #     "quadrotor_constrained_diffusion_seed_2",
    # ]


    data_parent_path_list = [f"/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/quadrotor/{data_type}" for data_type in data_type_list]

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
            data_path = "data/quadrotor/quadrotor_obs_goalperturb_time_control_num_179219.pkl"
        else:
            data_path = f"{data_parent_path}/{data_type}_num_{num}.pkl"

        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        if i == 0:
            total_data = copy.copy(data)
        else:
            total_data = np.vstack([total_data, data])

    data = torch.tensor(total_data).cuda()
    c = data[:, :17]
    x = data[:, 17:]

    violation_constraint = get_constraint_violation_quadrotor(x=x, c=c, scale=1.0, device=data.device)

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