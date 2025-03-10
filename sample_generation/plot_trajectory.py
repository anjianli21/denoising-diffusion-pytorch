import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
import torch


# Increase the resolution of the plots
mpl.rcParams['figure.dpi'] = 300


def plot_trajectory(data, title, sampling_step, show_ylabel=True):
    car_num = 2
    obs_num = 2
    timestep = 20

    state_x = data["state_x"][sampling_step, :, :].cpu()
    state_y = data["state_y"][sampling_step, :, :].cpu()
    car_start_pos = data["car_start_pos"].cpu()
    car_radius = data["car_radius"].cpu()
    car_goal_pos = data["car_goal_pos"].cpu()
    obs_pos = data["obs_pos"][sampling_step, :].cpu()
    obs_radius = data["obs_radius"][sampling_step, :].cpu()

    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 20})

    # Plot the trajectories, start, and goal positions for each car
    colors = ['blue', 'green', 'orange', 'purple']

    for i in range(car_num):
        # Plot trajectory with lines
        ax.plot(state_x[i], state_y[i], color=colors[i], label=f'car {i + 1}')

        # Mark each discrete step with a circle
        step = 0
        for x, y in zip(state_x[i], state_y[i]):
            step_marker = Circle((x, y), radius=car_radius, color=colors[i], alpha=0.8 * step / timestep)
            ax.add_patch(step_marker)
            step += 1

        # Plot start position with a square marker
        ax.plot(car_start_pos[i][0], car_start_pos[i][1], color=colors[i], marker='s', markersize=10,
                # label=f'Car {i} Start'
                )
        ax.text(car_start_pos[i][0], car_start_pos[i][1], ' Start', color=colors[i], verticalalignment='bottom',
                horizontalalignment='right')

        # Plot goal position with a circle marker
        ax.plot(car_goal_pos[i][0], car_goal_pos[i][1], color=colors[i], marker='o', markersize=10, alpha=0.5,
                # label=f'Car {i} Goal'
                )
        ax.text(car_goal_pos[i][0], car_goal_pos[i][1], ' Goal', color=colors[i], verticalalignment='bottom',
                horizontalalignment='right')

    # Plot the obstacles
    for i in range(obs_num):
        if i == 0:
            obs = Circle((obs_pos[i][0], obs_pos[i][1]), obs_radius[i], color='red', alpha=0.5, label='obs')
        else:
            obs = Circle((obs_pos[i][0], obs_pos[i][1]), obs_radius[i], color='red', alpha=0.5)
        ax.add_patch(obs)

    # Set plot limits and labels
    ax.set_xlabel('X position', fontsize=12)
    ax.set_ylabel('Y position', fontsize=18)

    # ax.legend(fontsize=20)
    # ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.xticks([0.0, 2.5, 5.0, 7.5, 10.0])

    x_min = -2
    y_min = -2
    x_max = 11
    y_max = 11

    min_limit = min(x_min, y_min)
    max_limit = max(x_max, y_max)

    # Set plot limits
    ax.set_xlim(min_limit, max_limit)
    ax.set_ylim(min_limit, max_limit)

    # Add legend
    ax.legend(fontsize=16)

    # Show the plot
    plt.show()


def get_original(x, c, device):
    # Scale back to original x (t_final, control)
    TIME_MIN = 7.81728
    TIME_MAX = 12.0
    CONTROL_MIN = - 1.0005
    CONTROL_MAX = 1.0005
    OBS_POS_MIN = 2.0
    OBS_POS_MAX = 8.0
    OBS_RADIUS_MIN = 0.5
    OBS_RADIUS_MAX = 1.5

    timestep = 20
    batch_size = x.size()[0]

    original_x = torch.zeros_like(x).to(device)
    original_x[:, 0] = x[:, 0] * (TIME_MAX - TIME_MIN) + TIME_MIN
    original_x[:, 1:] = x[:, 1:] * (CONTROL_MAX - CONTROL_MIN) + CONTROL_MIN

    x_sol = {}
    x_sol["t_final"] = original_x[:, 0]
    x_sol["car_0_u0"] = original_x[:, 1:1 + timestep]
    x_sol["car_0_u1"] = original_x[:, 1 + timestep:1 + 2 * timestep]
    x_sol["car_1_u0"] = original_x[:, 1 + 2 * timestep:1 + 3 * timestep]
    x_sol["car_1_u1"] = original_x[:, 1 + 3 * timestep:1 + 4 * timestep]

    car_start_pos = torch.tensor([[0.0, 10.0], [10.0, 10.0]]).to(device)
    car_start_v = torch.tensor([0.0] * 2).to(device)
    car_start_theta = torch.tensor([0.0, 0.0]).to(device)

    state_x, state_y, state_v, state_theta = integrate_dynamics(x_sol=x_sol,
                                                                car_num=2, u_num_per_car=2,
                                                                car_start_pos=car_start_pos, car_start_v=car_start_v,
                                                                car_start_theta=car_start_theta, timestep=timestep,
                                                                batch_size=batch_size,
                                                                device=device)

    # Unpack c (obs_pos, obs_radius)
    original_c = torch.zeros_like(c).to(device)
    original_c[:, :4] = c[:, :4] * (OBS_POS_MAX - OBS_POS_MIN) + OBS_POS_MIN
    original_c[:, 4:] = c[:, 4:6] * (OBS_RADIUS_MAX - OBS_RADIUS_MIN) + OBS_RADIUS_MIN

    obs_pos = original_c[:, :4].reshape(-1, 2, 2)
    obs_radius = original_c[:, 4:6]

    # Other parameters
    obs_num = 2
    car_num = 2
    car_v_bound = torch.tensor([-2.0, 2.0]).to(device)
    car_goal_radius = torch.tensor(0.2).to(device)
    car_radius = torch.tensor(0.2).to(device)
    car_goal_pos = torch.tensor([[10.0, 0.0], [0.0, 0.0]]).to(device)

    data = {}
    data["state_x"] = state_x
    data["state_y"] = state_y
    data["car_start_pos"] = car_start_pos
    data["car_radius"] = car_radius
    data["car_goal_pos"]= car_goal_pos
    data["obs_pos"] = obs_pos
    data["obs_radius"] = obs_radius

    return data

def integrate_dynamics(x_sol, car_num, u_num_per_car, car_start_pos, car_start_v, car_start_theta, timestep, batch_size, device):
    t_final = x_sol["t_final"]
    car_control = torch.zeros((batch_size, car_num, timestep, u_num_per_car)).to(device)
    for i in range(car_num):
        for k in range(u_num_per_car):
            car_control[:, i, :, k] = x_sol[f"car_{i}_u{k}"]

    # Integrate the x* solution through the dynamics
    dt = t_final.unsqueeze(1) / timestep  # Shape: (batch_size, 1)

    state_x = torch.zeros((batch_size, car_num, timestep + 1)).to(device)
    state_y = torch.zeros((batch_size, car_num, timestep + 1)).to(device)
    state_v = torch.zeros((batch_size, car_num, timestep + 1)).to(device)
    state_theta = torch.zeros((batch_size, car_num, timestep + 1)).to(device)

    # Initial value setup
    state_x[:, :, 0] = car_start_pos[:, 0].unsqueeze(0).expand(batch_size, -1)
    state_y[:, :, 0] = car_start_pos[:, 1].unsqueeze(0).expand(batch_size, -1)
    state_v[:, :, 0] = car_start_v.unsqueeze(0).expand(batch_size, -1)
    state_theta[:, :, 0] = car_start_theta.unsqueeze(0).expand(batch_size, -1)

    # Configure dynamics
    dx = lambda v, theta: v.clone() * torch.cos(theta.clone())
    dy = lambda v, theta: v.clone() * torch.sin(theta.clone())
    dv = lambda a: a.clone()
    dtheta = lambda omega: omega.clone()

    # RK4 integration without explicit batch loop
    for t in range(timestep):
        a = car_control[:, :, t, 0]
        omega = car_control[:, :, t, 1]

        k1_x = dx(state_v[:, :, t], state_theta[:, :, t])
        k1_y = dy(state_v[:, :, t], state_theta[:, :, t])
        k1_v = dv(a)
        k1_theta = dtheta(omega)

        k2_x = dx(state_v[:, :, t] + k1_v * dt / 2, state_theta[:, :, t] + k1_theta * dt / 2)
        k2_y = dy(state_v[:, :, t] + k1_v * dt / 2, state_theta[:, :, t] + k1_theta * dt / 2)
        k2_v = dv(a + k1_v * dt / 2)
        k2_theta = dtheta(omega + k1_theta * dt / 2)

        k3_x = dx(state_v[:, :, t] + k2_v * dt / 2, state_theta[:, :, t] + k2_theta * dt / 2)
        k3_y = dy(state_v[:, :, t] + k2_v * dt / 2, state_theta[:, :, t] + k2_theta * dt / 2)
        k3_v = dv(a + k2_v * dt / 2)
        k3_theta = dtheta(omega + k2_theta * dt / 2)

        k4_x = dx(state_v[:, :, t] + k3_v * dt, state_theta[:, :, t] + k3_theta * dt)
        k4_y = dy(state_v[:, :, t] + k3_v * dt, state_theta[:, :, t] + k3_theta * dt)
        k4_v = dv(a + k3_v * dt)
        k4_theta = dtheta(omega + k3_theta * dt)

        state_x[:, :, t + 1] = state_x[:, :, t] + (dt / 6) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        state_y[:, :, t + 1] = state_y[:, :, t] + (dt / 6) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
        state_v[:, :, t + 1] = state_v[:, :, t] + (dt / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        state_theta[:, :, t + 1] = state_theta[:, :, t] + (dt / 6) * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)

    return state_x, state_y, state_v, state_theta

def main():
    data_path = "/home/anjian/Desktop/project/denoising-diffusion-pytorch/results/dddas/x_data.pkl"

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # TODO: Now only use first data
    sampling_steps = 500

    x = [curr_x[0, :, :] for curr_x in data["x_list"]]
    t = data["t_list"]
    c = data["condition"][0, :]

    x = torch.concat(x, dim=0)
    c = c.unsqueeze(0).expand(sampling_steps, -1)
    original_data = get_original(x, c, x.device)

    # TODO: plot trajectory for a specific sampling step
    # for sampling_step in [0, 10, 20, 30, 100, 250]:
    for sampling_step in [50]:
        title = f"Noisy data at sampling step {sampling_step}"
        plot_trajectory(data=original_data, title=title, sampling_step=sampling_step)
    pass


if __name__ == "__main__":
    main()