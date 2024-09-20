import torch
import pickle


def get_constraint_violation_quadrotor(x, c, scale, device):
    # Scale back to original x (t_final, control)
    TIME_MIN = 4.33
    TIME_MAX = 4.37
    CONTROL_U1_MIN = - torch.pi / 9 - .001
    CONTROL_U1_MAX = torch.pi / 9 + .001
    CONTROL_U2_MIN = - torch.pi / 9 - .001
    CONTROL_U2_MAX = torch.pi / 9 + .001
    CONTROL_U3_MIN = 0. - .001
    CONTROL_U3_MAX = 1.5 * 9.81 + .001

    OBS_POS_MIN = -6.0
    OBS_POS_MAX = 6.0
    OBS_RADIUS_MIN = 2.0
    OBS_RADIUS_MAX = 4.0

    timestep = 80
    batch_size = x.size()[0]

    # Unpack x

    # Scale and unpack x
    original_x = torch.zeros_like(x).to(device)
    original_x[:, 0] = x[:, 0] * (TIME_MAX - TIME_MIN) + TIME_MIN
    original_x[:, 1:81] = x[:, 1:81] * (CONTROL_U1_MAX - CONTROL_U1_MIN) + CONTROL_U1_MIN
    original_x[:, 81:161] = x[:, 81:161] * (CONTROL_U2_MAX - CONTROL_U2_MIN) + CONTROL_U2_MIN
    original_x[:, 161:241] = x[:, 161:241] * (CONTROL_U3_MAX - CONTROL_U3_MIN) + CONTROL_U3_MIN

    x_sol = {}
    x_sol["t_final"] = original_x[:, 0]
    x_sol["agent_0_u0"] = original_x[:, 1:1 + timestep]
    x_sol["agent_0_u1"] = original_x[:, 1 + timestep:1 + 2 * timestep]
    x_sol["agent_0_u2"] = original_x[:, 1 + 2 * timestep:1 + 3 * timestep]

    agent_start_state = torch.tensor([[-12.0, 0., 0., 0., 0., 0., 0., 0., 0., 0.]]).to(device)
    disturbances = torch.zeros((timestep, 3)).to(device)

    state_history = integrate_dynamics(x_sol=x_sol,
                                       agent_num=1, u_num_per_agent=3,
                                       agent_start_state=agent_start_state,
                                       timestep=timestep,
                                       disturbances=disturbances,
                                       batch_size=batch_size,
                                       device=device)

    state_x, state_y, state_z = state_history[:, :, 0], state_history[:, :, 4], state_history[:, :, 8]

    # Unpack c (obs_pos, obs_radius)
    original_c = torch.zeros_like(c).to(device)
    original_c[:, :12] = c[:, :12] * (OBS_POS_MAX - OBS_POS_MIN) + OBS_POS_MIN
    original_c[:, 12:16] = c[:, 12:16] * (OBS_RADIUS_MAX - OBS_RADIUS_MIN) + OBS_RADIUS_MIN

    obs_pos = original_c[:, :12].reshape(-1, 4, 3)
    obs_radius = original_c[:, 12:16]

    # Other parameters
    obs_num = 4
    agent_num = 1
    agent_goal_radius = torch.tensor(1.0).to(device)
    agent_radius = torch.tensor(1.0).to(device)
    agent_goal_pos = torch.tensor([[12., 0., 0.]]).to(device)

    # goal reaching constraints
    goal_reaching_tolerance = torch.tensor(5e-2).to(device)
    obstacle_avoidance_tolerance = torch.tensor(5e-2).to(device)

    # goal_reaching_violation = torch.zeros((batch_size)).to(device)
    dist_to_goal_square = (state_x[:, -1] - agent_goal_pos[:, 0]) ** 2 + \
                          (state_y[:, -1] - agent_goal_pos[:, 1]) ** 2 + \
                          (state_z[:, -1] - agent_goal_pos[:, 2]) ** 2
    threshold = agent_goal_radius ** 2
    goal_reaching_violation = torch.max(torch.tensor(0.0).to(device),
                                              dist_to_goal_square - threshold - goal_reaching_tolerance)
    # goal_reaching_violation = torch.sum(goal_reaching_violation)

    # Obstacle avoidance constraints
    obstacle_avoidance_violation = torch.zeros((batch_size, obs_num, timestep + 1)).to(device)
    for j in range(obs_num):
        dist_to_obstacle_square = (state_x - obs_pos[:, j, 0].reshape(-1, 1)) ** 2 + \
                                  (state_y - obs_pos[:, j, 1].reshape(-1, 1)) ** 2 + \
                                  (state_z - obs_pos[:, j, 2].reshape(-1, 1)) ** 2
        threshold = (agent_radius + obs_radius[:, j]).reshape(-1, 1) ** 2
        obstacle_avoidance_violation[:, j, :] = torch.max(torch.tensor(0.0).to(device),
                                                             threshold - dist_to_obstacle_square - obstacle_avoidance_tolerance)
    obstacle_avoidance_violation = torch.sum(obstacle_avoidance_violation, dim=[1, 2])

    # print(f"goal_reaching_violation max {torch.max(goal_reaching_violation)}")
    # print(f"goal_reaching_violation min {torch.min(goal_reaching_violation)}")
    # print(f"obstacle_avoidance_violation max {torch.max(obstacle_avoidance_violation)}")

    violation = goal_reaching_violation + obstacle_avoidance_violation
    # print(f"max goal reaching violation {torch.max(goal_reaching_violation)}")
    # print(f"max obstacle avoidance violation {torch.max(obstacle_avoidance_violation)}")

    violation = violation * scale
    # violation = torch.mean(violation)

    return violation


def integrate_dynamics(x_sol, agent_num, u_num_per_agent, agent_start_state, timestep, disturbances, batch_size, device):
    t_final = x_sol["t_final"]
    agent_control = torch.zeros((batch_size, agent_num, timestep, u_num_per_agent), device=device)
    for i in range(agent_num):
        for k in range(u_num_per_agent):
            agent_control[:, i, :, k] = x_sol[f"agent_{i}_u{k}"]

    # Constants
    constants = torch.tensor([10.0, 8.0, 10.0, 0.91, 9.81], device=device)  # d_0, d_1, n_0, k_T, g

    dt = t_final / timestep

    state_history = torch.zeros((batch_size, timestep + 1, 10), device=device)
    state_history[:, 0] = agent_start_state.repeat(batch_size, 1)

    disturbances = disturbances.unsqueeze(0).repeat(batch_size, 1, 1)

    for t in range(timestep):
        state_history[:, t + 1] = rk4_step(
            state_history[:, t],
            agent_control[:, 0, t],  # Assuming agent_num is 1
            disturbances[:, t],
            dt,
            constants
        )

    return state_history

def quadrotor_dynamics(state, control, disturbance, constants):
    x, v_x, theta_x, omega_x, y, v_y, theta_y, omega_y, z, v_z = state.unbind(dim=-1)
    a_x, a_y, a_z = control.unbind(dim=-1)
    d_x, d_y, d_z = disturbance.unbind(dim=-1)

    d_0, d_1, n_0, k_T, g = constants

    return torch.stack([
        v_x + d_x,
        g * torch.tan(theta_x),
        -d_1 * theta_x + omega_x,
        -d_0 * theta_x + n_0 * a_x,
        v_y + d_y,
        g * torch.tan(theta_y),
        -d_1 * theta_y + omega_y,
        -d_0 * theta_y + n_0 * a_y,
        v_z + d_z,
        k_T * a_z - g
    ], dim=-1)


def rk4_step(state, control, disturbance, dt, constants):
    k1 = quadrotor_dynamics(state, control, disturbance, constants)
    k2 = quadrotor_dynamics(state + 0.5 * dt.unsqueeze(-1) * k1, control, disturbance, constants)
    k3 = quadrotor_dynamics(state + 0.5 * dt.unsqueeze(-1) * k2, control, disturbance, constants)
    k4 = quadrotor_dynamics(state + dt.unsqueeze(-1) * k3, control, disturbance, constants)

    return state + (dt.unsqueeze(-1) / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

if __name__ == "__main__":
    use_local_optimal_data = True
    # use_local_optimal_data = False
    device = "cuda:0"
    torch.autograd.set_detect_anomaly(True)
    data_num = 2000
    timestep = 80

    if use_local_optimal_data:
        data_path = "data/quadrotor/quadrotor_obs_time_control_num_103898.pkl"
        # data_path = "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/sample_data/tabletop_v2/tabletop_v2_diffusion_seed_0/tabletop_v2_diffusion_seed_0_num_2000.pkl"
        with open(data_path, 'rb') as f:
            local_optimal_data = pickle.load(f)

        local_optimal_data_to_use = local_optimal_data[:data_num, :]
        c = local_optimal_data_to_use[:, :16]
        x = local_optimal_data_to_use[:, 16:]

        # Random x
        x = torch.tensor(x).to(device)
        x.requires_grad_(True)
        c = torch.tensor(c).to(device)
    else:
        x = torch.rand(data_num, timestep * 3 + 1).to(device)
        x.requires_grad_(True)
        c = torch.rand(data_num, 16).to(device)
    scale = torch.ones(data_num).to(device)
    violation = get_constraint_violation_quadrotor(x, c, scale, device)
    violation = torch.mean(violation)
    print(f"mean total violation is {violation}")
    # print(torch.autograd.grad(violation, x, create_graph=True))
    print(torch.max(torch.autograd.grad(violation, x, create_graph=True)[0]))
    print(torch.min(torch.autograd.grad(violation, x, create_graph=True)[0]))
