import torch
import pickle

def get_constraint_violation_tabletop(x, c, scale, device):

    # Scale back to original x (t_final, control)
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

    timestep = 40
    batch_size = x.size()[0]

    # Unpack x

    # Scale and unpack x
    original_x = torch.zeros_like(x).to(device)
    original_x[:, 0] = x[:, 0] * (TIME_MAX - TIME_MIN) + TIME_MIN
    original_x[:, 1:] = x[:, 1:] * (CONTROL_MAX - CONTROL_MIN) + CONTROL_MIN

    x_sol = {}
    x_sol["t_final"] = original_x[:, 0]
    x_sol["car_0_u0"] = original_x[:, 1:1 + timestep]
    x_sol["car_0_u1"] = original_x[:, 1 + timestep:1 + 2 * timestep]

    car_start_pos = torch.tensor([[5.0, 5.0]]).to(device)

    state_x, state_y = integrate_dynamics(x_sol=x_sol,
                                          car_num=1, u_num_per_car=2,
                                          car_start_pos=car_start_pos,
                                          timestep=timestep,
                                          batch_size=batch_size)

    # print(f"state x is {state_x[0, :]}")

    # Unpack c (obs_pos, obs_radius)
    original_c = torch.zeros_like(c).to(device)
    original_c[:, :8] = c[:, :8] * (OBS_POS_MAX - OBS_POS_MIN) + OBS_POS_MIN
    original_c[:, 8:12] = c[:, 8:12] * (OBS_RADIUS_MAX - OBS_RADIUS_MIN) + OBS_RADIUS_MIN
    original_c[:, 12:14] = c[:, 12:14] * (GOAL_POS_MAX - GOAL_POS_MIN) + GOAL_POS_MIN

    obs_pos = original_c[:, :8].reshape(-1, 4, 2)
    obs_radius = original_c[:, 8:12]
    car_goal_pos = original_c[:, 12:14]

    # Other parameters
    obs_num = 4
    car_num = 1
    car_goal_radius = torch.tensor(0.2).to(device)
    car_radius = torch.tensor(0.2).to(device)

    # goal reaching constraints
    goal_reaching_violation = torch.zeros((batch_size, car_num)).to(device)
    for i in range(car_num):
        dist_to_goal_square = (state_x[:, i, -1] - car_goal_pos[:, 0]) ** 2 + (state_y[:, i, -1] - car_goal_pos[:, 1]) ** 2
        threshold = car_goal_radius ** 2
        goal_reaching_violation[:, i] = torch.max(torch.tensor(0.0).to(device), dist_to_goal_square - threshold)
    goal_reaching_violation = torch.sum(goal_reaching_violation, dim=1)

    # Obstacle avoidance constraints
    obstacle_avoidance_violation = torch.zeros((batch_size, car_num, obs_num, timestep + 1)).to(device)
    for i in range(car_num):
        for j in range(obs_num):
            dist_to_obstacle_square = (state_x[:, i, :] - obs_pos[:, j, 0].reshape(-1, 1)) ** 2 + (state_y[:, i, :] - obs_pos[:, j, 1].reshape(-1, 1)) ** 2
            threshold = (car_radius + obs_radius[:, j]).reshape(-1, 1) ** 2
            obstacle_avoidance_violation[:, i, j, :] = torch.max(torch.tensor(0.0).to(device), threshold - dist_to_obstacle_square)
    obstacle_avoidance_violation = torch.sum(obstacle_avoidance_violation, dim=[1, 2, 3])


    violation = goal_reaching_violation + obstacle_avoidance_violation

    violation = violation * scale
    violation = torch.mean(violation)

    return violation

def integrate_dynamics(x_sol, car_num, u_num_per_car, car_start_pos, timestep, batch_size):
    t_final = x_sol["t_final"]
    car_control = torch.zeros((batch_size, car_num, timestep, u_num_per_car)).to(device)
    for i in range(car_num):
        for k in range(u_num_per_car):
            car_control[:, i, :, k] = x_sol[f"car_{i}_u{k}"]

    # Integrate the x* solution through the dynamics
    dt = t_final.unsqueeze(1) / timestep  # Shape: (batch_size, 1)

    state_x = torch.zeros((batch_size, car_num, timestep + 1)).to(device)
    state_y = torch.zeros((batch_size, car_num, timestep + 1)).to(device)

    # Initial value setup
    state_x[:, :, 0] = car_start_pos[:, 0].unsqueeze(0).expand(batch_size, -1)
    state_y[:, :, 0] = car_start_pos[:, 1].unsqueeze(0).expand(batch_size, -1)

    # Configure dynamics
    dx = lambda v_x: v_x.clone()
    dy = lambda v_y: v_y.clone()

    # RK4 integration without explicit batch loop
    for t in range(timestep):
        v_x = car_control[:, :, t, 0]
        v_y = car_control[:, :, t, 1]

        # Compute the increments for each car using vectorized operations
        k1_x = dx(v_x)
        k1_y = dy(v_y)

        k2_x = dx(v_x + k1_x * dt / 2)
        k2_y = dy(v_y + k1_y * dt / 2)

        k3_x = dx(v_x + k2_x * dt / 2)
        k3_y = dy(v_y + k2_y * dt / 2)

        k4_x = dx(v_x + k3_x * dt)
        k4_y = dy(v_y + k3_y * dt)

        state_x[:, :, t + 1] = state_x[:, :, t] + (dt / 6) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        state_y[:, :, t + 1] = state_y[:, :, t] + (dt / 6) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)

    return state_x, state_y

if __name__ == "__main__":
    use_local_optimal_data = True
    # use_local_optimal_data = False
    device = "cuda:0"
    torch.autograd.set_detect_anomaly(True)
    data_num = 10

    if use_local_optimal_data:
        data_path = "/home/anjian/Desktop/project/trajectory_optimization/snopt_python/Data/local_optimal_data/tabletop/obstacle_goal_time_control_data_obj_6_num_202654.pkl"
        with open(data_path, 'rb') as f:
            local_optimal_data = pickle.load(f)

        local_optimal_data_to_use = local_optimal_data[:data_num, :]
        c = local_optimal_data_to_use[:, :14]
        x = local_optimal_data_to_use[:, 14:]

        # Random x
        x = torch.tensor(x).to(device)
        x.requires_grad_(True)
        c = torch.tensor(c).to(device)
    else:
        x = torch.rand(data_num, 81).to(device)
        x.requires_grad_(True)
        c = torch.rand(data_num, 14).to(device)
    scale = torch.ones(data_num).to(device)
    violation = get_constraint_violation_tabletop(x, c, scale, device)
    print(f"total violation is {violation}")
    # print(torch.autograd.grad(violation, x, create_graph=True))
    print(torch.max(torch.autograd.grad(violation, x, create_graph=True)[0]))
    print(torch.min(torch.autograd.grad(violation, x, create_graph=True)[0]))