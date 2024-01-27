import torch

def get_constraint_violation_car(x, c, device):


    # Scale back to original x (t_final, control)
    TIME_MIN = 7.81728
    TIME_MAX = 12.0
    CONTROL_MIN = - 1.0005
    CONTROL_MAX = 1.0005
    timestep = 20
    batch_size = x.size()[0]

    # Unpack x
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
    car_start_theta = torch.tensor([7 * torch.pi / 4, 5 * torch.pi / 4]).to(device)

    state_x, state_y, state_v, state_theta = integrate_dynamics(x_sol=x_sol,
                                                                car_num=2, u_num_per_car=2,
                                                                car_start_pos=car_start_pos, car_start_v=car_start_v,
                                                                car_start_theta=car_start_theta, timestep=timestep,
                                                                batch_size=batch_size)

    # print(f"state x is {state_x[0, :]}")

    # Unpack c (obs_pos, obs_radius)
    original_c = torch.zeros_like(c).to(device)
    original_c[:, :4] = c[:, :4] * 6.0 + 2.0
    original_c[:, 4:] = c[:, 4:6] + 0.5

    obs_pos = original_c[:, :4].reshape(-1, 2, 2)
    obs_radius = original_c[:, 4:6]

    # Other parameters
    obs_num = 2
    car_num = 2
    car_v_bound = torch.tensor([-2.0, 2.0]).to(device)
    car_goal_radius = torch.tensor(0.2).to(device)
    car_radius = torch.tensor(0.2).to(device)
    car_goal_pos = torch.tensor([[10.0, 0.0], [0.0, 0.0]]).to(device)

    # V constraints
    # TODO: max v over trajectory violation? or each step v violation
    v_min_violation = torch.max(torch.tensor(0.0).to(device), car_v_bound[0] - state_v)
    v_min_violation = torch.sum(v_min_violation, dim=[1, 2])
    v_max_violation = torch.max(torch.tensor(0.0).to(device), state_v - car_v_bound[1])
    v_max_violation = torch.sum(v_max_violation, dim=[1, 2])

    # goal reaching constraints
    goal_reaching_violation = torch.zeros((batch_size, 2)).to(device)
    for i in range(car_num):
        dist_to_goal_square = (state_x[:, i, -1] - car_goal_pos[i][0]) ** 2 + (state_y[:, i, -1] - car_goal_pos[i][1]) ** 2
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

    # Car avoidance constraints
    car_avoidance_violation = torch.zeros((batch_size, car_num, obs_num, timestep + 1)).to(device)
    for i in range(car_num):
        for j in range(obs_num):
            if i != j:
                dist_to_car_square = (state_x[:, i, :] - state_x[:, j, 0].reshape(-1, 1)) ** 2 + (
                            state_y[:, i, :] - state_y[:, j, 1].reshape(-1, 1)) ** 2
                threshold = (car_radius + car_radius).reshape(-1, 1) ** 2
                car_avoidance_violation[:, i, j, :] = torch.max(torch.tensor(0.0).to(device), threshold - dist_to_car_square)
    car_avoidance_violation = torch.sum(car_avoidance_violation, dim=[1, 2, 3])

    violation = v_min_violation + v_max_violation + goal_reaching_violation + obstacle_avoidance_violation + car_avoidance_violation

    violation = torch.mean(violation)

    return violation

def integrate_dynamics(x_sol, car_num, u_num_per_car, car_start_pos, car_start_v, car_start_theta, timestep, batch_size):
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
    dx = lambda v, theta: v * torch.cos(theta)
    dy = lambda v, theta: v * torch.sin(theta)
    dv = lambda a: a
    dtheta = lambda omega: omega

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

if __name__ == "__main__":
    device = "cuda:0"
    # Random x
    x = torch.ones((512, 81)).to(device)
    c = torch.ones((512, 6)).to(device)
    violation = get_constraint_violation_car(x, c, device)
    print(f"total violation is {violation}")