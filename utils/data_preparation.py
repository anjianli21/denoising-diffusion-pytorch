import numpy as np
import pickle
import matplotlib.pyplot as plt

def main():

    data_path = "data/CR3BP/cr3bp_time_mass_alpha_control_part_4_250k_each.pkl"

    with open(data_path, "rb") as f:
        cr3bp_time_mass_alpha_control = pickle.load(f)

    time = cr3bp_time_mass_alpha_control[:, :3]
    mass = cr3bp_time_mass_alpha_control[:, 3].reshape(-1, 1)
    alpha = cr3bp_time_mass_alpha_control[:, 4].reshape(-1, 1)
    control = cr3bp_time_mass_alpha_control[:, 5:]

    cr3bp_alpha_time_mass = np.hstack((np.hstack((alpha, time)), mass))
    cr3bp_alpha_time_mass_control = np.hstack((cr3bp_alpha_time_mass, control))

    alpha_time_mass_path = "data/CR3BP/cr3bp_alpha_time_mass.pkl"
    with open(alpha_time_mass_path, "wb") as fp:
        pickle.dump(cr3bp_alpha_time_mass, fp)
    print(f"{alpha_time_mass_path} is saved")

    alpha_time_mass_control_path = "data/CR3BP/cr3bp_alpha_time_mass_control.pkl"
    with open(alpha_time_mass_control_path, "wb") as fp:
        pickle.dump(cr3bp_alpha_time_mass_control, fp)
    print(f"{alpha_time_mass_control_path} is saved")

    return True

if __name__ == "__main__":
    main()