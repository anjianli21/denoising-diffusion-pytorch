## Diffusion model for trajectory optimization in Pytorch

Classifier free 1D conditional diffusion model.

## Environment configuration

Refer to "setup.py".

## Usage: training

For car problem, without constraint violation loss, we train the model by

```python
$ WANDB_MODE=offline python run/train_classifier_free_cond_1d_improved_constrained_diffusion.py --training_random_seed=0 --training_data_num=114570 --result_folder=results/car/results --data_path=data/fixed_car_vary_obs/obstacle_time_control_data_obj_12_num_114570.pkl --class_dim=6 --channel_num=1 --seq_length=81 --training_data_type=vanilla_diffusion_seed_0 --batch_size=128 --constraint_loss_type=NA --task_type=car
```

For CR3BP problem, without constraint violation loss, we train the model by

```python
$ WANDB_MODE=offline python run/train_classifier_free_cond_1d_improved_constrained_diffusion.py --training_random_seed=0 --training_data_num=300000 --result_folder=results/cr3bp/results --data_path=data/CR3BP/cr3bp_alpha_time_mass_control.pkl --class_dim=1 --channel_num=1 --seq_length=64 --training_data_type=cr3bp_vanilla_diffusion_seed_0 --batch_size=128 --constraint_loss_type=NA --task_type=cr3bp
```

