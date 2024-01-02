import sys

sys.path.append('../')
sys.path.append('./')

import torch
from denoising_diffusion_pytorch.classifier_free_guidance_cond_1d import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from torch.utils.data import TensorDataset
import pickle
import numpy as np
from datetime import datetime

import argparse

def main():

    ####################################################################################################################
    # Parse the arguments
    args = parse_args()
    machine = args.machine
    unet_dim = args.unet_dim
    unet_dim_mults = tuple(map(int, args.unet_dim_mults.split(',')))
    embed_class_layers_dims = tuple(map(int, args.embed_class_layers_dims.split(',')))
    timesteps = args.timesteps
    objective = str(args.objective)
    batch_size = args.batch_size
    data_path = args.data_path
    cond_drop_prob = args.cond_drop_prob
    wandb_project_name = str(args.wandb_project_name)
    class_dim = args.class_dim
    channel_num = args.channel_num
    seq_length = args.seq_length
    training_data_type = str(args.training_data_type)

    ####################################################################################################################
    # Build the model
    model = Unet1D(
        dim=unet_dim,
        channels=channel_num,
        dim_mults=unet_dim_mults,
        embed_class_layers_dims=embed_class_layers_dims,
        class_dim=class_dim,
        cond_drop_prob=cond_drop_prob
    )

    diffusion = GaussianDiffusion1D(
        model=model,
        seq_length=seq_length,
        timesteps=timesteps,
        objective=objective,
        # objective='pred_noise',
    ).cuda()

    # # Random dataset
    # training_data_num = 64
    # training_seq = torch.rand(training_data_num, 3, 20)  # images are normalized from 0 to 1
    # training_seq_classes = torch.rand(training_data_num, 5)  # say 10 classes
    # dataset = TensorDataset(training_seq, training_seq_classes)

    # CR3BP dataset
    # data_path = "data/CR3BP/cr3bp_time_mass_alpha_control_part_4_250k_each.pkl"
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    # set up the data
    x = data[:, class_dim:].astype(np.float32).reshape(data.shape[0], channel_num, seq_length)
    c = data[:, :class_dim].astype(np.float32).reshape(data.shape[0], class_dim)
    training_seq = torch.tensor(x)
    training_seq_classes = torch.tensor(c)
    dataset = TensorDataset(training_seq, training_seq_classes)

    # TODO: one loss step ##################################################
    # loss = diffusion(training_seq, classes=training_seq_classes)
    # loss.backward()
    #
    # TODO: use trainer ###################################################

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # Reconfigure the tuple variables to string
    unet_dim_mults_in_str = "_".join(map(str, unet_dim_mults))
    embed_class_layers_dims_in_str = "_".join(map(str, embed_class_layers_dims))
    if machine == "ubuntu":
        results_folder = f"results/{training_data_type}/unet_{unet_dim}_mults_{unet_dim_mults_in_str}_embed_class_{embed_class_layers_dims_in_str}_timesteps_{timesteps}_objective_{objective}_batch_size_{batch_size}/{current_time}"
        num_workers = 1
    elif machine == "della":
        results_folder = f"/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/results/{training_data_type}/unet_{unet_dim}_mults_{unet_dim_mults_in_str}_embed_class_{embed_class_layers_dims_in_str}_timesteps_{timesteps}_objective_{objective}_batch_size_{batch_size}/{current_time}"
        num_workers = 1

    step_per_epoch = int(data.shape[0] / batch_size)
    max_epoch = 150

    trainer = Trainer1D(
        diffusion_model=diffusion,
        dataset=dataset,
        train_batch_size=batch_size,
        train_lr=8e-5,
        train_num_steps=step_per_epoch * max_epoch,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        results_folder=results_folder,
        num_workers=num_workers,
        wandb_project_name=wandb_project_name
    )
    trainer.train()

    # do above for many steps
    sampled_seq = diffusion.sample(
        classes=training_seq_classes[:10, :].cuda(),
        cond_scale=6.,
        # condition scaling, anything greater than 1 strengthens the classifier free guidance. reportedly 3-8 is good empirically
    )

    print(sampled_seq.shape)  # (64, 3, 20)

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for diffusion models")

    # Machine
    parser.add_argument('--machine',
                        type=str,
                        default="ubuntu",
                        help="Machine to run this code")

    # Unet 1D parameters
    parser.add_argument('--unet_dim',
                        type=int,
                        default=64,
                        help='Dimension of the first layer of Unet')
    parser.add_argument('--unet_dim_mults',
                        type=str,
                        default="1,1,1",
                        help='List of dimension multipliers for Unet, currently at most 4 layers since we can only downsample 20 dim 4 times.')
    parser.add_argument('--embed_class_layers_dims',
                        type=str,
                        default="16,16",
                        help='List of dimension for embedding class layers')
    parser.add_argument('--cond_drop_prob',
                        type=float,
                        default=0.1,
                        help='Probability of dropping the condition input')
    parser.add_argument('--channel_num',
                        type=int,
                        default=3,
                        help='Channel number of the data')

    # GaussianDiffusion1D parameters
    parser.add_argument('--timesteps',
                        type=int,
                        default=500,
                        help='Timesteps for the diffusion process')
    parser.add_argument('--objective',
                        type=str,
                        default='pred_noise',
                        choices=['pred_v', 'pred_noise'],
                        help='Objectives for the diffusion model')
    parser.add_argument('--seq_length',
                        type=int,
                        default=20,
                        help='length of the data sequence')

    # Trainer1D parameters
    parser.add_argument('--batch_size',
                        type=int,
                        default=512,
                        help='Batch size for training')
    parser.add_argument('--data_path',
                        type=str,
                        default="data/CR3BP/cr3bp_time_mass_alpha_control_part_4_250k_each.pkl",
                        help="cr3bp data path")
    parser.add_argument('--wandb_project_name',
                        type=str,
                        default="diffusion_for_cr3bp_test",
                        help="project name for wandb")

    # Training data parameters
    parser.add_argument('--class_dim',
                        type=int,
                        default=5,
                        help='Dimension of the class variable')
    parser.add_argument('--training_data_type',
                        type=str,
                        default='cr3bp_cond_time_mass_alpha_data_control',
                        help='specify the condition input and the training data')

    return parser.parse_args()

if __name__ == "__main__":
    main()