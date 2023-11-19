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
    unet_dim = args.unet_dim
    unet_dim_mults = args.unet_dim_mults
    embed_class_layers_dims = args.embed_class_layers_dims
    timesteps = args.timesteps
    objective = args.objective
    batch_size = args.batch_size
    data_path = args.data_path

    ####################################################################################################################
    # Build the model
    class_dim = 5

    model = Unet1D(
        dim=unet_dim,
        channels=3,
        dim_mults=unet_dim_mults,
        embed_class_layers_dims=embed_class_layers_dims,
        class_dim=class_dim,
        cond_drop_prob=0.1
    )

    diffusion = GaussianDiffusion1D(
        model=model,
        seq_length=20,
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
    x = data[:, 5:].astype(np.float32).reshape(data.shape[0], 3, 20)
    c = data[:, :5].astype(np.float32)
    training_seq = torch.tensor(x)
    training_seq_classes = torch.tensor(c)
    dataset = TensorDataset(training_seq, training_seq_classes)

    # TODO: one loss step ##################################################
    # loss = diffusion(training_seq, classes=training_seq_classes)
    # loss.backward()
    #
    # TODO: use trainer ###################################################

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_folder = f"results/{current_time}"
    trainer = Trainer1D(
        diffusion_model=diffusion,
        dataset=dataset,
        train_batch_size=batch_size,
        train_lr=8e-5,
        train_num_steps=30000,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        results_folder=results_folder,
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

    # Unet 1D parameters
    parser.add_argument('--unet_dim',
                        type=int,
                        default=64,
                        help='Dimension of the first layer of Unet')
    parser.add_argument('--unet_dim_mults',
                        nargs='+',
                        type=tuple,
                        default=(1, 2, 4),
                        help='List of dimension multipliers for Unet, currently at most 4 layers since we can only downsample 20 dim 4 times.')
    parser.add_argument('--embed_class_layers_dims',
                        type=tuple,
                        default=(256, 256),
                        help='List of dimension for embedding class layers')

    # GaussianDiffusion1D parameters
    parser.add_argument('--timesteps',
                        type=int,
                        default=1000,
                        help='Timesteps for the diffusion process')
    parser.add_argument('--objective',
                        type=str,
                        default='pred_v',
                        choices=['pred_v', 'pred_noise'],
                        help='Objectives for the diffusion model')

    # Trainer1D parameters
    parser.add_argument('--batch_size',
                        type=int,
                        default=1024,
                        help='Batch size for training')
    parser.add_argument('--data_path',
                        type=str,
                        default="data/CR3BP/cr3bp_time_mass_alpha_control_part_4_250k_each.pkl",
                        help="cr3bp data path")

    return parser.parse_args()

if __name__ == "__main__":
    main()