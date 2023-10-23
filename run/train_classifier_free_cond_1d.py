import torch
from denoising_diffusion_pytorch.classifier_free_guidance_cond_1d import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from torch.utils.data import TensorDataset
import pickle
import numpy as np
from datetime import datetime


def main():

    class_dim = 5

    model = Unet1D(
        dim=64,
        channels=3,
        dim_mults=(1, 2, 4),
        # dim_mults = (1, 2, 4, 8, 16),
        class_dim=class_dim,
        cond_drop_prob=0.1
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length=20,
        timesteps=1000,
        objective='pred_v'
    ).cuda()

    # # Random dataset
    # training_data_num = 64
    # training_seq = torch.rand(training_data_num, 3, 20)  # images are normalized from 0 to 1
    # training_seq_classes = torch.rand(training_data_num, 5)  # say 10 classes
    # dataset = TensorDataset(training_seq, training_seq_classes)

    # CR3BP dataset
    data_path = "data/CR3BP/cr3bp_time_mass_alpha_control_part_4_250k_each.pkl"
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
    results_folder = f"/home/anjian/Desktop/project/denoising-diffusion-pytorch/results/{current_time}"
    trainer = Trainer1D(
        diffusion_model=diffusion,
        dataset=dataset,
        train_batch_size=1024,
        train_lr=8e-5,
        train_num_steps=30000,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        results_folder=results_folder,
    )
    trainer.train()

    # do above for many steps
    sampled_images = diffusion.sample(
        classes=training_seq_classes[:10, :].cuda(),
        cond_scale=6.,
        # condition scaling, anything greater than 1 strengthens the classifier free guidance. reportedly 3-8 is good empirically
    )

    print(sampled_images.shape)  # (64, 3, 20)


if __name__ == "__main__":
    main()