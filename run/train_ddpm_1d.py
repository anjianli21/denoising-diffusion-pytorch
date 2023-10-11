import torch
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D

def main():

    model = Unet1D(
        dim=32,
        dim_mults=(1, 2, 4),
        # dim_mults=(1, 2, 4, 8),
        channels=3
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length=20,
        timesteps=1000,
        objective='pred_v'  #
    )

    training_seq = torch.rand(64, 3, 20)  # features are normalized from 0 to 1
    dataset = Dataset1D(
        training_seq)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

    # Simple loss backward step ##################
    # loss = diffusion(training_seq)
    # loss.backward()
    #
    # # Or using trainer ############################
    trainer = Trainer1D(
        diffusion,
        dataset=dataset,
        train_batch_size=32,
        train_lr=8e-5,
        train_num_steps=100,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
    )
    trainer.train()

    # after a lot of training

    sampled_seq = diffusion.sample(batch_size=1)
    print(sampled_seq.shape)  # (4, 32, 128)
    print(sampled_seq)

if __name__ == "__main__":
    main()