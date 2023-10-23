import torch
from denoising_diffusion_pytorch.classifier_free_guidance_cond_1d import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from torch.utils.data import TensorDataset
import pickle
import numpy as np

# 1. Initialization

class_dim = 5

model = Unet1D(
    dim=64,
    channels=3,
    dim_mults=(1, 2, 4),
    class_dim=class_dim,
    cond_drop_prob=0.1
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length=20,
    timesteps=1000,
    objective='pred_v'
).cuda()

results_folder = "/home/anjian/Desktop/project/denoising-diffusion-pytorch/results/2023-10-22_18-59-35"

trainer = Trainer1D(
    diffusion_model=diffusion,
    dataset=[0, 0, 0],
    results_folder=results_folder,
)

# 2. Load the saved checkpoint
milestone = "epoch-102"
trainer.load(milestone)

# 3. Use the loaded model for sampling
test_data_num = 10
test_seq_classes = torch.rand(test_data_num, 5)
sampled_images = diffusion.sample(
    classes=test_seq_classes.cuda(),
    cond_scale=6.,
)

print(sampled_images.shape)
