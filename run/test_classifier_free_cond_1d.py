import torch
from denoising_diffusion_pytorch.classifier_free_guidance_cond_1d import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from torch.utils.data import TensorDataset
import pickle
import numpy as np

# 1. Initialization
unet_dim = 128
unet_dim_mults = (4,4,8)
embed_class_layers_dims = (64, 128)
timesteps = 500
objective = "pred_v"
batch_size = 512

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
    model,
    seq_length=20,
    timesteps=timesteps,
    objective=objective
).cuda()

results_folder = '/home/anjian/Desktop/project/denoising-diffusion-pytorch/from_della/checkpoint/top_10/unet_128_mults_(4, 4, 8)_embed_class_(64, 128)_timesteps_500_objective_pred_noise_batch_size_512/2023-11-22_06-05-41'

trainer = Trainer1D(
    diffusion_model=diffusion,
    dataset=[0, 0, 0],
    results_folder=results_folder,
)

# 2. Load the saved checkpoint
milestone = "epoch-141"
trainer.load(milestone)

# 3. Use the loaded model for sampling
test_data_num = 10
test_seq_classes = torch.rand(test_data_num, 5)
sampled_images = diffusion.sample(
    classes=test_seq_classes.cuda(),
    cond_scale=6.,
)

print(sampled_images.shape)
