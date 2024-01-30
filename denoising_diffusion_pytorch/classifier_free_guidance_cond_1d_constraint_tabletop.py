import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from accelerate import Accelerator
from ema_pytorch import EMA

from datetime import datetime
import wandb
import os
import copy

from tqdm.auto import tqdm

from denoising_diffusion_pytorch.version import __version__
from denoising_diffusion_pytorch.constraint_violation_function_tabletop import get_constraint_violation_tabletop

# constants

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


# helpers functions

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# normalization functions
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# TODO: used for classifier free ####################
def uniform(shape, device):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


# data

class Dataset1D(Dataset):
    def __init__(self, tensor: Tensor):
        super().__init__()
        self.tensor = tensor.clone()

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx].clone()


# small helper modules

class Residual(nn.Module):  # TODO: input a function f, return f(x) + x
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None, scale_factor=2):  # TODO: upsample and downsample using conv1d
    return nn.Sequential(
        nn.Upsample(scale_factor=scale_factor, mode='nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding=1)
    )

def Final_upsample_to_target_length(dim_out, dim_in, target_length):
    return nn.Sequential(
        nn.Upsample(size=target_length, mode='nearest'),
        nn.Conv1d(dim_out, dim_in, 3, padding=1)
    )

def Downsample(dim, dim_out=None, kernel=4):
    return nn.Conv1d(dim, default(dim_out, dim), kernel, 2, 1)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# sinusoidal positional embeds
# TODO： takes a tensor of shape (batch_size, 1) as input, return a tensor of shape (batch_size, dim), with dim being the dimensionality of the position embeddings
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):  # TODO: another positional encoding?
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules
# TODO: group number is set to 8 by default
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)  # TODO: ouput size is the same as input size
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, classes_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2)
        ) if exists(time_emb_dim) or exists(classes_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, class_emb=None):

        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            try:
                cond_emb = self.mlp(cond_emb)
            except:
                print("wrong size for conditional embedding!")
            cond_emb = rearrange(cond_emb, 'b c -> b c 1')

            scale_shift = cond_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):  # TODO: del the sequence
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads),
                      qkv)  # TODO: rearrange can help manage the dimension (even split or merge them)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h=self.heads)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

# model
class Unet1D(nn.Module):
    def __init__(
            self,
            dim,
            class_dim,
            seq_length,
            cond_drop_prob=0.5,
            mask_val=0.0,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            embed_class_layers_dims=(64, 64),
            channels=3,
            self_condition=False,
            resnet_block_groups=4,  # TODO: previously resnet_block_groups = 8
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            sinusoidal_pos_emb_theta=10000,
            attn_dim_head=32,
            attn_heads=4,
    ):
        super().__init__()

        # classifier free guidance stuff
        self.cond_drop_prob = cond_drop_prob
        self.mask_val = mask_val

        # determine dimensions
        self.dim = dim
        self.dim_mults = dim_mults
        self.embed_class_layers_dims = embed_class_layers_dims

        self.seq_length = seq_length

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, self.dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7,
                                   padding=3)  # TODO: currently the init convolution uses a kernel size = 7

        dims = [init_dim, *map(lambda m: self.dim * m, self.dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # TODO: partial function is used to create a "partial" version of a function. You can think of it as pre-setting some arguments of a function.
        #  Here, the ResnetBlock class is being partially set with the groups argument pre-defined. Any subsequent calls to block_klass will be the same as calling ResnetBlock with groups=resnet_block_groups.
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = self.dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(self.dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = self.dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # TODO: class embeddings ##, define the conditional embedding ####################################################
        # embedded_classes_dim = self.dim * 4
        # self.classes_mlp = nn.Sequential(
        #     nn.Linear(class_dim, embedded_classes_dim),
        #     nn.GELU(),
        #     nn.Linear(embedded_classes_dim, embedded_classes_dim)
        # )

        self.classes_mlp = self.build_classes_mlp(class_dim, self.embed_class_layers_dims)
        embedded_classes_dim = self.embed_class_layers_dims[-1]
        ###############################################################################################################

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            # TODO: Handle the case where seq_length = 1, no upsampling and downsampling, only conv1d
            if self.seq_length > 1:
                self.downs.append(nn.ModuleList([
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1)
                ]))
            else:
                # If sequence length is already 1, avoid downsampling and just use convolutional layers
                self.downs.append(nn.ModuleList([
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    # nn.Conv1d(dim_in, dim_out, 3, padding=1)
                    Downsample(dim_in, dim_out, 2) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1)
                ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head=attn_dim_head, heads=attn_heads)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            is_second_last = ind == (len(in_out) - 2)

            if self.seq_length > 1:
                self.ups.append(nn.ModuleList([
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    # TODO: since the original data size not necessarily can be divided by two,
                    #  so if is the second last year of upsample, we need make sure it upsamples to the original size
                    # Upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding=1),
                    Final_upsample_to_target_length(dim_out, dim_in, target_length=self.seq_length) if is_second_last else
                    (nn.Conv1d(dim_out, dim_in, 3, padding=1) if is_last else
                     Upsample(dim_out, dim_in))

                ]))
            else:
                self.ups.append(nn.ModuleList([
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    # nn.Conv1d(dim_out, dim_in, 3, padding=1),
                    (nn.Conv1d(dim_out, dim_in, 3, padding=1) if is_last else
                     Upsample(dim_out, dim_in, 1))
                ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(self.dim * 2, self.dim, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim)
        self.final_conv = nn.Conv1d(self.dim, self.out_dim, 1)

    def build_classes_mlp(self, class_dim, embed_class_layers_dims):

        layers = []
        input_dim = copy.copy(class_dim)
        for output_dim in embed_class_layers_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.GELU())
            input_dim = output_dim

        layers.pop()

        return nn.Sequential(*layers)

    # TODO: classifier free guidance loss, set conditional scale to be w,
    def forward_with_cond_scale(
            self,
            *args,
            cond_scale=1.,
            rescaled_phi=0.,
            **kwargs
    ):
        # TODO: fully conditional output
        logits = self.forward(*args, cond_drop_prob=0., **kwargs)

        if cond_scale == 1:
            return logits

        # TODO: unconditional output
        null_logits = self.forward(*args, cond_drop_prob=1., **kwargs)

        # weighted sum = unconditional + scale * (conditional - unconditional) = scale * conditional + (1 - scale) * unconditional
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if rescaled_phi == 0.:
            return scaled_logits

        # Rescale the cost with std
        # TODO: after rescale, the value of logit become nan
        std_fn = partial(torch.std, dim=tuple(range(1, scaled_logits.ndim)), keepdim=True)
        rescaled_logits = scaled_logits * (std_fn(logits) / (std_fn(scaled_logits) + 1e-6))
        if torch.isnan(rescaled_logits).any():
            return scaled_logits

        # weighted sum of rescaled and original sum. if rescaled_phi = 1.0, then fully use rescaled loss
        return rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)

    def forward(self,
                x,
                time,
                classes,
                cond_drop_prob=None):

        batch, device = x.shape[0], x.device

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # derive condition, with condition dropout for classifier free guidance ########################################
        # TODO: Depend on the conditional drop probability we set when initialized
        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device=device)

            classes_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes,
                torch.tensor(self.mask_val).cuda()  # TODO, when not keeping mask, using null_classes_emb to fill in
            )
            # TODO: embed the class to the conditional variable c
            c = self.classes_mlp(classes_emb)
        else:
            c = self.classes_mlp(classes)
        ################################################################################################################

        # Unet
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t, c)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        for block1, block2, attn, upsample in self.ups:
            # print(f"x size is {x.size()}, h pop size is {h[-1].size()}")
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, c)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t, c)
            x = attn(x)
            x = upsample(x)
            # print(f"x size is {x.size()}")

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t, c)
        return self.final_conv(x)


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion1D(nn.Module):
    def __init__(
            self,
            model,
            *,
            seq_length,
            timesteps=1000,
            sampling_timesteps=None,
            objective='pred_noise',
            beta_schedule='cosine',
            ddim_sampling_eta=0.,
            auto_normalize=True,
            constraint_violation_weight=0.001
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.seq_length = seq_length
        self.timesteps = timesteps
        self.objective = objective
        self.constraint_violation_weight = constraint_violation_weight

        assert objective in {'pred_noise', 'pred_x0',
                             'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        # define beta schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(self.timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(self.timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters
        # If sampling_timesteps is not defined, then use the number of training timesteps
        self.sampling_timesteps = default(sampling_timesteps,
                                          timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate loss weight
        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

        # whether to autonormalize
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, classes, cond_scale=6., rescaled_phi=0.7, clip_x_start=False,
                          rederive_pred_noise=False):
        model_output = self.model.forward_with_cond_scale(x, t, classes, cond_scale=cond_scale,
                                                          rescaled_phi=rescaled_phi)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, classes, cond_scale, rescaled_phi, clip_denoised=True):
        preds = self.model_predictions(x, t, classes, cond_scale, rescaled_phi)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    # @torch.no_grad()
    def p_sample(self, x, t: int, classes, cond_scale=6., rescaled_phi=0.7,
                 clip_denoised=True):  # TODO: sample function for diffusion
        b, *_, device = *x.shape, x.device
        if isinstance(t, int):
            batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        else:  # multiple different t
            batched_times = t.clone().to(device=x.device, dtype=torch.long)

        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, classes=classes,
                                                                          cond_scale=cond_scale,
                                                                          rescaled_phi=rescaled_phi,
                                                                          clip_denoised=clip_denoised)
        if isinstance(t, int):
            noise = torch.randn_like(x) if (t > 0) else 0.
        else:
            noise = torch.randn_like(x) if (len(t.shape) >= 1 or t > 0) else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, classes, shape, cond_scale=6., rescaled_phi=0.7):
        batch, device = shape[0], self.betas.device

        # TODO: default image to be randomly sampled from [0, 1]?
        img = torch.randn(shape, device=device)

        x_start = None

        # TODO: the giant sample loop, loop over each time steps, and sample the x_(t-1) at each steps
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img, x_start = self.p_sample(img, t, classes, cond_scale, rescaled_phi)

        img = self.unnormalize(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, classes, shape, cond_scale=6., rescaled_phi=0.7, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
                                                                                 0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, classes, cond_scale=cond_scale,
                                                             clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, classes, cond_scale=6., rescaled_phi=0.7):
        batch_size, seq_length, channels = classes.shape[0], self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(classes, (batch_size, channels, seq_length), cond_scale, rescaled_phi)

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device=device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled=False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, *, classes, noise=None):

        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # predict and take gradient step
        model_out = self.model(x, t, classes)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if self.is_ddim_sampling:
            t_next = t-1
            alpha = self.alphas_cumprod[t]
            alpha_next = self.alphas_cumprod[t_next]

            eta = 1. # TODO
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x_start)

            # x(t-1) = x(t) + epsilon
            x_t_1 = x_start * alpha_next.sqrt() + \
                    c * model_out + \
                    sigma * noise

        else:
            cond_scale=6.
            rescaled_phi=0.7
            x_t_1, _ = self.p_sample(x, t, classes, cond_scale, rescaled_phi)
        violation_loss = get_constraint_violation_tabletop(x_t_1.view(x_start.shape[0], -1), classes, 1./(t+1), x_start.device)
        coef = torch.tensor(self.constraint_violation_weight)
        loss = F.mse_loss(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean() + coef * violation_loss

    def forward(self, img, *args, **kwargs):
        b, c, n, device, seq_length, = *img.shape, img.device, self.seq_length
        assert n == seq_length, f'seq length must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)


# trainer class

class Trainer1D(object):
    def __init__(
            self,
            diffusion_model: GaussianDiffusion1D,
            dataset: Dataset,
            *,
            train_batch_size=16,
            gradient_accumulate_every=1,
            train_lr=1e-4,
            train_num_steps=100000,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.99),
            save_and_sample_every=1000,
            num_samples=25,
            results_folder='./results',
            amp=False,
            mixed_precision_type='fp16',
            split_batches=True,
            max_grad_norm=1.,
            num_workers=1,
            wandb_project_name="diffusion_for_cr3bp_test",
            training_data_range="0_1",
            training_data_num=300000,
    ):
        super().__init__()

        ##########################################################################################
        # Initialize wandb
        api_key = "0349e936b3e168fb936ff7d6e5bc1a0db52110d5"
        # if api_key:
        #     wandb.login(key=api_key)
        # else:
        #     raise Exception("not api key for wandb")
        wandb.login()
        hyperparameters = {
            'unet_dim': diffusion_model.model.dim,
            'unet_dim_mults': diffusion_model.model.dim_mults,
            'embed_class_layers_dims': diffusion_model.model.embed_class_layers_dims,
            'timesteps': diffusion_model.timesteps,
            'objective': diffusion_model.objective,
            'batch_size': train_batch_size,
            'cond_drop_prob': diffusion_model.model.cond_drop_prob,
            'mask_val': diffusion_model.model.mask_val,
            'training_data_range': training_data_range,
            'training_data_num': training_data_num
        }

        # Generate a unique name for the run
        run_name = f"unet_dim: {hyperparameters['unet_dim']}, unet_mults: ({','.join(map(str, hyperparameters['unet_dim_mults']))}), " \
                   f"embed_class_layer: ({','.join(map(str, hyperparameters['embed_class_layers_dims']))}), " \
                   f"steps: {hyperparameters['timesteps']}, obj: {hyperparameters['objective']}, " \
                   f"cond_drop_prob: {hyperparameters['cond_drop_prob']}, " \
                   f"batch_size: {hyperparameters['batch_size']}, mask_val: {hyperparameters['mask_val']}, " \
                   f"data_range: {hyperparameters['training_data_range']}, " \
                   f"data_num: {hyperparameters['training_data_num']}"

        wandb.init(project=wandb_project_name, name=run_name, config=hyperparameters)

        ############################################################################################
        # Configure the trainer
        # accelerator
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else 'no'
        )

        # model
        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps

        # dataset and dataloader
        # dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())
        #
        # dl = self.accelerator.prepare(dl)
        # self.dl = cycle(dl)

        # divide into training and validation dataset
        # Split the dataset
        train_length = int(0.9 * len(dataset))
        val_length = len(dataset) - train_length
        train_dataset, val_dataset = random_split(dataset, [train_length, val_length])

        # Create separate dataloaders
        train_dl = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True,
                              num_workers=num_workers)
        val_dl = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False, pin_memory=True,
                            num_workers=num_workers)
        # train_dl = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True,
        #                       num_workers=cpu_count())
        # val_dl = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False, pin_memory=True,
        #                     num_workers=cpu_count())

        train_dl = self.accelerator.prepare(train_dl)
        val_dl = self.accelerator.prepare(val_dl)

        self.train_dl = cycle(train_dl)
        self.val_dl = val_dl  # We don't need to cycle the validation dataloader

        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        # configure the results folder
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok = True)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # caculate batches/steps per epoch
        self.batches_per_epoch = len(dataset) // self.batch_size
        self.train_lr = train_lr

        # Save the best checkpoints
        self.best_checkpoints = []

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        #TODO #############################################################################################
        # Modify state dict if necessary
        ema_state_dict = data['ema']
        if 'initted' in data['ema'] and ema_state_dict['initted'].shape == torch.Size([]):
            ema_state_dict['initted'] = ema_state_dict['initted'].unsqueeze(0)
        if 'step' in ema_state_dict and ema_state_dict['step'].shape == torch.Size([]):
            ema_state_dict['step'] = ema_state_dict['step'].unsqueeze(0)

        # Print the shapes of initted and step
        print("Shape of initted in EMA state dict:", ema_state_dict.get('initted', 'Not found').shape)
        print("Shape of step in EMA state dict:", ema_state_dict.get('step', 'Not found').shape)
        #############################################################################################

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            # self.ema.load_state_dict(data["ema"])
            # TODO: use modified ema_state_dict
            self.ema.load_state_dict(ema_state_dict)

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            # Define a variable to track the best validation loss
            best_val_loss = torch.tensor(float("inf"))

            while self.step < self.train_num_steps:

                total_loss = 0.

                # Compute gradient for number of self.gradient_accumulate_every steps
                for _ in range(self.gradient_accumulate_every):
                    # data = next(self.train_dl).to(device)

                    # Unpack data into training_sequence and training_classes
                    training_sequence, training_classes = next(self.train_dl)
                    training_sequence, training_classes = training_sequence.to(device), training_classes.to(device)

                    with self.accelerator.autocast():
                        loss = self.model(training_sequence, classes=training_classes)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                # Log training loss
                wandb.log({'train_loss': total_loss, 'step': self.step})

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step % self.batches_per_epoch == 0 and self.step != 0:
                        milestone = self.step // self.batches_per_epoch  # this gives us the epoch number
                        print(f"Epoch {milestone}")

                        val_loss = self.compute_validation_loss()

                        # Log validation loss
                        wandb.log({'val_loss': val_loss, 'epoch': milestone})

                        # If validation loss is decreasing, save checkpoint Update the best validation loss and checkpoints
                        if val_loss < best_val_loss:
                            self.save(f"epoch-{milestone}")
                            best_val_loss = val_loss
                            self.update_best_checkpoints(val_loss, f"epoch-{milestone}")

                pbar.update(1)

        accelerator.print('training complete')

    def update_best_checkpoints(self, val_loss, milestone):
        # Adding the new checkpoint and sorting
        self.best_checkpoints.append((val_loss, str(self.results_folder / f'model-{milestone}.pt')))
        self.best_checkpoints.sort(key=lambda x: x[0])

        # Keeping only top 2 checkpoints
        if len(self.best_checkpoints) > 2:
            _, checkpoint_to_remove = self.best_checkpoints.pop(2)  # Remove the 4th checkpoint
            if os.path.exists(checkpoint_to_remove):
                os.remove(checkpoint_to_remove)  # Delete the checkpoint file

    def compute_validation_loss(self):
        self.model.eval()  # Set model to evaluation mode
        total_val_loss = 0.

        for val_seq, val_classes in self.val_dl:
            val_seq, val_classes = val_seq.to(self.device), val_classes.to(self.device)

            with torch.no_grad():
                val_loss = self.model(val_seq, classes=val_classes)
                total_val_loss += val_loss.item()

        average_val_loss = total_val_loss / len(self.val_dl)
        return average_val_loss