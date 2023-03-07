from diffusers import DDIMScheduler, DDIMPipeline
from diffusers.models import UNet2DModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize
from torchvision.datasets import MNIST

from accelerate import Accelerator
import wandb
from tqdm import tqdm
from PIL.Image import open
import matplotlib.pyplot as plt
from torchinfo import summary

num_epoch = 10
batch_size = 128
lr = 2e-4
mixed_precision = 'fp16'
gradient_accumulation = 1


class Normalize_img():
    def __call__(self, img):
        return 2 * img - 1


transform = Compose(
    [
        ToTensor(),
        Normalize_img(),
    ]
)

sample = transform(open("../cat-backpack.png"))

unet = UNet2DModel(
    sample_size=28,
    in_channels=1,
    out_channels=1,
    center_input_sample=False,
    time_embedding_type="positional",
    down_block_types=(
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D"
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D"
    ),
    block_out_channels=(64, 128, 256),
    layers_per_block=2,
    act_fn='silu',
    attention_head_dim=8,
    resnet_time_scale_shift="scale_shift",
)

ddim = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=1e-4,
    beta_end=2e-2,
    beta_schedule='squaredcos_cap_v2',
    prediction_type='epsilon'
)

mnist = MNIST(root="datasett", train=True, transform=transform, download=True)
train_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True,)

optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)

accelerator = Accelerator(mixed_precision=mixed_precision,
                          gradient_accumulation_steps=gradient_accumulation
                          )

unet, optimizer, train_loader = accelerator.prepare(
    unet, optimizer, train_loader
)

run = wandb.init(
    project='ddpm_mnist',
    entity="ima9ine",
    dir="wandb_"
)


def denormalize_img(img):
    return (img + 1) / 2


def train():
    global_step = 0
    for epoch in range(num_epoch):
        pbar = tqdm(total=len(train_loader))
        pbar.set_description(f"EPOCH {epoch}")
        for i, x in enumerate(train_loader):

            clean_img = x[0]
            device = clean_img.device
            bsz = clean_img.size(0)

            # timestep
            timestep = torch.randint(0, 1000, size=(bsz,), device=device)

            # noise
            noise = torch.randn_like(clean_img)

            # noisy image
            noisy_image = ddim.add_noise(clean_img, noise, timestep)

            model_output = unet(noisy_image, timestep).sample

            loss = F.mse_loss(model_output, noise)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            pbar.update(1)

            logs = {'loss': loss.detach().item()}
            pbar.set_postfix(**logs)
            wandb.log(logs, step=global_step)
            global_step += 1


        pipe = DDIMPipeline(unet=unet, scheduler=ddim,)
        generator = torch.Generator(device=pipe.device)

        images = pipe(
            generator=generator,
            batch_size=10,
            num_inference_steps=50,
            output_type='pil'
        ).images

        wandb.log({"img" : [wandb.Image(img) for img in images]})

train()
run.finish()