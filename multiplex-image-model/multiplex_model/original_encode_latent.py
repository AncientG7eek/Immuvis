#!/usr/bin/env python
# coding: utf-8


import os
import sys
from ruamel.yaml import YAML
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Compose, RandomRotation, Lambda, RandomCrop, Resize
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.functional import crop
from torchvision.ops import sigmoid_focal_loss

from torch.utils.data import DataLoader, Sampler, Dataset
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from functools import partial
from typing import Callable, Tuple, Type, Dict, List, Literal
from math import ceil

import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr

import neptune
from neptune.utils import stringify_unsupported
import matplotlib.pyplot as plt
from glob import glob
from cv2 import medianBlur
from skimage import filters
import tifffile
import gc

from sklearn.metrics import r2_score

# set seeds
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)


from data import DatasetFromTIFF, PanelBatchSampler, TestCrop, GridCrop
from losses import nll_loss
from utils import ClampWithGrad, plot_reconstructs_with_uncertainty, get_scheduler_with_warmup
from modules import MultiplexAutoencoder



device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {device}')

MODEL_NAME = 'ImmuVis-616-MSE-768-ma1'
SIZE = (128, 128)
BATCH_SIZE = 1
NUM_WORKERS = 4

PANEL_CONFIG = YAML().load(open('../configs/all_panels_config.yaml'))
TOKENIZER = YAML().load(open('../configs/all_markers_tokenizer.yaml'))
INV_TOKENIZER = {v: k for k, v in TOKENIZER.items()}

train_transform = GridCrop(SIZE[0])
test_transform = GridCrop(SIZE[0])

train_dataset = DatasetFromTIFF(
    panels_config=PANEL_CONFIG,
    split='train',
    marker_tokenizer=TOKENIZER,
    transform=train_transform,
    use_median_denoising=False,
    use_butterworth_filter=True,
    use_minmax_normalization=False,
    use_global_clip_limits=True,
    use_clip_normalization=True,
)

test_dataset = DatasetFromTIFF(
    panels_config=PANEL_CONFIG,
    split='test',
    marker_tokenizer=TOKENIZER,
    transform=test_transform,
    use_median_denoising=False,
    use_butterworth_filter=True,
    use_minmax_normalization=False,
    use_global_clip_limits=True,
    use_clip_normalization=True,
)

train_batch_sampler = PanelBatchSampler(train_dataset, BATCH_SIZE, shuffle=False)
test_batch_sampler = PanelBatchSampler(test_dataset, BATCH_SIZE, shuffle=False)

train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=NUM_WORKERS)
test_dataloader = DataLoader(test_dataset, batch_sampler=test_batch_sampler, num_workers=NUM_WORKERS)





model_config = {
    'num_channels': len(TOKENIZER),
    'superkernel_config': {
        'embedding_dim': 96,
        'num_layers': 0,
        'num_heads': None,
        'layer_type': 'linear',
        'kernel_size': None,
        'mlp_ratio': None
    },
    'encoder_config': {
        'layers_blocks': [3, 6, 9],
        'embedding_dims': [192, 384, 768]
    },
    'decoder_config': {
        'decoded_embed_dim': 512,
        'num_blocks': 1
    },
}

model = MultiplexAutoencoder(**model_config)
model.load_state_dict(torch.load('/raid_encrypted/immucan/models/' +MODEL_NAME+ '.pth', map_location=device)['model_state_dict'])
model.to(device)
model.eval()



def compute_embeddings(dataloader, model, device, split):

    
    latents = []
    image_names = []
    datasets = []
    coordinates = []


    rand_gen = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for crops, coords, channel_ids, dataset, img_path in tqdm(dataloader):

            for crop, coord in zip(crops,coords):
                crop = crop.to(device).to(torch.float32)

                output = model.encode_images(
                    x=crop, 
                    encoded_indices=channel_ids, 
                )['output']
                latents.append(output.cpu())
                image_names.append(img_path[0].split('/')[-1]) #.replace('.tiff', ''))
                coordinates.append(coord)
                datasets.append(dataset[0])
               

    latents = torch.stack(latents)     
    latents = latents.squeeze(1)
    latents = latents.mean(dim=(2,3)) # to average the whole crop by its patches

    save_dict = {
        'embeddings': latents,
        'image_name': image_names,
        'crop_coords': coordinates,
        'dataset': datasets
    }
    os.makedirs(os.path.expanduser('~/multiplex-image-model/expt'), exist_ok=True)
    output_file = os.path.expanduser(f'~/multiplex-image-model/expt/{MODEL_NAME}_{split}.pt')
    torch.save(save_dict, output_file)
    print(f'saved to: {output_file}')

    del latents
    gc.collect()
    
    if device == 'cuda':
        torch.cuda.empty_cache()
    


for split, dataloader in zip(['train', 'test'], [train_dataloader, test_dataloader]):
    compute_embeddings(dataloader, model, device, split)
    
    



