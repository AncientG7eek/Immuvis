import os
import sys

#import comet_ml  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ruamel.yaml import YAML
from torch.amp import GradScaler, autocast
from torch.nn.functional import normalize
from torch.utils.data import DataLoader

from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler, TestCrop, GridCrop
from multiplex_model.modules import MultiplexAutoencoder
from multiplex_model.utils import TrainingConfig
import gc

config_path = 'train_masked_config.yaml' #sys.argv[1]
yaml = YAML(typ="safe")
with open(config_path, "r") as f:
    raw_config = yaml.load(f)

# Validate configuration using Pydantic model
config = TrainingConfig(**raw_config)

device = config.device
print(f"Using device: {device}")

MODEL_NAME = 'ImmuVis-616-MSE-768-ma1'
SIZE = config.input_image_size
BATCH_SIZE = config.batch_size
NUM_WORKERS = config.num_workers

PANEL_CONFIG = YAML().load(open(config.panel_config))
TOKENIZER = YAML().load(open(config.tokenizer_config))
INV_TOKENIZER = {v: k for k, v in TOKENIZER.items()}

train_transform = GridCrop(SIZE[0])

test_transform = GridCrop(SIZE[0])

train_dataset = DatasetFromTIFF(
    panels_config=PANEL_CONFIG,
    split="train",
    marker_tokenizer=TOKENIZER,
    transform=train_transform,
    use_preprocessing=False,  # saved data is already preprocessed
    use_median_denoising=False,
    use_butterworth_filter=True,
    use_minmax_normalization=False,
    use_clip_normalization=True,
    file_extension="tiff",
)

test_dataset = DatasetFromTIFF(
    panels_config=PANEL_CONFIG,
    split="test",
    marker_tokenizer=TOKENIZER,
    transform=test_transform,
    use_preprocessing=False,  # saved data is already preprocessed
    use_median_denoising=False,
    use_butterworth_filter=True,
    use_minmax_normalization=False,
    use_clip_normalization=True,
    file_extension="tiff",
)

train_batch_sampler = PanelBatchSampler(train_dataset, BATCH_SIZE)
test_batch_sampler = PanelBatchSampler(test_dataset, BATCH_SIZE, shuffle=False)

def custom_collate(batch):
    """
    batch: list of tuples returned by DatasetFromTIFF.__getitem__:
      (crops: Tensor[n_crops, C, H, W],
       coords: np.ndarray[n_crops, ...] or list,
       channel_ids: Tensor[C],
       dataset: list_of_len_n_crops or list_of_len_n_crops,
       img_path: list_of_len_n_crops)
    Return:
      crops: Tensor[N_total_crops, C, H, W]
      coords: list of coords length N_total_crops
      channel_ids: Tensor[N_total_crops, C]
      datasets: list length N_total_crops
      img_paths: list length N_total_crops
    """
    all_crops = []
    all_coords = []
    all_channel_ids = []
    all_datasets = []
    all_img_paths = []

    for crops, coords, channel_ids, dataset, img_paths in batch:
        # crops is tensor (n_crops, C, H, W)
        n = crops.shape[0]
        all_crops.append(crops)

        # coords might be numpy array -> convert to list of tuples
        if isinstance(coords, np.ndarray):
            coords_list = [tuple(c) for c in coords.tolist()]
        else:
            coords_list = list(coords)
        all_coords.extend(coords_list)

        # repeat channel_ids per crop (channel_ids is 1D tensor of len C)
        all_channel_ids.extend([channel_ids] * n)

        # dataset and img_paths are lists of length n
        all_datasets.extend(list(dataset))
        all_img_paths.extend(list(img_paths))

    # concat crops along crop-dim
    crops = torch.cat(all_crops, dim=0)  # shape (N_total, C, H, W)

    # stack channel ids into shape (N_total, C)
    channel_ids = torch.stack(all_channel_ids, dim=0)  # (N_total, C)

    return crops, all_coords, channel_ids, all_datasets, all_img_paths

train_dataloader = DataLoader(
    train_dataset,
    batch_sampler=train_batch_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    collate_fn=custom_collate
)
test_dataloader = DataLoader(
    test_dataset,
    batch_sampler=test_batch_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    collate_fn=custom_collate
)

# Build model configuration
num_channels = len(TOKENIZER)
model = MultiplexAutoencoder(
    num_channels=num_channels,
    encoder_config=config.encoder_config.model_dump(),
    decoder_config=config.decoder_config.model_dump(),
)
ckpt = torch.load('models/' +MODEL_NAME+ '.pth',map_location=torch.device('cpu'))['model_state_dict']
#/raid_encrypted/immucan/
filtered_ckpt = {
    k: v for k, v in ckpt.items()
    if not k.startswith("decoder.pred")
}

model.load_state_dict(filtered_ckpt, strict=False)
model.to(device)
model.eval()

def save_chunk(latents, image_names, datasets, coordinates, split, crops_yielded, chunk_id):
    latents = torch.cat(latents)     
    latents = latents.squeeze(1)
    latents = latents.mean(dim=(2,3)) # to average the whole crop by its patches
    latents = latents.numpy()

    coords0,coords1 = zip(*coordinates)

    metadata = pd.DataFrame(
        {
        'image_path': image_names,
        'panel': datasets,
        'coords0': coords0,
        'coords1': coords1,
        }
    )
    os.makedirs(os.path.expanduser('~/Git/multiplex-image-model/expt'), exist_ok=True)
    latents_file = os.path.expanduser(f'~/Git/multiplex-image-model/expt/{MODEL_NAME}_{split}_image_patches_embeddings_{chunk_id}.npy')
    metadata_file = os.path.expanduser(f'~/Git/multiplex-image-model/expt/{MODEL_NAME}_{split}_image_patches_metadata_{chunk_id}.csv')
    np.save(latents_file, latents)
    metadata.to_csv(metadata_file, index=False)
    print(f'saved embeddings to: {latents_file}')
    print(f'saved metadata to {metadata_file}')

    del latents
    gc.collect()
        
    if device == 'cuda':
        torch.cuda.empty_cache()

def compute_embeddings(dataloader, model, device, split):

    
    latents = []
    image_paths = []
    datasets = []
    coordinates = []
    crops_yielded = 0
    chunk_id = 0

    with torch.no_grad():
        for crops, coords, channel_ids, dataset, img_paths in tqdm(dataloader):
            # flatten all crops from all samples
  
            print(coords)
            print(dataset)
            print(img_paths)

                # for crop, coord in zip(crops,coords):
            crops = crops.to(device).to(torch.float32)
            channel_ids = channel_ids.to(device)
            
            output = model.encode(
                x=crops, 
                encoded_indices=channel_ids, 
            )['output']
            latents.append(output.cpu())
            image_paths.extend(img_paths) #.replace('.tiff', ''))
            coordinates.extend(coords)
            datasets.extend(dataset)

            crops_yielded += len(crops)
                
            if crops_yielded > 5000:
                print(len(image_paths))
                print(len(datasets))
                print(len(coordinates))
                save_chunk(latents, image_paths, datasets, coordinates, split, crops_yielded, chunk_id)
                
                # reset
                latents, image_paths, datasets, coordinates = [], [], [], []
                crops_yielded = 0
                chunk_id += 1
    
    if len(latents) > 0:
        print(len(image_paths))
        print(len(datasets))
        print(len(coordinates))
        save_chunk(latents, image_paths, datasets, coordinates, split, crops_yielded, chunk_id)



for split, dataloader in zip(['test', 'train'], [test_dataloader, train_dataloader]):
    compute_embeddings(dataloader, model, device, split)