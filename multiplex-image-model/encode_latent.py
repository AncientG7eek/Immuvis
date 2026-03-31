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

from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler, TestCrop
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

train_transform = TestCrop(SIZE[0])

test_transform = TestCrop(SIZE[0])

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
    file_extension="npy",
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
    file_extension="npy",
)

train_batch_sampler = PanelBatchSampler(train_dataset, BATCH_SIZE)
test_batch_sampler = PanelBatchSampler(test_dataset, BATCH_SIZE, shuffle=False)

train_dataloader = DataLoader(
    train_dataset,
    batch_sampler=train_batch_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_sampler=test_batch_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
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

    metadata = pd.DataFrame(
        {
        'image_name': image_names,
        'dataset': datasets,
        'crop_coords': coordinates,
        }
    )
    os.makedirs(os.path.expanduser('~/multiplex-image-model/expt'), exist_ok=True)
    latents_file = os.path.expanduser(f'~/multiplex-image-model/expt/{MODEL_NAME}_{split}_image_patches_embeddings_{chunk_id}.npy')
    metadata_file = os.path.expanduser(f'~/multiplex-image-model/expt/{MODEL_NAME}_{split}_image_patches_metadata_{chunk_id}.csv')
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
    image_names = []
    datasets = []
    coordinates = []
    crops_yielded = 0
    chunk_id = 0

    with torch.no_grad():
        for crops, coords, channel_ids, dataset, img_paths in tqdm(dataloader):
            print(coords)
            print(dataset)
            print(img_paths)

                # for crop, coord in zip(crops,coords):
            crops = crops.to(device).to(torch.float32)

            output = model.encode(
                x=crops, 
                encoded_indices=channel_ids, 
            )['output']
            latents.append(output.cpu())
            image_names.extend([img_path.split('/')[-1] for img_path in img_paths]) #.replace('.tiff', ''))
            coordinates.extend([(x,y) for x,y in zip(coords[0].tolist(), coords[1].tolist())])
            datasets.extend(dataset)

            crops_yielded += len(crops)
                
            if crops_yielded > 5000:
                print(len(image_names))
                print(len(datasets))
                print(len(coordinates))
                save_chunk(latents, image_names, datasets, coordinates, split, crops_yielded, chunk_id)
                
                # reset
                latents, image_names, datasets, coordinates = [], [], [], []
                crops_yielded = 0
                chunk_id += 1
    
    if len(latents) > 0:
        print(len(image_names))
        print(len(datasets))
        print(len(coordinates))
        save_chunk(latents, image_names, datasets, coordinates, split, crops_yielded, chunk_id)



for split, dataloader in zip(['test', 'train'], [test_dataloader, train_dataloader]):
    compute_embeddings(dataloader, model, device, split)