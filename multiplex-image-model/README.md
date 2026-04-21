# ImmuVis — Clinical Fine-tuning Pipeline

Fine-tuning pipeline for the **ImmuVis** multiplex imaging backbone on clinical classification tasks (e.g. tumour grade, survival, receptor status). Loads a pretrained encoder, attaches a swappable classification head, and trains end-to-end on image-level clinical labels.

> **This repository is for fine-tuning, not pretraining.**
> The backbone (ImmuVis encoder) is trained separately. This code loads a checkpoint and attaches a task head.

---

## Architecture overview

```text
TIFF image  ──►  GridCrop  ──►  N crops (64×64, C channels)
                                      │
                          FinetuningModel.encode_crops()
                                      │
                        MultiplexImageEncoder (frozen or fine-tuned)
                        ├── Marker-agnostic ConvNeXt
                        ├── Hyperkernel (channel fusion)
                        └── Pan-marker ConvNeXt  ──►  (N, E) instance embeddings
                                      │
                              TaskHead  ◄──────── swappable
                        ┌─────────────┴─────────────────┐
                   CropClassifierHead             ABMILHead
                  (mean-pool crops → MLP)    (attention MIL → MLP)
                                      │
                              (1, num_classes) logits
                                      │
                          CrossEntropyLoss  ◄──  one label per image
```

One image = one bag. One label per bag. The head decides how crops are aggregated.

---

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.12 and PyTorch ≥ 2.7 with CUDA. All other dependencies are declared in `pyproject.toml`.

---

## Data preparation

Images must be multi-channel TIFF files (channels × H × W). Organise them as:

```text
/path/to/data/
├── train/
│   └── danenberg/
│       └── imgs/
│           ├── image_001.tiff
│           └── ...
└── test/
    └── danenberg/
        └── imgs/
            ├── image_042.tiff
            └── ...
```

`train/` and `test/` are the splits. Each subdirectory under a split is one **dataset/panel** (images sharing the same set of markers). Panel names must match the keys used in the panel config.

### Panel config (`configs/all_panels_config.yaml`)

```yaml
paths:
  server:
    train: /path/to/data/train
    test:  /path/to/data/test
  local:
    train: /local/path/train
    test:  /local/path/test

datasets:
  - danenberg
  - otherpanel

markers:
  danenberg:
    - CD3
    - CD8
    - FOXP3
    # ... one entry per channel, in channel order

clip_limits:
  danenberg: 5.0   # upper bound for intensity clipping normalisation
```

### Tokenizer config (`configs/all_markers_tokenizer.yaml`)

Maps each marker name to a unique integer index (shared vocabulary across all panels):

```yaml
CD3:   0
CD8:   1
FOXP3: 2
# ...
```

### Clinical labels (`melted_table.csv`)

A long-format CSV with one row per (image, feature) pair:

| img_name | feature | value | dataset |
| --- | --- | --- | --- |
| image_001 | Grade | Grade2 | danenberg |
| image_001 | ER_status | Positive | danenberg |
| image_042 | Grade | Grade3 | danenberg |

`img_name` must match the TIFF filename stem (no extension). The path is hardcoded at:

```python
MELTED_TABLE_PATH = "../melted_table/results/melted_table.csv"
```

Edit line 270 of `train_masked_model.py` if your path differs.

---

## Configuration

All training options live in `train_masked_config.yaml`. Annotated reference:

```yaml
# ── Model architecture ──────────────────────────────────────────────────────
encoder:
  ma_layers_blocks: [4]           # blocks in marker-agnostic ConvNeXt stage
  ma_embedding_dims: [16]         # channels per stage
  pm_layers_blocks: [4, 4, 4]    # blocks in pan-marker stages
  pm_embedding_dims: [192, 384, 768]  # channels per stage; last = latent dim E
  hyperkernel:
    kernel_size: 1
    padding: 0
    stride: 1
    use_bias: true

classifier:
  hidden_dims: [512, 128]         # MLP layers for CropClassifierHead

# ── Head selection ───────────────────────────────────────────────────────────
head_type: logistic               # "logistic" or "abmil"

abmil:                            # only used when head_type: abmil
  hidden_dim: 128                 # attention network width
  gated: true                     # gated attention (recommended)
  dropout: 0.0                    # instance dropout (try 0.25 for small bags)
  classifier_hidden_dims: []      # [] = single linear; [256] = one hidden layer

# ── Data ─────────────────────────────────────────────────────────────────────
dataset_subsets:
  - [danenberg, Grade]            # [panel_name, clinical_feature_to_predict]

panel_config:     configs/all_panels_config.yaml
tokenizer_config: configs/all_markers_tokenizer.yaml
input_image_size: [64, 64]        # crop size
max_crops_per_image: 64           # cap crops per image (OOM guard for large TIFFs)
num_workers: 8
batch_size: 1                     # images per DataLoader iteration

# ── Training ─────────────────────────────────────────────────────────────────
device: cuda
lr: 5e-4
final_lr: 1e-5
weight_decay: 0.0001
gradient_accumulation_steps: 4   # effective batch = 4 images
epochs: 10
frac_warmup_steps: 0.1

# ── Checkpoint ───────────────────────────────────────────────────────────────
from_checkpoint: /path/to/pretrained/ImmuVis.pth   # encoder weights to load
checkpoints_dir: models_finetuning/
save_checkpoint_freq: 5

# ── Logging (Comet.ml) ───────────────────────────────────────────────────────
comet_project: immuvis-finetuning
comet_workspace: null             # or set COMET_WORKSPACE env var
comet_api_key: null               # or set COMET_API_KEY env var
tags: [first-run]
```

---

## Running

```bash
python train_masked_model.py train_masked_config.yaml
```

Training logs one prediction per image (not per crop) to Comet.ml under `val/loss` and `val/macroF1`. Checkpoints are saved to `checkpoints_dir` every `save_checkpoint_freq` epochs, plus a `last_checkpoint-*.pth` after every epoch.

---

## Switching classification heads

Edit one line in the YAML:

### Logistic (mean-pool + MLP)

```yaml
head_type: logistic
```

All crops from one image are encoded → embeddings mean-pooled → MLP classifies. Fast baseline. Configured by the `classifier.hidden_dims` field.

### ABMIL (Attention-Based MIL)

```yaml
head_type: abmil
abmil:
  hidden_dim: 128
  gated: true
  dropout: 0.0
  classifier_hidden_dims: []
```

Each crop gets a learned attention weight; bag embedding is the weighted sum of instance embeddings. One prediction per image. Attention weights can be extracted for spatial interpretability (see below).

#### Extracting attention maps at inference

```python
from multiplex_model.modules.abmil import ABMILHead

model.eval()
with torch.no_grad():
    crops = crops.to(device, dtype=torch.float32)
    instance_emb = model.encode_crops(crops, channel_ids)          # (N, E)
    logits, weights = model.head.forward_with_attention(instance_emb)
    # weights: (N, 1) — pair with GridCrop coordinates to make a heatmap
```

---

## Adding a custom head

All heads implement the `TaskHead` interface from `multiplex_model/modules/immuvis.py`:

```python
from multiplex_model.modules.immuvis import TaskHead
import torch, torch.nn as nn

class MyHead(TaskHead):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, instance_embeddings: torch.Tensor) -> torch.Tensor:
        # instance_embeddings: (N_crops, E) — all crops from one image
        # must return: (1, num_classes)
        pooled = instance_embeddings.mean(dim=0)
        return self.linear(pooled).unsqueeze(0)
```

Then wire it into `__main__` in `train_masked_model.py` alongside the existing `if config.head_type` branch.

---

## Testing ABMIL

Run the built-in smoke test (no GPU required, no data needed):

```bash
python -m multiplex_model.modules.abmil
```

Checks: output shapes, attention weight normalisation, gradient flow, gated vs standard, dropout in eval mode, variable bag sizes, `forward_with_attention` consistency.

---

## Project structure

```text
multiplex-image-model/
├── train_masked_model.py          # training entry point
├── train_masked_config.yaml       # all config options
├── encode_latent.py               # feature extraction (inference only)
├── configs/
│   ├── all_panels_config.yaml     # dataset paths and marker lists
│   └── all_markers_tokenizer.yaml # marker → token index mapping
└── multiplex_model/
    ├── data.py                    # DatasetFromTIFF, GridCrop, PanelBatchSampler
    ├── clinical.py                # label lookup, LabelEncoder
    └── modules/
        ├── immuvis.py             # MultiplexImageEncoder, FinetuningModel,
        │                          #   TaskHead, CropClassifierHead
        ├── abmil.py               # ABMILHead (attention MIL)
        ├── convnext.py            # ConvNeXt blocks and encoder
        ├── vit.py                 # ViT encoder (alternative backbone)
        └── utils/
            └── configuration.py   # FinetuningConfig, ABMILHeadConfig
```
