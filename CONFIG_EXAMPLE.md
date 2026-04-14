# Example Config for Finetuning

```yaml
# ============================================================================
# TRAINING CONFIGURATION FOR FINETUNING
# ============================================================================

# Dataset and DataLoader
input_image_size: [256, 256]  # Height, width for RandomCrop
batch_size: 16
num_workers: 4

# Model Architecture
encoder_config:
  # Pan-Marker (per-marker aggregation)
  pm_layers_blocks: [3, 4, 6]
  pm_embedding_dims: [96, 192, 384]
  
  # Marker-Agnostic layers
  ma_layers_blocks: [2, 2, 2]
  ma_embedding_dims: [64, 128, 256]
  
  # Hyperkernel config
  hyperkernel_config:
    embedding_dim: 384
    num_factors: 64
    
  use_latent_norm: true
  encoder_type: convnext

classifier_config:
  hidden_dims: [512, 256]  # Hidden layers before output
  # Final structure will be: latent_dim -> 512 -> 256 -> num_classes

# Training Optimization
peak_lr: 1.0e-3
final_lr: 1.0e-5
weight_decay: 1.0e-4
frac_warmup_steps: 0.1
gradient_accumulation_steps: 2
epochs: 100

# Checkpointing
from_checkpoint: "./checkpoints/pretrained_encoder.pth"  # Encoder weights
checkpoints_dir: "./checkpoints/finetuning"
save_checkpoint_freq: 5  # Save every N epochs

# Dataset subsets and features
dataset_subsets:
  train: ["feature_name"]  # Feature to classify on
  val: ["feature_name"]

# Paths
panel_config: "configs/panel_config.yaml"
tokenizer_config: "configs/tokenizer.yaml"

# Logging
device: cuda:0
```

---

## Configuration Tips

### Encoder Config
The `encoder_config` should **match** the encoder used in pretraining:
- Same `pm_embedding_dims` and `ma_embedding_dims`
- Same `hyperkernel_config`
- Same `encoder_type` (usually "convnext")

Get these values from your **pretrained encoder checkpoint** or training logs.

### Classifier Hidden Dims
- Smaller networks (e.g., `[256, 128]`) for small datasets
- Larger networks (e.g., `[1024, 512, 256]`) for large datasets
- First dim often 2-4x the latent dim

### Learning Rates
For **finetuning** (don't change encoder much):
- `peak_lr: 1e-3` (if encoder frozen)
- `peak_lr: 1e-4` (if encoder trainable)

For **warm fine-tuning** (gradually unfreeze):
- `peak_lr: 1e-4`
- Later, unfreeze encoder and reduce LR further

### Batch Size
Adjust based on GPU memory:
- 12GB GPU: batch_size 16-32
- 24GB GPU: batch_size 32-64
- 40GB GPU: batch_size 64-128

---

## Minimal Config (for quick testing)

```yaml
input_image_size: [224, 224]
batch_size: 8
num_workers: 2

encoder_config:
  pm_layers_blocks: [2]
  pm_embedding_dims: [64]
  ma_layers_blocks: [1]
  ma_embedding_dims: [32]
  hyperkernel_config: { embedding_dim: 64, num_factors: 32 }
  encoder_type: convnext

classifier_config:
  hidden_dims: [128]

peak_lr: 1.0e-3
final_lr: 1.0e-5
weight_decay: 1.0e-4
frac_warmup_steps: 0.05
gradient_accumulation_steps: 1
epochs: 10

from_checkpoint: null  # No pretrained weights
checkpoints_dir: "./checkpoints_test"
save_checkpoint_freq: 5

dataset_subsets:
  train: ["feature1"]
  val: ["feature1"]

panel_config: "configs/panel_config.yaml"
tokenizer_config: "configs/tokenizer.yaml"
device: cuda:0
```

---

## Using the Config

```bash
# Run training
python train_masked_model.py config.yaml

# With environment variables
CUDA_VISIBLE_DEVICES=0 python train_masked_model.py config.yaml
```

