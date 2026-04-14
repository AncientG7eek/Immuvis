# Finetuning Script Guide: Adapting from Autoencoder to Classifier

## Overview
Your script has been updated to convert from **autoencoder training** (reconstruction task) to **finetuning** (classification task). Here's what changed and how to use it properly.

---

## 1. Key Changes Made

### 1.1 Loss Function Fix
**Problem:** You were using `NLLLoss` but passing raw predictions instead of log-probabilities.

**Solution:** Changed to `CrossEntropyLoss`
```python
# BEFORE (Wrong)
loss_fn = torch.nn.NLLLoss()
preds = torch.argmax(logits, dim=1)
loss = loss_fn(preds, y)  # ❌ NLLLoss expects log-probs, not predictions

# AFTER (Correct)
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(logits, y)  # ✅ Pass raw logits, loss handles softmax internally
preds = torch.argmax(logits.detach(), dim=1)
```

**Why:** 
- `CrossEntropyLoss` = `LogSoftmax` + `NLLLoss` in one operation
- More numerically stable than manual softmax
- Automatically computes log-probabilities

---

### 1.2 Label Loading from Melted Table
**How it works:**
```python
# Get all rows matching this image path
all_cli_features = get_a_subset(melted_table, "img_path", img_path)

# Filter to the feature you care about
selected_cli_feat = get_a_subset(all_cli_features, "feature", cli_feat_for_subset)

# Convert to tensor
y = torch.tensor(selected_cli_feat["value"].values, device=device, dtype=torch.long)
```

**Expected melted_table format:**
```
| img_path           | feature         | value | dataset |
|-------------------|-----------------|-------|---------|
| /path/to/img1.tif | marker1         | 0     | train   |
| /path/to/img1.tif | marker2         | 1     | train   |
| /path/to/img2.tif | marker1         | 2     | val     |
```

---

### 1.3 Autocast & GradScaler Explanation

#### **What is Autocast?**
Enables **mixed precision training** (FP32 + FP16) for faster computation.

```python
with autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = model(x=img, encoded_indices=channel_ids)
    loss = loss_fn(logits, y)
```

- Forward pass uses lower precision (bfloat16) → **faster, less memory**
- Loss computation stays in FP32 → **numerical stability**
- Only used during training (not testing)

#### **What is GradScaler?**
Prevents **gradient underflow** when using mixed precision.

```python
scaler = GradScaler()  # Initialize once before training loop

# Inside training loop:
scaler.scale(loss / gradient_accumulation_steps).backward()  # Scale loss before backward

if (batch_idx + 1) % gradient_accumulation_steps == 0:
    scaler.unscale_(optimizer)  # Unscale before clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)  # Scaled step
    scaler.update()  # Update scale for next iteration
    optimizer.zero_grad()
    scheduler.step()
```

**Why this order matters:**
1. `scale()` → multiply loss by large factor (e.g., 65536)
2. `backward()` → compute gradients (won't underflow)
3. `unscale_()` → divide gradients back to normal
4. `clip_grad_norm_()` → clip for stability
5. `step()` → update weights
6. `update()` → adjust scale for next batch

#### **Testing with Autocast**
In your test function, **remove autocast**:
```python
# REMOVE THIS in test_masked():
# with autocast(device_type="cuda", dtype=torch.bfloat16):
#     logits = model(x=img, encoded_indices=channel_ids)

# INSTEAD, use this (no autocast needed in eval mode):
with torch.no_grad():
    logits = model(x=img, encoded_indices=channel_ids)
    loss = loss_fn(logits, y)
```

**Why?** 
- No gradients needed in evaluation (torch.no_grad() is enough)
- Autocast adds overhead when not computing backward()
- Mixed precision is only beneficial during training

---

## 2. Testing Function Changes

### Before (Autoencoder)
```python
def test_masked(model, test_dataloader, device, epoch, marker_names_map, ...):
    # Computed reconstruction loss
    loss = nll_loss(img, mi, logvar)
    # Plotted reconstructed images
    plot_reconstructs_with_masks(...)
```

### After (Classifier)
```python
def test_masked(model, test_dataloader, device, epoch, marker_names_map,
                melted_table, cli_feat_for_subset, classes, step, ...):
    # Compute classification loss
    logits = model(x=img, encoded_indices=channel_ids)
    loss = loss_fn(logits, y)
    preds = torch.argmax(logits, dim=1)
    
    # Compute metrics
    metrics = log_finetuning_validation_metrics(
        loss=val_loss,
        preds=all_preds,
        y=all_y,
        classes=classes,
        step=step,
    )
```

**New parameters required:**
- `melted_table` - for label loading
- `cli_feat_for_subset` - which feature to extract
- `classes` - list of class labels
- `step` - training step for logging

---

## 3. Data Loading from Melted Table

### Structure your melted table like this:

```python
import pandas as pd

melted_table = pd.read_csv("path/to/melted_data.csv")
# Columns: img_path, feature, value, dataset, ...
```

### Example:
```python
# Create from multiple sources
melted_list = []
for img_path, features_dict in dataset.items():
    for feature_name, feature_value in features_dict.items():
        melted_list.append({
            "img_path": img_path,
            "feature": feature_name,
            "value": feature_value,  # Should be integer class label
            "dataset": "train"  # or "test" or "val"
        })

melted_table = pd.DataFrame(melted_list)
```

### Getting labels during training:
```python
# For a batch of images:
all_cli_features = get_a_subset(melted_table, "img_path", img_path)
selected_cli_feat = get_a_subset(all_cli_features, "feature", cli_feat_for_subset)
y = torch.tensor(selected_cli_feat["value"].values, device=device, dtype=torch.long)

# If img_path is a list/batch, get_a_subset should handle it
# If not, loop through batch:
y_list = []
for path in img_path:
    features = get_a_subset(melted_table, "img_path", [path])
    feat = get_a_subset(features, "feature", cli_feat_for_subset)
    y_list.append(feat["value"].values[0])
y = torch.tensor(y_list, device=device, dtype=torch.long)
```

---

## 4. Loading Pretrained Encoder Weights

The script now automatically loads encoder weights:

```python
# Load checkpoint if specified
if config.resolve_checkpoint():
    print(f"Loading encoder weights from: {config.from_checkpoint}")
    checkpoint = torch.load(config.from_checkpoint, map_location=device)
    encoder_state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # Filter to only encoder weights
    encoder_keys = {k: v for k, v in encoder_state_dict.items() 
                   if k.startswith("encoder.")}
    if encoder_keys:
        model.encoder.load_state_dict(encoder_keys)
```

### To freeze encoder weights (don't fine-tune):
```python
# After loading encoder
for param in model.encoder.parameters():
    param.requires_grad = False

# This way, only classifier weights will be updated
optimizer = optim.AdamW(model.classifier.parameters(), lr=config.peak_lr)
```

### To fine-tune encoder with lower learning rate:
```python
# Use parameter groups
optimizer = optim.AdamW([
    {'params': model.encoder.parameters(), 'lr': config.peak_lr * 0.1},  # 10x lower
    {'params': model.classifier.parameters(), 'lr': config.peak_lr}
], weight_decay=config.weight_decay)
```

---

## 5. Metrics and Logging

### Training Loop:
```python
# After each epoch
metrics = log_finetuning_validation_metrics(
    loss=loss.item(),
    preds=all_preds,
    y=all_y,
    classes=classes,
    step=step,
)
```

Expects:
- `loss`: scalar float
- `preds`: numpy array (N,) of predicted class indices
- `y`: numpy array (N,) of true class indices  
- `classes`: list of unique class labels
- `step`: training step for logging

### Testing Loop:
Same metrics computation, returns dict with keys like:
- `val/loss`
- `val/macroF1`
- `val/accuracy`
- etc.

---

## 6. Configuration File Requirements

Your config YAML should include:

```yaml
# Dataset
input_image_size: [224, 224]
batch_size: 32
num_workers: 8

# Model architecture
encoder_config:
  pm_embedding_dims: [64, 128, 256]
  ma_layers_blocks: [2, 2, 3]
  ma_embedding_dims: [32, 64, 128]
  # ... other encoder params

classifier_config:
  hidden_dims: [256, 128]  # Hidden layer sizes
  # Will auto-calculate: input_dim (from encoder) → 256 → 128 → num_classes

# Training
peak_lr: 1e-3
final_lr: 1e-5
weight_decay: 1e-4
frac_warmup_steps: 0.1
gradient_accumulation_steps: 4
epochs: 50

# Checkpoints
from_checkpoint: "path/to/encoder_weights.pth"
checkpoints_dir: "./checkpoints"

# Datasets and features
dataset_subsets:
  train: ["feature1", "feature2"]  # Features per subset
  val: ["feature1"]
```

---

## 7. Common Issues & Fixes

### Issue: `y` has wrong shape
```
RuntimeError: Expected target size (32,), got (32, 1)
```

**Fix:** Ensure y is 1D:
```python
y = torch.tensor(selected_cli_feat["value"].values, dtype=torch.long)  # shape: (batch,)
# NOT
y = selected_cli_feat["value"].values.reshape(-1, 1)  # shape: (batch, 1) ❌
```

---

### Issue: Classifier input size mismatch
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied
```

**Fix:** Ensure encoder output is flattened. Check `Finetuning.forward()`:
```python
def forward(self, x, encoded_indices):
    emb = self.encode(x, encoded_indices)
    # emb is dict: {"output": tensor}
    # Need to extract and possibly flatten:
    output_tensor = emb["output"]  # shape: (B, latent_dim, 1, 1) or (B, latent_dim)
    if output_tensor.dim() > 2:
        output_tensor = output_tensor.flatten(1)  # Flatten spatial dims
    pred = self.classifier(output_tensor)
    return pred
```

---

### Issue: `get_a_subset()` doesn't handle batches
```
ValueError: index is out of bounds for axis 0
```

**Fix:** Make sure your `get_a_subset()` function works with lists:
```python
# Your implementation should handle:
get_a_subset(melted_table, "img_path", ["path1.tif", "path2.tif"])

# Or loop through batch:
for path in img_path:
    get_a_subset(melted_table, "img_path", path)
```

---

## 8. Training Checklist

- [ ] Config file created with encoder/classifier/training parameters
- [ ] Melted table CSV prepared with img_path, feature, value, dataset columns
- [ ] Pretrained encoder checkpoint path specified (or from_checkpoint=None)
- [ ] DataLoader batch sizes match config
- [ ] Classes extracted correctly from melted_table
- [ ] Model instantiated with correct num_classes
- [ ] Loss function set to CrossEntropyLoss
- [ ] Autocast used only during training (not testing)
- [ ] GradScaler initialized and used correctly
- [ ] Encoder weights loaded before training
- [ ] Metrics logged to Comet.ml (or disabled)

---

## 9. Running the Script

```bash
python train_masked_model.py config.yaml

# With GPU
CUDA_VISIBLE_DEVICES=0 python train_masked_model.py config.yaml

# Multiple GPUs with DDP (if supported)
torchrun --nproc_per_node=2 train_masked_model.py config.yaml
```

---

## 10. Next Steps

1. **Test on small data**: Run with 1 epoch on 100 samples to check for crashes
2. **Monitor metrics**: Watch loss decrease and F1 increase
3. **Adjust hyperparameters**: LR, batch size, warmup steps
4. **Validate on held-out set**: Check generalization
5. **Ensemble predictions**: Combine multiple checkpoints for robustness

---

**Questions?** Check the fixes I made in the script and compare with this guide.
