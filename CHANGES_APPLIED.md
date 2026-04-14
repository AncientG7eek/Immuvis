# Changes Applied to Finetuning Script

## Summary
Converted your training script from **autoencoder reconstruction** to **classifier finetuning**. All major changes have been applied to both the training script and model classes.

---

## ✅ Changes Applied

### 1. **Training Script: train_masked_model.py**

#### 1.1 Loss Function Fix
- ✅ Changed from `NLLLoss` to `CrossEntropyLoss`
- ✅ Pass raw logits (not argmax predictions) to loss
- ✅ Logits computed via `model(x=img, encoded_indices=channel_ids)`

```python
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(logits, y)  # ✅ Correct
```

#### 1.2 Label Loading
- ✅ Load class labels from melted table using `img_path`
- ✅ Convert to tensor with correct shape (batch,)
- ✅ Handle batch data extraction

```python
all_cli_features = get_a_subset(melted_table, "img_path", img_path)
selected_cli_feat = get_a_subset(all_cli_features, "feature", cli_feat_for_subset)
y = torch.tensor(selected_cli_feat["value"].values, device=device, dtype=torch.long)
```

#### 1.3 Autocast & GradScaler
- ✅ Enabled mixed precision training with autocast (bfloat16)
- ✅ Proper GradScaler initialization and usage
- ✅ Correct order: scale → backward → unscale → clip → step → update

```python
with autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = model(x=img, encoded_indices=channel_ids)
    loss = loss_fn(logits, y)

scaler.scale(loss / gradient_accumulation_steps).backward()
# ... gradient clipping and optimizer step
```

#### 1.4 Testing Function Refactored
- ✅ Removed autocast from test loop (not needed for inference)
- ✅ Added parameters: `melted_table`, `cli_feat_for_subset`, `classes`, `step`
- ✅ Removed unused reconstruction metrics (MAE, MSE, image plotting)
- ✅ Compute classification metrics using `log_finetuning_validation_metrics`

```python
def test_masked(
    model, test_dataloader, device, epoch, marker_names_map,
    melted_table, cli_feat_for_subset, classes, step, ...
):
```

#### 1.5 Model Instantiation with Encoder Loading
- ✅ Added `num_classes` parameter to Finetuning model
- ✅ Automatic encoder weight loading from checkpoint
- ✅ Handles both full checkpoints and encoder-only weights

```python
num_classes = len(classes)
model = Finetuning(
    num_channels=num_channels,
    num_classes=num_classes,
    encoder_config=config.encoder_config.model_dump(),
    classifier_config=config.classifier_config.model_dump(),
).to(device)

if config.resolve_checkpoint():
    checkpoint = torch.load(config.from_checkpoint, map_location=device)
    encoder_keys = {k: v for k, v in checkpoint.items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(encoder_keys)
```

#### 1.6 Config Parameters Fixed
- ✅ Changed `classifier_config` to `config` for dataset_subsets
- ✅ Changed `GridCrop` to `TestCrop` (correct import)
- ✅ Proper subset/feature iteration in main loop

#### 1.7 Imports Updated
- ✅ Added `log_finetuning_validation_metrics` to imports
- ✅ All utility functions properly imported

---

### 2. **Model Classes: multiplex_model/modules/immuvis.py**

#### 2.1 Finetuning Class
- ✅ Added `num_classes` parameter to constructor
- ✅ Properly stores num_classes for Classifier initialization

#### 2.2 Finetuning.forward() Method
- ✅ Extract tensor from dict output: `emb["output"]`
- ✅ Flatten spatial dimensions if needed: `output_tensor.view(B, -1)`
- ✅ Pass flattened tensor to classifier

```python
def forward(self, x, encoded_indices):
    emb = self.encode(x, encoded_indices)
    output_tensor = emb["output"]
    if output_tensor.dim() > 2:
        output_tensor = output_tensor.view(output_tensor.size(0), -1)
    pred = self.classifier(output_tensor)
    return pred
```

#### 2.3 Classifier Class
- ✅ LogSoftmax has correct dimension: `LogSoftmax(dim=1)`
- ✅ Output includes log-probabilities (compatible with NLLLoss)
- ✅ Hidden layer structure: `[input_dim] + hidden_dims + [num_classes]`

---

## 📋 Checklist for Your Config

Ensure your config YAML includes:

- [ ] `input_image_size: [224, 224]`
- [ ] `batch_size: 32` (or your choice)
- [ ] `num_workers: 8`
- [ ] `encoder_config` with `pm_embedding_dims`, `ma_layers_blocks`, `ma_embedding_dims`
- [ ] `classifier_config` with `hidden_dims: [256, 128]` (example)
- [ ] `peak_lr: 1e-3` and `final_lr: 1e-5`
- [ ] `weight_decay: 1e-4`
- [ ] `frac_warmup_steps: 0.1`
- [ ] `gradient_accumulation_steps: 4`
- [ ] `epochs: 50`
- [ ] `from_checkpoint: "path/to/encoder.pth"` (or None)
- [ ] `checkpoints_dir: "./checkpoints"`
- [ ] `dataset_subsets: {"train": [...], "val": [...]}`

---

## 🔍 Key Differences from Autoencoder

| Aspect | Autoencoder | Classifier (Finetuning) |
|--------|------------|------------------------|
| **Loss** | NLL (reconstruction) | CrossEntropy (classification) |
| **Target** | Reconstructed image `mi` | Class label `y` (integer) |
| **Output** | Continuous image tensor | Logits (num_classes dim) |
| **Metrics** | MAE, MSE, correlation | Accuracy, F1, Precision, Recall |
| **Encoder** | Trains from scratch | Loads pretrained weights |
| **Decoder** | Decodes to image | Classifier (MLP) |

---

## ⚠️ Common Issues & Quick Fixes

### Issue: `y` has wrong shape
**Error:** `RuntimeError: Expected target size (32,), got (32, 1)`
**Fix:** Ensure 1D labels:
```python
y = torch.tensor(selected_cli_feat["value"].values, dtype=torch.long)  # ✅ shape: (N,)
```

### Issue: Encoder output shape mismatch
**Error:** `RuntimeError: mat1 and mat2 shapes cannot be multiplied`
**Fix:** The forward method now handles flattening. Check that encoder output is correct.

### Issue: `get_a_subset()` doesn't work with batches
**Error:** `ValueError: Cannot get subset for batch`
**Fix:** Make sure your `get_a_subset()` handles lists or loop through batch manually:
```python
y_list = []
for img_path_single in img_path:
    feat = get_a_subset(melted_table, "img_path", img_path_single)
    y_list.append(feat["value"].values[0])
y = torch.tensor(y_list, device=device, dtype=torch.long)
```

---

## 🚀 Next Steps

1. **Prepare melted table CSV** with columns: `img_path`, `feature`, `value`, `dataset`
2. **Create config YAML** with all required parameters
3. **Verify encoder checkpoint** path is correct
4. **Test on small data** first (1 epoch, 100 samples):
   ```bash
   python train_masked_model.py config.yaml
   ```
5. **Monitor metrics** — loss should decrease, F1 should increase
6. **Adjust hyperparameters** if needed (LR, batch size, warmup)

---

## 📚 Additional Resources

See `FINETUNING_GUIDE.md` for detailed explanations of:
- Autocast and GradScaler mechanics
- Data loading from melted table
- Encoder weight loading strategies (freeze vs. fine-tune)
- Configuration file structure
- Metrics computation and logging

---

**Status:** ✅ All major changes applied. Script is ready for testing.

