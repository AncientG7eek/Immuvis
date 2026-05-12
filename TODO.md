# Code Review: Fine-tuning Pipeline — Issues & TODOs

This document captures all findings from the code review of the `explain` branch, focused on
diagnosing the reported macro-F1 drop (from ~0.8 foundation-model baseline down to ~0.2 after
fine-tuning). Issues are grouped by severity and category.

---

## CRITICAL BUGS — Almost Certainly Causing F1 Collapse

### 1. Early stopping counter is never reset on improvement
**File:** `multiplex-image-model/finetuning.py` — `train_masked()`, ~line 322

**Code:**
```python
if top_macro_f1 >= current_macro_f1:
    no_improvement += 1
    if no_improvement > 7:
        return best_checkpoint_path
else:
    top_macro_f1 = current_macro_f1
    # BUG: no_improvement is NOT reset to 0 here
```

**Problem:** `no_improvement` is only ever incremented; it is never reset to 0 when the model
actually improves. This means early stopping fires after 7 *total* (not consecutive) non-improving
epochs across the entire training run. For example, with 20 epochs and oscillating F1:
`0.5 → 0.6 → 0.5 → 0.7 → 0.5 → 0.6 → 0.5 → 0.6 → 0.5` — training stops after epoch 8
(7 non-improving epochs total) even though the model keeps improving in between.
This causes training to abort very early.

**Fix:**
```python
else:
    top_macro_f1 = current_macro_f1
    no_improvement = 0   # reset counter on every genuine improvement
```

---

### 2. Validation metrics return `None` when Comet.ml is not configured — best checkpoint never saved
**File:** `multiplex-image-model/multiplex_model/utils/train_logging.py` — `log_finetuning_validation_metrics()`, ~line 826

**Code:**
```python
def log_finetuning_validation_metrics(...):
    if _experiment is None:
        return          # returns None, not a metrics dict
    ...
    return metrics      # only reached if Comet is active
```

**Problem:** The function conflates metric computation with Comet logging. When `_experiment is None`
(Comet not configured, or `comet_api_key: null` in YAML), the function returns `None`.
Back in `train_masked()`:

```python
val_metrics = test_masked(...)              # returns None
if not val_metrics or "val/macroF1" not in val_metrics:
    print("Validation metrics missing macro-F1; skipping best checkpoint update.")
```

This guard silently passes every single epoch → `top_macro_f1` stays at 0.0 → `best_checkpoint_path`
stays `None` → at the end of training, `model.load_state_dict(ckpt["model_state_dict"])` is
never called → the final test evaluation uses whatever state the model is in at the last epoch
(likely overfit), not the best checkpoint. Same bug applies to `log_finetuning_training_metrics`:
if it returns `None`, line `train_metrics['train/macroF1']` will throw `TypeError`.

**Fix:** Separate metric computation from Comet logging:
```python
def _compute_finetuning_metrics(preds, y, num_classes):
    """Pure computation — no Comet dependency."""
    confusion_matrix = ...
    ...
    return {"val/macroF1": macroF1, "val/loss": loss, ...}

def log_finetuning_validation_metrics(...):
    metrics = _compute_finetuning_metrics(preds, y, num_classes)
    if _experiment is not None:
        _experiment.log_metrics(metrics, epoch=epoch)
        _experiment.log_confusion_matrix(...)
        ...
    return metrics  # always return, regardless of Comet state
```

---

### 3. Gradient accumulation is broken when `batch_size > 1`
**File:** `multiplex-image-model/finetuning.py` — `train_masked()`, ~line 268

**Code:**
```python
for batch_idx, (bags, ...) in enumerate(train_dataloader):
    for crops, channel_ids, img_path_full in zip(bags, channel_ids_list, img_paths):
        # inner loop runs batch_size times per batch_idx
        loss = loss_fn(logits, y)
        scaler.scale(loss / gradient_accumulation_steps).backward()

    if (batch_idx + 1) % gradient_accumulation_steps == 0:
        scaler.step(optimizer)  # steps after gradient_accumulation_steps outer iters
```

**Problem:** With `batch_size=4` and `gradient_accumulation_steps=4` (as in the YAML config):
- Each `batch_idx` processes **4 images** (inner loop runs 4 times)
- Each image contributes `loss / 4` to the gradient
- Optimizer steps every 4 `batch_idx` steps
- Effective gradients per optimizer step: `4 × 4 = 16 image losses / 4 = 4×` the intended scale

The loss normalization only accounts for `gradient_accumulation_steps` but not for `batch_size`.
The effective learning rate is `batch_size` (4×) too large, causing unstable or diverging training.

**Fix:** Either set `batch_size=1` in the config (intended design for MIL where one image = one bag),
or normalize by the total number of backward calls per optimizer step:
```python
# Option A: simplest — always use batch_size=1 for MIL training
# In finetuning_config.yaml: batch_size: 1

# Option B: normalize correctly in code
effective_accum = gradient_accumulation_steps * len(bags)  # bags = batch
scaler.scale(loss / effective_accum).backward()
```

---

## HIGH SEVERITY BUGS — Correctness / Data Leakage

### 4. Test set is used as validation set by default → data leakage for checkpoint selection
**File:** `multiplex-image-model/finetuning.py`, ~line 581

**Code:**
```python
if val_dataset is None:
    val_dataset = test_dataset
    print("No val split requested; using test set for validation.")
```

**Config:** `val_split_ratio: 0.0` (disabled by default)

**Problem:** With the default config, `val_dataset = test_dataset`. The best checkpoint is selected
based on validation macro-F1, which is computed on the test set. This means the test split is used
during training for early stopping and checkpoint selection — a direct data leakage. The final
`test_masked()` call then re-evaluates the same data that was used to select the model.

**Fix:** Always use a proper held-out validation split. Set `val_split_ratio: 0.15` or `0.2` in
the config. The infrastructure for `val_split_ratio > 0` is already in place.

---

### 5. No patient-level stratification in train/val/test split
**File:** `multiplex-image-model/finetuning.py` and `multiplex-image-model/multiplex_model/data.py`

**Problem:** `DatasetFromTIFF` splits data by image filenames, not by patient ID. In cohort studies,
one patient may have multiple images (e.g., sequential sections, multiple cores). If images from the
same patient appear in both train and test, the model can memorise patient-specific tissue patterns
and report inflated test F1. This is a well-known source of performance overestimation in
pathology ML.

**Fix:**
1. Add a patient-ID column to the melted table (or derive it from image filenames).
2. Split `train_dataset.imgs` by patient ID, not by image index.
3. Ensure `random_split` is called on patient groups, not individual images.

---

### 6. `saliency_data` has a double-append: one entry has raw tensor weights, one has numpy
**File:** `multiplex-image-model/finetuning.py` — `test_masked()`, ~line 382

**Code:**
```python
if n_images_with_weight < 5:
    saliency_data.append([img, coords, weights])   # BUG: weights is still a raw Tensor here
    if weights is not None:
        w_np = np.asarray(w_t).squeeze()
        saliency_data.append([img, coords, w_np])  # second append with numpy weights
        n_images_with_weight += 1
```

**Problem:** For every saliency image, TWO entries are appended to `saliency_data`:
- First entry: raw `torch.Tensor` weights (not yet converted)
- Second entry: correct numpy weights

This causes `log_finetuning_validation_metrics` to receive entries with inconsistent types,
likely crashing `plot_attention_saliency_imc` on the tensor entry.

**Fix:** Remove the first (unconverted) append:
```python
if n_images_with_weight < 5:
    if weights is not None:
        w_np = weights.detach().cpu().numpy().squeeze()
        saliency_data.append([img, coords, w_np])
        n_images_with_weight += 1
```

---

### 7. `config` global variable referenced inside `train_masked()` — fragile scoping
**File:** `multiplex-image-model/finetuning.py` — `train_masked()`, ~line 238

**Code:**
```python
optimizer.add_param_group(
    {"params": model.encoder.parameters(), "lr": config.encoder_lr}
)
# ...
scheduler = get_scheduler_with_warmup(
    optimizer, ..., final_lr=config.final_lr, peak_lr=config.classifier_lr, ...
)
```

**Problem:** `config` is defined in the `if __name__ == "__main__":` block at module scope and is
accessed as a global variable inside `train_masked()`. The function signature does not accept
`config` as a parameter. This means:
- The function cannot be called from tests or other scripts without setting the global first
- IDE refactoring / moving code will silently break this
- Any use of `config` inside `train_masked` is invisible from the function signature

**Fix:** Pass `config` (or the needed fields: `encoder_lr`, `classifier_lr`, `final_lr`,
`frac_warmup_steps`) as an explicit function parameter.

---

## MEDIUM SEVERITY — Training Strategy Issues

### 8. No data augmentation during training — likely a major driver of overfitting
**File:** `multiplex-image-model/finetuning.py`, ~line 543

**Code:**
```python
train_transform = GridCrop(config.input_image_size[0], max_crops=config.max_crops_per_image)
test_transform  = GridCrop(config.input_image_size[0], max_crops=config.max_crops_per_image)
```

**Problem:** Training and test crops use identical deterministic grid cropping. No random flips,
rotations, intensity jitter, or stochastic crop offsets are applied during training. On small
cohorts (typical in IMC), this means:
- The model sees exactly the same crop positions every epoch
- No spatial invariance is learned
- Overfitting is almost guaranteed on small datasets (~100 images per cohort)

**Fix:**
- Add random horizontal/vertical flip augmentation during training
- Add random crop offset (shift the grid origin by ±`crop_size//4` pixels)
- Consider random intensity scaling per channel (simulate staining variability)
- Consider using a different transform class for training vs test

---

### 9. `classifier_lr = 5e-3` is likely too high for ABMIL head on a frozen foundation model
**File:** `multiplex-image-model/finetuning_config.yaml`, line `classifier_lr: 5e-3`

**Problem:** The ABMIL head is trained for 5 epochs at `5e-3` on top of frozen foundation model
features. For a 768-dim input bag → [128] attention network → 1 output, this learning rate causes
large weight updates in early epochs, potentially destroying the signal in the frozen features
before the encoder is unfrozen at epoch 5. Common practice for fine-tuning foundation models:
head LR should be in `[1e-4, 5e-4]` range.

**Fix:** Try `classifier_lr: 2e-4` to `5e-4`. Use a learning rate sweep.

---

### 10. `classifier_dropout = 0.5` in ABMILHead is hardcoded, not configurable
**File:** `multiplex-image-model/multiplex_model/modules/abmil.py`, ~line 103  
**File:** `multiplex-image-model/multiplex_model/utils/configuration.py` — `ABMILHeadConfig`

**Code in `abmil.py`:**
```python
def __init__(self, ..., classifier_dropout: float = 0.5):
```
**Code in `configuration.py`:**
```python
class ABMILHeadConfig(BaseModel):
    hidden_dim: int = ...
    gated: bool = ...
    dropout: float = ...
    classifier_hidden_dims: list[int] = ...
    # classifier_dropout is NOT here → always defaults to 0.5, no YAML override
```

**Problem:** The classifier dropout of 0.5 is applied inside the bag-level MLP but cannot be
controlled from the config. With a small training set (few dozen bags), 0.5 classifier dropout
combined with only [128] hidden dim may over-regularise and prevent the model from learning anything.

**Fix:** Add `classifier_dropout: float = 0.0` to `ABMILHeadConfig`, pass it through in
`finetuning.py` when constructing `ABMILHead`. Start with `classifier_dropout: 0.0` or `0.25`.

---

### 11. `weighted_sampler_num_samples: 400` is hardcoded without regard to dataset size
**File:** `multiplex-image-model/finetuning_config.yaml`, line `weighted_sampler_num_samples: 400`

**Problem:** With `imbalance_strategy: weighted_sampler`, the sampler draws 400 samples per epoch.
If the actual labeled dataset has fewer than 400 samples (e.g., 80–150 in small IMC cohorts),
the model sees each image 3–5 times per epoch with weighted resampling. Combined with no
augmentation, this massively increases overfitting. If the dataset has more than 400, minority
classes are undersampled.

**Fix:** Set `weighted_sampler_num_samples: null` to default to the labeled dataset size, and add
augmentation to avoid seeing identical crops on repeated samples.

---

### 12. `epochs: 20` with 5 freeze epochs leaves only 15 epochs to fine-tune encoder — too few
**File:** `multiplex-image-model/finetuning_config.yaml`, line `epochs: 20`

**Problem:** The encoder is frozen for 5 epochs, then unfrozen. With early stopping (bug #1 aside),
the actual useful training window is 15 epochs. Given the cosine scheduler restarts at epoch 5,
the effective LR starts high again, and 15 annealing epochs is very short for adapting a large
ConvNeXt backbone.

**Fix:** Increase to `epochs: 50` or more. With early stopping fixed (#1), this is safe.

---

### 13. Preprocessing inconsistency: arcsinh not applied during fine-tuning
**File:** `multiplex-image-model/finetuning.py`, ~line 503  
**File:** `multiplex-image-model/multiplex_model/data.py` — `DatasetFromTIFF.__getitem__`

**Code:**
```python
train_dataset = DatasetFromTIFF(
    ...
    use_preprocessing=False,   # arcsinh(x/5) is skipped
    use_butterworth_filter=True,
    use_clip_normalization=True,
    ...
)
```

**Problem:** `use_preprocessing=False` means the arcsinh transform is NOT applied. The pretraining
likely used a specific normalization pipeline. If the encoder was pretrained with arcsinh +
clip normalization but fine-tuning only uses Butterworth + clip, the features extracted from the
encoder will be out-of-distribution relative to what the encoder learned. This mismatch alone can
explain the large F1 drop.

**Fix:** Check what normalization was used during pretraining (`train_masked_config.yaml`) and
replicate it exactly in fine-tuning. Add a comment to the config explaining the normalization
contract. A good default is `use_preprocessing: true` (arcsinh) + clip normalization,
consistent with pretraining.

---

### 14. Encoder unfreeze at `epoch == 5` is fragile on checkpoint resume
**File:** `multiplex-image-model/finetuning.py` — `train_masked()`, ~line 236

**Code:**
```python
if epoch == 5:
    for p in model.encoder.parameters():
        p.requires_grad = True
    optimizer.add_param_group(...)
```

**Problem:** If training is resumed from a checkpoint saved at epoch 6+, the `if epoch == 5`
condition is never triggered. The encoder will stay frozen for the rest of training and a second
`add_param_group` will NOT be called. Conversely, if resumed from epoch 4, the encoder will be
correctly unfrozen at epoch 5.

**Fix:** Store an `encoder_unfrozen` boolean in the checkpoint, and unfreeze based on that flag
rather than on the epoch number:
```python
encoder_unfrozen = checkpoint.get("encoder_unfrozen", False)
if not encoder_unfrozen and epoch >= 5:
    # unfreeze...
    encoder_unfrozen = True
# save to checkpoint: "encoder_unfrozen": encoder_unfrozen
```

---

### 15. `no_improvement` early stopping threshold is `> 7` (not `>= 7`), so it fires at 8
**File:** `multiplex-image-model/finetuning.py` — `train_masked()`, ~line 325

**Code:**
```python
if no_improvement > 7:
    return best_checkpoint_path
```

**Problem:** Minor — the threshold is 8 (fires when `no_improvement` hits 8), not 7 as the
constant implies. After fixing bug #1 (never reset), the threshold should be made explicit with a
config parameter rather than a hardcoded value.

**Fix:** Add `early_stopping_patience: 10` to `FinetuningConfig` and use it here. Expose it in
the YAML. Default patience should be larger (10–15 epochs) given the small datasets.

---

## LOW SEVERITY — Code Quality / Logging

### 16. Melted table path is hardcoded as a relative path
**File:** `multiplex-image-model/finetuning.py`, ~line 540

**Code:**
```python
MELTED_TABLE_PATH = "../melted_table/results/melted_table.csv"
```

**Problem:** This path is relative to the working directory, not to the script location. Running
`finetuning.py` from a different directory will silently fail with a misleading `FileNotFoundError`.

**Fix:** Add `melted_table_path` to `FinetuningConfig` and resolve it relative to the config file's
location, or use an absolute path.

---

### 17. Saliency images are only collected for the first 5 validation images, all from the same class likely
**File:** `multiplex-image-model/finetuning.py` — `test_masked()`, ~line 376

**Code:**
```python
if n_images_with_weight < 5:
    saliency_data.append(...)
```

**Problem:** The first 5 images encountered in the validation DataLoader are used. Since the loader
is not shuffled for validation, these 5 images are always the same ones and may all belong to the
same class.

**Fix:** Collect one saliency image per class (stratified by `y.item()`):
```python
classes_seen = set()
if len(classes_seen) < num_classes and y.item() not in classes_seen:
    saliency_data.append(...)
    classes_seen.add(y.item())
```

---

### 18. `PanelBatchSampler` is not used in fine-tuning — its batching logic is silently bypassed
**File:** `multiplex-image-model/finetuning.py`, ~line 557

**Problem:** The `DataLoader` is created with `shuffle=True` and no custom sampler. Since all
images in fine-tuning belong to a single `subset`, they are from one panel, so cross-panel mixing
cannot happen. However, the `PanelBatchSampler` exists specifically for this use case. Not using it
is inconsistent and may cause subtle issues if multi-panel fine-tuning is added later.

**Fix:** Use `PanelBatchSampler(train_dataset, batch_size=config.batch_size)` as the sampler for
consistency with the pretraining pipeline, and remove `shuffle=True`.

---

### 19. `log_finetuning_training_metrics` and `log_finetuning_validation_metrics` crash if `_experiment is None` and code accesses their return value
**File:** `multiplex-image-model/multiplex_model/utils/train_logging.py`, ~line 626  
**File:** `multiplex-image-model/finetuning.py`, ~line 294

**Code in `train_logging.py`:**
```python
def log_finetuning_training_metrics(...):
    if _experiment is None:
        return   # returns None
    ...
    return metrics
```

**Code in `finetuning.py`:**
```python
train_metrics = log_finetuning_training_metrics(...)
print(f"  macro-F1={train_metrics['train/macroF1']:.4f}")  # TypeError if None
```

**Problem:** If `_experiment` is `None` (no Comet configured), both `log_finetuning_training_metrics`
and `log_finetuning_validation_metrics` return `None`, causing `TypeError: 'NoneType' object is
not subscriptable`. This also means no metrics are ever printed to stdout, making debugging
impossible without Comet access.

**Fix:** Same as #2 — separate computation from logging. Metrics should always be computed and
returned; Comet logging is a side effect.

---

## Summary Table

| # | Severity | File | Issue | Fix |
|---|----------|------|-------|-----|
| 1 | **Critical** | `finetuning.py` | `no_improvement` never reset → early stop fires after 7 total (not consecutive) bad epochs | Add `no_improvement = 0` in `else` branch |
| 2 | **Critical** | `train_logging.py` | `log_finetuning_validation_metrics` returns `None` when Comet not active → best checkpoint never saved | Separate metric computation from logging; always return dict |
| 3 | **Critical** | `finetuning.py` | Gradient accumulation broken with `batch_size > 1` → LR effectively `batch_size × ` too large | Use `batch_size: 1` in config, or fix normalization |
| 4 | **High** | `finetuning.py` | Test set used as validation by default (`val_split_ratio: 0.0`) → data leakage | Set `val_split_ratio: 0.15` |
| 5 | **High** | `finetuning.py` | No patient-level stratification in splits | Implement patient-level grouping |
| 6 | **High** | `finetuning.py` | `saliency_data` double-append: raw tensor + numpy → likely crash in visualization | Remove first (unconverted) append |
| 7 | **Medium** | `finetuning.py` | `config` global used inside `train_masked()` | Pass as explicit parameter |
| 8 | **Medium** | `finetuning.py` | No augmentation during training → severe overfitting on small datasets | Add random flips, crop offset, intensity jitter |
| 9 | **Medium** | `finetuning_config.yaml` | `classifier_lr: 5e-3` too high for ABMIL on frozen foundation model | Try `2e-4` – `5e-4` |
| 10 | **Medium** | `abmil.py`, `configuration.py` | `classifier_dropout=0.5` hardcoded, not configurable | Add to `ABMILHeadConfig` |
| 11 | **Medium** | `finetuning_config.yaml` | `weighted_sampler_num_samples: 400` hardcoded regardless of dataset size | Set to `null` or derive from actual dataset size |
| 12 | **Medium** | `finetuning_config.yaml` | `epochs: 20` too few with 5-epoch freeze | Increase to 50+ |
| 13 | **Medium** | `finetuning.py` | Preprocessing mismatch: `use_preprocessing=False` in fine-tuning may differ from pretraining | Verify normalization matches pretraining contract |
| 14 | **Medium** | `finetuning.py` | Encoder unfreeze `if epoch == 5` breaks on checkpoint resume | Store `encoder_unfrozen` flag in checkpoint |
| 15 | **Low** | `finetuning.py` | Early-stopping patience hardcoded to `> 7` | Add `early_stopping_patience` to config |
| 16 | **Low** | `finetuning.py` | Melted table path is a hardcoded relative path | Add to `FinetuningConfig` |
| 17 | **Low** | `finetuning.py` | Saliency from first 5 images only; not class-stratified | Collect one per class |
| 18 | **Low** | `finetuning.py` | `PanelBatchSampler` not used in fine-tuning | Use it for consistency |
| 19 | **Low** | `train_logging.py`, `finetuning.py` | Metrics functions crash with `TypeError` if Comet not active and caller indexes result | See #2 fix |
