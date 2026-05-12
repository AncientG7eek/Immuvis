# ImmuVis

Self-supervised foundation model for multiplex imaging (IMC / MIBI) with a clinical fine-tuning pipeline. The model learns to encode multi-channel tissue images into a shared latent space via masked reconstruction, then is fine-tuned end-to-end on image-level clinical labels.

---

## Repository layout

```text
Immuvis/
├── melted_table/          # data preparation: build clinical label table
├── multiplex-image-model/ # model definition, pretraining, fine-tuning
└── predict_clinical/      # baseline: logistic regression on frozen embeddings
```

---

## Modules

### 1. `melted_table/`

Converts heterogeneous per-cohort clinical spreadsheets into a single long-format CSV used by all downstream code.

**Output:** `melted_table/results/melted_table.csv`

| Column | Description |
|---|---|
| `dataset` | Cohort identifier (e.g. `danenberg`) |
| `img_name` | TIFF filename stem |
| `patient` | Patient identifier |
| `feature` | Clinical variable name (e.g. `Grade`, `ER_status`) |
| `value` | Clinical variable value (e.g. `Grade2`, `Positive`) |

**Run:**
```bash
cd melted_table && python melted_table.py
```

Requires `img_data/full_images_list_20122025_mapping_to_full_dir.csv` (image path mapping) and `configs/master_clinical.tsv` (cohort-to-clinical-table mapping).

---

### 2. `multiplex-image-model/`

Core model and training code. See [`multiplex-image-model/README.md`](multiplex-image-model/README.md) for full configuration reference.

#### Architecture

```
TIFF (C × H × W)
  └─► GridCrop → N crops (64×64)
        └─► MultiplexImageEncoder
              ├─ Marker-agnostic ConvNeXt  (per-channel, independent)
              ├─ Hyperkernel               (channel-specific learned fusion)
              └─ Pan-marker ConvNeXt       (cross-channel reasoning)
                    └─► (N, E=768) instance embeddings
                          └─► TaskHead
                                ├─ CropClassifierHead  (mean-pool MIL)
                                └─ ABMILHead           (attention MIL)
                                      └─► (1, K) logits
```

The **Hyperkernel** is the key architectural component: it generates per-marker convolution weights from a shared embedding table, allowing the model to process panels with different marker sets without duplicating the encoder.

#### Training stages

**Stage 1 — Self-supervised pretraining** (`train_masked_model.py`)

The full encoder–decoder autoencoder is trained with masked reconstruction:
- Random channel subset sampling + full channel masking
- Patch-level spatial masking
- Predicts mean and variance per pixel per channel; loss: Beta-NLL

```bash
cd multiplex-image-model
python train_masked_model.py train_masked_config.yaml
```

**Stage 2 — Clinical fine-tuning** (`finetuning.py`)

Loads pretrained encoder weights, attaches a classification head, and trains on image-level labels:
- Epochs 0–4: encoder frozen, head trained at `classifier_lr`
- Epoch 5+: encoder unfrozen, trained at `encoder_lr` (lower)
- Best checkpoint selected on validation macro-F1

```bash
python finetuning.py finetuning_config.yaml
# optional overrides:
python finetuning.py finetuning_config.yaml danenberg Grade cuda
```

**Feature extraction** (`encode_latent.py`)

Extracts and saves per-image latent embeddings from a trained encoder (no labels required).

```bash
python encode_latent.py
```

#### Key configuration (`finetuning_config.yaml`)

```yaml
dataset_subsets: [[danenberg, Grade]]   # cohort + clinical feature
head_type: abmil                        # "logistic" or "abmil"
encoder_lr: 5e-5
classifier_lr: 5e-3
epochs: 20
from_checkpoint: ImmuVis-616-MSE-768-ma1.pth
imbalance_strategy: weighted_sampler    # "class_weight" | "weighted_sampler" | "none"
val_split_ratio: 0.0                    # set > 0 to hold out a validation set
```

#### Module reference

| File | Role |
|---|---|
| `multiplex_model/modules/immuvis.py` | `MultiplexImageEncoder`, `FinetuningModel`, `TaskHead`, `Hyperkernel` |
| `multiplex_model/modules/abmil.py` | `ABMILHead` — attention MIL (Ilse et al., ICML 2018) |
| `multiplex_model/modules/convnext.py` | ConvNeXt blocks and encoder |
| `multiplex_model/data.py` | `DatasetFromTIFF`, `GridCrop`, `PanelBatchSampler` |
| `multiplex_model/clinical.py` | `LabelEncoder`, melted-table utilities |
| `multiplex_model/losses.py` | Beta-NLL loss, RankMe metric |
| `multiplex_model/utils/configuration.py` | Pydantic configs: `FinetuningConfig`, `ABMILHeadConfig` |
| `multiplex_model/utils/train_logging.py` | Comet.ml logging, ROC curves, attention saliency overlays |
| `multiplex_model/utils/masking.py` | Channel and spatial masking for pretraining |

---

### 3. `predict_clinical/`

Baseline pipeline that trains a logistic regression classifier on **frozen** ImmuVis embeddings. Useful for evaluating the foundation model without fine-tuning.

**Pipeline:**
1. Load pre-extracted embeddings (`.npy`) and metadata (`.csv`) from `encode_latent.py` output
2. Merge with `melted_table.csv` to attach clinical labels
3. Optional PCA dimensionality reduction
4. Fit `sklearn.LogisticRegression` per cohort per feature
5. Report confusion matrix, classification report, ROC-AUC

```bash
cd predict_clinical && python predict_clinical.py
```

---

## Installation

```bash
pip install -e multiplex-image-model/
```

Requires Python ≥ 3.12 and PyTorch ≥ 2.7 with CUDA.

---

## Known issues

See [`TODO.md`](TODO.md) for a prioritised list of bugs and training improvements identified in code review, including critical issues affecting fine-tuning macro-F1.

---

## Citation

If you use this code, please cite the ImmuVis preprint (TBD).
