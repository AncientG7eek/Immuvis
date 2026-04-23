import os
import sys

import comet_ml  # noqa: F401
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from ruamel.yaml import YAML
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

from multiplex_model.clinical import LabelEncoder, get_a_subset
from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler, GridCrop
from multiplex_model.modules.abmil import ABMILHead
from multiplex_model.modules.immuvis import CropClassifierHead, FinetuningModel
from multiplex_model.utils import (
    finish_experiment,
    get_run_name,
    get_scheduler_with_warmup,
    init_experiment,
    log_finetuning_validation_metrics,
)
from multiplex_model.utils.configuration import FinetuningConfig


def calc_class_imbalance(melted_table, subset, cli_feat_for_subset, classes, device):

    # --- START: Add class weight calculation ---
    # Get labels for the specific subset and feature
    train_labels_df = melted_table[
        (melted_table["dataset"] == subset) &
        (melted_table["feature"] == cli_feat_for_subset)
    ]
    # Ensure we only use labels that are in the defined classes
    train_labels_df = train_labels_df[train_labels_df['value'].isin(classes)]

    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.array(classes, dtype=train_labels_df['value'].dtype),
        y=train_labels_df['value'].values
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    print(f"Using class weights: {class_weights_tensor}")
    return class_weights_tensor


def bag_collate(batch):
    """Preserve per-image (bag) structure for ABMIL-compatible batching.

    Each element in `batch` is one image from DatasetFromTIFF.__getitem__:
        (crops: Tensor[n_crops, C, H, W],
         coords: np.ndarray,
         channel_ids: Tensor[C],
         dataset: list[str],
         img_paths: list[str])

    Returns:
        bags        : list of Tensor[n_crops_i, C, H, W], one per image
        channel_ids : list of Tensor[C], one per image
        img_paths   : list of str (full path), one per image
    """
    bags, channel_ids_list, img_paths = [], [], []
    for crops, _coords, channel_ids, _dataset, img_paths_item in batch:
        bags.append(crops)
        channel_ids_list.append(channel_ids)
        img_paths.append(
            img_paths_item[0] if isinstance(img_paths_item, list) else img_paths_item
        )
    return bags, channel_ids_list, img_paths


def _get_label(
    img_stem: str,
    melted_table: pd.DataFrame,
    cli_feat_for_subset: str,
    label_encoder: LabelEncoder,
    device: torch.device,
) -> torch.Tensor | None:
    """Return (1,) long tensor with encoded label, or None if unavailable."""
    img_rows = get_a_subset(melted_table, "img_name", img_stem)
    selected = get_a_subset(img_rows, "feature", cli_feat_for_subset).dropna()
    if len(selected) == 0:
        return None
    label_val = selected.iloc[0]["value"]
    return torch.tensor(
        label_encoder.encode([label_val]), device=device, dtype=torch.long
    )


def train_masked(
    model,
    optimizer,
    scheduler,
    train_dataloader,
    val_dataloader,
    device,
    cli_feat_for_subset,
    melted_table,
    classes,
    marker_names_map,
    epochs=10,
    gradient_accumulation_steps=1,
    start_epoch=0,
    save_checkpoint_every=5,
    checkpoints_path="checkpoints",
):
    """Fine-tune model on clinical classification, one image (bag) per step."""
    
    scaler = GradScaler()
    run_name = get_run_name()
    label_encoder = LabelEncoder(classes)
    class_weights = calc_class_imbalance(melted_table, subset, cli_feat_for_subset, classes, device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    os.makedirs(checkpoints_path, exist_ok=True)

    step = start_epoch * (len(train_dataloader) // gradient_accumulation_steps)

    for epoch in range(start_epoch, epochs):
        model.train()
        all_preds, all_y = [], []
        running_loss = 0.0
        n_images = 0

        for batch_idx, (bags, channel_ids_list, img_paths) in enumerate(
            tqdm(train_dataloader, desc=f"Epoch {epoch}")
        ):
            # With PanelBatchSampler batch_size=1 each outer iter = 1 image.
            # Inner loop keeps code correct for future batch_size > 1.
            for crops, channel_ids, img_path_full in zip(bags, channel_ids_list, img_paths):
                img_stem = img_path_full.split("/")[-1].split(".")[0]
                y = _get_label(img_stem, melted_table, cli_feat_for_subset, label_encoder, device)
                if y is None:
                    print(f"Skipping {img_stem} — no clinical data")
                    continue

                crops = crops.to(device, dtype=torch.float32)
                channel_ids = channel_ids.to(device, dtype=torch.long)

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(crops, channel_ids)   # (1, num_classes)
                    loss = loss_fn(logits, y)

                all_preds.append(torch.argmax(logits.detach(), dim=1).cpu())
                all_y.append(y.cpu())
                running_loss += loss.item()
                n_images += 1

                scaler.scale(loss / gradient_accumulation_steps).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                step += 1

        if n_images == 0:
            print(f"Epoch {epoch}: no valid batches, skipping")
            continue

        epoch_loss = running_loss / n_images
        all_preds_np = torch.cat(all_preds).numpy()
        all_y_np = torch.cat(all_y).numpy()

        train_metrics = log_finetuning_validation_metrics(
            val_loss=epoch_loss,
            val_preds=all_preds_np,
            val_y=all_y_np,
            label_encoder=label_encoder,
            epoch=epoch,
        )
        print(
            f"[Train] loss={train_metrics['val/loss']:.4f}"
            f"  macro-F1={train_metrics['val/macroF1']:.4f}"
        )

        test_masked(
            model,
            val_dataloader,
            device,
            epoch,
            marker_names_map=marker_names_map,
            melted_table=melted_table,
            cli_feat_for_subset=cli_feat_for_subset,
            classes=classes,
            label_encoder=label_encoder,
            step=step,
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
        }
        if (epoch + 1) % save_checkpoint_every == 0:
            torch.save(
                checkpoint,
                f"{checkpoints_path}/checkpoint-{run_name}-epoch_{epoch}.pth",
            )
        torch.save(checkpoint, f"{checkpoints_path}/last_checkpoint-{run_name}.pth")

    final_model_path = f"{checkpoints_path}/final_model-{run_name}.pth"
    print(f"Training completed. Saving final model at {final_model_path}...")
    torch.save({"model_state_dict": model.state_dict()}, final_model_path)


def test_masked(
    model,
    test_dataloader,
    device,
    epoch,
    marker_names_map,
    melted_table,
    cli_feat_for_subset,
    classes,
    label_encoder,
    step,
):
    """Validate model on the test set. One prediction per image (bag)."""
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    running_loss = 0.0
    all_preds, all_y = [], []
    n_images = 0

    with torch.no_grad():
        for bags, channel_ids_list, img_paths in tqdm(
            test_dataloader, desc=f"Validation epoch {epoch}"
        ):
            for crops, channel_ids, img_path_full in zip(bags, channel_ids_list, img_paths):
                img_stem = img_path_full.split("/")[-1].split(".")[0]
                y = _get_label(img_stem, melted_table, cli_feat_for_subset, label_encoder, device)
                if y is None:
                    continue

                crops = crops.to(device, dtype=torch.float32)
                channel_ids = channel_ids.to(device, dtype=torch.long)

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(crops, channel_ids)   # (1, num_classes)
                    loss = loss_fn(logits, y)

                all_preds.append(torch.argmax(logits.detach(), dim=1).cpu())
                all_y.append(y.cpu())
                running_loss += loss.item()
                n_images += 1

    if n_images == 0:
        print(f"Epoch {epoch}: no valid validation batches")
        return {}

    val_loss = running_loss / n_images
    all_preds_np = torch.cat(all_preds).numpy()
    all_y_np = torch.cat(all_y).numpy()

    val_metrics = log_finetuning_validation_metrics(
        val_loss=val_loss,
        val_preds=all_preds_np,
        val_y=all_y_np,
        label_encoder=label_encoder,
        epoch=epoch,
    )
    print(f"{'=' * 40} EPOCH {epoch + 1} {'=' * 40}")
    print(f"Val loss: {val_loss:.4f}")
    if "val/macroF1" in val_metrics:
        print(f"Val macro-F1: {val_metrics['val/macroF1']:.4f}")
    print("=" * 90)
    return val_metrics


if __name__ == "__main__":
    config_path = sys.argv[1]
    yaml = YAML(typ="safe")
    with open(config_path, "r") as f:
        raw_config = yaml.load(f)

    config = FinetuningConfig(**raw_config)

    device = config.device
    print(f"Using device: {device}")

    PANEL_CONFIG = YAML().load(open(config.panel_config))
    TOKENIZER = YAML().load(open(config.tokenizer_config))
    INV_TOKENIZER = {v: k for k, v in TOKENIZER.items()}

    MELTED_TABLE_PATH = "../melted_table/results/melted_table.csv"
    melted_table = pd.read_csv(MELTED_TABLE_PATH)

    train_transform = GridCrop(config.input_image_size[0], max_crops=config.max_crops_per_image)
    test_transform  = GridCrop(config.input_image_size[0], max_crops=config.max_crops_per_image)

    for subset, cli_feat_for_subset in config.dataset_subsets:
        classes = melted_table[melted_table["dataset"] == subset]
        classes = classes[classes["feature"] == cli_feat_for_subset].dropna()
        classes = np.unique(classes["value"])
        print(f"subset={subset}  feature={cli_feat_for_subset}  classes={classes}")

        train_dataset = DatasetFromTIFF(
            panels_config=PANEL_CONFIG,
            split="train",
            marker_tokenizer=TOKENIZER,
            subset=subset,
            transform=train_transform,
            use_preprocessing=False,
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
            subset=subset,
            transform=test_transform,
            use_preprocessing=False,
            use_median_denoising=False,
            use_butterworth_filter=True,
            use_minmax_normalization=False,
            use_clip_normalization=True,
            file_extension="tiff",
        )
        print(f"train size: {len(train_dataset)}  test size: {len(test_dataset)}")

        train_batch_sampler = PanelBatchSampler(train_dataset, config.batch_size)
        test_batch_sampler  = PanelBatchSampler(test_dataset, config.batch_size, shuffle=False)

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=config.num_workers,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=bag_collate,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_sampler=test_batch_sampler,
            num_workers=config.num_workers,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=bag_collate,
        )

        num_channels = len(TOKENIZER)
        num_classes = len(classes)
        input_dim = config.encoder_config.pm_embedding_dims[-1]

        if config.head_type == "abmil":
            abmil = config.abmil_config
            head = ABMILHead(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dim=abmil.hidden_dim,
                gated=abmil.gated,
                dropout=abmil.dropout,
                classifier_hidden_dims=abmil.classifier_hidden_dims,
            )
            print(f"Head: ABMILHead (gated={abmil.gated}, hidden_dim={abmil.hidden_dim})")
        else:
            head = CropClassifierHead(
                input_dim=input_dim,
                hidden_dims=config.classifier_config.hidden_dims,
                num_classes=num_classes,
            )
            print(f"Head: CropClassifierHead (hidden_dims={config.classifier_config.hidden_dims})")

        model = FinetuningModel(
            num_channels=num_channels,
            encoder_config=config.encoder_config.model_dump(),
            head=head,
        ).to(device)

        if config.resolve_checkpoint():
            print(f"Loading encoder weights from: {config.from_checkpoint}")
            ckpt = torch.load(config.from_checkpoint, map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            encoder_keys = {
                k[len("encoder."):]: v
                for k, v in state_dict.items()
                if k.startswith("encoder.")
            }
            if encoder_keys:
                missing, unexpected = model.encoder.load_state_dict(encoder_keys, strict=False)
                print(f"Encoder loaded — missing: {len(missing)}  unexpected: {len(unexpected)}")
            else:
                print("Warning: no 'encoder.*' keys found in checkpoint")

        total_steps = (
            len(train_dataloader) * config.epochs // config.gradient_accumulation_steps
        )
        num_warmup_steps = int(total_steps * config.frac_warmup_steps)
        num_annealing_steps = total_steps - num_warmup_steps

        optimizer = optim.AdamW(
            model.parameters(), lr=config.peak_lr, weight_decay=config.weight_decay
        )
        scheduler = get_scheduler_with_warmup(
            optimizer,
            num_warmup_steps,
            num_annealing_steps,
            final_lr=config.final_lr,
            peak_lr=config.peak_lr,
            type="cosine",
        )

        init_experiment(config.model_dump())

        train_masked(
            model,
            optimizer,
            scheduler,
            train_dataloader,
            test_dataloader,
            device,
            cli_feat_for_subset,
            melted_table,
            classes,
            marker_names_map=INV_TOKENIZER,
            epochs=config.epochs,
            start_epoch=0,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            save_checkpoint_every=config.save_checkpoint_freq,
            checkpoints_path=config.checkpoints_dir,
        )

        finish_experiment()
