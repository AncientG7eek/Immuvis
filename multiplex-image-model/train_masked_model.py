import os
import sys

import comet_ml  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from ruamel.yaml import YAML
from torch.amp import GradScaler, autocast
from torch.nn.functional import normalize
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotation,
)
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from multiplex_model.clinical import LabelEncoder, concat_metadata, merge_metadata_with_melted, get_a_subset
from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler, TestCrop, GridCrop
from multiplex_model.losses import RankMe, beta_nll_loss, nll_loss
from multiplex_model.modules.immuvis import Finetuning
from multiplex_model.utils import (
    ClampWithGrad,
    TrainingConfig,
    apply_channel_masking,
    apply_spatial_masking,
    finish_experiment,
    get_run_name,
    get_scheduler_with_warmup,
    init_experiment,
    log_finetuning_validation_metrics,
    log_validation_metrics,
    plot_reconstructs_with_masks,
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
    beta=1.0,
    min_channels_frac=0.75,
    fully_masked_channels_max_frac=0.5,
    spatial_masking_ratio=0.6,
    mask_patch_size=8,
    start_epoch=0,
    save_checkpoint_every=5,
    checkpoints_path="checkpoints",
):
    """Train a masked autoencoder (decode the remaining channels) with the given parameters."""
    model.train()
    scaler = GradScaler()
    run_name = get_run_name()
    print(f"classes: {classes}")
    label_encoder = LabelEncoder(classes)
    print(f"label dict: {label_encoder.get_dict()}")

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path, exist_ok=True)
        print(f"Created checkpoints directory at {checkpoints_path}")

    step = start_epoch * (len(train_dataloader) // gradient_accumulation_steps)

    for epoch in range(start_epoch, epochs):
        model.train()
        all_preds = []
        all_y = []
        print(len(train_dataloader))
        for batch_idx, batch_data in enumerate(
            tqdm(train_dataloader, desc=f"Epoch {epoch}")
        ):
            # Handle both GridCrop (with coords) and regular transforms
            if len(batch_data) == 5:  # GridCrop: (crops, coords, channel_ids, dataset, img_path)
                img, coords, channel_ids, panel_idx, img_path = batch_data
            else:  # Regular: (img, channel_ids, dataset, img_path)
                img, channel_ids, panel_idx, img_path = batch_data
            
            img = img.to(device, dtype=torch.float32)
            print(f"num crops: {len(img)}", flush=True)
            channel_ids = channel_ids.to(device, dtype=torch.long)
            print(f"img shape: {img.shape}")
            
            # Handle batch of crops: reshape (batch_size, num_crops, C, H, W) -> (batch_size * num_crops, C, H, W)
            if img.dim() == 5:  # (batch, num_crops, C, H, W)
                batch_size, num_crops = img.shape[0], img.shape[1]
                img = img.view(batch_size * num_crops, *img.shape[2:])
                # Repeat channel_ids and img_path for each crop
                if isinstance(channel_ids, torch.Tensor) and channel_ids.dim() == 1:
                    channel_ids = channel_ids.unsqueeze(0).repeat(batch_size * num_crops, 1).squeeze()
                img_path = [p.split("/")[-1].split('.')[0] for p in img_path for _ in range(num_crops)] if isinstance(img_path, list) else img_path.split("/")[-1].split('.')[0]
            else:
                img_path = [p.split("/")[-1].split('.')[0] for p in img_path]
                
            print(img_path)
            # here use img_path to get clinical 
            all_cli_features = get_a_subset(melted_table, "img_name", img_path)
            print(f"all cli feat: {all_cli_features}")
            
            selected_cli_feat = get_a_subset(all_cli_features, "feature", cli_feat_for_subset)
            print(f"selected cli feat: {selected_cli_feat}")
            print("Encoding labels")
            label_per_crop = []
            for img_p in img_path:
                per_crop = selected_cli_feat[selected_cli_feat["img_name"].astype(str)==img_p]
                print(f"per crop: {per_crop}")
                label_per_crop.append(per_crop)
            label_per_crop = pd.concat(label_per_crop)
            print(f"label per crop: {label_per_crop}")
            print(f"labels to encode: {label_per_crop["value"]}")
            y = torch.tensor(label_encoder.encode(label_per_crop["value"].values), device=device, dtype=torch.long)
            print(f"encoded labels: {y}")
            loss_fn = torch.nn.CrossEntropyLoss()

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                print("Calculating logits")
                logits = model(x=img, encoded_indices=channel_ids)
                print("Calculating loss")
                loss = loss_fn(logits, y)
            print("Calculating preds")

            preds = torch.argmax(logits.detach(), dim=1)
            all_preds.append(preds.cpu())
            all_y.append(y.cpu())

            scaler.scale(loss / gradient_accumulation_steps).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                step += 1

        all_preds = torch.cat(all_preds).cpu().numpy()
        all_y = torch.cat(all_y).cpu().numpy()

        metrics = log_finetuning_validation_metrics(
            val_loss=loss.item(),
            val_preds=all_preds,
            val_y=all_y,
            label_encoder=label_encoder,
            epoch=epoch,
        )

        print(f"Loss; {metrics['val/loss']}")
        print(f"macro-f1: {metrics['val/macroF1']}")
            

        test_masked(
            model,
            val_dataloader,
            device,
            epoch,
            marker_names_map=INV_TOKENIZER,
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
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, final_model_path)


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
    num_plots=4,
    spatial_masking_ratio=0.6,
    fully_masked_channels_max_frac=0.5,
    mask_patch_size=8,
):
    """Validate the finetuning model on the test set."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_y = []

    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for idx, batch_data in enumerate(
            tqdm(test_dataloader, desc=f"Testing epoch {epoch}")
        ):
            # Handle both GridCrop (with coords) and regular transforms
            if len(batch_data) == 5:  # GridCrop: (crops, coords, channel_ids, dataset, img_path)
                img, coords, channel_ids, panel_idx, img_path = batch_data
            else:  # Regular: (img, channel_ids, dataset, img_path)
                img, channel_ids, panel_idx, img_path = batch_data
            
            img = img.to(device, dtype=torch.float32)
            print(f"num crops: {len(img)}", flush=True)
            channel_ids = channel_ids.to(device, dtype=torch.long)
            print(f"img shape: {img.shape}")
            
            # Handle both GridCrop (with coords) and regular transforms
            if img.dim() == 5:  # (batch, num_crops, C, H, W)
                batch_size, num_crops = img.shape[0], img.shape[1]
                img = img.view(batch_size * num_crops, *img.shape[2:])
                # Repeat channel_ids and img_path for each crop
                if isinstance(channel_ids, torch.Tensor) and channel_ids.dim() == 1:
                    channel_ids = channel_ids.unsqueeze(0).repeat(batch_size * num_crops, 1).squeeze()
                img_path = [p.split("/")[-1].split('.')[0] for p in img_path for _ in range(num_crops)] if isinstance(img_path, list) else img_path.split("/")[-1].split('.')[0]
            else:
                img_path = [p.split("/")[-1].split('.')[0] for p in img_path]
                
            print(img_path)
            # here use img_path to get clinical 
            all_cli_features = get_a_subset(melted_table, "img_name", img_path)
            print(f"all cli feat: {all_cli_features}")
            
            selected_cli_feat = get_a_subset(all_cli_features, "feature", cli_feat_for_subset)
            print(f"selected cli feat: {selected_cli_feat}")
            print("Encoding labels")
            label_per_crop = []
            for img_p in img_path:
                per_crop = selected_cli_feat[selected_cli_feat["img_name"].astype(str)==img_p]
                print(f"per crop: {per_crop}")
                label_per_crop.append(per_crop)
            label_per_crop = pd.concat(label_per_crop)
            print(f"label per crop: {label_per_crop}")
            print(f"labels to encode: {label_per_crop["value"]}")
            y = torch.tensor(label_encoder.encode(label_per_crop["value"].values), device=device, dtype=torch.long)
            print(f"encoded labels: {y}")
            loss_fn = torch.nn.CrossEntropyLoss()

            
            print("Calculating logits")
            logits = model(x=img, encoded_indices=channel_ids)
            print("Calculating loss")
            loss = loss_fn(logits, y)
            print("Calculating preds")

            preds = torch.argmax(logits.detach(), dim=1)
            all_preds.append(preds.cpu())
            all_y.append(y.cpu())

            running_loss += loss.item()
            #     )
            #     log_validation_images(
            #         fig=reconstr_img,
            #         panel_idx=panel_idx[0],
            #         img_path=img_path[0],
            #         epoch=epoch,
            #         masked_channels_names=masked_channels_names,
            #         img_idx=idx,
            #     )
            #     plt.close("all")

    val_loss = running_loss / len(test_dataloader)

    all_preds = torch.cat(all_preds).cpu().numpy()
    all_y = torch.cat(all_y).cpu().numpy()

    val_metrics = log_finetuning_validation_metrics(
        val_loss=val_loss,
        val_preds=all_preds,
        val_y=all_y,
        label_encoder=label_encoder,
        epoch=epoch,
    )
        
    print(f"{'=' * 40} EPOCH {epoch + 1} {'=' * 40}")
    print(f"NLL: {val_loss:.4f}")
    if "val/macroF1" in val_metrics:
        print(f"macro-f1: {val_metrics['val/macroF1']:.4f}")
    print("=" * 90)
    print()

    return val_metrics


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

    return crops[:10], all_coords[:10], channel_ids[:10], all_datasets[:10], all_img_paths[:10]


if __name__ == "__main__":
    # Load the configuration file
    config_path = sys.argv[1]
    yaml = YAML(typ="safe")
    with open(config_path, "r") as f:
        raw_config = yaml.load(f)

    # Validate configuration using Pydantic model
    config = TrainingConfig(**raw_config)

    device = config.device
    print(f"Using device: {device}")

    SIZE = config.input_image_size
    BATCH_SIZE = config.batch_size
    NUM_WORKERS = config.num_workers

    PANEL_CONFIG = YAML().load(open(config.panel_config))
    TOKENIZER = YAML().load(open(config.tokenizer_config))
    INV_TOKENIZER = {v: k for k, v in TOKENIZER.items()}

    MELTED_TABLE_PATH = '/home/kacper/Documents/oświata/UW/2nd_yr/magisterka/Immuvis/melted_table/results/melted_table.csv'
    melted_table = pd.read_csv(MELTED_TABLE_PATH)

    train_transform = GridCrop(SIZE[0])

    test_transform = GridCrop(SIZE[0])

    dataset_subsets = config.dataset_subsets
    for subset, cli_feat_for_subset in dataset_subsets: #dict

        # make datasetform tiff take the subset
        classes = melted_table[melted_table["dataset"]==subset]
        classes = classes[classes["feature"]==cli_feat_for_subset]
        classes = np.unique(classes["value"])
        
        print(f"subset and feature: {subset, cli_feat_for_subset}")
        train_dataset = DatasetFromTIFF(
            panels_config=PANEL_CONFIG,
            split="train",
            marker_tokenizer=TOKENIZER,
            subset=subset,
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
            subset=subset,
            transform=test_transform,
            use_preprocessing=False,  # saved data is already preprocessed
            use_median_denoising=False,
            use_butterworth_filter=True,
            use_minmax_normalization=False,
            use_clip_normalization=True,
            file_extension="tiff",
        )
        print(f"train_dataset len: {len(train_dataset)}")
        train_batch_sampler = PanelBatchSampler(train_dataset, BATCH_SIZE)
        test_batch_sampler = PanelBatchSampler(test_dataset, BATCH_SIZE, shuffle=False)
        print(f"train_batch_sampler len: {len(train_batch_sampler)}")
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            collate_fn=custom_collate,
        )
        print(f"train_dataloader len: {len(train_dataloader)}")
        test_dataloader = DataLoader(
            test_dataset,
            batch_sampler=test_batch_sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            collate_fn=custom_collate,
        )

        # Build model configuration
        num_channels = len(TOKENIZER)
        num_classes = len(classes)
        
        model = Finetuning(
            num_channels=num_channels,
            num_classes=num_classes,
            encoder_config=config.encoder_config.model_dump(),
            classifier_config=config.classifier_config.model_dump(),
        ).to(device)
        
        # Load pretrained encoder weights if specified
        if config.resolve_checkpoint():
            print(f"Loading encoder weights from: {config.from_checkpoint}")
            checkpoint = torch.load(config.from_checkpoint, map_location=device)
            encoder_state_dict = checkpoint.get("model_state_dict", checkpoint)
            
            # Filter to only encoder weights and strip "encoder." prefix
            encoder_keys = {k[8:]: v for k, v in encoder_state_dict.items() if "encoder." in k}
            
            # Try loading full state dict (assumes it's just the encoder)
            try:
                model.encoder.load_state_dict(encoder_keys)
                print(f"Loaded encoder state dict")
            except Exception as e:
                print(f"Warning: Could not load encoder: {e}")


        # Setup optimizer and scheduler
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

        # Initialize Comet.ml experiment
        comet_config = config.model_dump()
        init_experiment(comet_config)

        # Load checkpoint if specified
        start_epoch = 0
        # if config.resolve_checkpoint():
        #     print(f"Loading model from checkpoint: {config.from_checkpoint}")
        #     checkpoint = torch.load(config.from_checkpoint, map_location=device)
        #     #model.load_state_dict(checkpoint["model_state_dict"])
        #     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        #     scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        #     start_epoch = checkpoint["epoch"] + 1

        # Train the model
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
            start_epoch=start_epoch,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            min_channels_frac=config.min_channels_frac,
            spatial_masking_ratio=config.spatial_masking_ratio,
            fully_masked_channels_max_frac=config.fully_masked_channels_max_frac,
            mask_patch_size=config.mask_patch_size,
            save_checkpoint_every=config.save_checkpoint_freq,
            checkpoints_path=config.checkpoints_dir,
            beta=config.beta,
        )

        finish_experiment()
