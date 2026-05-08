"""Logging and visualization utilities for training and validation."""

import re
import os
from datetime import datetime
from io import BytesIO
from math import ceil
from typing import Any

import comet_ml
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from itertools import cycle

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay, roc_auc_score, auc, roc_curve

from matplotlib.colors import to_rgb

# Disable PIL's decompression bomb limit
Image.MAX_IMAGE_PIXELS = None

# Global experiment instance
_experiment: comet_ml.Experiment | None = None


# ...existing code...
def _make_imc_rgb_from_groups(
    full_img,
    channel_groups: dict[str, list],
    markers_names_map: dict[int, str] | dict[str, int] | None = None,
    clip_percentile: tuple[float, float] = (1.0, 99.0),
    group_colors = {"nuclear": "blue", "membrane": "green", "neoplasm": "red"},
) -> np.ndarray:
    """Build RGB preview from IMC full image (C,H,W or H,W,C) and 3 channel groups.
    channel_groups keys: 'nuclear','membrane','neoplasm' -> list of channel indices or names.
    Returns HxWx3 float image in [0,1].
    """
    import numpy as np

    # accept torch or numpy
    if isinstance(full_img, torch.Tensor):
        img = full_img.detach().cpu().numpy()
    else:
        img = np.asarray(full_img)

  
    C, H, W = img.shape
    channels = img
    

    def _resolve_idx(x):
        if isinstance(x, int):
            return x
        if isinstance(x, str):
            if markers_names_map is None:
                raise ValueError("marker name used but markers_names_map is None")
            # markers_names_map may be name->idx or idx->name
            if isinstance(next(iter(markers_names_map.keys())), str):
                return markers_names_map[x]
            else:
                # invert mapping
                inv = {v: k for k, v in markers_names_map.items()}
                return inv[x]
        raise ValueError("channel spec must be int or str")

    # accumulate colored layers
    rgb_img = np.zeros((H, W, 3), dtype=float)

    for group_name, color_name in group_colors.items():
        ids = channel_groups.get(group_name, [])
        if len(ids) == 0:
            chan = np.zeros((H, W), dtype=float)
        else:
            idxs = [_resolve_idx(i) if not isinstance(i, (list, tuple)) else i for i in ids]
            flat_idxs = []
            for it in idxs:
                if isinstance(it, (list, tuple)):
                    flat_idxs.extend(it)
                else:
                    flat_idxs.append(it)
            arrs = []
            for ii in flat_idxs:
                if ii < 0 or ii >= channels.shape[0]:
                    raise IndexError(f"channel index {ii} out of range (0..{channels.shape[0]-1})")
                arrs.append(channels[ii])
            chan = np.maximum.reduce(arrs) if len(arrs) > 1 else arrs[0].astype(float)

        # clip and normalize
        lo, hi = np.percentile(chan, clip_percentile)
        chan = np.clip(chan, lo, hi)
        if hi - lo <= 1e-6:
            norm = np.zeros_like(chan, dtype=float)
        else:
            norm = (chan - lo) / (hi - lo)

        # tint by color_name and add to rgb image
        rgb_color = np.array(to_rgb(color_name), dtype=float)  # (3,)
        rgb_img += norm[..., None] * rgb_color[None, None, :]

    # optional: scale if any value >1
    maxv = rgb_img.max()
    if maxv > 1.0:
        rgb_img = rgb_img / maxv

    rgb_img = np.clip(rgb_img, 0.0, 1.0)
    return rgb_img


def plot_attention_saliency_imc(
    full_img,
    crop_coords,
    weights,
    crop_size,
    channel_groups: dict[str, list],
    markers_names_map: dict[int, str] | None = None,
    cmap="Reds",
    overlay_color=(1.0, 0.0, 0.0),
    max_alpha: float = 0.8,
    show_weights: bool = False,
    annotate_topk: int | None = None,
) -> "plt.Figure":
    """Create IMC RGB preview and overlay crop attention saliency.
    - full_img: CxHxW or HxWxC torch / np array
    - crop_coords: sequence of (x,y,w,h) in pixels
    - weights: length-N array (attention per crop)
    - channel_groups: {'nuclear':[...], 'membrane':[...], 'neoplasm':[...]}
    """
    import numpy as np
    import matplotlib.patches as patches

    group_colors = {"nuclear": "blue", "membrane": "green", "neoplasm": "red"}

    rgb = _make_imc_rgb_from_groups(
        full_img,
        channel_groups,
        markers_names_map,
        group_colors=group_colors,
    )

    if hasattr(weights, 'cpu'):  # Check if it's a torch.Tensor
        weights = weights.detach().cpu().numpy()
    weights = np.asarray(weights).squeeze()
    crop_coords = np.asarray(crop_coords)
    if weights.ndim == 0:
        weights = np.array([float(weights)])
    if crop_coords.shape[0] != weights.shape[0]:
        raise ValueError(f"num crops {crop_coords.shape[0]} != num weights {weights.shape[0]}")
    
    # Convert (x,y) top-left to (x,y,w,h) top-left if shape is Nx2
   
    if crop_coords.ndim == 3 and crop_coords.shape[1:] == (2, 2):
        top = crop_coords[:, 0, 0]
        bottom = crop_coords[:, 0, 1]
        left = crop_coords[:, 1, 0]
        right = crop_coords[:, 1, 1]
        
        w = right - left
        h = bottom - top
        
        # Stack into (x, y, w, h)
        crop_coords = np.stack([left, top, w, h], axis=-1)


    # normalize weights to 0..1
    if np.allclose(weights, weights[0]):
        norm_w = np.ones_like(weights) if weights[0] > 0 else np.zeros_like(weights)
    else:
        norm_w = (weights - weights.min()) / (weights.max() - weights.min())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    ax.axis("off")

    # optionally annotate only top-k to reduce clutter
    if annotate_topk is not None:
        order = np.argsort(-norm_w)
        top_mask = np.zeros_like(norm_w, dtype=bool)
        top_mask[order[:annotate_topk]] = True
    else:
        top_mask = np.ones_like(norm_w, dtype=bool)

    for (x, y, w, h), alpha, raw_w, show in zip(crop_coords, norm_w, weights, top_mask):
        a = float(alpha) * float(max_alpha)
        rect = patches.Rectangle((x, y), w, h, linewidth=0, edgecolor=None,
                                 facecolor=overlay_color, alpha=a, zorder=2)
        ax.add_patch(rect)
        if show_weights and show:
            ax.text(x + 2, y + 10, f"{raw_w:.2f}", color="white", fontsize=8, zorder=3)

    # Add legend text for the marker colors drawn in RGB
    
    y_pos = 0.98
    for group_name in ("nuclear", "membrane", "neoplasm"):
        markers = channel_groups.get(group_name, [])
        if not markers:
            continue
        # resolve names
        names = []
        for m in (markers[0] if isinstance(markers, list) and len(markers) > 0 and isinstance(markers[0], list) else markers):
            if isinstance(m, str):
                names.append(m)
            elif markers_names_map is not None:
                # markers_names_map can be int->str or str->int
                if isinstance(next(iter(markers_names_map.keys())), str):
                    inv = {v: k for k, v in markers_names_map.items()}
                    names.append(inv.get(m, str(m)))
                else:
                    names.append(markers_names_map.get(m, str(m)))
            else:
                names.append(str(m))
                
        label_text = f"{group_name.capitalize()}: {', '.join(names)}"
        ax.text(0.02, y_pos, label_text, color=group_colors[group_name],
                fontsize=9, fontweight='bold', transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=1))
        y_pos -= 0.05

    fig.tight_layout()
    return fig


def log_attention_saliency_imc(
    full_img,
    crop_coords,
    weights,
    crop_size,
    channel_groups: dict[str, list],
    markers_names_map: dict[int, str] | None = None,
    name: str = "attention_saliency_imc",
    epoch: int | None = None,
    panel_idx: int | None = None,
    img_idx: int | None = None,
    **plot_kwargs,
):
    """Build figure and upload to Comet if initialized. Returns the figure."""
    fig = plot_attention_saliency_imc(
        full_img, crop_coords, weights, crop_size, channel_groups, markers_names_map, **plot_kwargs
    )
    global _experiment
    if _experiment is not None:
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        meta = {}
        if epoch is not None:
            meta["epoch"] = epoch
        if panel_idx is not None:
            meta["panel_idx"] = panel_idx
        if img_idx is not None:
            meta["img_idx"] = img_idx
        _experiment.log_image(buf, name=name, metadata=meta, step=epoch)
    return fig
# ...existing code...


def plot_reconstructs_with_uncertainty(
    orig_img: torch.Tensor,
    reconstructed_img: torch.Tensor,
    sigma_plot: torch.Tensor,
    channel_ids: torch.Tensor,
    masked_ids: torch.Tensor,
    markers_names_map: dict[int, str],
    ncols: int = 9,
    scale_by_max: bool = True,
    partially_masked_ids: list[int] = [],
):
    """Plot the original image and the reconstructed image with uncertainty.

    Args:
        orig_img (torch.Tensor): Original image
        reconstructed_img (torch.Tensor): Reconstructed image
        sigma_plot (torch.Tensor): Uncertainty image
        channel_ids (torch.Tensor): Indices of the original channels
        masked_ids (torch.Tensor): Indices of the masked/reconstructed channels
        markers_names_map (Dict[int, str]): Channel index to marker name mapping
        ncols (int, optional): Number of columns on the plot. Defaults to 9.
        scale_by_max (bool, optional): Whether to scale the images by their maximum value. Defaults to True.
        partially_masked_ids (List[int], optional): List of channel IDs that were only partially masked. Defaults to [].

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # plot original image
    num_channels = orig_img.shape[1]

    nrows = ceil(num_channels / (ncols // 3))
    fig_orig, axs_orig = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    ax_flat = axs_orig.flatten()
    for i in range(0, len(ax_flat), 3):
        j = i // 3

        # first original image
        ax_img = ax_flat[i]
        ax_img.axis("off")

        ax_reconstructed = ax_flat[i + 1]
        ax_reconstructed.axis("off")

        ax_uncertainty = ax_flat[i + 2]
        ax_uncertainty.axis("off")

        if j < num_channels:
            marker_name = markers_names_map[channel_ids[0, j].item()]
            ax_img.imshow(orig_img[0, j].cpu().numpy(), cmap="CMRmap", vmin=0, vmax=1)
            ax_img.set_title(f"Original\n{marker_name}")

            ax_reconstructed.imshow(
                reconstructed_img[0, j].cpu().numpy(), cmap="CMRmap", vmin=0, vmax=1
            )
            is_masked = channel_ids[0, j].item() in masked_ids
            is_partially_masked = channel_ids[0, j].item() in partially_masked_ids
            if is_partially_masked:
                masked_str = " (partially masked)"
            elif is_masked:
                masked_str = " (masked)"
            else:
                masked_str = ""
            ax_reconstructed.set_title(f"Reconstructed{masked_str}\n{marker_name}")

            if scale_by_max:
                var_min = sigma_plot[0, j].min().item()
                var_max = sigma_plot[0, j].max().item()
            else:
                var_min = None
                var_max = None

            ax_uncertainty.imshow(
                sigma_plot[0, j].cpu().numpy(),
                cmap="CMRmap",
                vmin=var_min,
                vmax=var_max,
            )
            ax_uncertainty.set_title(f"Variance\n{marker_name}")

    fig_orig.tight_layout()

    return fig_orig


def plot_reconstructs_with_masks(
    orig_img: torch.Tensor,
    reconstructed_img: torch.Tensor,
    pixel_masks: torch.Tensor,
    channel_ids: torch.Tensor,
    fully_masked_ids: list[int],
    markers_names_map: dict[int, str],
    ncols: int = 9,
):
    """Plot the original image, masked image (with white pixels where masked), and reconstruction.

    Args:
        orig_img (torch.Tensor): Original image [B, C, H, W]
        reconstructed_img (torch.Tensor): Reconstructed image [B, C_all, H, W] (all channels)
        pixel_masks (torch.Tensor): Boolean pixel-level masks [B, C_active, H, W] where True = masked
        channel_ids (torch.Tensor): Indices of all channels [B, C_all]
        fully_masked_ids (list[int]): List of channel IDs that were fully masked (dropped)
        markers_names_map (dict[int, str]): Channel index to marker name mapping
        ncols (int, optional): Number of columns on the plot. Defaults to 9.

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    num_channels = orig_img.shape[1]

    nrows = ceil(num_channels / (ncols // 3))
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    ax_flat = axs.flatten()

    # Create mapping from channel_id to index in masked_img
    active_channel_ids = [
        cid for cid in channel_ids[0].tolist() if cid not in fully_masked_ids
    ]
    channel_to_masked_idx = {cid: idx for idx, cid in enumerate(active_channel_ids)}

    for i in range(0, len(ax_flat), 3):
        j = i // 3

        ax_orig = ax_flat[i]

        ax_masked = ax_flat[i + 1]

        ax_reconstructed = ax_flat[i + 2]

        if j < num_channels:
            channel_id = channel_ids[0, j].item()
            marker_name = markers_names_map[channel_id]

            # Show original
            ax_orig.imshow(orig_img[0, j].cpu().numpy(), cmap="CMRmap", vmin=0, vmax=1)
            ax_orig.set_title(f"Original\n{marker_name}")
            ax_orig.set_xticks([])
            ax_orig.set_yticks([])
            # Add black frame
            for spine in ax_orig.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(1)
                spine.set_visible(True)

            # Show masked version
            if channel_id in fully_masked_ids:
                # Fully masked channel - show all white (RGBA)
                white_img = np.ones((*orig_img[0, j].shape, 4))
                white_img[..., :3] = 1.0  # RGB = white
                white_img[..., 3] = 1.0  # Alpha = 100% opaque
                ax_masked.imshow(white_img)
                ax_masked.set_title(f"Masked (fully)\n{marker_name}")
                ax_masked.set_xticks([])
                ax_masked.set_yticks([])
                # Add black frame
                for spine in ax_masked.spines.values():
                    spine.set_edgecolor("black")
                    spine.set_linewidth(1)
                    spine.set_visible(True)
            else:
                # Partially masked channel - show with white pixels where masked
                masked_idx = channel_to_masked_idx[channel_id]

                # Convert grayscale to RGBA using colormap (image already normalized to 0-1)
                cmap = plt.cm.CMRmap
                img_data = orig_img[0, j].cpu().numpy()
                rgba_img = cmap(img_data)  # Apply colormap directly

                # Set masked pixels to pure white with 100% opacity
                mask_np = pixel_masks[0, masked_idx].cpu().numpy()
                rgba_img[mask_np] = [1.0, 1.0, 1.0, 1.0]  # Pure white, fully opaque

                ax_masked.imshow(rgba_img)
                ax_masked.set_title(f"Masked\n{marker_name}")
                ax_masked.set_xticks([])
                ax_masked.set_yticks([])
                # Add black frame
                for spine in ax_masked.spines.values():
                    spine.set_edgecolor("black")
                    spine.set_linewidth(1)
                    spine.set_visible(True)

            # Show reconstruction
            ax_reconstructed.imshow(
                reconstructed_img[0, j].cpu().numpy(), cmap="CMRmap", vmin=0, vmax=1
            )
            ax_reconstructed.set_title(f"Reconstructed\n{marker_name}")
            ax_reconstructed.set_xticks([])
            ax_reconstructed.set_yticks([])
            # Add black frame
            for spine in ax_reconstructed.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(1)
                spine.set_visible(True)
        else:
            # Turn off empty subplots
            ax_orig.axis("off")
            ax_masked.axis("off")
            ax_reconstructed.axis("off")

    fig.tight_layout()
    return fig


def get_next_version_number(
    project_name: str,
    workspace: str | None = None,
    api_key: str | None = None,
) -> int:
    """Query Comet.ml API to get the next version number for experiments.

    Looks for existing experiments with names matching the pattern 'ImVs-' followed
    by a number (e.g., 'ImVs-1', 'ImVs-42', 'ImVs-100') and returns the next
    available version number.

    Args:
        project_name (str): Name of the Comet.ml project
        workspace (str | None): Comet.ml workspace name
        api_key (str | None): Comet.ml API key (can also use env var COMET_API_KEY)

    Returns:
        int: Next version number to use
    """
    try:
        api = comet_ml.API(api_key=api_key)

        # --- START FIX ---
        # If workspace/project not passed, check environment variables, just like comet_ml.start()
        if workspace is None:
            workspace = os.getenv("COMET_WORKSPACE")
        if project_name is None:
            project_name = os.getenv("COMET_PROJECT_NAME")
        # --- END FIX ---

        version_pattern = r"^ImVs-(\d+)"
        # Get all experiments in the project
        experiments = api.get_experiments(
            workspace=workspace,
            project_name=project_name,
            pattern=version_pattern,
            sort_by="startTime",
            sort_order="desc",
        )

        # Return next version (1 if no versions exist)
        if not experiments:
            return 1

        latest_experiment = experiments[0]

        version = re.match(version_pattern, latest_experiment.name)
        version = int(version.group(1))

        return version + 1

    except Exception as e:
        print(f"Warning: Could not query Comet.ml for version number: {e}")
        print("Falling back to version 1")
        return 1


def init_experiment(config: dict[str, Any]) -> None:
    """Initialize Comet.ml experiment with the given configuration.

    Args:
        config (dict[str, Any]): Configuration dictionary containing Comet.ml settings
    """
    global _experiment
    _experiment = comet_ml.start(
        project_name=config["comet_project"],
        workspace=config.get("comet_workspace"),
        api_key=config.get(
            "comet_api_key"
        ),  # Can also be set via env var COMET_API_KEY
    )
    run_name = config.get("run_name", None)
    if run_name is None:
        # Get next version number from Comet.ml
        if config.get("use_versioning", True):
            version = get_next_version_number(
                project_name=config["comet_project"],
                workspace=config.get("comet_workspace"),
                api_key=config.get("comet_api_key"),
            )
            run_name = f"ImVs-{version}"
        else:
            # Fallback to date-time as default run name
            run_name = datetime.now().strftime("%m%d_%H:%M:%S")

    print(f"Run name: {run_name}")
    _experiment.set_name(run_name)
    _experiment.add_tags(config.get("tags", []))
    _experiment.log_parameters(config)


def log_training_metrics(
    loss: float,
    lr: float,
    mu: float,
    logvar: float,
    mae: float,
    mse: float,
    step: int | None = None,
) -> None:
    """Log training metrics to Comet.ml.

    Args:
        loss (float): Training loss
        lr (float): Learning rate
        mu (float): Mean of predicted values
        logvar (float): Log variance
        mae (float): Mean absolute error
        mse (float): Mean squared error
        step (int | None): Step number for logging
    """
    if _experiment is None:
        return

    metrics = {
        "train/loss": loss,
        "train/lr": lr,
        "train/µ": mu,
        "train/logvar": logvar,
        "train/mae": mae,
        "train/mse": mse,
    }
    _experiment.log_metrics(metrics, step=step)


def log_validation_metrics(
    val_loss: float,
    val_mae: float,
    val_mse: float,
    latent_rankme: float,
    epoch: int,
    variance_mae_correlation: float | None = None,
) -> None:
    """Log validation metrics to Comet.ml.

    Args:
        val_loss (float): Validation loss
        val_mae (float): Validation MAE
        val_mse (float): Validation MSE
        latent_rankme (float): RankMe metric for latent representations
        epoch (int): Current epoch number
        variance_mae_correlation (Optional[float]): Pearson correlation between predicted variances and MAEs per channel
    """
    if _experiment is None:
        return

    metrics = {
        "val/loss": val_loss,
        "val/mae": val_mae,
        "val/mse": val_mse,
        "val/latent_RankMe": latent_rankme,
    }
    if variance_mae_correlation is not None:
        metrics["val/variance_mae_correlation"] = variance_mae_correlation
    _experiment.log_metrics(metrics, epoch=epoch)

def log_finetuning_training_metrics(
    lr: float,
    loss: float,
    preds: np.array,
    y: np.array,
    label_encoder,
    epoch: int,
) -> None:
    """Log validation metrics to Comet.ml.

    Args:
        loss (float): Validation loss
        macroF1 (float): Validation macro-f1 score
        epoch (int): Current epoch number
        variance_mae_correlation (Optional[float]): Pearson correlation between predicted variances and MAEs per channel
    """
    if _experiment is None:
        return

    num_classes = len(label_encoder.get_dict())

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for truth, pred in zip(y, preds):
        confusion_matrix[truth, pred] += 1

    TP = np.diag(confusion_matrix)
    FP = confusion_matrix.sum(axis=0) - TP
    FN = confusion_matrix.sum(axis=1) - TP
    
    precision = TP / (TP+FP+1e-10)
    recall = TP / (TP+FN+1e-10)

    F1score = 2*precision*recall / (precision+recall+1e-10)
    macroF1 = np.mean(F1score)


    metrics = {
        "train/lr": lr,
        "train/loss": loss,
        "train/macroF1": macroF1,
        "train/precission": precision,
        "train/recall": recall
    }
    _experiment.log_metrics(metrics, epoch=epoch)
    

    return metrics

def plot_roc_curve(n_classes, y_test, y_score, label_encoder):
    # Normalize inputs


    y_score = torch.cat(y_score).to(torch.float32).detach().cpu().numpy()
    
    y_test = np.asarray(y_test)

    # Collapse any singleton channel dim like (n,1,c) -> (n,c) or (n,c,1) -> (n,c)
    if y_score.ndim == 3:
        if y_score.shape[1] == 1:
            y_score = y_score.reshape((y_score.shape[0], y_score.shape[2]))
        elif y_score.shape[2] == 1:
            y_score = y_score.reshape((y_score.shape[0], y_score.shape[1]))
        else:
            # try generic squeeze (will drop any axis==1)
            y_score = np.squeeze(y_score)


    # Debug-friendly shapes
   

    # Binarize true labels
    y_onehot_test = label_encoder.binarize(y_test)
    y_onehot_test = np.asarray(y_onehot_test)
  

    # Ensure scores are (n_samples, n_classes)
    if y_score.ndim == 2 and y_score.shape[0] == n_classes and y_score.shape[1] != n_classes:
        # common mistake: scores provided as (n_classes, n_samples)
        y_score = y_score.T
        print("Transposed y_score to shape:", y_score.shape)

    if y_score.shape != y_onehot_test.shape:
        raise ValueError(
            f"Shape mismatch: y_onehot_test {y_onehot_test.shape} vs y_score {y_score.shape}. "
            "Both should be (n_samples, n_classes)."
        )

    fpr, tpr, roc_auc = {}, {}, {}

    # Per-class ROC
    for i in range(n_classes):
        # If a class has only one label in y_true, roc_curve will error — guard for that.
        if np.unique(y_onehot_test[:, i]).size == 1:
            # skip or set NaNs
            fpr[i], tpr[i], roc_auc[i] = np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")
            continue
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro and macro
    # micro
    try:
        fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    except Exception:
        fpr["micro"], tpr["micro"], roc_auc["micro"] = np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    # macro: interpolate per-class TPRs on a common grid and average (skip classes with NaN)
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    mean_tpr = np.zeros_like(fpr_grid)
    valid_classes = 0
    for i in range(n_classes):
        if np.isnan(roc_auc.get(i, np.nan)):
            continue
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])
        valid_classes += 1
    if valid_classes > 0:
        mean_tpr /= valid_classes
        fpr["macro"] = fpr_grid
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    else:
        fpr["macro"], tpr["macro"], roc_auc["macro"] = np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    if "micro" in fpr:
        ax.plot(fpr["micro"], tpr["micro"], label=f"micro-average ROC (AUC = {roc_auc.get('micro', float('nan')):.2f})",
                color="deeppink", linestyle=":", linewidth=2)
    if "macro" in fpr:
        ax.plot(fpr["macro"], tpr["macro"], label=f"macro-average ROC (AUC = {roc_auc.get('macro', float('nan')):.2f})",
                color="navy", linestyle=":", linewidth=2)

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "olive", "purple", "teal", "gold"])
    if hasattr(label_encoder, "get_dict"):
        class_dict = label_encoder.get_dict()
        class_names = [None] * n_classes
        for cls_name, cls_idx in class_dict.items():
            if 0 <= cls_idx < n_classes:
                class_names[cls_idx] = str(cls_name)
        class_names = [name if name is not None else str(i) for i, name in enumerate(class_names)]
    else:
        class_names = [str(i) for i in range(n_classes)]

    for class_id, color in zip(range(n_classes), colors):
        if np.isnan(roc_auc.get(class_id, np.nan)):
            continue
        ax.plot(
            fpr[class_id],
            tpr[class_id],
            color=color,
            linewidth=1.5,
            label=f"ROC for {class_names[class_id]} (AUC = {roc_auc[class_id]:.2f})",
        )

    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC (One-vs-Rest multiclass)")
    ax.legend(loc="lower right")
    return fig

def log_finetuning_validation_metrics(
    loss: float,
    logits: np.array,
    preds: np.array,
    y: np.array,
    saliency_data: list,
    current_saliency_config,
    crop_size: int,
    label_encoder,
    epoch: int,
) -> None:
    """Log validation metrics to Comet.ml.

    Args:
        loss (float): Validation loss
        macroF1 (float): Validation macro-f1 score
        epoch (int): Current epoch number
        variance_mae_correlation (Optional[float]): Pearson correlation between predicted variances and MAEs per channel
    """
    if _experiment is None:
        return

    classes = label_encoder.get_dict()
    num_classes = len(classes)

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for truth, pred in zip(y, preds):
        confusion_matrix[truth, pred] += 1

    TP = np.diag(confusion_matrix)
    FP = confusion_matrix.sum(axis=0) - TP
    FN = confusion_matrix.sum(axis=1) - TP
    
    precision = TP / (TP+FP+1e-10)
    recall = TP / (TP+FN+1e-10)

    F1score = 2*precision*recall / (precision+recall+1e-10)
    macroF1 = np.mean(F1score)


    metrics = {
        "val/loss": loss,
        "val/macroF1": macroF1,
        "val/precission": precision,
        "val/recall": recall
    }

    roc = plot_roc_curve(num_classes, y, logits, label_encoder)
    
    _experiment.log_metrics(metrics, epoch=epoch)
    _experiment.log_confusion_matrix(y_true=y, y_predicted=preds)
    _experiment.log_figure(figure_name="roc", figure=roc)

    channel_groups = current_saliency_config.get("channel_groups", {})
    markers_names_map = current_saliency_config.get("markers_names_map", {})

    for full_img, crop_coords, weights in saliency_data:
        log_attention_saliency_imc(
            full_img,
            crop_coords,
            weights,
            crop_size,
            channel_groups,
            markers_names_map=markers_names_map,
            epoch=epoch,
            
        )
    return metrics

def log_validation_images(
    fig: plt.Figure,
    panel_idx: int,
    img_path: str,
    epoch: int,
    masked_channels_names: str,
    img_idx: int,
) -> None:
    """Log validation reconstruction images to Comet.ml.

    Args:
        fig (plt.Figure): Matplotlib figure to log
        panel_idx (int): Panel index
        img_path (str): Path to the image
        epoch (int): Current epoch number
        masked_channels_names (str): Names of masked channels
        img_idx (int): Index of the image in the batch
    """
    if _experiment is None:
        return

    # Convert figure to image
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)

    _experiment.log_image(
        img,
        name=f"val/reconstructions_panel-{panel_idx}_epoch-{epoch + 1}_img-{img_idx}",
        step=epoch,
        metadata={
            "panel_idx": panel_idx,
            "img_path": img_path,
            "masked_channels": masked_channels_names,
        },
    )

    buf.close()


def get_run_name() -> str:
    """Get the current Comet.ml experiment name.

    Returns:
        str : Current experiment name or "unknown" if no experiment is active
    """
    return _experiment.get_name() if _experiment else "unknown"


def finish_experiment() -> None:
    """Finish the current Comet.ml experiment."""
    global _experiment
    if _experiment is not None:
        _experiment.end()
        _experiment = None
