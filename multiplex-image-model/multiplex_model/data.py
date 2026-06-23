import os
import random
from glob import glob
from typing import Literal, List

import numpy as np
import tifffile
import torch
from cv2 import medianBlur
from skimage import filters
from torch.utils.data import Dataset, Sampler
from torchvision.transforms.functional import crop


class DatasetFromTIFF(Dataset):
    def __init__(
        self,
        panels_config: dict,
        split: str,
        marker_tokenizer: dict[str, int],
        machine: str = "local",
        subset=None,
        transform=None,
        use_preprocessing: bool = True,
        use_median_denoising: bool = False,
        use_butterworth_filter: bool = True,
        use_minmax_normalization: bool = False,
        use_clip_normalization: bool = True,
        global_upper_bound: float = 5.0,
        use_global_clip_limits: bool = False,
        file_extension: Literal["tiff", "npy"] = "tiff",
    ):
        """Dataset for loading multiplex images from multiple panels.

        Args:
            panels_config (dict): Configuration dictionary for panels.
            split (str): Name of the data split (e.g., 'train', 'val', 'test').
            marker_tokenizer (dict[str, int]): Tokenizer for marker names.
            transform (_type_, optional): Transform to be applied to the images. Defaults to None.
            use_preprocessing (bool, optional): Whether to use preprocessing. Defaults to True.
            use_median_denoising (bool, optional): Whether to use median denoising. Defaults to False.
            use_butterworth_filter (bool, optional): Whether to use Butterworth filter. Defaults to True.
            use_minmax_normalization (bool, optional): Whether to use min-max normalization. Defaults to True.
            use_clip_normalization (bool, optional): Whether to use clipping normalization. Defaults to False.
            global_upper_bound (float, optional): Global upper bound for clipping normalization if `clip_limits`
                is not provided in config. Defaults to 5.0.
            use_global_clip_limits (bool, optional): Whether to use global clip limits for all datasets. Defaults to False.
            file_extension (Literal['tiff', 'npy'], optional): File extension of the images. Defaults to 'tiff'.
        """
        assert "paths" in panels_config, (
            "Panels config must have 'paths' attribute with paths of splits of the data."
        )
        # assert split in panels_config["paths"], (
        #     f"Panels config must have '{split}' attribute with data path."
        # )
        assert "datasets" in panels_config, (
            "Panels config must have 'datasets' attribute with subdirectories."
        )
        assert "markers" in panels_config, (
            "Panels config must have 'markers' attribute with channel IDs."
        )

        self.channel_ids = {
            dataset: torch.tensor(
                [
                    marker_tokenizer[marker]
                    for marker in panels_config["markers"][dataset]
                ],
                dtype=torch.long,
            )
            for dataset in panels_config["datasets"]
        }
        
        if machine == "szary":
            img_path = panels_config["paths"]["szary"][split]
        elif machine == "local": 
            img_path = panels_config["paths"]["local"][split]

        self.imgs = []  # tuples of (img_path, dataset)
        for dataset in panels_config["datasets"]:
            if subset:
                
                if dataset == subset:
                    print(f"dataset: {dataset}")
                    tiffs = glob(os.path.join(img_path, dataset, "imgs", f"*.{file_extension}"))
                    #tiffs = tiffs[:10]
                    self.imgs.extend([(tiff, dataset) for tiff in tiffs])
        
        if use_global_clip_limits:
            self.clip_limits = {}
        else:
            self.clip_limits = panels_config.get("clip_limits", {})
        self.global_upper_bound = global_upper_bound

        self.transform = transform
        self.use_denoising = use_median_denoising
        self.use_butterworth = use_butterworth_filter
        self.use_minmax_normalization = use_minmax_normalization
        self.use_clip_normalization = use_clip_normalization
        self.use_preprocessing = use_preprocessing
        self.file_extension = file_extension
        self.read_file_func = (
            tifffile.imread if self.file_extension == "tiff" else np.load
        )

    @staticmethod
    def preprocess(img):
        return np.arcsinh(img / 5.0)

    @staticmethod
    def denoise(img):
        denoised_channels = [
            medianBlur(img[i].astype("float32"), 3) for i in range(img.shape[0])
        ]
        return np.stack(denoised_channels)

    @staticmethod
    def butterworth(img):
        filtered_channels = [
            filters.butterworth(img[i], cutoff_frequency_ratio=0.2, high_pass=False)
            for i in range(img.shape[0])
        ]
        return np.stack(filtered_channels)

    @staticmethod
    def norm_minmax(img):
        min_val = np.min(img, axis=(1, 2), keepdims=True)
        max_val = np.max(img, axis=(1, 2), keepdims=True)
        scaled_img = np.where(
            max_val == min_val, img, (img - min_val) / (max_val - min_val + 1e-8)
        )
        scaled_img = np.clip(scaled_img, 0, 1)
        return scaled_img

    def norm_clip(self, img, dataset):
        """Normalize image channels to [0, 1] range using global (per dataset) clipping."""
        upper_bound = self.clip_limits.get(dataset, self.global_upper_bound)
        img = np.clip(img, 0, upper_bound) / upper_bound
        return img

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, dataset = self.imgs[idx]
        channel_ids = self.channel_ids[dataset]

        img = self.read_file_func(img_path)
        if self.use_preprocessing:
            img = self.preprocess(img)

        # if self.transform:
        #     img, coords = self.transform(torch.tensor(img))
        #     img = img.numpy()

        # if self.use_butterworth:
        #     img = self.butterworth(img)

        # if self.use_denoising:
        #     img = self.denoise(img)

        # if self.use_clip_normalization:
        #     img = self.norm_clip(img, dataset)

        # elif self.use_minmax_normalization:
        #     img = self.norm_minmax(img)

        if self.transform:
            crops, coords = self.transform(torch.tensor(img))

        if self.use_butterworth:
            crops = [self.butterworth(crop) for crop in crops]

        if self.use_denoising:
            crops = [self.denoise(crop) for crop in crops]

        if self.use_clip_normalization:
            crops = [self.norm_clip(crop, dataset) for crop in crops]

        elif self.use_minmax_normalization:
            crops = [self.norm_minmax(crop) for crop in crops]
        crops = torch.stack([torch.from_numpy(crop) for crop in crops])
        dataset = [dataset] * len(crops)
        img_path = [img_path] * len(crops)
    #     return torch.tensor(img), channel_ids, dataset, img_path
    
        if self.transform:
            return crops, coords, channel_ids, dataset, img_path, img
        return crops, channel_ids, dataset, img_path, img
        



class PanelBatchSampler(Sampler):
    """Sampler that yields batches of indices grouped by panels."""

    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group indices by panel
        self.panel_to_indices = {}
        for idx, (_, panel_idx) in enumerate(dataset.imgs):
            if panel_idx not in self.panel_to_indices:
                self.panel_to_indices[panel_idx] = []
            self.panel_to_indices[panel_idx].append(idx)

        # Convert to list of (panel, indices) pairs for easier random selection
        self.panels = list(self.panel_to_indices.keys())

        self.epoch_batches = []  # Store batches for an epoch
        self._generate_batches()  # Prepare the first epoch

    def _generate_batches(self):
        """Generate batches ensuring each sample is used exactly once per epoch."""
        self.epoch_batches = []  # Reset batches for the new epoch

        # Shuffle panels if needed
        if self.shuffle:
            random.shuffle(self.panels)

        for panel in self.panels:
            indices = self.panel_to_indices[panel]

            # Shuffle indices within the panel if needed
            if self.shuffle:
                random.shuffle(indices)

            # Split indices into batches of batch_size
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                self.epoch_batches.append(batch)

        # Shuffle the final batch order for diversity
        if self.shuffle:
            random.shuffle(self.epoch_batches)

    def __iter__(self):
        """Yield batches, ensuring all images are used exactly once per epoch."""
        for batch in self.epoch_batches:
            yield batch
        self._generate_batches()  # Prepare for next epoch

    def __len__(self):
        """Return number of batches per epoch."""
        return len(self.epoch_batches)


class TestCrop:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # Crop the image and mask
        h, w = img.shape[-2], img.shape[-1]
        top = (h - self.size) // 2
        left = (w - self.size) // 2
        img = crop(img, top, left, self.size, self.size)

        return img, (top,left)

class GridCrop:
    def __init__(self, crop_size: int, max_crops: int = 64, augmentation: int | bool = False):
        """
        Grid-based cropping with optional augmentation.
        
        Args:
            crop_size (int): Size of each crop.
            max_crops (int): Maximum number of crops to extract.
            augmentation (int | bool): 
                - False or 0: No augmentation (default)
                - True or 1: Single augmented version per crop
                - N (N>1): N augmented versions per crop
        """
        self.crop_size = crop_size
        self.max_crops = max_crops
        
        # Convert bool to int for uniform handling
        if isinstance(augmentation, bool):
            self.num_augmentations = int(augmentation)
        else:
            self.num_augmentations = int(augmentation)

    def _augment_crop(self, crop_img: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to a single crop.
        
        Args:
            crop_img (np.ndarray): Input crop of shape (C, H, W) or (H, W, C)
            
        Returns:
            np.ndarray: Augmented crop
        """
        aug_img = crop_img.copy()
        
        # Random horizontal flip
        if random.random() > 0.5:
            aug_img = np.fliplr(aug_img)
        
        # Random vertical flip
        if random.random() > 0.5:
            aug_img = np.flipud(aug_img)
        
        # Random intensity scaling per channel
        # Assuming crop_img is (C, H, W) based on the code
        if aug_img.ndim == 3:
            for c in range(aug_img.shape[0]):
                # Scale intensity by 0.8 to 1.2 to simulate staining variability
                scale = np.random.uniform(0.8, 1.2)
                aug_img[c] = np.clip(aug_img[c] * scale, 0, aug_img[c].max())
        
        return aug_img

    def _random_offset_crop(self, img: torch.Tensor, base_top: int, base_left: int) -> np.ndarray:
        """
        Crop with random offset to add spatial variability.
        
        Args:
            img (torch.Tensor): Full image tensor
            base_top (int): Base top coordinate
            base_left (int): Base left coordinate
            
        Returns:
            np.ndarray: Cropped region with random offset
        """
        h, w = img.shape[-2], img.shape[-1]
        offset_range = self.crop_size // 4
        
        # Random offset within ±crop_size//4
        offset_top = random.randint(-offset_range, offset_range)
        offset_left = random.randint(-offset_range, offset_range)
        
        # Clamp to valid bounds
        top = max(0, min(base_top + offset_top, h - self.crop_size))
        left = max(0, min(base_left + offset_left, w - self.crop_size))
        
        c = crop(img, top, left, self.crop_size, self.crop_size)
        return c.numpy()

    def __call__(self, img: torch.Tensor) -> tuple[List[np.ndarray], np.ndarray]:
        """
        Extract crops with optional augmentation.
        
        Returns:
            tuple: (crops, coordinates) where
                - crops: List of augmented crop arrays
                - coordinates: Array of (top, left) coordinates for each base crop
        """
        h, w = img.shape[-2], img.shape[-1]
        crops = []
        coordinates = [] # [((top, top+size), (left, left+size))] crop's span in x and y dim
        num_rows = h // self.crop_size
        num_cols = w // self.crop_size
        
        if self.max_crops and (num_rows * num_cols) > self.max_crops:
            print(f"Number of crops exceeded {self.max_crops}. Taking a central square...")
            # Find the largest square side (in number of crops) that fits in max_crops
            side_len = min(int(self.max_crops ** 0.5), num_rows, num_cols)
            
            # Center the square
            row_start = (num_rows - side_len) // 2
            col_start = (num_cols - side_len) // 2
            row_end = row_start + side_len
            col_end = col_start + side_len
        else:
            row_start, col_start = 0, 0
            row_end, col_end = num_rows, num_cols
            
        for i in range(row_start, row_end):
            for j in range(col_start, col_end):
                top = i * self.crop_size
                left = j * self.crop_size
                
                # Store base coordinates only once (not per augmentation)
                coordinates.append(((int(top), int(top) + self.crop_size),
                                    (int(left), int(left) + self.crop_size)))
                
                if self.num_augmentations == 0:
                    # No augmentation: single deterministic crop
                    c = crop(img, top, left, self.crop_size, self.crop_size)
                    crops.append(c.numpy())
                else:
                    # Generate num_augmentations versions of this crop
                    for _ in range(self.num_augmentations):
                        # Extract crop with random offset
                        aug_crop = self._random_offset_crop(img, top, left)
                        # Apply random flips and intensity scaling
                        aug_crop = self._augment_crop(aug_crop)
                        crops.append(aug_crop)
        
        num_base_crops = len(coordinates)
        num_total_crops = len(crops)
        augmentation_factor = num_total_crops // num_base_crops if num_base_crops > 0 else 0
        
        print(f"GridCrop: {num_total_crops} total crops ({num_base_crops} base × {augmentation_factor} factor), augmentation={self.num_augmentations}")
        
        # Sanity check: ensure crops and coordinates have consistent relationship
        if num_base_crops > 0 and num_total_crops != num_base_crops * augmentation_factor:
            raise RuntimeError(
                f"Crop count mismatch: {num_total_crops} crops != {num_base_crops} base_crops × {augmentation_factor} aug_factor"
            )
        
        return crops, np.array(coordinates)