#!/usr/bin/env python3

from pathlib import Path
from typing import Union

from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader


def image_fit_collate_fn(batch):
    uvs, colors = zip(*batch)
    return np.asarray(uvs), np.asarray(colors)


class SampledPixelsDataset(Dataset):
    image: np.ndarray  # [H, W, C]
    loop: int

    def __init__(
            self,
            image: Union[Image.Image, Path, np.ndarray],
            loop: int,
        ):
        super().__init__()

        if isinstance(image, Image.Image):
            self.image = np.asarray(image)
        elif isinstance(image, Path):
            self.image = np.asarray(Image.open(image))
        elif isinstance(image, np.ndarray):
            self.image = image
        else:
            raise ValueError("Unexpected image source type '{}'".format(type(image)))

        self.image = self.image[..., :3] * (self.image[..., -1:] / 255)

        self.loop = loop
        self.normalizer_fn = self.to_unit_cube

    def __len__(self) -> int:
        return self.loop * self.image.shape[0] * self.image.shape[1]

    def __getitem__(self, index):
        H, W = self.image.shape[:2]
        index %= H * W
        x, y = index % W, index // W
        return self.normalizer_fn(W, H, x, y), self.image[y, x] / 255

    @staticmethod
    def to_canonical_cube(W: int, H: int, x: int, y: int):
        """Normalizes coordinate (x, y) into range [-1, 1], where 0<=x<W, 0<=y<H"""
        # assertion disabled: slow for un-jitted functions
        # chex.assert_type([W, H, x, y], [int] * 4)
        u = x / (W - 1) * 2 - 1
        v = y / (H - 1) * 2 - 1
        return (u, v)

    @staticmethod
    def to_unit_cube(W: int, H: int, x: int, y: int):
        """Normalizes coordinate (x, y) into range [0, 1], where 0<=x<W, 0<=y<H"""
        # assertion disabled: slow for un-jitted functions
        # chex.assert_type([W, H, x, y], [int] * 4)
        u = x / (W - 1)
        v = y / (H - 1)
        return (u, v)


class ImageFitLoader(DataLoader):
    def __init__(
            self,
            dataset: SampledPixelsDataset,
            batch_size: int,
            num_workers: int,
            shuffle: bool=True,
            drop_last: bool=False,
        ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=image_fit_collate_fn,
        )
