#!/usr/bin/env python3


from pathlib import Path
from typing import Union
from dataclasses import dataclass

from PIL import Image
import numpy as np
import tensorflow as tf
Dataset = tf.data.Dataset


def to_unit_cube_2d(xys: np.ndarray, W: int, H: int):
    "Normalizes coordinate (x, y) into range [0, 1], where 0<=x<W, 0<=y<H"
    uvs = xys / np.asarray([[W-1, H-1]])
    return uvs


@dataclass(frozen=True)
class SampledPixelsDatasetInputs:
    H: int
    W: int
    idcs: np.ndarray
    rgbs: np.ndarray


def make_sampled_pixels_dataset_inputs(
        image: Union[np.ndarray, Image.Image, Path, str],
        use_white_bg: bool = True,
    ):
    if isinstance(image, Image.Image):
        image = np.asarray(image)
    elif isinstance(image, (Path, str)):
        image = np.asarray(Image.open(image))
    elif isinstance(image, np.ndarray):
        image = image
    else:
        raise ValueError("Unexpected image source type '{}'".format(type(image)))

    H, W, C = image.shape
    rgbs = image.reshape(H*W, C) / 255
    if C == 4:
        alpha = rgbs[..., -1:]
        rgbs = rgbs[..., :3]
        if use_white_bg:
            rgbs = rgbs * alpha
            rgbs += 1 - alpha

    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x, y = x.reshape(-1, 1), y.reshape(-1, 1)
    idcs = np.concatenate([x, y], axis=-1)
    idcs = to_unit_cube_2d(idcs, W, H)

    return SampledPixelsDatasetInputs(
        H=H,
        W=W,
        idcs=idcs,
        rgbs=rgbs
    )


def make_sampled_pixels_dataset(
        inputs: SampledPixelsDatasetInputs,
        shuffle: True,
    ):
    size = inputs.H * inputs.W
    if shuffle:
        perm = np.random.permutation(size)
    else:
        perm = np.arange(size)

    idcs = Dataset.from_tensor_slices(inputs.idcs[perm])
    rgb = Dataset.from_tensor_slices(inputs.rgbs[perm])

    ds = Dataset.zip((idcs, rgb))

    return ds


if __name__ == "__main__":
    from utils.common import tqdm_format
    from tqdm import tqdm

    ds_in = make_sampled_pixels_dataset_inputs("./h.jpg", use_white_bg=True)
    ds = make_sampled_pixels_dataset(ds_in, shuffle=True)
    print(ds.element_spec)
    # _NumpyIterator does not support len()
    print(len(ds.batch(3).as_numpy_iterator()))
    for x in tqdm(ds.batch(4).as_numpy_iterator(), bar_format=tqdm_format):
        pass
