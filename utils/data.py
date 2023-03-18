#!/usr/bin/env python3


from dataclasses import dataclass
from pathlib import Path
from typing import Union

from PIL import Image
import jax
import jax.numpy as jnp
import jax.random as jran
import numpy as np
import tensorflow as tf
Dataset = tf.data.Dataset


def to_unit_cube_2d(xys: jax.Array, W: int, H: int):
    "Normalizes coordinate (x, y) into range [0, 1], where 0<=x<W, 0<=y<H"
    uvs = xys / jnp.asarray([[W-1, H-1]])
    return uvs


@dataclass(frozen=True)
class ImageData:
    H: int
    W: int
    idcs: jax.Array  # int:[H*W, 2]  original integer coordinates in range [0, W] for x and [0, H] for y
    uvs: jax.Array  # float:[H*W, 2] normalized coordinates in range [0, 1]
    rgbs: jax.Array  # float:[H*W, 3] normalized rgb values in range [0, 1]


def make_image_data(
        image: Union[np.ndarray, Image.Image, Path, str],
        use_white_bg: bool = True,
    ) -> ImageData:
    if isinstance(image, Image.Image):
        image = jnp.asarray(image)
    elif isinstance(image, (Path, str)):
        image = jnp.asarray(Image.open(image))
    elif isinstance(image, np.ndarray):
        image = jnp.asarray(image)
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

    x, y = jnp.meshgrid(jnp.arange(W), jnp.arange(H))
    x, y = x.reshape(-1, 1), y.reshape(-1, 1)
    idcs = jnp.concatenate([x, y], axis=-1)
    uvs = to_unit_cube_2d(idcs, W, H)

    return ImageData(
        H=H,
        W=W,
        idcs=jnp.asarray(idcs),
        uvs=jnp.asarray(uvs),
        rgbs=jnp.asarray(rgbs),
    )


def make_permutation_dataset(
        key: jran.KeyArray,
        size: int,
        shuffle: True,
    ) -> Dataset:
    if shuffle:
        perm = jran.permutation(key, size)
    else:
        perm = jnp.arange(size)

    return Dataset.from_tensor_slices(perm)


if __name__ == "__main__":
    from utils.common import tqdm_format
    from tqdm import tqdm

    imdata = make_image_data("./h.jpg", use_white_bg=True)
    K, key = jran.split(jran.PRNGKey(0xabcdef), 2)
    ds = make_permutation_dataset(key, imdata.H*imdata.W, shuffle=True)
    print(ds.element_spec)
    # _NumpyIterator does not support len()
    print(len(ds.batch(3).as_numpy_iterator()))
    for x in tqdm(ds.batch(4).as_numpy_iterator(), bar_format=tqdm_format):
        pass
