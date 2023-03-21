#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Literal, Tuple, Union

from PIL import Image
import chex
from flax.struct import dataclass
import jax
import jax.numpy as jnp
import jax.random as jran
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils.common import mkValueError, tqdm_format
from utils.types import AABB, PinholeCamera, RayMarchingOptions, RigidTransformation
Dataset = tf.data.Dataset


@dataclass
class ImageMetadata:
    H: int
    W: int
    xys: jax.Array  # int,[H*W, 2]: original integer coordinates in range [0, W] for x and [0, H] for y
    uvs: jax.Array  # float,[H*W, 2]: normalized coordinates in range [0, 1]
    rgbs: jax.Array  # float,[H*W, 3]: normalized rgb values in range [0, 1]


@dataclass
class ViewMetadata:
    H: int
    W: int
    xys: jax.Array  # int,[H*W, 2]: original integer coordinates in range [0, W] for x and [0, H] for y
    rgbs: jax.Array  # float,[H*W, 3]: normalized rgb values in range [0, 1]
    transform: RigidTransformation
    file: Path


@dataclass
class SceneMetadata:
    aabb: AABB
    # TODO:
    #   Make this `camera`'s H, W configurable and resize loaded images accordingly (specified H,W
    #   must have same aspect ratio as the loaded images).
    #   For now it's just read from the dataset.
    camera: PinholeCamera  # the camera model used to render this scene
    all_xys: jax.Array  # int,[n_pixels, 2], flattened xy coordinates from loaded images
    all_rgbs: jax.Array  # float,[n_pixels, 3+9+3], flattened rgb values from loaded images
    all_transforms: jax.Array  # float,[n_views, 9+3] each row comprises of R(flattened,9), T(3), from loaded images


def to_unit_cube_2d(xys: jax.Array, W: int, H: int):
    "Normalizes coordinate (x, y) into range [0, 1], where 0<=x<W, 0<=y<H"
    uvs = xys / jnp.asarray([[W-1, H-1]])
    return uvs


def psnr(lhs: jax.Array, rhs: jax.Array):
    chex.assert_type([lhs, rhs], [jnp.uint8, jnp.uint8])
    mse = ((lhs.astype(float) - rhs.astype(float)) ** 2).mean()
    if mse == 0:
        return 100
    else:
        return float(20 * jnp.log10(255 / jnp.sqrt(mse)))


@jax.jit
def set_pixels(imgarr: jax.Array, xys: jax.Array, selected: jax.Array, preds: jax.Array) -> jax.Array:
    H, W, C = imgarr.shape
    interm = imgarr.reshape(H*W, C)
    idcs = xys[selected, 1] * W + xys[selected, 0]
    interm = interm.at[idcs].set(jnp.clip(preds * 255, 0, 255).astype(jnp.uint8))
    return interm.reshape(H, W, C)


def blend_alpha_channel(imgarr, use_white_bg: bool):
    chex.assert_shape(imgarr, [..., 4])
    rgbs, alpha = imgarr[..., :-1], imgarr[..., -1:]
    if use_white_bg:
        rgbs = rgbs * alpha + (1 - alpha)
    return rgbs


def get_xyrgbs(imgarr: jax.Array, use_white_bg: bool) -> Tuple[jax.Array, jax.Array]:
    assert imgarr.dtype == jnp.uint8
    H, W, C = imgarr.shape

    x, y = jnp.meshgrid(jnp.arange(W), jnp.arange(H))
    x, y = x.reshape(-1, 1), y.reshape(-1, 1)
    xys = jnp.concatenate([x, y], axis=-1)

    flattened = imgarr.reshape(H*W, C) / 255
    if C == 3:
        rgbs = flattened
        return xys, rgbs
    elif C == 4:
        rgbs = blend_alpha_channel(flattened, use_white_bg)
        return xys, rgbs
    else:
        raise mkValueError(
            desc="number of image channels",
            value=C,
            type=Literal[3, 4],
        )


_ImageSourceType = Union[jax.Array, np.ndarray, Image.Image, Path, str],
def make_image_metadata(
        image: _ImageSourceType,
        use_white_bg: bool = True,
    ) -> ImageMetadata:
    if isinstance(image, jax.Array):
        pass
    elif isinstance(image, Image.Image):
        image = jnp.asarray(image)
    elif isinstance(image, (Path, str)):
        image = jnp.asarray(Image.open(image))
    elif isinstance(image, np.ndarray):
        image = jnp.asarray(image)
    else:
        raise mkValueError(
            desc="image source type",
            value=image,
            type=_ImageSourceType,
        )

    xys, rgbs = get_xyrgbs(image, use_white_bg=use_white_bg)

    H, W = image.shape[:2]
    uvs = to_unit_cube_2d(xys, W, H)

    return ImageMetadata(
        H=H,
        W=W,
        xys=jnp.asarray(xys),
        uvs=jnp.asarray(uvs),
        rgbs=jnp.asarray(rgbs),
    )


def make_view(
        image_path: Union[Path, str],
        transform_4x4: jax.Array,
        use_white_bg: bool,
    ) -> ViewMetadata:
    image_path = Path(image_path)
    image = jnp.asarray(Image.open(image_path))
    xys, rgbs = get_xyrgbs(image, use_white_bg=use_white_bg)
    H, W = image.shape[:2]
    return ViewMetadata(
        H=H,
        W=W,
        xys=xys,
        rgbs=rgbs,
        transform=RigidTransformation(
            rotation=transform_4x4[:3, :3],
            translation=transform_4x4[:3, 3],
        ),
        file=image_path.absolute(),
    )


def make_nerf_synthetic_scene_metadata(
        rootdir: Union[Path, str],
        split: Literal["train", "val", "test"],
        aabb: AABB,
        near: float,
        far: float,
        use_white_bg: bool=True,
    ) -> Tuple[SceneMetadata, list[ViewMetadata]]:
    rootdir = Path(rootdir)

    transforms_path = rootdir.joinpath("transforms_{}.json".format(split))
    transforms = json.load(open(transforms_path))

    views = list(
        map(
            lambda frame: make_view(
                    image_path=rootdir.joinpath(frame["file_path"] + ".png"),
                    transform_4x4=jnp.asarray(frame["transform_matrix"]),
                    use_white_bg=use_white_bg,
                ),
            tqdm(transforms["frames"], desc="loading views (split={})".format(split), bar_format=tqdm_format)
        )
    )

    # shared camera model
    W, H = views[0].W, views[0].H
    fovx = transforms["camera_angle_x"]
    focal = float(.5 * W / np.tan(fovx / 2))
    camera = PinholeCamera(
        W=W,
        H=H,
        near=near,
        far=far,
        focal=focal,
    )

    # flatten
    # int,[n_pixels, 2]
    all_xys = jnp.concatenate(
        list(map(lambda view: view.xys, views)),
        axis=0,
    )

    # float,[n_pixels, 3]
    all_rgbs = jnp.concatenate(
        list(map(lambda view: view.rgbs, views)),
        axis=0,
    )

    # float,[n_pixels, 9]
    all_Rs = jnp.concatenate(
        list(map(lambda view: view.transform.rotation.reshape(-1, 9), views)),
        axis=0,
    )
    # float,[n_pixels, 3]
    all_Ts = jnp.concatenate(
        list(map(lambda view: view.transform.translation.reshape(-1, 3), views)),
        axis=0,
    )
    # float,[n_views, 3+9+3]
    all_transforms = jnp.concatenate([all_Rs, all_Ts], axis=-1)

    scene = SceneMetadata(
        aabb=aabb,
        camera=camera,
        all_xys=all_xys,
        all_rgbs=all_rgbs,
        all_transforms=all_transforms,
    )

    return scene, views


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


def main():
    sce = make_nerf_synthetic_scene_metadata(
        rootdir="data/nerf/nerf_synthetic/lego",
        split="train",
        aabb=[[-1, 1], [-1, 1], [-1, 1]],
        near=2,
        far=6,
        use_white_bg=True,
    )
    print(sce.all_xys.shape)
    print(sce.all_rgbs.shape)
    print(sce.all_transforms.shape)

    # imdata = make_image_metadata("./h.jpg", use_white_bg=True)
    # K, key = jran.split(jran.PRNGKey(0xabcdef), 2)
    # ds = make_permutation_dataset(key, imdata.H*imdata.W, shuffle=True)
    # print(ds.element_spec)
    # # _NumpyIterator does not support len()
    # print(len(ds.batch(3).as_numpy_iterator()))
    # for x in tqdm(ds.batch(4).as_numpy_iterator(), bar_format=tqdm_format):
    #     pass


if __name__ == "__main__":
    main()
