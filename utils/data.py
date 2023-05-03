import json
import math
from pathlib import Path
from typing import List, Literal, Sequence, Tuple, Union

from PIL import Image
import chex
import ffmpeg
import imageio
import jax
import jax.numpy as jnp
import jax.random as jran
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils.common import jit_jaxfn_with, mkValueError, tqdm_format
from utils.types import (
    ImageMetadata,
    PinholeCamera,
    RGBColor,
    RigidTransformation,
    SceneMetadata,
    ViewMetadata,
)
Dataset = tf.data.Dataset


def to_cpu(array: jnp.DeviceArray) -> jnp.DeviceArray:
    return jax.device_put(array, device=jax.devices("cpu")[0])


@jax.jit
def f32_to_u8(img: jax.Array) -> jax.Array:
    return jnp.clip(img * 255, 0, 255).astype(jnp.uint8)


@jax.jit
def mono_to_rgb(img: jax.Array) -> jax.Array:
    return jnp.tile(img[..., None], (1, 1, 3))


def video_to_images(
    video_in: Path,
    images_dir: Path,
    fmt: str="%03d.png",
    fps: int=3,
):
    video_in, images_dir = Path(video_in), Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    (ffmpeg.input(video_in)
        .output(
            images_dir.joinpath(fmt).as_posix(),
            r=fps,
            pix_fmt="rgb24",  # colmap only supports 8-bit color depth
        )
        .run(
            capture_stdout=False,
            capture_stderr=False,
        )
    )


def to_unit_cube_2d(xys: jax.Array, W: int, H: int):
    "Normalizes coordinate (x, y) into range [0, 1], where 0<=x<W, 0<=y<H"
    uvs = xys / jnp.asarray([[W-1, H-1]])
    return uvs


@jit_jaxfn_with(static_argnames=["H", "W", "vertical", "gap", "gap_color"])
def side_by_side(
    lhs: jax.Array,
    rhs: jax.Array,
    H: int=None,
    W: int=None,
    vertical: bool=False,
    gap: int=5,
    gap_color: RGBColorU8=(0xab, 0xcd, 0xef),
) -> jax.Array:
    chex.assert_not_both_none(H, W)
    chex.assert_scalar_non_negative(vertical)
    chex.assert_type([lhs, rhs], jnp.uint8)
    if len(lhs.shape) == 2 or lhs.shape[-1] == 1:
        lhs = mono_to_rgb(lhs)
    if len(rhs.shape) == 2 or rhs.shape[-1] == 1:
        rhs = mono_to_rgb(rhs)
    if vertical:
        chex.assert_axis_dimension(lhs, 1, W)
        chex.assert_axis_dimension(rhs, 1, W)
    else:
        chex.assert_axis_dimension(lhs, 0, H)
        chex.assert_axis_dimension(rhs, 0, H)
    concat_axis = 0 if vertical else 1
    if gap > 0:
        gap_color = jnp.asarray(gap_color, dtype=jnp.uint8)
        gap = jnp.broadcast_to(gap_color, (gap, W, 3) if vertical else (H, gap, 3))
        return jnp.concatenate([lhs, gap, rhs], axis=concat_axis)
    else:
        return jnp.concatenate([lhs, rhs], axis=concat_axis)


@jit_jaxfn_with(static_argnames=["width", "color"])
def add_border(
    img: jax.Array,
    width: int=5,
    color: RGBColorU8=(0xfe, 0xdc, 0xba)
) -> jax.Array:
    chex.assert_rank(img, 3)
    chex.assert_axis_dimension(img, -1, 3)
    chex.assert_scalar_non_negative(width)
    chex.assert_type(img, jnp.uint8)
    color = jnp.asarray(color, dtype=jnp.uint8)
    H, W = img.shape[:2]
    leftright = jnp.broadcast_to(color, (H, width, 3))
    img = jnp.concatenate([leftright, img, leftright], axis=1)
    topbottom = jnp.broadcast_to(color, (width, W+2*width, 3))
    img = jnp.concatenate([topbottom, img, topbottom], axis=0)
    return img


@jax.jit
def linear_to_db(val: float, maxval: float):
    return 20 * jnp.log10(jnp.sqrt(maxval / val))


@jax.jit
def psnr(lhs: jax.Array, rhs: jax.Array):
    chex.assert_type([lhs, rhs], jnp.uint8)
    mse = ((lhs.astype(float) - rhs.astype(float)) ** 2).mean()
    return jnp.clip(20 * jnp.log10(255 / jnp.sqrt(mse + 1e-15)), 0, 100)


def write_video(dest: Path, images: Sequence, *, fps: int=24, loop: int=3):
    images = list(images) * loop
    assert len(images) > 0, "cannot write empty video"
    video_writer = imageio.get_writer(dest, mode="I", fps=fps)
    for im in tqdm(images, desc="writing video to {}".format(dest.as_posix()), bar_format=tqdm_format):
        video_writer.append_data(np.asarray(im))


def cascades_from_bound(bound: float) -> int:
    return max(1, int(1 + math.ceil(math.log2(bound))))


@jax.jit
def set_pixels(imgarr: jax.Array, xys: jax.Array, selected: jax.Array, preds: jax.Array) -> jax.Array:
    H, W = imgarr.shape[:2]
    if len(imgarr.shape) == 3:
        interm = imgarr.reshape(H*W, -1)
    else:
        interm = imgarr.ravel()
    idcs = xys[selected, 1] * W + xys[selected, 0]
    interm = interm.at[idcs].set(jnp.clip(preds * 255, 0, 255).astype(jnp.uint8))
    if len(imgarr.shape) == 3:
        return interm.reshape(H, W, -1)
    else:
        return interm.reshape(H, W)


# this does not give better results than a plain `jnp.unifrom(0, 1)` for supervising alpha via color
# blending.
# TODO: for the "mic" scene, using white as background color actually performs better than using
# random background on depth supervision, why?
def alternate_color(KEY: jran.KeyArray, bg: RGBColor, n_pixels: int, dtype) -> jax.Array:
    KEY, key_randcolor, key_choice = jran.split(KEY, 3)
    alternate_options = (
        (0., 0., 0.),  # black
        (1., 0., 0.),  # red
        (0., 1., 0.),  # green
        (0., 0., 1.),  # blue
        (1., 1., 0.),  # yellow
        (1., 0., 1.),  # magenta
        (0., 1., 1.),  # cyan
        (1., 1., 1.),  # white
        jran.uniform(key_randcolor, (3,), dtype, minval=0., maxval=1.),
        bg,
    )
    alternate_options = jnp.asarray(alternate_options, dtype=dtype)
    return jran.choice(key_choice, alternate_options, shape=(n_pixels,))


def blend_rgba_image_array(imgarr, bg: RGBColor):
    """
    Blend the given background color according to the given alpha channel from `imgarr`.
    WARN: this function SHOULD NOT be used for blending background colors into volume-rendered
          pixels because the colors of volume-rendered pixels already have the alpha channel
          factored-in.  To blend background for volume-rendered pixels, directly add the scaled
          background color.
          E.g.: `final_color = ray_accumulated_color + (1 - ray_opacity) * bg_color`
    """
    if isinstance(imgarr, Image.Image):
        imgarr = np.asarray(imgarr)
    chex.assert_shape(imgarr, [..., 4])
    chex.assert_type(imgarr, bg.dtype)
    rgbs, alpha = imgarr[..., :-1], imgarr[..., -1:]
    bg_color = jnp.asarray(bg)
    bg_color = jnp.broadcast_to(bg_color, rgbs.shape)
    if imgarr.dtype == jnp.uint8:
        rgbs, alpha = rgbs.astype(float) / 255, alpha.astype(float) / 255
        rgbs = rgbs * alpha + bg_color * (1 - alpha)
        rgbs = (rgbs * 255).astype(jnp.uint8)
    else:
        rgbs = rgbs * alpha + bg_color * (1 - alpha)
    return rgbs


def get_xyrgbas(imgarr: jax.Array) -> Tuple[jax.Array, jax.Array]:
    assert imgarr.dtype == jnp.uint8
    H, W, C = imgarr.shape

    x, y = jnp.meshgrid(jnp.arange(W), jnp.arange(H))
    x, y = x.reshape(-1, 1), y.reshape(-1, 1)
    xys = jnp.concatenate([x, y], axis=-1)

    flattened = imgarr.reshape(H*W, C) / 255
    if C == 3:
        # images without an alpha channel is equivalent to themselves with an all-opaque alpha
        # channel
        rgbas = jnp.concatenate([flattened, jnp.ones_like(flattened[:, :1])])
        return xys, rgbas
    elif C == 4:
        rgbas = flattened
        return xys, rgbas
    else:
        raise mkValueError(
            desc="number of image channels",
            value=C,
            type=Literal[3, 4],
        )


_ImageSourceType = Union[jax.Array, np.ndarray, Image.Image, Path, str],
def make_image_metadata(
    image: _ImageSourceType,
    bg: RGBColor,
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

    raise NotImplementedError(
        "function get_xyrgbs has been renamed to get_xyrgbas and this part has not been updated "
        "accordingly"
    )
    xys, rgbs = get_xyrgbs(image, bg=bg)

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
) -> ViewMetadata:
    image_path = Path(image_path)
    image = jnp.asarray(Image.open(image_path))
    xys, rgbas = get_xyrgbas(image)
    H, W = image.shape[:2]
    return ViewMetadata(
        H=H,
        W=W,
        xys=xys,
        rgbas=rgbas,
        transform=RigidTransformation(
            rotation=transform_4x4[:3, :3],
            translation=transform_4x4[:3, 3],
        ),
        file=image_path.absolute(),
    )


def make_nerf_synthetic_scene_metadata(
    rootdir: Union[Path, str],
    split: Literal["train", "val", "test"],
    scale: float,
) -> Tuple[SceneMetadata, List[ViewMetadata]]:
    rootdir = Path(rootdir)

    transforms_path = rootdir.joinpath("transforms_{}.json".format(split))
    transforms = json.load(open(transforms_path))

    views = list(
        map(
            lambda frame: make_view(
                    image_path=rootdir.joinpath(frame["file_path"] + ".png"),
                    transform_4x4=jnp.asarray(frame["transform_matrix"]),
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
        focal=focal,
    )

    # flatten
    # int,[n_pixels, 2]
    all_xys = jnp.concatenate(
        list(map(lambda view: view.xys, views)),
        axis=0,
    )

    # float,[n_pixels, 4]
    all_rgbas = jnp.concatenate(
        list(map(lambda view: view.rgbas, views)),
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
    all_Ts *= scale
    # float,[n_views, 3+9+3]
    all_transforms = jnp.concatenate([all_Rs, all_Ts], axis=-1)

    scene = SceneMetadata(
        camera=camera,
        all_xys=all_xys,
        all_rgbas=all_rgbas,
        all_transforms=all_transforms,
    )

    return scene, views


@jit_jaxfn_with(static_argnames=["size", "loop", "shuffle"])
def make_permutation(
    key: jran.KeyArray,
    size: int,
    loop: int=1,
    shuffle: bool=True,
) -> jax.Array:
    if shuffle:
        perm = jran.permutation(key, size * loop)
    else:
        perm = jnp.arange(size * loop)
    return perm % size


def main():
    scene, views = make_nerf_synthetic_scene_metadata(
        rootdir="data/nerf/nerf_synthetic/lego",
        split="train",
        scale=.8,
    )
    print(scene.all_xys.shape)
    print(scene.all_rgbas.shape)
    print(scene.all_transforms.shape)
    print(len(views))


if __name__ == "__main__":
    main()
