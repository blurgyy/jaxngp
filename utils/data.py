import collections
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import functools
import json
import math
from pathlib import Path
from typing import List, Literal, Sequence, Tuple

from PIL import Image
import chex
import ffmpeg
import imageio
import jax
import jax.numpy as jnp
import jax.random as jran
import numpy as np
from tqdm import tqdm

from . import sfm
from .common import jit_jaxfn_with, mkValueError, tqdm_format
from .types import (
    ColmapMatcherType,
    ImageMetadata,
    PinholeCamera,
    RGBColor,
    RGBColorU8,
    RigidTransformation,
    SceneData,
    SceneMeta,
    TransformJsonFrame,
    TransformJsonNGP,
    TransformJsonNeRFSynthetic,
    ViewMetadata,
)


def to_cpu(array: jnp.DeviceArray) -> jnp.DeviceArray:
    return jax.device_put(array, device=jax.devices("cpu")[0])


@jax.jit
def f32_to_u8(img: jax.Array) -> jax.Array:
    return jnp.clip(jnp.round(img * 255), 0, 255).astype(jnp.uint8)


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


def qvec2rotmat(qvec):
    "copied from NVLabs/instant-ngp/scripts/colmap2nerf.py"
    return np.asarray([
        [
            1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
        ], [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ], [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
        ]
    ])
def rotmat(a, b):
    "copied from NVLabs/instant-ngp/scripts/colmap2nerf.py"
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))
def closest_point_2_lines(oa, da, ob, db): 
    """
    (copied from NVLabs/instant-ngp/scripts/colmap2nerf.py)
    returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    """
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom


def write_transforms_json(
    dataset_root_dir: Path,
    images_dir: Path,
    text_model_dir: Path,
    # given that the cameras' average distance to the origin is 4.0, what would the scene's bound be?
    bound: float,
):
    "adapted from NVLabs/instant-ngp/scripts/colmap2nerf.py"
    dataset_root_dir, images_dir, text_model_dir = (
        Path(dataset_root_dir),
        Path(images_dir),
        Path(text_model_dir),
    )
    rel_prefix = images_dir.relative_to(dataset_root_dir)

    camera = PinholeCamera.from_colmap_txt(text_model_dir.joinpath("cameras.txt"))

    images_txt = text_model_dir.joinpath("images.txt")
    images_lines = list(filter(lambda line: line[0] != "#", open(images_txt).readlines()))[::2]
    up = np.zeros(3)
    bottom_row = np.asarray((0, 0, 0, 1.0)).reshape(1, 4)
    frames: List[TransformJsonFrame] = []
    for line in images_lines:
        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        _, qw, qx, qy, qz, tx, ty, tz, _, name = line.strip().split()
        R = qvec2rotmat(tuple(map(float, (qw, qx, qy, qz))))
        T = np.asarray(tuple(map(float, (tx, ty, tz)))).reshape(3, 1)
        m = np.concatenate([R, T], axis=-1)
        m = np.concatenate([m, bottom_row], axis=0)
        c2w = np.linalg.inv(m)

        c2w[0:3,2] *= -1 # flip the y and z axis
        c2w[0:3,1] *= -1
        c2w = c2w[[1,0,2,3],:]
        c2w[2,:] *= -1 # flip whole world upside down
        up += c2w[0:3,1]

        frames.append(TransformJsonFrame(
            file_path=rel_prefix.joinpath(name).as_posix(),
            transform_matrix=c2w.tolist(),
        ))

    # reorient the scene to be easier to work with
    up = up / np.linalg.norm(up)
    print("up vector:", up, "->", [0, 0, 1])
    R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
    R = np.pad(R,[0,1])
    R[-1, -1] = 1

    for i, f in enumerate(frames):
        frames[i] = dataclasses.replace(f, transform_matrix=np.matmul(R, f.transform_matrix_numpy).tolist())

    # find a central point they are all looking at
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in frames:
        mf = f.transform_matrix_numpy[0:3,:]
        for g in frames:
            mg = g.transform_matrix_numpy[0:3,:]
            p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
            if w > 1e-5:
                totp += p*w
                totw += w
    if totw > 0.0:
        totp /= totw
    # the cameras are looking at totp
    print("the cameras are looking at:", totp, "->", [0, 0, 0])
    for i, f in enumerate(frames):
        new_m = f.transform_matrix_numpy
        new_m[0:3,3] -= totp
        frames[i] = dataclasses.replace(f, transform_matrix=new_m.tolist())

    avglen = 0.
    for f in frames:
        avglen += np.linalg.norm(f.transform_matrix_numpy[0:3,3])
    avglen /= len(frames)
    print("average camera distance from origin:", avglen, "->", 4.0)
    for i, f in enumerate(frames):
        # scale to "nerf sized"
        new_m = f.transform_matrix_numpy
        new_m[0:3, 3] *= 4.0 / avglen
        frames[i] = dataclasses.replace(f, transform_matrix=new_m.tolist())

    print("scene bound (i.e. half width of scene's aabb):", bound * 2)
    all_transform_json = TransformJsonNGP(
        frames=frames,
        fl_x=camera.fx,
        fl_y=camera.fy,
        cx=camera.cx,
        cy=camera.cy,
        w=camera.W,
        h=camera.H,
        aabb_scale=bound,
    )
    train_tj = dataclasses.replace(all_transform_json, frames=frames[:len(frames) // 2])
    val_tj = dataclasses.replace(all_transform_json, frames=frames[len(frames) // 2:len(frames) // 2 + len(frames) // 4])
    test_tj = dataclasses.replace(all_transform_json, frames=frames[len(frames) // 2 + len(frames) // 4:])
    train_tj.save(dataset_root_dir.joinpath("transforms_train.json"))
    val_tj.save(dataset_root_dir.joinpath("transforms_val.json"))
    test_tj.save(dataset_root_dir.joinpath("transforms_test.json"))
    return all_transform_json


def create_dataset_from_single_camera_image_collection(
    raw_images_dir: Path,
    dataset_root_dir: Path,
    matcher: ColmapMatcherType,
    bound: float,
):
    raw_images_dir, dataset_root_dir = Path(raw_images_dir), Path(dataset_root_dir)
    dataset_root_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = dataset_root_dir.joinpath("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    db_path = artifacts_dir.joinpath("colmap.db")
    sparse_reconstruction_dir = artifacts_dir.joinpath("sparse")
    undistorted_images_dir = dataset_root_dir.joinpath("images-undistorted")
    text_model_dir = artifacts_dir.joinpath("text")

    sfm.extract_features(images_dir=raw_images_dir, db_path=db_path)

    sfm.match_features(matcher=matcher, db_path=db_path)

    maps = sfm.sparse_reconstruction(
        images_dir=raw_images_dir,
        sparse_reconstruction_dir=sparse_reconstruction_dir,
        db_path=db_path,
    )
    if len(maps) == 0:
        raise RuntimeError("mapping with colmap failed")
    elif len(maps) > 1:
        raise RuntimeError("colmap reconstructed more than 1 maps")

    sfm.undistort(
        images_dir=raw_images_dir,
        sparse_reconstruction_dir=sparse_reconstruction_dir.joinpath("0"),
        undistorted_images_dir=undistorted_images_dir,
    )

    sfm.export_text_format_model(
        undistorted_sparse_reconstruction_dir=undistorted_images_dir.joinpath("sparse"),
        text_model_dir=text_model_dir,
    )

    write_transforms_json(
        dataset_root_dir=dataset_root_dir,
        images_dir=undistorted_images_dir.joinpath("images"),
        text_model_dir=text_model_dir,
        bound=bound,
    )


def create_dataset_from_video(
    video_path: Path,
    dataset_root_dir: Path,
    bound: float,
    fps: int=3,
):
    video_path, dataset_root_dir = Path(video_path), Path(dataset_root_dir)
    raw_images_dir = dataset_root_dir.joinpath("images-raw")
    video_to_images(
        video_in=video_path,
        images_dir=raw_images_dir,
        fps=fps,
    )
    create_dataset_from_single_camera_image_collection(
        raw_images_dir=raw_images_dir,
        dataset_root_dir=dataset_root_dir,
        matcher="Sequential",
        bound=bound,
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
    interm = interm.at[idcs].set(jnp.clip(jnp.round(preds * 255), 0, 255).astype(jnp.uint8))
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
        rgbs = jnp.clip(jnp.round(rgbs * 255), 0, 255).astype(jnp.uint8)
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
        rgbas = jnp.concatenate([flattened, jnp.ones_like(flattened[:, :1])], axis=-1)
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


_ImageSourceType = jax.Array | np.ndarray | Image.Image | Path | str
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


def merge_transforms(transforms: Sequence[TransformJsonNGP | TransformJsonNeRFSynthetic]) -> TransformJsonNGP | TransformJsonNeRFSynthetic:
    return functools.reduce(
        lambda lhs, rhs: lhs.merge(rhs) if lhs is not None else rhs,
        transforms,
    )


def load_transform_json_recursive(src: Path | str) -> TransformJsonNGP | TransformJsonNeRFSynthetic | None:
    """
    returns a single transforms object with the `file_path` in its `frames` attribute converted to
    absolute paths
    """
    src = Path(src)

    if src.is_dir():
        all_transforms = tuple(filter(
            lambda xform: xform is not None,
            map(load_transform_json_recursive, src.iterdir()),
        ))
        if len(all_transforms) == 0:
            return None

        # merge transforms found from descendants if any
        transforms = merge_transforms(all_transforms)

    elif src.suffix == ".json":  # skip other files for speed
        try:
            transforms = json.load(open(src))
        except:
            # unreadable, or not a json
            return None
        try:
            transforms = (
                TransformJsonNeRFSynthetic(**transforms)
                if transforms.get("camera_angle_x") is not None
                else TransformJsonNGP(**transforms)
            )
            transforms = transforms.make_absolute(src.parent)
        except TypeError:
            # not a valid transform.json
            return None

    else:
        return None

    return transforms


def load_scene(
    srcs: Sequence[Path | str],
    world_scale: float,
    image_scale: float,
    load_views: bool=True,
) -> SceneMeta | Tuple[SceneData, List[ViewMetadata]]:
    assert isinstance(srcs, collections.abc.Sequence) and not isinstance(srcs, str), (
        "load_scene accepts a sequence of paths as srcs to load, did you mean '{}'?".format([srcs])
    )
    srcs = map(Path, srcs)

    transforms = merge_transforms(map(load_transform_json_recursive, srcs))

    if transforms is None:
        raise FileNotFoundError("could not find any valid transforms in {}".format(srcs))
    if len(transforms.frames) == 0:
        raise ValueError("could not find any frame in {}".format(srcs))

    def try_image_extensions(
        file_path: str,
        extensions: List[str]=["png", "jpg", "jpeg"],
    ) -> Path:
        if "" not in extensions:
            extensions = [""] + list(extensions)
        for ext in extensions:
            if len(ext) > 0 and ext[0] != ".":
                ext = "." + ext
            p = Path(file_path + ext)
            if p.exists():
                return p
        raise FileNotFoundError(
            "could not find a file at {} with any extension of {}".format(file_path, extensions)
        )

    # shared camera model
    if isinstance(transforms, TransformJsonNeRFSynthetic):
        _img = Image.open(try_image_extensions(transforms.frames[0].file_path))
        W, H = int(_img.width * image_scale), int(_img.height * image_scale)
        fovx = transforms.camera_angle_x
        focal = float(.5 * W / np.tan(fovx / 2))
        camera = PinholeCamera(
            W=W,
            H=H,
            fx=focal * image_scale,
            fy=focal * image_scale,
            cx=W / 2,
            cy=H / 2,
        )

    elif isinstance(transforms, TransformJsonNGP):
        camera = PinholeCamera(
            W=int(transforms.w * image_scale),
            H=int(transforms.h * image_scale),
            fx=transforms.fl_x * image_scale,
            fy=transforms.fl_y * image_scale,
            cx=transforms.cx * image_scale,
            cy=transforms.cy * image_scale,
        )

    else:
        raise TypeError("unexpected type for transforms: {}, expected one of {}".format(
            type(transforms),
            [TransformJsonNeRFSynthetic, TransformJsonNGP],
        ))

    scene_meta = SceneMeta(
        bound=transforms.aabb_scale * world_scale,
        camera=camera,
        frames=transforms.frames,
    )

    if not load_views:
        return scene_meta

    views = list(
        map(
            lambda frame: ViewMetadata(
                scale=image_scale,
                transform=RigidTransformation(
                    rotation=frame.transform_matrix_numpy[:3, :3],
                    translation=frame.transform_matrix_numpy[:3, 3],
                ),
                file=try_image_extensions(frame.file_path),
            ),
            transforms.frames,
        )
    )

    # uint8,[n_pixels, 4]
    all_rgbas = jnp.concatenate(
        list(tqdm(
            ThreadPoolExecutor().map(lambda view: view.rgba_u8, views),
            total=len(views),
            desc="pre-loading views",
            bar_format=tqdm_format),
        ),
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
    all_Ts *= world_scale
    # float,[n_views, 3+9+3]
    all_transforms = jnp.concatenate([all_Rs, all_Ts], axis=-1)

    scene_data = SceneData(
        meta=scene_meta,
        all_rgbas_u8=all_rgbas,
        all_transforms=all_transforms,
    )

    return scene_data, views


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
    scene, views = load_scene(
        rootdir="data/nerf/nerf_synthetic/lego",
        split="train",
        world_scale=.6,
    )
    print(scene.all_xys.shape)
    print(scene.all_rgbas.shape)
    print(scene.all_transforms.shape)
    print(len(views))


if __name__ == "__main__":
    main()
