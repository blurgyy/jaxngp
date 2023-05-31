import collections
from concurrent.futures import ThreadPoolExecutor
import functools
import json
from pathlib import Path
from typing import List, Literal, Sequence, Tuple
import warnings

from PIL import Image, ImageFilter, UnidentifiedImageError
import chex
import ffmpeg
import imageio
import jax
import jax.numpy as jnp
import jax.random as jran
from matplotlib import pyplot as plt
from natsort import natsorted
import numpy as np

from . import sfm
from .common import jit_jaxfn_with, mkValueError, tqdm
from .types import (
    ColmapMatcherType,
    ImageMetadata,
    OrbitTrajectoryOptions,
    PinholeCamera,
    RGBColor,
    RGBColorU8,
    RigidTransformation,
    SceneData,
    SceneMeta,
    SceneOptions,
    TransformJsonFrame,
    TransformJsonNGP,
    TransformJsonNeRFSynthetic,
    ViewMetadata,
)


def to_cpu(array: jax.Array) -> jax.Array:
    return jax.device_put(array, device=jax.devices("cpu")[0])


@jax.jit
def f32_to_u8(img: jax.Array) -> jax.Array:
    return jnp.clip(jnp.round(img * 255), 0, 255).astype(jnp.uint8)


def mono_to_rgb(img: jax.Array, cm: Literal["inferno", "jet", "turbo"]="inferno") -> jax.Array:
    return plt.get_cmap(cm)(img)


def sharpness_of(path: str | Path) -> float | None:
    try:
        image = Image.open(path)
    except (IsADirectoryError, UnidentifiedImageError) as e:
        warnings.warn(
            "failed loading '{}': {}".format(path, str(e)))
        return None
    laplacian = image.convert("L").filter(ImageFilter.FIND_EDGES)
    return float(np.asarray(laplacian).var())


def video_to_images(
    video_in: Path,
    images_dir: Path,
    fmt: str="%04d.png",
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


def write_sharpness_json(raw_images_dir: str | Path):
    raw_images_dir = Path(raw_images_dir)
    out_filename = "sharpnesses.jaxngp.json"
    image_candidates = tuple(filter(lambda x: x.name != out_filename, raw_images_dir.iterdir()))
    sharpnesses = ThreadPoolExecutor().map(sharpness_of, image_candidates)
    path_sharpness_tuples = zip(map(lambda p: p.absolute().as_posix(), raw_images_dir.iterdir()), sharpnesses)
    path_sharpness_tuples = filter(lambda tup: tup[1] is not None, path_sharpness_tuples)
    path_sharpness_tuples = sorted(tqdm(path_sharpness_tuples, desc="estimating sharpness of image collection"), key=lambda tup: tup[1], reverse=True)
    with open(raw_images_dir.joinpath(out_filename), "w") as f:
        json.dump(path_sharpness_tuples, f)


def write_transforms_json(
    scene_root_dir: Path,
    images_dir: Path,
    text_model_dir: Path,
    # given that the cameras' average distance to the origin is (4.0 * `camera_scale`), what would
    # the scene's bound be?
    bound: float,
    camera_scale: float,
    bg: bool,
):
    "adapted from NVLabs/instant-ngp/scripts/colmap2nerf.py"
    scene_root_dir, images_dir, text_model_dir = (
        Path(scene_root_dir),
        Path(images_dir),
        Path(text_model_dir),
    )
    rel_prefix = images_dir.relative_to(scene_root_dir)

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

    # estimate sharpness
    sharpnesses = ThreadPoolExecutor().map(lambda f: sharpness_of(scene_root_dir.joinpath(f.file_path)), frames)
    for i, sharpness in enumerate(tqdm(sharpnesses, total=len(frames), desc="estimating sharpness of image collection")):
        frames[i] = frames[i].replace(sharpness=sharpness)

    # reorient the scene to be easier to work with
    up = up / np.linalg.norm(up)
    print("up vector:", up, "->", [0, 0, 1])
    R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
    R = np.pad(R,[0,1])
    R[-1, -1] = 1

    for i, f in enumerate(frames):
        frames[i] = f.replace(transform_matrix=np.matmul(R, f.transform_matrix_numpy).tolist())

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
        frames[i] = f.replace(transform_matrix=new_m.tolist())

    avglen = 0.
    for f in frames:
        avglen += np.linalg.norm(f.transform_matrix_numpy[0:3,3])
    avglen /= len(frames)
    print("average camera distance from origin:", avglen, "->", 4.0)
    for i, f in enumerate(frames):
        # scale to "nerf sized"
        new_m = f.transform_matrix_numpy
        new_m[0:3, 3] *= 4.0 / avglen
        frames[i] = f.replace(transform_matrix=new_m.tolist())

    print("scene bound (i.e. half width of scene's aabb):", bound)
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
    all_transform_json: TransformJsonNGP = all_transform_json.replace(
        scale=camera_scale,
        bg=bg,
        up=[0, 0, 1],
    )
    train_tj = all_transform_json.replace(frames=frames[:len(frames) // 2])
    val_tj = all_transform_json.replace(frames=frames[len(frames) // 2:len(frames) // 2 + len(frames) // 4])
    test_tj = all_transform_json.replace(frames=frames[len(frames) // 2 + len(frames) // 4:])
    train_tj.save(scene_root_dir.joinpath("transforms_train.json"))
    val_tj.save(scene_root_dir.joinpath("transforms_val.json"))
    test_tj.save(scene_root_dir.joinpath("transforms_test.json"))
    return all_transform_json


def create_scene_from_single_camera_image_collection(
    raw_images_dir: Path,
    scene_root_dir: Path,
    matcher: ColmapMatcherType,
    bound: float,
    camera_scale: float,
    bg: bool,
):
    raw_images_dir, scene_root_dir = Path(raw_images_dir), Path(scene_root_dir)
    scene_root_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = scene_root_dir.joinpath("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    db_path = artifacts_dir.joinpath("colmap.db")
    sparse_reconstruction_dir = artifacts_dir.joinpath("sparse")
    undistorted_images_dir = scene_root_dir.joinpath("images-undistorted")
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
        warnings.warn(
            "colmap reconstructed more than 1 maps")

    sfm.undistort(
        images_dir=raw_images_dir,
        sparse_reconstruction_dir=sparse_reconstruction_dir.joinpath("0"),  # just use the first map reconstructed by colmap
        undistorted_images_dir=undistorted_images_dir,
    )

    sfm.export_text_format_model(
        undistorted_sparse_reconstruction_dir=undistorted_images_dir.joinpath("sparse"),
        text_model_dir=text_model_dir,
    )

    write_transforms_json(
        scene_root_dir=scene_root_dir,
        images_dir=undistorted_images_dir.joinpath("images"),
        text_model_dir=text_model_dir,
        bound=bound,
        camera_scale=camera_scale,
        bg=bg,
    )


def create_scene_from_video(
    video_path: Path,
    scene_root_dir: Path,
    bound: float,
    camera_scale: float,
    bg: bool,
    fps: int,
):
    video_path, scene_root_dir = Path(video_path), Path(scene_root_dir)
    raw_images_dir = scene_root_dir.joinpath("images-raw")
    video_to_images(
        video_in=video_path,
        images_dir=raw_images_dir,
        fps=fps,
    )
    create_scene_from_single_camera_image_collection(
        raw_images_dir=raw_images_dir,
        scene_root_dir=scene_root_dir,
        matcher="Sequential",
        bound=bound,
        camera_scale=camera_scale,
        bg=bg,
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
        lhs = jnp.tile(lhs[..., None], (1, 1, 3))
    if len(rhs.shape) == 2 or rhs.shape[-1] == 1:
        rhs = jnp.tile(rhs[..., None], (1, 1, 3))
    if rhs.shape[-1] == 3:
        rhs = jnp.concatenate([rhs, 255 * jnp.ones_like(rhs[..., -1:], dtype=jnp.uint8)], axis=-1)
    if lhs.shape[-1] == 3:
        lhs = jnp.concatenate([lhs, 255 * jnp.ones_like(lhs[..., -1:], dtype=jnp.uint8)], axis=-1)
    if vertical:
        chex.assert_axis_dimension(lhs, 1, W)
        chex.assert_axis_dimension(rhs, 1, W)
    else:
        chex.assert_axis_dimension(lhs, 0, H)
        chex.assert_axis_dimension(rhs, 0, H)
    concat_axis = 0 if vertical else 1
    if gap > 0:
        gap_color = jnp.asarray(gap_color + (0xff,), dtype=jnp.uint8)
        gap = jnp.broadcast_to(gap_color, (gap, W, 4) if vertical else (H, gap, 4))
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
    chex.assert_axis_dimension(img, -1, 4)
    chex.assert_scalar_non_negative(width)
    chex.assert_type(img, jnp.uint8)
    color = jnp.asarray(color + (0xff,), dtype=jnp.uint8)
    H, W = img.shape[:2]
    leftright = jnp.broadcast_to(color, (H, width, 4))
    img = jnp.concatenate([leftright, img, leftright], axis=1)
    topbottom = jnp.broadcast_to(color, (width, W+2*width, 4))
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
    try:
        for im in tqdm(images, desc="writing video to {}".format(dest.as_posix())):
            video_writer.append_data(np.asarray(im))
    except (BrokenPipeError, IOError) as e:  # sometimes ffmpeg encounters io error for no apparent reason
        warnings.warn(
            "failed writing video: {}".format(str(e)), RuntimeWarning)
        warnings.warn(
            "skipping saving video '{}'".format(dest.as_posix()), RuntimeWarning)


@jax.jit
def set_pixels(imgarr: jax.Array, xys: jax.Array, selected: jax.Array, preds: jax.Array) -> jax.Array:
    chex.assert_type(imgarr, jnp.uint8)
    H, W = imgarr.shape[:2]
    if len(imgarr.shape) == 3:
        interm = imgarr.reshape(H*W, -1)
    else:
        interm = imgarr.ravel()
    idcs = xys[selected, 1] * W + xys[selected, 0]
    interm = interm.at[idcs].set(f32_to_u8(preds))
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


def blend_rgba_image_array(imgarr, bg: jax.Array):
    """
    Blend the given background color according to the given alpha channel from `imgarr`.
    WARN: this function SHOULD NOT be used for blending background colors into volume-rendered
          pixels because the colors of volume-rendered pixels already have the alpha channel
          factored-in.  To blend background for volume-rendered pixels, directly add the scaled
          background color.
          E.g.: `final_color = ray_accumulated_color + (1 - ray_opacity) * bg`
    """
    if isinstance(imgarr, Image.Image):
        imgarr = np.asarray(imgarr)
    chex.assert_shape(imgarr, [..., 4])
    chex.assert_type(imgarr, bg.dtype)
    rgbs, alpha = imgarr[..., :-1], imgarr[..., -1:]
    bg = jnp.broadcast_to(bg, rgbs.shape)
    if imgarr.dtype == jnp.uint8:
        rgbs, alpha = rgbs.astype(float) / 255, alpha.astype(float) / 255
        rgbs = rgbs * alpha + bg * (1 - alpha)
        rgbs = f32_to_u8(rgbs)
    else:
        rgbs = rgbs * alpha + bg * (1 - alpha)
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
        if isinstance(transforms, dict):
            try:
                transforms = (
                    TransformJsonNeRFSynthetic(**transforms)
                    if transforms.get("camera_angle_x") is not None
                    else TransformJsonNGP(**transforms)
                )
                transforms = transforms.make_absolute(src.parent).rotate_world_up().scale_camera_positions()
            except TypeError:
                # not a valid transform.json
                return None
        else:
            return None

    else:
        return None

    return transforms


def try_image_extensions(
    file_path: Path,
    extensions: List[str]=["png", "jpg", "jpeg"],
) -> Path | None:
    if "" not in extensions:
        extensions = [""] + list(extensions)
    for ext in extensions:
        if len(ext) > 0 and ext[0] != ".":
            ext = "." + ext
        p = Path(file_path.as_posix() + ext)
        if p.exists():
            return p
        p = Path(file_path.with_suffix(ext))
        if p.exists():
            return p
    warnings.warn(
        "could not find a file at '{}' with any extension of {}".format(file_path, extensions),
        RuntimeWarning,
    )
    return None


def load_scene(
    srcs: Sequence[Path | str],
    scene_options: SceneOptions,
    sort_frames: bool=False,
    orbit_options: OrbitTrajectoryOptions | None=None,
) -> SceneMeta | Tuple[SceneData, List[ViewMetadata]]:
    """
    Inputs:
        srcs: sequence of paths to recursively load transforms.json
        scene_options: see :class:`SceneOptions`
        sort_frames: whether to sort the frames by their filenames, (uses natural sort if enabled)
        orbit_options: if not `None`, generate a sequence of camera poses and ignore the loaded
                       camera poses
    """

    assert isinstance(srcs, collections.abc.Sequence) and not isinstance(srcs, str), (
        "load_scene accepts a sequence of paths as srcs to load, did you mean '{}'?".format([srcs])
    )
    srcs = list(map(Path, srcs))

    transforms = merge_transforms(map(load_transform_json_recursive, srcs))
    if transforms is None:
        raise FileNotFoundError("could not load transforms from any of {}".format(srcs))

    loaded_frames, discarded_frames = functools.reduce(
        lambda prev, frame: (
            prev[0] + ((frame,) if (
                frame.file_path is not None
                and frame.sharpness >= scene_options.sharpness_threshold
            ) else ()),
            prev[1] + (() if (
                frame.file_path is not None
                and frame.sharpness >= scene_options.sharpness_threshold
            ) else (frame,)),
        ),
        map(
            lambda f: f.replace(file_path=try_image_extensions(f.file_path)),
            transforms.frames,
        ),
        (tuple(), tuple()),
    )

    if len(loaded_frames) == 0:
        raise RuntimeError("loaded 0 frame from '{}' (discarded {} frame(s))".format(srcs, len(discarded_frames)))

    transforms = transforms.replace(frames=loaded_frames)

    if sort_frames:
        transforms = transforms.replace(
            frames=natsorted(transforms.frames, key=lambda f: f.file_path),
        )

    # shared camera model
    if isinstance(transforms, TransformJsonNeRFSynthetic):
        _img = Image.open(try_image_extensions(transforms.frames[0].file_path))
        fovx = transforms.camera_angle_x
        focal = float(.5 * _img.width / np.tan(fovx / 2))
        camera = PinholeCamera(
            W=_img.width,
            H=_img.height,
            fx=focal,
            fy=focal,
            cx=_img.width / 2,
            cy=_img.height / 2,
        )

    elif isinstance(transforms, TransformJsonNGP):
        camera = PinholeCamera(
            W=transforms.w,
            H=transforms.h,
            fx=transforms.fl_x,
            fy=transforms.fl_y,
            cx=transforms.cx,
            cy=transforms.cy,
        )

    else:
        raise TypeError("unexpected type for transforms: {}, expected one of {}".format(
            type(transforms),
            [TransformJsonNeRFSynthetic, TransformJsonNGP],
        ))

    scene_meta = SceneMeta(
        bound=transforms.aabb_scale * scene_options.world_scale,
        bg=transforms.bg,
        camera=(
            camera
                .scale_world(scene_options.world_scale)
                .scale_resolution(scene_options.resolution_scale)
        ),
        frames=transforms.frames,
    )

    if orbit_options is not None:
        assert isinstance(orbit_options, OrbitTrajectoryOptions)
        return scene_meta.make_frames_with_orbiting_trajectory(orbit_options)

    views = tuple(map(
        lambda frame: ViewMetadata(
            scale=scene_options.resolution_scale,
            transform=RigidTransformation(
                rotation=frame.transform_matrix_numpy[:3, :3],
                translation=frame.transform_matrix_numpy[:3, 3],
            ),
            file=frame.file_path,
        ),
        transforms.frames,
    ))

    # uint8,[n_pixels, 4]
    all_rgbas = jnp.concatenate(
        list(tqdm(
            ThreadPoolExecutor().map(lambda view: view.rgba_u8, views),
            total=len(views),
            desc="pre-loading views"
        )),
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
    all_Ts *= scene_options.world_scale
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
