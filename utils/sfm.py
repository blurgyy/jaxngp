from pathlib import Path
from typing import Dict
from typing_extensions import get_args

import pycolmap

from .common import mkValueError
from .types import ColmapMatcherType


def extract_features(
    images_dir: Path,
    db_path: Path,
):
    images_dir, db_path = Path(images_dir), Path(db_path)
    pycolmap.extract_features(
        database_path=db_path,
        image_path=images_dir,
        # REF:
        #   <https://github.com/colmap/colmap/blob/43de802cfb3ed2bd155150e7e5e3e8c8dd5aaa3e/src/exe/feature.h#L44-L52>
        #   <https://github.com/colmap/pycolmap/blob/bdcdf47e0d40240c6f53dc463d7ceaa2cef923fd/pipeline/extract_features.cc#L63>
        camera_mode="SINGLE",
        # REF:
        #   <https://github.com/colmap/pycolmap/blob/bdcdf47e0d40240c6f53dc463d7ceaa2cef923fd/pipeline/extract_features.cc#L64>
        camera_model="OPENCV",
        reader_options=pycolmap.ImageReaderOptions(
            # camera_model="OPENCV",  # NOTE: this is obsolete, see camera_model above
            # single_camera=True,  # NOTE: this is obsolete, see camera_mode above
        ),
        sift_options=pycolmap.SiftExtractionOptions(
            estimate_affine_shape=True,
            domain_size_pooling=True,
        ),
    )


def match_features(
    matcher: ColmapMatcherType,
    db_path: Path,
):
    db_path = Path(db_path)
    if matcher not in get_args(ColmapMatcherType):
        raise mkValueError(
            desc="colmap matcher",
            value=matcher,
            type=ColmapMatcherType,
        )
    match_fn = getattr(pycolmap, "match_{}".format(matcher.lower()))
    return match_fn(
        database_path=db_path,
        sift_options=pycolmap.SiftMatchingOptions(
            guided_matching=True,
        ),
    )


def sparse_reconstruction(
    images_dir: Path,
    sparse_reconstruction_dir: Path,
    db_path: Path,
) -> Dict[int, pycolmap.Reconstruction]:
    images_dir, sparse_reconstruction_dir = Path(images_dir), Path(sparse_reconstruction_dir)
    maps = pycolmap.incremental_mapping(
        database_path=db_path,
        image_path=images_dir,
        output_path=sparse_reconstruction_dir,
        options=pycolmap.IncrementalMapperOptions(
            ba_refine_principal_point=True,
        ),
    )
    return maps


def undistort(
    images_dir: Path,
    sparse_reconstruction_dir: Path,
    undistorted_images_dir: Path,
):
    images_dir, sparse_reconstruction_dir, undistorted_images_dir = (
        Path(images_dir),
        Path(sparse_reconstruction_dir),
        Path(undistorted_images_dir),
    )
    pycolmap.undistort_images(
        output_path=undistorted_images_dir,
        input_path=sparse_reconstruction_dir,
        image_path=images_dir,
    )


def export_text_format_model(
    undistorted_sparse_reconstruction_dir: Path,
    text_format_model_dir: Path,
):
    undistorted_sparse_reconstruction_dir, text_format_model_dir = (
        Path(undistorted_sparse_reconstruction_dir),
        Path(text_format_model_dir),
    )
    text_format_model_dir.mkdir(parents=True, exist_ok=True)
    reconstruction = pycolmap.Reconstruction(undistorted_sparse_reconstruction_dir)
    reconstruction.write_text(text_format_model_dir.as_posix())
