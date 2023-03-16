#!/usr/bin/env python3

from pathlib import Path
from typing import Literal

from PIL import Image
from flax.training.train_state import TrainState
from icecream import ic
import jax
import jax.numpy as jnp
import jax.random as jran
import numpy as np
import optax
from tqdm import tqdm
import tyro

from models.imagefit import ImageFitter
from utils import data, common
from utils.args import ImageFitArgs


logger, (debug, info, warn, err, crit) = common.setup_logging("imagefit")


@jax.jit
def train_step(state: TrainState, x, y):
    def loss(params, x, y):
        preds = state.apply_fn({"params": params}, x)
        loss = jnp.square(preds - y).mean()
        return loss

    loss_grad_fn = jax.value_and_grad(loss)

    loss, grads = loss_grad_fn(state.params, x, y)
    state = state.apply_gradients(grads=grads)
    metrics = {
        "loss": loss,
    }
    return state, metrics


def train_epoch(
        loader: data.ImageFitLoader,
        state: TrainState,
        ep_log: int,
    ):
    loss = 0
    for uvs, colors in tqdm(loader, desc="ep#{:03d}".format(ep_log), bar_format=common.tqdm_format):
        state, metrics = train_step(state, uvs, colors)
        loss += metrics["loss"]
    return loss, state


def eval(
        image_array,
        state: TrainState,
    ):
    H, W, C = image_array.shape

    @jax.jit
    def set_pixels(imgarr, xy, preds):
        interm = imgarr.reshape(H*W, C)
        x = xy[:, 0]
        y = xy[:, 1]
        idcs = y * W + x
        interm = interm.at[idcs].set(jnp.clip(preds * 255, 0, 255).astype(jnp.uint8))
        return interm.reshape(H, W, C)

    # Retain exact pixel coordinates for later use in set_pixels, otherwise many pixel locations
    # will be skipped due to floating point precision issues, resulting in black strips in the
    # output image.
    x, y = jnp.meshgrid(jnp.arange(W), jnp.arange(H))
    all_xys = jnp.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=-1)
    all_uvs = all_xys / jnp.asarray([[W-1, H-1]])
    chunk_size = 2**20
    debug("evaluating with chunk_size={} (totally {} batches to eval)".format(chunk_size,
                                                                              all_uvs.shape[0] // chunk_size))
    for beg in tqdm(range(0, all_uvs.shape[0], chunk_size), desc="evaluating", bar_format=common.tqdm_format):
        uv = all_uvs[beg:beg+chunk_size]
        preds = state.apply_fn({"params": state.params}, uv)
        xy = all_xys[beg:beg+chunk_size]
        image_array = set_pixels(image_array, xy, preds)

    return image_array


def main(
        args: ImageFitArgs,
        in_image: Path,
        out_path: Path,
        encoding: Literal["hashgrid", "frequency"],
        # Enable this to suppress prompt if out_path exists and directly overwrite the file.
        overwrite: bool = False,
        encoding_prec: int = 32,
        model_summary: bool = False,
    ):
    logger.setLevel(args.common.logging.upper())

    if not out_path.parent.is_dir():
        err("Output path's parent '{}' does not exist or is not a directory!".format(out_path.parent))
        exit(1)

    if out_path.exists() and not overwrite:
        warn("Output path '{}' exists and will be overwritten!".format(out_path))
        try:
            r = input("Continue? [y/N] ")
            if (r.strip() + "n").lower()[0] != "y":
                exit(0)
        except EOFError:
            print()
            exit(0)
        except KeyboardInterrupt:
            print()
            exit(0)

    encoding_dtype = getattr(jnp, "float{}".format(encoding_prec))
    dtype = getattr(jnp, "float{}".format(args.common.prec))

    # deterministic
    K = common.set_deterministic(args.common.seed)

    # model parameters
    K, key = jran.split(K, 2)
    model, init_input = (
        ImageFitter(encoding=encoding, encoding_dtype=encoding_dtype),
        jnp.zeros((1, 2), dtype=dtype),
    )
    variables = model.init(key, init_input)
    if model_summary:
        print(model.tabulate(key, init_input))

    # training state
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optax.adam(
            learning_rate=args.train.lr,
            b1=0.9,
            b2=0.99,
            # paper:
            #   the small value of 𝜖 = 10^{−15} can significantly accelerate the convergence of the
            #   hash table entries when their gradients are sparse and weak.
            eps=1e-15,
        ),
    )

    # data
    dataset = data.SampledPixelsDataset(
        image=in_image,
        loop=args.data.loop,
    )
    loader = data.ImageFitLoader(
        dataset=dataset,
        batch_size=args.train.bs,
        num_workers=args.data.n_workers,
    )

    for ep in range(args.train.n_epochs):
        ep_log = ep + 1
        loss, state = train_epoch(loader, state, ep_log)
        info("epoch#{:03d}: loss={}".format(ep_log, loss))

        image = np.asarray(Image.new("RGB", dataset.image.shape[:2][::-1]))
        image = eval(image, state)
        debug("saving image of shape {} to {}".format(image.shape, out_path))
        Image.fromarray(np.asarray(image)).save(out_path)


if __name__ == "__main__":
    tyro.cli(main)
