#!/usr/bin/env python3

from pathlib import Path
from typing import Literal

from PIL import Image
from flax.training.train_state import TrainState
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
def train_step(state: TrainState, uvs, rgbs, perm):
    def loss(params, x, y):
        preds = state.apply_fn({"params": params}, x)
        loss = jnp.square(preds - y).mean()
        return loss

    loss_grad_fn = jax.value_and_grad(loss)

    loss, grads = loss_grad_fn(state.params, uvs[perm], rgbs[perm])
    state = state.apply_gradients(grads=grads)
    metrics = {
        "loss": loss * perm.shape[0],
    }
    return state, metrics


def train_epoch(
        data: data.ImageData,
        permutation: data.Dataset,
        total_batches: int,
        state: TrainState,
        ep_log: int,
    ):
    loss = 0
    for perm in tqdm(permutation, total=total_batches, desc="ep#{:03d}".format(ep_log), bar_format=common.tqdm_format):
        state, metrics = train_step(state, data.uvs, data.rgbs, perm)
        loss += metrics["loss"]
    return loss, state


@jax.jit
def eval_step(state, uvs, perm):
    preds = state.apply_fn({"params": state.params}, uvs[perm])
    return preds


def eval(
        image_array,
        data: data.ImageData,
        state: TrainState,
    ):
    H, W, C = image_array.shape

    @jax.jit
    def set_pixels(imgarr, idcs, perm, preds):
        interm = imgarr.reshape(H*W, C)
        idcs = idcs[perm, 1] * W + idcs[perm, 0]
        interm = interm.at[idcs].set(jnp.clip(preds * 255, 0, 255).astype(jnp.uint8))
        return interm.reshape(H, W, C)
    @common.jit_jaxfn_with(static_argnames=["chunk_size"])
    def get_perms(chunk_size: int) -> list[jax.Array]:
        all_perms = jnp.arange(H*W)
        n_chunk = H*W // chunk_size
        perms = jnp.array_split(all_perms, n_chunk)
        return perms

    for perm in tqdm(get_perms(chunk_size=2**15), desc="evaluating", bar_format=common.tqdm_format):
        # preds = state.apply_fn({"params": state.params}, uv)
        preds = eval_step(state, data.uvs, perm)
        image_array = set_pixels(image_array, data.idcs, perm, preds)

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
    in_image = np.asarray(Image.open(in_image))
    image_data = data.make_image_data(
        image=in_image,
        use_white_bg=True,
    )

    for ep in range(args.train.n_epochs):
        ep_log = ep + 1
        K, key = jran.split(K, 2)
        permutation = data.make_permutation_dataset(
            key,
            size=image_data.W * image_data.H,
            shuffle=True
        )\
            .batch(args.train.bs, drop_remainder=True)\
            .repeat(args.data.loop)
        loss, state = train_epoch(
            data=image_data,
            permutation=permutation.as_numpy_iterator(),
            total_batches=len(permutation),
            state=state,
            ep_log=ep_log,
        )
        info("epoch#{:03d}: loss={}".format(ep_log, loss))

        image = np.asarray(Image.new("RGB", in_image.shape[:2][::-1]))
        image = eval(image, image_data, state)
        debug("saving image of shape {} to {}".format(image.shape, out_path))
        Image.fromarray(np.asarray(image)).save(out_path)


if __name__ == "__main__":
    tyro.cli(main)
