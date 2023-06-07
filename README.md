jaxngp
======

This repository contains [JAX] implementations of:
* a multiresolution hash encoder (JAX)
* an accelerated volume renderer for fast training of NeRFs (CUDA + JAX), with
  * occupancy grid pruning during ray marching
  * early stop during ray color integration
* an inference-time renderer for real-time rendering of NeRFs (CUDA + JAX)
* a GUI for visualizing \& interacting \& exploring NeRFs **[@seimeicyx]**

Benchmarks
----------

### [NeRF-synthetic]

|                                            | mic   | ficus | chair | hotdog | materials | drums | ship  | lego  | average |
|:--- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| jaxngp _@33.7k steps_ <br> (this codebase) | 36.07 | 33.23 | 35.06 | 37.09  | 29.54     | 25.78 | 30.93 | 36.08 | 32.973  |
| jaxngp _@51.2k steps_ <br> (this codebase) | 35.99 | 33.28 | 34.99 | 37.16  | 29.57     | 25.83 | 30.93 | 36.11 | 32.983  |
| paper ([instant-ngp])                      | 36.22 | 33.51 | 35.00 | 37.40  | 29.78     | 26.02 | 31.10 | 36.39 | 33.176  |

<sup>
For each scene, the network is trained on 100 training images (800x800 each) for 30k steps with
default parameters, reported PSNR is averaged across 200 test images.
</sup>

Environment Setup
-----------------

jaxngp manages environments with Nix, but it's also possible to setup the environment with any other package manager (e.g. Conda).

### With Nix (recommended)

1. Install Nix with the [official installer](https://nixos.org/download) or the [nix-installer](https://github.com/DeterminateSystems/nix-installer/releases).
2. With the `nix` executable available, clone this repository and setup environment:
   ```bash
   $ git clone https://github.com/blurgyy/jaxngp.git
   $ cd jaxngp/
   $ NIXPKGS_ALLOW_UNFREE=1 nix develop --impure
   ```
   This will download (or build if necessary) all the dependencies, and opens a new shell with all the dependencies configured.
   > **Note**: to avoid the built environment being garbage collected when `nix gc` or `nix-collect-garbage` is called, append a `--profile <PATH>` argument:
   > ```bash
   > $ NIXPKGS_ALLOW_UNFREE=1 nix develop --impure --profile .git/devshell.profile
   > ```

### With Conda

TODO

Running
-------

<!-- > **Note**: All the commands below are run after the environment has been setup. -->

The program's entrance is at `python3 -m app.nerf`.  It provides three subcommands: [`train`](./app/nerf/train.py), [`test`](./app/nerf/test.py), and [`gui`](./app/nerf/gui.py).  Pass `-h|--help` to any of the subcommand to see its usage, e.g.:

<details>
<summary>
  <code>python3 -m app.nerf train --help</code>
</summary>

```markdown
usage: __main__.py train [-h] --exp-dir PATH [--raymarch.diagonal-n-steps INT]
                         [--raymarch.perturb | --raymarch.no-perturb]
                         [--raymarch.density-grid-res INT] [--render.bg FLOAT FLOAT FLOAT]
                         [--render.random-bg | --render.no-random-bg]
                         [--scene.sharpness-threshold FLOAT] [--scene.world-scale FLOAT]
                         [--scene.resolution-scale FLOAT] [--scene.camera-near FLOAT]
                         [--logging {DEBUG,INFO,WARN,WARNING,ERROR,CRITICAL}] [--seed INT]
                         [--summary | --no-summary] [--frames-val PATH [PATH ...]]
                         [--ckpt {None}|PATH] [--lr FLOAT] [--tv-scale FLOAT] [--bs INT]
                         [--n-epochs INT] [--n-batches INT] [--data-loop INT] [--validate-every INT]
                         [--keep INT] [--keep-every {None}|INT]
                         [--raymarch-eval.diagonal-n-steps INT]
                         [--raymarch-eval.perturb | --raymarch-eval.no-perturb]
                         [--raymarch-eval.density-grid-res INT] [--render-eval.bg FLOAT FLOAT FLOAT]
                         [--render-eval.random-bg | --render-eval.no-random-bg]
                         PATH [PATH ...]

╭─ positional arguments ───────────────────────────────────────────────────────────────────────────╮
│ PATH [PATH ...]         directories or transform.json files containing data for training         │
│                         (required)                                                               │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ arguments ──────────────────────────────────────────────────────────────────────────────────────╮
│ -h, --help              show this help message and exit                                          │
│ --exp-dir PATH          experiment artifacts are saved under this directory (required)           │
│ --frames-val PATH [PATH ...]                                                                     │
│                         directories or transform.json files containing data for validation       │
│                         (default: )                                                              │
│ --ckpt {None}|PATH      if specified, continue training from this checkpoint (default: None)     │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ raymarch arguments ─────────────────────────────────────────────────────────────────────────────╮
│ raymarching/rendering options during training                                                    │
│ ──────────────────────────────────────────────────────────────────────────────────────────────── │
│ --raymarch.diagonal-n-steps INT                                                                  │
│                         for calculating the length of a minimal ray marching step, the NGP paper │
│                         uses 1024 (appendix E.1) (default: 1024)                                 │
│ --raymarch.perturb, --raymarch.no-perturb                                                        │
│                         whether to fluctuate the first sample along the ray with a tiny          │
│                         perturbation (default: True)                                             │
│ --raymarch.density-grid-res INT                                                                  │
│                         resolution for the auxiliary density/occupancy grid, the NGP paper uses  │
│                         128 (appendix E.2) (default: 128)                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ render arguments ───────────────────────────────────────────────────────────────────────────────╮
│ raymarching/rendering options during training                                                    │
│ ──────────────────────────────────────────────────────────────────────────────────────────────── │
│ --render.bg FLOAT FLOAT FLOAT                                                                    │
│                         background color for transparent parts of the image, has no effect if    │
│                         `random_bg` is True (default: 1.0 1.0 1.0)                               │
│ --render.random-bg, --render.no-random-bg                                                        │
│                         ignore `bg` specification and use random color for transparent parts of  │
│                         the image (default: True)                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ scene arguments ────────────────────────────────────────────────────────────────────────────────╮
│ raymarching/rendering options during training                                                    │
│ ──────────────────────────────────────────────────────────────────────────────────────────────── │
│ --scene.sharpness-threshold FLOAT                                                                │
│                         images with sharpness lower than this value will be discarded (default:  │
│                         -1.0)                                                                    │
│ --scene.world-scale FLOAT                                                                        │
│                         scale both the scene's camera positions and bounding box with this       │
│                         factor (default: 1.0)                                                    │
│ --scene.resolution-scale FLOAT                                                                   │
│                         scale input images in case they are too large, camera intrinsics are     │
│                         also scaled to match the updated image resolution. (default: 1.0)        │
│ --scene.camera-near FLOAT                                                                        │
│                         (default: 0.3)                                                           │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ common arguments ───────────────────────────────────────────────────────────────────────────────╮
│ --logging {DEBUG,INFO,WARN,WARNING,ERROR,CRITICAL}                                               │
│                         log level (default: INFO)                                                │
│ --seed INT              random seed (default: 1000000007)                                        │
│ --summary, --no-summary                                                                          │
│                         display model information after model init (default: False)              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ train arguments ────────────────────────────────────────────────────────────────────────────────╮
│ training hyper parameters                                                                        │
│ ──────────────────────────────────────────────────────────────────────────────────────────────── │
│ --lr FLOAT              learning rate (default: 0.01)                                            │
│ --tv-scale FLOAT        scalar multiplied to total variation loss, set this to a positive value  │
│                         to enable calculation of TV loss (default: 0.0)                          │
│ --bs INT                batch size (default: 1048576)                                            │
│ --n-epochs INT          training epochs (default: 50)                                            │
│ --n-batches INT         batches per epoch (default: 1024)                                        │
│ --data-loop INT         loop within training data for this number of iterations, this helps      │
│                         reduce the effective dataloader overhead. (default: 1)                   │
│ --validate-every INT    will validate every `validate_every` epochs, set this to a large value   │
│                         to disable validation (default: 10)                                      │
│ --keep INT              number of latest checkpoints to keep (default: 1)                        │
│ --keep-every {None}|INT                                                                          │
│                         how many epochs should a new checkpoint to be kept (in addition to       │
│                         keeping the last `keep` checkpoints) (default: 8)                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ raymarch-eval arguments ────────────────────────────────────────────────────────────────────────╮
│ raymarching/rendering options for validating during training                                     │
│ ──────────────────────────────────────────────────────────────────────────────────────────────── │
│ --raymarch-eval.diagonal-n-steps INT                                                             │
│                         for calculating the length of a minimal ray marching step, the NGP paper │
│                         uses 1024 (appendix E.1) (default: 1024)                                 │
│ --raymarch-eval.perturb, --raymarch-eval.no-perturb                                              │
│                         whether to fluctuate the first sample along the ray with a tiny          │
│                         perturbation (default: False)                                            │
│ --raymarch-eval.density-grid-res INT                                                             │
│                         resolution for the auxiliary density/occupancy grid, the NGP paper uses  │
│                         128 (appendix E.2) (default: 128)                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ render-eval arguments ──────────────────────────────────────────────────────────────────────────╮
│ raymarching/rendering options for validating during training                                     │
│ ──────────────────────────────────────────────────────────────────────────────────────────────── │
│ --render-eval.bg FLOAT FLOAT FLOAT                                                               │
│                         background color for transparent parts of the image, has no effect if    │
│                         `random_bg` is True (default: 0.0 0.0 0.0)                               │
│ --render-eval.random-bg, --render-eval.no-random-bg                                              │
│                         ignore `bg` specification and use random color for transparent parts of  │
│                         the image (default: False)                                               │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

<sup>
  Above is just an example and might not reflect the state of the latest codebase.
</sup>
</details>

### Examples

* `train`
  * Train for 10 epochs, with a batch size of 262144, on all the 400 (100\*train + 100\*validation + 200\*test) images from the `lego` scene of [NeRF-synthetic] dataset:
    ```bash
    $ python3 -m app.nerf train data/nerf_synthetic/lego --exp-dir=logs/lego-trainvaltest --{n-epochs=10,bs=262144}
    ```
  * Train on the training and validation splits of the `drums` scene, with a weight of 1e-5 on the Total Variation (TV) loss (by default this weight is 0):
    ```bash
    $ python3 -m app.nerf train data/nerf_synthetic/drums/transforms_{train,val}.json --exp-dir=logs/drums-trainval --tv-scale=1e-5
    ```
  * Train on the training split of the `mic` scene, validate with the validation split, validate after every epoch:
    ```bash
    $ python3 -m app.nerf train data/nerf_synthetic/mic/transforms_train.json --frames-val=data/nerf_synthetic/mic/transforms_val.json --exp-dir=logs/mic --validate-every=1
    ```
    > **Note**: The validated images are logged to [tensorboard], located under `--exp-dir`'s `logs/` directory.  View it in browser with:
    > ```bash
    > $ tensorboard serve --logdir logs/mic/logs/ --bind_all
    > TensorBoard 2.10.0 at http://localhost:6006/ (Press CTRL+C to quit)
    > ```
* `test`
  * Test using the latest checkpoint under `logs/mic` directory, with the camera intrinsics and extrinsics of the `mic` scene's test split
    ```bash
    $ python3 -m app.nerf test data/nerf_synthetic/mic/transforms_test.json --ckpt=logs/mic/ --exp-dir=output
    ```
  * Test on the `mic` scene with given camera extrinsics, but override the camera's resolution to 1920x1080, and use white as background color:
    ```bash
    $ python3 -m app.nerf test data/nerf_synthetic/mic/transforms_test.json --ckpt=logs/mic/ --exp-dir=output --camera-override.{width=1920,height=1080} --render.bg 1 1 1
    ```
  * Test with a generated orbiting trajectory (see [Demos] for an example) on the `mic` scene, with resolution 1920x1080:
    ```bash
    $ python3 -m app.nerf test data/nerf_synthetic/mic --trajectory=orbit --ckpt=logs/mic/ --exp-dir=output --camera-override.{width=1920,height=1080}
    ```
* `gui`
  > **Note**: The `gui` subcommand accepts all the parameters of the `train` subcommand, and additionally a `--viewport` parameter (but the default values of `--viewport` are sane enough to leave as-is).
  * Train on all the 400 images from the `lego` scene, with 1280x720 as the default size of the rendering viewport:
    ```bash
    $ python3 -m app.nerf gui data/nerf_synthetic/lego --exp-dir=logs/gui-lego --viewport.{W=1280,H=720}
    ```

Running on Custom Data
----------------------

A helper CLI (just a [colmap] wrapper via [pycolmap]) is provided for creating an Instant-NGP-compatible scene from a casually captured video or a directory of images:

```bash
$ python3 -m utils create-scene --help  # see full usage
usage: ...
$
$ # capture or download a video
$ mkdir -p data/_src
$ curl https://github.com/blurgyy/jaxngp/assets/44701880/022a7b3c-344d-418f-aba0-0ccb9bfeb374 -Lo data/_src/gundam.mp4
$
$ # create a scene from the video, set scene bound to 16, with a background color model
$ python3 -m utils create-scene data/_src/gundam.mp4 --root-dir=data/gundam --matcher=Sequential --fps=5 --bound=16 --bg
[...]
```

After the scene has been created, the rest (training/validating/testing) are the same:

```bash
$ # train on all the registered images
$ python3 -m utils train data/gundam --exp-dir=logs/gundam
[...]
$
$ # Render novel views, with a resolution of 1920x1080, save results as images and a video (video shown below in the Demo section)
$ python3 -m app.nerf test data/custom/gundam --{ckpt,exp-dir}=logs/gundam --trajectory=orbit --camera-override.{width=1920,height=1080} --orbit.high=1 --save-as="video and images"
```

### Demos

* Orbiting camera trajectory:

  [gundam-nvs]

  <sup>[Source video][gundam-source] courtesy of **[@lzyronrico]** | Rendering time: \~18min for 288 frames | Resolution: 1920x1080 | Hardware: 1 GTX 1080Ti</sup>

* Interactive rendering in the GUI:

  [gui-teaser]

  <sup>[Source video][metasun-source] courtesy of **[@filiptronicek]** | Hardware: 1 GTX 1080Ti</sup>


[JAX]: <https://github.com/google/jax>
[instant-ngp]: <https://github.com/NVLabs/instant-ngp>
[colmap]: <https://github.com/colmap/colmap>
[pycolmap]: <https://github.com/colmap/pycolmap>
[tensorboard]: <https://github.com/tensorflow/tensorboard>

[@seimeicyx]: https://github.com/seimeicyx
[@lzyronrico]: https://github.com/lzyronrico
[@filiptronicek]: https://github.com/filiptronicek

[NeRF-synthetic]: <https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi>

[gundam-source]: <https://github.com/blurgyy/jaxngp/assets/44701880/022a7b3c-344d-418f-aba0-0ccb9bfeb374>
[metasun-source]: <https://twitter.com/filiptronicek/status/1654894133801103360>

[gundam-nvs]: <https://github.com/blurgyy/jaxngp/assets/44701880/2ce8e57c-e179-469a-9ca2-10219fcba58d>
[gui-teaser]: <https://github.com/blurgyy/jaxngp/assets/44701880/b94dcd0f-a66d-404e-aee2-87f91ddf52fe>

[Demos]: <#demos>
