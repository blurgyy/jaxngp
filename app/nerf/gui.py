from copy import deepcopy
from enum import Enum
import logging
from pathlib import Path
import numpy as np
from typing import List, Any, Tuple, Union
import jax
import jax.random as jran
import jax.numpy as jnp
from flax.training import checkpoints
from dataclasses import dataclass, field
import threading
import dearpygui.dearpygui as dpg
import ctypes
from utils.args import NeRFGUIArgs
from .train import *
from utils.types import (RGBColor, SceneData, SceneMeta, Camera)
from models.nerfs import (NeRF, SkySphereBg)
from PIL import Image
import time


@dataclass
class CKPT():
    need_load_ckpt = False
    ckpt_file_path: Path = Path("")
    step: int = 0

    def parse_ckpt(self, ckpt_name: str, ckpt_path: str) -> str:
        success = False
        s = ckpt_name.split("_")
        if s[0] == "checkpoint" and Path(ckpt_path).exists:
            try:
                self.step = int(s[1].split(".")[0])
                self.ckpt_file_path = Path(ckpt_path)
                self.need_load_ckpt = True
                success = True
            except TypeError or ValueError as e:
                self.logger.error(e)
            finally:
                if success:
                    return "checkpoint loaded from '{}'".format(self.ckpt_file_path)
                return "Fail to load checkpoint, causing the file is not a checkpoint"


@dataclass
class CameraPose():
    theta: float = 160.0
    phi: float = -30.0
    radius: float = 4.0
    tx: float = 0.0
    ty: float = 0.0
    centroid: np.ndarray = np.asarray([0., 0., 0.])

    def pose_spherical(self, theta, phi, radius):
        trans_t = lambda t: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t],
                                      [0, 0, 0, 1]], np.float32)
        rot_phi = lambda phi: np.array(
            [[1, 0, 0, 0], [0, np.cos(phi), -np.sin(phi), 0],
             [0, np.sin(phi), np.cos(phi), 0], [0, 0, 0, 1]], np.float32)
        rot_theta = lambda theta: np.array(
            [[np.cos(theta), 0, -np.sin(theta), 0], [0, 1, 0, 0],
             [np.sin(theta), 0, np.cos(theta), 0], [0, 0, 0, 1]], np.float32)
        c2w = trans_t(radius)
        #rotate
        c2w = np.matmul(rot_phi(phi / 180. * np.pi), c2w)
        c2w = np.matmul(rot_theta(theta / 180. * np.pi), c2w)

        return c2w

    @property
    def pose(self):
        mod = lambda x: x % 360
        self.theta = mod(self.theta)
        self.phi = mod(self.phi)
        c2w = self.pose_spherical(self.theta, self.phi, self.radius)
        #translate
        self.centroid = np.asarray(
            self.centroid) + self.tx * c2w[:3, 0] + self.ty * c2w[:3, 1]
        self.tx, self.ty = 0, 0
        trans_centroid = np.array(
            [[1, 0, 0, self.centroid[0]], [0, 1, 0, self.centroid[1]],
             [0, 0, 1, self.centroid[2]], [0, 0, 0, 1]], np.float32)
        c2w = np.matmul(trans_centroid, c2w)
        c2w = np.matmul(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0,1]]), c2w)
        return jnp.asarray(c2w)

    def move(self, dx, dy):
        self.theta += .3 * dx
        self.phi -= .2 * dy
        return self.pose

    def trans(self, dx, dy):
        velocity = 8e-4
        self.tx -= dx * velocity * self.radius
        self.ty += dy * velocity * self.radius
        return self.pose

    def change_radius(self, rate):
        self.radius *= 1.1**(-rate)
        return self.pose


@dataclass
class Gui_trainer():
    KEY: jran.KeyArray
    args: NeRFGUIArgs
    logger: common.Logger
    camera_pose: jnp.array
    back_color: RGBColor

    scene_train: SceneData = field(init=False)
    scene_meta: SceneMeta = field(init=False)

    nerf_model_train: NeRF = field(init=False)
    nerf_model_inference: NeRF = field(init=False)
    nerf_variables: Any = field(init=False)
    bg_model: SkySphereBg = field(init=False, default=None)
    bg_variables: Any = field(init=False)
    optimizer: Any = field(init=False)

    state: NeRFState = field(init=False)
    last_resample_step: int = -1
    cur_step: int = 0
    log_step: int = 0
    loss_log: str = "--"

    istraining: bool = field(init=False)

    data_step: List[int] = field(default_factory=list, init=False)
    data_pixel_quality: List[float] = field(default_factory=list, init=False)

    compacted_batch: int = -1
    not_compacted_batch: int = -1
    rays_num: int = -1
    mean_effective_samples_per_ray: int = -1
    mean_samples_per_ray: int = -1
    camera_near: float = 0.1
    camera: Camera = field(init=False)
    need_exit: bool = False

    loading_ckpt: bool = False
    ckpt: CKPT = CKPT()

    def __post_init__(self):
        self.data_step = []
        self.data_pixel_quality = []
        self.cur_step = 0

        self.istraining = True

        self.args.exp_dir.mkdir(parents=True, exist_ok=True)
        save_dir = common.backup_current_codebase(self.args.exp_dir, name_prefix="gui-",
                                                  note=self.args.note)
        config_save_path = save_dir.joinpath("config.yaml")
        config_save_path.write_text(tyro.to_yaml(self.args))

        logs_dir = save_dir.joinpath("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.logger = common.setup_logging(
            "nerf.gui",
            file=logs_dir.joinpath("gui.log"),
            with_tensorboard=True,
            level=self.args.common.logging.upper(),
            file_level="DEBUG",
        )
        self.logger.write_hparams(dataclasses.asdict(self.args))

        self.logger.info("code saved to '{}', configurations saved at '{}'".format(
            save_dir,
            config_save_path,
        ))

        # load data
        self.scene_train = data.load_scene(
            srcs=self.args.frames_train,
            scene_options=self.args.scene,
        )
        self.scene_meta = self.scene_train.meta

        # model parameters
        self.nerf_model_train, self.nerf_model_inference, init_input = (
            make_nerf_ngp(bound=self.scene_meta.bound,
                          inference=False,
                          tv_scale=self.args.train.tv_scale),
            make_nerf_ngp(bound=self.scene_meta.bound,
                          inference=True), (jnp.zeros((1, 3), dtype=jnp.float32),
                                            jnp.zeros((1, 3), dtype=jnp.float32),
                                            jnp.zeros((1, self.scene_meta.n_extra_learnable_dims),
                                                      dtype=jnp.float32)))
        self.KEY, key = jran.split(self.KEY, 2)
        self.nerf_variables = self.nerf_model_train.init(key, *init_input)
        if self.args.common.summary:
            print(self.nerf_model_train.tabulate(key, *init_input))

        if self.scene_meta.bg:
            self.bg_model, init_input = (make_skysphere_background_model_ngp(
                bound=self.scene_meta.bound), (jnp.zeros((1, 3),dtype=jnp.float32),
                                               jnp.zeros((1, 3),dtype=jnp.float32),
                                               jnp.zeros((1, self.scene_meta.n_extra_learnable_dims),
                                                      dtype=jnp.float32)))
            self.KEY, key = jran.split(self.KEY, 2)
            self.bg_variables = self.bg_model.init(key, *init_input)

        self.optimizer = make_optimizer(self.args.train.lr)
        if self.ckpt.need_load_ckpt:
            self.load_checkpoint(self.ckpt.ckpt_file_path, self.ckpt.step)
        else:
            self.KEY, key = jran.split(self.KEY, 2)
            self.state = NeRFState.create(
                ogrid=OccupancyDensityGrid.create(
                    cascades=self.scene_meta.cascades,
                    grid_resolution=self.args.raymarch.density_grid_res,
                ),
                raymarch=self.args.raymarch,
                render=self.args.render,
                scene_options=self.args.scene,
                scene_meta=self.scene_meta,
                # unfreeze the frozen dict so that the weight_decay mask can apply, see:
                #   <https://github.com/deepmind/optax/issues/160>
                #   <https://github.com/google/flax/issues/1223>
                nerf_fn=self.nerf_model_train.apply,
                bg_fn=self.bg_model.apply if self.scene_meta.bg else None,
                params={
                    "nerf":
                    self.nerf_variables["params"].unfreeze(),
                    "bg":
                    self.bg_variables["params"].unfreeze()
                    if self.scene_meta.bg else None,
                    "appearance_embeddings": jran.uniform(
                        key=key,
                        shape=(len(self.scene_meta.frames), self.scene_meta.n_extra_learnable_dims),
                        dtype=jnp.float32,
                        minval=-1,
                        maxval=1,
                    )
                },
                tx=self.optimizer,
            )
            self.state = self.state.mark_untrained_density_grid()
        self.camera = Camera(
            width=self.args.viewport.W,
            height=self.args.viewport.H,
            fx=self.scene_meta.camera.fx,
            fy=self.scene_meta.camera.fy,
            cx=self.args.viewport.W / 2,
            cy=self.args.viewport.H / 2,
            near=self.camera_near,
        )

    def set_render_camera(self, _scale, _H, _W) -> Camera:
        self.camera = Camera(
            width=_W,
            height=_H,
            fx=self.scene_meta.camera.fx,
            fy=self.scene_meta.camera.fy,
            cx=_W / 2,
            cy=_H / 2,
            near=self.camera_near,
        )
        self.camera = self.camera.scale_resolution(_scale)

    def render_frame(self, _scale: float, _H: int, _W: int, render_cost: bool):
        self.set_render_camera(_scale, _H, _W)
        #camera pose
        transform = RigidTransformation(
            rotation=self.camera_pose[:3, :3],
            translation=jnp.squeeze(self.camera_pose[:3, 3].reshape(-1, 3),
                                    axis=0))
        self.KEY, key = jran.split(self.KEY, 2)
        bg, rgb, depth, cost = render_image_inference(
            KEY=key,
            transform_cw=transform,
            state=self.state.replace(
                raymarch=self.args.raymarch_eval,
                render=self.args.render_eval.replace(bg=self.back_color),
                nerf_fn=self.nerf_model_inference.apply,
            ),
            camera_override=self.camera,
            render_cost=render_cost)

        bg = self.get_npf32_image(bg,
                                  W=self.args.viewport.W,
                                  H=self.args.viewport.H)
        rgb = self.get_npf32_image(rgb,
                                   W=self.args.viewport.W,
                                   H=self.args.viewport.H)
        depth = self.color_depth(depth,
                                 W=self.args.viewport.W,
                                 H=self.args.viewport.H)
        if render_cost:
            cost = self.get_cost_image(cost,
                                       W=self.args.viewport.W,
                                       H=self.args.viewport.H)
        return (bg, rgb, depth, cost)

    def get_cost_image(self, cost, W, H):
        img = Image.fromarray(np.array(cost, dtype=np.uint8))
        img = img.convert('RGB')
        img = img.resize(size=(W, H), resample=Image.NEAREST)
        cost = np.array(img, dtype=np.float32) / 255.
        return cost

    def color_depth(self, depth, W, H):
        depth = np.array(data.f32_to_u8(data.mono_to_rgb(depth)),
                         dtype=np.uint8)
        img = Image.fromarray(depth, mode='RGBA')
        img = img.convert('RGB')
        img = img.resize(size=(W, H), resample=Image.NEAREST)
        depth = np.array(img, dtype=np.float32) / 255.
        return depth

    def load_checkpoint(self, path: Path, step: int):
        self.loading_ckpt = True
        try:
            if not path.exists():
                raise FileNotFoundError("{} does not exist".format(path))
            self.logger.info("loading checkpoint from '{}'".format(path))
            state: NeRFState = checkpoints.restore_checkpoint(
                path,
                target=NeRFState.empty(
                    raymarch=self.args.raymarch,
                    render=self.args.render,
                    scene_options=self.args.scene,
                    scene_meta=self.scene_meta,
                    nerf_fn=self.nerf_model_train.apply,
                    bg_fn=self.bg_model.apply if self.scene_meta.bg else None,
                    tx=self.optimizer,
                ),
            )
            # WARN:
            #   flax.checkpoints.restore_checkpoint() returns a pytree with all arrays of numpy's array type,
            #   which slows down inference.  use jax.device_put() to move them to jax's default device.
            # REF: <https://github.com/google/flax/discussions/1199#discussioncomment-635132>
            self.state = jax.device_put(state)
            self.state = self.state.mark_untrained_density_grid()
            self.logger.info("checkpoint loaded from '{}'".format(path))
            self.cur_step = step
            self.loading_ckpt = False
            return "checkpoint loaded from '{}'".format(path)
        except BaseException as e:
            self.logger.error(e)
            return e

    def train_steps(self, steps: int) -> Tuple[np.array, np.array, np.array]:
        if self.loading_ckpt:
            return
        gc.collect()
        try:
            if self.istraining:
                if self.last_resample_step < 0 or self.state.step - self.last_resample_step >= 1024:
                    self.KEY, key_resample = jran.split(self.KEY, 2)
                    self.scene_train = self.scene_train.resample_pixels(
                        KEY=key_resample,
                        new_max_mem_mbytes=self.args.scene.max_mem_mbytes,
                    )
                    self.last_resample_step = self.state.step
                self.KEY, key_train = jran.split(self.KEY, 2)
                self.state = self.gui_train_epoch(
                    KEY=key_train,
                    state=self.state,
                    scene=self.scene_train,
                    iters=steps,
                    total_samples=self.args.train.bs,
                    #total_samples=self.args.train.bs,
                    cur_steps=self.cur_step,
                    logger=self.logger,
                )
                self.cur_step = self.cur_step + steps
        except UnboundLocalError as e:
            self.logger.exception(e)

    def get_npf32_image(self, img: jnp.array, W, H) -> np.array:
        img = Image.fromarray(np.array(img, dtype=np.uint8))
        img = img.resize(size=(W, H), resample=Image.NEAREST)
        img = np.array(img, dtype=np.float32) / 255.
        return img

    def gui_train_epoch(
        self,
        KEY: jran.KeyArray,
        state: NeRFState,
        scene: SceneData,
        iters: int,
        total_samples: int,
        cur_steps: int,
        logger: common.Logger,
    ):
        self.log_step = 0
        for _ in (pbar := common.tqdm(range(iters),
                                      desc="Training step#{:03d}".format(cur_steps),
                                      leave=False)):
            if self.need_exit:
                raise KeyboardInterrupt
            if not self.istraining:
                logger.warn("aborted at step {}".format(cur_steps))
                logger.debug("exiting cleanly ...")
                exit()
            KEY, key_perm, key_train_step = jran.split(KEY, 3)
            perm = jran.choice(key_perm,
                               scene.n_pixels,
                               shape=(total_samples, ),
                               replace=True)
            state, metrics = train_step(
                state,
                KEY=key_train_step,
                total_samples=total_samples,
                scene=scene,
                perm=perm,
            )
            self.log_step += 1
            cur_steps = cur_steps + 1
            loss = metrics["loss"]

            self.data_step, self.data_pixel_quality = (  # the 2 lists are ploted so should be updated simultaneously
                self.data_step + [self.log_step + self.cur_step],
                self.data_pixel_quality +
                [data.linear_to_db(loss["rgb"], maxval=1)])

            self.mean_effective_samples_per_ray = metrics["measured_batch_size"] / metrics["n_valid_rays"]
            self.mean_samples_per_ray = metrics["measured_batch_size_before_compaction"] / metrics["n_valid_rays"]

            pbar.set_description_str(
                desc="Training step#{:03d} ".format(cur_steps) + format_metrics(metrics))

            if state.should_call_update_ogrid:
                # update occupancy grid
                for cas in range(state.scene_meta.cascades):
                    KEY, key = jran.split(KEY, 2)
                    state = state.update_ogrid_density(
                        KEY=key,
                        cas=cas,
                        update_all=bool(state.should_update_all_ogrid_cells),
                        max_inference=total_samples,
                    )
                state = state.threshold_ogrid()

            self.compacted_batch = metrics["measured_batch_size"]
            self.not_compacted_batch = metrics[
                "measured_batch_size_before_compaction"]
            self.rays_num = metrics["n_valid_rays"]
            if state.should_write_batch_metrics:
                logger.write_metrics_to_tensorboard(metrics, state.step)

        return state

    def stop_trainer(self):
        self.istraining = False

    def setBackColor(self, color: RGBColor):
        self.back_color = color

    def get_currentStep(self):
        return self.cur_step

    def get_logStep(self):
        return self.log_step

    def get_state(self) -> NeRFState:
        return self.state

    def get_plotData(self):
        return (self.data_step, self.data_pixel_quality)

    def get_effective_samples_nums(self):
        return self.mean_effective_samples_per_ray

    def get_samples_nums(self):
        return self.mean_samples_per_ray

    def get_compactedBatch(self):
        return self.compacted_batch

    def get_notCompactedBatch(self):
        return self.not_compacted_batch

    def get_raysNum(self):
        return self.rays_num


class TrainThread(threading.Thread):

    def __init__(self, KEY, args: NeRFGUIArgs, logger, camera_pose, step,
                 back_color, ckpt):
        super(TrainThread, self).__init__()

        self.KEY = KEY
        self.args = args
        self.logger = logger
        self.camera_pose = camera_pose

        self.istraining = True
        self.needUpdate = True
        self.istesting = False
        self.needtesting = False
        self.step = step
        self.scale = self.args.viewport.resolution_scale

        self.H, self.W = self.args.viewport.H, self.args.viewport.W
        self.back_color = back_color
        self.framebuff = None
        self.rgb = None
        self.depth = None
        self.trainer = None
        self.initFrame()
        self.train_infer_time = -1
        self.render_infer_time = -1
        self.data_step = []
        self.data_pixel_quality = []

        self.compacted_batch = -1
        self.not_compacted_batch = -1
        self.rays_num = -1

        self.frame_updated = False
        self.mode = Mode.Render
        self.havestart = False
        self.ckpt = ckpt

    def initFrame(self):
        frame_init = np.tile(np.asarray(self.back_color, dtype=np.float32),
                             (self.H, self.W, 1))
        self.framebuff = frame_init.copy()
        self.rgb = frame_init.copy()
        self.depth = frame_init.copy()
        self.cost = frame_init.copy()
        self.frame_updated = True

    def setMode(self, mode):
        self.mode = mode

    def setBackColor(self, color: RGBColor):
        self.back_color = color
        if self.trainer:
            self.trainer.setBackColor(self.back_color)

    def run(self):
        try:
            self.trainer = Gui_trainer(KEY=self.KEY,
                                       args=self.args,
                                       logger=self.logger,
                                       camera_pose=self.camera_pose,
                                       back_color=self.back_color,
                                       ckpt=self.ckpt)
        except Exception as e:
            self.logger.exception(e)
            self.needUpdate = False
        while self.needUpdate:
            try:
                if self.istraining and self.trainer:
                    start_time = time.time()
                    self.trainer.train_steps(self.step)
                    end_time = time.time()
                    self.train_infer_time = end_time - start_time
                    self.test()
                if self.istesting and self.needtesting:
                    self.havestart = True
                    start_time = time.time()
                    self.trainer.setBackColor(self.back_color)
                    _, self.rgb, self.depth, self.cost = self.trainer.render_frame(
                        self.scale, self.H, self.W, self.mode == Mode.Cost)
                    if self.mode == Mode.Render:
                        self.framebuff = self.rgb
                    elif self.mode == Mode.depth:
                        self.framebuff = self.depth
                    elif self.mode == Mode.Cost:
                        if self.cost is not None:
                            self.framebuff = self.cost
                        else:
                            self.framebuff = np.tile(np.asarray(self.back_color, dtype=np.float32),
                                                     (self.H, self.W, 1))
                    else:
                        raise NotImplementedError("visualization mode '{}' is not implemented"
                                                  .format(self.mode))
                    self.frame_updated = True
                    end_time = time.time()
                    self.render_infer_time = end_time - start_time
                    self.istesting = False
            except Exception as e:
                self.logger.exception(e)
                break

    def get_TrainInferTime(self):
        if self.train_infer_time != -1:
            return "{:.6f}".format(self.train_infer_time)
        else:
            return "no data"

    def get_RenderInferTime(self):
        if self.render_infer_time != -1:
            return "{:.6f}".format(self.render_infer_time)
        else:
            return "no data"

    def get_Fps(self):
        if self.train_infer_time == -1 and self.render_infer_time == -1:
            return "no data"
        elif self.render_infer_time == -1:
            return "{:.3f}".format(1.0 / (self.train_infer_time))
        elif self.train_infer_time == -1 or not self.istraining:
            return "{:.3f}".format(1.0 / (self.render_infer_time))
        else:
            return "{:.3f}".format(
                1.0 / (self.render_infer_time + self.train_infer_time))

    def get_compactedBatch(self):
        if self.trainer:
            self.compacted_batch = self.trainer.get_compactedBatch()
            if self.compacted_batch != -1:
                return "{:d}".format(self.compacted_batch)
            else:
                return "no data"
        return "no data"

    def get_notCompactedBatch(self):
        if self.trainer:
            self.not_compacted_batch = self.trainer.get_notCompactedBatch()
            if self.not_compacted_batch != -1:
                return "{:d}".format(self.not_compacted_batch)
            else:
                return "no data"
        return "no data"

    def get_raysNum(self):
        if self.trainer:
            self.rays_num = self.trainer.get_raysNum()
            if self.rays_num != -1:
                return "{:d}".format(self.rays_num)
            else:
                return "no data"
        return "no data"

    def stop(self):
        self.istraining = False
        self.needUpdate = False
        if self.trainer:

            self.trainer.stop_trainer()
        thread_id = self.get_id()
        self.logger.debug("throwing training thread exit Exception")
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            thread_id, ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            self.logger.warn("exception raise failure",
                             category=None,
                             stacklevel=1)

    def set_scale(self, _scale):
        self.scale = _scale

    def get_scale(self):
        return self.scale

    def get_id(self):
        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def get_state(self) -> NeRFState:
        return self.trainer.get_state()

    def set_camera_pose(self, camera_pose):
        if self.trainer:
            self.trainer.camera_pose = camera_pose

    def change_WH(self, W, H):
        self.W = W
        self.H = H

    def get_logStep(self):
        if self.trainer:
            return self.trainer.get_logStep()
        return 0

    def get_currentStep(self):
        if self.trainer:
            return self.trainer.get_currentStep()
        return 0

    def get_plotData(self):
        if self.trainer:
            self.data_step, self.data_pixel_quality = self.trainer.get_plotData(
            )
        return (self.data_step, self.data_pixel_quality)

    def get_effective_samples_nums(self):
        if self.trainer:
            return "{:.3f}".format(self.trainer.mean_effective_samples_per_ray)
        else:
            return "no data"

    def get_samples_nums(self):
        if self.trainer:
            return "{:.3f}".format(
                self.trainer.mean_samples_per_ray)
        else:
            return "no data"

    def test(self):
        self.istesting = True

    def finishUpdate(self):
        self.frame_updated = False

    def canUpdate(self):
        return self.frame_updated

    def setStep(self, step):
        self.step = step

    def setCamNear(self, near):
        if self.trainer:
            self.trainer.camera_near = near

    def getPinholeCam(self):
        if self.trainer:
            return self.trainer.camera
        return None


class Mode(Enum):
    Render = 1
    depth = 2
    Cost = 3


@dataclass
class NeRFGUI():
    framebuff: Any = field(init=False)
    H: int = field(init=False)
    W: int = field(init=False)

    need_train: bool = False
    istesting: bool = False
    train_thread: TrainThread = field(init=False)

    args: NeRFGUIArgs = None

    KEY: jran.KeyArray = None
    logger: logging.Logger = None
    cameraPose: CameraPose = CameraPose()
    cameraPosePrev: CameraPose = CameraPose()
    cameraPoseNext: CameraPose = CameraPose()
    scale_slider: Union[int, str] = field(init=False)
    back_color: RGBColor = field(init=False)
    scale: float = field(init=False)
    data_step: List[int] = field(default_factory=list, init=False)
    data_pixel_quality: List[float] = field(default_factory=list, init=False)

    texture_H: int = field(init=False)
    texture_W: int = field(init=False)
    View_H: int = field(init=False)
    View_W: int = field(init=False)

    exit_flag: bool = False
    mode: Mode = Mode.Render

    mouse_pressed: bool = False
    need_test: bool = True

    #ckpt
    ckpt: CKPT = CKPT()

    @property
    def _effective_resolution_display(self) -> str:
        return "{}x{}".format(
            *map(lambda val: int(val * self.scale), (self.W, self.H)))

    def __post_init__(self):
        self.H, self.W = self.args.viewport.H, self.args.viewport.W
        self.back_color = self.args.render_eval.bg
        self.scale = self.args.viewport.resolution_scale
        self.texture_H, self.texture_W = self.H, self.W
        self.framebuff = np.tile(np.asarray(self.back_color, dtype=np.float32),
                                 (self.H, self.W, 3))
        radius_init = 4.
        self.cameraPose, self.cameraPosePrev, self.cameraPoseNext = (
            CameraPose(radius=radius_init),
            CameraPose(radius=radius_init),
            CameraPose(radius=radius_init),
        )
        dpg.create_context()
        self.train_thread = None
        self.ItemsLayout()

    def ItemsLayout(self):

        def callback_backgroundColor():
            self.back_color = tuple(
                map(lambda val: val / 255,
                    dpg.get_value("_BackColor")[:3]))
            self.setFrameColor()

        def callback_mouseDrag(_, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            if not self.need_test:
                return
            dx = app_data[1]
            dy = app_data[2]
            self.cameraPoseNext = deepcopy(self.cameraPosePrev)
            self.cameraPoseNext.move(dx, dy)
            if self.train_thread:
                self.train_thread.set_camera_pose(self.cameraPoseNext.pose)
                self.train_thread.test()
                self.show_cam_angle(self.cameraPoseNext.theta,
                                    self.cameraPoseNext.phi)

        def callback_midmouseDrag(_, app_data):
            if not self.need_test:
                return
            dx = app_data[1]
            dy = app_data[2]
            self.cameraPoseNext = deepcopy(self.cameraPosePrev)
            self.cameraPoseNext.trans(dx, dy)
            if self.train_thread:
                self.train_thread.set_camera_pose(self.cameraPoseNext.pose)
                self.train_thread.test()
                self.show_cam_centroid(self.cameraPoseNext.centroid[0],
                                       self.cameraPoseNext.centroid[1],
                                       self.cameraPoseNext.centroid[2])

        def callback_mouseDown(_, app_data):
            if not dpg.is_item_hovered("_primary_window"):
                return
            if not self.need_test:
                return
            self.mouse_pressed = True
            if app_data[1] < 1e-5:
                self.cameraPosePrev = self.cameraPose
            if self.train_thread:
                self.train_thread.setStep(1)

        def callback_mouseRelease():
            if not self.need_test:
                return
            self.mouse_pressed = False
            self.cameraPose = self.cameraPoseNext
            if self.train_thread:
                self.train_thread.setStep(self.args.train.iters)

        def callback_mouseWheel(_, app_data):
            if not dpg.is_item_hovered("_primary_window"):
                return
            if not self.need_test:
                return
            if self.train_thread:
                self.cameraPose.change_radius(app_data)
                self.train_thread.set_camera_pose(self.cameraPose.pose)
                self.train_thread.test()
                self.show_cam_radius(self.cameraPose.radius)

        def callback_train():
            if self.need_train:
                self.need_train = False
                self.istesting = True
                if self.train_thread:
                    self.train_thread.istraining = False
                _label = "continue" if (self.train_thread != None) else "start"
                dpg.configure_item("_button_train", label=_label)
            else:
                dpg.configure_item("_button_train", label="pause")
                self.need_train = True
                if self.train_thread:
                    self.train_thread.istraining = True
                else:
                    self.train_thread = TrainThread(
                        KEY=self.KEY,
                        args=self.args,
                        logger=self.logger,
                        camera_pose=self.cameraPose.pose,
                        step=self.args.train.iters,
                        back_color=self.back_color,
                        ckpt=self.ckpt)
                    self.train_thread.setDaemon(True)
                    self.train_thread.start()

        def callback_checkpoint(sender):
            if sender == "_button_check_save":
                if self.train_thread and self.train_thread.trainer:
                    self.logger.info("saving training state ... ")
                    ckpt_name = checkpoints.save_checkpoint(
                        self.args.exp_dir,
                        self.train_thread.get_state(),
                        step=self.train_thread.get_currentStep(),
                        overwrite=True,
                        keep=self.args.train.keep,
                    )
                    dpg.set_value(
                        "_log_ckpt",
                        "Checkpoint saved path: {}".format(ckpt_name))
                    self.logger.info(
                        "training state saved to: {}".format(ckpt_name))
                else:
                    dpg.set_value(
                        "_log_ckpt",
                        "Checkpoint save path: failed ,cause no training")
                    self.logger.info(
                        "saving training state failed ,cause no training")

        def callback_change_scale(_, new_scale):
            self.scale = new_scale
            dpg.set_value("_cam_WH", self._effective_resolution_display)
            if self.train_thread:
                self.train_thread.set_scale(self.scale)
                if self.train_thread.havestart:
                    self.train_thread.test()

        def callback_reset():
            self.need_train = False
            if self.train_thread:
                self.train_thread.stop()
                dpg.configure_item("_button_train", label="start")
                self.train_thread = None
            self.framebuff = np.tile(
                np.asarray(self.back_color, dtype=np.float32),
                (self.texture_H, self.texture_W, 3))
            self.clear_plot()
            self.ckpt = CKPT()
            dpg.set_value("_log_ckpt", "")

        def callback_Render():
            if self.need_test:
                dpg.configure_item("_button_Render",
                                   label="continue rendering")
            else:
                dpg.configure_item("_button_Render", label="pause rendering")
            self.need_test = not self.need_test

        def callback_mode(_, app_data):
            if app_data == "render":
                self.mode = Mode.Render
            elif app_data == "depth":
                self.mode = Mode.depth
            elif app_data == "cost":
                self.mode = Mode.Cost
            else:
                raise NotImplementedError("visualization mode '{}' is not implemented"
                                          .format(self.mode))
            if self.train_thread:
                self.train_thread.test()

        def callback_loadCheckpoint(_, app_data):
            file_name = app_data['file_name']
            file_path_name = app_data['file_path_name'][:-2]
            dpg.set_value('_log_ckpt',
                          self.ckpt.parse_ckpt(file_name, file_path_name))

        self.View_W, self.View_H = self.W + self.args.viewport.control_window_width, self.H
        dpg.create_viewport(title='NeRF',
                            width=self.View_W,
                            height=self.View_H,
                            min_width=250 +
                            self.args.viewport.control_window_width,
                            min_height=250,
                            x_pos=0,
                            y_pos=0)

        with dpg.window(tag="_main_window", no_scrollbar=True):
            dpg.set_primary_window("_main_window", True)

            with dpg.file_dialog(directory_selector=False,
                                 show=False,
                                 callback=callback_loadCheckpoint,
                                 tag="checkpoint_file_dialog",
                                 width=700,
                                 height=400):
                dpg.add_file_extension(".*")
                dpg.add_file_extension("",
                                       color=(150, 255, 150, 255),
                                       custom_text="[Checkpoint]")

            with dpg.group(horizontal=True):
                #texture
                with dpg.group(tag="_render_texture"):
                    with dpg.texture_registry(show=False):
                        dpg.add_raw_texture(width=self.W,
                                            height=self.H,
                                            default_value=self.framebuff,
                                            format=dpg.mvFormat_Float_rgb,
                                            tag="_texture")
                    with dpg.child_window(tag="_primary_window",
                                          width=self.W,
                                          no_scrollbar=True):
                        dpg.add_image("_texture",
                                      tag="_img",
                                      parent="_primary_window",
                                      width=self.W - 15,
                                      height=self.H - 32)
                #control panel
                with dpg.child_window(tag="_control_window",
                                      no_scrollbar=True):
                    with dpg.theme() as theme_head:
                        with dpg.theme_component(dpg.mvAll):
                            dpg.add_theme_color(dpg.mvThemeCol_Header,
                                                (0, 62, 89))
                    #control
                    with dpg.collapsing_header(tag="_control_panel",
                                               label="Control Panel",
                                               default_open=True):
                        dpg.bind_item_theme("_control_panel", theme_head)
                        #mode
                        with dpg.group(horizontal=True):
                            dpg.add_text("Visualization mode:   ")
                            items = ["render", "depth", "cost"]
                            dpg.add_combo(items=items,
                                          callback=callback_mode,
                                          width=max(map(len, items)) * 10,
                                          default_value="render")
                        # train / stop/reset
                        with dpg.group(horizontal=True):
                            dpg.add_text("Train: ")
                            dpg.add_button(label="start",
                                           tag="_button_train",
                                           callback=callback_train)
                            dpg.add_button(label="reset",
                                           tag="_button_reset",
                                           callback=callback_reset)
                        #need render
                        with dpg.group(horizontal=True):
                            dpg.add_text("Render: ")
                            dpg.add_button(label="pause rendering",
                                           tag="_button_Render",
                                           callback=callback_Render)
                        # save ckpt
                        with dpg.group(horizontal=True):
                            dpg.add_text("Checkpoint: ")
                            dpg.add_button(label="save",
                                           tag="_button_check_save",
                                           callback=callback_checkpoint)
                            dpg.add_button(label="load",
                                           tag="_button_check_load",
                                           callback=lambda: dpg.show_item(
                                               'checkpoint_file_dialog'))
                        dpg.add_text(
                            "",
                            tag="_log_ckpt",
                            wrap=self.args.viewport.control_window_width - 40)
                        #resolution
                        dpg.add_text("resolution scale:")
                        self.scale_slider = dpg.add_slider_float(
                            tag="_resolutionScale",
                            label="",
                            default_value=self.args.viewport.resolution_scale,
                            clamped=True,
                            min_value=0.1,
                            max_value=1.0,
                            width=self.args.viewport.control_window_width - 40,
                            format="%.1f",
                            callback=callback_change_scale,
                        )
                        dpg.add_text("Background color: ")
                        dpg.add_color_edit(
                            tag="_BackColor",
                            default_value=tuple(
                                map(lambda val: int(val * 255 + .5),
                                    self.args.render_eval.bg)),
                            no_alpha=True,
                            width=self.args.viewport.control_window_width - 40,
                            callback=callback_backgroundColor)
                        with dpg.value_registry():
                            dpg.add_float_value(default_value=0.0,
                                                tag="float_value")
                        #camera
                        dpg.add_text("camera set:")
                        with dpg.group(horizontal=True):
                            dpg.add_text("near plane")
                            dpg.add_input_text(tag="_camera_near",
                                               width=40,
                                               default_value=0.1,
                                               decimal=True)
                        with dpg.group(horizontal=True):
                            dpg.add_text("centroid:")
                            dpg.add_text("x")
                            dpg.add_input_text(
                                tag="_centroid_x",
                                width=40,
                                default_value=self.cameraPose.centroid[0],
                                decimal=True)
                            dpg.add_text("y")
                            dpg.add_input_text(
                                tag="_centroid_y",
                                width=40,
                                default_value=self.cameraPose.centroid[1],
                                decimal=True)
                            dpg.add_text("z")
                            dpg.add_input_text(
                                tag="_centroid_z",
                                width=40,
                                default_value=self.cameraPose.centroid[2],
                                decimal=True)
                        with dpg.group(horizontal=True):
                            dpg.add_text("theta")
                            dpg.add_input_text(
                                tag="_theta",
                                width=40,
                                default_value=self.cameraPose.theta,
                                decimal=True)
                            dpg.add_text("phi")
                            dpg.add_input_text(
                                tag="_phi",
                                width=40,
                                default_value=self.cameraPose.phi,
                                decimal=True)
                            dpg.add_text("radius")
                            dpg.add_input_text(
                                tag="_radius",
                                width=40,
                                default_value=self.cameraPose.radius,
                                decimal=True)
                    with dpg.collapsing_header(tag="_para_panel",
                                               label="Parameter Monitor",
                                               default_open=True):
                        dpg.bind_item_theme("_para_panel", theme_head)
                        with dpg.group(horizontal=True):
                            dpg.add_text("Resolution(W*H): ")
                            dpg.add_text(self._effective_resolution_display,
                                         tag="_cam_WH")
                        with dpg.group(horizontal=True):
                            dpg.add_text("Current training step: ")
                            dpg.add_text("no data", tag="_cur_train_step")
                        with dpg.group(horizontal=True):
                            dpg.add_text("Train time: ")
                            dpg.add_text("no data", tag="_log_train_time")
                        with dpg.group(horizontal=True):
                            dpg.add_text("Infer time: ")
                            dpg.add_text("no data", tag="_log_infer_time")
                        with dpg.group(horizontal=True):
                            dpg.add_text("FPS: ")
                            dpg.add_text("no data", tag="_fps")
                        with dpg.group(horizontal=True):
                            dpg.add_text("Mean samples/ray: ")
                            dpg.add_text("no data", tag="_samples")
                        with dpg.group(horizontal=True):
                            dpg.add_text("Mean effective samples/ray: ")
                            dpg.add_text("no data", tag="_effective_samples")

                        with dpg.group(horizontal=True):
                            dpg.add_text("Batch size: ")
                            dpg.add_text("no data",
                                         tag="_not_compacted_batch_size")
                        with dpg.group(horizontal=True):
                            dpg.add_text("Batch size(compacted): ")
                            dpg.add_text("no data",
                                         tag="_compacted_batch_size")
                        with dpg.group(horizontal=True):
                            dpg.add_text("Number of rays: ")
                            dpg.add_text("no data", tag="_rays_num")
                        # create plot
                        with dpg.plot(
                                label="pixel quality",
                                height=self.args.viewport.control_window_width
                                - 40,
                                width=self.args.viewport.control_window_width -
                                40):
                            # optionally create legend
                            dpg.add_plot_legend()
                            # REQUIRED: create x and y axes
                            dpg.add_plot_axis(dpg.mvXAxis,
                                              label="step",
                                              tag="x_axis")
                            dpg.add_plot_axis(dpg.mvYAxis,
                                              label="PSNR (estimated)",
                                              tag="y_axis")
                            # series belong to a y axis
                            dpg.add_line_series(self.data_step,
                                                self.data_pixel_quality,
                                                label="~PSNR",
                                                parent="y_axis",
                                                tag="_plot")
                    with dpg.collapsing_header(tag="_tip_panel",
                                               label="Tips",
                                               default_open=True):
                        dpg.bind_item_theme("_tip_panel", theme_head)
                        tip1 = "* Drag the left mouse button to rotate the camera\n"
                        tip2 = "* The mouse wheel zooms the distance between the camera and the object\n"
                        tip3 = "* Drag the window to resize\n"
                        tip4 = "* Drag the middle mouse button to translate the camera\n"
                        dpg.add_text(
                            tip1,
                            wrap=self.args.viewport.control_window_width - 40)
                        dpg.add_text(
                            tip2,
                            wrap=self.args.viewport.control_window_width - 40)
                        dpg.add_text(
                            tip3,
                            wrap=self.args.viewport.control_window_width - 40)
                        dpg.add_text(
                            tip4,
                            wrap=self.args.viewport.control_window_width - 40)

        def callback_key(_, appdata):
            if appdata == dpg.mvKey_Q:
                self.exit_flag = True

        #IO
        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left,
                                       callback=callback_mouseDrag)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left,
                                          callback=callback_mouseRelease)
            dpg.add_mouse_down_handler(button=dpg.mvMouseButton_Left,
                                       callback=callback_mouseDown)
            dpg.add_mouse_wheel_handler(callback=callback_mouseWheel)
            #mouse middle
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle,
                                       callback=callback_midmouseDrag)
            dpg.add_mouse_down_handler(button=dpg.mvMouseButton_Middle,
                                       callback=callback_mouseDown)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Middle,
                                          callback=callback_mouseRelease)

            dpg.add_key_release_handler(callback=callback_key)

        dpg.setup_dearpygui()
        dpg.show_viewport()

    def update_frame(self):
        self.framebuff = self.train_thread.framebuff
        dpg.set_value("_texture", self.framebuff)

    def adapt_size(self):
        if self.View_H != dpg.get_viewport_height(
        ) or self.View_W != dpg.get_viewport_width():
            self.View_H = dpg.get_viewport_height()
            self.View_W = dpg.get_viewport_width()
            self.H, self.W = self.View_H, self.View_W - self.args.viewport.control_window_width
            dpg.set_item_width("_primary_window", self.W)
            dpg.delete_item("_img")
            dpg.add_image("_texture",
                          tag="_img",
                          parent="_primary_window",
                          width=self.W - 15,
                          height=self.H - 32)
            dpg.configure_item("_control_panel",
                               label="Control Panel",
                               default_open=True)
            dpg.configure_item("_para_panel",
                               label="Parameter Monitor",
                               default_open=True)
            dpg.configure_item("_tip_panel", label="Tips", default_open=True)
            dpg.set_value("_cam_WH", self._effective_resolution_display)
            if self.train_thread:
                self.train_thread.test()

    def setFrameColor(self):
        if self.train_thread:
            self.train_thread.setBackColor(self.back_color)
        if self.train_thread and self.train_thread.havestart:
            self.train_thread.test()
        else:
            self.framebuff = np.tile(
                np.asarray(self.back_color, dtype=np.float32),
                (self.texture_H, self.texture_W, 1))
        dpg.set_value("_texture", self.framebuff)

    def clear_plot(self):
        self.data_step.clear()
        self.data_pixel_quality.clear()
        self.update_plot()

    def update_plot(self):
        if len(self.data_pixel_quality
               ) > self.args.viewport.max_show_loss_step:
            self.data_pixel_quality = self.data_pixel_quality[
                -self.args.viewport.max_show_loss_step - 1:]
            self.data_step = self.data_step[
                -self.args.viewport.max_show_loss_step - 1:]
        dpg.set_value('_plot', [self.data_step, self.data_pixel_quality])
        dpg.fit_axis_data("y_axis")
        dpg.fit_axis_data("x_axis")

    def set_cam_angle(self):
        try:
            theta = float(dpg.get_value("_theta"))
            phi = float(dpg.get_value("_phi"))
            radius = float(dpg.get_value("_radius"))
            if theta != self.cameraPose.theta or phi != self.cameraPose.phi or radius != self.cameraPose.radius:
                self.cameraPose.theta = theta
                self.cameraPose.phi = phi
                self.cameraPose.radius = radius
                self.train_thread.set_camera_pose(self.cameraPose.pose)
                self.train_thread.test()
        except BaseException as e:
            self.logger.error(e)

    def show_cam_angle(self, _theta, _phi):
        _theta = float('{:.3f}'.format(_theta))
        _phi = float('{:.3f}'.format(_phi))
        try:
            theta = float(dpg.get_value("_theta"))
            phi = float(dpg.get_value("_phi"))
            if theta != _theta or phi != _phi:
                dpg.set_value("_theta", _theta)
                dpg.set_value("_phi", _phi)
        except BaseException as e:
            self.logger.error(e)

    def show_cam_radius(self, _radius):
        _radius = float('{:.3f}'.format(_radius))
        try:
            radius = float(dpg.get_value("_radius"))
            if radius != _radius:
                dpg.set_value("_radius", _radius)
        except BaseException as e:
            self.logger.error(e)

    def set_cam_centroid(self):
        try:
            x = float(dpg.get_value("_centroid_x"))
            y = float(dpg.get_value("_centroid_y"))
            z = float(dpg.get_value("_centroid_z"))
            if x != self.cameraPose.centroid[
                    0] or y != self.cameraPose.centroid[
                        1] or z != self.cameraPose.centroid[2]:
                self.cameraPose.centroid[0] = x
                self.cameraPose.centroid[1] = y
                self.cameraPose.centroid[2] = z
                self.train_thread.set_camera_pose(self.cameraPose.pose)
                self.train_thread.test()
        except BaseException as e:
            self.logger.error(e)

    def set_cam_near(self):
        try:
            cam_near = float(dpg.get_value("_camera_near"))
            camera = self.train_thread.getPinholeCam()
            if camera and camera.near != cam_near:
                self.train_thread.setCamNear(cam_near)
                self.train_thread.test()
        except BaseException as e:
            self.logger.exception(e)

    def show_cam_centroid(self, _x, _y, _z):
        _x = float('{:.3f}'.format(_x))
        _y = float('{:.3f}'.format(_y))
        _z = float('{:.3f}'.format(_z))
        try:
            x = float(dpg.get_value("_centroid_x"))
            y = float(dpg.get_value("_centroid_y"))
            z = float(dpg.get_value("_centroid_z"))
            if x != _x or y != _y or z != _z:
                dpg.set_value("_centroid_x", _x)
                dpg.set_value("_centroid_y", _y)
                dpg.set_value("_centroid_z", _z)
        except BaseException as e:
            self.logger.error(e)

    def update_panel(self):
        dpg.set_value(
            "_cur_train_step",
            "{} (+{}/{})".format(self.train_thread.get_currentStep(),
                                 self.train_thread.get_logStep(),
                                 self.train_thread.step))
        dpg.set_value("_log_train_time",
                      "{}".format(self.train_thread.get_TrainInferTime()))
        dpg.set_value("_log_infer_time",
                      "{}".format(self.train_thread.get_RenderInferTime()))
        dpg.set_value("_fps", "{}".format(self.train_thread.get_Fps()))
        dpg.set_value("_samples",
                      "{}".format(self.train_thread.get_samples_nums()))
        dpg.set_value(
            "_effective_samples",
            "{}".format(self.train_thread.get_effective_samples_nums()))

        dpg.set_value("_compacted_batch_size",
                      "{}".format(self.train_thread.get_compactedBatch()))
        dpg.set_value("_not_compacted_batch_size",
                      "{}".format(self.train_thread.get_notCompactedBatch()))
        dpg.set_value("_rays_num",
                      "{}".format(self.train_thread.get_raysNum()))
        self.data_step, self.data_pixel_quality = self.train_thread.get_plotData(
        )
        self.update_plot()

    def load_ckpt(self):
        if self.train_thread and self.train_thread.havestart:
            self.train_thread.trainer.load_checkpoint(self.ckpt.ckpt_file_path,
                                                      self.ckpt.step)
            self.clear_plot()
            self.ckpt.need_load_ckpt = False

    def render(self) -> int:
        while dpg.is_dearpygui_running():
            self.adapt_size()
            if self.train_thread:
                if self.ckpt.need_load_ckpt:
                    self.load_ckpt()
                if not self.mouse_pressed:
                    self.set_cam_angle()
                    self.set_cam_centroid()
                    self.set_cam_near()
                self.train_thread.setMode(self.mode)
                self.train_thread.set_scale(self.scale)
                self.train_thread.change_WH(self.W, self.H)
                self.update_panel()
                if self.need_test:
                    self.train_thread.needtesting = True
                    if self.train_thread.canUpdate():
                        self.update_frame()
                        self.train_thread.finishUpdate()
                else:
                    self.train_thread.needtesting = False
            else:
                dpg.set_value("_texture", self.framebuff)
            dpg.render_dearpygui_frame()
            time.sleep(.01 if self.train_thread and self.train_thread.istraining else .1)
            if self.exit_flag:
                if self.train_thread:
                    self.train_thread.stop()
                while self.train_thread and self.train_thread.is_alive():
                    pass
                self.logger.debug("thread killed successfully")
                self.logger.info("exiting cleanly ...")
                break
        dpg.destroy_context()
        return 0


def gui_exit():
    import sys
    sys.exit()


def GuiWindow(KEY: jran.KeyArray, args: NeRFGUIArgs, logger: logging.Logger) -> int:
    nerfGui = NeRFGUI(args=args, KEY=KEY, logger=logger)
    return nerfGui.render()
