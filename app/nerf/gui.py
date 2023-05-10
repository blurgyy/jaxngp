import logging
from pathlib import Path,PosixPath
import numpy as np
from typing import List, Literal, Optional,Any,Tuple,Union
import jax.random as jran
import jax.numpy as jnp
from dataclasses import dataclass,field
from tqdm import tqdm

import threading

import dearpygui.dearpygui as dpg
from utils.args import GuiWindowArgs

import matplotlib.image as mpimg
from utils.args import NeRFTestingArgs, NeRFTrainingArgs,GuiWindowArgs

from .train import *

_state:NeRFState


from utils.types import (
    SceneData,
    SceneMeta,
    ViewMetadata,
    PinholeCamera
)
from models.nerfs import (NeRF,SkySphereBg)
@dataclass
class Gui_trainer():
    KEY: jran.KeyArray
    args: NeRFTrainingArgs
    logger: common.Logger
    camera_pose:jnp.array
    gui_args:GuiWindowArgs
    
    scene_train:SceneData=field(init=False)
    scene_val:SceneData=field(init=False)
    scene_meta:SceneMeta=field(init=False)
    test_scene_meta:SceneMeta=field(init=False)
    val_views:ViewMetadata=field(init=False)
    
    nerf_model:NeRF=field(init=False)
    init_input: tuple=field(init=False)
    nerf_variables: Any=field(init=False)
    bg_model: SkySphereBg=field(init=False)
    bg_variables: Any=field(init=False)
    optimizer: Any=field(init=False)
    
    state:NeRFState=field(init=False)
    cur_step:int=field(init=False)
    loss_log:str="--"
    

    def __post_init__(self):
        self.cur_step=0
        logs_dir = self.args.exp_dir.joinpath("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.logger = common.setup_logging(
        "nerf.train",
        file=logs_dir.joinpath("train.log"),
        with_tensorboard=True,
        level=self.args.common.logging.upper(),
        file_level="DEBUG",
        )
        self.args.exp_dir.joinpath("config.yaml").write_text(tyro.to_yaml(self.args))
        self.logger.write_hparams(dataclasses.asdict(self.args))
        self.logger.info("configurations saved to '{}'".format(self.args.exp_dir.joinpath("config.yaml")))

        #load data
        self.scene_train, _ = data.load_scene(
            srcs=self.args.frames_train,
            world_scale=self.args.scene.world_scale,
            image_scale=self.args.scene.image_scale,
        )
        self.scene_meta=self.scene_train.meta
        #self.scene_meta.camera=self.scene_meta.replace(camera=_camera)
        # model parameters
        self.nerf_model, self.init_input = (
            make_nerf_ngp(bound=self.scene_meta.bound),
            (jnp.zeros((1, 3), dtype=jnp.float32), jnp.zeros((1, 3), dtype=jnp.float32))
        )
        self.KEY, key = jran.split(self.KEY, 2)
        self.nerf_variables =self.nerf_model.init(key, *self.init_input)
        if self.args.common.summary:
            print(self.nerf_model.tabulate(key, *self.init_input))
        
        if self.scene_meta.bg:
            self.bg_model, self.init_input = (
                make_skysphere_background_model_ngp(bound=self.scene_meta.bound),
                (jnp.zeros((1, 3), dtype=jnp.float32), jnp.zeros((1, 3), dtype=jnp.float32))
            )
            self.KEY, key = jran.split(self.KEY, 2)
            self.bg_variables = self.bg_model.init(key, *self.init_input)
            
        lr_sch = optax.exponential_decay(
        init_value=self.args.train.lr,
        transition_steps=10_000,
        decay_rate=1/3,  # decay to `1/3 * init_lr` after `transition_steps` steps
        staircase=True,  # use integer division to determine lr drop step
        transition_begin=10_000,  # hold the initial lr value for the initial 10k steps (but first lr drop happens at 20k steps because `staircase` is specified)
        end_value=self.args.train.lr / 100,  # stop decaying at `1/100 * init_lr`
        )
        self.optimizer = optax.adamw(
            learning_rate=lr_sch,
            b1=0.9,
            b2=0.99,
            # paper:
            #   the small value of 𝜖 = 10^{−15} can significantly accelerate the convergence of the
            #   hash table entries when their gradients are sparse and weak.
            eps=1e-15,
            eps_root=1e-15,
            # In NeRF experiments, the network can converge to a reasonably low loss during the
            # frist ~50k training steps (with 1024 rays per batch and 1024 samples per ray), but the
            # loss becomes NaN after about 50~150k training steps.
            # paper:
            #   To prevent divergence after long training periods, we apply a weak L2 regularization
            #   (factor 10^{−6}) to the neural network weights, ...
            weight_decay=1e-6,
            # paper:
            #   ... to the neural network weights, but not to the hash table entries.
            mask={
                "nerf": {
                    "density_mlp": True,
                    "rgb_mlp": True,
                    "position_encoder": False,
                },
                "bg": self.scene_meta.bg,
            },
        )
        
         # training state
        self.state = NeRFState.create(
            ogrid=OccupancyDensityGrid.create(
                        cascades=self.scene_meta.cascades,
                        grid_resolution=self.args.raymarch.density_grid_res,
                    ),
            batch_config=NeRFBatchConfig(
                mean_effective_samples_per_ray=self.args.raymarch.diagonal_n_steps,
                mean_samples_per_ray=self.args.raymarch.diagonal_n_steps,
                n_rays=self.args.train.bs // self.args.raymarch.diagonal_n_steps,
            ),
            raymarch=self.args.raymarch,
            render=self.args.render,
            scene_options=self.args.scene,
            scene_meta=self.scene_meta,
            # unfreeze the frozen dict so that the weight_decay mask can apply, see:
            #   <https://github.com/deepmind/optax/issues/160>
            #   <https://github.com/google/flax/issues/1223>
            nerf_fn=self.nerf_model.apply,
            bg_fn=self.bg_model.apply if self.scene_meta.bg else None,
            params={
                "nerf": self.nerf_variables["params"].unfreeze(),
                "bg": self.bg_variables["params"].unfreeze() if self.scene_meta.bg else None,
            },
            tx=self.optimizer,
        )
        self.cur_step=0
   
    def train_steps(self,steps:int,_scale:float)->Tuple[np.array,np.array,np.array]:
        gc.collect()
        
        if self.cur_step<self.gui_args.max_step:
            
            _scene_meta = self.scene_meta
            #_scale=self.gui_args.resolution_scale
            _camera=PinholeCamera(
                                H=int(_scene_meta.camera.H*_scale),
                                W=int(_scene_meta.camera.H*_scale),
                                fx=int(_scene_meta.camera.fx*_scale),
                                fy=int(_scene_meta.camera.fy*_scale),
                                cx=int(_scene_meta.camera.cx*_scale),
                                cy=int(_scene_meta.camera.cy*_scale)
                                )
            self.test_scene_meta=SceneMeta(bound=_scene_meta.bound,
                                    bg=_scene_meta.bg,
                                    camera=_camera,
                                    frames=_scene_meta.frames)
            
            self.KEY, key = jran.split(self.KEY, 2)
            loss_log, self.state = gui_train_epoch(
                KEY=key,
                state=self.state,
                scene=self.scene_train,
                n_batches=steps,
                total_samples=self.args.train.bs,
                cur_steps=self.cur_step,
                logger=self.logger,
            )
            self.cur_step=self.cur_step+steps
            self.loss_log=str(loss_log)
        loss_db = data.linear_to_db(loss_log, maxval=1)
        self.logger.info("epoch#{:03d}: loss={:.2e}({:.2f}dB)".format(self.cur_step, loss_log, loss_db))

        #camera pose
        transform = RigidTransformation(rotation=self.camera_pose[:3, :3],
                                        translation=jnp.squeeze(self.camera_pose[:3, 3].reshape(-1,3),axis=0))
        self.KEY, key = jran.split(self.KEY, 2)
        bg, rgb, depth = render_image_inference(
            KEY=key,
            transform_cw=transform,
            state=self.state.replace(render=self.state.render.replace(random_bg=False),
                                     scene_meta=self.test_scene_meta)
        )
        bg=self.get_npf32_image(bg)
        rgb=self.get_npf32_image(rgb)
        depth=self.get_npf32_image(depth)
        # bg=np.array(bg,dtype=np.float32)/255.
        # rgb=np.array(rgb,dtype=np.float32)/255.
        # depth=np.array(depth,dtype=np.float32)/255.
        return (bg, rgb, depth)
    
    def get_npf32_image(self,img:jnp.array)->np.array:
        from PIL import Image
        img=Image.fromarray(np.array(img,dtype=np.uint8))
        img=img.resize(size=(800,800), resample=1)
        img=np.array(img,dtype=np.float32)/255.
        self.logger.info("img shape:{}".format(img.shape))
        return img    
        
def gui_train_epoch(
    KEY: jran.KeyArray,
    state: NeRFState,
    scene: SceneData,
    n_batches: int,
    total_samples: int,
    cur_steps: int,
    logger: common.Logger,
):
    n_processed_rays = 0
    total_loss = 0
    running_mean_effective_samp_per_ray = state.batch_config.mean_effective_samples_per_ray
    running_mean_samp_per_ray = state.batch_config.mean_samples_per_ray

    for _ in (pbar := tqdm(range(n_batches), desc="Training epoch#{:03d}".format(cur_steps), bar_format=common.tqdm_format)):
        KEY, key_perm, key_train_step = jran.split(KEY, 3)
        perm = jran.choice(key_perm, scene.meta.n_pixels, shape=(state.batch_config.n_rays,), replace=True)
        state, metrics = train_step(
            KEY=key_train_step,
            state=state,
            total_samples=total_samples,
            scene=scene,
            perm=perm,
        )
        cur_steps=cur_steps+1
        n_processed_rays += state.batch_config.n_rays
        total_loss += metrics["loss"]
        loss_log = metrics["loss"] / state.batch_config.n_rays
        running_mean_samp_per_ray, running_mean_effective_samp_per_ray = (
            running_mean_samp_per_ray * .95 + .05 * metrics["measured_batch_size_before_compaction"] / state.batch_config.n_rays,
            running_mean_effective_samp_per_ray * .95 + .05 * metrics["measured_batch_size"] / state.batch_config.n_rays,
        )

        loss_db = data.linear_to_db(loss_log, maxval=1)
        
        pbar.set_description_str(
            desc="Training step#{:03d} batch_size={}/{} samp./ray={}/{} n_rays={} loss={:.3e}({:.2f}dB)".format(
                cur_steps,
                metrics["measured_batch_size"],
                metrics["measured_batch_size_before_compaction"],
                state.batch_config.mean_effective_samples_per_ray,
                state.batch_config.mean_samples_per_ray,
                state.batch_config.n_rays,
                loss_log,
                loss_db,
            )
        )
 
        if state.should_call_update_ogrid:
            # update occupancy grid
            for cas in range(state.scene_meta.cascades):
                KEY, key = jran.split(KEY, 2)
                state = state.update_ogrid_density(
                    KEY=key,
                    cas=cas,
                    update_all=bool(state.should_update_all_ogrid_cells),
                )
            state = state.threshold_ogrid()

        if state.should_update_batch_config:
            new_mean_effective_samples_per_ray = int(running_mean_effective_samp_per_ray + 1.5)
            new_mean_samples_per_ray = int(running_mean_samp_per_ray + 1.5)
            new_n_rays = total_samples // new_mean_samples_per_ray
            state = state.replace(
                batch_config=NeRFBatchConfig(
                    mean_effective_samples_per_ray=new_mean_effective_samples_per_ray,
                    mean_samples_per_ray=new_mean_samples_per_ray,
                    n_rays=new_n_rays,
                ),
            )

        if state.should_write_batch_metrics:
            logger.write_scalar("batch/↓loss", loss_log, state.step)
            logger.write_scalar("batch/↑loss (db)", loss_db, state.step)
            logger.write_scalar("batch/effective batch size (not compacted)", metrics["measured_batch_size_before_compaction"], state.step)
            logger.write_scalar("batch/↑effective batch size (compacted)", metrics["measured_batch_size"], state.step)
            logger.write_scalar("rendering/↓effective samples per ray", state.batch_config.mean_effective_samples_per_ray, state.step)
            logger.write_scalar("rendering/↓marched samples per ray", state.batch_config.mean_samples_per_ray, state.step)
            logger.write_scalar("rendering/↑number of rays", state.batch_config.n_rays, state.step)
        
        
    return total_loss / n_processed_rays, state

class TrainThread(threading.Thread):
    def __init__(self,KEY,args,gui_args,logger,camera_pose,step):
        super(TrainThread,self).__init__()   
        self.istraining=True
        self.KEY=KEY
        self.args=args
        self.gui_args=gui_args
        self.logger=logger
        self.camera_pose=camera_pose
        self.trainer=Gui_trainer(KEY=self.KEY,args=self.args,logger=self.logger,camera_pose=self.camera_pose,gui_args=self.gui_args)
        self.step=step
        self.scale=gui_args.resolution_scale
        
        self.framebuff=np.ones(shape=(self.gui_args.H,self.gui_args.W,3),dtype=np.float32)
    def run(self):
        while self.istraining and self.trainer.cur_step<self.gui_args.max_step:
            _,self.framebuff,_=self.trainer.train_steps(self.step,self.scale)
        self.logger.info("training finished")    
    def stop(self):
        self.istraining=False
    def set_scale(self,_scale):
        self.scale=_scale
    

@dataclass
class NeRFGUI():
    
    framebuff:Any= field(init=False)
    H:int= field(init=False)
    W:int= field(init=False)
    isTest:bool=False
    need_train:bool=False
    need_update:bool=False
    train_thread:TrainThread= field(init=False)
    
    train_args: NeRFTrainingArgs= field(init=False)
    gui_args:GuiWindowArgs=None
    
    KEY: jran.KeyArray=None
    logger: logging.Logger=None
    camera_pose:jnp.array=None
    
    trainer:Gui_trainer=field(init=False)
    
    scale_slider:Union[int,str]=field(init=False)
    
    def __post_init__(self):
        #self.scale_slider=dpg.generate_uuid()
        self.train_args=NeRFTrainingArgs(frames_train=self.gui_args.frames_train,exp_dir=self.gui_args.exp_dir)
        self.H,self.W=self.gui_args.H,self.gui_args.W
        self.framebuff=np.ones(shape=(self.W,self.H,3),dtype=np.float32)#default background is white
        dpg.create_context()
        self.ItemsLayout()
        self.train_thread=None
        self.camera_pose=jnp.asarray([
                [
                    0.3681268095970154,
                    0.2467726171016693,
                    -0.8964295387268066,
                    -3.6136231422424316
                ],
                [
                    -0.929775595664978,
                    0.09770487248897552,
                    -0.3549240529537201,
                    -1.4307446479797363
                ],
                [
                    -7.450580596923828e-09,
                    0.9641353487968445,
                    0.2654109597206116,
                    1.0699057579040527
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ])
        self.trainer=None
        
    def ItemsLayout(self):
        # def resolutionControls():
        #     dpg.add_text("resolution scale:",parent=)
        #     dpg.add_slider_float(label="",parent=,width=300,default_value=self.step_sleep,clamped=True)
        
    
        dpg.create_viewport(title='NeRf', width=self.W, height=self.H)
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(width=self.W, height=self.H,default_value=self.framebuff, format=dpg.mvFormat_Float_rgb, tag="_texture")

        with dpg.window(tag="_primary_window", width=self.W, height=self.H):
            dpg.add_image("_texture")
        dpg.set_primary_window("_primary_window", True)


        #control
        with dpg.window(label="Control", tag="_control_window", width=400, height=300):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)
            # time
            if not self.isTest:
                with dpg.group(horizontal=True):
                    dpg.add_text("Train time: ")
                    dpg.add_text("no data", tag="_log_train_time")
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")
            with dpg.group(horizontal=True):
                dpg.add_text("SPP: ")
                dpg.add_text("1", tag="_log_spp")
            # train button
            if not self.isTest:
                with dpg.collapsing_header(label="Train", default_open=True):
                    # train / stop
                    with dpg.group(horizontal=True):
                        dpg.add_text("Train: ")
                        def callback_train(sender, app_data):
                            if self.need_train:
                                self.logger.info("set need_train:{}".format(self.need_train))
                                self.need_train = False
                                self.train_thread.stop()
                                dpg.configure_item("_button_train", label="start")
                            else:
                                
                                dpg.configure_item("_button_train", label="stop")
                                self.need_train = True
                                self.train_thread=TrainThread(KEY=self.KEY,args=self.train_args,gui_args=self.gui_args,logger=self.logger,camera_pose=self.camera_pose,step=5)
                                self.train_thread.start()

                        dpg.add_button(label="start", tag="_button_train", callback=callback_train)
                        dpg.bind_item_theme("_button_train", theme_button)
                    # save ckpt
                    with dpg.group(horizontal=True):
                        dpg.add_text("Checkpoint: ")

                        def callback_save(sender, app_data):
                            pass
                        dpg.add_button(label="save", tag="_button_save", callback=callback_save)
                        dpg.bind_item_theme("_button_save", theme_button)

                        dpg.add_text("", tag="_log_ckpt")
            #resolution
            with dpg.group(horizontal=True):
                dpg.add_text("resolution scale:")
                self.scale_slider=dpg.add_slider_float(tag="_resolutionScale",label="",default_value=self.gui_args.resolution_scale,clamped=True,min_value=0.0,max_value=1.0)
        dpg.setup_dearpygui()
        dpg.show_viewport()
    def train_step(self):
        self.framebuff=self.train_thread.framebuff
        dpg.set_value("_texture", self.framebuff)

            
    def render(self):
        while dpg.is_dearpygui_running():
            if self.need_train and self.train_thread: 
                self.train_thread.set_scale(dpg.get_value(self.scale_slider))
                self.train_step() 
                
            dpg.render_dearpygui_frame()
        dpg.destroy_context()



def GuiWindow(KEY: jran.KeyArray, args: GuiWindowArgs, logger: logging.Logger):
    nerfGui=NeRFGUI(gui_args=args,KEY=KEY,logger=logger)
    nerfGui.render()
    