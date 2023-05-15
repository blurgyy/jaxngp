import logging
from pathlib import Path
import numpy as np
from typing import List,Any,Tuple,Union
import jax.random as jran
import jax.numpy as jnp
from dataclasses import dataclass,field
from tqdm import tqdm
import threading
import dearpygui.dearpygui as dpg
import ctypes
from utils.args import GuiWindowArgs, NeRFTrainingArgs,GuiWindowArgs
from .train import *
from utils.data import load_transform_json_recursive,merge_transforms
from utils.types import (
    SceneData,
    SceneMeta,
    ViewMetadata,
    TransformJsonNeRFSynthetic,
    TransformJsonNGP
)
from models.nerfs import (NeRF,SkySphereBg)


@dataclass
class CameraPose():
    theta:float=160.0
    phi:float=-30.0
    radius:float=4.0
    def pose_spherical(self,theta, phi, radius):
        trans_t=lambda t: np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,t],
        [0,0,0,1]],np.float32)
        rot_phi=lambda phi:np.array([
            [1,0,0,0],
            [0,np.cos(phi),-np.sin(phi),0],
            [0,np.sin(phi),np.cos(phi),0], 
            [0,0,0,1]],np.float32)
        rot_theta= lambda theta:np.array([
            [np.cos(theta),0,-np.sin(theta),0],
            [0,1,0,0],
            [np.sin(theta),0,np.cos(theta),0],
            [0,0,0,1]],np.float32)
        c2w=trans_t(radius)
        c2w=np.matmul(rot_phi(phi/180.*np.pi),c2w)
        c2w=np.matmul(rot_theta(theta/180.*np.pi),c2w)
        c2w =np.matmul(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) , c2w) 
        return c2w
    @property
    def pose(self):
        return jnp.asarray(self.pose_spherical(self.theta,self.phi,self.radius))
    def move(self,dx,dy):
        velocity=0.01
        self.theta+=velocity*dx
        self.phi-=velocity*dy
        return self.pose
    
@dataclass
class Gui_trainer():
    KEY: jran.KeyArray
    args: NeRFTrainingArgs
    logger: common.Logger
    camera_pose:jnp.array
    gui_args:GuiWindowArgs
    H:int
    W:int
    
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
    
    istraining:bool=field(init=False)
    def __post_init__(self):
        
        self.istraining=True
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
            scene_options=self.args.scene,
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
            batch_config=NeRFBatchConfig.create(
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
        self.state=self.state.mark_untrained_density_grid()
        self.cur_step=0
    def render_frame(self):
        self.logger.info("render_frame")
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
        return (bg, rgb, depth)
    def train_steps(self,steps:int,_scale:float)->Tuple[np.array,np.array,np.array]:
        gc.collect()
        
        if self.cur_step<self.gui_args.max_step and self.istraining:
            
            _scene_meta = self.scene_meta
            #_scale=self.gui_args.resolution_scale
            _camera=_scene_meta.camera.scale_resolution(_scale)
            self.test_scene_meta=SceneMeta(bound=_scene_meta.bound,
                                    bg=_scene_meta.bg,
                                    camera=_camera,
                                    frames=_scene_meta.frames)
            
            self.KEY, key = jran.split(self.KEY, 2)
            loss_log, self.state = self.gui_train_epoch(
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
        return self.render_frame()
    def get_npf32_image(self,img:jnp.array)->np.array:
        from PIL import Image
        img=Image.fromarray(np.array(img,dtype=np.uint8))
        img=img.resize(size=(self.W,self.H), resample=1)
        img=np.array(img,dtype=np.float32)/255.
        return img    
        
    def gui_train_epoch(
        self,
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
        
        for _ in (pbar := tqdm(range(n_batches), desc="Training epoch#{:03d}".format(cur_steps), bar_format=common.tqdm_format)):
            if not self.istraining:
                logger.warn("aborted at step {}".format(cur_steps))
                # logger.info("saving training state ... ")
                # ckpt_name = checkpoints.save_checkpoint(args.exp_dir, state, step="ep{}aborted".format(ep_log), overwrite=True, keep=2**30)
                # logger.info("training state of epoch {} saved to: {}".format(ep_log, ckpt_name))
                logger.info("exiting cleanly ...")
                exit()
                #break
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

            loss_db = data.linear_to_db(loss_log, maxval=1)
            
            pbar.set_description_str(
                desc="Training step#{:03d} batch_size={}/{} samp./ray={:.1f}/{:.1f} n_rays={} loss={:.3e}({:.2f}dB)".format(
                    cur_steps,
                    metrics["measured_batch_size"],
                    metrics["measured_batch_size_before_compaction"],
                    state.batch_config.running_mean_effective_samples_per_ray,
                    state.batch_config.running_mean_samples_per_ray,
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

            state = state.update_batch_config(
                new_measured_batch_size=metrics["measured_batch_size"],
                new_measured_batch_size_before_compaction=metrics["measured_batch_size_before_compaction"],
            )
            if state.should_commit_batch_config:
                state = state.replace(batch_config=state.batch_config.commit(total_samples))

            if state.should_write_batch_metrics:
                logger.write_scalar("batch/↓loss", loss_log, state.step)
                logger.write_scalar("batch/↑loss (db)", loss_db, state.step)
                logger.write_scalar("batch/effective batch size (not compacted)", metrics["measured_batch_size_before_compaction"], state.step)
                logger.write_scalar("batch/↑effective batch size (compacted)", metrics["measured_batch_size"], state.step)
                logger.write_scalar("rendering/↓effective samples per ray", state.batch_config.mean_effective_samples_per_ray, state.step)
                logger.write_scalar("rendering/↓marched samples per ray", state.batch_config.mean_samples_per_ray, state.step)
                logger.write_scalar("rendering/↑number of rays", state.batch_config.n_rays, state.step)
    
            
        return total_loss / n_processed_rays, state
    def stop_trainer(self):
        self.istraining=False
class TrainThread(threading.Thread):
    def __init__(self,KEY,args,gui_args,logger,camera_pose,step,H,W):
        super(TrainThread,self).__init__()   
        
        self.KEY=KEY
        self.args=args
        self.gui_args=gui_args
        self.logger=logger
        self.camera_pose=camera_pose
        self.scale=gui_args.resolution_scale
        
        self.istraining=True
        self.needUpdate=True
        try:
           
            self.trainer=Gui_trainer(KEY=self.KEY,args=self.args,logger=self.logger,camera_pose=self.camera_pose,gui_args=self.gui_args,H=H,W=W)
            self.step=step
            self.scale=gui_args.resolution_scale
            
            self.framebuff=np.ones(shape=(H,W,3),dtype=np.float32)
        except Exception as e:
            self.logger.warning(e)
    def run(self):
        try:
            while self.needUpdate:
                while self.istraining and self.trainer.cur_step<self.gui_args.max_step:
                    _,self.framebuff,_=self.trainer.train_steps(self.step,self.scale)
                _,self.framebuff,_=self.trainer.render_frame()
                #self.logger.info("training finished")  
        except Exception as e:
            self.logger.warning(e)
    def render(self):
        self.trainer.render_frame()  

    def stop(self):
        self.istraining=False
        self.needUpdate=False
        self.trainer.stop_trainer()
        thread_id = self.get_id() 
        self.logger.warning("Throw training thread exit Exception")
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 
            ctypes.py_object(SystemExit)) 
        if res > 1: 
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0) 
            self.logger.warn("Exception raise failure", category=None, stacklevel=1)
    def set_scale(self,_scale):
        self.scale=_scale
    def get_scale(self):
        return self.scale
    def get_id(self): 
    # returns id of the respective thread 
        if hasattr(self, '_thread_id'): 
            return self._thread_id 
        for id, thread in threading._active.items(): 
            if thread is self: 
                return id
    def get_state(self):
        return self.trainer.state
    def set_camera_pose(self,camera_pose):
        self.trainer.camera_pose=camera_pose
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
    cameraPose:CameraPose=field(init=False)
    scale_slider:Union[int,str]=field(init=False)
    def init_HW(self):
        def try_image_extensions(
            file_path: str,
            extensions: List[str]=["png", "jpg", "jpeg"],) -> Path:
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
        srcs = list(map(Path, self.train_args.frames_train))
        transforms = merge_transforms(map(load_transform_json_recursive, srcs))
        if transforms is None:
            raise FileNotFoundError("could not find any valid transforms in {}".format(srcs))
        if len(transforms.frames) == 0:
            raise ValueError("could not find any frame in {}".format(srcs))
        if isinstance(transforms, TransformJsonNeRFSynthetic):
            from PIL import Image
            _img = Image.open(try_image_extensions(transforms.frames[0].file_path))
            self.W, self.H = _img.width, _img.height
        elif isinstance(transforms, TransformJsonNGP):
            self.W,self.H=transforms.w,transforms.h
        else:
            raise TypeError("unexpected type for transforms: {}, expected one of {}".format(
                type(transforms),
                [TransformJsonNeRFSynthetic, TransformJsonNGP],
            ))
    def __post_init__(self):
        self.train_args=NeRFTrainingArgs(frames_train=self.gui_args.frames_train,exp_dir=self.gui_args.exp_dir)
        self.init_HW()
        self.framebuff=np.ones(shape=(self.W,self.H,3),dtype=np.float32)#default background is white
        dpg.create_context()
        self.ItemsLayout()
        self.train_thread=None
        self.cameraPose=CameraPose()
    def ItemsLayout(self):
        
        def callback_mouseDrag(sender,app_data):
            if not dpg.is_item_focused("_primary_window"):
                return 
            dx=app_data[1]
            dy=app_data[2]
            
            self.logger.info("dx:{},dy:{}".format(dx,dy))
            self.cameraPose.move(dx,dy)
            if self.train_thread:
                self.train_thread.set_camera_pose(self.cameraPose.pose)
            
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
                    # train / stop/reset
                    with dpg.group(horizontal=True):
                        dpg.add_text("Train: ")
                        def callback_train(sender, app_data):
                            if self.need_train:
                                self.need_train = False
                                if self.train_thread:
                                    self.train_thread.istraining=False
                                #self.train_thread.stop()
                                _label="continue" if (self.train_thread!=None) else "start"
                                dpg.configure_item("_button_train", label=_label)
                            else:
                                dpg.configure_item("_button_train", label="pause")
                                self.need_train = True
                                if self.train_thread:
                                    self.train_thread.istraining=True
                                else:
                                    self.train_thread=TrainThread(KEY=self.KEY,args=self.train_args,gui_args=self.gui_args,logger=self.logger,camera_pose=self.cameraPose.pose,step=5,H=self.H,W=self.W)
                                    self.train_thread.setDaemon(True)
                                    self.train_thread.start()

                        dpg.add_button(label="start", tag="_button_train", callback=callback_train)
                        dpg.bind_item_theme("_button_train", theme_button)
                        def callback_reset(sender, app_data):
                            self.need_train=True
                            if self.train_thread:
                                self.train_thread.stop()
                                dpg.configure_item("_button_train", label="start")
                                self.train_thread=None
                            self.framebuff=np.ones(shape=(self.H,self.W,3),dtype=np.float32)
                        dpg.add_button(label="reset", tag="_button_reset", callback=callback_reset)
                    
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
                self.scale_slider=dpg.add_slider_float(tag="_resolutionScale",label="",default_value=self.gui_args.resolution_scale,clamped=True,min_value=0.1,max_value=1.0)
        #drag
        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left,callback=callback_mouseDrag)
        dpg.setup_dearpygui()
        dpg.show_viewport()
    def change_scale(self):    
        if dpg.get_value(self.scale_slider)!=self.train_thread.get_scale():
            self.train_thread.set_scale(dpg.get_value(self.scale_slider))
    def update_frame(self):
        self.framebuff=self.train_thread.framebuff
    def render(self):
        try:
            while dpg.is_dearpygui_running():
                if self.train_thread:
                    self.change_scale()
                    self.update_frame()
                dpg.set_value("_texture", self.framebuff)
                dpg.render_dearpygui_frame()
        except KeyboardInterrupt:
            if self.train_thread:
                self.train_thread.stop()
            self.logger.info("exiting cleanly ...")
            exit()    
        dpg.destroy_context()


def gui_exit(): 
    import sys
    sys.exit()

def GuiWindow(KEY: jran.KeyArray, args: GuiWindowArgs, logger: logging.Logger):
    # import keyboard
    # keyboard.add_hotkey('q', gui_exit)
    nerfGui=NeRFGUI(gui_args=args,KEY=KEY,logger=logger)
    nerfGui.render()
    
