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
    TransformJsonNGP,
    PinholeCamera
)
from ._utils import(
    color_float2int,
    color_int2float
)
from models.nerfs import (NeRF,SkySphereBg)
# from guis import *
from PIL import Image
import time

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
        velocity=0.08
        self.theta+=velocity*dx
        self.phi-=velocity*dy
        return self.pose
    def change_radius(self,rate):
        self.radius*=1.1**(-rate)
        return self.pose
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
    val_views:ViewMetadata=field(init=False)
    
    nerf_model:NeRF=field(init=False)
    init_input: tuple=field(init=False)
    nerf_variables: Any=field(init=False)
    bg_model: SkySphereBg=field(init=False)
    bg_variables: Any=field(init=False)
    optimizer: Any=field(init=False)
    
    state:NeRFState=field(init=False)
    cur_step:int=0
    log_step:int=0
    loss_log:str="--"
    
    istraining:bool=field(init=False)
    back_color:Tuple[float,float,float]=(1.0,1.0,1.0)
    
    data_step:List[int]=field(default_factory=list,init=False)
    data_loss:List[float]=field(default_factory=list,init=False)
    
    compacted_batch:int=-1
    not_compacted_batch:int=-1
    rays_num:int=-1
    
    def __post_init__(self):
        self.data_step=[]
        self.data_loss=[]
        self.cur_step=0
        
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
        

    def set_render_camera(self,_scale,_H,_W)->PinholeCamera:
        _scene_meta = self.scene_meta
        camera = PinholeCamera(
            W=_W,
            H=_H,
            fx=_scene_meta.camera.fx,
            fy=_scene_meta.camera.fy,
            cx=_W/ 2,
            cy=_H / 2
        )
        camera=camera.scale_resolution(_scale)
        return camera
        
    def render_frame(self,_scale:float,_H:int,_W:int):
        self.logger.info("update frame which resolution scale is {},resolution is H:{},W:{}".format(_scale,_H,_W))
        camera=self.set_render_camera(_scale,_H,_W)
        #camera pose
        transform = RigidTransformation(rotation=self.camera_pose[:3, :3],
                                        translation=jnp.squeeze(self.camera_pose[:3, 3].reshape(-1,3),axis=0))
        self.KEY, key = jran.split(self.KEY, 2)
        bg, rgb, depth = render_image_inference(
            KEY=key,
            transform_cw=transform,
            state=self.state.replace(render=self.state.render.replace(random_bg=False,bg=self.back_color)),
            #state=self.state.replace(render=self.state.render.replace(random_bg=False)),
            camera_override=camera
        )
        bg=self.get_npf32_image(bg,W=self.gui_args.W,H=self.gui_args.H)
        rgb=self.get_npf32_image(rgb,W=self.gui_args.W,H=self.gui_args.H)
        depth=self.get_npf32_image(depth,W=self.gui_args.W,H=self.gui_args.H)
        return (bg, rgb, depth)

    def train_steps(self,steps:int)->Tuple[np.array,np.array,np.array]:
        gc.collect()
        
        if self.cur_step<self.gui_args.max_step and self.istraining:
            self.KEY, key = jran.split(self.KEY, 2)
            loss_log, self.state = self.gui_train_epoch(
                KEY=key,
                state=self.state,
                scene=self.scene_train,
                n_batches=steps,
                total_samples=self.gui_args.bs,
                #total_samples=self.args.train.bs,
                cur_steps=self.cur_step,
                logger=self.logger,
            )
            self.cur_step=self.cur_step+steps
            self.loss_log=str(loss_log)
        loss_db = data.linear_to_db(loss_log, maxval=1)
        self.logger.info("epoch#{:03d}: loss={:.2e}({:.2f}dB)".format(self.cur_step, loss_log, loss_db))
        #return self.render_frame(_scale)
    def get_npf32_image(self,img:jnp.array,W,H)->np.array:
        img=Image.fromarray(np.array(img,dtype=np.uint8))
        img=img.resize(size=(W,H), resample=Image.NEAREST)
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
        self.log_step=0
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
            self.log_step+=1
            cur_steps=cur_steps+1
            n_processed_rays += state.batch_config.n_rays
            total_loss += metrics["loss"]
            loss_log = metrics["loss"] / state.batch_config.n_rays

            loss_db = data.linear_to_db(loss_log, maxval=1)
            self.data_step.append(self.log_step+self.cur_step)
            temp_db=float(loss_log)
            # temp_db=np.array(loss_db,dtype=np.float32)
            self.data_loss.append(temp_db)
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
            self.compacted_batch=metrics["measured_batch_size_before_compaction"]
            self.not_compacted_batch=metrics["measured_batch_size"]
            self.rays_num=state.batch_config.n_rays
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
    
    def setBackColor(self,color:Tuple[float,float,float]):
        self.back_color=color
    def get_currentStep(self):
        return self.cur_step
    def get_logStep(self):
        return self.log_step
    def get_state(self):
        return self.state
    def get_plotData(self):
        return(self.data_step,self.data_loss)
    def get_effective_samples_nums(self):
        return self.get_state().batch_config.running_mean_effective_samples_per_ray
    def get_samples_nums(self):
        return self.get_state().batch_config.running_mean_samples_per_ray
    
    def get_compactedBatch(self):
        return self.compacted_batch
    def get_notCompactedBatch(self):
        return self.not_compacted_batch
    def get_raysNum(self):
        return self.rays_num
class TrainThread(threading.Thread):
    def __init__(self,KEY,args,gui_args,logger,camera_pose,step,back_color):
        super(TrainThread,self).__init__()   
        
        self.KEY=KEY
        self.args=args
        self.gui_args=gui_args
        self.logger=logger
        self.camera_pose=camera_pose
        self.scale=gui_args.resolution_scale
        
        self.istraining=True
        self.needUpdate=True
        self.istesting=False
        self.step=step
        self.scale=gui_args.resolution_scale
        
        self.H,self.W=self.gui_args.H,self.gui_args.W
        self.back_color=back_color
        self.framebuff=None
        self.trainer=None
        self.initFrame()
        self.train_infer_time=-1
        self.render_infer_time=-1
        self.data_step=[]
        self.data_loss=[]
        
        self.compacted_batch=-1
        self.not_compacted_batch=-1
        self.rays_num=-1
        try:   
            pass
            #self.trainer=Gui_trainer(KEY=self.KEY,args=self.args,logger=self.logger,camera_pose=self.camera_pose,gui_args=self.gui_args,H=H,W=W)
        except Exception as e:
            self.logger.warning(e)
    def initFrame(self):
        img=Image.new("RGB",(self.W,self.H),color_float2int(self.back_color))
        self.framebuff=np.array(img,dtype=np.float32)/255.
        
    def setBackColor(self,color:Tuple[float,float,float]):
        self.back_color=color
        if self.trainer:
            self.trainer.setBackColor(self.back_color)
    def run(self):
        self.trainer=Gui_trainer(KEY=self.KEY,args=self.args,logger=self.logger,camera_pose=self.camera_pose,gui_args=self.gui_args,back_color=self.back_color)
        try:
            while self.needUpdate:
                if self.istraining and self.trainer and self.trainer.cur_step<self.gui_args.max_step:
                    start_time=time.time()
                    self.trainer.train_steps(self.step)
                    end_time=time.time()
                    self.train_infer_time=end_time-start_time
                    self.test()
                    # start_time=time.time()
                    # _,self.framebuff,_=self.trainer.render_frame(self.scale,self.H,self.W)
                    # end_time=time.time()
                    # self.render_infer_time=end_time-start_time
                if self.istesting:
                    start_time=time.time()    
                    _,self.framebuff,_=self.trainer.render_frame(self.scale,self.H,self.W)
                    end_time=time.time()
                    self.render_infer_time=end_time-start_time
                    self.istesting=False
        except Exception as e:
            self.logger.error(e)
    def get_TrainInferTime(self):
        if self.train_infer_time!=-1:
            return "{:.6f}".format(self.train_infer_time)
        else:
            return "no data"
    def get_RenderInferTime(self):
        if self.render_infer_time!=-1:
            return "{:.6f}".format(self.render_infer_time)
        else:
            return "no data"
    def get_Fps(self):
        if self.train_infer_time==-1 and self.render_infer_time==-1:
            return "no data"
        elif self.render_infer_time==-1 :
            return "{:.3f}".format(1.0/(self.train_infer_time))
        elif self.train_infer_time==-1 or not self.istraining:
            return "{:.3f}".format(1.0/(self.render_infer_time))
        else:
            return "{:.3f}".format(1.0/(self.render_infer_time+self.train_infer_time))

    def get_compactedBatch(self):
        if self.trainer:
            self.compacted_batch=self.trainer.get_compactedBatch()
            if self.compacted_batch!=-1:
                return "{:d}".format(self.compacted_batch)
            else:
                return "no data"
        return "no data"
    def get_notCompactedBatch(self):
        if self.trainer:
            self.not_compacted_batch=self.trainer.get_notCompactedBatch()
            if self.not_compacted_batch!=-1:
                return "{:d}".format(self.not_compacted_batch)
            else:
                return "no data"
        return "no data"
    def get_raysNum(self):
        if self.trainer:
            self.rays_num=self.trainer.get_raysNum()
            if self.rays_num!=-1:
                return "{:d}".format(self.rays_num)
            else:
                return "no data"
        return "no data"
    def stop(self):
        self.istraining=False
        self.needUpdate=False
        if self.trainer:
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
        return self.trainer.get_state()
    def set_camera_pose(self,camera_pose):
        if self.trainer:
            self.trainer.camera_pose=camera_pose
    def change_WH(self,W,H):
        self.W=W
        self.H=H
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
            self.data_step,self.data_loss=self.trainer.get_plotData()
        return(self.data_step,self.data_loss)
    def get_effective_samples_nums(self):
        if self.trainer:
            return "{:.3f}".format(self.get_state().batch_config.running_mean_effective_samples_per_ray)
        else :
            return "no data"
    def get_samples_nums(self):
        if self.trainer:
            return "{:.3f}".format(self.get_state().batch_config.running_mean_samples_per_ray)
        else :
            return "no data"
    def test(self):
        self.istesting=True
@dataclass
class NeRFGUI():
    
    framebuff:Any= field(init=False)
    H:int= field(init=False)
    W:int= field(init=False)

    need_train:bool=False
    istesting:bool=False
    train_thread:TrainThread= field(init=False)
    
    train_args: NeRFTrainingArgs= field(init=False)
    gui_args:GuiWindowArgs=None
    
    KEY: jran.KeyArray=None
    logger: logging.Logger=None
    cameraPose:CameraPose=field(init=False)
    scale_slider:Union[int,str]=field(init=False)
    back_color:Tuple[float,float,float]=(1.0,1.0,1.0)
    scale:float=1.0
    data_step:List[int]=field(default_factory=list,init=False)
    data_loss:List[float]=field(default_factory=list,init=False)
    
    texture_H:int= field(init=False)
    texture_W:int= field(init=False)
    View_H:int= field(init=False)
    View_W:int= field(init=False)
    dx:float=0.0
    dy:float=0.0

    def __post_init__(self):
        self.train_args=NeRFTrainingArgs(frames_train=self.gui_args.frames_train,exp_dir=self.gui_args.exp_dir)

        self.H,self.W=self.gui_args.H,self.gui_args.W
        self.texture_H,self.texture_W=self.H,self.W
        self.framebuff=np.ones(shape=(self.W,self.H,3),dtype=np.float32)#default background is white
        dpg.create_context()
        self.ItemsLayout()
        self.train_thread=None
        self.cameraPose=CameraPose()
    def ItemsLayout(self):
        def callback_backgroundColor():
            self.back_color= color_int2float(dpg.get_value("_BackColor"))
            self.logger.info("get color:{}".format(color_int2float(dpg.get_value("_BackColor"))))
            self.setFrameColor()
        def callback_mouseDrag(sender,app_data):
            if not dpg.is_item_focused("_primary_window"):
                return 
            dx=app_data[1]
            dy=app_data[2]
            if self.dx!=dx or self.dy!=dy:
                self.dx,self.dy=dx,dy
                self.logger.info("dx:{},dy:{}".format(dx,dy))
                self.cameraPose.move(dx,dy)
                if self.train_thread:
                    self.train_thread.set_camera_pose(self.cameraPose.pose)
                    self.train_thread.test()
              

        def callback_mouseRelease(sender,app_data):
            if not dpg.is_item_focused("_primary_window"):
                return 
    
            self.dx,self.dy=0.0,0.0
            
        def callback_mouseWheel(sender,app_data):
            if not dpg.is_item_focused("_primary_window"):
                return 
            delta=app_data
            self.cameraPose.change_radius(delta)
            self.logger.info("self.cameraPose.radius:{}".format(self.cameraPose.radius))
            if self.train_thread:
                self.train_thread.set_camera_pose(self.cameraPose.pose)
                self.train_thread.test()
        def callback_train(sender, app_data):
            if self.need_train:
                self.need_train = False
                self.istesting=True
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
                    self.train_thread=TrainThread(KEY=self.KEY,args=self.train_args,gui_args=self.gui_args,logger=self.logger,
                                                    camera_pose=self.cameraPose.pose,step=self.gui_args.train_steps,back_color=self.back_color)
                    self.train_thread.setDaemon(True)
                    self.train_thread.start()
        def callback_save(sender, app_data):
            if self.train_thread and self.train_thread.trainer:
                self.logger.info("saving training state ... ")
                ckpt_name = checkpoints.save_checkpoint(
                    self.gui_args.exp_dir,
                    self.train_thread.get_state(),
                    step=self.train_thread.get_currentStep(),
                    overwrite=True,
                    keep=self.gui_args.keep,
                )
                #self.gui_args.keep+=1
                dpg.set_value("_log_ckpt", "Checkpoint saved path: {}".format(ckpt_name))
                self.logger.info("training state of step {} saved to: {}".format(self.train_thread.get_logStep(), ckpt_name))
            else:
                dpg.set_value("_log_ckpt", "Checkpoint save path: failed ,cause no training")
                self.logger.info("saving training state failed ,cause no training")
        def callback_reset(sender, app_data):
            self.need_train=True
            if self.train_thread:
                self.train_thread.stop()
                dpg.configure_item("_button_train", label="start")
                self.train_thread=None
            self.framebuff=np.ones(shape=(self.texture_H,self.texture_W,3),dtype=np.float32) 
            self.data_step.clear()
            self.data_loss.clear() 
            self.update_plot()        
                  
        self.View_W,self.View_H=self.W+self.gui_args.control_window_width,self.H
        dpg.create_viewport(title='NeRF', width=self.View_W, height=self.View_H,
                            min_width=250+self.gui_args.control_window_width,min_height=250,x_pos=0, y_pos=0)
        with dpg.window(tag="_main_window",
               ) as main_window:
            dpg.set_primary_window("_main_window", True)
            with dpg.group(horizontal=True):
                #texture
                with dpg.group(tag="_render_texture"):
                    with dpg.texture_registry(show=False):
                        dpg.add_raw_texture(width=self.W, height=self.H,default_value=self.framebuff, format=dpg.mvFormat_Float_rgb, tag="_texture")
                    with dpg.child_window(tag="_primary_window", width=self.W,no_scrollbar=True):
                        dpg.add_image("_texture",tag="_img",parent="_primary_window")
                #control panel
                with dpg.child_window(tag="_control_window",no_scrollbar=True):
                    with dpg.theme() as theme_head:
                        with dpg.theme_component(dpg.mvAll):
                            dpg.add_theme_color(dpg.mvThemeCol_Header, (0,62,89))
                    #control
                    with dpg.collapsing_header(tag="_conrol_panel",label="Control Panel", default_open=True):
                        dpg.bind_item_theme("_conrol_panel", theme_head)
                        # train / stop/reset
                        with dpg.group(horizontal=True):
                            dpg.add_text("Train: ")
                            dpg.add_button(label="start", tag="_button_train", callback=callback_train)
                            dpg.add_button(label="reset", tag="_button_reset", callback=callback_reset)            
                        # save ckpt
                        with dpg.group(horizontal=True):
                            dpg.add_text("Checkpoint: ")
                            dpg.add_button(label="save", tag="_button_save", callback=callback_save)
                        dpg.add_text("", tag="_log_ckpt",wrap=self.gui_args.control_window_width-40)     
                        #resolution
                        dpg.add_text("resolution scale:")
                        self.scale_slider=dpg.add_slider_float(tag="_resolutionScale",label="",default_value=self.gui_args.resolution_scale,
                                                            clamped=True,min_value=0.1,max_value=1.0,width=self.gui_args.control_window_width-40)
                        dpg.add_color_edit(tag="_BackColor",label="Background color", default_value=[255, 255, 255], no_alpha=True,
                                                         width=self.gui_args.control_window_width-40, callback=callback_backgroundColor)
                    with dpg.collapsing_header(tag="_para_panel",label="Parameter Monitor", default_open=True):
                        dpg.bind_item_theme("_para_panel", theme_head)
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
                            dpg.add_text("SPP: ")
                            dpg.add_text("1", tag="_log_spp")
                        with dpg.group(horizontal=True):
                            dpg.add_text("Mean samples/ray: ")
                            dpg.add_text("no data", tag="_samples")
                        with dpg.group(horizontal=True):
                            dpg.add_text("Mean effective samples/ray: ")
                            dpg.add_text("no data", tag="_effective_samples")
                        
                        with dpg.group(horizontal=True):
                            dpg.add_text("not compacted batch size: ")
                            dpg.add_text("no data", tag="_not_compacted_batch_size")
                        with dpg.group(horizontal=True):
                            dpg.add_text("compacted batch size: ")
                            dpg.add_text("no data", tag="_compacted_batch_size")
                        with dpg.group(horizontal=True):
                            dpg.add_text("number of rays: ")
                            dpg.add_text("no data", tag="_rays_num")
                        # create plot
                        with dpg.plot(label="Loss", height=self.gui_args.control_window_width-40, width=self.gui_args.control_window_width-40):
                            # optionally create legend
                            dpg.add_plot_legend()

                            # REQUIRED: create x and y axes
                            dpg.add_plot_axis(dpg.mvXAxis, label="step",tag="x_axis")
                            dpg.add_plot_axis(dpg.mvYAxis, label="loss", tag="y_axis")
                            # series belong to a y axis
                            dpg.add_line_series(self.data_step, self.data_loss, label="loss", parent="y_axis",tag="_plot")
                    with dpg.collapsing_header(tag="_tip_panel",label="Tips", default_open=True):
                        dpg.bind_item_theme("_tip_panel", theme_head)
                        tip1="* Drag the left mouse button to change the camera perspective\n"
                        tip2="* The mouse wheel zooms the distance between the camera and the object\n"
                        tip3="* Drag the window to resize\n"
                        dpg.add_text(tip1,wrap=self.gui_args.control_window_width-40)
                        dpg.add_text(tip2,wrap=self.gui_args.control_window_width-40)
                        dpg.add_text(tip3,wrap=self.gui_args.control_window_width-40)
        
        #drag       
        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left,callback=callback_mouseDrag)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left,callback=callback_mouseRelease)
            dpg.add_mouse_wheel_handler(callback=callback_mouseWheel)
        dpg.setup_dearpygui()
        dpg.show_viewport()
    def change_scale(self):    
        if dpg.get_value(self.scale_slider)!=self.scale:
            self.scale=dpg.get_value(self.scale_slider)
            self.train_thread.set_scale(self.scale)
            self.train_thread.test()
    def update_frame(self):
        self.framebuff=self.train_thread.framebuff
    def adapt_size(self):
        if self.View_H!=dpg.get_viewport_height() or self.View_W!=dpg.get_viewport_width():
            self.View_H=dpg.get_viewport_height()
            self.View_W=dpg.get_viewport_width()
            self.H,self.W=self.View_H,self.View_W-self.gui_args.control_window_width
            # dpg.set_item_width("_main_window",self.View_W)
            # dpg.set_item_height("_main_window",self.View_H)
            dpg.set_item_width("_primary_window",self.W)
            dpg.set_item_height("_primary_window",self.H) 
            dpg.delete_item("_img")
            dpg.add_image("_texture",tag="_img",parent="_primary_window",width=self.W, height=self.H)
            if self.train_thread:
                self.train_thread.test()
            #dpg.configure_item("_texture",width=self.W, height=self.H,default_value=self.framebuff, format=dpg.mvFormat_Float_rgb)
    def setFrameColor(self):
        if self.train_thread and self.train_thread.trainer:
            self.train_thread.setBackColor(self.back_color)
            self.train_thread.test()
        else:              
            img=Image.new("RGB",(self.texture_W,self.texture_H),color_float2int(self.back_color))
            self.framebuff=np.array(img,dtype=np.float32)/255.
        dpg.set_value("_texture", self.framebuff)
    def update_plot(self):
        dpg.set_value('_plot', [self.data_step,self.data_loss])
        dpg.fit_axis_data("y_axis")
        dpg.fit_axis_data("x_axis") 
    
    def update_panel(self):
        dpg.set_value("_cur_train_step","{} (+{}/{})".format(self.train_thread.get_currentStep(),
                                                             self.train_thread.get_logStep(),
                                                             self.gui_args.train_steps))
        dpg.set_value("_log_train_time","{}".format(self.train_thread.get_TrainInferTime()))
        dpg.set_value("_log_infer_time","{}".format(self.train_thread.get_RenderInferTime()))
        dpg.set_value("_fps","{}".format(self.train_thread.get_Fps()))
        dpg.set_value("_samples","{}".format(self.train_thread.get_samples_nums()))
        dpg.set_value("_effective_samples","{}".format(self.train_thread.get_effective_samples_nums()))
        
        dpg.set_value("_compacted_batch_size","{}".format(self.train_thread.get_compactedBatch()))
        dpg.set_value("_not_compacted_batch_size","{}".format(self.train_thread.get_notCompactedBatch()))
        dpg.set_value("_rays_num","{}".format(self.train_thread.get_raysNum()))
        self.data_step,self.data_loss=self.train_thread.get_plotData()
        self.update_plot()
    
    def render(self):
        try:       
            while dpg.is_dearpygui_running():
                self.adapt_size()
                #self.setFrameColor()
                if self.train_thread:
                    self.train_thread.change_WH(self.W,self.H)
                    self.change_scale()
                    # if self.istesting:
                    #     self.train_thread.test()
                    self.update_frame()
                    self.update_panel()
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
    
