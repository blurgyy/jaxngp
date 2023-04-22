#import sys
#sys.path.append("/home/cad_83/E/chenyingxi/jaxngp")

import logging
from pathlib import Path
import numpy as np
from typing import List, Literal, Optional,Any
import jax.random as jran
import jax.numpy as jnp
from dataclasses import dataclass,field

import threading

import dearpygui.dearpygui as dpg
from utils.args import GuiWindowArgs

import matplotlib.image as mpimg
from utils.args import NeRFTestingArgs, NeRFTrainingArgs,GuiWindowArgs


class TrainThread(threading.Thread):
    # istraining:bool=False
    # KEY: jran.KeyArray=None
    # args: GuiWindowArgs=None
    # logger: logging.Logger=None
    def __init__(self,istraining,KEY,args,logger):
        super(TrainThread,self).__init__()   
        self.istraining=istraining
        self.KEY=KEY
        self.args=args
        self.logger=logger
    def run(self):
        from app.nerf.train import train
        train(self.KEY, self.args, self.logger)
        for i in range(1000):
            print(i)
        import time
        while self.istraining:
            print(time.time())
            time.sleep(3)
    def stop(self):
        self.istraining=False
    

@dataclass
class NerfGUI():
    
    framebuff:Any
    H:int=300
    W:int=600
    isTest:bool=False
    training:bool=False
    need_update:bool=False
    train_thread:Any= field(init=False)
    
    train_args: NeRFTrainingArgs=None
    test_args:NeRFTestingArgs=None
    
    KEY: jran.KeyArray=None
    logger: logging.Logger=None
    camera_pose:jnp.array=None
    
    def __post_init__(self):
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
        #self.train_thread=TrainThread(self.training,KEY=self.KEY,args=self.args,logger=self.logger)
        #self.train_thread=TrainThread(istraining=self.training,KEY=self.KEY,args=self.args,logger=self.logger)
        #self.train_thread.start()
        #train_thread.join()
        
    def ItemsLayout(self):
        dpg.create_viewport(title='Nerf', width=self.W, height=self.H)
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
                                    if self.training:
                                        self.training = False
                                        dpg.configure_item("_button_train", label="start")
                                    else:
                                        self.training = True
                                        dpg.configure_item("_button_train", label="stop")
                                        self.train_thread=TrainThread(self.training,KEY=self.KEY,args=self.train_args,logger=self.logger)
                                        self.train_thread.start()
                                        #self.train_thread.join()

                                dpg.add_button(label="start", tag="_button_train", callback=callback_train)
                                dpg.bind_item_theme("_button_train", theme_button)
                            # save ckpt
                            with dpg.group(horizontal=True):
                                dpg.add_text("Checkpoint: ")

                                def callback_save(sender, app_data):
                                    
                                    self.camera_pose=jnp.array([
                                                                [
                                                                    0.30901381373405457,
                                                                    -0.5914012789726257,
                                                                    0.7448187470436096,
                                                                    3.002460479736328
                                                                ],
                                                                [
                                                                    0.9510576128959656,
                                                                    0.19215571880340576,
                                                                    -0.24200350046157837,
                                                                    -0.9755473136901855
                                                                ],
                                                                [
                                                                    0.0,
                                                                    0.7831478714942932,
                                                                    0.621835470199585,
                                                                    2.5066988468170166
                                                                ],
                                                                [
                                                                    0.0,
                                                                    0.0,
                                                                    0.0,
                                                                    1.0
                                                                ]
                                                            ])
                                    self.need_update=True
                                    print("save")
                                    pass
                                    # self.trainer.save_checkpoint(full=True, best=False)
                                    # dpg.set_value("_log_ckpt", "saved " + os.path.basename(self.trainer.stats["checkpoints"][-1]))
                                    # self.trainer.epoch += 1 # use epoch to indicate different calls.
                                dpg.add_button(label="save", tag="_button_save", callback=callback_save)
                                dpg.bind_item_theme("_button_save", theme_button)

                                dpg.add_text("", tag="_log_ckpt")
        dpg.setup_dearpygui()
        dpg.show_viewport()
    def train_step(self):
        pass
    def test_step(self):
        
        if self.need_update:
            from app.nerf.test import test_render
            # self.framebuff=test_render(self.KEY, self.test_args, self.logger)
            
            self.framebuff,_=test_render(self.KEY, self.test_args, self.logger,camera_pose=self.camera_pose)
            dpg.set_value("_texture", self.framebuff)
            self.need_update=False
            pass
    def render(self):
        while dpg.is_dearpygui_running():
            ##resize
            # View_H=dpg.get_viewport_height()
            # View_W=dpg.get_viewport_width()
            # W=View_W
            # H=View_H
            # with Image.open(filename) as img:
            #     img = np.array(img.resize((W,H), PIL.Image.Resampling.BILINEAR),dtype=np.float)    
            #     img=img/256. 
            # dpg.set_item_width("_primary_window",W)
            # dpg.set_item_height("_primary_window",H)
            # dpg.set_value("_texture", img)
            #print("render")
            if self.train_thread!=None and not self.training:
                self.train_thread.stop()
            self.test_step()
            dpg.render_dearpygui_frame()

        #dpg.destroy_context()



def GuiWindow(KEY: jran.KeyArray, args: GuiWindowArgs, logger: logging.Logger):
    #todo:--args
    testArgs=NeRFTestingArgs(data_root=Path("/home/cad_83/E/chenyingxi/jaxngp/data/nerf_synthetic/lego"),
                             exp_dir=Path("/home/cad_83/E/chenyingxi/jaxngp/data/gui/test"),
                             test_ckpt=Path("/home/cad_83/E/chenyingxi/jaxngp/data/train1/checkpoint_3072"))
    trainArgs=NeRFTrainingArgs(data_root=Path("/home/cad_83/E/chenyingxi/jaxngp/data/nerf_synthetic/lego"),
                               exp_dir=Path("--exp-dir /home/cad_83/E/chenyingxi/jaxngp/data/gui/train1"))
    #(for debug)init shows
    filename="/home/cad_83/E/chenyingxi/jaxngp/data/test1/rgb.png"
    img = mpimg.imread(filename) 
    img=np.array(img)
    H,W=img.shape[:2]
    #H,W=800,000
    
    # train_thread=TrainThread(istraining=True,KEY=KEY,args=trainArgs,logger=logger)
    # train_thread.start()
    # train_thread.join()
    
    #create gui and run it
    nerfGui=NerfGUI(framebuff=img,H=H,W=W,KEY=KEY,train_args=trainArgs,test_args=testArgs,logger=logger)
    nerfGui.render()
    
    
    # from app.nerf.test import test_render
    # test_render(KEY, testArgs, logger)
    # from app.nerf.train import train
    # train(KEY, trainArgs, logger)