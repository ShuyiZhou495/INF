import os, importlib
from util import log, update_timer
import torch
import torch.utils.tensorboard
from easydict import EasyDict as edict
import time
import tqdm
from typing import Union, List, Tuple, Optional, Any
import util


class Model():
    """model is to control the whole training process
    """
    def __init__(self, opt: edict[str, Any]) -> None:
        super().__init__()
        # tensorboard
        self.tb = torch.utils.tensorboard.SummaryWriter(log_dir=opt.output_path,flush_secs=10)
        # variables to control training process
        self.ep, self.iter_start, self.status = 0, 0, "base"
        # variables to control whether to optimize density/color field, whether to use sheduler for the optimizers
        self.field_req, self.pose_req, self.sched_f, self.sched_p = True, True, False, False
        self.optim_lr = {}

    def build_network(self, opt: edict[str, Any]) -> None:
        """
        build renderer class in self.renderer and pose class in self.pose. 
        And restore checkpoints.

        Args:
            opt (edict[str, Any]): opt
        """
        log.info("building networks...")
        renderer = importlib.import_module(f"model.{opt.model}")
        self.renderer = renderer.Renderer(opt).to(opt.device)
        if opt.poses:
            pose = importlib.import_module(f"model.{opt.model}")
            self.pose = pose.Pose(opt).to(opt.device)
        self.restore_checkpoint(opt)

    def set_optimizer(self, opt: edict[str, Any]) -> None:
        """
        set optim_f for field optimizer and optim_p for pose optimizer.
        learning rate will be restore from checkpoints or reset for training lidar poses during training process 

        Args:
            opt (edict[str, Any]): opt
        """
        log.info("setting up optimizers...")
        # optimizer for color/density field
        self.field_lr = opt.lr.field
        self.optim_f = torch.optim.Adam(params=self.renderer.field.parameters(),lr=self.field_lr)
        # optimizer for lidar poses/extrinsic parameters
        if opt.poses:
            self.pose_lr = opt.lr.pose
            self.optim_p = torch.optim.Adam(params=self.pose.parameters(), lr=self.pose_lr)

    def create_dataset(self, opt: edict[str, Any]) -> None:
        """
        create dataset variables. Basically data are not loaded unless necessary.

        Args:
            opt (edict[str, Any]): opt
        """
        data = importlib.import_module(f"data.{opt.data.sensor}")
        log.info("creating dataset...")
        self.train_data = data.Dataset(opt, target="train")
        if opt.model=="color":
            self.vis_data = data.Dataset(opt,target="vis")

    def train(self, opt: edict[str, Any]) -> None:
        """
        the whole traning process

        Args:
            opt (edict[str, Any]): opt
        """
        log.title("TRAINING START")
        self.timer = edict(start=time.time(),it_mean=None)

        # detail training process is in child model

    def train_iteration(self, opt: edict[str, Any], var: edict, loader: tqdm.std.tqdm) -> None:
        """
        in every training iteration, we retrieve poses, trasform rays to world space and render with these rays
        and update the parameters

        Args:
            opt (edict[str, Any]): opt
            var (edict[str, torch.Tensor]): edict of used training data
            loader (tqdm.std.tqdm): tqdm progress bar
        """
        # --- before train iteration ---
        
        self.renderer.train()
        self.timer.it_start = time.time()
        self.field_req and self.optim_f.zero_grad()
        self.pose_req and self.optim_p.zero_grad()

        # --- train iteration ---

        # get rays, render and calculate loss
        dirs, origins, gt = self.get_rays(opt, var)
        res = self.renderer(opt, dirs, origins, mode="train") 
        self.loss = self.compute_loss(opt, gt, res)
        # optimizer
        self.loss.all.backward()
        self.field_req and self.optim_f.step()
        self.field_req and self.sched_f and self.sched_f.step()
        self.pose_req and self.optim_p.step()
        self.pose_req and self.sched_p and self.sched_p.step()

        # --- after training iteration ---
        
        if self.it==0 or (self.it + 1) % opt.freq.scalar == 0: 
            self.log_scalars(opt, var.idx)
        if self.it==0 or (self.it + 1) % opt.freq.val == 0: 
            self.validate(opt)
        if (self.it + 1) % opt.freq.ckpt == 0:
            self.save_checkpoint(opt, status=self.status)
        loader.set_postfix(it=self.it,loss=f"{self.loss.all:.3f}")
        self.timer.it_end = time.time()
        update_timer(opt,self.timer,self.ep,len(loader))

    def end_process(self, opt: edict[str, Any]) -> None:
        """
        close every opened files

        Args:
            opt (edict[str, Any]): opt
        """
        self.tb.flush()
        self.tb.close()
        log.title("PROCESS END")

    def save_checkpoint(self, opt: edict[str, Any], status: str="base") -> None:
        """
        save checkpoint, including 
        current epochs (only change when optimizing LiDAR poses), 
        current iteration, learning rates, status (only change when optimizing LiDAR poses)
        and the parameters in renderer and pose classes

        Args:
            opt (edict[str, Any]): opt
            status (str, optional): used when optimizing LiDAR poses. Defaults to "base".
        """
        checkpoint = dict(
            epoch=self.ep,
            iter=self.it,
            status=status,
            renderer=self.renderer.state_dict(),
        )
        opt.poses and checkpoint.update(pose=self.pose.state_dict())
        self.it>0 and self.field_req and self.sched_f and checkpoint.update(lr_f=self.optim_f.param_groups[0]["lr"])
        self.it>0 and self.pose_req and self.sched_p and checkpoint.update(lr_p=self.optim_p.param_groups[0]["lr"])

        torch.save(checkpoint,f"{opt.output_path}/model.ckpt")
        log.info(f"checkpoint saved: (epoch {self.ep} (iteration {self.it})") 

    def restore_checkpoint(self, opt: edict[str, Any]) -> None:
        """
        restore the information saved including
        current epochs (only change when optimizing LiDAR poses), 
        current iteration, learning rates, status (only change when optimizing LiDAR poses)
        and the parameters in renderer and pose classes

        Args:
            opt (edict[str, Any]): opt
        """
        log.info("resuming from previous checkpoint...")
        
        get_child_state_dict = lambda state_dict, key: { ".".join(k.split(".")[1:]): v for k,v in state_dict.items() if k.startswith(f"{key}.")}
        load_name = f"{opt.output_path}/model.ckpt"
        
        if not os.path.exists(load_name):
            return
        checkpoint = torch.load(load_name,map_location=opt.device)
        # load modules in renderer (mlp for density/color fields)
        child_state_dict = get_child_state_dict(checkpoint["renderer"],"field")
        if child_state_dict:
            print("restoring field...")
            self.renderer.field.load_state_dict(child_state_dict)
        # load pose parameters
        if opt.poses:
            self.pose.load_state_dict(checkpoint["pose"])
        # load train progress
        self.ep = checkpoint["epoch"]
        self.iter_start = checkpoint["iter"] 
        self.status = checkpoint["status"]
        print(f"resuming from epoch {self.ep} (iteration {self.iter_start})")
        # load learning rate
        for key in ["lr_f", "lr_p"] :
            if key in checkpoint:
                self.optim_lr[key] = checkpoint[key]

    def reset_field_lr(self, iter: int, sched: Union[bool, float] = False) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        reset or restore learning rate for field optimizer
        and return the correcsponding scheduler if necessary

        Args:
            iter (int): number of iterations left
            sched (Union[bool, float]): False if scheduler is not required. 
            Otherwise, it should be a float number indicating the decay rate.

        Returns:
            Optional[torch.optim.lr_scheduler._LRScheduler]: scheduler linked with self.optim_f if necessaray, otherwise return None
        """
        if "lr_f" in self.optim_lr:
            lr_f = self.optim_lr["lr_f"]
            self.optim_lr.pop("lr_f")
        else:
            lr_f = self.field_lr
        if iter < 1:
            return None
        for g in self.optim_f.param_groups:
            g['lr'] = lr_f
        if sched is not False: 
            return torch.optim.lr_scheduler.ExponentialLR(self.optim_f, sched**(1./iter))
        return None

    def reset_pose_lr(self, iter: int, sched: Union[bool, float] = False) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        reset or restore learning rate for pose optimizer
        and return the correcsponding scheduler if necessary

        Args:
            iter (int): number of iterations left
            sched (Union[bool, float]): False if scheduler is not required. 
            Otherwise, it should be a float number indicating the decay rate.

        Returns:
            Optional[torch.optim.lr_scheduler._LRScheduler]: scheduler linked with self.optim_p　if necessaray, otherwise return None
        """
        if "lr_p" in self.optim_lr:
            lr_p = self.optim_lr["lr_p"]
            self.optim_lr.pop("lr_p")
        else:
            lr_p = self.pose_lr
        if iter < 1:
            return None
        for g in self.optim_p.param_groups:
            g['lr'] = lr_p
        if sched is not False: 
            return torch.optim.lr_scheduler.ExponentialLR(self.optim_p, sched**(1./iter))
        return None

    @torch.no_grad()
    def render_by_slices(self, opt: edict[str, Any], pose: torch.Tensor) -> edict[str, torch.Tensor]:
        """
        render a whole frame with a pose

        Args:
            opt (edict[str, Any]):  opt
            pose (tensor): [3, 4]
            
        Returns:
            edict[str, torch.Tensor]: depth=tensor with shape [HW]; rgb=tensor with shape [HW, 3] if there is rgb
        """
        if not hasattr(self, "render_dirs"):
            camera = importlib.import_module(f"camera.{opt.render_camera}")
            self.render_dirs = util.rays_from_image(opt, opt.
            render_W, opt.render_H, camera, opt.render_intr)
        ret_all = edict(depth=[])
        if opt.model=="color":
            ret_all.update(rgb=[])
        # render the image by slices for memory considerations
        for c in range(0,opt.render_H * opt.render_W,opt.train.rand_rays):
            ray_idx = torch.arange(c,min(c+opt.train.rand_rays,opt.render_H * opt.render_W))
            dirs = self.render_dirs[ray_idx].to(opt.device) # [HW, 3]
            # transform to world spaces
            dirs = dirs @ pose[..., :3].transpose(0, 1) # [HW, 3]
            origins = pose[:, 3].repeat(ray_idx.shape[0], 1) # [HW, 3]
            # render the rays
            ret = self.renderer(opt, dirs[None], origins[None], mode="vis") # [1, HW, 3]
            for key in ret_all: ret_all[key].append(ret[key][0]) # [HW] or [HW, 3]
        # group all slices of images
        for k in ret_all: ret_all[k] = torch.cat(ret_all[k],dim=0)
        return ret_all  

    def get_rays(self, opt: edict[str, Any], var: edict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, edict[str, torch.Tensor]]:
        """
        select rays and transform to world space

        Args:
            opt (edict[str, Any]): opt
            var (edict[str, torch.Tensor]): edict including ray frame information

        Returns:
            Tuple[torch.Tensor, torch.Tensor, edict[str, torch.Tensor]]: 
            1. directions of rays (toch.Tensor): shape [B, HW, 3]
            2. origin points of rays (torch.Tensor): shape [B, HW, 3]
            3. ground truth information (edict): dictionary of gt values that will be used during calculating losses
        """
        raise NotImplementedError
    
    def log_scalars(self, opt: edict[str, Any], idx: List[int] = [0]) -> None:
        """
        log scalar information including loss, poses, learning rates to tensorboard

        Args:
            opt (edict[str, Any]): opt
            idx (List[int], optional): the indices of frames to load. Defaults to [0].
        """
        raise NotImplementedError

    def compute_loss(self, opt: edict[str, Any], gt: edict[str, torch.Tensor], res: edict[str, torch.Tensor]) -> edict[str, torch.Tensor]:
        """compute loss and return the dictionary including all the losses 
        and the summation of all the losses

        Args:
            opt (edict[str, Any]): opt
            gt (edict[str, torch.Tensor]): easy dict object including ground truth information
            res (edict[str, torch.Tensor]): easy dict object including rendered results

        Returns:
            edict[str, torch.Tensor]: easy dict object including all kinds of losses. 
            The summation of them is saved in the key "all"
        """
        raise NotImplementedError

    @torch.no_grad()
    def validate(self, opt: edict[str, Any]) -> None:
        """
        render the density field/color field of current frame to tensorboard

        Args:
            opt (edict[str, Any]): opt

        """
        raise NotImplementedError

class Renderer(torch.nn.Module):
    """
    renderer class is used to perform volume rendering for given rays.
    """

    def sample_points(self, opt: edict[str, Any], dirs: torch.Tensor, origins: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        sample points along the ray
        
        Args:
            dirs: shape of (..., 3)
            origins: shape of (..., 3)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
            1. point coordinates: shape of (..., #points every ray, 3)
            2. depths(distance) of the points on the ray: shape of (..., #points every ray)
        """
        depth_min,depth_max = opt.data.near_far
        si = dirs.shape[:-1]
        rand_samples = torch.rand(*si,opt.train.sample_intvs,device=opt.device) 
        add_ = torch.arange(opt.train.sample_intvs,device=opt.device).reshape(*[1 for _ in range(len(si))], opt.train.sample_intvs)
        rand_samples += add_.float()
        depth_samples = rand_samples/opt.train.sample_intvs*(depth_max-depth_min)+depth_min # (-1, N)
        points = depth_samples.unsqueeze(-1) * dirs.unsqueeze(-2) + origins.unsqueeze(-2)
        return points, depth_samples
