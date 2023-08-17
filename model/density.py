import util
from . import base
import torch
import torch.nn.functional as torch_F
import transforms
from easydict import EasyDict as edict
import tqdm
import os
import numpy as np
import util_vis
from typing import List, Union, Any, Tuple, Optional


class Model(base.Model):   

    def train(self, opt: edict[str, Any]) -> None:
        """
        Optimize neural radiance field. Estimate LiDAR poses if necessary.

        Args:
            opt (edict[str, Any]): opt
        """
        super().train(opt)
        if not opt.train.poses:
            self.train_data.prefetch_all_data()
        else:
            while self.ep < len(self.train_data):
                if self.status == "base":
                    self.train_base(opt)   
                elif self.status == "pose":
                    self.train_pose(opt)
                elif self.status=="together":
                    self.train_together(opt)
            if self.train_data.poses is not None:
                transforms.evaluate_poses(opt, self.pose.get_all(), self.train_data.poses[:, :3])
        self.save_poses(opt)
        self.status="all"
        self.train_all(opt)

    def train_process(self, 
                      opt: edict[str, Any], 
                      title: str, 
                      idx: List[int], 
                      iter: int, 
                      sched_f: Union[bool, float] = False, 
                      sched_p: Union[bool, float] = False, 
                      ) -> None:
        """
        train process including loading frames and optimize nerual density field/poses 
        with respect to the loaed frames

        Args:
            opt (edict[str, Any]): opt
            title (str): title for tqdm progress bar
            idx (List[int]): list of id of frames
            iter (int): number of total iteration numbers
            sched_f (Union[bool, float], optional): whether use scheduler to optimizer for neural field. False if scheduler is not required. Otherwise, it should be a float number indicating the decay rate. Defaults to False.
            sched_p (Union[bool, float], optional): whether use scheduler to optimizer for pose. False if scheduler is not required. Otherwise, it should be a float number indicating the decay rate. Defaults to False.
        """
        loader = tqdm.trange(self.iter_start, iter,desc=title,leave=True)
        # set the learning rates
        if self.field_req:
            self.sched_f = self.reset_field_lr(iter - self.iter_start, sched_f)
        if self.pose_req:
            self.sched_p = self.reset_pose_lr(iter - self.iter_start, sched_p)
        
        # collect the frames
        vars = [self.train_data[id] for id in idx]
        var = edict(idx=idx)
        self.train_data.collect(vars, var)
        
        # train for iterations
        for self.it in loader:
            self.train_iteration(opt,var,loader)
        
        # prepare for the next process
        self.iter_start = 0

    def train_base(self, opt: edict[str, Any]) -> None:
        """
        training the density field with key frame

        Args:
            opt (edict[str, Any]): opt
        """
        if self.iter_start==0:
            # save current frame as keyframe
            self.pose.save_keyframe(self.ep)

        if self.ep == len(self.train_data) - 1:
            # if it is the last frame, this is no longer necessary
            self.ep += 1
            return
        
        # set which frames to use, what to optimize
        idx = [self.ep]
        self.pose_req, self.field_req = False, True   
        
        # train process
        self.train_process(opt, f"base {self.ep}", idx, opt.train.iteration.base, sched_f=0.1)
        
        # to next phase
        self.ep += 1
        self.status = "pose"

    def train_pose(self, opt: edict[str, Any]) -> None:
        """
        estimate pose of one frame with density field. 
        if the current frame pose is far from keyframe pose, 
        last frame will become keyframe

        Args:
            opt (edict[str, Any]): opt
        """
        
        # set the keyframe for current frame 
        base = self.pose.curr_keyframe(self.ep)
        
        # initialize current frame pose as the last frame pose
        if self.iter_start==0: 
            self.pose.assign(self.ep)
        
        # set which frames to use, what to optimize
        idx = [self.ep]
        self.field_req, self.pose_req= False, True

        # train process
        self.train_process(opt, f"pose {self.ep}", idx, opt.train.iteration.pose, sched_p=0.2)
        
        # plot
        self.plot_res_poses(opt)
        
        # decide whether this frame is keyframe
        distance = self.pose.get_distance(self.ep)
        if distance > opt.train.distance:
            self.status = "together"
            self.ep = self.ep - 1 if (self.ep - base > 1) else self.ep  # if it is just the next frame of keyframe, then do not go back to last frame
        elif self.ep == len(self.train_data) - 1 :
            # last frame
            self.status = "together"
        else:
            # next frame
            self.ep += 1

    def train_together(self, opt: edict[str, Any]) -> None:
        """
        optimize density field and local poses together

        Args:
            opt (edict[str, Any]): opt
        """
        
        # set which frames to use, what to optimize
        base = self.pose.keyframe_map[self.ep].tolist()
        idx = list(range(base, self.ep+1))
        self.field_req, self.pose_req = True, True
        
        # train process
        self.train_process(opt, f"toge {base}-{self.ep}", idx, opt.train.iteration.together)
        
        # plot
        self.plot_res_poses(opt)
        
        # next local map
        self.status = "base"

    def train_all(self, opt: edict[str, Any]) -> None:
        """
        use the given lidar poses to train density field.
        this function is used when lidar poses are known or already trained.

        Args:
            opt (edict[str, Any]): opt
        """
        # set which frames to use, what to optimize
        idx = list(range(len(self.train_data)))
        self.field_req, self.pose_req = True, False
        
        # train process
        self.train_process(opt, f"all", idx, opt.train.iteration.all, sched_f=0.1)

    def get_rays(self, opt: edict[str, Any], var: edict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, edict[str, torch.Tensor]]:
        
        # get poses
        if opt.train.poses:
            if self.status=="base" or self.status=="all":
                # not to optimize pose
                pose = self.pose.get(var.idx).detach() 
            elif self.status=="pose":
                # optimize pose
                pose = self.pose.get(var.idx) 
            elif self.status=="together":
                # fix the first frame pose (keyframe)
                pose = self.pose.get(var.idx)
                pose = torch.row_stack((pose[:1].detach(), pose[1:])) 
        else:
            pose = self.train_data.poses[var.idx]

        # get lidar space rays
        indices = self.random_select(opt, var.size) # [B, HW]
        dirs = var.dir[indices].to(opt.device) # [B, HW, 3]
        origins = var.origin[indices].to(opt.device) # [B, HW, 3]
        gt = edict({
            "depth": var.z[indices].to(opt.device), # [B, HW]
            "dweight": var.weight[indices].to(opt.device) # [B, HW] this weight emphasizes edge points
        })
        # transform to world spaces
        dirs = dirs @ pose[..., :3].transpose(1, 2)
        origins = origins @ pose[..., :3].transpose(1, 2) + pose[..., 3].unsqueeze(1)

        return dirs, origins, gt

    def random_select(self, opt: edict[str, Any], sizes: torch.Tensor) -> torch.Tensor:
        """
        randomly select rays from LiDAR frames and return the indices of these rays
        Args:
            opt (edict[str, Any]): opt
            sizes (torch.Tensor): number of total rays each frame, tensor with shape [B]

        Returns:
            torch.Tensor: selected indices, shape [B, HW]
        """
        batch_size = sizes.shape[0]
        per_batch = opt.train.rand_rays // batch_size
        random = torch.rand(batch_size, per_batch) * sizes[..., None]
        cumsum = torch.zeros_like(sizes)
        cumsum[1:] = sizes.cumsum(dim=0)[:-1]
        return cumsum[..., None] + random.to(int) 

    def compute_loss(self, opt: edict[str, Any], gt: edict[str, torch.Tensor], res: edict[str, torch.Tensor]) -> edict[str, torch.Tensor]:
        
        loss = edict({})
        
        # --- depth loss ---
        f = gt.depth > 0 # in ouster, the range of unreached points are all 0
        rendered_depth = (res.weights * res.depth_samples).sum(dim=-1) # [B, HW]
        depth_loss = torch.sum(torch.abs(rendered_depth[f] - gt.depth[f]) * gt.dweight[f]) / torch.sum(gt.dweight[f])
        loss.update(depth=depth_loss) # l1 loss

        # --- opacity loss ---
        rendered_opacity = res.weights.sum(dim=-1) # [B, HW]
        gt_opacity = (gt.depth != 0).to(torch.float)
        opacity_loss = torch.mean(-gt_opacity * torch.log(rendered_opacity + 1e-6) - (1 - gt_opacity) * torch.log(1 - rendered_opacity +1e-6)) # BCE loss
        loss.update(opacity=opacity_loss)

        # --- empty loss ---
        epsilon = opt.data.near_far[1]/opt.train.sample_intvs
        z_diff = res.depth_samples-gt.depth[..., None] #[B, HW, N]/[B, HW]
        empty_f = z_diff < -epsilon
        empty_loss = torch.sum(res.weights[empty_f]**2) / torch.sum(f)
        loss.update(empty=empty_loss)

        # --- sum them all ---
        loss_all = 0
        for key in loss:
            loss_all += opt.train.loss_weight[key] * loss[key]
        loss.update(all=loss_all)
        return loss

    def plot_res_poses(self, opt: edict[str, Any]) -> None:
        """
        plot the poses up till now

        Args:
            opt (edict[str, Any]): opt
        """
        info = {}
        if self.train_data.poses is not None:
            info["gt"] = self.train_data.poses.detach().cpu()
        poses = self.pose.get_all().detach().cpu()
        info["res"] = poses[:self.ep+1]
        info["key"] = poses[self.pose.keyframes()]
        util.plot_paths(info, os.path.join(opt.output_path, f"result"))

    def log_scalars(self, opt: edict[str, Any], idx: List[int] = [0]) -> None:
        split = f"{self.status}-{self.ep}" if opt.poses else "train"
        # log loss:
        for key, value in self.loss.items():
            self.tb.add_scalar(f"{split}/{key}_loss", value, self.it+1)
            
        # log learning rate:
        self.field_req and self.tb.add_scalar(f"{split}/lr_field", self.optim_f.param_groups[0]["lr"], self.it+1)
        self.pose_req and self.tb.add_scalar(f"{split}/lr_pose", self.optim_p.param_groups[0]["lr"], self.it+1)

        # log pose if there is no ground truth pose, otherwise log pose errors
        if self.pose_req:
            # since the first frame is keyframe, the pose is fixed
            idu = idx[1:] if self.status=="together" else idx
            res = self.pose.get(idu).detach()
            if self.train_data.poses is not None:
                res = transforms.pose.relative(self.train_data.poses[idu], res)
            euler, trans = transforms.get_ang_tra(res)
            for id, e, t in zip(idu, euler, trans):
                util_vis.tb_log_ang_tra(self.tb, f"pose-{id}", self.status, e, t, self.it+1)

    def save_poses(self, opt: edict[str, Any]) -> None:
        """
        save the poses to be used later when optimizing neural color field.

        Args:
            opt (edict[str, Any]): opt
        """
        if opt.poses:
            poses = self.pose.get_all()
        else:
            poses = self.train_data.poses
            
        out_pose = poses.detach().cpu().numpy()
        np.save(f"{opt.output_path}/poses.npy", out_pose)

    @torch.no_grad()
    def validate(self, opt: edict[str, Any]) -> None:
        self.renderer.eval()
        
        # if optimizing with all frames, we use the middle frame pose to render
        id = self.ep if self.status != "all" else len(self.train_data)//2
        
        # get pose
        if opt.poses:
            pose = self.pose.get(id)
        else:
            pose = self.train_data.poses[id]
        
        # render
        var = self.render_by_slices(opt,pose)
        
        # colorize depth and post on tensorboard
        depth_c = util_vis.to_color_img(opt, var.depth)
        util_vis.tb_image(opt, self.tb, self.it + 1, self.status, f"{self.ep}", depth_c)

class Renderer(base.Renderer):
    def __init__(self, opt: edict[str, Any]) -> None:
        super().__init__()
        self.field = Density(opt)
        
    def forward(self, opt: edict[str, Any], dirs: torch.Tensor, origins: torch.Tensor, mode: str="train") -> edict[str, torch.Tensor]:
        """
        given rays and return the rendered depths, opacity, weights

        Args:
            opt (edict[str, Any]): opt
            dirs (torch.Tensor): tensor with shape [B, HW, 3]
            origins (torch.Tensor): tensor with shape [B, HW, 3]
            mode (str, optional): "vis" or "train". Defaults to "train".

        Returns:
            edict[str, torch.Tensor]: 
            if mode is "train": key "weights" is tensor with shape [B, HW, N]. key "depth_samples" is tensor with shape [B, HW, N]. if mode is "vis": key "depth" is tensor with shape [B, HW]
        """

        point_samples, depth_samples = self.sample_points(opt, dirs, origins) # [F, N, 3], [F, N]
        density_samples = self.field(opt, point_samples)
        weights = self.field.composite(opt,density_samples,depth_samples) # [F, N]
        if mode=="vis":
            # when validating, only depth values are necessary
            return edict(depth=(weights * depth_samples).sum(dim=-1)) #[F]
        else:
            return edict(weights=weights, depth_samples=depth_samples)

class Density(torch.nn.Module):
    def __init__(self, opt: edict[str, Any]) -> None:
        super().__init__()
        self.define_network(opt)

    def define_network(self, opt: edict[str, Any]) -> None:
        """
        define the linear layers, follow BARF implementation

        Args:
            opt (edict[str, Any]): opt
        """
        get_layer_dims = lambda layers: list(zip(layers[:-1],layers[1:]))
        input_3D_dim = 3+6*opt.arch.posenc.L_3D
        self.mlp_feat = torch.nn.ModuleList()
        L = get_layer_dims(opt.arch.layers_feat)
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = input_3D_dim
            if li in opt.arch.skip: k_in += input_3D_dim
            if li==len(L)-1: k_out = 1
            linear = torch.nn.Linear(k_in,k_out)
            self.tensorflow_init_weights(opt,linear,out="first" if li==len(L)-1 else None)
            self.mlp_feat.append(linear)

    def tensorflow_init_weights(self, opt: edict[str, Any], linear: torch.nn.Linear, out:Optional[str] = None) -> None:
        """
        use Xavier init instead of Kaiming init, follow BARF implementation

        Args:
            opt (edict[str, Any]): opt
            linear (torch.nn.Linear): linear layer to initialzied
            out (Optional[str], optional): whether it is the last layer. Defaults to None.
        """
        # 
        relu_gain = torch.nn.init.calculate_gain("relu")
        if out=="first":
            torch.nn.init.xavier_uniform_(linear.weight[:1])
            torch.nn.init.xavier_uniform_(linear.weight[1:],gain=relu_gain)
        else:
            torch.nn.init.xavier_uniform_(linear.weight,gain=relu_gain)
        torch.nn.init.zeros_(linear.bias)

    def forward(self, opt: edict[str, Any], points_3D: torch.Tensor) -> torch.Tensor: 
        """
        give density values for 3d points, follow BARF implementation

        Args:
            opt (edict[str, Any]): opt
            points_3D: tensor with shape [B, HW, N, 3]

        Returns:
            torch.Tensor: tensor with shape [B, HW, N, 3]
        """
        points_enc = self.positional_encoding(opt,points_3D,L=opt.arch.posenc.L_3D)
        points_enc = torch.cat([points_3D,points_enc],dim=-1) # [B, HW, N, 6L+3]
        feat = points_enc
        for li,layer in enumerate(self.mlp_feat):
            if li in opt.arch.skip: feat = torch.cat([feat,points_enc],dim=-1)
            feat = layer(feat)
            if li==len(self.mlp_feat)-1:
                density = feat[...,0]
                density = torch_F.relu(density)
            feat = torch_F.relu(feat)
        return density

    def positional_encoding(self, opt: edict[str, Any], input: torch.Tensor, L: int) -> torch.Tensor:
        """
        positional encoding, following BARF implementation

        Args:
            opt (edict[str, Any]): opt
            input (torch.Tensor): tensor with shape [B, HW, N, 3]
            L (int): maximum frequencies

        Returns:
            torch.Tensor: tensor with shape [B, HW, N, 6L]
        """
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float,device=opt.device) * torch.pi / opt.train.range # (L)
        spectrum = input[...,None] *freq # [B, HW, N, 3, L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B, HW, N, 3, L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B, HW, N, 3, 2, L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B, HW, N, 6L]
        return input_enc

    def composite(self, opt: edict[str, Any], density_samples: torch.Tensor, depth_samples: torch.Tensor):
        """
        given density samples and depth samples, calculate weights.
        the direction of the ray is normalized vector, so depth sample intervals are distance intervals

        Args:
            opt (edict[str, Any]): opt
            density_samples (torch.Tensor): tensor with shape [B, HW, N]
            depth_samples (torch.Tensor): depth samples are actually distance samples. tensor with shape [B, HW, N]

        Returns:
            weight: [B, HW, N]
        """
        # volume rendering: compute probability (using quadrature)
        _, depth_max = opt.data.near_far
        depth_intv_samples = depth_samples[...,1:]-depth_samples[...,:-1] # [B, HW, N-1]
        dist_samples = torch.cat([depth_intv_samples,depth_max - depth_samples[..., -1:]], dim=-1) # [B, HW, N]
        sigma_delta = density_samples*dist_samples # [B, HW, N]
        alpha = 1-(-sigma_delta).exp_() # [B, HW, N]
        T = (-torch.cat([torch.zeros_like(sigma_delta[...,:1]),sigma_delta[...,:-1]],dim=-1).cumsum(dim=-1)).exp_() # [B, HW, N]
        weight = (T*alpha) # [B, HW, N]
        return weight

class Pose(torch.nn.Module):
    """
    LiDAR poses, only global poses will be output
    """
    def __init__(self, opt: edict[str, Any]) -> None:
        super().__init__()
        # __local_poses are the LiDAR poses relative to the most recent keyframes
        self.__local_poses = torch.nn.Embedding(opt.data.length,6).to(opt.device)
        torch.nn.init.zeros_(self.__local_poses.weight)
        # register the parameters so that we can save the keyframe poses in checkpoint
        # keyframe_map[i] gives the keyframe that i frame is based on
        self.register_parameter(
            name="keyframe_map",
            param=torch.nn.Parameter(torch.zeros(opt.data.length, dtype=torch.int), requires_grad=False),
        )
        # keyframe poses
        self.register_parameter(
            name="keyframe_poses",
            param=torch.nn.Parameter(torch.eye(3,4).repeat(opt.data.length,1,1), requires_grad=False)
        )
    
    def save_keyframe(self, i: int) -> None:
        """
        save current frame pose to keyframe pose variable, 
        set current frame local pose to 0,
        save the correspoinding keyframe index of current frame

        Args:
            i (int): index of frame
        """
        p = self.get(i).detach()
        self.keyframe_poses[i].data.copy_(p) # save current frame pose to keyframe pose variable
        self.__local_poses.weight[i].data.copy_(torch.zeros(6)) # set current frame local pose to 0
        self.keyframe_map[i] = i # the corresponding keyframe of current frame is itself

    def keyframes(self) -> List[int]:
        """
        get indices of keyframes:
        
        Returns: 
            List(int): list of key frames indices
        """
        return sorted(list(set(self.keyframe_map.tolist())))

    def curr_keyframe(self, i: int) -> int:
        """
        keyframe of current frame is the same as that of the last frame
        this function will called before single pose training
        
        Args:
            i (int): index of queried frame
        
        Returns:
            int: the keyframe index of the current frame
        """
        self.keyframe_map[i] = self.keyframe_map[i-1]
        return self.keyframe_map[i]

    def assign(self, i: int) -> None:
        """
        assign last frame local pose to this frame
        
        Args:
            i (int): index of the frame
        """
        self.__local_poses.weight[i].data.copy_(self.__local_poses.weight[i-1].detach())

    @torch.no_grad()
    def get_distance(self, i: int) -> float:
        """
        get distance from the closest keyframe
        
        Args:
            i (int): index of queried frame
            
        Returns:
            float: distance from the closest keyframe
        """
        local_pose = transforms.lie.se3_to_SE3(self.__local_poses.weight[i])
        distance = torch.linalg.norm(local_pose[:3, 3])
        return distance

    def get(self, idx: Union[int, List[int]]) -> torch.Tensor:
        """get the global poses of queried frames

        Args:
            idx (Union[int, List[int]]): indices of the queried frames

        Returns:
            torch.Tensor: pose in SE3 with shape [3, 4] or [B, 3, 4]
        """
        local_pose = transforms.lie.se3_to_SE3(self.__local_poses.weight[idx])
        keyframe_pose = self.keyframe_poses[self.keyframe_map[idx].tolist()]
        global_pose = transforms.pose.compose_pair(local_pose,keyframe_pose)
        return global_pose

    def get_all(self) -> torch.Tensor:
        """
        get all poses
        
        Returns: 
            torch.Tensor: all poses in SE3 with shape [B, 3, 4]

        """
        indices = list(range(len(self.keyframe_map)))
        return self.get(indices)

    def load_state_dict(self, state_dict, strict=True):
        # overwirte because we maybe trained with different frame numbers in different trials
        # and the length of poses may vary
        print(state_dict.keys())
        len_state = len(state_dict["_Pose__local_poses.weight"])
        if len(self.__local_poses.weight) > len_state:
            self.__local_poses.weight[:len_state].data.copy_(state_dict["_Pose__local_poses.weight"])
            self.keyframe_poses[:len_state].data.copy_(state_dict["keyframe_poses"])
            self.keyframe_map[:len_state].data.copy_(state_dict["keyframe_map"])
        else:
            self.__local_poses.weight.data.copy_(state_dict["_Pose__local_poses.weight"][:len(self.__local_poses.weight)])
            self.keyframe_poses.data.copy_(state_dict["keyframe_poses"][:len(self.__local_poses.weight)])
            self.keyframe_map.data.copy_(state_dict["keyframe_map"][:len(self.__local_poses.weight)])
