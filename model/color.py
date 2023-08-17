from . import base, density
from easydict import EasyDict as edict
from util import log
import torch
import tqdm
import util_vis
import os
import torch.nn.functional as torch_F
import transforms
import importlib
import imageio
import json
import util
from typing import Any, Tuple, List
import numpy as np

class Model(base.Model):

    def train(self, opt: edict[str, Any]) -> None:
        super().train(opt)
        
        iter = opt.train.iteration
        loader = tqdm.trange(self.iter_start, iter, desc="calib", leave=True)
        self.sched_f, self.sched_p = False, False
        
        # get all data
        self.train_data.prefetch_all_data()
        var = self.train_data.all

        # train iteration
        for self.it in loader:
            self.train_iteration(opt, var, loader)

    def get_rays(self, opt: edict[str, Any], var: edict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, edict[str, torch.Tensor]]:
        
        # --- pre-calculate directions for a image frame ---
        if not hasattr(self, "train_dirs"):
            self.camera = importlib.import_module(f"camera.{opt.camera}")
            self.train_dirs = util.rays_from_image(opt, opt.W, opt.H, self.camera, self.train_data.intr)
        
        # --- scale of image ---
        scale_mask = torch.zeros((opt.H, opt.W), device=opt.device, dtype=torch.bool)
        steps = [m for m, _ in opt.train.multi_scale] # opt.train.multi_scale: [[iter, scale], [iter, scale], ...] -> step:[iter, iter, ...]
        for n, s in enumerate(zip(steps, steps[1:] + [float("inf")])): 
            # [[iter1, iter2], [iter2, iter3], ..., [iter_last, inf]]
            if self.it >= s[0] and self.it < s[1]:
                scale = opt.train.multi_scale[n][1]
                break
        scale_mask[::scale, ::scale] = True 
        scale_mask = scale_mask.reshape(-1)
        # --- masked points ---
        if var.mask is not None:
            scale_mask = scale_mask & var.mask
        # --- samples indices ---
        all_indices = torch.arange(opt.H*opt.W, device=opt.device)[scale_mask]
        selected = (torch.rand(opt.train.rand_rays // len(var.idx), device=opt.device) * len(all_indices)).to(int) # shape of ray per frames
        indices = all_indices[selected]
        # --- project the indices to a cosine distribution (for panorama camera)
        if opt.camera == "panorama":
            row = torch.div(indices, opt.W, rounding_mode="floor")
            column = indices % opt.W
            # inverse transform sampling
            row_c = ((torch.arcsin((row/opt.H - 0.5) * 2) / torch.pi + 0.5)* opt.H).to(int)
            new_indices = row_c * opt.W + column
        # --- remove the indices that fall in to masked area after inverse transform sampling ---
            new_indices = new_indices[var.mask[new_indices]] # [HW]
            new_indices = new_indices.cpu()
        else:
            new_indices = indices.cpu()
        # --- transform camera rays to world space
        cdirs = self.train_dirs[new_indices].to(opt.device).repeat(len(var.idx), 1, 1) # [B, HW, 3]
        c2w = self.pose(var.pose) # [B, 3, 4]
        dirs = cdirs @ c2w[..., :3].transpose(1, 2) # [B, HW, 3]
        origins = c2w[..., 3][:, None, :].expand_as(dirs) # [B, HW, 3]

        # --- ground truth rgb ---
        gt_rgb = var.image[:, new_indices].to(opt.device) # [B, HW, 3]
        return dirs, origins, edict(rgb=gt_rgb, dirs=dirs, origins=origins, cposes=c2w, lposes=var.pose)

    def compute_loss(self, opt: edict[str, Any], gt: edict[str, torch.Tensor], res: edict[str, torch.Tensor]) -> edict[str, torch.Tensor]:
        rgb_loss = ((gt.rgb - res.rgb) ** 2).mean()
        return edict(all=rgb_loss)

    @torch.no_grad()
    def validate(self, opt: edict[str, Any]) -> None:
        log.info("validating")
        self.renderer.eval()

        os.makedirs(os.path.join(opt.output_path, "figures"), exist_ok=True)
        for var_original in self.vis_data:
            var_original = edict(var_original)
            # save the rendered images
            os.makedirs(os.path.join(opt.output_path, "figures", f"{var_original.idx}_depth"), exist_ok=True)
            os.makedirs(os.path.join(opt.output_path, "figures", f"{var_original.idx}_rgb"), exist_ok=True)
            
            # render
            var_original = edict(var_original)
            pose = self.pose(var_original.pose.to(opt.device))
            var = self.render_by_slices(opt, pose)

            # depth result
            depth_c = util_vis.to_color_img(opt, var.depth)
            util_vis.tb_image(opt, self.tb, self.it + 1, "vis", f"{var_original.idx}_depth", depth_c)
            imageio.imwrite(os.path.join(opt.output_path, "figures", f"{var_original.idx}_depth", f"{self.it}.png"),
                            util_vis.to_np_img(depth_c))
            # rgb image
            rgb_map = var.rgb.view(-1,opt.render_H,opt.render_W,3).permute(0,3,1,2) # [B,3,H,W]
            util_vis.tb_image(opt,self.tb,self.it+1,"vis", f"{var_original.idx}_rgb",rgb_map)
            imageio.imwrite(os.path.join(opt.output_path, "figures", f"{var_original.idx}_rgb", f"{self.it}.png"), 
                            util_vis.to_np_img(rgb_map))
            # original image
            origin_image = var_original.image.reshape(1, opt.render_H, opt.render_W, 3).permute(0, 3, 1, 2)
            util_vis.tb_image(opt,self.tb,self.it+1,"vis",f"{var_original.idx}_origin",origin_image)

    @torch.no_grad()
    def log_scalars(self, opt: edict[str, Any], idx: List[int] = [0]) -> None:
        # log loss:
        for key, value in self.loss.items():
            self.tb.add_scalar(f"train/{key}_loss", value, self.it + 1)

        # log learning rate:
        self.field_req and self.tb.add_scalar(f"train/lr_field", self.optim_f.param_groups[0]["lr"], self.it + 1)
        self.pose_req and self.tb.add_scalar(f"train/lr_pose", self.optim_p.param_groups[0]["lr"], self.it + 1)

        # log pose
        pose = self.pose.SE3()
        euler, trans = transforms.get_ang_tra(pose)
        util_vis.tb_log_ang_tra(self.tb, "ext", None, euler, trans, self.it + 1)
        res = {"rotation": euler, "translation": trans}
        
        # log pose error if there is ground truth pose
        if self.pose.ref_ext is not None:
            pose_e = transforms.pose.relative(pose, self.pose.ref_ext)
            euler_e, trans_e = transforms.get_ang_tra(pose_e)
            util_vis.tb_log_ang_tra(self.tb, "ext_error", None, euler_e, trans_e, self.it + 1)
            res.update(
                rotation_error=euler_e, 
                rotation_norm_error=np.linalg.norm(euler_e),
                translation_error=trans_e,
                translation_norm_error = np.linalg.norm(trans_e))
            
        # save in a json file
        with open(os.path.join(opt.output_path,"res.json"), "w") as f:
            json.dump(res, f)

class Renderer(base.Renderer):
    def __init__(self,opt):
        super().__init__()
        self.density_field = Density(opt.density_opt)
        self.field = Color(opt)
        
    def forward(self, opt: edict[str, Any], dirs: torch.Tensor, origins: torch.Tensor, mode: str="train") -> edict[str, torch.Tensor]:
        """
        given rays and return the rendered depths, colors

        Args:
            opt (edict[str, Any]): opt
            dirs (torch.Tensor): tensor with shape [B, HW, 3]
            origins (torch.Tensor): tensor with shape [B, HW, 3]
            mode (str, optional): not used here. Defaults to "train".

        Returns:
            edict[str, torch.Tensor]: 1. rgb: tensor with shape [B, HW, 3]; 2. depth: tensor with shape [B, HW]
        """
        point_samples, depth_samples = self.sample_points(opt, dirs, origins) # [B, HW, N, 3], [B, HW, N]
        density_samples = self.density_field(opt.density_opt, point_samples)
        weights = self.density_field.composite(opt.density_opt,density_samples,depth_samples) # [B, HW, N]
        depth = (weights * depth_samples).sum(dim=-1)
        # give the farthest point color, 
        # otherwise the unreached rays will be all black
        weights[..., -1] = 1 - weights[..., :-1].sum(dim=-1)
        colors = self.field(opt, point_samples)
        rgb = (weights[..., None] * colors).sum(dim=-2)
        return edict(rgb=rgb,depth=depth) #[B, HW]

class Color(density.Density):

    def define_network(self, opt: edict[str, Any]) -> None:
        get_layer_dims = lambda layers: list(zip(layers[:-1],layers[1:]))
        input_3D_dim = 3+6*opt.arch.posenc.L_3D
        self.mlp_feat = torch.nn.ModuleList()
        L = get_layer_dims(opt.arch.layers_feat)
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = input_3D_dim
            if li in opt.arch.skip: k_in += input_3D_dim
            linear = torch.nn.Linear(k_in,k_out)
            self.tensorflow_init_weights(opt,linear,out="first" if li==len(L)-1 else None)
            self.mlp_feat.append(linear)

    def forward(self, opt: edict[str, Any], points_3D: torch.Tensor) -> torch.Tensor: 
        """
        give colors of the sampled potins

        Args:
            opt (edict[str, Any]): opt
            points_3D (torch.Tensor): with shape [B, HW, N, 3]

        Returns:
            torch.Tensor: with shape [B, HW, N, 3]
        """
        points_enc = self.positional_encoding(opt,points_3D,L=opt.arch.posenc.L_3D)
        points_enc = torch.cat([points_3D,points_enc],dim=-1) # [B, HW, N,6L+3]
        feat = points_enc
        # extract coordinate-based features
        for li, layer in enumerate(self.mlp_feat):
            if li in opt.arch.skip: feat = torch.cat([feat, points_enc], dim=-1)
            feat = layer(feat)
            if li!=len(self.mlp_feat)-1:
                feat = torch_F.relu(feat)
        rgb = feat.sigmoid_() # [B,HW, N,3]
        return rgb

class Density(density.Density):
    def __init__(self, opt: edict[str, Any]) -> None:
        super().__init__(opt)
        # load the trained density field
        self.restore_param(opt)
        for p in self.parameters():
            p.requires_grad_(False)

    def restore_param(self, opt: edict[str, Any]) -> None:
        """
        restore the trained density field parameters
        
        Args:
            opt (edict[str, Any]): opt
        """
        load_name = f"{opt.output_path}/model.ckpt"
        assert os.path.exists(load_name), "density field not found"
        checkpoint = torch.load(load_name,map_location=opt.device)
        get_child_state_dict = lambda state_dict, key: { ".".join(k.split(".")[1:]): v for k,v in state_dict.items() if k.startswith(f"{key}.")}
        child_state_dict = get_child_state_dict(checkpoint["renderer"], "field")
        self.load_state_dict(child_state_dict)

class Pose(torch.nn.Module):
    """
    extrinsic parameters
    """
    def __init__(self, opt):
        super().__init__()
        # --- extrinsic parameters requiring gradient---     
        self.ext = torch.nn.Embedding(1, 6).to(opt.device)
        torch.nn.init.zeros_(self.ext.weight)

        # --- initial value ---
        # init euler angles are in degrees, xyz arrangement
        rot = opt.extrinsic[:3] if "extrinsic" in opt else [0, 0, 0] 
        trans = opt.extrinsic[-3:] if "extrinsic" in opt else [0, 0, 0]
        pose = transforms.pose.invert(transforms.ang_tra_to_SE3(opt, rot, trans))
        self.init = transforms.lie.SE3_to_se3(pose) #(6)
        
        # --- load the reference extrinsic parameter ----
        ref_path = os.path.join("data", opt.data.scene, "ref_ext.json")
        if os.path.exists(ref_path):
            with open(ref_path, "r") as file:
                ref = json.load(file)
            # angles are euler angles in xyz order and in degrees
            self.ref_ext = transforms.ang_tra_to_SE3(opt, ref["rotation"], ref["translation"])
            # In our reference extrinsic parameters, we use lidar-to-camera transformation. While in this work, we first project camera poses to LiDAR spaces, causing the result extrinsic parameters to be camera-to-lidar transformation. 
            # Thus, we need to inverse it here.
            self.ref_ext = transforms.pose.invert(self.ref_ext)
        else:
            self.ref_ext = None

    def SE3(self) -> torch.Tensor:
        """
        return SE3 matrix of ext

        Returns:
            torch.Tensor: tensor with shape [3, 4]
        """
        return transforms.lie.se3_to_SE3(self.ext.weight[0] + self.init)

    def forward(self, l2w) -> torch.Tensor:
        """
        given lidar to world poses, return camera to world poses

        Args:
            l2w (tensor): with shape [..., 3, 4]

        Returns:
            torch.Tensor: with shape [..., 3, 4]
        """
        c2l = self.SE3() # [3, 4]
        new_poses = transforms.pose.compose_pair(c2l, l2w) # l2w @ c2l
        return new_poses

