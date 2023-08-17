import numpy as np
import torch
from scipy.spatial.transform import Rotation
from typing import Any, Union, List, Tuple
from easydict import EasyDict as edict
import os

def matrix_to_euler_angles_scipy(matrix: torch.Tensor) -> np.ndarray:
    """
    SO3 matrix to euler angle

    Args:
        matrix (torch.Tensor): SO3 matrix

    Returns:
        np.ndarray: euler angle in degree in xyz order
    """
    return Rotation.from_matrix(matrix.detach().cpu().numpy()).as_euler("xyz", degrees=True)

def euler_angles_to_matrix_scipy(euler: Union[List, np.ndarray]) -> torch.Tensor:
    """
    euler angle in degree in xyz order to SO3 matrix

    Args:
        euler (Union[List, np.ndarray]): euler angle in degree in xyz order

    Returns:
        torch.Tensor: SO3 matrix
    """
    return torch.tensor(Rotation.from_euler("xyz", euler, degrees=True).as_matrix(), dtype=torch.float)

@torch.no_grad()
def get_ang_tra(pose: torch.Tensor) -> Tuple[List, List]:
    """get the current extrinsic parameters

    Args:
        pose (torch.Tensor): tensor with shape [..., 3, 4] or [..., 4, 4]
    Returns:
        euler angles (XYZ), translation
    """
    ang = matrix_to_euler_angles_scipy(pose[..., :3, :3]).tolist()
    tra = pose[..., :3, 3].tolist()
    return ang, tra

def ang_tra_to_SE3(opt: edict[str, Any], euler: List, trans: List) -> torch.Tensor:
    """given euler angles and translation, make it to SE3

    Args:
        opt (edict[str, Any]): opt
        euler (List): euler angle in xyz order in degree
        trans (List): translation in xyz order

    Returns:
        torch.Tensor: with shape [3, 4]
    """
    rot = euler_angles_to_matrix_scipy(euler)
    t_tensor = torch.tensor(trans).to(torch.float)
    pose = torch.concat([rot, t_tensor[..., None]], dim=-1) # [3, 4]
    return pose.to(opt.device)

@torch.no_grad()
def evaluate_poses(opt: edict[str, Any], pose1: torch.Tensor, pose2: torch.Tensor) -> None:
    """
    evaluate RPE and APE and save in a txt file

    Args:
        opt (edict[str, Any]): opt
        pose1 (torch.Tensor): shape with (N, 3, 4)
        pose2 (torch.Tensor):  shape with (N, 3, 4)
    """
    metrics = {"translation": {}, "rotation": {}}
    rmse = lambda x: (x ** 2).mean() ** 0.5
    torch_rotvec = lambda x: Rotation.from_matrix(x.detach().cpu().numpy()).as_rotvec()
    # --- ape ---
    diff = pose.relative(pose1, pose2)
    # translation
    metrics["translation"]["ape"] = rmse(torch.linalg.norm(diff[:, :3, 3], dim=-1)).tolist()
    # rotation
    rotvec = torch_rotvec(diff[:, :3, :3])
    metrics["rotation"]["ape"] = np.rad2deg(rmse(np.linalg.norm(rotvec, axis=-1))).tolist()
    
    # --- rpe ---
    pose1_r = pose.relative(pose1[:-1], pose1[1:])
    pose2_r = pose.relative(pose2[:-1], pose2[1:])
    diff_r = pose.relative(pose1_r, pose2_r)
    # translation
    metrics["translation"]["rpe"] = rmse(torch.linalg.norm(diff_r[:, :3, 3], dim=-1)).tolist()
    # rotation
    rotvec = torch_rotvec(diff_r[:, :3, :3])
    metrics["rotation"]["rpe"] = np.rad2deg(rmse(np.linalg.norm(rotvec, axis=-1))).tolist()
    
    # save in txt file
    titles = ["translation ape", "translation rpe", "rotation ape", "rotation rpe"]
    data = [metrics["translation"]["ape"], metrics["translation"]["rpe"], metrics["rotation"]["ape"], metrics["rotation"]["rpe"]]
    data = [f"{d:.3f}" for d in data]
    with open(os.path.join(opt.output_path, "pose_eva.txt"), "w") as file:
        for row in zip(titles, data):
            file.write("\t".join(row) + "\n")

class Pose():
    """
    A class of operations on camera poses (PyTorch tensors with shape [...,3,4])
    each [3,4] camera pose takes the form of [R|t]
    """

    def __call__(self,R=None,t=None):
        # construct a camera pose from the given R and/or t
        assert(R is not None or t is not None)
        if R is None:
            if not isinstance(t,torch.Tensor): t = torch.tensor(t)
            R = torch.eye(3,device=t.device).repeat(*t.shape[:-1],1,1)
        elif t is None:
            if not isinstance(R,torch.Tensor): R = torch.tensor(R)
            t = torch.zeros(R.shape[:-1],device=R.device)
        else:
            if not isinstance(R,torch.Tensor): R = torch.tensor(R)
            if not isinstance(t,torch.Tensor): t = torch.tensor(t)
        assert(R.shape[:-1]==t.shape and R.shape[-2:]==(3,3))
        R = R.float()
        t = t.float()
        pose = torch.cat([R,t[...,None]],dim=-1) # [...,3,4]
        assert(pose.shape[-2:]==(3,4))
        return pose

    def invert(self,pose,use_inverse=False):
        # invert a camera pose
        R,t = pose[...,:3],pose[...,3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
        t_inv = (-R_inv@t)[...,0]
        pose_inv = self(R=R_inv,t=t_inv)
        return pose_inv

    def compose(self,pose_list):
        """compose a sequence of poses together
        pose_new(x) = poseN o ... o pose2 o pose1(x)
        """
        pose_new = pose_list[0]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair(pose_new,pose)
        return pose_new

    def compose_pair(self,pose_a,pose_b):
        """compose a sequence of poses together
        pose_new(x) = pose_b o pose_a
        """
        R_a,t_a = pose_a[...,:3],pose_a[...,3:]
        R_b,t_b = pose_b[...,:3],pose_b[...,3:]
        R_new = R_b@R_a
        t_new = (R_b@t_a+t_b)[...,0]
        pose_new = self(R=R_new,t=t_new)
        return pose_new

    def relative(self, pose_a: torch.Tensor, pose_b: torch.Tensor) -> torch.Tensor:
        """relative between poses

        Args:
            pose_a (torch.Tensor): poses with shape (..., 3, 4)
            pose_b (torch.Tensor): poses with shape (..., 3, 4)

        Returns:
            torch.Tensor: relative pose between pose a and pose b
        """
        return self.compose_pair(pose_b, self.invert(pose_a))

class Lie():
    """
    Lie algebra for SO(3) and SE(3) operations in PyTorch
    """

    def so3_to_SO3(self,w): # [...,3]
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        R = I+A*wx+B*wx@wx
        return R

    def SO3_to_so3(self,R,eps=1e-7): # [...,3,3]
        trace = R[...,0,0]+R[...,1,1]+R[...,2,2]
        theta = ((trace-1)/2).clamp(-1+eps,1-eps).acos_()[...,None,None]%np.pi # ln(R) will explode if theta==pi
        lnR = 1/(2*self.taylor_A(theta)+1e-8)*(R-R.transpose(-2,-1)) # FIXME: wei-chiu finds it weird
        w0,w1,w2 = lnR[...,2,1],lnR[...,0,2],lnR[...,1,0]
        w = torch.stack([w0,w1,w2],dim=-1)
        return w

    def se3_to_SE3(self,wu): # [...,3]
        w,u = wu.split([3,3],dim=-1)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        C = self.taylor_C(theta)
        R = I+A*wx+B*wx@wx
        V = I+B*wx+C*wx@wx
        Rt = torch.cat([R,(V@u[...,None])],dim=-1)
        return Rt

    def SE3_to_se3(self,Rt,eps=1e-8): # [...,3,4]
        R,t = Rt.split([3,1],dim=-1)
        w = self.SO3_to_so3(R)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        invV = I-0.5*wx+(1-A/(2*B))/(theta**2+eps)*wx@wx
        u = (invV@t)[...,0]
        wu = torch.cat([w,u],dim=-1)
        return wu    

    def skew_symmetric(self,w):
        w0,w1,w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack([torch.stack([O,-w2,w1],dim=-1),
                          torch.stack([w2,O,-w0],dim=-1),
                          torch.stack([-w1,w0,O],dim=-1)],dim=-2)
        return wx

    def taylor_A(self,x,nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            if i>0: denom *= (2*i)*(2*i+1)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans
    def taylor_B(self,x,nth=10):
        # Taylor expansion of (1-cos(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+1)*(2*i+2)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans
    def taylor_C(self,x,nth=10):
        # Taylor expansion of (x-sin(x))/x**3
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+2)*(2*i+3)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans

pose = Pose()
lie = Lie()
