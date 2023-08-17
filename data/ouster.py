from .base import Dataset3D
import os
import torch
import numpy as np
import open3d as o3d
from typing import Dict, Any

class Dataset(Dataset3D):
    """
    Specified for current data that are measured from Ouster
    
    """

    def get_scan(self, idx: int) -> Dict[str, Any]:
        # In current data, a lidar scan is saved in a .npy file as the follows:
        # 0th column, 1st column: x, y coordinate of the origins of LiDAR beams
        # 2nd column, 3rd column: phi and theta angle in spherical coordinates
        # 4th column: distance of the surface that the beam reached or 0 if no surface reached
        
        pc = self.load_file(idx)
        origin = torch.column_stack([pc[:, :2], torch.zeros_like(pc[:, 0])])
        z = torch.sin(pc[:, 2])
        y = torch.cos(pc[:, 2]) * torch.sin(pc[:, 3])
        x = torch.cos(pc[:, 2]) * torch.cos(pc[:, 3])
        dirs = torch.column_stack((x, y, z))
        r = pc[:, 4]
        
        # filter out the too far points
        f = (r<=self.pc_range[1])

        # filter out the points outside the mask
        if self.mask is not None:
            sy, sx = self.mask.shape[:2]
            x = ((-pc[:, 3] + torch.pi) / (torch.pi * 2) * sx).to(torch.long) 
            y = ((0.5-pc[:, 2] / torch.pi) * sy).to(torch.long)
            mask = self.mask[y, x]
            f = f & mask

        # load or calculate the weights of the rays
        if self.opt.train.use_weight:
            os.makedirs(os.path.join(self.path, "weight"), exist_ok=True)
            weight_path = os.path.join(self.path, "weight", f"{self.list[idx]:04}.npy")
            if os.path.exists(weight_path):
                weight = torch.from_numpy(np.load(weight_path))
            else:
                pc = origin + dirs * r[..., None]
                weight = self.cal_weight(pc.numpy(), (r>0).numpy())
                np.save(weight_path, weight)
                weight = torch.from_numpy(weight).float()
        else:
            weight = torch.ones_like(r)

        return {
            "dir": dirs[f], # mandatory
            "z": r[f], # mandatory
            "size": r[f].shape[0], # mandatory
            "weight": weight[f], # if not applicable, fill it with torch.ones()
            "origin": origin[f] # if not applicable, fill it with torch.zeros()
        }
    
    def load_file(self, idx: int) -> torch.Tensor:
        """load the LiDAR scan of certain index

        Args:
            idx (int): index of the scan

        Returns:
            torch.Tensor: torch.Tensor with raw data, dtype is torch.float
        """
        file_path = os.path.join(self.path, "scans", f"{self.list[idx]:04}.npy")
        pc = torch.from_numpy(np.load(file_path)).to(torch.float)
        return pc
    
    def cal_weight(self, pointcloud: np.ndarray, filter_: np.ndarray) -> np.ndarray:
        """calculate the weights of the rays

        Args:
            pointcloud (np.ndarray): shape (-1, 3), xyz coordinates
            filter_ (np.ndarray): filter that tells the reached points

        Returns:
            np.ndarray: weights of the rays
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud[filter_])
        pcd.estimate_normals()
        pcd.normalize_normals()
        nm = np.asarray(pcd.normals)

        nms = np.zeros((filter_.shape[0], 3))
        nms[filter_] = nm
        nms = nms.reshape((128, -1, 3))
        nms_padding = np.zeros((nms.shape[0]+2, nms.shape[1]+2, 3))
        nms_padding[1:-1, 1:-1] = nms
        edges = np.stack([
            nms_padding[0:-2, 1:-1]*nms,
            nms_padding[0:-2, 0:-2]*nms,
            nms_padding[1:-1, 2:]*nms,
            nms_padding[1:-1, :-2]*nms,
            nms_padding[2:, 2:]*nms,
            nms_padding[2:, :-2]*nms,
            nms_padding[:-2, 2:]*nms,
            nms_padding[2:, 1:-1]*nms
        ])
        edges = edges.sum(axis=-1).mean(axis=0).reshape(-1)
        e = (1-edges)/2
        e[~filter_] = 0
        weight = e * 0.8 + 0.2 # hard-coded here!
        return weight
