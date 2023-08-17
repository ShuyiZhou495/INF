import torch
from typing import List

def cam2img(X: torch.Tensor, cam_intr: torch.Tensor) -> torch.Tensor:
    """
    calculate image coordinates from 3D points.
    Args:
        X (): ..., 3
        cam_intr (): ..., 2 [height, width]

    Returns:
        torch.Tensor: y, x coordinate for camera images
    """
    r = torch.linalg.norm(X, axis=-1)
    theta = torch.pi - torch.arctan2(X[..., 1], X[..., 0])  # theta \in [0, 2*pi]
    phi = torch.arccos(X[..., 2] / (r + 1e-6)) # phi \in [0, pi]
    x = theta / (torch.pi * 2) * cam_intr[1]
    y = phi / torch.pi * cam_intr[0]
    return (torch.column_stack((y, x)) - 0.5).to(torch.long)

def img2cam(X: torch.Tensor, cam_intr: torch.Tensor) -> torch.Tensor:
    """
    calculate rays from camera images.
    Args:
        X (torch.Tensor): tensor with shape [N, ..., 2]
        cam_intr (torch.Tensor): tensor with shape [2], inside is (height, width)

    Returns:
        torch.Tensor: directions of image rays. tensor with shape [N, ..., 3], the norm of xyz is 1 rather than z is 1
    """
    theta = torch.pi - X[..., 0] / cam_intr[1] * (torch.pi * 2) # theta \in [0, 2*pi]
    phi = X[..., 1] / cam_intr[0] * torch.pi # phi \in [0, pi]
    xyz = torch.ones((*X.shape[:-1], 3)) 
    phi_sin = torch.sin(phi)
    xyz[..., 0] = phi_sin * torch.cos(theta) # x: forward
    xyz[..., 1] = phi_sin * torch.sin(theta) # y: left
    xyz[..., 2] = torch.cos(phi) # z: upward
    return xyz