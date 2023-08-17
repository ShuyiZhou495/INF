import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
import torch.utils.tensorboard
from typing import List, Optional

@torch.no_grad()
def colorize_depth(dep: torch.Tensor, r: int=30) -> torch.Tensor:
    """
    colorize the rays according to depth value

    Args:
        dep (torch.Tensor): depths of a image
        r (int, optional): max depth value when rendering. Defaults to 30.

    Returns:
        torch.Tensor: colorsized depths. shape with (3, ...)
    """
    dvalue = dep.clamp_max(r)
    dvalue = dvalue / r
    # draw a color map
    cm = torch.from_numpy(plt.cm.rainbow(dvalue.cpu().detach().numpy())).float()[..., :3]
    return cm.permute(-1, *list(range(cm.dim() - 1)))

def to_color_img(opt: edict, depth: torch.Tensor) -> torch.Tensor:
    """
    given a render depth tensor with shape (-1), colorize it with rainbow colormap,
    and reshape it

    Args:
        opt (edict): opt
        depth (torch.Tensor): rendered depth with shape (opt.render_H * opt.render_W)

    Returns:
        torch.Tensor: colorized tensor with shape (1, 3, opt.render_H, opt.render_W), color in 0-1
    """
    depth_c = colorize_depth(depth, opt.render.depth)
    depth_c = depth_c.view(-1, opt.render_H, opt.render_W, 1)
    return depth_c.permute(3,0,1,2)

def to_np_img(image: torch.Tensor) -> np.ndarray:
    """
    reshape the torch tensor prepared for tensorboard to a normal numpy array

    Args:
        image (torch.Tensor): shape of (1, 3, H, W), scaled in 0-1

    Returns:
        np.ndarray: reshape to (H, W, 3), color in 8-bit unsigned integer
    """
    return (image.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

def tb_log_ang_tra(
    tb: torch.utils.tensorboard.SummaryWriter, 
    split: str, 
    name: Optional[str], 
    euler: List, 
    trans: List, 
    iter: int) -> None:
    """
    log euler angles and translation vectors

    Args:
        tb (torch.utils.tensorboard.SummaryWriter): tensorboard summary write
        split (str): split of plot
        name (Optional[str]): name of plot
        euler (List): list of euler angles in degree in x, y, z, order
        trans (List): list of translation vector in x, y, z order
        iter (int): number of iteration
    """
    t = "" if name is None else name + "-"
    for n, title in enumerate(["rx", "ry", "rz"]):
        tb.add_scalar(f"{split}/{t}{title}", euler[n], iter + 1)
    for n, title in enumerate(["x", "y", "z"]):
        tb.add_scalar(f"{split}/{t}{title}", trans[n], iter + 1)

@torch.no_grad()
def tb_image(opt,tb,step,group,name,images,num_vis=None,from_range=(0,1),cmap="gray"):
    images = preprocess_vis_image(opt,images,from_range=from_range,cmap=cmap)
    num_H,num_W = num_vis or opt.tb.num_images
    images = images[:num_H*num_W]
    image_grid = torchvision.utils.make_grid(images[:,:3],nrow=num_W,pad_value=1.)
    if images.shape[1]==4:
        mask_grid = torchvision.utils.make_grid(images[:,3:],nrow=num_W,pad_value=1.)[:1]
        image_grid = torch.cat([image_grid,mask_grid],dim=0)
    tag = "{0}/{1}".format(group,name)
    tb.add_image(tag,image_grid,step)

def preprocess_vis_image(opt,images,from_range=(0,1),cmap="gray"):
    min,max = from_range
    images = (images-min)/(max-min)
    images = images.clamp(min=0,max=1).cpu()
    if images.shape[1]==1:
        images = get_heatmap(opt,images[:,0].cpu(),cmap=cmap)
    return images

def get_heatmap(opt,gray,cmap): # [N,H,W]
    color = plt.get_cmap(cmap)(gray.numpy())
    color = torch.from_numpy(color[...,:3]).permute(0,3,1,2).float() # [N,3,H,W]
    return color

