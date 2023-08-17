import time
import torch
import termcolor
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from typing import Any


# convert to colored strings
def red(message, **kwargs): return termcolor.colored(str(message), color="red",
                                                     attrs=[k for k, v in kwargs.items() if v is True])


def green(message, **kwargs): return termcolor.colored(str(message), color="green",
                                                       attrs=[k for k, v in kwargs.items() if v is True])


def blue(message, **kwargs): return termcolor.colored(str(message), color="blue",
                                                      attrs=[k for k, v in kwargs.items() if v is True])


def cyan(message, **kwargs): return termcolor.colored(str(message), color="cyan",
                                                      attrs=[k for k, v in kwargs.items() if v is True])


def yellow(message, **kwargs): return termcolor.colored(str(message), color="yellow",
                                                        attrs=[k for k, v in kwargs.items() if v is True])


def magenta(message, **kwargs): return termcolor.colored(str(message), color="magenta",
                                                         attrs=[k for k, v in kwargs.items() if v is True])


def grey(message, **kwargs): return termcolor.colored(str(message), color="grey",
                                                      attrs=[k for k, v in kwargs.items() if v is True])


def get_time(sec):
    d = int(sec // (24 * 60 * 60))
    h = int(sec // (60 * 60) % 24)
    m = int((sec // 60) % 60)
    s = int(sec % 60)
    return d, h, m, s

class Log:
    def __init__(self):
        pass

    def process(self, pid):
        print(grey("Process ID: {}".format(pid), bold=True))

    def title(self, message):
        print(yellow(message, bold=True, underline=True))

    def info(self, message):
        print(magenta(message, bold=True))

    def options(self, opt, level=0):
        for key, value in sorted(opt.items()):
            if isinstance(value, (dict, edict)):
                print("   " * level + cyan("* ") + green(key) + ":")
                self.options(value, level + 1)
            else:
                print("   " * level + cyan("* ") + green(key) + ":", yellow(value))

    def loss_train(self, opt, ep, lr, loss, timer):
        if not opt.max_epoch: return
        message = grey("[train] ", bold=True)
        message += "epoch {}/{}".format(cyan(ep, bold=True), opt.max_epoch)
        message += ", lr:{}".format(yellow("{:.2e}".format(lr), bold=True))
        message += ", loss:{}".format(red("{:.3e}".format(loss), bold=True))
        message += ", time:{}".format(blue("{0}-{1:02d}:{2:02d}:{3:02d}".format(*get_time(timer.elapsed)), bold=True))
        message += " (ETA:{})".format(blue("{0}-{1:02d}:{2:02d}:{3:02d}".format(*get_time(timer.arrival))))
        print(message)

    def loss_val(self, opt, loss):
        message = grey("[val] ", bold=True)
        message += "loss:{}".format(red("{:.3e}".format(loss), bold=True))
        print(message)


log = Log()


def update_timer(opt, timer, ep, it_per_ep):
    momentum = 0.99
    timer.elapsed = time.time() - timer.start
    timer.it = timer.it_end - timer.it_start
    # compute speed with moving average
    timer.it_mean = timer.it_mean * momentum + timer.it * (1 - momentum) if timer.it_mean is not None else timer.it
    timer.arrival = timer.it_mean * it_per_ep * (opt.data.length - ep)


# move tensors to device in-place
def move_to_device(X, device):
    if isinstance(X, dict):
        for k, v in X.items():
            X[k] = move_to_device(v, device)
    elif isinstance(X, list):
        for i, e in enumerate(X):
            X[i] = move_to_device(e, device)
    elif isinstance(X, tuple) and hasattr(X, "_fields"):  # collections.namedtuple
        dd = X._asdict()
        dd = move_to_device(dd, device)
        return type(X)(**dd)
    elif isinstance(X, torch.Tensor):
        return X.to(device=device)
    return X


def to_dict(D, dict_type=dict):
    D = dict_type(D)
    for k, v in D.items():
        if isinstance(v, dict):
            D[k] = to_dict(v, dict_type)
    return D


def rays_from_image(opt:edict[str, Any], width: float, height: float, camera: Any, intr: torch.Tensor) -> torch.Tensor:
    """
    calculate rays from image
    
    Args:
        opt (edict[str, Any]): opt
        width (float): image width
        height (float): image height
        camera (Any): imported camera library
        intr (torch.Tensor): camera intrinsic
    
    Returns:
        torch.Tensor: camera rays. tensor with shape [HW, 2]
    """
    y_range = torch.arange(height,dtype=torch.float32,device=opt.device)
    x_range = torch.arange(width,dtype=torch.float32,device=opt.device)
    Y,X = torch.meshgrid(y_range,x_range, indexing="ij") # [H,W]
    uv = torch.stack([X,Y],dim=-1).view(-1,2) # [HW,2]
    return camera.img2cam(uv, intr)


def plot_paths(data: dict[str, torch.Tensor], path: str) -> None:
    """
    plot LiDAR poses trajectory
    
    Args:
        data (dict[str, torch.Tensor]): {legend: poses}, legend should be one of "gt", "res", "key", poses should be np array with shape(-1, 4, 4) or (-1, 3, 4)
        path (str): do not include extension

    """
    res_file = path + "-poses.png"
    _, axs = plt.subplots(1, 1)
    axs.set_ylabel("Y(m)", fontweight="bold")
    axs.set_xlabel("X(m)", fontweight="bold")
    axs.grid = True

    axs.set_ylim([-0.005, 0.005])
    axs.set_xlim([-0.005, 0.005])
    colors = {
        "gt": "#7e6a9d",
        "res": "#df8244",
        "key": "#ff0000"
    }
    markers = {"gt": "o", "res": "v", "key": "D"}
    legend_elements = []

    def decide_lim(origin_lim, poses):
        mi, ma = poses.min(), poses.max()
        length = ma - mi
        mi, ma = mi - length / 10, ma + length / 10
        omi, oma = origin_lim
        return [min(mi, omi), max(ma, oma)]

    def plot_path(legend, poses):
        axs.scatter(poses[:, 0, 3], poses[:, 1, 3], c=colors[legend], marker=markers[legend])
        if legend != "key":
            axs.plot(poses[:, 0, 3], poses[:, 1, 3], c=colors[legend])
            axs.set_xlim(decide_lim(axs.get_xlim(), poses[:, 0, 3]))
            axs.set_ylim(decide_lim(axs.get_ylim(), poses[:, 1, 3]))
            legend_elements.append(Line2D([0], [0], color=colors[legend], marker=markers[legend], label=legend))
        else:
            legend_elements.append(Line2D([0], [0], color="w", markerfacecolor=colors[legend], marker=markers[legend], label=legend))

    for key, value in data.items():
        plot_path(key, value)
        
    axs.set_aspect('equal', adjustable='box')
    axs.axis("scaled")
    axs.legend(handles=legend_elements)
    # plt.rcParams["text.usetex"] = True
    plt.savefig(res_file, format="png", dpi=300, bbox_inches="tight")
