import numpy as np
import os
import torch
import torchvision.transforms.functional as torchvision_F
import PIL
import tqdm
import threading,queue
import transforms
from easydict import EasyDict as edict
import imageio
import util
from util import log
from typing import Optional, Union, List, Dict, Any, Callable, Tuple

class DatasetBase(torch.utils.data.Dataset):
    """
    A base dataset class. It will be first inherited for 2D images and 3D scans.

    """
    def __init__(self, opt: edict[str, Any], target: str = "train") -> None:
        super().__init__()
        self.opt = opt
        self.target = target
        self.path = os.path.join("data", opt.data.scene)
        self.list = self.get_list()
        self.mask = self.get_mask()

    def get_list(self) -> List[int]:
        """
        Retrieve a list of frame indices to use from the available scans/images.

        Returns:
            List[int]: A list of frame indices representing the frames to be used.
        """
        raise NotImplementedError

    def get_mask(self) -> Optional[Union[torch.Tensor, None]]:
        """
        Retrieve the mask tensor from a specified path.

        Returns:
            Optional[Union[torch.Tensor, None]]: The lidar/camera mask tensor as 2D/1D tensor. Tensor if the mask file exists,
            or None if the file is not found.
        """
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve an item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Dict[str, Any]: A dictionary containing information about the retrieved item, including its index and detail information.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.list)
    
    def prefetch_all_data(self) -> None:
        """
        Preload and save all data into the 'self.all' dictionary.
        
        """
        raise NotImplementedError

    def preload_worker(self, data_list: List, load_func: callable, q: queue.Queue, lock: threading.Lock, idx_tqdm: tqdm.tqdm) -> None:
        # from BARF code
        while True:
            idx = q.get()
            data_list[idx] = load_func(idx)
            with lock:
                idx_tqdm.update()
            q.task_done()

    def preload_threading(self,load_func: Callable, data_str: str = "images") -> List[Any]:
        # from BARF code
        num = len(self)
        data_list = [None]*num
        q = queue.Queue(maxsize=num)
        idx_tqdm = tqdm.tqdm(range(num),desc="preloading {}".format(data_str),leave=False)
        for i in range(num): q.put(i)
        lock = threading.Lock()
        for _ in range(self.opt.data.num_workers):
            t = threading.Thread(target=self.preload_worker,
                                 args=(data_list,load_func,q,lock,idx_tqdm),daemon=True)
            t.start()
        q.join()
        idx_tqdm.close()
        assert(all(map(lambda x: x is not None,data_list)))
        return data_list

class DatasetRGB(DatasetBase):
    """
    Base dataset class for RGB images. It can be inherited for different sensors.

    """

    def __init__(self, opt: edict[str, Any], target: str="train") -> None:
        super().__init__(opt, target)
        self.raw_poses = self.load_poses()
        self.images = [None] * len(self)
        self.cameras = [None] * len(self)

    def get_list(self) -> List[int]:
        all_scans = len(os.listdir(os.path.join(self.path, "images")))
        # use the same frames that density field used
        self.opt.data.length = min(self.opt.density_opt.data.length, all_scans)
        if self.target == "train":
            return list(range(self.opt.data.length))
        elif self.target == "vis":
            # only when calibtration, there is visualization target
            # we just render the first and last frames for your reference
            return [0, self.opt.data.length-1] # first and last

    def get_mask(self) -> torch.Tensor:
        mask_path = os.path.join(self.path, "camera_mask.png") # mask path is hard-coded here
        if os.path.exists(mask_path):
            log.info("Logging camera image mask")
            mask = torchvision_F.to_tensor(self.load_img(mask_path))
            # We use the white area
            mask = mask[0] > 0.8
        else:
            mask = torch.ones((self.opt.H, self.opt.W), dtype=torch.bool)
        mask = mask.view(-1)
        return mask.to(self.opt.device)

    def load_poses(self) -> torch.Tensor:
        """
        load LiDAR poses from the neural density field training result directory.

        Returns:
            torch.Tensor: lidar poses. Shape (-1, 3, 4)
        """
        path = os.path.join(self.opt.density_opt.output_path, "poses.npy")
        raw_poses = torch.from_numpy(np.load(path))
        if self.target=="vis":
            raw_poses = raw_poses[[0, -1]] # the first and last frames will be used
        return raw_poses

    def load_img(self, path:str) -> PIL.Image.Image:
        """load image from path

        Args:
            path (str): image path

        Returns:
            PIL.Image.Image: loaded image
        """
        return PIL.Image.fromarray(imageio.imread(path))

    def get_image(self, idx: int) -> torch.Tensor:
        """retrieve image from certain index

        Args:
            idx (int): index

        Returns:
            torch.Tensor: shape (-1, 3)
        """
        id = self.list[idx]
        image_folder = os.path.join(self.path, "images")
        image = self.load_img(os.path.join(image_folder, f"{id:04}.jpg"))
        # resize image
        if self.target=="vis":
            image = image.resize((self.opt.render_W, self.opt.render_H))
        image = torchvision_F.to_tensor(image)
        return image.reshape(image.shape[0], -1).t()
    
    def get_camera(self, idx: int) -> Tuple[Any, torch.Tensor]:
        """
        Retrieve camera information

        Args:
            idx (int): index

        Returns:
            Tuple[Any, torch.Tensor]: camera intrinsic parameters and corresponding LiDAR poses
        """
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = dict(idx=self.list[idx])
        if self.cameras[idx] is None:
            self.cameras[idx] = self.get_camera(idx)
        intr, pose = self.cameras[idx]

        if self.images[idx] is None:
            self.images[idx] = self.get_image(idx)
        image = self.images[idx]
        
        item.update(
            image=image, 
            intr=intr,
            pose=pose # corresponding LiDAR pose
            )
        return item

    def prefetch_all_data(self) -> None:
        self.images = self.preload_threading(self.get_image)
        all = torch.utils.data._utils.collate.default_collate([s for s in self])
        self.all = edict(
            idx=all["idx"],
            pose=util.move_to_device(all["pose"], self.opt.device),
            intr=util.move_to_device(all["intr"], self.opt.device),
            image=all["image"]
        )
        self.all.update(mask=self.mask)

class Dataset3D(DatasetBase):
    """
    Base dataset class for 3D scans. It can be inherited for different LiDAR sensors.

    """
    
    def __init__(self, opt: edict[str, Any], target: str = "train") -> None:
        super().__init__(opt, target)
        self.pc_range=self.opt.data.near_far
        self.poses = self.load_poses()
        self.scans = [None] * len(self)

    def get_list(self) -> List[int]:
        # use the first several frames
        all_scans = len(os.listdir(os.path.join(self.path, "scans")))
        self.opt.data.length = min(self.opt.data.length, all_scans)
        return list(range(self.opt.data.length))

    def get_mask(self) -> Optional[Union[torch.Tensor, None]]:
        # lidar mask is hard-coded as lidar_mask.png
        lmask_path = os.path.join(self.path, "lidar_mask.png") 
        if os.path.exists(lmask_path):
            log.info("logging LiDAR mask")
            lmask = torch.tensor(np.array(PIL.Image.fromarray(imageio.imread(lmask_path))))[..., 0] > 0.8
            return lmask
        else:
            return None

    def load_poses(self) -> Optional[Union[torch.Tensor, None]]:
        """
        Retrieve reference LiDAR poses from specified path

        Returns:
            Optional[Union(torch.Tensor, None)]: If there exists the gt LiDAR poses, return (n, 3, 4) shaped torch.Tensor.
            Otherwise return None.
                                                
        """
        pose_file = os.path.join(self.path, "poses.npy")
        if not self.opt.train.poses:
            assert os.path.exists(pose_file), "No lidar poses file. Have to estimate LiDAR poses first."
        if os.path.exists(pose_file):
            poses = np.load(os.path.join(self.path, "poses.npy"))[:, :3][self.list]
            # pose of the first frame is the origin of world coodinate.
            poses = transforms.pose.compose_pair(poses, transforms.pose.invert(poses[0]))
            # plot a graph
            util.plot_paths({"gt": poses}, os.path.join(self.opt.output_path, "gt"))
            return poses.float().to(self.opt.device)
        else:
            return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.scans[idx] is None:
            self.scans[idx] = self.get_scan(idx)
        res = self.scans[idx]
        item = dict(idx=idx)
        item.update(res)
        return item

    def get_scan(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve information about a scan point cloud.

        Args:
            idx (int): The index of the scan

        Returns:
            Dict[str, Any]: A dictionary containing information about the scan, including direction (dir), distance (z),
            origin point of the ray (origin), weight (weight), and size of the frame (size).
        """
        raise NotImplementedError  

    def collect(self, src: List[Dict[str, Any]], des: Dict[str, torch.Tensor]):
        """
        concatenate the list of frame information. Similar to torch.utils.data._utils.collate.
        Because not all pointclouds are the same size, do it by manually.
        e.g. directions will be a (-1, 3) tensor with everything stacked, 
        rather than (N, hw, 3) tensor separate as camera photos.

        Args:
            src (List[Dict[str, Any]]): list of dictionary including information for every frame.
            des (Dict[str, torch.Tensor]): the detination diction to update.
        """
        for key in src[0].keys():
            contents = [scan[key] for scan in src]
            if isinstance(contents[0], int):
                des.update({key: torch.tensor(contents)})
            else:
                des.update({key: torch.cat(contents)})  

    def prefetch_all_data(self) -> None:
        self.scans = self.preload_threading(self.get_scan, data_str="lidar scans")
        self.all = edict(idx=list(range(len(self.list))))
        self.collect(self.scans, self.all)

 