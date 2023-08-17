import numpy as np
import os
import torch
import random
import string
import yaml
from easydict import EasyDict as edict

import util
from util import log

def parse_arguments(args):
    """
    Parse arguments from command line.
    Syntax: --key1.key2.key3=value --> value
            --key1.key2.key3=      --> None
            --key1.key2.key3       --> True
            --key1.key2.key3!      --> False
    """
    opt_cmd = {}
    for arg in args:
        assert(arg.startswith("--"))
        if "=" not in arg[2:]:
            key_str,value = (arg[2:-1],"false") if arg[-1]=="!" else (arg[2:],"true")
        else:
            key_str,value = arg[2:].split("=")
        keys_sub = key_str.split(".")
        opt_sub = opt_cmd
        for k in keys_sub[:-1]:
            if k not in opt_sub: opt_sub[k] = {}
            opt_sub = opt_sub[k]
        assert keys_sub[-1] not in opt_sub,keys_sub[-1]
        opt_sub[keys_sub[-1]] = yaml.safe_load(value)
    opt_cmd = edict(opt_cmd)
    return opt_cmd

def set(arg):
    opt_cmd = parse_arguments(arg)
    log.info("setting configurations...")
    assert("model" in opt_cmd)
    # load config from yaml file
    assert("yaml" in opt_cmd)
    fname = "options/{}.yaml".format(opt_cmd.yaml)
    opt_base = load_options(fname)
    # override with command line arguments
    opt = override_options(opt_base,opt_cmd,key_stack=[],safe_check=True)
    process_options(opt)
    save_options_file(opt)
    return opt

def load_options(fname):
    with open(fname) as file:
        opt = edict(yaml.safe_load(file))
    if "_parent_" in opt:
        # load parent yaml file(s) as base options
        parent_fnames = opt.pop("_parent_")
        if type(parent_fnames) is str:
            parent_fnames = [parent_fnames]
        for parent_fname in parent_fnames:
            opt_parent = load_options(parent_fname)
            opt_parent = override_options(opt_parent,opt,key_stack=[])
            opt = opt_parent
    print("loading {}...".format(fname))
    return opt

def override_options(opt,opt_over,key_stack=None,safe_check=False):
    for key,value in opt_over.items():
        if isinstance(value,dict):
            # parse child options (until leaf nodes are reached)
            opt[key] = override_options(opt.get(key,dict()),value,key_stack=key_stack+[key],safe_check=safe_check)
        elif value is not None:
            opt[key] = value
    return opt

def process_options(opt):
    # set seed
    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        if opt.seed!=0:
            opt.name = str(opt.name)+f"_seed{opt.seed}"
    else:
        # create random string as run ID
        randkey = "".join(random.choice(string.ascii_uppercase) for _ in range(4))
        opt.name = str(opt.name)+"_{}".format(randkey)
    # other default options
    opt.output_path = f"output/{opt.group}/{opt.name}"

    os.makedirs(opt.output_path,exist_ok=True)
    assert(isinstance(opt.gpu,int)) # disable multi-GPU support for now, single is enough
    opt.device = "cuda:{}".format(opt.gpu)
    
    # use panorama camera model to render
    opt.render_H, opt.render_W = opt.render.image_size
    opt.render_camera = "panorama"
    opt.render_intr = [opt.render_H, opt.render_W]

    # whether there are poses to train
    opt.poses = opt.model=="color" or opt.train.poses 

    # set the optioin for calibration
    if opt.model == "color":
        # for convenience
        opt.H, opt.W = opt.data.image_size
        # load density field settings if train color field
        assert "density_name" in opt, "need to train density field first"
        density_path = f"output/{opt.group}/{opt.density_name}"
        opt.density_opt = load_options(os.path.join(density_path, "options.yaml"))
        opt.train.range = opt.density_opt.train.range
        
def save_options_file(opt):
    opt_fname = "{}/options.yaml".format(opt.output_path)
    if os.path.isfile(opt_fname):
        with open(opt_fname) as file:
            opt_old = yaml.safe_load(file)
        if opt!=opt_old:
            # prompt if options are not identical
            opt_new_fname = "{}/options_temp.yaml".format(opt.output_path)
            with open(opt_new_fname,"w") as file:
                yaml.safe_dump(util.to_dict(opt),file,default_flow_style=False,indent=4)
            print("existing options file found (different from current one)...")
            os.system("diff {} {}".format(opt_fname,opt_new_fname))
            os.system("rm {}".format(opt_new_fname))
        else: print("existing options file found (identical)")
    else: print("(creating new options file...)")
    with open(opt_fname,"w") as file:
        yaml.safe_dump(util.to_dict(opt),file,default_flow_style=False,indent=4)
