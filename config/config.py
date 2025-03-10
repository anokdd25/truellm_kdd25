import datetime
import json
import os
import random
from pathlib import Path
from torch import optim

import numpy as np
import torch
import yaml

__all__ = ["config"]




def update_config(conf, new_conf):
    for item in new_conf.keys():
        if type(new_conf[item]) == dict and item in conf.keys():
            conf[item] = update_config(conf[item], new_conf[item])
        else:
            conf[item] = new_conf[item]
    return conf


def load_yaml(file):
    try:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    except AttributeError:
        conf_dict = yaml.load(file)
    return conf_dict


class Config:
    def __init__(self, path="config/", cfg=None):
        self.__is_none = False
        self.__data = cfg if cfg is not None else {}
        if path is not None and cfg is None:
            self.__path = os.path.abspath(os.path.join(os.curdir, path))
            with open(os.path.join(self.__path, "default.yaml"), "rb") as default_config:
                self.__data.update(load_yaml(default_config))
            for cfg in sorted(os.listdir(self.__path)):
                if cfg != "default.yaml" and cfg[-4:] in ["yaml", "yml"]:
                    with open(os.path.join(self.__path, cfg), "rb") as config_file:
                        self.__data = update_config(self.__data, load_yaml(config_file))

    def set_(self, key, value):
        self.__data[key] = value

    def set_subkey(self, key, subkey, value):
        self.__data[key][subkey] = value

    def values_(self):
        return self.__data

    def save_(self, file):
        file = os.path.abspath(os.path.join(os.curdir, file))
        with open(file, 'w') as f:
            yaml.dump(self.__data, f)

    def __getattr__(self, item):
        if type(self.__data[item]) == dict:
            return Config(cfg=self.__data[item])
        return self.__data[item]

    def __getitem__(self, item):
        return self.__data[item]


class Singleton:
    def __init__(self, cls):
        self.cls = cls
        self.instance = None

    def __call__(self, *args, **kwargs):
        if self.instance is None:
            self.instance = self.cls(*args, **kwargs)
            return self.instance


config = Singleton(Config)


def transform_input_filters(x):
    x = str(x).replace(" ", "")
    if "[" in x:
        inputs_type1 = []
        inputs_type2 = x.replace("[", "").replace("]", "")
        inputs_type1 = inputs_type2.split(",")
        z = [int(y) for y in inputs_type1]
    return z


def transform_input_thr(x):
    x = str(x).replace(" ", "")
    if "[" in x:
        inputs_type1 = []
        inputs_type2 = x.replace("[", "").replace("]", "")
        inputs_type1 = inputs_type2.split(",")
        z = [float(y) for y in inputs_type1]
    return z

def transform_input_str(x):
    x = str(x).replace(" ", "")
    if "[" in x:
        inputs_type1 = []
        inputs_type2 = x.replace("[", "").replace("]", "")
        inputs_type1 = inputs_type2.split(",")
        z = [str(y) for y in inputs_type1]
    return z

def transform_input_bool(x):
    x = str(x).replace(" ", "")
    if "[" in x:
        inputs_type1 = []
        inputs_type2 = x.replace("[", "").replace("]", "")
        inputs_type1 = inputs_type2.split(",")
        z = [bool(y) for y in inputs_type1]
    return z


def transform_input_filters2(x):
    x = str(x).replace(" ", "")
    if "[" in x:
        inputs_type2 = x.replace("[", "").replace("]", "")
        inputs_type1 = inputs_type2.split(",")
        z = [str(y) for y in inputs_type1]
    return z


def transform_input_filters_multiple(x):
    x = str(x).replace(" ", "")
    if "(" in x:
        inputs_type2 = x.replace("(", "").replace(")", "")
        inputs_type1 = inputs_type2.split(",")
        # print(inputs_type1)
        z = [int(y) for y in inputs_type1]
    return z


def transform_input_eps(x):
    if "/" in x:
        x1 = x.split("/")
        value = float(x1[0]) / float(x1[1])
    else:
        value = float(x)
    return value


def transform_input_downsampling(x):
    if "[" in x:
        inputs_type1 = []
        inputs_type2 = x.replace("[", "").replace("]", "")
        inputs_type1 = inputs_type2.split(",")
        v = []
        for y in inputs_type1:
            # print(y)
            if y.lower() in ('yes', 'true', 't', 'y', '1'):
                v.append(True)
            else:
                v.append(False)
    return v


def transform_input_lr(x):
    if "[" in x:
        inputs_type2 = x.replace("[", "").replace("]", "")
        inputs_type1 = inputs_type2.split(",")
        z = [float(y) for y in inputs_type1]
    else:
        z = float(x)
    return z


def transform_input_type(x):
    if "[" in x:
        inputs_type1 = []
        inputs_type2 = x.replace("[", "").replace("]", "")
        inputs_type1 = inputs_type2.split(", ")
        x = inputs_type1
    return x

def transform_input_optim(x):
    x = str(x).replace(" ", "")
    if "[" in x:
        inputs_type1 = []
        inputs_type2 = x.replace("[", "").replace("]", "")
        inputs_type1 = inputs_type2.split(",")
        z = []
        for y in inputs_type1:
            if y == "optim.Adagrad":
                z.append(optim.Adagrad)
            elif y == "optim.Adam":
                z.append(optim.Adam)
            elif y == "optim.AdamW":
                z.append(optim.AdamW)
            elif y == "optim.RMSprop":
                z.append(optim.RMSprop)
            elif y == "optim.SGD":
                z.append(optim.SGD)
    return z


def str2list(v):
    if isinstance(v, str):
        v1 = v.replace("[", "").replace("]", "")
        return [float(v1)]
    else:
        return v


def str2list_bol(v):
    if isinstance(v, str):
        return [float(v)]
    else:
        return v





def str2hexa(x):
    try:
        x2 = x.replace(")", "").replace("(", "").split(", ")
        return int(x2[0], 16), int(x2[1], 16)
    except:
        return x


def two_args_str_int(x):
    try:
        return int(x)
    except:
        return x


def two_args_str_float(x):
    try:
        return float(x)
    except:
        return x


def transform_input_type(x):
    if "[" in x:
        inputs_type1 = []
        inputs_type2 = x.replace("[", "").replace("]", "")
        inputs_type1 = inputs_type2.split(", ")
        x = inputs_type1
    return x


def transform_input_list_list(x):
    if "[" in x:
        x_str = x.replace(" ", "")
        x_str_list = x_str.split("),(")  # .replace("]", "")
        res = []
        for x_bit in x_str_list:
            xpixel = x_bit.split(",")
            bit = []
            for xvalue in xpixel:
                # print(xvalue)
                xvalueint = int(
                    xvalue.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace(",", ""))
                bit.append(xvalueint)
            assert len(bit) == 3
            res.append(bit)
        x = res
    return x


def init_all_for_run(args):
    date = str(datetime.datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").replace(".", "_")
    slsh = "/"
    path_save_model = args.models_path + date + slsh  # args.cipher + slsh +str(args.nombre_round_eval) +slsh + name_input + slsh + date + slsh
    path_save_model_train = args.models_path + date + slsh
    print()
    print("Folder save: ", path_save_model)
    if not os.path.exists(path_save_model):
        os.makedirs(path_save_model)
    with open(path_save_model + 'commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    Path(path_save_model_train).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:" + str(args.device_ids[0]) if torch.cuda.is_available() else "cpu")
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2,4'
    print()
    print("Use Hardware : ", device)
    print()
    # Reproductibilites
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return device, path_save_model




__all__ = ["Config", "transform_input_filters", "transform_input_filters2", "transform_input_optim", \
    "transform_input_thr", "two_args_str_int" ]