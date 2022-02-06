import torch
import numpy as np
from tqdm import tqdm
import logging
import os
import pickle
import torch.nn.functional as F
import argparse
import torch.distributed as dist
import yaml
import copy
from typing import List
from ast import literal_eval


def main_process(distributed) -> bool:
    if distributed:
        rank = dist.get_rank()
        if rank == 0:
            return True
        else:
            return False
    else:
        return True


def setup(port: int,
          rank: int,
          world_size: int) -> None:
    """
    Used for distributed learning
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup() -> None:
    """
    Used for distributed learning
    """
    dist.destroy_process_group()


def find_free_port() -> int:
    """
    Used for distributed learning
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def to_np_image(x):
    x = x.permute(1, 2, 0).cpu().numpy()
    # x *= std
    # x += mean
    return x


def get_model_dir(args: argparse.Namespace):
    return os.path.join(args.res_dir,
                        f'source={str(args.source_data)}',
                        f'target={str(args.target_data)}',
                        f'method={args.method}')


def get_logs_path(model_path, method, shot):
    exp_path = '_'.join(model_path.split('/')[1:])
    file_path = os.path.join('tmp', exp_path, method)
    os.makedirs(file_path, exist_ok=True)
    return os.path.join(file_path, f'{shot}.txt')


def get_features(model, samples):
    features, _ = model(samples, True)
    features = F.normalize(features.view(features.size(0), -1), dim=1)
    return features


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0

    def update(self, val, init, alpha=0.2):
        self.val = val
        if init:
            self.avg = val
        else:
            self.avg = alpha * val + (1 - alpha) * self.avg


def setup_logger(filepath):
    file_formatter = logging.Formatter(
        "[%(asctime)s %(filename)s:%(lineno)s] %(levelname)-8s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger('example')
    # handler = logging.StreamHandler()
    # handler.setFormatter(file_formatter)
    # logger.addHandler(handler)

    file_handle_name = "file"
    if file_handle_name in [h.name for h in logger.handlers]:
        return
    if os.path.dirname(filepath) != '':
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
    file_handle = logging.FileHandler(filename=filepath, mode="a")
    file_handle.set_name(file_handle_name)
    file_handle.setFormatter(file_formatter)
    logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    return logger


def warp_tqdm(data_loader, disable_tqdm):
    if disable_tqdm:
        tqdm_loader = data_loader
    else:
        tqdm_loader = tqdm(data_loader, total=len(data_loader))
    return tqdm_loader


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', folder='result/default'):
    os.makedirs(folder, exist_ok=True)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        torch.save(state, os.path.join(folder, 'model_best.pth.tar'))


def load_checkpoint(model, model_path, type='best'):
    if type == 'best':
        checkpoint = torch.load('{}/model_best.pth.tar'.format(model_path))
        print(f'Loaded model from {model_path}/model_best.pth.tar')
    elif type == 'last':
        checkpoint = torch.load('{}/checkpoint.pth.tar'.format(model_path))
        print(f'Loaded model from {model_path}/checkpoint.pth.tar')
    else:
        assert False, 'type should be in [best, or last], but got {}'.format(type)
    state_dict = checkpoint['state_dict']
    names = []
    for k, v in state_dict.items():
        names.append(k)
    model.load_state_dict(state_dict)


def compute_confidence_interval(data, axis=0):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a, axis=axis)
    std = np.std(a, axis=axis)
    pm = 1.96 * (std / np.sqrt(a.shape[axis]))
    m = m.astype(np.float64)
    pm = pm.astype(np.float64)
    return m, pm


class CfgNode(dict):
    """
    CfgNode represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                # Convert dict to CfgNode
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())


def _decode_cfg_value(v):
    if not isinstance(v, str):
        return v
    try:
        v = literal_eval(v)
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    original_type = type(original)

    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    casts = [(tuple, list), (list, tuple)]
    try:
        casts.append((str, unicode))  # noqa: F821
    except Exception:
        pass

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )


def load_cfg_from_cfg_file(file: str):
    cfg = {}
    assert os.path.isfile(file) and file.endswith('.yaml'), \
        '{} is not a yaml file'.format(file)

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    for key in cfg_from_file:
        for k, v in cfg_from_file[key].items():
            cfg[k] = v

    cfg = CfgNode(cfg)
    return cfg


def merge_cfg_from_list(cfg: CfgNode,
                        cfg_list: List[str]):
    new_cfg = copy.deepcopy(cfg)
    assert len(cfg_list) % 2 == 0, cfg_list
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        subkey = full_key.split('.')[-1]
        assert subkey in cfg, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, cfg[subkey], subkey, full_key
        )
        setattr(new_cfg, subkey, value)

    return new_cfg
