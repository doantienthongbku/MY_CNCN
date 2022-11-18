import yaml
import torch
import torch.optim as optim
import torch.distributed as dist

def read_cfg(cfg_file):
    with open(cfg_file, 'r') as rf:
        cfg = yaml.safe_load(rf)
        return cfg
    
def get_device(cfg):
    device = None
    if cfg['device'] == 'cpu':
        device = torch.device("cpu")
    elif cfg['device'].startswith("cuda"):
        device = torch.device(cfg['device'])
    else:
        raise NotImplementedError
    return device

def get_optimizer(cfg, network):
    optimizer = None
    if cfg['train']['optimizer'] == 'adam':
        optimizer = optim.Adam(network.parameters(), lr=cfg['train']['lr'])
    elif cfg['train']['optimizer'] == 'sgd':
        optimizer = optim.SGD(network.parameters(), lr=cfg['train']['lr'], momentum=0.92, weight_decay=1e-5)
    else:
        raise NotImplementedError

    return optimizer

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()