import matplotlib as mpl
mpl.use('TkAgg')
import argparse,os
import pandas as pd
import cv2
import numpy as np
import random
import math
import copy
import pdb
import os
import sys

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.build_model import build_model
from metrics.loss import ContrastDepthLoss, DepthLoss
from dataset.transform import RandomGammaCorrection
from dataset.FAS_dataset import FASDataset
from utils.utils import read_cfg, get_device, get_optimizer, get_rank
from utils.compute_mean_std import compute_mean_std
from engine.CDCN_trainer import FASTrainer

# read config file from config/config.yaml
cfg = read_cfg('./config/config.yaml')

# fix the seed for reproducibility
seed = cfg['seed'] + get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
cudnn.benchmark = True

# build model and engine
device = get_device(cfg)
model = build_model(cfg)
optimizer = get_optimizer(cfg, model)
lr_scheduler = StepLR(optimizer=optimizer, step_size=5, gamma=0.1)
criterion = DepthLoss(device=device)
writer = SummaryWriter(cfg['log_dir'])

dump_input = torch.randn((1, 3, cfg['model']['input_size'][0], cfg['model']['input_size'][1]))
writer.add_graph(model, dump_input)

if cfg['dataset']['mean'] == 'auto' and cfg['dataset']['std'] == 'auto':
    mean, std = compute_mean_std(cfg)
else:
    print("Use mean and std from config file...")
    mean = cfg['dataset']['mean']
    std = cfg['dataset']['std']

train_transform = transforms.Compose([
    RandomGammaCorrection(max_gamma=cfg['dataset']['augmentation']['gamma_correction'][1],
                            min_gamma=cfg['dataset']['augmentation']['gamma_correction'][0]),
    transforms.RandomResizedCrop(cfg['model']['input_size'][0]),
    transforms.ColorJitter(
        brightness=cfg['dataset']['augmentation']['brightness'],
        contrast=cfg['dataset']['augmentation']['contrast'],
        saturation=cfg['dataset']['augmentation']['saturation'],
    ),
    transforms.RandomRotation(cfg['dataset']['augmentation']['rotation_range']),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(cfg['model']['input_size']),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

val_transform = transforms.Compose([
    transforms.Resize(cfg['model']['input_size']),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

trainset = FASDataset(
    root_dir=cfg['dataset']['root'],
    csv_file=cfg['dataset']['train_set'],
    depth_map_size=cfg['model']['depth_map_size'],
    transform=train_transform,
    smoothing=cfg['train']['smoothing']
)

valset = FASDataset(
    root_dir=cfg['dataset']['root'],
    csv_file=cfg['dataset']['val_set'],
    depth_map_size=cfg['model']['depth_map_size'],
    transform=val_transform,
    smoothing=cfg['train']['smoothing']
)

trainloader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=cfg['train']['batch_size'],
    shuffle=True,
    num_workers=4
)

valloader = torch.utils.data.DataLoader(
    dataset=valset,
    batch_size=cfg['val']['batch_size'],
    shuffle=True,
    num_workers=4
)

trainer = FASTrainer(
    cfg=cfg, 
    network=model,
    optimizer=optimizer,
    criterion=criterion,
    lr_scheduler=lr_scheduler,
    device=device,
    trainloader=trainloader,
    valloader=valloader,
    writer=writer
)

print("Start training...")
trainer.train()

writer.close()
