import torch
import os
import sys
import torchvision.transforms as transforms
import torch.utils.data as data
from tqdm import tqdm

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dataset.FAS_dataset import FASDataset
from utils.utils import read_cfg

def compute_mean_std(cfg):
    print("Auto compute mean and std of dataset...")
    transform = transforms.Compose([
        transforms.Resize(cfg['model']['input_size']),
        transforms.ToTensor()
    ])

    dataset = FASDataset(
        root_dir=cfg['dataset']['root'],
        csv_file=cfg['dataset']['train_set'],
        depth_map_size=cfg['model']['depth_map_size'],
        transform=transform,
        smoothing=cfg['train']['smoothing']
    )

    loader = data.DataLoader(dataset,
                            batch_size=1,
                            num_workers=4,
                            shuffle=False)

    mean = 0.
    std = 0.
    for images, depth_map, label in tqdm(loader):
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)

    print("mean of dataset: ", mean)
    print("std of dataset: ", std)
    
    return mean, std
