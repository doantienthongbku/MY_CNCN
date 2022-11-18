#!/usr/bin/env python3
"""PyTorch Inference Script

An example inference script that outputs top-k class ids for images in a folder into a csv.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import os
import time
import argparse
import logging
import numpy as np
import torch
import glob
import pandas as pd
from tqdm import tqdm
import cv2
import shutil

from timm.models import create_model, apply_test_time_pool
from timm.data import ImageDataset, create_loader, resolve_data_config
from timm.utils import AverageMeter, setup_default_logging

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('inference')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--output_dir', metavar='DIR', default='./',
                    help='path to output files')
parser.add_argument('--model', '-m', metavar='MODEL', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--topk', default=1, type=int,
                    metavar='N', help='Top-k to output to CSV')

def cut_frames(video_path, saved_path):
    deltaFrame = 10
    vid_name, ext = os.path.splitext(os.path.basename(video_path))

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        if count % deltaFrame == 0:
            cv2.imwrite(os.path.join(saved_path, "{}frame{}.jpg".format(vid_name, count)), image)     # save frame as JPEG file      
        success, image = vidcap.read()
        count += 1

def main():
    setup_default_logging()
    args = parser.parse_args()
    # might as well try to do something useful...
    args.pretrained = args.pretrained or not args.checkpoint

    # create model
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint)

    _logger.info('Model %s created, param count: %d' %
                 (args.model, sum([m.numel() for m in model.parameters()])))

    config = resolve_data_config(vars(args), model=model)
    model, test_time_pool = (model, False) if args.no_test_pool else apply_test_time_pool(model, config)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model = model.cuda()
    
    model.eval()

    fname = []
    preds = []
    folders = glob.glob(args.data + '/*.mp4')
    saved_path = '/home/lake/Project/olaz/dataset/test_frame_video'
    for folder in tqdm(folders):
        cut_frames(folder, saved_path)
        loader = create_loader(
            ImageDataset(saved_path),
            input_size=config['input_size'],
            batch_size=args.batch_size,
            use_prefetcher=True,
            interpolation=config['interpolation'],
            mean=config['mean'],
            std=config['std'],
            num_workers=args.workers,
            crop_pct=1.0 if test_time_pool else config['crop_pct'])

        k = min(args.topk, args.num_classes)
        batch_time = AverageMeter()
        end = time.time()
        topk_ids = []
        with torch.no_grad():
            for batch_idx, (input, _) in enumerate(loader):
                input = input.cuda()
                labels = model(input)
                topk = labels.topk(k)[1]
                topk_ids.append(topk.cpu().numpy())

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % args.log_freq == 0:
                    _logger.info('Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                        batch_idx, len(loader), batch_time=batch_time))

        topk_ids = np.concatenate(topk_ids, axis=0)
        lst = topk_ids[:,0].tolist()
        print(lst)
        label = max(set(list(lst)), key=lst.count)
        fname.append(folder.split('/')[-1])
        preds.append(label)
        shutil.rmtree(saved_path)
        os.mkdir(saved_path)

    # debug_df = pd.DataFrame({   'fname': fname,
    #                             'label': preds})

    submit_df = pd.DataFrame({ 'fname': fname,
                        'liveness_score': preds})
    submit_df.to_csv('./submit.csv', index=False)

if __name__ == '__main__':
    main()