import argparse
import numpy as np
import os
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import matplotlib.pyplot as plt

import data
import models
import train
import utils
import math


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='/data/kaggle-freesound-2019')
    parser.add_argument('--outpath', default='./runs')
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    return parser.parse_args()


def cyclical_lr(half_period, min_lr=3e-2, max_lr=3e-3):
    scaler = lambda x: 2 / x
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, half_period)

    def relative(it, half_period):
        cycle = math.floor(1 + it / (2 * half_period))
        x = abs(it / half_period - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)
    return lr_lambda

def main(args):
    np.random.seed(432)
    torch.random.manual_seed(432)
    try:
        os.makedirs(args.outpath)
    except OSError:
        pass
    experiment_path = utils.get_new_model_path(args.outpath)
    print(experiment_path)
    train_writer = SummaryWriter(os.path.join(experiment_path, 'train_logs'))
    val_writer = SummaryWriter(os.path.join(experiment_path, 'val_logs'))
    trainer = train.Trainer(train_writer, val_writer)

    # todo: add config
    train_transform = data.build_preprocessing()
    eval_transform = data.build_preprocessing()

    trainds, evalds = data.build_dataset(args.datadir, None)
    trainds.transform = train_transform
    evalds.transform = eval_transform

    model = models.resnet34()
    opt = torch.optim.Adam(model.parameters())

    trainloader = DataLoader(trainds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    evalloader = DataLoader(evalds, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)


    export_path = os.path.join(experiment_path, 'last.pth')

    best_lwlrap = 0

    scheduler = cyclical_lr(5, 1e-4, 1e-2)

    for epoch in range(args.epochs):
        print('lr {}'.format(scheduler(epoch)))
        trainer.train_epoch(model, opt, trainloader, scheduler(epoch))
        metrics = trainer.eval_epoch(model, evalloader)

        print('Epoch: {} - lwlrap: {:.4f}'.format(epoch, metrics['lwlrap']))

        if metrics['lwlrap'] > best_lwlrap:
            best_lwlrap = metrics['lwlrap']
            torch.save(model.state_dict(), export_path)

    print('Best metrics {:.4f}'.format(best_lwlrap))


if __name__ == "__main__":
    args = _parse_args()
    main(args)
