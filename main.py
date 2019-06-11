import argparse
import numpy as np
import os
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from torchcontrib.optim import SWA

import data
import models
import train
import utils
import math

#python main.py --outpath './runs' --epochs 60 --batch_size 64
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='/data/kaggle-freesound-2019')
    parser.add_argument('--outpath', default='./runs')
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    return parser.parse_args()


def cyclical_lr(half_period, min_lr=3e-2, max_lr=3e-3):
    half_period = float(half_period)
    scaler = lambda x: 3. / x
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, half_period)

    def relative(it, half_period):
        cycle = math.floor(1. + it / (2. * half_period))
        x = abs(it / half_period - 2. * cycle + 1.)
        return max(0, (1. - x)) * scaler(cycle)
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
    scheduler = cyclical_lr(5, 1e-5, 2e-3)
    trainer = train.Trainer(train_writer, val_writer, scheduler=scheduler)

    train_transform = data.build_preprocessing()
    eval_transform = data.build_preprocessing()

    trainds, evalds = data.build_dataset(args.datadir, None)
    trainds.transform = train_transform
    evalds.transform = eval_transform

    model = models.resnet34()
    base_opt = torch.optim.Adam(model.parameters())
    opt = SWA(base_opt, swa_start=30, swa_freq=10)

    trainloader = DataLoader(trainds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    evalloader = DataLoader(evalds, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)


    export_path = os.path.join(experiment_path, 'last.pth')

    best_lwlrap = 0

    for epoch in range(args.epochs):
        print('Epoch {} - lr {:.6f}'.format(epoch, scheduler(epoch)))
        trainer.train_epoch(model, opt, trainloader, scheduler(epoch))
        metrics = trainer.eval_epoch(model, evalloader)

        print('Epoch: {} - lwlrap: {:.4f}'.format(epoch, metrics['lwlrap']))

        # save best model
        if metrics['lwlrap'] > best_lwlrap:
            best_lwlrap = metrics['lwlrap']
            torch.save(model.state_dict(), export_path)

    print('Best metrics {:.4f}'.format(best_lwlrap))
    opt.swap_swa_sgd()

if __name__ == "__main__":
    args = _parse_args()
    main(args)
