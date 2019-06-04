import argparse
import numpy as np
import os
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


import data_v2 as data
import model_v2 as models
import train
import utils


def _parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--datadir', default='/data/kaggle-freesound-2019')
    parser.add_argument('--datadir', default='/data')
    parser.add_argument('--outpath', default='./../runs/')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    return parser.parse_args()


class LinearLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, num_epochs, last_epoch=-1):
        self.num_epochs = max(num_epochs, 1)
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        res = []
        for lr in self.base_lrs:
            res.append(np.maximum(lr * np.minimum(-(self.last_epoch + 1) * 1. / self.num_epochs + 1., 1.), 0.))
        return res


transforms_dict = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ]),
}

def main(args):

    np.random.seed(432)
    torch.random.manual_seed(432)
    try:
        os.makedirs(args.outpath)
    except OSError:
        pass

    experiment_path = utils.get_new_model_path(args.outpath)

    train_writer = SummaryWriter(os.path.join(experiment_path, 'train_logs'))
    val_writer = SummaryWriter(os.path.join(experiment_path, 'val_logs'))
    trainer = train.Trainer(train_writer, val_writer)

    # todo: add config
    trainds, evalds  = data.build_dataset(args.datadir, transforms_dict['train'])

    model = models.cnn()
    opt = torch.optim.Adam(model.parameters())

    for param_group in opt.param_groups:
        param_group['lr'] = 0.001
    #scheduler = LinearLR(opt, 100)
    scheduler = CosineAnnealingLR(opt, T_max=5, eta_min=1e-5)
    #scheduler = StepLR(opt, step_size=30, gamma=0.1)

    trainloader = DataLoader(trainds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    evalloader = DataLoader(evalds, batch_size=256, shuffle=False, num_workers=16, pin_memory=True)

    best_metrics = 0

    for epoch in range(args.epochs):
        print('Epoch:', epoch, 'lr', scheduler.get_lr()[-1])
        trainer.train_epoch(model, opt, trainloader, scheduler.get_lr()[-1])
        metrics = trainer.eval_epoch(model, evalloader)
        scheduler.step()

        state = dict(
            epoch=epoch,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=opt.state_dict(),
            loss=metrics['loss'],
            lwlrap=metrics['lwlrap'],
            global_step=trainer.global_step,
        )
        print('val_loss', metrics['loss'], 'val_lwlrap', metrics['lwlrap'])

        if metrics['lwlrap'] > best_metrics:
            best_metrics = metrics['lwlrap']
            export_path_model = os.path.join(experiment_path, 'last.pth')
            torch.save(state['model_state_dict'], export_path_model)
    #
    #     #torch.save(state, export_path)


if __name__ == "__main__":
    args = _parse_args()
    main(args)
