from collections import defaultdict
from sklearn.metrics import label_ranking_average_precision_score
import torch
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.autograd import Variable

import utils

plt.switch_backend('agg')

def mixup_data(x, y, alpha=1.0, use_cuda=False):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def apply_wd(model, gamma):
    for name, tensor in model.named_parameters():
        if 'bias' in name:
            continue
        tensor.data.add_(-gamma * tensor.data)


def grad_norm(model):
    grad = 0.0
    count = 0
    for name, tensor in model.named_parameters():
        if tensor.grad is not None:
            grad += torch.sqrt(torch.sum((tensor.grad.data) ** 2))
            count += 1
    return grad.cpu().numpy() / count


class Trainer:
    global_step = 0

    def __init__(self, train_writer=None, eval_writer=None, compute_grads=True, device=None):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        self.device = device
        self.train_writer = train_writer
        self.eval_writer = eval_writer
        self.compute_grads = compute_grads
        self.alpha = 0.7

    def train_epoch(self, model, optimizer, dataloader, lr, log_prefix=""):
        device = self.device

        model = model.to(device)
        model.train()
        criterion = nn.BCEWithLogitsLoss()

        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr

        for batch in tqdm(dataloader):
            x = batch['logmel'].to(device)
            y = batch['labels'].to(device)

            inputs, targets_a, targets_b, lam = mixup_data(x, y, self.alpha)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

            optimizer.zero_grad()
            out = model(inputs)

            # out_ravel = out.view(-1,)
            # y_ravel = y.view(-1,)
            # loss = (criterion(out, y) +
            #         criterion(out_ravel, y_ravel)) / 2

            loss = mixup_criterion(criterion, out, targets_a, targets_b, lam)

            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(out).cpu().data.numpy()
            lrap = label_ranking_average_precision_score(batch['labels'], probs)

            log_entry = dict(
                lrap=lrap,
                loss=loss.item(),
                lr=lr,
            )
            if self.compute_grads:
                log_entry['grad_norm'] = grad_norm(model)

            for name, value in log_entry.items():
                if log_prefix != '':
                    name = log_prefix + '/' + name
                self.train_writer.add_scalar(name, value, global_step=self.global_step)
            self.global_step += 1

    def eval_epoch(self, model, dataloader, log_prefix=""):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        criterion = nn.BCEWithLogitsLoss()
        model = model.to(device)
        model.eval()
        metrics = defaultdict(list)
        lwlrap = utils.lwlrap_accumulator()
        for batch in tqdm(dataloader):
            with torch.no_grad():
                x = batch['logmel'].to(device)
                y = batch['labels'].to(device)

                inputs, targets_a, targets_b, lam = mixup_data(x, y, self.alpha)
                inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

                out = model(inputs)

                # out_ravel = out.view(-1, )
                # y_ravel = y.view(-1, )
                # loss = (criterion(out, y) +
                #         criterion(out_ravel, y_ravel)) / 2

                loss = mixup_criterion(criterion, out, targets_a, targets_b, lam)

                probs = torch.sigmoid(out).cpu().data.numpy()
                lrap = label_ranking_average_precision_score(batch['labels'], probs)
                lwlrap.accumulate_samples(batch['labels'], probs)

                metrics['loss'].append(loss.item())
                metrics['lrap'].append(lrap)

        metrics = {key: np.mean(values) for key, values in metrics.items()}
        metrics['lwlrap'] = lwlrap.overall_lwlrap()

        for name, value in metrics.items():
            if log_prefix != '':
                name = log_prefix + '/' + name
            self.eval_writer.add_scalar(name, value, global_step=self.global_step)

        fig = plt.figure(figsize=(12, 9))
        z = lwlrap.per_class_lwlrap() * lwlrap.per_class_weight()
        plt.bar(np.arange(len(z)), z)
        plt.hlines(np.mean(z), 0, len(z), linestyles='dashed')
        plt.ylim([0, 0.013])
        plt.xlim([-1, 80])
        plt.grid()
        self.eval_writer.add_figure('per_class_weighted_lwlrap', fig, global_step=self.global_step)

        fig = plt.figure(figsize=(12, 9))
        z = lwlrap.per_class_lwlrap()
        plt.bar(np.arange(len(z)), z)
        plt.hlines(np.mean(z), 0, len(z), linestyles='dashed')
        plt.xlim([-1, 80])
        plt.grid()
        self.eval_writer.add_figure('per_class_lwlrap', fig, global_step=self.global_step)

        return metrics
