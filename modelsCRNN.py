import torch
from torch import nn
from torch.nn import functional as F
import math

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, activation):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.GRU(nIn, nHidden, bidirectional=True, batch_first=True, bias=False)
        self.embedding = nn.Sequential(nn.Linear(nHidden * 2, nHidden * 2, bias=False), nn.ReLU(),
                                       nn.Linear(nHidden * 2, nOut, bias=False))
        self.activation = activation

    def forward(self, input):
        recurrent, h = self.rnn(input)
        recurrent = recurrent.contiguous()

        b, T, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)

        output = output.view(b, T, -1)
        return self.activation(output)

class CNNBlock(nn.Module):
    def __init__(self):
        super(CNNBlock, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        out_relu = self.relu(x)
        out_sigmoid = self.sigmoid(x)
        x = out_sigmoid * out_relu

        x = self.conv(x)
        out_relu = self.relu(x)
        out_sigmoid = self.sigmoid(x)
        x = out_sigmoid * out_relu

        x = self.conv(x)
        out_relu = self.relu(x)
        out_sigmoid = self.sigmoid(x)
        x = out_sigmoid * out_relu
        return x


class SingleChannelResnet(nn.Module):
    def __init__(self):
        self.inplanes = 64
        super(SingleChannelResnet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(10, 1), stride=(2,1), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(10, 1), stride=(5, 1), padding=0, bias=False),
        )
        self.layer1 = nn.Sequential(CNNBlock(), nn.MaxPool2d((1, 2)))
        self.layer2 = nn.Sequential(CNNBlock(), nn.MaxPool2d((1, 2)))
        self.layer3 = nn.Sequential(CNNBlock(), nn.MaxPool2d((1, 2)))
        self.layer4 = nn.Sequential(nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False), nn.MaxPool2d((1, 8)))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.rnn_sigmoid = BidirectionalLSTM(256, 128, 80, nn.Sigmoid())
        self.rnn_softmax = BidirectionalLSTM(256, 128, 80, nn.Softmax(dim=-1))

        self.fc = nn.Linear(64, 80)

    def outfunc(self, vects):
        sigmoid, soft_max = vects
        soft_max = torch.clamp(soft_max, 1e-7, 1.)
        out = torch.sum(sigmoid * soft_max, dim=1) / torch.sum(sigmoid, dim=1)
        out = torch.clamp(out, 1e-9, 1.)
        return out

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size()[0], -1, x.size()[1])

        x_sig = self.rnn_sigmoid(x)
        #print()
        #print('sig', torch.sqrt(torch.sum((x_sig.data) ** 2)))

        x_soft_max = self.rnn_softmax(x)
        #print('soft', torch.sqrt(torch.sum((x_soft_max.data) ** 2)))

        x = self.outfunc([x_sig, x_soft_max])
        #print(torch.sqrt(torch.sum((x.data) ** 2)))
        #print(torch.min(x), torch.max(x))

        return x


def resnet34(**kwargs):
    print('m', torch.cuda.max_memory_allocated(torch.device('cuda')))
    model = SingleChannelResnet(**kwargs)
    return model
