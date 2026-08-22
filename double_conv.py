import torch.nn as nn
def double_conv(in_channel,out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel,out_channel,3,padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channel),
        nn.Conv2d(out_channel, out_channel, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channel)
    )