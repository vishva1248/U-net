import torch.nn as nn
from double_conv import double_conv
def encoder(pad_image,device,saved_tools=None):
    x = pad_image.float()
    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    skip_connections = []
    if saved_tools is None:
        saved_tools = nn.ModuleList()
        current_in_channel = x.shape[1]
        current_out_channel = 64
        while x.shape[2] > 10:
            layer = double_conv(current_in_channel,current_out_channel).to(device)
            saved_tools.append(layer)
            x = layer(x)
            skip_connections.append(x)
            x = pool(x)
            current_in_channel = current_out_channel
            current_out_channel = current_out_channel * 2
    else:
        for layer in saved_tools:
            x = layer(x)
            skip_connections.append(x)
            x = pool(x)
    return x, skip_connections, saved_tools
#completed