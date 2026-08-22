import torch
import torch.nn as nn
from double_conv import double_conv
def decoder(x, skip_connections, saved_tools=None, num_classes=17):
    if saved_tools is None:
        # PyTorch upgrade for the dictionary lists
        saved_tools = nn.ModuleDict({'up': nn.ModuleList(), 'blend': nn.ModuleList(), 'final': None})
        for skip in reversed(skip_connections):
            current_in_channel = x.shape[1]
            current_out_channel= skip.shape[1]
            up_layer = nn.ConvTranspose2d(current_in_channel,current_out_channel, kernel_size=2, stride=2).to(x.device)
            blend_layer = double_conv(current_out_channel* 2, current_out_channel).to(x.device)
            saved_tools['up'].append(up_layer)
            saved_tools['blend'].append(blend_layer)
            up_x = up_layer(x)
            concat_x = torch.cat((skip, up_x), dim=1)
            x = blend_layer(concat_x)
        final_layer = nn.Conv2d(x.shape[1], num_classes, kernel_size=1).to(x.device)
        saved_tools['final'] = final_layer
        predictions = final_layer(x)
    else:
        for i, skip in enumerate(reversed(skip_connections)):
            up_layer = saved_tools['up'][i]
            blend_layer = saved_tools['blend'][i]
            up_x = up_layer(x)
            concat_x = torch.cat((skip, up_x), dim=1)
            x = blend_layer(concat_x)
        predictions = saved_tools['final'](x)
    return predictions, saved_tools