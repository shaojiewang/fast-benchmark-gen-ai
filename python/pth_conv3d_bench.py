
import torch
import torch.nn as nn
import torch.nn.functional as F

SPLIT_SIZE = 16
torch_conv3d = F.conv3d

def split_conv(input, weight, *args, **kwargs):
    out_channels, in_channels_over_groups, kT, kH, kW = weight.shape
    element_num = in_channels_over_groups * input.shape[2] * input.shape[3] * input.shape[4]
    if element_num < (1 << 31) and out_channels != 1024:
        return torch_conv3d(input, weight, *args, **kwargs)
    else:
        output = None
        if out_channels != 1024 and out_channels != 3 and in_channels_over_groups != 256:
            split_inputs = torch.chunk(input, 32, dim=1)
            split_conv_weight = torch.chunk(weight, 32, dim=1)
        elif in_channels_over_groups == 256:
            split_inputs = torch.chunk(input, 64, dim=1)
            split_conv_weight = torch.chunk(weight, 64, dim=1)
        else:
            split_inputs = torch.chunk(input, SPLIT_SIZE, dim=1)
            split_conv_weight = torch.chunk(weight, SPLIT_SIZE, dim=1)
        for i in range(len(split_inputs)):
            if i == 0:
                output = torch_conv3d(split_inputs[i], split_conv_weight[i], *args, **kwargs)
                #  since bias only needs to added once, we set it to None after i==0
                args = list(args)
                args[0] = None
            else:
                output += torch_conv3d(split_inputs[i], split_conv_weight[i], *args, **kwargs)
        return output

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--out_channels", type=int)
    parser.add_argument("--in_channels", type=int)
    parser.add_argument("--time", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--weight", type=int)
    parser.add_argument("--kernel_time", type=int)
    parser.add_argument("--kernel_height", type=int)
    parser.add_argument("--kernel_weight", type=int)
    args = parser.parse_args()
    
    bs = 1
    T = 30
    H = 32
    W = 32
    in_channels = 32
    out_channels = 32
    kT = 1
    kH = 1
    kW = 1

    dt = torch.half
    cuda = torch.device('cuda:0')
    tensor_input = torch.randn(bs, in_channels, T, H, W, dtype=dt, device='cuda')
    tensor_weight = torch.randn(out_channels, in_channels, kT, kH, kW, dtype=dt, device='cuda')

    split_conv(tensor_input, tensor_weight)



