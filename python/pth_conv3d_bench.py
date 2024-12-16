import os 
import csv
import timeit

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

def try_conv3d_ndhwc(input, model, *args, **kwargs):
    output = model(input)
    return output

def profile_op(func, *args, **kargs):
    # print(f"args={args}")
    # timeit.timeit(f"{func}({*args})", number=10) 
    output_tensor = func(*args, **kargs)
    # print(f"output_tensor=[{output_tensor.size()}]")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    ncalls = 5
    for i in range(ncalls):
        func(*args, **kargs)
    
    torch.cuda.synchronize()
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event) / ncalls

    return elapsed_time_ms, output_tensor.size()
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", type=int)
    parser.add_argument("-oc", "--out_channels", type=int)
    parser.add_argument("-ic", "--in_channels", type=int)
    parser.add_argument("-f", "--frame", type=int)
    parser.add_argument("-hi", "--height", type=int)
    parser.add_argument("-wi", "--width", type=int)
    parser.add_argument("-kf", "--kernel_frame", type=int)
    parser.add_argument("-kh", "--kernel_height", type=int)
    parser.add_argument("-kw", "--kernel_width", type=int)
    parser.add_argument("-ts", "--temporal_stride", type=int, default=1)
    parser.add_argument("-ss", "--space_stride", type=int, default=1)
    parser.add_argument("-of", "--output_file", default='prof/performance.csv')
    args = parser.parse_args()

    device_name = torch.cuda.get_device_name(0)
    PEAK_TFLOPS = 2048.0 * torch.cuda.get_device_properties(0).multi_processor_count
    if torch.cuda.get_device_properties(0).multi_processor_count == 80:
        freq = 1.41
    elif torch.cuda.get_device_properties(0).multi_processor_count == 304:
        freq = 2.1
    PEAK_TFLOPS *= freq / 1000
    
    bs = args.batch_size
    T = args.frame
    H = args.height
    W = args.width
    in_channels = args.in_channels
    out_channels = args.out_channels
    kT = args.kernel_frame
    kH = args.kernel_height
    kW = args.kernel_width
    TS = args.temporal_stride
    SS = args.space_stride

    dt = torch.bfloat16
    cuda = torch.device('cuda:0')
    tensor_input = torch.randn(bs, in_channels, T, H, W, dtype=dt, device='cuda')
    tensor_weight = torch.randn(out_channels, in_channels, kT, kH, kW, dtype=dt, device='cuda')
    tensor_bias = torch.randn(out_channels, dtype=dt, device='cuda')

    # split_conv(tensor_input, tensor_weight)
    #elapsed_time = profile_op(split_conv, tensor_input, tensor_weight, padding='same')
    elapsed_time, ouput_shape = profile_op(split_conv, tensor_input, tensor_weight, tensor_bias, stride=(TS,SS,SS))
    gflops = ouput_shape[0] * ouput_shape[2] * ouput_shape[3] * ouput_shape[4] * in_channels * out_channels * kT * kH * kW * 2.0 / 1000 / 1000 / 1000
    TFLOPS = gflops / elapsed_time

    # print("ncdhw:")
    print(f"input [bs, ic, F, H, W]=[tensor_input.size()], weight [oc, ic, kF, kH, kW]=[tensor_weight.size()], ouput [bs, oc, F, H, W]=[{ouput_shape}] time={elapsed_time:.3f}ms, TFLOPS={TFLOPS:.3f}TFLOPS")

    if not os.path.exists(args.output_file):
        with open(args.output_file, 'w') as f:
            writer = csv.writer(f)
            row = ['bs', 'o_c', 'i_c', 'i_f', 'i_h', 'i_w', 'k_f', 'k_h', 'k_w', 'o_f', 'o_h', 'o_w', 'gflop', 'time(ms)', 'TFLOPS', 'efficiency']
            writer.writerow(row)

    with open(args.output_file, 'a') as f:
        writer = csv.writer(f)
        row = [bs, out_channels, in_channels, T, H, W, kT, kH, kW, ouput_shape[2], ouput_shape[3], ouput_shape[4], f'{gflops:.3f}', f'{elapsed_time:.3f}', f'{TFLOPS:.3f}', f'{TFLOPS / PEAK_TFLOPS :.3f}']
        writer.writerow(row)

    exit(0)

    print("ndhwc:")
    model = nn.Sequential(nn.Conv3d(out_channels, in_channels, kT)).cuda().bfloat16()
    # model = nn.utils.convert_conv3d_weight_memory_format(model, torch.channels_last_3d)
    tensor_input = tensor_input.to(memory_format=torch.channels_last_3d)
    model = model.to(memory_format=torch.channels_last_3d)
    input_ndhwc = torch.randn(bs, T, H, W, in_channels, dtype=dt, device='cuda')
    elapsed_time = profile_op(try_conv3d_ndhwc, tensor_input, model)
    TFLOPS = gflops / elapsed_time
    print(f"input [bs, ic, F, H, W]=[{bs}, {in_channels}, {T}, {H}, {W}], weight [oc, ic, kF, kH, kW]=[{out_channels}, {in_channels}, {kT}, {kH}, {kW}], time={elapsed_time:.3f}ms, TFLOPS={TFLOPS:.3f}TFLOPS")


