import time
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from Conv1dOpti_ext import Conv1dOpti, ReLU_bf16                    # Import Layer from the extension


"""
Set parameters here for testing the convolutional layer. By default layer run in single-precsion (FP32) format

To run code in BFloat16 set enable_BF16 flag to True. BFloat16 code runs only when parameters of
Input width, number of filters and input channels to the layer are even number.
Ex. -  Input_width = 60000, Filters = 16, Channels = 16, enable_BF16 = True  ------ BF16 run

If any of the previous parameters is an odd number than code runs in FP32 format.


Keep batch size as multiple of CPU (Ex. - 28, 56, 84, 128 .... on a 28 core cascade lake) for optimal
performance with the Conv1dOpti layer. Each batch will run on a seperate thread thus performance
may go down if some core are not free, or batch size is not equal to the number of free cores.
Keep the batch size as power of 2 with the MKLDNN backend (Conv1d) for optimal performance.

"""

Input_width = 60400              # Width of the input signal track (60400)
Channels = 16                    # Number of channels in the input (15)
Filters = 16                     # Number of filter in the layer (15)
Dilation = 8                     # Amount of dilation (8)
Kernel_size = 51                 # Size of each filter (51)
enable_BF16 = False              # Enable layer compute in BFloat16 (Only works when Filters and channels are both even numbers)


class Net1(nn.Module):                      # First network containing inbuilt PyTorch layer
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=Filters, out_channels=Channels, kernel_size=Kernel_size, \
                                stride=1, padding=0, dilation=Dilation, bias=False)                # PyTorch Convolutional layer

    def forward(self, x):
        x = self.conv1(x)
        # x = F.relu(x)                       # If applying relu
        return x


class Net2(nn.Module):                      # Second network containing our optimized layer
    def __init__(self):
        super(Net2, self).__init__()
        self.conv2 = Conv1dOpti(in_channels=Filters, out_channels=Channels, kernel_size=Kernel_size, \
                                stride=1, padding=0, dilation=Dilation, bias=False, enable_BF16=enable_BF16)                # Optimized convolutional layer

    def forward(self, x):
        x = self.conv2(x)
        # x = ReLU_bf16.apply(x)              # If applying BF16 relu
        return x


net1 = Net1()                   # Initilize neural networks
net2 = Net2()

torch.manual_seed(11)           # Fixed Random Seed for comparison

random_weights = torch.randn(Filters, Channels, Kernel_size)                 # Random weights
net1.conv1.weight.data = random_weights                         # Assign random weights to the layer
net2.conv2.weight.data = random_weights

###------------------------------------- Timing check part -----------------------------------###

forward1 = 0                    # variables to store time values
forward2 = 0
backward1 = 0
backward2 = 0


Batch_size_1 = 64                  # Batch size for oneDNN (64)
X = torch.randn(Batch_size_1, Channels, Input_width, requires_grad=True)               # Random Input (Batch_size, channel, width)

N = 20                                      # Number of iterations
for _ in range(N):                          # MKLDNN PyTorch layer Forward and Backward pass timing
    start = time.time()
    Y1 = net1.forward(X)
    forward1 += time.time() - start

    start = time.time()
    Y1.sum().backward()
    backward1 += time.time() - start


Batch_size_2 = 56                             # Multiple of core count for optimized layer
X = torch.randn(Batch_size_2, Channels, Input_width, requires_grad=True)               # Random Input (Batch_size, channel, width)

if enable_BF16 == True:                     # if BFloat16 computation is enabled
    X = X.to(torch.bfloat16)

for _ in range(N):                          # Optimized PyTorch layer Forward and Backward pass timing
    start = time.time()
    Y2 = net2.forward(X)
    forward2 += time.time() - start

    start = time.time()
    Y2.sum().backward()
    backward2 += time.time() - start

print('Forward pass time (PyTorch layer): {:.3f} ms | Forward pass time (Optimized layer): {:.3f} ms'.format(forward1 * 1e3/N, forward2 * 1e3/N))
print('Backward pass time (PyTorch layer): {:.3f} ms | Backward pass time (Optimized layer): {:.3f} ms'.format(backward1 * 1e3/N, backward2 * 1e3/N))

forward1_flops = 2*Batch_size_1*Channels*Filters*Kernel_size*(Input_width - (Kernel_size - 1)*Dilation)/(forward1 / N)
backward1_flops = 2*2*Batch_size_1*Channels*Filters*Kernel_size*(Input_width - (Kernel_size - 1)*Dilation)/(backward1 / N)

forward2_flops = 2*Batch_size_2*Channels*Filters*Kernel_size*(Input_width - (Kernel_size - 1)*Dilation)/(forward2 / N)
backward2_flops = 2*2*Batch_size_2*Channels*Filters*Kernel_size*(Input_width - (Kernel_size - 1)*Dilation)/(backward2 / N)



print("\n")
print('Forward pass flops (PyTorch layer): {:e} | Forward pass flops (Optimized layer): {:e} '.format(forward1_flops, forward2_flops))
print('Backward pass flops (PyTorch layer): {:e} | Backward pass flops (Optimized layer): {:e} '.format(backward1_flops, backward2_flops))
