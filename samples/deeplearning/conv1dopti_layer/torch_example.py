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
Batch_size = 64                  # Batch size (64)
Input_width = 60000              # Width of the input signal track (60000)
Channels = 15                    # Number of channels in the input (15)
Filters = 15                     # Number of filter in the layer (15)
Dilation = 8                     # Amount of dilation (8)
Kernel_size = 51                 # Size of each filter (51)
enable_BF16 = False              # Enable layer compute in BFloat16 (Only works when Filters and channels are both even numbers)

class ZeroSamePad1d(nn.Module):
    """Apply SAME zero padding to input."""

    def __init__(self, interval_size, kernel_size, stride, dilation):
        """Initialize layer.

        Args:
            interval_size : Genome interval size.
            kernel_size : Size of filter.
            stride : Stride for filter.
            dilation : Filter dilation.

        """
        super(ZeroSamePad1d, self).__init__()

        required_total_padding = ZeroSamePad1d._get_total_same_padding(
            interval_size, kernel_size, stride, dilation)
        padding_left = required_total_padding // 2
        padding_right = required_total_padding - padding_left
        self.pad = nn.ConstantPad1d((padding_left, padding_right), 0)

    @staticmethod
    def _get_total_same_padding(interval_size, kernel_size, stride, dilation):
        """Calculate total required padding.

        Args:
            interval_size : Genome interval size.
            kernel_size : Size of filter.
            stride : Stride for filter.
            dilation : Filter dilation.

        Return:
            Total padding required around the input for SAME padding.

        """
        effective_kernel_size = (kernel_size - 1) * dilation + 1
        required_total_padding = (interval_size - 1) * \
            stride + effective_kernel_size - interval_size
        return required_total_padding

    def forward(self, x):
        """Execute layer on input.

        Args:
            x : Input data.

        """
        return self.pad(x)


class Net1(nn.Module):                      # First network containing inbuilt PyTorch layer
    def __init__(self):
        super(Net1, self).__init__()
        self.padding_layer = ZeroSamePad1d(Input_width, Kernel_size, 1, Dilation)
        self.conv1 = nn.Conv1d(in_channels=Filters, out_channels=Channels, kernel_size=Kernel_size, \
                                stride=1, padding=0, dilation=Dilation, bias=False)                # PyTorch Convolutional layer

    def forward(self, x):
        x = self.padding_layer(x)           # Explicit padding
        x = self.conv1(x)
        # x = F.relu(x)                       # If applying relu
        return x


class Net2(nn.Module):                      # Second network containing our optimized layer
    def __init__(self):
        super(Net2, self).__init__()
        self.padding_layer = ZeroSamePad1d(Input_width, Kernel_size, 1, Dilation)
        self.conv2 = Conv1dOpti(in_channels=Filters, out_channels=Channels, kernel_size=Kernel_size, \
                                stride=1, padding=0, dilation=Dilation, bias=False, enable_BF16=enable_BF16)                # Optimized convolutional layer

    def forward(self, x):
        x = self.padding_layer(x)           # Explicit padding needed for our optimzed convolutional layer
        x = self.conv2(x)
        # x = ReLU_bf16.apply(x)              # If applying BF16 relu
        return x


net1 = Net1()                   # Initilize neural networks
net2 = Net2()

torch.manual_seed(11)           # Fixed Random Seed for comparison

X = torch.randn(Batch_size, Channels, Input_width, requires_grad=True)               # Random Input (Batch_size, channel, width)
random_weights = torch.randn(Filters, Channels, Kernel_size)                 # Random weights

###------------------------------------- Accuracy check part -----------------------------------###

net1.conv1.weight.data = random_weights                         # Assign random weights to the layer
net2.conv2.weight.data = random_weights

Y1 = net1.forward(X)
Y1.sum().backward()
for p in net1.parameters():
    wgrad1 = p.grad

Y2 = net2.forward(X)
Y2.sum().backward()
for p in net2.parameters():
    wgrad2 = p.grad

r = wgrad1.max() - wgrad1.min()
print("Backward weight check: ",((torch.abs(wgrad1 - wgrad2)/r < 0.00001).sum() == Filters*Channels*Kernel_size).item())

Y1 = net1.forward(X)
Y2 = net2.forward(X)
r = Y1.max() - Y1.min()
print("    Foward pass check: ", ((torch.abs(Y1 - Y2)/r < 0.00001).sum() == Batch_size*Filters*Input_width).item())

dgrad1 = torch.autograd.grad(Y1.sum(),X)
dgrad2 = torch.autograd.grad(Y2.sum(),X)
r = dgrad1[0].max() - dgrad2[0].min()
print("  Backward data check: ", ((torch.abs(dgrad1[0] - dgrad2[0])/r < 0.00001).sum() == Batch_size*Channels*Input_width).item())


###------------------------------------- Timing check part -----------------------------------###

forward1 = 0                    # variables to store time values
forward2 = 0
backward1 = 0
backward2 = 0

N = 20                                      # Number of iterations
for _ in range(N):                          # MKLDNN PyTorch layer Forward and Backward pass timing
    start = time.time()
    Y1 = net1.forward(X)
    forward1 += time.time() - start

    start = time.time()
    Y1.sum().backward()
    backward1 += time.time() - start

if enable_BF16 == True:
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
