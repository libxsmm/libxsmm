
import Conv1dOpti_cpp                       # import the CPP extension module

import torch
import math

import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Function
from torch.nn.modules.utils import _single


class _ConvNd(Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

class ReLU_bf16(Function):
    @staticmethod
    def forward(ctx, input):                        # Forward pass method
        input = input.contiguous()
        result = Conv1dOpti_cpp.relu_forward_bf16(input)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad):                        # Backward pass method
        r = ctx.saved_tensors
        output = r[0]
        d_input = Conv1dOpti_cpp.relu_backward_bf16(grad.contiguous(), output.contiguous())
        return d_input


class Conv1dOptiFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, dilation, enable_BF16):
        input = input.contiguous()
        if (enable_BF16==False):            # Run the FP32 version
            weight = weight.contiguous()
            result = Conv1dOpti_cpp.forward(input, \
                                        weight, \
                                        dilation[0])
        else:
            weight = weight.to(torch.bfloat16).contiguous()             # Run the BF16 version
            result = Conv1dOpti_cpp.forward_bf16(input, \
                                            weight, \
                                            dilation[0])

        ctx.backward_cache = (dilation,enable_BF16)
        ctx.save_for_backward(input, weight)

        return result

    @staticmethod
    def backward(ctx, grad):
        dilation = ctx.backward_cache[0]
        enable_BF16 = ctx.backward_cache[1]
        r = ctx.saved_tensors
        inp= r[0]
        weight = r[1]

        if (enable_BF16==False):                # Run the FP32 version
            d_input, d_weight = Conv1dOpti_cpp.backward(grad.contiguous(), \
                                                    inp, \
                                                    weight, \
                                                    dilation[0])
            return d_input, d_weight, None, None

        else:                                                               # Run the BF16 version
            if grad.dtype == torch.bfloat16:                                # Check if gradiant is in BF16 format
                d_input, d_weight = Conv1dOpti_cpp.backward_bf16(grad.contiguous(), \
                                                        inp, \
                                                        weight, \
                                                        dilation[0])
            else:
                d_input, d_weight = Conv1dOpti_cpp.backward_bf16(grad.to(torch.bfloat16).contiguous(), \
                                                        inp, \
                                                        weight, \
                                                        dilation[0])
            return d_input, d_weight, None, None

class Conv1dOpti(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', enable_BF16=False):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        self.enable_BF16 = enable_BF16
        super(Conv1dOpti, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, padding_mode)


    def forward(self, input):
        # Conditions to enable BF16 compute. Width should be sufficiently greater than padding amount
        # and Input width, filters and channels should all be even numbers
        self.enable_BF16 = (self.enable_BF16) and \
                            (input.size(2) >= (2*(self.weight.size(2) - 1)*self.dilation[0] + 96 - 1)) and \
                            (input.size(2)%2 == 0) and (self.weight.size(0)%2 == 0) and (self.weight.size(1)%2 == 0)

        if (self.enable_BF16==False):          # Convert to run the FP32 version
            if input.dtype != torch.float32:
                input = input.to(torch.float32)
        else:                                                                   # Convert to run the BF16 version
            if input.dtype != torch.bfloat16:
                input = input.to(torch.bfloat16)

        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            self.weight, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)

        elif (self.stride[0] != 1)  or (self.padding[0] != 0):
            return F.conv1d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        elif (self.bias is None):
            # if self.weight.size(1) == 1:    # For the first layer
            #     return Conv1dOptiFunction.apply(input, self.weight, self.dilation, self.enable_BF16).to(torch.bfloat16)
            # else:
                return Conv1dOptiFunction.apply(input, self.weight, self.dilation, self.enable_BF16)

        if input.dtype != torch.bfloat16:                   # Run the FP32 version
            # if self.weight.size(1) == 1:                  # For the first layer
            #     return Conv1dOptiFunction.apply(input, self.weight, self.dilation, self.enable_BF16).to(torch.bfloat16) + self.bias.view([1,-1,1]).to(torch.bfloat16)
            # else:
                return Conv1dOptiFunction.apply(input, self.weight, self.dilation, self.enable_BF16) + self.bias.view([1,-1,1])
        else:                                               # Run the BF16 version
            return Conv1dOptiFunction.apply(input, self.weight, self.dilation, self.enable_BF16) + self.bias.view([1,-1,1]).to(torch.bfloat16)
