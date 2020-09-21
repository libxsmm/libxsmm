import torch
import pcl_mlp

MB = 128
N = MB
K= 128 #128
C=128 #64
"""

MB = 64
K= 64 #128
C= 64 #64
"""


fc = pcl_mlp.XsmmLinear(C, K, sparse_kernel_mode=True)
#fc = pcl_mlp.XsmmLinear(C, K)
tl = torch.nn.Linear(C, K)

sparsity_rate = 0.9
weight = torch.zeros(K, C, requires_grad=True)

from random import random
# Populate weight matrix
for k in range(K):
    for c in range(C):
        # Creating simple permutation matrix
        # This causes segmentation error
        """
        if K-k-1 == c:
            weight[k, c] = 1.
        """
        # This doesn't cause segmentation error
        if random() > sparsity_rate:
            weight[k, c] = random()
# bias = torch.randn(K, requires_grad=True)
bias = torch.zeros(K, requires_grad=True)
#print("Weight: ", weight)
#print("Bias: ", bias)

"""
x1 = torch.zeros(MB, C, requires_grad=True)
for n in range(N):
    for c in range(C):
        x1[n][c] = (c + n) / 20.
"""

x1 = torch.randn(MB, C, requires_grad=True)
x2 = x1.clone().detach().requires_grad_()

fc.weight.data = weight.clone()
#fc.reset_weight_shape(torch.bfloat16)
tl.weight.data = weight.clone()
fc.bias.data = bias.clone()
tl.bias.data = bias.clone()

y1 = fc(x1)
# y2 = tl(x2.to_mkldnn())
y2 = tl(x2)
#y2 = y2.to_dense()
z1 = y1.mean()
z2 = y2.mean()

print("xsmm: {}".format(z1))
print("ref: {}".format(z2))

if not y1.allclose(y2, rtol=1e-4, atol=1e-4):
    print("forward pass invalid")
    print("ref")
    print(y2)

    print("xsmm")
    print(y1)

z1.backward()
z2.backward()


# Testing input grad
if not x1.grad.allclose(x2.grad, rtol=1e-6, atol=1e-6):
  print("InputGrad:")
  print(x1.grad.size())
  print("xsmm: ", x1.grad)
  print(x2.grad.size())
  print("ref: ", x2.grad)
  #print((x2.grad-x1.grad).sort(descending=True))
  print(x2.grad-x1.grad)

print("xsmm: {}".format(x1.grad.mean()))
print("ref: {}".format(x2.grad.mean()))

# Testing weight grad
if not tl.weight.grad.allclose(fc.weight.grad):
  print("WeightGrad:")
  print(fc.weight.grad.size())
  print("xsmm: ", fc.weight.grad)
  print(tl.weight.grad.size())
  print("ref: ", tl.weight.grad)
  print("Org weight: ", weight)

print("xsmm: {}".format(fc.weight.grad.mean()))
print("ref: {}".format(tl.weight.grad.mean()))

"""

print(fc.weight.grad.size())
print(fc.weight.grad.mean())
print(tl.weight.grad.size())
print(tl.weight.grad.mean())
"""

"""
if not x1.grad.allclose(x2.grad, rtol=1e-4, atol=1e-4):
  print("InputGrad:")
  print(x1.grad.size())
  print("F: ", x1.grad)
  print(x2.grad.size())
  print("T: ", x2.grad)
  print((x2.grad-x1.grad).sort(descending=True))

if not tl.weight.grad.allclose(fc.weight.grad):
  print("WeightGrad:")
  print(fc.weight.grad.size())
  print("F: ", fc.weight.grad)
  print(tl.weight.grad.size())
  print("T: ", tl.weight.grad)
"""

"""
if not tl.bias.grad.allclose(fc.bias.grad):
  print("BiasGrad:")
  print(fc.bias.grad.size())
  print("F: ", fc.bias.grad)
  print(tl.bias.grad.size())
  print("T: ", tl.bias.grad)
# print(x1.grad)
# print(x2.grad)

print("X Allclose: ", x1.grad.allclose(x2.grad, rtol=1e-4, atol=1e-4))
print("Y Allclose: ", tl.bias.grad.allclose(fc.bias.grad))
print("(x1.grad - x2.grad).abs().sum() = ", (x1.grad - x2.grad).abs().sum())
"""
