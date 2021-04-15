import torch
import pcl_mlp
K=32
C=14
MB=32

torch.manual_seed(10)
dtype = torch.bfloat16

fc = pcl_mlp.XsmmLinear(C, K)
tl = torch.nn.Linear(C, K)

fc = fc.to(dtype)

weight = torch.randn(K, C, requires_grad=True)
bias = torch.randn(K, requires_grad=True)
#print("Weight: ", weight)
#print("Bias: ", bias)
x1 = torch.randn(MB, C, requires_grad=True)
x2 = x1.clone().detach().requires_grad_()

fc.weight.data = weight.clone()
fc.reset_weight_shape(dtype)
tl.weight.data = weight.clone()
fc.bias.data = bias.clone()
tl.bias.data = bias.clone()

y1 = fc(x1.to(dtype))
#y1 = fc(x1)
#y2 = tl(x2.to(dtype))
y2 = tl(x2)
#y2 = y2.to_dense()
z1 = y1.mean()
z2 = y2.mean()

print("z1", z1)
print("z2", z2)
z1.backward()
z2.backward()

print("x1.grad.mean", x1.grad.mean())
print("x2.grad.mean", x2.grad.mean())

print("fc.wt.grad.mean", fc.weight.grad.mean())
print("tl.wt.grad.mean", tl.weight.grad.mean())

print("fc.bias.grad.mean", fc.bias.grad.mean())
print("tl.bias.grad.mean", tl.bias.grad.mean())

