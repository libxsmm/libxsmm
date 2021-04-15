import torch
from torch import nn
from torch.autograd import Function

import pcl_embedding_bag_cpp
from pcl_embedding_bag_cpp import bf16_update
torch_embedding_bag = torch.embedding_bag
torch_tensor_add_ = torch.Tensor.add_

def pcl_embedding_bag(weight, input, offsets, scale_grad_by_freq, mode_enum, sparse, per_sample_weights, include_last_offset=False):
  if sparse and mode_enum == 0 and per_sample_weights is None and scale_grad_by_freq == False and weight.device == torch.device('cpu'): # and weight.dtype == torch.float32:
    ret = PclEmbeddingBagFunction.apply(weight, input.contiguous(), offsets.contiguous())
    ret = (ret, None, None, None)
  else:
    ret = torch_embedding_bag(weight, input, offsets, scale_grad_by_freq, mode_enum, sparse, per_sample_weights, include_last_offset)

  return ret

def pcl_dense_sparse_add(self, *args, **kwargs):
  alpha = 1
  other = None
  if len(args) > 0:
    if isinstance(args[0], torch.Tensor):
      other = args[0]
      if 'alpha' in kwargs.keys(): alpha = kwargs['alpha']
    else:
      alpha = args[0]
      if len(args) > 1: other = args[1]

  assert(other is not None)
  if not self.is_sparse and other.is_sparse and self.device == torch.device('cpu'): # and self.dtype == torch.float32:
    pcl_embedding_bag_cpp.dense_sparse_add(self, other, alpha)
  else:
    torch_tensor_add_(self, other, alpha=alpha)

def bdot(input):
    #print("In pcl_dot: dtype: %s, sizes = %s" % (input.dtype, input.size()))
    return BDotFunc.apply(input)

torch.embedding_bag = pcl_embedding_bag
torch.Tensor.add_ = pcl_dense_sparse_add
print("Using PCL EmbeddingBag Implementation")

class PclEmbeddingBagFunction(Function):
    @staticmethod
    def forward(ctx, weight, input, offsets):
        ctx.save_for_backward(weight, input, offsets)
        output = pcl_embedding_bag_cpp.forward(weight, input, offsets)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        weight, input, offsets = ctx.saved_tensors
        grad_weight = grad_input = grad_offsets = None
        grad_weight = pcl_embedding_bag_cpp.backward(grad_out, weight, input, offsets)
        return (grad_weight, grad_input, grad_offsets)

class BDotFunc(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = pcl_embedding_bag_cpp.bdot_forward(input)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        (input,) = ctx.saved_tensors
        grad_inp = pcl_embedding_bag_cpp.bdot_backward(grad_out, input)
        return grad_inp

