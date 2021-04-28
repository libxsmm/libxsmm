import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Function
import mlpcell_ext

#torch.autograd.set_detect_anomaly(True)

mlpcell_ext.init_libxsmm()
mlpcell_ext.set_rnd_seed(727)

#TODO: Add data type to the handle
class MLPCellHandle:
    def __init__(self, N, C, K, bn, bc, bk, data_type, bias, skip, activation, norm,  p, train):
        self.handle = mlpcell_ext.create_handle(N, C, K, bn, bc, bk, bias, skip, activation, norm, p, train, data_type)
        #print(f"Handle = {self.handle}")
        self.N = N
        self.C = C
        self.K = K
        self.bn = bn
        self.bc = bc
        self.bk = bk
        self.bias = bias
        self.skip = skip
        self.activation = activation
        self.norm = norm
        self.p = p
        self.train = train
        self.data_type = data_type

    def __del__(self):
        if self.handle:
            mlpcell_ext.destroy_handle(self.handle, self.data_type)
            self.handle = None

class XsmmMLPCellFunction(Function):
    @staticmethod
    def forward(ctx, handle, data_type, *inputs):
        # print("FWD Called")
        output, relumask_b, relumask_r, dropout_mask_b, dropout_mask_r = mlpcell_ext.mlpcell_forward(handle.handle, inputs, data_type)
        input_l, input_r, wt_l, wt_r, bias_l, bias_r = inputs

        ctx.save_for_backward(input_l, input_r, wt_l, wt_r, relumask_b, relumask_r, dropout_mask_b, dropout_mask_r)
        ctx.backward_cache = data_type
        ctx.handle = handle
        return output

    @staticmethod
    def backward(ctx, *grad_outs):
        #print("BWD Called")
        # print((grad_outs[0]))

        inputs = []
        inputs.append(grad_outs[0])
        input_l, input_r, wt_l, wt_r, relumask_b, relumask_r, dropout_mask_b, dropout_mask_r = ctx.saved_tensors
        data_type = ctx.backward_cache


        inputs.append(input_l)
        inputs.append(input_r)
        inputs.append(wt_l)
        inputs.append(wt_r)
        inputs.append(relumask_b)
        inputs.append(relumask_r)
        inputs.append(dropout_mask_b)
        inputs.append(dropout_mask_r)

        handle = ctx.handle
        for i in range(len(inputs)):
          if i==2 and inputs[i] is None:
            inputs[i] = torch.tensor([],dtype=input_l.dtype)
          elif i==4 and inputs[i] is None:
            inputs[i] = torch.tensor([],dtype=wt_l.dtype)
          elif (i==5 or i==6) and inputs[i] is None:
            inputs[i] = torch.tensor([],dtype=torch.uint8)
          elif (i==7 or i==8) and inputs[i] is None:
            inputs[i] = torch.tensor([],dtype=torch.uint8)

        grad_input_l, grad_input_r, grad_wt_l, grad_wt_r, grad_bias_l, grad_bias_r = mlpcell_ext.mlpcell_backward(handle.handle, inputs, data_type)

        return (None, None, grad_input_l, grad_input_r, grad_wt_l, grad_wt_r, grad_bias_l, grad_bias_r)

class XsmmMLPCell(nn.Module):
  def __init__(self, C, K, data_type="f32", bias=True, activation=0, norm=0, skip=False, p=0.5, output_stays_blocked=False):
      super(XsmmMLPCell, self).__init__()
      self.N = 0
      self.padded_NR = 0
      self.C = C
      self.padded_C = C
      self.C_pad = 0
      self.K = K
      self.bias = bias
      self.skip = skip
      self.activation = activation
      self.norm = norm
      self.p = p
      self.nbk = 0
      self.bk = 0
      self.nbc = 0
      self.bc = 0
      self.def_N_block_factor = 64
      self.output_stays_blocked = output_stays_blocked
      self.data_type = data_type
      self.weight_l = Parameter(torch.Tensor(K, C))

      if bias:
        self.bias_l = Parameter(torch.Tensor(K))
      else:
        self.bias_l = torch.Tensor()
      if skip:
        self.weight_r = Parameter(torch.Tensor(K, C))
        if bias:
          self.bias_r = Parameter(torch.Tensor(K))
        else:
          self.bias_r = torch.Tensor()
      else:
        self.weight_r = torch.Tensor()
        self.bias_r = torch.Tensor()

      # print("XsmmMLPCell")
      self.reset_parameters()

  def reset_parameters(self):
      init.kaiming_uniform_(self.weight_l, a=math.sqrt(5))
      if self.skip > 0:
        init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))
      if self.bias:
        bound = 1 / math.sqrt(self.C)
        init.uniform_(self.bias_l, -bound, bound)

        if self.skip:
          init.uniform_(self.bias_r, -bound, bound)

  def get_blocking_factor(self, dim_size, default_blocking=None):
      blocking_prio_list = [64, 48, 32, 50]
      if default_blocking:
        if not default_blocking in blocking_prio_list:
          blocking_prio_list = [default_blocking] + blocking_prio_list
      for bs in blocking_prio_list:
        if dim_size % bs == 0:
          #print("Returning block size of %d for dim_size of %d" % ( bs, dim_size))
          return bs
      #print("Returning block size of %d for dim_size of %d" % ( dim_size, dim_size))
      return dim_size

  def is_dtype_supported(self, dtype):
      if dtype == torch.float32:
        return True
      elif dtype == torch.bfloat16 and self.C % 2 == 0:
        return True
      else:
        return False

  def maybe_pad_input(self, input, Npads):
      if input.dim() == 2 and input.size(0) != self.padded_NR:
        input = torch.cat([input, input.new_zeros([Npads, input.size(1)])], dim=0)
      return input

  def maybe_pad_weight(self, weight):
      if weight.dim() == 2 and weight.size(1) != self.padded_C:
        weight = torch.cat([weight, weight.new_zeros([self.K, self.C_pad])], dim=1)
      return weight

  def update_blocking(self, dtype):
      if dtype == torch.bfloat16 and self.padded_C % 2 != 0:
        self.C_pad = 1
        self.padded_C = self.C + self.C_pad
      self.bc = self.get_blocking_factor(self.padded_C, self.default_blocking)
      if dtype == torch.bfloat16 and self.bc % 2 != 0: self.bc *= 2
      self.nbc = self.padded_C // self.bc
      self.bk = self.get_blocking_factor(self.K, self.default_blocking)
      self.nbk = self.K // self.bk

  def forward(self, input_l, input_r=None):
      input_l = input_l.contiguous()
      N = input_l.size(0)

      if input_r == None:
        input_r = torch.Tensor()
        weight_r = torch.Tensor()
        bias_r = torch.Tensor()
      else:
        input_r = input_r.contiguous()
        assert N == input_r.size(0)
      # print ("inputs: ", input_l.shape)
      bn = self.get_blocking_factor(N, self.def_N_block_factor)

      if bn == N and N > self.def_N_block_factor:
        bn = self.def_N_block_factor

      if self.nbc == 0 and self.bc == 0:
        self.nbc = 1
        self.bc = self.C

      if self.nbk == 0 and self.bk == 0:
        self.nbk = 1
        self.bk = self.K

      self.handle = MLPCellHandle(N, self.nbc, self.nbk, bn, self.bc, self.bk, self.data_type, self.bias, self.skip, self.activation, self.norm, self.p, self.training)
      # print("Created handle: ", N, self.nbc, self.nbc, bn, self.bc, self.bk, self.bias, self.skip, self.activation, self.norm, self.p, self.training, self.data_type)
      inputs = [input_l]#.to(torch.bfloat16).contiguous()]
      inputs.append(input_r)#.to(torch.bfloat16).contiguous())
      '''
      if input_l.dtype == torch.bfloat16:
        self.weight_l = Parameter(self.weight_l.to(torch.bfloat16))
        self.weight_r = Parameter(self.weight_r.to(torch.bfloat16))
        self.bias_l = Parameter(self.bias_l.to(torch.bfloat16))
        self.bias_r = Parameter(self.bias_r.to(torch.bfloat16))
      '''
      inputs.append(self.weight_l)
      inputs.append(self.weight_r)
      inputs.append(self.bias_l)
      inputs.append(self.bias_r)

      if input_l.dtype == torch.bfloat16:
        self.data_type = "bf16"
      output = XsmmMLPCellFunction.apply(self.handle, self.data_type, *inputs)

      return output

class DropoutFn(Function):
  @staticmethod
  def forward(ctx, input, p, train, data_type):
    output, mask = mlpcell_ext.dropout_forward(input, p, train, data_type)
    # print("DP-----Fwd")
    ctx.save_for_backward(mask)
    ctx.backward_cache = data_type
    ctx.p = p

    return output

  @staticmethod
  def backward(ctx, grad_output):

    mask, = ctx.saved_tensors
    data_type = ctx.backward_cache
    p = ctx.p

    grad_input = mlpcell_ext.dropout_backward(grad_output, mask, p, data_type)
    return (grad_input, None, None, None)

class Dropout(nn.Module):
  __constants__ = ['p', 'inplace']

  def __init__(self, data_type, p: float = 0.5, inplace: bool = False):
    super(Dropout, self).__init__()
    if p < 0 or p > 1:
       raise ValueError("dropout probability has to be between 0 and 1, "
                                                       "but got {}".format(p))
    self.p = p
    self.inplace = inplace
    # self.data_type = data_type

  def forward(self, input):
    input = input.contiguous()
    if input.dtype == torch.float32:
      data_type = "f32"
    else:
      data_type = "bf16"
    output = DropoutFn.apply(input, self.p, self.training, data_type)

    return output
