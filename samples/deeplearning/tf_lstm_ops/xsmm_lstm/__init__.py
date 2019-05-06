from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import abc

from tensorflow.contrib.rnn.ops import gen_lstm_ops
from tensorflow.contrib.util import loader
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.platform import resource_loader
from tensorflow.contrib.rnn import LSTMBlockWrapper
from tensorflow.python.keras.utils import tf_utils
import tensorflow as tf
import os

module_dir = os.path.dirname(__file__)
lib_name = os.path.join(module_dir, 'libxsmm_lstm.so')

xsmm_lstm = tf.load_op_library(lib_name)

@tf.RegisterGradient("XsmmLSTMCell")
def _LSTMBlockCellGrad(op, *grad):
  """Gradient for XsmmLSTMCell."""
  w_in_kcck = False
  try:
    w_in_kcck = op.get_attr("w_in_kcck")
  except:
    pass

  if w_in_kcck:
    (x, cs_prev, h_prev, w, wT, wci, wcf, wco, b) = op.inputs
  else:
    (x, cs_prev, h_prev, w, wci, wcf, wco, b) = op.inputs
    wT = w
  (i, cs, f, o, ci, co, _) = op.outputs
  (_, cs_grad, _, _, _, _, h_grad) = grad

  (cs_prev_grad, h_prev_grad, x_grad, w_grad, b_grad, wci_grad, wcf_grad,
   wco_grad) = xsmm_lstm.xsmm_lstm_cell_grad(
       x=x,
       cs_prev=cs_prev,
       h_prev=h_prev,
       w=w,
       w_t=wT,
       wci=wci,
       wcf=wcf,
       wco=wco,
       b=b,
       i=i,
       cs=cs,
       f=f,
       o=o,
       ci=ci,
       co=co,
       cs_grad=cs_grad,
       h_grad=h_grad,
       use_peephole=op.get_attr("use_peephole"),
       w_in_kcck=w_in_kcck)

  if w_in_kcck:
    return (x_grad, cs_prev_grad, h_prev_grad, w_grad, None, wci_grad, wcf_grad,
          wco_grad, b_grad)
  else:
    return (x_grad, cs_prev_grad, h_prev_grad, w_grad, wci_grad, wcf_grad,
          wco_grad, b_grad)


@ops.RegisterGradient("XsmmFusedLSTM")
def _XsmmFusedLSTMGrad(op, *grad):
  """Gradient for XsmmFusedLSTM."""
  seq_len_max, x, cs_prev, h_prev, w, wci, wcf, wco, b = op.inputs
  i, cs, f, o, ci, co, h = op.outputs

  cs_grad = grad[1]
  h_grad = grad[6]

  (x_grad, cs_prev_grad, h_prev_grad, w_grad, wci_grad, wcf_grad, wco_grad,
   b_grad) = xsmm_lstm.xsmm_fused_lstm_grad(
       seq_len_max,
       x,
       cs_prev,
       h_prev,
       w,
       wci,
       wcf,
       wco,
       b,
       i,
       cs,
       f,
       o,
       ci,
       co,
       h,
       cs_grad,
       h_grad,
       use_peephole=op.get_attr("use_peephole"),
       use_residue=op.get_attr("use_residue"),
       use_dropout=op.get_attr("use_dropout"))

  return [
      None, x_grad, cs_prev_grad, h_prev_grad, w_grad, wci_grad, wcf_grad,
      wco_grad, b_grad
  ]

class XsmmFusedLSTM(LSTMBlockWrapper):
  """XsmmFusedLSTM implementation of LSTM.
  This is an extremely efficient LSTM implementation, that uses a single TF op
  for the entire LSTM. It should be both faster and more memory-efficient than
  LSTMBlockCell defined above.
  The implementation is based on: http://arxiv.org/abs/1409.2329.
  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.
  The variable naming is consistent with `rnn_cell_impl.LSTMCell`.
  """

  def __init__(self,
               num_units,
               forget_bias=1.0,
               cell_clip=None,
               use_peephole=False,
               dropout=0.0,
               residual_connection=False,
               reuse=None,
               dtype=None,
               name="lstm_fused_cell"):
    """Initialize the LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      cell_clip: clip the cell to this value. Default is no cell clipping.
      use_peephole: Whether to use peephole connections or not.
      residual_connection: Whether to add residue connections or not.
      dropout: Whether to apply dropout or not.
      reuse: (optional) boolean describing whether to reuse variables in an
        existing scope.  If not `True`, and the existing scope already has the
        given variables, an error is raised.
      dtype: the dtype of variables of this layer.
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.  By default this is "lstm_cell", for variable-name compatibility
        with `tf.nn.rnn_cell.LSTMCell`.
    """
    super(XsmmFusedLSTM, self).__init__(
        _reuse=reuse, name=name, dtype=dtype)
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._cell_clip = cell_clip if cell_clip is not None else -1
    self._use_peephole = use_peephole
    self._residual_connection = residual_connection
    self._dropout = dropout

    # Inputs must be 3-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=3)

  @property
  def num_units(self):
    """Number of units in this cell (output dimension)."""
    return self._num_units

  def build(self, input_shape):
    input_size = input_shape[2].value
    self._kernel = self.add_variable(
        "kernel", [input_size + self._num_units, self._num_units * 4])
    self._bias = self.add_variable(
        "bias", [self._num_units * 4],
        initializer=init_ops.constant_initializer(0.0))
    if self._use_peephole:
      self._w_i_diag = self.add_variable("w_i_diag", [self._num_units])
      self._w_f_diag = self.add_variable("w_f_diag", [self._num_units])
      self._w_o_diag = self.add_variable("w_o_diag", [self._num_units])

    self.built = True

  def _call_cell(self,
                 inputs,
                 initial_cell_state=None,
                 initial_output=None,
                 dtype=None,
                 sequence_length=None):
    """Run this LSTM on inputs, starting from the given state.
    Args:
      inputs: `3-D` tensor with shape `[time_len, batch_size, input_size]`
      initial_cell_state: initial value for cell state, shape `[batch_size,
        self._num_units]`
      initial_output: initial value of cell output, shape `[batch_size,
        self._num_units]`
      dtype: The data type for the initial state and expected output.
      sequence_length: Specifies the length of each sequence in inputs. An
        `int32` or `int64` vector (tensor) size `[batch_size]`, values in `[0,
        time_len)` or None.
    Returns:
      A pair containing:
      - Cell state (cs): A `3-D` tensor of shape `[time_len, batch_size,
                         output_size]`
      - Output (h): A `3-D` tensor of shape `[time_len, batch_size,
                    output_size]`
    """

    inputs_shape = inputs.get_shape().with_rank(3)
    time_len = inputs_shape[0].value
    if time_len is None:
      time_len = array_ops.shape(inputs)[0]

    if self._use_peephole:
      wci = self._w_i_diag
      wco = self._w_o_diag
      wcf = self._w_f_diag
    else:
      wci = wcf = wco = array_ops.zeros([self._num_units], dtype=dtype)

    if sequence_length is None:
      max_seq_len = math_ops.to_int64(time_len)
    else:
      max_seq_len = math_ops.to_int64(math_ops.reduce_max(sequence_length))

    print("  Xsmm LSTM Fused Cell: dropout = %.3f, Resudue = %s" % (self._dropout, self._residual_connection))
    orig_inputs = inputs
    if self._dropout > 0.0:
      inputs = tf.nn.dropout(inputs, 1 - self._dropout)

    '''
    _, cs, _, _, _, _, h = gen_lstm_ops.block_lstm(
        seq_len_max=max_seq_len,
        x=inputs,
        cs_prev=initial_cell_state,
        h_prev=initial_output,
        w=self._kernel,
        wci=wci,
        wcf=wcf,
        wco=wco,
        b=self._bias,
        forget_bias=self._forget_bias,
        cell_clip=self._cell_clip,
        use_peephole=self._use_peephole)
        '''

    _, cs, _, _, _, _, h = xsmm_lstm.xsmm_fused_lstm(
        seq_len_max=max_seq_len,
        x=inputs,
        cs_prev=initial_cell_state,
        h_prev=initial_output,
        w=self._kernel,
        wci=wci,
        wcf=wcf,
        wco=wco,
        b=self._bias,
        forget_bias=self._forget_bias,
        cell_clip=self._cell_clip,
        use_peephole=self._use_peephole,
        use_residue=False,
        use_dropout=False)

    if self._residual_connection:
      with tf.name_scope("fused_residual_connection"):
        h = h + orig_inputs

    return cs, h


class XsmmLSTMCell(rnn_cell_impl.RNNCell):
  """LIbxsmm LSTM Cell"""
  def __init__(self,
               num_units,
               forget_bias=1.0,
               state_is_tuple=True,
               activation=None,
               reuse=None,
               name=None,
               dtype=None,
               w_in_kcck=True,
               **kwargs):
    """Initialize the libxsmm LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
        Must set to `0.0` manually when restoring from CudnnLSTM-trained
        checkpoints.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`. It
        could also be string that is within Keras activation function names.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.
      dtype: Default dtype of the layer (default of `None` means use the type
        of the first input). Required when `build` is called before `call`.
      **kwargs: Dict, keyword named properties for common layer attributes, like
        `trainable` etc when constructing the cell from configs of get_config().
      When restoring from CudnnLSTM-trained checkpoints, must use
      `CudnnCompatibleLSTMCell` instead.
    """
    super(XsmmLSTMCell, self).__init__(
        _reuse=reuse, name=name, dtype=dtype, **kwargs)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._w_in_kcck = w_in_kcck
    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = math_ops.tanh

  @property
  def state_size(self):
    return (rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  @tf_utils.shape_type_conversion
  def build(self, inputs_shape):
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % str(inputs_shape))

    input_depth = inputs_shape[-1]
    h_depth = self._num_units
    C = input_depth + h_depth
    K = 4 * self._num_units
    ctxt = tf.get_default_graph()._get_control_flow_context()
    if ctxt: ctxt = ctxt.GetWhileContext()
    self._kernel = self.add_variable(
        "kernel",
        shape=[input_depth + h_depth, 4 * self._num_units])
    self._bias = self.add_variable(
        "bias",
        shape=[4 * self._num_units],
        initializer=init_ops.zeros_initializer(dtype=self.dtype))

    if self._w_in_kcck:
      if ctxt: ctxt.Exit()
      def block_transpose(inp, C, BC, K, BK):
        inp_packed = tf.reshape(tf.transpose(tf.reshape(inp, [C//BC, BC, K//BK, BK]), perm=[2, 0, 1, 3]), [C, K])
        inp_packed_trans = tf.reshape(tf.transpose(tf.reshape(inp, [C//BC, BC, 4, K//(BK*4), BK]), perm=[2, 0, 3, 4, 1]), [C, K])
        return inp_packed, inp_packed_trans
      with tf.variable_scope("kernel_transpose") as vs:
       with tf.name_scope(""), tf.name_scope(vs.name):
        BC = 64 if input_depth % 64 == 0 else input_depth
        BK = 64 if h_depth % 64 == 0 else h_depth
        W, R = tf.split(self._kernel, [input_depth, h_depth], 0)
        W, WT = block_transpose(W, input_depth, BC, K, BK)
        R, RT = block_transpose(R, h_depth, BK, K, BK)
        self._kernel = tf.concat([W, R], 0)
        self._kernel_trans = tf.concat([WT, RT], 0)
      if ctxt: ctxt.Enter()
    else:
      self._kernel_trans = self._kernel
    self.built = True

  def call(self, inputs, state):
    """Long short-term memory cell (LSTM).
    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: An `LSTMStateTuple` of state tensors, each shaped
        `[batch_size, num_units]`, if `state_is_tuple` has been set to
        `True`.  Otherwise, a `Tensor` shaped
        `[batch_size, 2 * num_units]`.
    Returns:
      A pair containing the new hidden state, and the new state (either a
        `LSTMStateTuple` or a concatenated state, depending on
        `state_is_tuple`).
    """

    if len(state) != 2:
      raise ValueError("Expecting state to be a tuple with length 2.")

    if False: #self._use_peephole:
      wci = self._w_i_diag
      wcf = self._w_f_diag
      wco = self._w_o_diag
    else:
      wci = wcf = wco = array_ops.zeros([self._num_units])

    (cs_prev, h_prev) = state

    (_, cs, _, _, _, _, h) = xsmm_lstm.xsmm_lstm_cell(
        x=inputs,
        cs_prev=cs_prev,
        h_prev=h_prev,
        w=self._kernel,
        w_t=self._kernel_trans,
        wci=wci,
        wcf=wcf,
        wco=wco,
        b=self._bias,
        forget_bias=self._forget_bias,
        cell_clip=-1,
        use_peephole=False,
        w_in_kcck=self._w_in_kcck,
        name=self._name)

    new_state = rnn_cell_impl.LSTMStateTuple(cs, h)
    return h, new_state

  def get_config(self):
    config = {
        "num_units": self._num_units,
        "forget_bias": self._forget_bias,
        "state_is_tuple": self._state_is_tuple,
        "activation": activations.serialize(self._activation),
        "reuse": self._reuse,
    }
    base_config = super(XsmmLSTMCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

