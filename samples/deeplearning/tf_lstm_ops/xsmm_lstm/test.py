import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import xsmm_lstm
import sys
from os import isatty

GREEN=''
RED=''
BOLD=''
ENDC=''
if isatty(sys.stdout.fileno()):
  GREEN='\033[92m'
  RED ='\033[91m'
  BOLD='\033[1m'
  ENDC='\033[0m'

def isclose(buf, ref, xmm):
  avg_ref = np.mean(ref)
  avg_abs_ref_orig = np.mean(np.absolute(ref))
  avg_abs_ref = avg_abs_ref_orig if avg_abs_ref_orig != 0 else 0.1
  avg_xmm = np.mean(xmm)
  avg_abs_xmm = np.mean(np.absolute(xmm))
  if avg_abs_ref_orig == avg_abs_xmm == 0: return
  size = ref.size
  it = np.nditer([ref, xmm], flags=['multi_index'])
  count = 0
  print_count = 0
  max_print = 5
  print_always = 1
  for x, y in it:
    rdiff = abs(x - y) / avg_abs_ref
    diff = abs((x - y) / x) if x != 0 else rdiff
    if (diff > 1e-5 and rdiff > 1e-5) or print_count < print_always:
      if print_count < max_print: print("  %-10s %-10s: ref: %10s  xmm: %10s  diff: %9e" % (buf, it.multi_index, x, y, diff))
      if diff > 1e-5: count += 1
      print_count += 1
  if count > 0:
    print("%s %sdoes NOT match%s, error count = %d (out of %d) AVG=%g ABSAVG=%g" % (buf, RED+BOLD, ENDC, count, size, avg_ref, avg_abs_ref_orig))
  else:
    print("%s %sDOES match%s, size = %d AVG=%g ABSAVG=%g" % (buf, GREEN+BOLD, ENDC, size, avg_ref, avg_abs_ref_orig))

N=64
C=128
K=192
T=10
forget_bias=1.0
tf.set_random_seed(1)
#x = tf.constant(-0.1, shape=[N,C], dtype = tf.float32)
#x2 = tf.constant(0.1, shape=[N,C], dtype = tf.float32)
x = tf.random_normal(shape=[N,C], dtype = tf.float32) #+ 0.5
x2 = tf.random_normal(shape=[N,C], dtype = tf.float32) #+ 0.5
lstm_cell_ref = rnn.LSTMBlockCell(K, forget_bias=forget_bias, name='test')
#lstm_cell_ref = rnn.BasicLSTMCell(K, forget_bias=forget_bias, name='test')
#lstm_cell = rnn.LSTMBlockCell(K, forget_bias=forget_bias, name='test', reuse=True)
lstm_cell = xsmm_lstm.XsmmLSTMCell(K, forget_bias=forget_bias, name='test', reuse=True)
init_state = lstm_cell_ref.zero_state(N, dtype=tf.float32)
x_fused = tf.convert_to_tensor([x] + [x2 for _ in range(T-1)])
print("x_fused is: %s" % x_fused)
outputs_ref, states_ref = tf.nn.dynamic_rnn(lstm_cell_ref, x_fused, dtype=tf.float32, initial_state=init_state, time_major=True)
outputs, states = tf.nn.dynamic_rnn(lstm_cell, x_fused, dtype=tf.float32, initial_state=init_state, time_major=True)

init = tf.global_variables_initializer()
W = tf.global_variables()[0]
B = tf.global_variables()[1]

g_ref = tf.gradients(outputs_ref, [x_fused] + [W, B, init_state.c, init_state.h])
g = tf.gradients(outputs, [x_fused] + [W, B, init_state.c, init_state.h])
g_names = ["dx_fused"] + ["dW", "dB", "dcsp", "dhp"]
#print(tf.get_default_graph().as_graph_def())

with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1)) as sess:
    sess.run(init)
    g_print, g_print_ref = sess.run([g,g_ref])

    for t,t_ref, p, p_ref, name in zip(g, g_ref, g_print, g_print_ref, g_names):
      if t.name != t_ref.name: isclose("TEST: %-4s " % name + t.name, p_ref, p)
