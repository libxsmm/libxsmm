#!/usr/bin/env python3
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/hfp/libxsmm/                        #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Anand Venkat (Intel Corp.)
###############################################################################

import logging
import sys
import numpy as np
import tvm
import topi
import time
from topi.util import get_const_tuple
import math
import topi.testing
import xlwt
import argparse

import os
import ctypes
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

parser = argparse.ArgumentParser()
parser.add_argument("-d", nargs=1, type=str, default=["resnet3"])
args = parser.parse_args()
layer = args.d[0]

#Resnet-50  layers (excluding first layer)
_resnet_layers ={
    'resnet2':[1,256,64,56,56,1,1,0],
    'resnet3':[1,64,64,56,56,1,1,0],
    'resnet4':[1,64,64,56,56,3,1,1],
    'resnet5':[1,64,256,56,56,1,1,0],
    'resnet6':[1,512,256,56,56,1,2,0],
    'resnet7':[1,128,256,56,56,1,2,0],
    'resnet8':[1,128,128,28,28,3,1,1],
    'resnet9':[1,512,128,28,28,1,1,0],
    'resnet10':[1,128,512,28,28,1,1,0],
    'resnet11':[1,1024,512,28,28,1,2,0],
    'resnet12':[1,256,512,28,28,1,2,0],
    'resnet13':[1,256,256,14,14,3,1,1],
    'resnet14':[1,1024,256,14,14,1,1,0],
    'resnet15':[1,256,1024,14,14,1,1,0],
    'resnet16':[1,2048,1024,14,14,1,2,0],
    'resnet17':[1,512,1024,14,14,1,2,0],
    'resnet18':[1,512,512,7,7,3,1,1],
    'resnet19':[1,2048,512,7,7,1,1,0],
    'resnet20':[1,512,2048,7,7,1,1,0]
}

'''
Convert input from NCHW format to NCHW16C format where the innermost data dimension is vectorized for AVX-512
'''
def convert_input(a_np, batch, in_channel,input_height,input_width,pad_height,pad_width,vlen,A):
    to_return = np.zeros((batch, math.ceil(in_channel/vlen),input_height + 2*pad_height, input_width+ 2*pad_width,vlen),dtype = A.dtype)

    for i in range(batch):
      for j in range(math.ceil(in_channel/vlen)):
        for k in range(input_height + 2*pad_height):
          for l in range(input_width + 2*pad_width):
              for m in range(vlen):
                if k < pad_height or k >= input_height + pad_height or l < pad_width or l >= input_width+ pad_width or j*vlen + m >= in_channel:
                      to_return[i,j,k,l,m] = float(0)
                else:
                      to_return[i,j,k,l,m] = a_np[i,j*vlen + m,k-pad_height,l-pad_width]

    return to_return
'''
Convert output from NCHW format to NCHW16C format where the innermost data dimension is vectorized for AVX-512
'''

def convert_output(a_np, batch, out_channel,output_height,output_width,vlen):
    to_return = np.zeros((batch, out_channel,output_height, output_width), dtype = float)
    for i in range(batch):
      for j in range(math.ceil(out_channel/vlen)):
        for k in range(output_height):
          for l in range(output_width):
              for m in range(vlen):
                  to_return[i,j*vlen + m,k,l] = a_np[i,j,k,l,m]



    return to_return

'''
Convert weights from KCRS format to KCRS16C16K format where the innermost data dimension is vectorized for AVX-512
'''

def convert_weight(w_np, in_channel, out_channel, kernel_height, kernel_width, vlen,W):
    to_return = np.zeros((math.ceil(out_channel/vlen), math.ceil(in_channel/vlen),kernel_height, kernel_width,vlen,vlen), dtype = W.dtype)

    for i in range(math.ceil(out_channel/vlen)):
      for j in range(math.ceil(in_channel/vlen)):
        for k in range(kernel_height):
          for l in range(kernel_width):
            for m in range(vlen):
              for n in range(vlen):
                if i*vlen + n >= out_channel or j*vlen + m >= in_channel:
                   to_return[i,j,k,l,m,n] =float(0)
                else:
                   to_return[i,j,k,l,m,n] = w_np[i*vlen + n,j*vlen+ m,k,l]



    return to_return


# Get the reference output tensor for correctness check
def get_ref_data(batch,out_channel,in_channel,input_height,input_width,kernel_height,kernel_width,stride_height,padding):
            a_np = np.random.uniform(size=(batch,in_channel,input_height,input_width)).astype(float)
            w_np = np.random.uniform(size=(out_channel,in_channel,kernel_height,kernel_width)).astype(float)
            if batch == 1:
                b_np = topi.testing.conv2d_nchw_python(a_np, w_np, stride_height, padding)
            #b_np =  topi.nn.conv2d_NCHWc(a_np, w_np,out_channel,kernel_height,stride_height,
            #     padding, layout="NCHWc", out_layout="NCHWc", out_dtype='float32')

            if batch == 1:
                  return a_np, w_np, b_np
            else:
                  return a_np, w_np


#special case for small height and width (e.g.. h = w = 7), where (h*w) becomes dimension of the brgemm (M)
def intrin_libxsmm_hxw(ofmblock,ofw,ifmblock, stride_width,ifw,rco, ifh,r,s, ifh_stride, ifw_stride,\
                       ofh, stride_height, out_channel,output_height, output_width, in_channel):

    last_input_width_index = (ofw-1)*stride_width + s-1

    last_input_height_index = (ofh-1)*stride_height + r-1
    ry = tvm.reduce_axis((0, r), name='ry')
    rx = tvm.reduce_axis((0, s), name='rx')


    A = tvm.placeholder((rco,r,s,ifmblock, ofmblock), name='w')
    B = tvm.placeholder((rco,last_input_height_index + 1,last_input_width_index + 1,ifmblock), name='b')
    k = tvm.reduce_axis((0, ifmblock), name='k')
    k_outer = tvm.reduce_axis((0, rco), name='k_outer')
    C = tvm.compute(
          (ofh,ofw,ofmblock),
           lambda z,m,n: tvm.sum(A[k_outer,ry,rx,k,n] * B[k_outer,ry + z*stride_height,rx + m*stride_width,k], axis=[k_outer,ry,rx,k]),
           name='out')

    s1 = tvm.create_schedule(C.op)

    ifw1,ofw1,ofmblock1  = s1[C].op.axis

    rco_outer,ry,rx,rci = s1[C].op.reduce_axis
    s1[C].reorder(ifw1,rco_outer,ry,rx,ofw1,ofmblock1,rci)

    xx_ptr = tvm.decl_buffer(A.shape, A.dtype,
                        name="W",offset_factor = 1,
                        data_alignment=64)


    yy_ptr = tvm.decl_buffer(B.shape, B.dtype,
                        name="X",offset_factor=1,\
                        strides=[tvm.var("s3"),tvm.var("s2"), ifmblock, 1],#offset_factor=16
                        data_alignment=64)

    zz_ptr = tvm.decl_buffer(C.shape, C.dtype,
                        name="OUT",offset_factor=1,#offset_factor=1,
                        strides=[output_width*ofmblock, ofmblock, 1],
                        data_alignment=64)

    def intrin_func(ins, outs):
         # tvm call extern is used to interface to libxsmm bacth reduce kernel gemm implementation
         # rco*r*s is the number of batches
         init_and_compute = tvm.call_extern ("int32","batch_reduce_kernel_init_update", ins[0].access_ptr("r"),ins[1].access_ptr("r"),outs[0].access_ptr("w"),\
                                                rco*r*s,ofmblock,ifmblock,r,s,ifh_stride,ifw_stride, ofw*ofh, stride_width)
         reset = tvm.call_extern ("int32","batch_reduce_kernel_init", outs[0].access_ptr("w"),ofmblock, ofw*ofh)
         body = tvm.call_extern ("int32","batch_reduce_kernel_update", ins[0].access_ptr("r"),ins[1].access_ptr("r"),outs[0].access_ptr("w"), rco*r*s,ofmblock,\
                                        ifmblock,ofw*ofh, stride_width,r,s, ifh_stride,ifw_stride)
         if math.ceil(in_channel/ifmblock) == rco:
            return init_and_compute, None, init_and_compute
         else:
            return init_and_compute,reset,body

    with tvm.build_config(data_alignment=64):
        return tvm.decl_tensor_intrin(C.op, intrin_func,   name="GEMM",
                                  binds= {A: xx_ptr,
                                         B: yy_ptr,
                                         C: zz_ptr})

# regular case of batch reduce gemm with ofw corresponding to batch reduce brgemm dimension(M)
def intrin_libxsmm_tuned(ofmblock,ofw,ifmblock, stride_width,ifw,rco, ifh,r,s, ifh_stride, ifw_stride, in_channel):
    last_input_width_index = (ofw-1)*stride_width + s-1
    A = tvm.placeholder((rco,r,s,ifmblock, ofmblock), name='w')
    B = tvm.placeholder((rco,r,last_input_width_index + 1,ifmblock), name='b')
    k = tvm.reduce_axis((0, ifmblock), name='k')
    k_outer = tvm.reduce_axis((0, rco), name='k_outer')
    ry = tvm.reduce_axis((0, r), name='ry')
    rx = tvm.reduce_axis((0, s), name='rx')
    C = tvm.compute(
          (ofw,ofmblock),
           lambda m,n: tvm.sum(A[k_outer,ry,rx,k,n] * B[k_outer,ry, rx + m*stride_width,k], axis=[k_outer,ry,rx,k]),
           name='out')
    s1 = tvm.create_schedule(C.op)
    w,ofm  = s1[C].op.axis
    kco,ky,kx,kci = s1[C].op.reduce_axis
    s1[C].reorder(kco,ky,kx,w,ofm,kci)
    xx_ptr = tvm.decl_buffer(A.shape, A.dtype,
                        name="W",offset_factor=1,
                        data_alignment=64)

    yy_ptr = tvm.decl_buffer(B.shape, B.dtype,
                        name="some", offset_factor=1,strides=[tvm.var("s3"), tvm.var("s2"), ifmblock, 1],
                        data_alignment=64)

    zz_ptr = tvm.decl_buffer(C.shape, C.dtype,
                        name="OUT",offset_factor=1,
                        data_alignment=64)

    def intrin_func(ins, outs):
         # tvm call extern is used to interface to libxsmm batch reduce kernel gemm implementation
         # rco*r*s is the number of batches
         init_and_compute = tvm.call_extern ("int32","batch_reduce_kernel_init_update", ins[0].access_ptr("r"),ins[1].access_ptr("r"),outs[0].access_ptr("w"),\
                                                rco*r*s,ofmblock,ifmblock,r,s,ifh_stride,ifw_stride, ofw, stride_width)
         reset = tvm.call_extern ("int32","batch_reduce_kernel_init", outs[0].access_ptr("w"),ofmblock, ofw)
         body = tvm.call_extern ("int32","batch_reduce_kernel_update", ins[0].access_ptr("r"),ins[1].access_ptr("r"),outs[0].access_ptr("w"), rco*r*s,ofmblock,\
                                        ifmblock,ofw, stride_width,r,s, ifh_stride,ifw_stride)
         if math.ceil(in_channel/ifmblock) == rco:
            return init_and_compute, None, init_and_compute
         else:
            return init_and_compute,reset,body

    with tvm.build_config(data_alignment=64):
        return tvm.decl_tensor_intrin(C.op, intrin_func,   name="GEMM",
                                         binds={A: xx_ptr,
                                                B: yy_ptr,
                                                C: zz_ptr})

#AutoTVM template for libxmm brgemm based tensorize implementation
@autotvm.template
def conv_auto_tuned(ofmblock,ofw, ifmblock, stride_width,input_width,\
                    in_channel,input_height, filter_height, filter_width,ofh, stride_height, batch, out_channel):

  A1 = tvm.placeholder((batch,math.ceil(in_channel/ifmblock),input_height, input_width, ifmblock), name='input')
  W1 = tvm.placeholder((math.ceil(out_channel/ofmblock), math.ceil(in_channel/ifmblock), filter_height, filter_width, ifmblock,ofmblock), name='weight')

  rco1 = tvm.reduce_axis((0, math.ceil(in_channel/ifmblock)), name='rco1')
  ry1 = tvm.reduce_axis((0, filter_height), name='ry1')
  rx1 = tvm.reduce_axis((0, filter_width), name='rx1')
  rci1 = tvm.reduce_axis((0, ifmblock), name='rci1')
  cfg = autotvm.get_config()

  cfg.define_knob("pack", [0,1])
  pack = False
  w_tile =  []

  factor_found = False


  for i in range(6, min(ofw+1,29)):
      if ofw % i == 0:
          w_tile.append((i, ofw//i) )
          factor_found = True

  if factor_found == False:
      w_tile.append((ofw,1))

  #tile factors for output width
  cfg.define_knob("tile_w", w_tile)

  # pack data when stride > 1 and pack flag set so that data for brgemm is continuous
  if filter_height == 1 and filter_width == 1 and stride_width >  1 and stride_height > 1 and cfg['pack'].val == 1 :
      A2 =  tvm.compute((batch, math.ceil(in_channel/ifmblock),ofh,ofw,ifmblock),
          lambda n,c,h,w,vlen1: A1[n, c,h*stride_height,w*stride_width,vlen1])
      B1 = tvm.compute((batch, math.ceil(out_channel/ofmblock),ofh, ofw,ofmblock),
          lambda nn,ff,yy, xx, vlen1: tvm.sum(
               W1[ff,rco1,ry1,rx1,rci1,vlen1] * A2[nn, rco1, ry1 + yy, rx1 + xx,rci1],
        axis=[rco1,ry1, rx1, rci1]),name='output')
      pack = True
  else:
     # Compute the convolution
      B1 = tvm.compute((batch, math.ceil(out_channel/ofmblock),ofh, ofw,ofmblock),
          lambda nn,ff,yy, xx, vlen1: tvm.sum(
               W1[ff,rco1,ry1,rx1,rci1,vlen1] * A1[nn, rco1, ry1 + stride_height*yy, rx1 + stride_width*xx,rci1],
              axis=[rco1,ry1, rx1, rci1]), name='output')

  s = tvm.create_schedule(B1.op)
  n,ko,h,w,ki  = s[B1].op.axis
  rco,ry,rx, rci = s[B1].op.reduce_axis
  cfg.define_split("tile_h", h, num_outputs=3)#output height
  cfg.define_split("tile_c", rco, num_outputs=2) #input channel dimension
  cfg.define_split("tile_k",ko, num_outputs=2)   #output channel dimension
  w_factor_inner, _ =  cfg["tile_w"].val
  wo, wi = s[B1].split(w, w_factor_inner)        #tiling
  rco_o,rco_i =           cfg["tile_c"].apply(s, B1, rco)
  ko_o, ko_i =      cfg["tile_k"].apply(s, B1, ko)
  ho,hm, hi =  cfg["tile_h"].apply(s, B1, h)

  s[B1].reorder(n,ko_o,ho,ko_i,rco_o,hm,wo,hi,rco_i,ry,rx,wi,ki,rci)
  cfg.define_reorder("reorder_outer", [ko_i,rco_o,hm,wo], policy="all")
  cfg.add_flop(np.prod(get_const_tuple(B1.shape))*in_channel*filter_height*filter_width*2)
  cfg["reorder_outer"].apply(s, B1,[ko_i,rco_o,hm,wo])
  if (filter_height == 1 and filter_width == 1 and stride_width == 1 and stride_height == 1) or pack:
      if cfg["tile_h"].size[1] > 1 and w_factor_inner == ofw:#cfg["tile_w"].size[2] == ofw:
          libxsmm_tensorize = intrin_libxsmm_hxw(ofmblock,w_factor_inner,ifmblock, 1, w_factor_inner,
                                              cfg["tile_c"].size[1],cfg["tile_h"].size[2],\
                                               filter_height, filter_width,ofh,ofw,cfg["tile_h"].size[2],1, out_channel, ofh,ofw, in_channel)
          s[B1].tensorize(hi, libxsmm_tensorize)
      else:
          libxsmm_tensorize = intrin_libxsmm_tuned(ofmblock,w_factor_inner,ifmblock, 1, w_factor_inner,
                                              cfg["tile_c"].size[1], cfg["tile_h"].size[2],\
                                               filter_height, filter_width,ofh, ofw, in_channel)
          s[B1].tensorize(rco_i, libxsmm_tensorize)

  else:

      libxsmm_tensorize = intrin_libxsmm_tuned(ofmblock,w_factor_inner,ifmblock, stride_width, w_factor_inner,\
                                              cfg["tile_c"].size[1],  cfg["tile_h"].size[2],\
                                              filter_height, filter_width,input_height,input_width, in_channel)
      s[B1].tensorize(rco_i, libxsmm_tensorize)

  par = s[B1].fuse(n,ko_o,ho)
  s[B1].parallel(par)
  if pack:
     n1,c1,h1,w1,v1 = s[A2].op.axis
     par2 = s[A2].fuse(n1,c1,h1)
     s[A2].parallel(par)
     s[A2].vectorize(v1)

  s = s.normalize()

  return s, [W1, A1, B1]

def driver():


        book = xlwt.Workbook(encoding="utf-8")
        sheet1 = book.add_sheet("Sheet 1")
        row1=0
        sheet1.write(0,0,"Layer")
        sheet1.write(0,1,"AutoTVM_FLOPS")
        row1 = row1 + 1



        batch = _resnet_layers[layer][0]
        in_channel = _resnet_layers[layer][2]
        out_channel = _resnet_layers[layer][1]
        input_height = _resnet_layers[layer][3]
        input_width = _resnet_layers[layer][4]
        kernel_height = _resnet_layers[layer][5]
        kernel_width = _resnet_layers[layer][5]
        pad_height = _resnet_layers[layer][7]
        pad_width = _resnet_layers[layer][7]
        stride_height = _resnet_layers[layer][6]
        stride_width = _resnet_layers[layer][6]
        vlen = 64
        assert(pad_height == pad_width)
        assert(stride_height == stride_width)
        assert(kernel_height == kernel_width)

        output_width = ((input_width + 2 * pad_width - kernel_width) // stride_width) + 1
        output_height = ((input_height + 2 * pad_height - kernel_height) // stride_height) + 1
        assert(output_height == output_width)
        assert(input_height == input_width)


        ctx = tvm.context('llvm', 0)
        sheet1.write(row1,0,layer)



        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return


        task = autotvm.task.create(conv_auto_tuned, args=(vlen,output_width, vlen, stride_width,input_width + 2*pad_width, in_channel,\
               input_height + 2*pad_height, kernel_height, kernel_width,output_height, stride_height, batch, out_channel),\
                       target='llvm -mtriple=x86_64 -mcpu=skylake-avx512 -mattr=+skx,+fma,+fma4,+avx512ifma,+avx512f,+avx512cd,+avx512bw,+avx512vl,+avx512dq')

        logging.getLogger('autotvm').setLevel(logging.DEBUG)
        logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

        measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(), runner=autotvm.LocalRunner(number=1000, repeat=1,min_repeat_ms=1000))

        tuner = autotvm.tuner.RandomTuner(task)
        #Please limit n_trial to reduce tuning time
        n_trial= len(task.config_space)
        log_file = layer + ".log"

        #comment out the following call to tuner to just run the best case from log file history
        tuner.tune(n_trial=n_trial,
           measure_option=measure_option,
           callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=layer),

                           autotvm.callback.log_to_file(log_file)])
        with autotvm.apply_history_best( layer+'.log'):
            with tvm.target.create("llvm"):

                  a_np, w_np, b_np  = get_ref_data(batch,out_channel,in_channel,input_height,input_width,kernel_height, kernel_width,stride_height,pad_height)
                  s, arg_bufs = conv_auto_tuned(vlen,output_width, vlen, stride_width,input_width + 2*pad_width, in_channel,\
                                      input_height + 2*pad_height, kernel_height, kernel_width,output_height, stride_height, batch, out_channel)

                  a_np2 = convert_input(a_np, batch, in_channel,input_height,input_width,pad_height,pad_width,vlen, arg_bufs[1])
                  w_np2 = convert_weight(w_np, in_channel, out_channel, kernel_height, kernel_width,vlen,arg_bufs[0])
                  ctx = tvm.context('llvm', 0)
                  b = tvm.nd.array(np.zeros((batch, math.ceil(out_channel/vlen),output_height, output_width,vlen), dtype=arg_bufs[2].dtype), ctx)
                  a = tvm.nd.array(a_np2, ctx)
                  w = tvm.nd.array(w_np2, ctx)

                  func = tvm.build(s, arg_bufs,target=\
                          'llvm -mtriple=x86_64 -mcpu=skylake-avx512 -mattr=+skx,+fma,+fma4,+avx512ifma,+avx512f,+avx512cd,+avx512bw,+avx512vl,+avx512dq', name="conv2d")
                  func(w,a,b)
                  b_np_A = convert_output(b.asnumpy(), 1,out_channel, output_height, output_width,vlen)
                  np.testing.assert_allclose(b_np_A, b_np, rtol=1e-5)
                  evaluator1 = func.time_evaluator(func.entry_name, ctx, number=1000,repeat=1, min_repeat_ms=1)

                  t1 = evaluator1(w,a, b).mean
                  gflops_tvm1 = np.prod(get_const_tuple(arg_bufs[2].shape))*in_channel*kernel_height*kernel_width*2
                  gflops_tvm1 = gflops_tvm1/1e9/t1

                  print("Time for conv(tuned) is : {0:.6f}".format(t1))
                  print("GFLOPS  : {0:.3f} ".format( gflops_tvm1))


                  sheet1.write(row1,1,gflops_tvm1)

        row1 = row1 + 1
        book.save( "AutoTVM_tensorize_resnet" + layer +".xls")


if __name__ == "__main__":
    driver()

