/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(_OPENMP)
#  include <omp.h>
#endif

/* include c-based dnn library */
#include "../common/dnn_common.h"

#define CHKERR_LIBXSMM_DNN(A) \
  { \
    const int chkerr_libxsmm_dnn_ = A; \
    if (LIBXSMM_DNN_SUCCESS != chkerr_libxsmm_dnn_) { \
      fprintf(stderr, "%s\n", libxsmm_dnn_get_error(chkerr_libxsmm_dnn_)); \
      global_status = chkerr_libxsmm_dnn_; \
    } \
  }

int main(int argc, char* argv[]) {
  void* scratch = NULL;
  size_t scratch_size = 0;

  /* some parameters we can overwrite via cli,
     default is some inner layer of overfeat */
  int warm_up_iters = 10;
  int iters = 10; /* repetitions of benchmark */
  int MB = 1; /* mini-batch size, "N" */


  int layers[48][11] = {{56, 56, 64, 64, 1, 1, 0, 0, 1, 1, 1}, {56, 56, 64, 64, 3, 3, 1, 1, 0, 0, 1},
    {56, 56, 64, 256, 1, 1, 0, 0, 0, 0, 1}, {56, 56, 256, 64, 1, 1, 0, 0, 1, 1, 1}, {56, 56, 64, 64, 3, 3, 1, 1, 0, 0, 1},
    {56, 56, 64, 256, 1, 1, 0, 0, 0, 0, 1}, {56, 56, 256, 64, 1, 1, 0, 0, 1, 1, 1}, {56, 56, 64, 64, 3, 3, 1, 1, 0, 0, 1},
    {56, 56, 64, 256, 1, 1, 0, 0, 0, 0, 1}, {56, 56, 256, 128, 1, 1, 0, 0, 1, 1, 2}, {28, 28, 128, 128, 3, 3, 1, 1, 0, 0, 1},
    {28, 28, 128, 512, 1, 1, 0, 0, 0, 0, 1}, {28, 28, 512, 128, 1, 1, 0, 0, 1, 1, 1}, {28, 28, 128, 128, 3, 3, 1, 1, 0, 0, 1},
    {28, 28, 128, 512, 1, 1, 0, 0, 0, 0, 1}, {28, 28, 512, 128, 1, 1, 0, 0, 1, 1, 1}, {28, 28, 128, 128, 3, 3, 1, 1, 0, 0, 1},
    {28, 28, 128, 512, 1, 1, 0, 0, 0, 0, 1}, {28, 28, 512, 128, 1, 1, 0, 0, 1, 1, 1}, {28, 28, 128, 128, 3, 3, 1, 1, 0, 0, 1},
    {28, 28, 128, 512, 1, 1, 0, 0, 0, 0, 1}, {28, 28, 512, 256, 1, 1, 0, 0, 1, 1, 2}, {14, 14, 256, 256, 3, 3, 1, 1, 0, 0, 1},
    {14, 14, 256, 1024, 1, 1, 0, 0, 0, 0, 1}, {14, 14, 1024, 256, 1, 1, 0, 0, 1, 1, 1}, {14, 14, 256, 256, 3, 3, 1, 1, 0, 0, 1},
    {14, 14, 256, 1024, 1, 1, 0, 0, 0, 0, 1}, {14, 14, 1024, 256, 1, 1, 0, 0, 1, 1, 1}, {14, 14, 256, 256, 3, 3, 1, 1, 0, 0, 1},
    {14, 14, 256, 1024, 1, 1, 0, 0, 0, 0, 1}, {14, 14, 1024, 256, 1, 1, 0, 0, 1, 1, 1}, {14, 14, 256, 256, 3, 3, 1, 1, 0, 0, 1},
    {14, 14, 256, 1024, 1, 1, 0, 0, 0, 0, 1}, {14, 14, 1024, 256, 1, 1, 0, 0, 1, 1, 1}, {14, 14, 256, 256, 3, 3, 1, 1, 0, 0, 1},
    {14, 14, 256, 1024, 1, 1, 0, 0, 0, 0, 1}, {14, 14, 1024, 256, 1, 1, 0, 0, 1, 1, 1}, {14, 14, 256, 256, 3, 3, 1, 1, 0, 0, 1},
    {14, 14, 256, 1024, 1, 1, 0, 0, 0, 0, 1}, {14, 14, 1024, 512, 1, 1, 0, 0, 1, 1, 2}, {7, 7, 512, 512, 3, 3, 1, 1, 0, 0, 1},
    {7, 7, 512, 2048, 1, 1, 0, 0, 0, 0, 1}, {7, 7, 2048, 512, 1, 1, 0, 0, 1, 1, 1}, {7, 7, 512, 512, 3, 3, 1, 1, 0, 0, 1},
    {7, 7, 512, 2048, 1, 1, 0, 0, 0, 0, 1}, {7, 7, 2048, 512, 1, 1, 0, 0, 1, 1, 1}, {7, 7, 512, 512, 3, 3, 1, 1, 0, 0, 1},
    {7, 7, 512, 2048, 1, 1, 0, 0, 0, 0, 1}};

#if defined(_OPENMP)
  int nThreads = omp_get_max_threads(); /* number of threads */
#else
  int nThreads = 1; /* number of threads */
#endif

  unsigned long long l_start, l_end;
  double l_total = 0.0;
  double gflop = 0.0;
  int i, j;
  double fil_size = 0.0;
  unsigned long long max_act_size = 0;

  libxsmm_dnn_fullyconnected_desc fullyconnected_desc;
  libxsmm_dnn_fullyconnected** libxsmm_fc_layer;
  libxsmm_dnn_optimizer_desc optimizer_desc;
  libxsmm_dnn_optimizer** libxsmm_opt;
  libxsmm_dnn_softmaxloss_desc softmaxloss_desc;
  libxsmm_dnn_softmaxloss* libxsmm_softmax;
  libxsmm_dnn_tensor** libxsmm_delact;
  libxsmm_dnn_tensor** libxsmm_fil;
  libxsmm_dnn_tensor** libxsmm_delfil;
  libxsmm_dnn_tensor** libxsmm_bias;
  libxsmm_dnn_tensor** libxsmm_delbias;
  libxsmm_dnn_tensor** libxsmm_relumask;
  libxsmm_dnn_tensor* libxsmm_label;
  libxsmm_dnn_tensor_datalayout* libxsmm_layout;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status = LIBXSMM_DNN_SUCCESS;
  libxsmm_dnn_layer** libxsmm_conv_layers;
  float* filter_rsck[48];
  float* act_ring_buffer[2];
  libxsmm_dnn_tensor* libxsmm_act[2];
  libxsmm_dnn_tensor* libxsmm_filter[48];

  libxsmm_conv_layers = (libxsmm_dnn_layer**)malloc(48 * sizeof(libxsmm_dnn_layer*));

  libxsmm_rng_set_seed(1);
  for (i = 0; i < 48; ++i) {
    int ifh = layers[i][0];
    int ifw = layers[i][1];
    int ifm = layers[i][2];
    int ofm = layers[i][3];
    int kh = layers[i][4];
    int kw = layers[i][5];
    int pad_h_in = layers[i][6];
    int pad_w_in = layers[i][7];
    int pad_h_out = layers[i][8];
    int pad_w_out = layers[i][9];
    int stride = layers[i][10];
    libxsmm_dnn_conv_desc conv_desc;

    printf("\n");
    printf("##########################################\n");
    printf("#    Setting Up - (NHWC/RSCK-Storage)    #\n");
    printf("##########################################\n");

    /* setup LIBXSMM handle */
    conv_desc.N = MB;
    conv_desc.C = ifm;
    conv_desc.H = ifh;
    conv_desc.W = ifw;
    conv_desc.K = ofm;
    conv_desc.R = kh;
    conv_desc.S = kw;
    conv_desc.u = stride;
    conv_desc.v = stride;
    conv_desc.pad_h = pad_h_in;
    conv_desc.pad_w = pad_w_in;
    conv_desc.pad_h_in = pad_h_in;
    conv_desc.pad_w_in = pad_w_in;
    conv_desc.pad_h_out = pad_h_out;
    conv_desc.pad_w_out = pad_w_out;
    conv_desc.threads = nThreads;
    conv_desc.algo = LIBXSMM_DNN_CONV_ALGO_DIRECT;
    conv_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_NHWC;
    conv_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_RSCK;
#ifdef USE_OVERWRITE
    conv_desc.options = LIBXSMM_DNN_CONV_OPTION_OVERWRITE;
#else
    conv_desc.options = LIBXSMM_DNN_CONV_OPTION_NONE;
#endif
    conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
    conv_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    conv_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;

    libxsmm_conv_layers[i] = libxsmm_dnn_create_conv_layer(conv_desc, &status);
    CHKERR_LIBXSMM_DNN(status);
    filter_rsck[i] = (float*)libxsmm_aligned_malloc(ofm * ifm * kh * kw * sizeof(float), 2097152);
    init_buf(filter_rsck[i], ofm * ifm * kh * kw, 0, 0);

    unsigned long long layer_act_size;

    if (i == 0) {
      max_act_size = MB * (ifh + 2 * pad_h_in) * (ifw + 2 * pad_w_in) * ifm * sizeof(float);

      printf("SIZE Input activations  %i (%dx%dx%dx%d): %10.2f MiB\n", i, MB, ifh + 2 * pad_h_in, ifw + 2 * pad_w_in, ifm,
        (double)(max_act_size) / (1024.0 * 1024.0));
    }

    layer_act_size = MB * (ifh / stride + 2 * pad_h_out) * (ifw / stride + 2 * pad_w_out) * ofm * sizeof(float);

    printf("SIZE Output Activations  %i (%dx%dx%dx%d): %10.2f MiB\n", i, MB, ifh / stride + 2 * pad_h_out,
      ifw / stride + 2 * pad_w_out, ofm, (double)(layer_act_size) / (1024.0 * 1024.0));

    if (layer_act_size > max_act_size) max_act_size = layer_act_size;
  }

  printf("max act size %ld\n", max_act_size);
  act_ring_buffer[0] = (float*)libxsmm_aligned_malloc(max_act_size, 2097152);
  init_buf(act_ring_buffer[0], max_act_size / sizeof(float), 0, 0);
  act_ring_buffer[1] = (float*)libxsmm_aligned_malloc(max_act_size, 2097152);
  zero_buf(act_ring_buffer[1], max_act_size / sizeof(float));

  for (i = 0; i < 48; ++i) {
    /* setup LIBXSMM buffers */
    if (i % 2 == 0) {
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout(libxsmm_conv_layers[i], LIBXSMM_DNN_INPUT, &status);
      CHKERR_LIBXSMM_DNN(status);
      libxsmm_act[0] = libxsmm_dnn_link_tensor(libxsmm_layout, act_ring_buffer[0], &status);
      CHKERR_LIBXSMM_DNN(status);
      libxsmm_dnn_destroy_tensor_datalayout(libxsmm_layout);

      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout(libxsmm_conv_layers[i], LIBXSMM_DNN_OUTPUT, &status);
      CHKERR_LIBXSMM_DNN(status);
      libxsmm_act[1] = libxsmm_dnn_link_tensor(libxsmm_layout, act_ring_buffer[1], &status);
      CHKERR_LIBXSMM_DNN(status);
      libxsmm_dnn_destroy_tensor_datalayout(libxsmm_layout);
    }
    else {
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout(libxsmm_conv_layers[i], LIBXSMM_DNN_INPUT, &status);
      CHKERR_LIBXSMM_DNN(status);
      libxsmm_act[1] = libxsmm_dnn_link_tensor(libxsmm_layout, act_ring_buffer[1], &status);
      CHKERR_LIBXSMM_DNN(status);
      libxsmm_dnn_destroy_tensor_datalayout(libxsmm_layout);

      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout(libxsmm_conv_layers[i], LIBXSMM_DNN_OUTPUT, &status);
      CHKERR_LIBXSMM_DNN(status);
      libxsmm_act[0] = libxsmm_dnn_link_tensor(libxsmm_layout, act_ring_buffer[0], &status);
      CHKERR_LIBXSMM_DNN(status);
      libxsmm_dnn_destroy_tensor_datalayout(libxsmm_layout);
    }


    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout(libxsmm_conv_layers[i], LIBXSMM_DNN_FILTER, &status);
    CHKERR_LIBXSMM_DNN(status);
    libxsmm_filter[i] = libxsmm_dnn_link_tensor(libxsmm_layout, filter_rsck[i], &status);
    CHKERR_LIBXSMM_DNN(status);
    libxsmm_dnn_destroy_tensor_datalayout(libxsmm_layout);

    if (i % 2 == 0) {
      /* bind buffers and filter to handle */
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor(libxsmm_conv_layers[i], libxsmm_act[0], LIBXSMM_DNN_REGULAR_INPUT));
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor(libxsmm_conv_layers[i], libxsmm_act[1], LIBXSMM_DNN_REGULAR_OUTPUT));
    }
    else {
      /* bind buffers and filter to handle */
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor(libxsmm_conv_layers[i], libxsmm_act[1], LIBXSMM_DNN_REGULAR_INPUT));
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor(libxsmm_conv_layers[i], libxsmm_act[0], LIBXSMM_DNN_REGULAR_OUTPUT));
    }
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor(libxsmm_conv_layers[i], libxsmm_filter[i], LIBXSMM_DNN_REGULAR_FILTER));

    /* let's allocate and bind scratch */
    scratch_size = libxsmm_dnn_get_scratch_size(libxsmm_conv_layers[i], LIBXSMM_DNN_COMPUTE_KIND_ALL, &status);
    CHKERR_LIBXSMM_DNN(status);
    scratch = libxsmm_aligned_scratch(scratch_size, 2097152);
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_scratch(libxsmm_conv_layers[i], LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch));
    /* set scratch to bogus to make sure that libxsmm takes care of zeroing internally */
    init_buf((float*)scratch, scratch_size / 4, 0, 0);
  }

  printf("Warming up..\n");

#if defined(_OPENMP)
#  pragma omp parallel private(i, j)
#endif
  {
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    for (j = 0; j < warm_up_iters; ++j) {
      for (i = 0; i < 48; ++i) {
        libxsmm_dnn_execute_st(libxsmm_conv_layers[i], LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid);
      }
    }
  }


  printf("##########################################\n");
  printf("#   Performance - FWD (NHWC/RSCK)   #\n");
  printf("##########################################\n");
  l_start = libxsmm_timer_tick();
#if defined(_OPENMP)
#  pragma omp parallel private(i, j)
#endif
  {
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    for (j = 0; j < iters; ++j) {
      for (i = 0; i < 48; ++i) {
        libxsmm_dnn_execute_st(libxsmm_conv_layers[i], LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid);
      }
    }
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  double average_inference_time = (double)(l_total / iters);


  for (j = 0; j < 48; ++j) {
    l_start = libxsmm_timer_tick();
    for (i = 0; i < iters; ++i) {
#if defined(_OPENMP)
#  pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        libxsmm_dnn_execute_st(libxsmm_conv_layers[j], LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid);
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    double flops = (double)MB * (double)layers[j][2] * (double)layers[j][3] * (double)(layers[j][0] / layers[j][10]) *
                   (double)(layers[j][1] / layers[j][10]) * (double)(2 * layers[j][4] * layers[j][5]) * (double)iters;
    printf("PARAMS: W:%d  H:%d  N:%d  C:%d  K:%d  R:%d  S:%d  P:%d  Q:%d  STRIDE:%d\n", layers[j][1], layers[j][0], MB,
      layers[j][2], layers[j][3], layers[j][4], layers[j][5], layers[j][0] / layers[j][10], layers[j][1] / layers[j][10],
      layers[j][10]);
    printf("PARAMS: ITERS:%d\n", iters);
    printf("Threads:%d\n", nThreads);
    printf("GFLOP for layer%d (NHWC,RSCK)  = %.5g\n", j, flops * 1e-9 / (double)iters);
    printf("fp time (NHWC,RSCK) = %.5g\n", ((double)(l_total / iters)));
    printf("GFLOPS (NHWC,RSCK) = %.5g\n\n", (flops * 1e-9) / l_total);
  }


  printf("\nAverage Inference time = %.5gs\n", average_inference_time);


  for (i = 0; i < 48; ++i) {
    /* clean-up */
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_release_scratch(libxsmm_conv_layers[i], LIBXSMM_DNN_COMPUTE_KIND_ALL));
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_release_tensor(libxsmm_conv_layers[i], LIBXSMM_DNN_REGULAR_INPUT));
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_release_tensor(libxsmm_conv_layers[i], LIBXSMM_DNN_REGULAR_OUTPUT));
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_release_tensor(libxsmm_conv_layers[i], LIBXSMM_DNN_REGULAR_FILTER));
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_destroy_conv_layer(libxsmm_conv_layers[i]));
  }

  for (i = 0; i < 48; ++i) {
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_destroy_tensor(libxsmm_filter[i]));
  }


  CHKERR_LIBXSMM_DNN(libxsmm_dnn_destroy_tensor(libxsmm_act[0]));
  CHKERR_LIBXSMM_DNN(libxsmm_dnn_destroy_tensor(libxsmm_act[1]));

  /* deallocate data */
  libxsmm_free(scratch);
  for (i = 0; i < 48; ++i) {
    libxsmm_free(filter_rsck[i]);
  }
  libxsmm_free(act_ring_buffer[0]);
  libxsmm_free(act_ring_buffer[1]);


  /* some empty lines at the end */
  printf("\n\n\n");

  return global_status;
}
