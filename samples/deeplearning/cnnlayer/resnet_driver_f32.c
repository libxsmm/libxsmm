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

#if 0
#define USE_CORE_PERF_COUNTERS
#endif

#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(_OPENMP)
#  include <omp.h>
#endif
#if defined(USE_CORE_PERF_COUNTERS)
#  include "../../external_aux/counters_skx.h"
#endif
#include <sys/time.h>
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


inline double sec(struct timeval start, struct timeval end);

inline double sec(struct timeval start, struct timeval end) {
  return ((double)(((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)))) / 1.0e6;
}


void print_help_message();
void print_help_message() {
  printf(
    "Usage: ./resnet_driver_f32 iters nImg activation_format filter_format buffer_format layer_mode [layer_mode_options]...\n");
  printf("activation_format         =      'T':  (TensorFlow) or  'L' (LIBXSMM)\n");
  printf("filter_format             =      'T':  (TensorFlow) or  'L' (LIBXSMM)\n");
  printf("buffer_format             =      'N':  (Normal)     or  'R' (Ring)\n");
  printf("layer_mode                =      'S':  (Single layer )     or  'R' (Range of sequential resnet layers) or  'A' (All "
         "Resnet layers)\n");
  printf(
    "layer_mode_options(layer_mode = 'S')        =      ifh ifw nIfm nOfm kw kw pad_w_in pad_h_in pad_w_out pad_h_out stride\n");
  printf("layer_mode_options(layer_mode = 'R')        =      range_start(1-48) range_end(1-48)\n");
}

void dump_layer_params(int (*layers)[11], int index, int MB);
void dump_layer_params(int (*layers)[11], int index, int MB) {
  printf("PARAMS: W:%d  H:%d  N:%d  C:%d  K:%d  R:%d  S:%d  P:%d  Q:%d  STRIDE:%d\n", layers[index][0], layers[index][1], MB,
    layers[index][2], layers[index][3], layers[index][5], layers[index][4], layers[index][1] / layers[index][10],
    layers[index][0] / layers[index][10], layers[index][10]);
}


void write_perf_to_csv_file(int layer, FILE* f, double min_time, double max_time, double average_time, double flops, int ifw,
  int ifh, int nImg, int nIfm, int nOfm, int kw, int kh, int stride, double bw_min, double bw_max, double bw_avg);

void write_perf_to_csv_file(int layer, FILE* f, double min_time, double max_time, double average_time, double flops, int ifw,
  int ifh, int nImg, int nIfm, int nOfm, int kw, int kh, int stride, double bw_min, double bw_max, double bw_avg) {
  if (bw_min >= 0 && bw_max >= 0 && bw_avg >= 0)
    fprintf(f, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f\n", layer, nImg, nOfm, nIfm, ifh, ifw, kh, kw, stride, min_time,
      max_time, average_time, flops, bw_min, bw_max, bw_avg);
  else
    fprintf(f, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%f,%f,%f,%f\n", layer, nImg, nOfm, nIfm, ifh, ifw, kh, kw, stride, min_time, max_time,
      average_time, flops);
}


int main(int argc, char* argv[]) {
  void* scratch = NULL;
  size_t scratch_size = 0;

  /* some parameters we can overwrite via cli,
     default is some inner layer of overfeat */
  int warm_up_iters = 10;
  int iters = 1000; /* repetitions of benchmark */
  int MB = 1; /* mini-batch size, "N" */
  char activation_format = 'T'; /* 'T' for TensorFlow(NHWC) and 'L' for LIBXSMM(NCbHWC) */
  char filter_format = 'T'; /* 'T' for TensorFlow(RSCK) and 'L' for LIBXSMM (KbCbRSCK) */
  char buffer_format = 'R'; /* 'R' for ring buffer, 'N' for normal buffer */
  char layer_mode = 'A'; /* 'A' for all layers, 'S' for single layer mode, 'R' for a range of layers */
  int ifw, ifh, nIfm, nOfm, pad_w_in, pad_h_in, pad_w_out, pad_h_out, kw, kh, stride, range_start, range_end;
  double** per_layer_time;

#if defined(USE_CORE_PERF_COUNTERS)
  bw_gibs bw_tot;
  ctrs_skx_core s;
  ctrs_skx_core(**a)[2];
#endif

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
  int i, j;
  unsigned long long max_act_size = 0;
  unsigned long long max_scratch_size = 0;

  ifw = 56;
  ifh = 56;
  nIfm = 64;
  nOfm = 64;
  pad_w_in = 0;
  pad_h_in = 0;
  pad_w_out = 1;
  pad_h_out = 1;
  kw = 1;
  kh = 1;
  stride = 1;
  range_start = 1;
  range_end = 48;

  libxsmm_dnn_tensor_datalayout* libxsmm_layout;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status = LIBXSMM_DNN_SUCCESS;
  libxsmm_dnn_layer** libxsmm_conv_layers;

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    print_help_message();
    return 0;
  }

  i = 1;

  if (argc > i) iters = atoi(argv[i++]);
  if (argc > i) MB = atoi(argv[i++]);
  if (argc > i) activation_format = *(argv[i++]);
  if (activation_format != 'T' && activation_format != 'L') {
    printf("Wrong number and/or type of arguments\n");
    print_help_message();
    return 0;
  }

  if (argc > i) filter_format = *(argv[i++]);
  if (filter_format != 'T' && filter_format != 'L') {
    printf("Wrong number and/or type of arguments\n");
    print_help_message();
    return 0;
  }

  if (argc > i) buffer_format = *(argv[i++]);
  if (buffer_format != 'N' && buffer_format != 'R') {
    printf("Wrong number and/or type of arguments\n");
    print_help_message();
    return 0;
  }

  if (argc > i) layer_mode = *(argv[i++]);
  if (layer_mode != 'S' && layer_mode != 'R' && layer_mode != 'A') {
    printf("Wrong number and/or type of arguments\n");
    print_help_message();
    return 0;
  }
  if (argc > i) {
    if (layer_mode == 'R') {
      range_start = atoi(argv[i++]);
      if (argc > i)
        range_end = atoi(argv[i++]);
      else
        range_end = range_start;
      if (range_start > 48 || range_start < 1 || range_end < range_start || range_end > 48) {
        printf("Wrong number and/or type of arguments\n");
        printf(
          "range_start and range_end should be within 1 and 48 inclusive with range_end greater than or equal to range_start\n");
        print_help_message();
        return 0;
      }
    }
    else if (layer_mode == 'S') {
      range_start = 1;
      range_end = 1;
      if (argc > i) ifw = atoi(argv[i++]);
      if (argc > i) ifh = atoi(argv[i++]);
      if (argc > i) nIfm = atoi(argv[i++]);
      if (argc > i) nOfm = atoi(argv[i++]);
      if (argc > i) kw = atoi(argv[i++]);
      if (argc > i) kh = atoi(argv[i++]);
      if (argc > i) pad_w_in = atoi(argv[i++]);
      if (argc > i) pad_h_in = atoi(argv[i++]);
      if (argc > i) pad_w_out = atoi(argv[i++]);
      if (argc > i) pad_h_out = atoi(argv[i++]);
      if (argc > i) stride = atoi(argv[i++]);
    }
  }

  float** filter;
  float** act_ring_buffer;
  libxsmm_dnn_tensor** libxsmm_act;
  libxsmm_dnn_tensor** libxsmm_filter;
  FILE* f;
  per_layer_time = (double(**))malloc((range_end - range_start + 1) * (sizeof(double(*))));

  f = fopen("results.csv", "w");

  for (i = 0; i < range_end - range_start + 1; ++i) per_layer_time[i] = (double(*))malloc(iters * (sizeof(double)));

#if defined(USE_CORE_PERF_COUNTERS)

  a = (ctrs_skx_core(**)[2])malloc((range_end - range_start + 2) * (sizeof(ctrs_skx_core(*)[2])));

  for (i = 0; i < range_end - range_start + 2; ++i) {
    a[i] = (ctrs_skx_core(*)[2])malloc(iters * (sizeof(ctrs_skx_core[2])));
    for (j = 0; j < iters; j++) {
      zero_skx_core_ctrs(&a[i][j][0]);
      zero_skx_core_ctrs(&a[i][j][1]);
    }
  }

  zero_skx_core_ctrs(&s);

  setup_skx_core_ctrs(CTRS_EXP_L2_BW);
  fprintf(f, "layer,N,K,C,H,W,R,S,stride,min time,max time,average time,flops, min bw, max bw, avg bw\n");
#else
  fprintf(f, "layer,N,K,C,H,W,R,S,stride,min time,max time,average time,flops\n");
#endif
  libxsmm_conv_layers = (libxsmm_dnn_layer**)malloc((range_end - range_start + 1) * sizeof(libxsmm_dnn_layer*));
  filter = (float**)malloc((range_end - range_start + 1) * sizeof(float*));
  libxsmm_filter = (libxsmm_dnn_tensor**)malloc((range_end - range_start + 1) * sizeof(libxsmm_dnn_tensor*));

  if (buffer_format == 'R') {
    act_ring_buffer = (float**)malloc(2 * sizeof(float*));
    libxsmm_act = (libxsmm_dnn_tensor**)malloc(2 * sizeof(libxsmm_dnn_tensor*));
  }
  else {
    act_ring_buffer = (float**)malloc(2 * (range_end - range_start + 1) * sizeof(float*));
    libxsmm_act = (libxsmm_dnn_tensor**)malloc(2 * (range_end - range_start + 1) * sizeof(libxsmm_dnn_tensor*));
  }

  libxsmm_rng_set_seed(1);
  printf("\n");
  printf("##########################################\n");
  printf("#            Setting Up ...              #\n");
  printf("##########################################\n");

  for (i = range_start - 1; i <= range_end - 1; ++i) {
    if (layer_mode != 'S') {
      ifh = layers[i][0];
      ifw = layers[i][1];
      nIfm = layers[i][2];
      nOfm = layers[i][3];
      kh = layers[i][4];
      kw = layers[i][5];
      pad_h_in = layers[i][6];
      pad_w_in = layers[i][7];
      pad_h_out = layers[i][8];
      pad_w_out = layers[i][9];
      stride = layers[i][10];
    }
    libxsmm_dnn_conv_desc conv_desc;
    /* setup LIBXSMM handle */
    conv_desc.N = MB;
    conv_desc.C = nIfm;
    conv_desc.H = ifh;
    conv_desc.W = ifw;
    conv_desc.K = nOfm;
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

    if (activation_format == 'T')
      conv_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_NHWC;
    else
      conv_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;


    if (filter_format == 'T')
      conv_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_RSCK;
    else
      conv_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;

    conv_desc.options = LIBXSMM_DNN_CONV_OPTION_OVERWRITE;
    conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
    conv_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    conv_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;

    libxsmm_conv_layers[i - range_start + 1] = libxsmm_dnn_create_conv_layer(conv_desc, &status);
    CHKERR_LIBXSMM_DNN(status);
    filter[i - range_start + 1] = (float*)libxsmm_aligned_malloc(nOfm * nIfm * kh * kw * sizeof(float), 2097152);
    init_buf(filter[i - range_start + 1], nOfm * nIfm * kh * kw, 0, 0);

    unsigned long long input_act_size, output_act_size;

    input_act_size = MB * (ifh + 2 * pad_h_in) * (ifw + 2 * pad_w_in) * nIfm * sizeof(float);
    if (i == 0) max_act_size = input_act_size; /* MB * (ifh + 2 * pad_h_in) * (ifw + 2 * pad_w_in) * ifm * sizeof(float); */


    if (buffer_format == 'N') {
      act_ring_buffer[2 * (i - range_start + 1)] = (float*)libxsmm_aligned_malloc(input_act_size, 2097152);
      init_buf(act_ring_buffer[2 * (i - range_start + 1)], input_act_size / sizeof(float), 0, 0);
    }


    printf("SIZE Input activations  %i (%dx%dx%dx%d): %10.2f MiB\n", i - range_start + 1, MB, ifh + 2 * pad_h_in,
      ifw + 2 * pad_w_in, nIfm, (double)(input_act_size) / (1024.0 * 1024.0));

    output_act_size = MB * (ifh / stride + 2 * pad_h_out) * (ifw / stride + 2 * pad_w_out) * nOfm * sizeof(float);

    if (buffer_format == 'N') {
      act_ring_buffer[2 * (i - range_start + 1) + 1] = (float*)libxsmm_aligned_malloc(output_act_size, 2097152);
      zero_buf(act_ring_buffer[2 * (i - range_start + 1) + 1], output_act_size / sizeof(float));
    }
    printf("SIZE Output Activations  %i (%dx%dx%dx%d): %10.2f MiB\n", i - range_start + 1, MB, ifh / stride + 2 * pad_h_out,
      ifw / stride + 2 * pad_w_out, nOfm, (double)(output_act_size) / (1024.0 * 1024.0));

    if (buffer_format == 'N') {
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout(libxsmm_conv_layers[i - range_start + 1], LIBXSMM_DNN_INPUT, &status);
      CHKERR_LIBXSMM_DNN(status);
      libxsmm_act[2 * (i - range_start + 1)] = libxsmm_dnn_link_tensor(
        libxsmm_layout, act_ring_buffer[2 * (i - range_start + 1)], &status);
      CHKERR_LIBXSMM_DNN(status);
      libxsmm_dnn_destroy_tensor_datalayout(libxsmm_layout);

      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout(libxsmm_conv_layers[i - range_start + 1], LIBXSMM_DNN_OUTPUT, &status);
      CHKERR_LIBXSMM_DNN(status);
      libxsmm_act[2 * (i - range_start + 1) + 1] = libxsmm_dnn_link_tensor(
        libxsmm_layout, act_ring_buffer[2 * (i - range_start + 1) + 1], &status);
      CHKERR_LIBXSMM_DNN(status);
      libxsmm_dnn_destroy_tensor_datalayout(libxsmm_layout);

      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout(libxsmm_conv_layers[i - range_start + 1], LIBXSMM_DNN_FILTER, &status);
      CHKERR_LIBXSMM_DNN(status);
      libxsmm_filter[i - range_start + 1] = libxsmm_dnn_link_tensor(libxsmm_layout, filter[i - range_start + 1], &status);
      CHKERR_LIBXSMM_DNN(status);
      libxsmm_dnn_destroy_tensor_datalayout(libxsmm_layout);

      /* bind buffers and filter to handle */
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor(
        libxsmm_conv_layers[i - range_start + 1], libxsmm_act[2 * (i - range_start + 1)], LIBXSMM_DNN_REGULAR_INPUT));
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor(
        libxsmm_conv_layers[i - range_start + 1], libxsmm_act[2 * (i - range_start + 1) + 1], LIBXSMM_DNN_REGULAR_OUTPUT));
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor(
        libxsmm_conv_layers[i - range_start + 1], libxsmm_filter[i - range_start + 1], LIBXSMM_DNN_REGULAR_FILTER));

      /* let's allocate and bind scratch */
      scratch_size = libxsmm_dnn_get_scratch_size(libxsmm_conv_layers[i - range_start + 1], LIBXSMM_DNN_COMPUTE_KIND_ALL, &status);
      CHKERR_LIBXSMM_DNN(status);
    }


    if (scratch_size > max_scratch_size) max_scratch_size = scratch_size;
    if (output_act_size > max_act_size) max_act_size = output_act_size;
  }

  if (buffer_format == 'R') {
    act_ring_buffer[0] = (float*)libxsmm_aligned_malloc(max_act_size, 2097152);
    init_buf(act_ring_buffer[0], max_act_size / sizeof(float), 0, 0);
    act_ring_buffer[1] = (float*)libxsmm_aligned_malloc(max_act_size, 2097152);
    zero_buf(act_ring_buffer[1], max_act_size / sizeof(float));

    for (i = range_start - 1; i <= range_end - 1; ++i) {
      /* setup LIBXSMM buffers */
      if (i % 2 == 0) {
        libxsmm_layout = libxsmm_dnn_create_tensor_datalayout(libxsmm_conv_layers[i - range_start + 1], LIBXSMM_DNN_INPUT, &status);
        CHKERR_LIBXSMM_DNN(status);
        libxsmm_act[0] = libxsmm_dnn_link_tensor(libxsmm_layout, act_ring_buffer[0], &status);
        CHKERR_LIBXSMM_DNN(status);
        libxsmm_dnn_destroy_tensor_datalayout(libxsmm_layout);

        libxsmm_layout = libxsmm_dnn_create_tensor_datalayout(
          libxsmm_conv_layers[i - range_start + 1], LIBXSMM_DNN_OUTPUT, &status);
        CHKERR_LIBXSMM_DNN(status);
        libxsmm_act[1] = libxsmm_dnn_link_tensor(libxsmm_layout, act_ring_buffer[1], &status);
        CHKERR_LIBXSMM_DNN(status);
        libxsmm_dnn_destroy_tensor_datalayout(libxsmm_layout);
      }
      else {
        libxsmm_layout = libxsmm_dnn_create_tensor_datalayout(libxsmm_conv_layers[i - range_start + 1], LIBXSMM_DNN_INPUT, &status);
        CHKERR_LIBXSMM_DNN(status);
        libxsmm_act[1] = libxsmm_dnn_link_tensor(libxsmm_layout, act_ring_buffer[1], &status);
        CHKERR_LIBXSMM_DNN(status);
        libxsmm_dnn_destroy_tensor_datalayout(libxsmm_layout);

        libxsmm_layout = libxsmm_dnn_create_tensor_datalayout(
          libxsmm_conv_layers[i - range_start + 1], LIBXSMM_DNN_OUTPUT, &status);
        CHKERR_LIBXSMM_DNN(status);
        libxsmm_act[0] = libxsmm_dnn_link_tensor(libxsmm_layout, act_ring_buffer[0], &status);
        CHKERR_LIBXSMM_DNN(status);
        libxsmm_dnn_destroy_tensor_datalayout(libxsmm_layout);
      }


      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout(libxsmm_conv_layers[i - range_start + 1], LIBXSMM_DNN_FILTER, &status);
      CHKERR_LIBXSMM_DNN(status);
      libxsmm_filter[i - range_start + 1] = libxsmm_dnn_link_tensor(libxsmm_layout, filter[i - range_start + 1], &status);
      CHKERR_LIBXSMM_DNN(status);
      libxsmm_dnn_destroy_tensor_datalayout(libxsmm_layout);

      if (i % 2 == 0) {
        /* bind buffers and filter to handle */
        CHKERR_LIBXSMM_DNN(
          libxsmm_dnn_bind_tensor(libxsmm_conv_layers[i - range_start + 1], libxsmm_act[0], LIBXSMM_DNN_REGULAR_INPUT));
        CHKERR_LIBXSMM_DNN(
          libxsmm_dnn_bind_tensor(libxsmm_conv_layers[i - range_start + 1], libxsmm_act[1], LIBXSMM_DNN_REGULAR_OUTPUT));
      }
      else {
        /* bind buffers and filter to handle */
        CHKERR_LIBXSMM_DNN(
          libxsmm_dnn_bind_tensor(libxsmm_conv_layers[i - range_start + 1], libxsmm_act[1], LIBXSMM_DNN_REGULAR_INPUT));
        CHKERR_LIBXSMM_DNN(
          libxsmm_dnn_bind_tensor(libxsmm_conv_layers[i - range_start + 1], libxsmm_act[0], LIBXSMM_DNN_REGULAR_OUTPUT));
      }
      CHKERR_LIBXSMM_DNN(
        libxsmm_dnn_bind_tensor(libxsmm_conv_layers[i - range_start + 1], libxsmm_filter[i], LIBXSMM_DNN_REGULAR_FILTER));
    }
  }

  /* let's allocate and bind scratch */
  scratch = libxsmm_aligned_scratch(max_scratch_size, 2097152);
  init_buf((float*)scratch, scratch_size / 4, 0, 0);

  for (i = range_start - 1; i <= range_end - 1; ++i) {
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_scratch(libxsmm_conv_layers[i - range_start + 1], LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch));
  }

  printf("\n");
  printf("##########################################\n");
  printf("#         Setting Up ... done            #\n");
  printf("##########################################\n");

  printf("\n");
  printf("##########################################\n");
  printf("#            Warming Up ...              #\n");
  printf("##########################################\n");

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
      for (i = 0; i < range_end - range_start + 1; ++i) {
        libxsmm_dnn_execute_st(libxsmm_conv_layers[i], LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid);
      }
    }
  }

  printf("\n");
  printf("##########################################\n");
  printf("#          Warming Up ... done           #\n");
  printf("##########################################\n");

  printf("\n");
  printf("##########################################\n");
  printf("#  Performance: One OpenMP, full topo    #\n");
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
      for (i = 0; i < range_end - range_start + 1; ++i) {
        libxsmm_dnn_execute_st(libxsmm_conv_layers[i], LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid);
      }
    }
#if defined(_OPENMP)
  }
#endif
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  double average_inference_time_a = (double)(l_total / iters);
  printf("\nAverage Inference time (one   OpenMP region) = %.5gs\n", average_inference_time_a);

  printf("\n");
  printf("##########################################\n");
  printf("# Performance: Single OpenMP, full topo  #\n");
  printf("##########################################\n");

  l_start = libxsmm_timer_tick();
  for (j = 0; j < iters; ++j) {
    for (i = 0; i < range_end - range_start + 1; ++i) {
#if defined(_OPENMP)
#  pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        libxsmm_dnn_execute_st(libxsmm_conv_layers[i], LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid);
#if defined(_OPENMP)
      }
#endif
    }
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  double average_inference_time_b = (double)(l_total / iters);
  printf("\nAverage Inference time (layer OpenMP region) = %.5gs\n", average_inference_time_b);

  printf("\n");
  printf("##########################################\n");
  printf("# Performance: Single OpenMP, layerwise  #\n");
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
      unsigned long long start, end;
      if (tid == 0) {
#if defined(USE_CORE_PERF_COUNTERS)
        read_skx_core_ctrs(&a[0][j][0]);
#endif
        start = libxsmm_timer_tick();
      }
      for (i = 0; i < range_end - range_start + 1; ++i) {
        libxsmm_dnn_execute_st(libxsmm_conv_layers[i], LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid);

        if (tid == 0) {
          end = libxsmm_timer_tick();
#if defined(USE_CORE_PERF_COUNTERS)
          read_skx_core_ctrs(&a[i+1][j][0]);
#endif
          per_layer_time[i][j] = libxsmm_timer_duration(start, end);
          start = end;
        }
      }
    }
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  double average_inference_time_c = (double)(l_total / iters);
  printf("\nAverage Inference time (layerwise measuremenr execution) = %.5gs\n", average_inference_time_c);

  for (i = 0; i < range_end - range_start + 1; ++i) {
    if (iters > 0) {
      double l_total = per_layer_time[i][0];
      double l_min = l_total;
      double l_max = l_total;
      double bwmax = -1.0;
      double bwmin = -1.0;
      double bwtotal = -1.0;

#if defined(USE_CORE_PERF_COUNTERS)
      difa_skx_core_ctrs(&a[i][0][0], &a[i+1][0][0], &s);
      get_l2_bw_skx(&s, l_total, &bw_tot);
      zero_skx_core_ctrs(&s);
      bwmax = bw_tot.rd;
      bwmin = bw_tot.rd;
      bwtotal = bw_tot.rd;
#endif
      for (j = 1; j < iters; ++j) {
        l_total += per_layer_time[i][j];
        if (l_min > per_layer_time[i][j]) l_min = per_layer_time[i][j];
        if (l_max < per_layer_time[i][j]) l_max = per_layer_time[i][j];

#if defined(USE_CORE_PERF_COUNTERS)
        difa_skx_core_ctrs(&a[i][j][0], &a[i+1][j][0], &s);
        double bwcurr;
        bw_gibs bw_curr;
        get_l2_bw_skx(&s, per_layer_time[i][j], &bw_curr);
        zero_skx_core_ctrs(&s);
        bwcurr = bw_curr.rd;

        bwtotal += bwcurr;
        if (bwmin > bwcurr) bwmin = bwcurr;
        if (bwmax < bwcurr) bwmax = bwcurr;
#endif
      }

      double flops;

      if (layer_mode == 'S') {
#if 0
        printf("PARAMS: W:%d  H:%d  N:%d  C:%d  K:%d  R:%d  S:%d  P:%d  Q:%d  STRIDE:%d\n", ifw, ifh, MB, nIfm, nOfm, kh, kw,
          ifh / stride, ifw / stride, stride);
#endif
      }
      else {
        ifw = layers[i + range_start - 1][0];
        ifh = layers[i + range_start - 1][1];
        nIfm = layers[i + range_start - 1][2];
        nOfm = layers[i + range_start - 1][3];
        kw = layers[i + range_start - 1][4];
        kh = layers[i + range_start - 1][5];
        stride = layers[i + range_start - 1][10];
#if 0
        dump_layer_params((int(*)[11])layers, i + range_start - 1, MB);
#endif
      }
      flops = (double)MB * (double)nIfm * (double)nOfm * (double)(ifw / stride) * (double)(ifh / stride) * (double)(2 * kw * kh) *
              (double)iters;

#if 0
      printf("l_total:%f\n", l_total);
      printf("PARAMS: ITERS:%d\n", iters);
      printf("Threads:%d\n", nThreads);
      printf("GFLOP for layer%d (NHWC,RSCK)  = %.5g\n", i, flops * 1e-9 / (double)iters);
      printf("fp time (NHWC,RSCK) = %.5g\n", ((double)(l_total / iters)));
      printf("GFLOPS (NHWC,RSCK) = %.5g\n\n", (flops * 1e-9) / l_total);
#endif
      write_perf_to_csv_file(i, f, l_min, l_max, (double)(l_total / iters), (flops * 1e-9) / l_total, ifw, ifh, MB, nIfm, nOfm, kw,
        kh, stride, bwmin, bwmax, (double)(bwtotal / iters));
    }
  }

  printf("dumped layer-wise results into results.csv\n");

  printf("\n");
  printf("##########################################\n");
  printf("#              Cleaning up               #\n");
  printf("##########################################\n");

  for (i = 0; i < range_end - range_start + 1; ++i) {
    /* clean-up */
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_release_scratch(libxsmm_conv_layers[i], LIBXSMM_DNN_COMPUTE_KIND_ALL));
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_release_tensor(libxsmm_conv_layers[i], LIBXSMM_DNN_REGULAR_INPUT));
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_release_tensor(libxsmm_conv_layers[i], LIBXSMM_DNN_REGULAR_OUTPUT));
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_release_tensor(libxsmm_conv_layers[i], LIBXSMM_DNN_REGULAR_FILTER));
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_destroy_conv_layer(libxsmm_conv_layers[i]));
    free(per_layer_time[i]);
#if defined(USE_CORE_PERF_COUNTERS)
    free(a[i]);
#endif
  }

  for (i = 0; i < range_end - range_start + 1; ++i) {
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_destroy_tensor(libxsmm_filter[i]));
  }

  if (buffer_format == 'R') {
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_destroy_tensor(libxsmm_act[0]));
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_destroy_tensor(libxsmm_act[1]));

    libxsmm_free(act_ring_buffer[0]);
    libxsmm_free(act_ring_buffer[1]);
  }
  else {
    for (i = 0; i < range_end - range_start + 1; ++i) {
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_destroy_tensor(libxsmm_act[2 * i]));
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_destroy_tensor(libxsmm_act[2 * i + 1]));
      libxsmm_free(act_ring_buffer[2 * i]);
      libxsmm_free(act_ring_buffer[2 * i + 1]);
    }
  }

  /* deallocate data */
  libxsmm_free(scratch);
  for (i = 0; i < range_end - range_start + 1; ++i) {
    libxsmm_free(filter[i]);
  }

  free(libxsmm_filter);
  free(filter);
  free(act_ring_buffer);
  free(per_layer_time);
#if defined(USE_CORE_PERF_COUNTERS)
  free(a);
#endif
  free(libxsmm_act);
  free(libxsmm_conv_layers);
  fclose(f);
  /* some empty lines at the end */
  printf("\n\n\n");

  return global_status;
}
