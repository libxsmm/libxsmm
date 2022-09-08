/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/

#include <libxsmm_dnn.h>
#include <dnn_common.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

int main(int argc, char* argv[])
{
  float *naive_input, *naive_output, *naive_output_save, *naive_filter, *naive_filter_wu, *naive_output_bp, *naive_output_wu, *naive_libxsmm_output;
  float *naive_libxsmm_input, *naive_libxsmm_filter, *naive_input_save, *naive_filter_save, *naive_filter_kcrs;
  float *input_nhwc, *output_nhwc, *filter_rsck, *dinput_nhwc, *doutput_nhwc, *dfilter_rsck, *naive_output_nhwc, *naive_input_nhwc;
  float *input_libxsmm, *filter_libxsmm, *output_libxsmm, *dinput_libxsmm, *dfilter_libxsmm, *doutput_libxsmm, *filtertr_libxsmm;
  float *bias_libxsmm;

  libxsmm_bfloat16 *input_libxsmm_bf16, *filter_libxsmm_bf16, *output_libxsmm_bf16, *dinput_libxsmm_bf16, *dfilter_libxsmm_bf16, *doutput_libxsmm_bf16, *filtertr_libxsmm_bf16;
  libxsmm_bfloat16 *bias_libxsmm_bf16;

  unsigned char *relumask_libxsmm = NULL;
  libxsmm_dnn_conv_eltwise_fuse my_fuse = LIBXSMM_DNN_CONV_ELTWISE_FUSE_NONE;
  libxsmm_dnn_conv_config libxsmm_dnn_conv_cfg;

  int ifhp, ifwp, ofhp, ofwp, ofh, ofw;
  int stride_h, stride_w, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out, pad_w_out;
  int bc = 64, bk = 64;
  int overwrite_output = 1;
  int avoid_bwd_wt_trans = 0;
  naive_conv_t naive_param;
  void* scratch = NULL;
  int fuse_type = 0;
  int zero_output_rims_fwd = 0;
  libxsmm_datatype cnn_dtype = LIBXSMM_DATATYPE_F32;

  /* some parameters we can overwrite via cli,
     default is some inner layer of overfeat */
  int iters = 10;         /* repetitions of benchmark */
  int ifw = 14;           /* input width, "W" */
  int ifh = 20;           /* input height, "H" */
  int nImg = 32;          /* mini-batch size, "N" */
  int nIfm = 256;         /* number of input feature maps, "C" */
  int nOfm = 512;         /* number of output feature maps, "K" */
  int kh = 3;             /* filter height, "R" */
  int kw = 3;             /* filter width, "S" */
  int padh = 0;           /* padding in input, height */
  int padw = 0;           /* padding in input, width */
  int stride = 1;         /* stride when accessing inputs */
  int padding_mode = 0;   /* padding mode */
  char type = 'A';        /* 'A': ALL, 'F': FP, 'B': BP, 'U', WU */
  char format = 'A';      /* 'A': ALL, 'L': LIBXSMM, 'T': Tensorflow, 'M', Mixed */

  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 1 : atof(env_check));

#if defined(_OPENMP)
  int nThreads = omp_get_max_threads(); /* number of threads */
#else
  int nThreads = 1; /* number of threads */
#endif

  unsigned long long l_start, l_end;
  double l_total = 0.0;
  double flops = 0.0;
  int i;
  int prec_bf16 = 0;

  libxsmm_matdiff_info norms_fwd, norms_bwd, norms_upd, diff;
  libxsmm_matdiff_clear(&norms_fwd);
  libxsmm_matdiff_clear(&norms_bwd);
  libxsmm_matdiff_clear(&norms_upd);
  libxsmm_matdiff_clear(&diff);

  naive_input = NULL;
  naive_output = NULL;
  naive_output_save = NULL;
  naive_filter = NULL;
  naive_filter_wu = NULL;
  naive_output_bp = NULL;
  naive_output_wu = NULL;
  naive_libxsmm_output = NULL;
  naive_libxsmm_input = NULL;
  naive_libxsmm_filter = NULL;
  naive_input_save = NULL;
  naive_filter_save = NULL;
  naive_filter_kcrs = NULL;
  input_nhwc = NULL;
  output_nhwc = NULL;
  filter_rsck = NULL;
  dinput_nhwc = NULL;
  doutput_nhwc = NULL;
  dfilter_rsck = NULL;
  naive_output_nhwc = NULL;
  naive_input_nhwc = NULL;
  input_libxsmm = NULL;
  filter_libxsmm = NULL;
  output_libxsmm = NULL;
  dinput_libxsmm = NULL;
  dfilter_libxsmm = NULL;
  doutput_libxsmm = NULL;
  filtertr_libxsmm = NULL;
  bias_libxsmm = NULL;
  input_libxsmm_bf16 = NULL;
  filter_libxsmm_bf16 = NULL;
  output_libxsmm_bf16 = NULL;
  dinput_libxsmm_bf16 = NULL;
  dfilter_libxsmm_bf16 = NULL;
  doutput_libxsmm_bf16 = NULL;
  filtertr_libxsmm_bf16 = NULL;
  bias_libxsmm_bf16 = NULL;

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("Usage: %s iters inpWidth inpHeight nImg nIfm nOfm kw kh pad stride type format padding_mode\n", argv[0]);
    return 0;
  }
  libxsmm_rng_set_seed(1);

  /* reading new values from cli */
  i = 1;
  if (argc > i) iters      = atoi(argv[i++]);
  if (argc > i) ifw        = atoi(argv[i++]);
  if (argc > i) ifh        = atoi(argv[i++]);
  if (argc > i) nImg       = atoi(argv[i++]);
  if (argc > i) nIfm       = atoi(argv[i++]);
  if (argc > i) nOfm       = atoi(argv[i++]);
  if (argc > i) kw         = atoi(argv[i++]);
  if (argc > i) kh         = atoi(argv[i++]);
  if (argc > i) padw       = atoi(argv[i++]);
  if (argc > i) padh       = atoi(argv[i++]);
  if (argc > i) stride     = atoi(argv[i++]);
  if (argc > i) type       = *(argv[i++]);
  if (argc > i) format     = *(argv[i++]);
  if (argc > i) padding_mode = atoi(argv[i++]);
  if (argc > i) fuse_type = atoi(argv[i++]);
  if (argc > i) bc = atoi(argv[i++]);
  if (argc > i) bk = atoi(argv[i++]);
  if (argc > i) prec_bf16 = atoi(argv[i++]);
  if (argc > i) overwrite_output   = atoi(argv[i++]);
  if (argc > i) avoid_bwd_wt_trans = atoi(argv[i++]);
  if (argc > i) zero_output_rims_fwd = atoi(argv[i++]);

  LIBXSMM_UNUSED(format);

  if ( fuse_type == 0 ) {
    my_fuse = LIBXSMM_DNN_CONV_ELTWISE_FUSE_NONE;
  } else if ( fuse_type == 1 ) {
    my_fuse = LIBXSMM_DNN_CONV_ELTWISE_FUSE_BIAS;
  } else if ( fuse_type == 2 ) {
    my_fuse = LIBXSMM_DNN_CONV_ELTWISE_FUSE_RELU;
  } else if ( fuse_type == 3 ) {
    my_fuse = LIBXSMM_DNN_CONV_ELTWISE_FUSE_BIAS_RELU;
  } else {
    /* cannot happen */
  }

  if (type != 'A' && type != 'F' && type != 'B' && type != 'U') {
    printf("type needs to be 'A' (All), 'F' (FP only), 'B' (BP only), 'U' (WU only)\n");
    return 0;
  }

  if ((type != 'F') && (zero_output_rims_fwd > 0)) {
    printf("Supporting zero_output rims only for fwd pass...\n");
    return 0;
  }

  stride_w = stride;
  stride_h = stride;
  pad_w = padw;
  pad_h = padh;

  if (0 == padding_mode) {
    pad_h_in = 0;
    pad_w_in = 0;
    pad_h_out = 0;
    pad_w_out = 0;
  }
  else {
    /* TODO: change "1" to "0" if "padding_mode = -1" is acknowledged */
    if (1 < padding_mode) pad_w = padding_mode;
    pad_h_in = pad_h;
    pad_w_in = pad_w;
    pad_h_out = pad_h;
    pad_w_out = pad_w;
  }

  if (zero_output_rims_fwd > 0) {
    pad_h_out = zero_output_rims_fwd;
    pad_w_out = zero_output_rims_fwd;
  }

  /* deriving some values for naive code */
  ofh = (ifh + 2 * pad_h - kh) / stride_h + 1;
  ofw = (ifw + 2 * pad_w - kw) / stride_w + 1;
  ifhp = ifh + 2 * pad_h_in;
  ifwp = ifw + 2 * pad_w_in;
  ofhp = ofh + 2 * pad_h_out;
  ofwp = ofw + 2 * pad_w_out;

  /* set struct for naive convolution */
  naive_param.nImg = nImg;
  naive_param.nIfm = nIfm;
  naive_param.nOfm = nOfm;
  naive_param.ifhp = ifhp;
  naive_param.ifwp = ifwp;
  naive_param.ofhp = ofhp;
  naive_param.ofwp = ofwp;
  naive_param.ifh = ifh;
  naive_param.ifw = ifw;
  naive_param.ofh = ofh;
  naive_param.ofw = ofw;
  naive_param.pad_h = pad_h;
  naive_param.pad_w = pad_w;
  naive_param.pad_h_in = pad_h_in;
  naive_param.pad_w_in = pad_w_in;
  naive_param.pad_h_out = pad_h_out;
  naive_param.pad_w_out = pad_w_out;
  naive_param.kh = kh;
  naive_param.kw = kw;
  naive_param.stride_h = stride_h;
  naive_param.stride_w = stride_w;

#if defined(__SSE3__)
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
#endif

/* print some summary */
  printf("##########################################\n");
  printf("#          Setting Up (Common)           #\n");
  printf("##########################################\n");
  printf("PARAMS: W:%d  H:%d  N:%d  C:%d  K:%d  R:%d  S:%d  P:%d  Q:%d  STRIDE:%d\n", ifw, ifh, nImg, nIfm, nOfm, kw, kh, ofh, ofw, stride);
  printf("PARAMS: ITERS:%d", iters); if (LIBXSMM_FEQ(0, check)) printf("  Threads:%d\n", nThreads); else printf("\n");
  printf(" InImg %dx%d Padded (%dx%d)\n", ifh, ifw, ifhp, ifwp);
  printf("OutImg %dx%d Padded (%dx%d)\n", ofh, ofw, ofhp, ofwp);
  printf("SIZE Input  (MB): %10.2f MiB\n", (double)((size_t)nImg*nIfm*ifhp*ifwp*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Output (MB): %10.2f MiB\n", (double)((size_t)nImg*nOfm*ofhp*ofwp*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Input   (1): %10.2f MiB\n", (double)((size_t)1*nIfm*ifhp*ifwp*   sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Output  (1): %10.2f MiB\n", (double)((size_t)1*nOfm*ofhp*ofwp*   sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Weight     : %10.2f MiB\n", (double)((size_t)nIfm*nOfm*kw*kh*    sizeof(float))/(1024.0*1024.0) );
  if (overwrite_output > 0 ) {
    printf("Using Overwrite Option\n");
  }
  if (avoid_bwd_wt_trans > 0) {
    printf("External transpose of weights\n");
  }
  if ( fuse_type == 0 ) {
    my_fuse = LIBXSMM_DNN_CONV_ELTWISE_FUSE_NONE;
  } else if ( fuse_type == 1 ) {
    printf("Fusion of bias\n");
  } else if ( fuse_type == 2 ) {
    printf("Fusion of relu\n");
  } else if ( fuse_type == 4 ) {
    printf("Fusion of bias+relu\n");
  } else {
    /* cannot happen */
  }
  /* allocate data */
  naive_input           = (float*)libxsmm_aligned_malloc( (size_t)nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  naive_input_save      = (float*)libxsmm_aligned_malloc( (size_t)nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  naive_output          = (float*)libxsmm_aligned_malloc( (size_t)nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  naive_output_save     = (float*)libxsmm_aligned_malloc( (size_t)nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  naive_output_bp       = (float*)libxsmm_aligned_malloc( (size_t)nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  naive_output_wu       = (float*)libxsmm_aligned_malloc( (size_t)nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  naive_libxsmm_output  = (float*)libxsmm_aligned_malloc( (size_t)nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  naive_libxsmm_input   = (float*)libxsmm_aligned_malloc( (size_t)nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  naive_filter          = (float*)libxsmm_aligned_malloc( (size_t)nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  naive_filter_save     = (float*)libxsmm_aligned_malloc( (size_t)nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  naive_filter_wu       = (float*)libxsmm_aligned_malloc( (size_t)nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  naive_filter_kcrs     = (float*)libxsmm_aligned_malloc( (size_t)nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  naive_libxsmm_filter  = (float*)libxsmm_aligned_malloc( (size_t)nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  input_nhwc            = (float*)libxsmm_aligned_malloc( (size_t)nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  doutput_nhwc          = (float*)libxsmm_aligned_malloc( (size_t)nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  dinput_nhwc           = (float*)libxsmm_aligned_malloc( (size_t)nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  output_nhwc           = (float*)libxsmm_aligned_malloc( (size_t)nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  naive_output_nhwc     = (float*)libxsmm_aligned_malloc( (size_t)nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  naive_input_nhwc      = (float*)libxsmm_aligned_malloc( (size_t)nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  filter_rsck           = (float*)libxsmm_aligned_malloc( (size_t)nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  dfilter_rsck          = (float*)libxsmm_aligned_malloc( (size_t)nOfm*nIfm*kh*kw*    sizeof(float), 2097152);

  input_libxsmm         = (float*)libxsmm_aligned_malloc( (size_t)nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  filter_libxsmm        = (float*)libxsmm_aligned_malloc( (size_t)nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  output_libxsmm        = (float*)libxsmm_aligned_malloc( (size_t)nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  dinput_libxsmm        = (float*)libxsmm_aligned_malloc( (size_t)nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  dfilter_libxsmm       = (float*)libxsmm_aligned_malloc( (size_t)nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  doutput_libxsmm       = (float*)libxsmm_aligned_malloc( (size_t)nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  filtertr_libxsmm      = (float*)libxsmm_aligned_malloc( (size_t)nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  bias_libxsmm          = (float*)libxsmm_aligned_malloc( (size_t)nOfm*               sizeof(float), 2097152);
  if ( prec_bf16 > 0 ) {
    /* Allocate bf16 counterparts */
    input_libxsmm_bf16         = (libxsmm_bfloat16*)libxsmm_aligned_malloc( (size_t)nImg*nIfm*ifhp*ifwp*sizeof(libxsmm_bfloat16), 2097152);
    filter_libxsmm_bf16        = (libxsmm_bfloat16*)libxsmm_aligned_malloc( (size_t)nOfm*nIfm*kh*kw*    sizeof(libxsmm_bfloat16), 2097152);
    output_libxsmm_bf16        = (libxsmm_bfloat16*)libxsmm_aligned_malloc( (size_t)nImg*nOfm*ofhp*ofwp*sizeof(libxsmm_bfloat16), 2097152);
    dinput_libxsmm_bf16        = (libxsmm_bfloat16*)libxsmm_aligned_malloc( (size_t)nImg*nIfm*ifhp*ifwp*sizeof(libxsmm_bfloat16), 2097152);
    dfilter_libxsmm_bf16       = (libxsmm_bfloat16*)libxsmm_aligned_malloc( (size_t)nOfm*nIfm*kh*kw*    sizeof(libxsmm_bfloat16), 2097152);
    doutput_libxsmm_bf16       = (libxsmm_bfloat16*)libxsmm_aligned_malloc( (size_t)nImg*nOfm*ofhp*ofwp*sizeof(libxsmm_bfloat16), 2097152);
    filtertr_libxsmm_bf16      = (libxsmm_bfloat16*)libxsmm_aligned_malloc( (size_t)nOfm*nIfm*kh*kw*    sizeof(libxsmm_bfloat16), 2097152);
    bias_libxsmm_bf16          = (libxsmm_bfloat16*)libxsmm_aligned_malloc( (size_t)nOfm*               sizeof(libxsmm_bfloat16), 2097152);
  }

  /* initialize data */
  if (padding_mode == 0 ) {
    init_buf(naive_input,          nImg*nIfm*ifhp*ifwp, 0, 0);
  } else {
    float *naive_input_tmp = (float*)libxsmm_aligned_malloc( (size_t)nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
    init_buf(naive_input_tmp,          nImg*nIfm*ifh*ifw, 0, 0);
    copy_internal_nchw( naive_input , naive_input_tmp, nImg, nIfm, ifh, ifw, pad_h, pad_w);
    libxsmm_free(naive_input_tmp);
  }

  if (padding_mode == 0 ) {
    init_buf(naive_output_bp,      nImg*nOfm*ofhp*ofwp, 0, 0);
    init_buf(naive_output_wu,      nImg*nOfm*ofhp*ofwp, 0, 0);
  } else {
    float *naive_output_bp_tmp = (float*)libxsmm_aligned_malloc( (size_t)nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
    float *naive_output_wu_tmp = (float*)libxsmm_aligned_malloc( (size_t)nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
    init_buf(naive_output_bp_tmp,      nImg*nOfm*ofh*ofw, 0, 0);
    copy_internal_nchw( naive_output_bp , naive_output_bp_tmp, nImg, nOfm, ofh, ofw, pad_h, pad_w);
    init_buf(naive_output_wu_tmp,      nImg*nOfm*ofh*ofw, 0, 0);
    copy_internal_nchw( naive_output_wu , naive_output_wu_tmp, nImg, nOfm, ofh, ofw, pad_h, pad_w);
    libxsmm_free(naive_output_bp_tmp);
    libxsmm_free(naive_output_wu_tmp);
  }
  set_zeropad_nchw(naive_input, nImg, nIfm, ifhp, ifwp, pad_h_in, pad_w_in);
  set_zeropad_nchw(naive_output_bp, nImg, nOfm, ofhp, ofwp, pad_h_out, pad_w_out);
  set_zeropad_nchw(naive_output_wu, nImg, nOfm, ofhp, ofwp, pad_h_out, pad_w_out);

  copy_buf(naive_input, naive_input_save, nImg*nIfm*ifhp*ifwp);
  zero_buf(naive_output_save,    nImg*nOfm*ofhp*ofwp);

  if (padding_mode == 0 ) {
    init_buf(naive_output,       nImg*nOfm*ofhp*ofwp, 0, 0);
  } else {
    float *naive_output_tmp = (float*)libxsmm_aligned_malloc( (size_t)nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
    init_buf(naive_output_tmp,       nImg*nOfm*ofh*ofw, 0, 0);
    libxsmm_free(naive_output_tmp);
  }
  set_zeropad_nchw(naive_output, nImg, nOfm, ofhp, ofwp, pad_h_out, pad_w_out);

  copy_buf(naive_output, naive_output_save, nImg*nOfm*ofhp*ofwp);
  zero_buf(naive_libxsmm_output, nImg*nOfm*ofhp*ofwp);
  zero_buf(naive_libxsmm_input,  nImg*nIfm*ifhp*ifwp);
  init_buf(naive_filter,         nOfm*nIfm*kh*kw, 0, 0);
  copy_buf(naive_filter, naive_filter_wu, nOfm*nIfm*kh*kw);
  zero_buf(naive_libxsmm_filter, nOfm*nIfm*kh*kw);
  naive_copy_NCHW_to_NHWC(naive_input, input_nhwc, nImg, ifhp, ifwp, nIfm);
  zero_buf(output_nhwc,          nImg*nOfm*ofhp*ofwp);
  zero_buf(naive_output_nhwc,    nImg*nOfm*ofhp*ofwp);
  zero_buf(naive_input_nhwc,     nImg*nIfm*ifhp*ifwp);
  naive_copy_KCRS_to_RSCK(naive_filter, filter_rsck, kh, kw, nIfm, nOfm);
  init_buf(bias_libxsmm,         nOfm, 0, 0);

  /* first touch LIBXSMM */
  zero_buf( input_libxsmm    , nImg*nIfm*ifhp*ifwp );
  zero_buf( filter_libxsmm   , nOfm*nIfm*kh*kw );
  zero_buf( output_libxsmm   , nImg*nOfm*ofhp*ofwp );
  zero_buf( dinput_libxsmm   , nImg*nIfm*ifhp*ifwp );
  zero_buf( dfilter_libxsmm  , nOfm*nIfm*kh*kw );
  zero_buf( doutput_libxsmm  , nImg*nOfm*ofhp*ofwp );
  zero_buf( filtertr_libxsmm , nOfm*nIfm*kh*kw );

  if (prec_bf16 > 0) {
    libxsmm_rne_convert_fp32_bf16( naive_input,      input_libxsmm_bf16,     nImg*nIfm*ifhp*ifwp );
    libxsmm_convert_bf16_f32( input_libxsmm_bf16, naive_input, nImg*nIfm*ifhp*ifwp );
    libxsmm_rne_convert_fp32_bf16( naive_input_save,      input_libxsmm_bf16,     nImg*nIfm*ifhp*ifwp );
    libxsmm_convert_bf16_f32( input_libxsmm_bf16, naive_input_save, nImg*nIfm*ifhp*ifwp );
    libxsmm_rne_convert_fp32_bf16( naive_filter,     filter_libxsmm_bf16,     nOfm*nIfm*kh*kw );
    libxsmm_convert_bf16_f32( filter_libxsmm_bf16, naive_filter, nOfm*nIfm*kh*kw );
    libxsmm_rne_convert_fp32_bf16( naive_output,     output_libxsmm_bf16,     nImg*nOfm*ofhp*ofwp );
    libxsmm_convert_bf16_f32( output_libxsmm_bf16, naive_output, nImg*nOfm*ofhp*ofwp );
    libxsmm_rne_convert_fp32_bf16( naive_output_bp,     output_libxsmm_bf16,     nImg*nOfm*ofhp*ofwp );
    libxsmm_convert_bf16_f32( output_libxsmm_bf16, naive_output_bp, nImg*nOfm*ofhp*ofwp );
    libxsmm_rne_convert_fp32_bf16( bias_libxsmm,     bias_libxsmm_bf16,     nOfm );
    libxsmm_convert_bf16_f32( bias_libxsmm_bf16, bias_libxsmm, nOfm );
  }

  if (LIBXSMM_NEQ(0, check)) {
    printf("##########################################\n");
    printf("#         Computing Reference ...        #\n");
    printf("##########################################\n");
    if (type == 'A' || type == 'F') {
      if (overwrite_output > 0 ) {
        zero_buf(naive_output,    nImg*nOfm*ofhp*ofwp);
      }
      naive_fused_conv_fp(&naive_param, naive_input, naive_output, naive_filter, bias_libxsmm, (libxsmm_blasint)my_fuse);
    }
    if ( (type == 'A' || type == 'B') && (nIfm > 3) ) {
      if (overwrite_output > 0 ) {
        zero_buf(naive_input,         nImg*nIfm*ifhp*ifwp);
      }
      naive_conv_bp(&naive_param, naive_input, naive_output_bp, naive_filter, naive_input_save);
    }
    if (type == 'A' || type == 'U') {
      /* NB: We reuse naive_input_save for weight update because the input should not
       * have been modified between forward propagation and weight update; it further
       * helps in exploiting reuse to converted data. */
      if (overwrite_output > 0 ) {
        zero_buf(naive_filter_wu,          nOfm*nIfm*kh*kw);
      }
      naive_conv_wu(&naive_param, naive_input_save, naive_output_wu, naive_filter_wu);
    }
    printf("##########################################\n");
    printf("#      Computing Reference ... done      #\n");
    printf("##########################################\n");
  }

  printf("\n");
  printf("##########################################\n");
  printf("#      Setting Up  (custom-Storage)      #\n");
  printf("##########################################\n");

  if (prec_bf16 > 0) {
    cnn_dtype = LIBXSMM_DATATYPE_BF16;
  }

  libxsmm_dnn_conv_cfg = setup_libxsmm_dnn_conv(cnn_dtype, cnn_dtype, nImg, ifh, ifw, nIfm, nOfm, kh, kw, stride_h, stride_w,
      pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out, pad_w_out, bc, bk, nThreads, my_fuse, overwrite_output, avoid_bwd_wt_trans, zero_output_rims_fwd);

  /* Copy input/output/weight tensors to correct format */
  tensor_copy_NCHW_to_NCHWc (naive_input_save , input_libxsmm,  nImg, nIfm, ifhp, ifwp, libxsmm_dnn_conv_cfg.ifmblock);
  tensor_copy_NCHW_to_NCHWc (naive_output_save, output_libxsmm, nImg, nOfm, ofhp, ofwp, libxsmm_dnn_conv_cfg.ofmblock);
  /* In this case put garbage in libxsmm_output rim to make sure zero riming works as expected  */
  if (zero_output_rims_fwd > 0) {
    tensor_pollute_rim_NCHWc(output_libxsmm, nImg, nOfm, ofhp, ofwp, libxsmm_dnn_conv_cfg.ofmblock, pad_h_out, pad_w_out, 42.0);
  }
  tensor_copy_KCRS_to_KCRSck(naive_filter     , filter_libxsmm, nOfm, nIfm, kh, kw, libxsmm_dnn_conv_cfg.ifmblock, libxsmm_dnn_conv_cfg.ofmblock);
  if (avoid_bwd_wt_trans > 0) {
    tensor_transpose_KCRSck_to_CKRSkc(filter_libxsmm, filtertr_libxsmm, nOfm, nIfm, kh, kw, libxsmm_dnn_conv_cfg.ifmblock, libxsmm_dnn_conv_cfg.ofmblock);
  }

  if ( (libxsmm_dnn_conv_cfg.scratch_size) > 0 ) {
    size_t alloc_size = libxsmm_dnn_conv_cfg.scratch_size;
    scratch = libxsmm_aligned_malloc( alloc_size, 2097152 );
    init_buf( (float*)(scratch), (alloc_size)/4, 0, 0 );
  }

  if (prec_bf16 > 0) {
    tensor_copy_KCRS_to_KCRSck_bf16(naive_filter,     filter_libxsmm_bf16, nOfm, nIfm, kh, kw, libxsmm_dnn_conv_cfg.ifmblock, libxsmm_dnn_conv_cfg.ofmblock);
    libxsmm_rne_convert_fp32_bf16( input_libxsmm,     input_libxsmm_bf16,     nImg*nIfm*ifhp*ifwp );
    libxsmm_rne_convert_fp32_bf16( output_libxsmm,     output_libxsmm_bf16,     nImg*nOfm*ofhp*ofwp );
    libxsmm_rne_convert_fp32_bf16( bias_libxsmm,     bias_libxsmm_bf16,     nOfm );
  }

  if ((type == 'A' || type == 'F') && LIBXSMM_NEQ(0, check)) {
    printf("##########################################\n");
    printf("#   Correctness - FWD (custom-Storage)   #\n");
    printf("##########################################\n");
    /* run LIBXSMM convolutions */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      if (prec_bf16 > 0) {
        libxsmm_dnn_conv_fwd_exec_bf16( libxsmm_dnn_conv_cfg, filter_libxsmm_bf16, input_libxsmm_bf16, output_libxsmm_bf16,
            bias_libxsmm_bf16, relumask_libxsmm, 0, tid, scratch );
      } else {
        libxsmm_dnn_conv_fwd_exec( libxsmm_dnn_conv_cfg, filter_libxsmm, input_libxsmm, output_libxsmm,
            bias_libxsmm, relumask_libxsmm, 0, tid, scratch );
      }
    }
    /* copy out data */
    if (prec_bf16 > 0) {
      libxsmm_convert_bf16_f32( output_libxsmm_bf16, output_libxsmm, nImg*nOfm*ofhp*ofwp );
    }
    tensor_copy_NCHWc_to_NCHW (output_libxsmm, naive_libxsmm_output, nImg, nOfm, ofhp, ofwp, libxsmm_dnn_conv_cfg.ofmblock);

    /* compare */
    libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, nImg*nOfm*ofhp*ofwp, 1, naive_output, naive_libxsmm_output, 0, 0);
    printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
    printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
    printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
    libxsmm_matdiff_reduce(&diff, &norms_fwd);
  }

  if (prec_bf16 > 0) {
    if (avoid_bwd_wt_trans > 0) {
      tensor_transpose_KCRSck_to_CKRSkc_bf16(filter_libxsmm, filtertr_libxsmm_bf16, nOfm, nIfm, kh, kw, libxsmm_dnn_conv_cfg.ifmblock, libxsmm_dnn_conv_cfg.ofmblock);
    }
  }

  if ( (type == 'A' || type == 'B') && (nIfm > 3) && LIBXSMM_NEQ(0, check) ) {
    printf("##########################################\n");
    printf("#   Correctness - BWD (custom-Storage)   #\n");
    printf("##########################################\n");
    /* let's do some additional init such that we can run passes standalone */
    tensor_copy_NCHW_to_NCHWc (naive_output_bp , doutput_libxsmm,  nImg, nOfm, ofhp, ofwp, libxsmm_dnn_conv_cfg.ofmblock);
    tensor_copy_NCHW_to_NCHWc (naive_input_save, dinput_libxsmm, nImg, nIfm, ifhp, ifwp, libxsmm_dnn_conv_cfg.ifmblock);
    if (prec_bf16 > 0) {
      libxsmm_rne_convert_fp32_bf16( dinput_libxsmm,     dinput_libxsmm_bf16,     nImg*nIfm*ifhp*ifwp );
      libxsmm_rne_convert_fp32_bf16( doutput_libxsmm,   doutput_libxsmm_bf16,     nImg*nOfm*ofhp*ofwp );
    }

    /* run LIBXSMM convolutions */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      if (prec_bf16 > 0) {
        libxsmm_dnn_conv_bwd_exec_bf16( libxsmm_dnn_conv_cfg, filter_libxsmm_bf16, filtertr_libxsmm_bf16,  doutput_libxsmm_bf16, dinput_libxsmm_bf16,
          relumask_libxsmm, 0, tid, scratch );
      } else {
        libxsmm_dnn_conv_bwd_exec( libxsmm_dnn_conv_cfg, filter_libxsmm, filtertr_libxsmm,  doutput_libxsmm, dinput_libxsmm,
          relumask_libxsmm, 0, tid, scratch );
      }
    }

    /* copy out data */
    if (prec_bf16 > 0) {
      libxsmm_convert_bf16_f32( dinput_libxsmm_bf16, dinput_libxsmm, nImg*nIfm*ifhp*ifwp );
    }
    tensor_copy_NCHWc_to_NCHW (dinput_libxsmm, naive_libxsmm_input, nImg, nIfm, ifhp, ifwp, libxsmm_dnn_conv_cfg.ifmblock);

    /* compare */
    libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, nImg*nIfm*ifhp*ifwp, 1, naive_input, naive_libxsmm_input, 0, 0);
    printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
    printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
    printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
    libxsmm_matdiff_reduce(&diff, &norms_bwd);
  }

  if ((type == 'A' || type == 'U') && LIBXSMM_NEQ(0, check)) {
    printf("##########################################\n");
    printf("#   Correctness - UPD (custom-Storage)   #\n");
    printf("##########################################\n");
    tensor_copy_NCHW_to_NCHWc (naive_input_save , input_libxsmm,  nImg, nIfm, ifhp, ifwp, libxsmm_dnn_conv_cfg.ifmblock);
    tensor_copy_NCHW_to_NCHWc (naive_output_wu,   doutput_libxsmm, nImg, nOfm, ofhp, ofwp, libxsmm_dnn_conv_cfg.ofmblock);
    tensor_copy_KCRS_to_KCRSck(naive_filter     , dfilter_libxsmm, nOfm, nIfm, kh, kw, libxsmm_dnn_conv_cfg.ifmblock, libxsmm_dnn_conv_cfg.ofmblock);
    if (prec_bf16 > 0) {
      libxsmm_rne_convert_fp32_bf16( input_libxsmm,     input_libxsmm_bf16,     nImg*nIfm*ifhp*ifwp );
      libxsmm_rne_convert_fp32_bf16( doutput_libxsmm,   doutput_libxsmm_bf16,     nImg*nOfm*ofhp*ofwp );
      tensor_copy_KCRS_to_KCRSck_bf16(naive_filter,     dfilter_libxsmm_bf16, nOfm, nIfm, kh, kw, libxsmm_dnn_conv_cfg.ifmblock, libxsmm_dnn_conv_cfg.ofmblock);
    }

    /* run LIBXSMM convolutions */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      if (prec_bf16 > 0) {
        libxsmm_dnn_conv_upd_exec_bf16( libxsmm_dnn_conv_cfg, input_libxsmm_bf16, doutput_libxsmm_bf16, dfilter_libxsmm_bf16,
            NULL, 0, tid, scratch );
      } else {
        libxsmm_dnn_conv_upd_exec( libxsmm_dnn_conv_cfg, input_libxsmm, doutput_libxsmm, dfilter_libxsmm,
            NULL, 0, tid, scratch );
      }
    }
    if (prec_bf16 > 0) {
      tensor_copy_KCRSck_vnni_to_norm_f32( dfilter_libxsmm_bf16, dfilter_libxsmm, nOfm, nIfm, kh, kw, libxsmm_dnn_conv_cfg.ifmblock, libxsmm_dnn_conv_cfg.ofmblock );
    }
    tensor_copy_KCRSck_to_KCRS( dfilter_libxsmm, naive_libxsmm_filter, nOfm, nIfm, kh, kw, libxsmm_dnn_conv_cfg.ifmblock, libxsmm_dnn_conv_cfg.ofmblock);

    /* compare */
    libxsmm_matdiff(&norms_upd, LIBXSMM_DATATYPE_F32, nOfm*nIfm*kh*kw, 1, naive_filter_wu, naive_libxsmm_filter, 0, 0);
    printf("L1 reference  : %.25g\n", norms_upd.l1_ref);
    printf("L1 test       : %.25g\n", norms_upd.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_upd.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_upd.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_upd.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_upd.linf_rel);
    printf("Check-norm    : %.24f\n", norms_upd.normf_rel);
    libxsmm_matdiff_reduce(&diff, &norms_upd);
  }

  if ((type == 'A' || type == 'F') && LIBXSMM_NEQ(0, check)) {
    printf("##########################################\n");
    printf("#   Performance - FWD (custom-Storage)   #\n");
    printf("##########################################\n");
    /* run LIBXSMM convolution for performance */
    l_start = libxsmm_timer_tick();
#if defined(_OPENMP)
#     pragma omp parallel private(i)
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      if (prec_bf16 > 0) {
        for (i = 0; i < iters; ++i) {
          libxsmm_dnn_conv_fwd_exec_bf16( libxsmm_dnn_conv_cfg, filter_libxsmm_bf16, input_libxsmm_bf16, output_libxsmm_bf16,
              bias_libxsmm_bf16, relumask_libxsmm, 0, tid, scratch );
        }
      } else {
        for (i = 0; i < iters; ++i) {
          libxsmm_dnn_conv_fwd_exec( libxsmm_dnn_conv_cfg, filter_libxsmm, input_libxsmm, output_libxsmm,
              bias_libxsmm, relumask_libxsmm, 0, tid, scratch );
        }
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

    printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
    printf("fp time = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

    printf("PERFDUMP,FP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
        ifw, ifh, kw, kh, stride, padw, padh, ((double)(l_total/iters)), (flops*1e-9)/l_total, norms_fwd.l1_ref, norms_fwd.l1_tst,
        norms_fwd.l2_abs, norms_fwd.l2_rel, norms_fwd.linf_abs, norms_fwd.linf_rel, norms_fwd.normf_rel);
  }

  if ( (type == 'A' || type == 'B') && (nIfm > 3) ) {
    printf("##########################################\n");
    printf("#   Performance - BWD (custom-Storage)   #\n");
    printf("##########################################\n");
    /* run LIBXSMM convolution for performance */
    l_start = libxsmm_timer_tick();

#if defined(_OPENMP)
#     pragma omp parallel  private(i)
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif

      if (prec_bf16 > 0) {
        for (i = 0; i < iters; ++i) {
          libxsmm_dnn_conv_bwd_exec_bf16( libxsmm_dnn_conv_cfg, filter_libxsmm_bf16, filtertr_libxsmm_bf16,  doutput_libxsmm_bf16, dinput_libxsmm_bf16,
            relumask_libxsmm, 0, tid, scratch );
        }
      } else {
        for (i = 0; i < iters; ++i) {
          libxsmm_dnn_conv_bwd_exec( libxsmm_dnn_conv_cfg, filter_libxsmm, filtertr_libxsmm,  doutput_libxsmm, dinput_libxsmm,
            relumask_libxsmm, 0, tid, scratch );
        }
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

    printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
    printf("bp time = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

    printf("PERFDUMP,BP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
        ifw, ifh, kw, kh, stride, padw, padh, ((double)(l_total/iters)), (flops*1e-9)/l_total, norms_bwd.l1_ref, norms_bwd.l1_tst,
        norms_bwd.l2_abs, norms_bwd.l2_rel, norms_bwd.linf_abs, norms_bwd.linf_rel, norms_bwd.normf_rel);
  }

  if (type == 'A' || type == 'U') {
    printf("##########################################\n");
    printf("#   Performance - UPD (custom-Storage)   #\n");
    printf("##########################################\n");
    /* run LIBXSMM convolution for performance */
    l_start = libxsmm_timer_tick();

#if defined(_OPENMP)
#     pragma omp parallel private(i)
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      if (prec_bf16 > 0) {
        for (i = 0; i < iters; ++i) {
          libxsmm_dnn_conv_upd_exec_bf16( libxsmm_dnn_conv_cfg, input_libxsmm_bf16, doutput_libxsmm_bf16, dfilter_libxsmm_bf16,
              NULL, 0, tid, scratch );
        }
      } else {
        for (i = 0; i < iters; ++i) {
          libxsmm_dnn_conv_upd_exec( libxsmm_dnn_conv_cfg, input_libxsmm, doutput_libxsmm, dfilter_libxsmm,
             NULL, 0, tid, scratch );
        }
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

    printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
    printf("wu time = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

    printf("PERFDUMP,WU,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
        ifw, ifh, kw, kh, stride, padw, padh, ((double)(l_total/iters)), (flops*1e-9)/l_total, norms_upd.l1_ref, norms_upd.l1_tst,
        norms_upd.l2_abs, norms_upd.l2_rel, norms_upd.linf_abs, norms_upd.linf_rel, norms_upd.normf_rel);
  }

  /* deallocate data */
  libxsmm_free(naive_input);
  libxsmm_free(naive_input_save);
  libxsmm_free(naive_output);
  libxsmm_free(naive_output_save);
  libxsmm_free(naive_output_bp);
  libxsmm_free(naive_output_wu);
  libxsmm_free(naive_libxsmm_output);
  libxsmm_free(naive_libxsmm_input);
  libxsmm_free(naive_filter);
  libxsmm_free(naive_filter_save);
  libxsmm_free(naive_filter_wu);
  libxsmm_free(naive_filter_kcrs);
  libxsmm_free(naive_libxsmm_filter);
  libxsmm_free(input_nhwc);
  libxsmm_free(output_nhwc);
  libxsmm_free(dinput_nhwc);
  libxsmm_free(doutput_nhwc);
  libxsmm_free(naive_output_nhwc);
  libxsmm_free(naive_input_nhwc);
  libxsmm_free(filter_rsck);
  libxsmm_free(dfilter_rsck);
  libxsmm_free(input_libxsmm);
  libxsmm_free(filter_libxsmm);
  libxsmm_free(output_libxsmm);
  libxsmm_free(dinput_libxsmm);
  libxsmm_free(dfilter_libxsmm);
  libxsmm_free(doutput_libxsmm);
  libxsmm_free(filtertr_libxsmm);
  libxsmm_free(bias_libxsmm);
  if ( prec_bf16 > 0 ) {
    libxsmm_free(input_libxsmm_bf16);
    libxsmm_free(filter_libxsmm_bf16);
    libxsmm_free(output_libxsmm_bf16);
    libxsmm_free(dinput_libxsmm_bf16);
    libxsmm_free(dfilter_libxsmm_bf16);
    libxsmm_free(doutput_libxsmm_bf16);
    libxsmm_free(filtertr_libxsmm_bf16);
    libxsmm_free(bias_libxsmm_bf16);
  }
  if ( (libxsmm_dnn_conv_cfg.scratch_size) > 0 ) {
    libxsmm_free(scratch);
  }

  destroy_libxsmm_dnn_conv(&libxsmm_dnn_conv_cfg );

  { const char *const env_check_scale = getenv("CHECK_SCALE");
    const double check_scale = LIBXSMM_ABS(0 == env_check_scale ? 100.0 : atof(env_check_scale));
    if (LIBXSMM_NEQ(0, check) && (check < 100.0 * check_scale * diff.normf_rel)) {
      fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
      exit(EXIT_FAILURE);
    }
  }

  /* some empty lines at the end */
  printf("\n\n\n");

  return 0;
}

