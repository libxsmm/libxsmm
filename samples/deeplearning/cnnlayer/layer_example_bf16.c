/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke, Hans Pabst, Dhiraj Kalamkar,
 * Ankush Mandal (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <libxsmm_intrinsics_x86.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

#define USE_OVERWRITE
/*#define USE_BWD_NO_FILTER_TRANSPOSE_OVERWRITE*/

/* include c-based dnn library */
#include "../common/dnn_common.h"

#define CHKERR_LIBXSMM_DNN(A) { const int chkerr_libxsmm_dnn_ = A; if (LIBXSMM_DNN_SUCCESS != chkerr_libxsmm_dnn_) { \
  fprintf(stderr, "%s\n", libxsmm_dnn_get_error(chkerr_libxsmm_dnn_)); global_status = chkerr_libxsmm_dnn_; } \
}

int main(int argc, char* argv[])
{
  float *naive_input, *naive_input_save, *naive_output_save, *naive_filter, *naive_output, *naive_output_bp, *naive_output_fp, *naive_input_bp , *naive_filter_wu, *naive_input_tmp, *naive_libxsmm_output_f32, *naive_libxsmm_input_f32 ,*naive_libxsmm_filter_f32;
  libxsmm_bfloat16 *naive_input_bf16, *naive_input_bp_bf16, *naive_filter_bf16, *naive_output_bf16, *naive_output_bp_bf16, *naive_filter_wu_bf16;
  libxsmm_bfloat16 *input_libxsmm, *filter_libxsmm, *filtertr_libxsmm, *output_libxsmm, *naive_libxsmm_output, *naive_libxsmm_input, *naive_libxsmm_filter, *dinput_libxsmm, *doutput_libxsmm, *dfilter_libxsmm;
  int ifhp, ifwp, ofhp, ofwp, ofh, ofw;
  int stride_h, stride_w, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out, pad_w_out;
  naive_conv_t naive_param;
  void* scratch;
  size_t scratch_size;

  /* some parameters we can overwrite via cli,
     default is some inner layer of overfeat */
  int iters = 10;         /* repetitions of benchmark */
  int ifw = 14;           /* input width, "W" */
  int ifh = 18;           /* input height, "H" */
  int nImg = 32;          /* mini-batch size, "N" */
  int nIfm = 256;         /* number of input feature maps, "C" */
  int nOfm = 512;         /* number of output feature maps, "K" */
  int kh = 3;             /* filter height, "R" */
  int kw = 3;             /* filter width, "S" */
  int padh = 1;           /* padding in input, height */
  int padw = 1;           /* padding in input, width */
  int stride = 1;         /* stride when accessing inputs */
  char type = 'A';        /* 'A': ALL, 'F': FP, 'B': BP, 'U', WU */
  char format = 'L';

  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 1 : atof(env_check));

#if defined(_OPENMP)
  int nThreads = omp_get_max_threads();       /* number of threads */
#else
  int nThreads = 1;       /* number of threads */
#endif
  int padding_mode = 0;   /* padding mode */

  unsigned long long l_start, l_end;
  double l_total = 0.0;
  double lpOps = 0.0; /* number of low precision operations */
  int i;

  libxsmm_dnn_conv_desc conv_desc;
  libxsmm_dnn_layer* libxsmm_handle;
  libxsmm_dnn_tensor* libxsmm_input;
  libxsmm_dnn_tensor* libxsmm_output;
  libxsmm_dnn_tensor* libxsmm_filter;
  libxsmm_dnn_tensor* libxsmm_filter_tr;
  libxsmm_dnn_tensor* libxsmm_dinput;
  libxsmm_dnn_tensor* libxsmm_doutput;
  libxsmm_dnn_tensor* libxsmm_dfilter;

  libxsmm_dnn_tensor_datalayout* libxsmm_layout;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status = LIBXSMM_DNN_SUCCESS;

  libxsmm_matdiff_info norms_fwd, norms_bwd, norms_upd, diff;
  libxsmm_matdiff_clear(&norms_fwd);
  libxsmm_matdiff_clear(&norms_bwd);
  libxsmm_matdiff_clear(&norms_upd);
  libxsmm_matdiff_clear(&diff);

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("Usage: %s iters inpWidth inpHeight nImg nIfm nOfm kw kh pad stride type padding_mode\n", argv[0]);
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

  if (type != 'A' && type != 'F' && type != 'B'&& type != 'U') {
    printf("type needs to be 'A' (All), 'F' (FP only), 'B' (BP only), 'U' (WU only)\n");
    return 0;
  }

  if (format != 'L') {
    printf("format needs to be 'L'\n");
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
  naive_param.ifh = ifh;
  naive_param.ifw = ifw;
  naive_param.ofhp = ofhp;
  naive_param.ofwp = ofwp;
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
  printf("#    Setting Up Common    #\n");
  printf("##########################################\n");
  printf("PARAMS: W:%d  H:%d  N:%d  C:%d  K:%d  R:%d  S:%d  P:%d  Q:%d  STRIDE:%d\n", ifw, ifh, nImg, nIfm, nOfm, kw, kh, ofh, ofw, stride);
  printf("PARAMS: ITERS:%d", iters); if (LIBXSMM_FEQ(0, check)) printf("  Threads:%d\n", nThreads); else printf("\n");
  printf(" InImg %dx%d Padded (%dx%d)\n", ifh, ifw, ifhp, ifwp);
  printf("OutImg %dx%d Padded (%dx%d)\n", ofh, ofw, ofhp, ofwp);
  printf("SIZE Input  (MB): %10.2f MiB\n", (double)(nImg*nIfm*ifhp*ifwp*sizeof(libxsmm_bfloat16))/(1024.0*1024.0) );
  printf("SIZE Output (MB): %10.2f MiB\n", (double)(nImg*nOfm*ofhp*ofwp*sizeof(libxsmm_bfloat16))/(1024.0*1024.0) );
  printf("SIZE Input   (1): %10.2f MiB\n", (double)(1*nIfm*ifhp*ifwp*   sizeof(libxsmm_bfloat16))/(1024.0*1024.0) );
  printf("SIZE Output  (1): %10.2f MiB\n", (double)(1*nOfm*ofhp*ofwp*   sizeof(libxsmm_bfloat16))/(1024.0*1024.0) );
  printf("SIZE Weight     : %10.2f MiB\n", (double)(nIfm*nOfm*kw*kh*    sizeof(libxsmm_bfloat16))/(1024.0*1024.0) );

  /* allocate data */
  naive_input               = (float*           )libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float           ), 2097152);
  naive_input_save          = (float*           )libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float           ), 2097152);
  naive_input_tmp           = (float*           )libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float           ), 2097152);
  naive_output              = (float*           )libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float           ), 2097152);
  naive_output_fp           = (float*           )libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float           ), 2097152);
  naive_output_save         = (float*           )libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float           ), 2097152);
  naive_output_bp           = (float*           )libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float           ), 2097152);
  naive_input_bp            = (float*           )libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float           ),   2097152);
  naive_filter              = (float*           )libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float           ), 2097152);
  naive_filter_wu           = (float*           )libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float           ), 2097152);
  naive_input_bf16          = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(libxsmm_bfloat16), 2097152);
  naive_input_bp_bf16       = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(libxsmm_bfloat16), 2097152);
  naive_output_bf16         = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(libxsmm_bfloat16), 2097152);
  naive_output_bp_bf16      = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(libxsmm_bfloat16), 2097152);
  naive_filter_bf16         = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(libxsmm_bfloat16), 2097152);
  naive_filter_wu_bf16      = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(libxsmm_bfloat16), 2097152);
  naive_libxsmm_output      = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(libxsmm_bfloat16), 2097152);
  naive_libxsmm_output_f32  = (float*           )libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float           ), 2097152);
  naive_libxsmm_input_f32   = (float*           )libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float           ), 2097152);
  naive_libxsmm_filter_f32   = (float*          )libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float           ), 2097152);
  naive_libxsmm_input       = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(libxsmm_bfloat16), 2097152);
  naive_libxsmm_filter      = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(libxsmm_bfloat16), 2097152);
  input_libxsmm             = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(libxsmm_bfloat16), 2097152);
  filter_libxsmm            = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(libxsmm_bfloat16), 2097152);
  filtertr_libxsmm          = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(libxsmm_bfloat16), 2097152);
  output_libxsmm            = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(libxsmm_bfloat16), 2097152);
  dinput_libxsmm            = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(libxsmm_bfloat16), 2097152);
  doutput_libxsmm           = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(libxsmm_bfloat16), 2097152);
  dfilter_libxsmm           = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(libxsmm_bfloat16), 2097152);

  /* initialize data */
  zero_buf(naive_input, nImg*nIfm*ifhp*ifwp);
  zero_buf(naive_output_bp, nImg*nOfm*ofhp*ofwp);
  if (padding_mode == 0 ) {
    init_buf(naive_input,          nImg*nIfm*ifhp*ifwp, 0, 0);
    init_buf(naive_output_bp,      nImg*nOfm*ofhp*ofwp, 0, 0);
  } else {
    float *naive_output_bp_tmp       = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
    init_buf(naive_input_tmp,      nImg*nIfm*ifh*ifw, 0, 0);
    init_buf(naive_output_bp_tmp,      nImg*nOfm*ofh*ofw, 0, 0);
    copy_internal_nchw( naive_input , naive_input_tmp, nImg, nIfm, ifh, ifw, pad_h, pad_w);
    copy_internal_nchw( naive_output_bp , naive_output_bp_tmp, nImg, nOfm, ofh, ofw, pad_h, pad_w);
    libxsmm_free(naive_output_bp_tmp);
  }

#if defined(USE_FUSED_RELU_BWD)
  /* Initialize some entries with zeros */
  for (i = 0; i < nImg*nIfm*ifhp*ifwp; i++ ) {
    if ( ((i%16) == 2) || ((i%16) == 3) || ((i%16) == 7) || ((i%16) == 14) ) {
      naive_input[i] = 0.0;
    }
  }
#endif

  copy_buf(naive_input, naive_input_save, nImg*nIfm*ifhp*ifwp);
  copy_buf(naive_output_bp, naive_output_save, nImg*nOfm*ofhp*ofwp);
  init_buf(naive_filter, nIfm*nOfm*kh*kw, 0, 0);
  zero_buf(naive_output_fp, nImg*nOfm*ofhp*ofwp);
  zero_buf(naive_input_bp,  nImg*nIfm*ifhp*ifwp);
  zero_buf(naive_filter_wu, nOfm*nIfm*kh*kw);
  /*zero_buf(output_libxsmm,      nImg*nOfm*ofhp*ofwp);
    zero_buf(dinput_libxsmm,      nImg*nIfm*ifhp*ifwp);
    zero_buf(naive_libxsmm_output, nImg*nOfm*ofhp*ofwp);
    zero_buf(naive_libxsmm_input,  nImg*nIfm*ifhp*ifwp);
    zero_buf(naive_libxsmm_filter, nOfm*nIfm*kh*kw);*/

  if (LIBXSMM_NEQ(0, check)) {
    printf("##########################################\n");
    printf("#         Computing Reference ...        #\n");
    printf("##########################################\n");
    /* run naive convolutions */
    if (type == 'A' || type == 'F') {
      naive_conv_fp(&naive_param, naive_input, naive_output_fp, naive_filter, NULL);
    }
    /* run naive convolutions */
    if (type == 'A' || type == 'B') {
      naive_conv_bp(&naive_param, naive_input_bp, naive_output_bp, naive_filter, naive_input_save);
    }
    /* run naive convolutions */
    if (type == 'A' || type == 'U') {
      naive_conv_wu(&naive_param, naive_input, naive_output_bp, naive_filter_wu);
    }
    printf("##########################################\n");
    printf("#      Computing Reference ... done      #\n");
    printf("##########################################\n");
  }

  /* make things bf16 */
  truncate_mask_fp32_bf16( naive_input, naive_input, nImg*nIfm*ifhp*ifwp );
  truncate_mask_fp32_bf16( naive_input_bp, naive_input_bp, nImg*nIfm*ifhp*ifwp );
  truncate_mask_fp32_bf16( naive_output_fp, naive_output_fp, nImg*nOfm*ofhp*ofwp );
  truncate_mask_fp32_bf16( naive_output_bp, naive_output_bp, nImg*nOfm*ofhp*ofwp );
  truncate_mask_fp32_bf16( naive_filter, naive_filter, nIfm*nOfm*kh*kw );
  truncate_mask_fp32_bf16( naive_filter_wu, naive_filter_wu, nIfm*nOfm*kh*kw );
  libxsmm_truncate_convert_f32_bf16( naive_input, naive_input_bf16, nImg*nIfm*ifhp*ifwp );
  libxsmm_truncate_convert_f32_bf16( naive_input_bp, naive_input_bp_bf16, nImg*nIfm*ifhp*ifwp );
  libxsmm_truncate_convert_f32_bf16( naive_output_fp, naive_output_bf16, nImg*nOfm*ofhp*ofwp );
  libxsmm_truncate_convert_f32_bf16( naive_output_bp, naive_output_bp_bf16, nImg*nOfm*ofhp*ofwp );
  libxsmm_truncate_convert_f32_bf16( naive_filter, naive_filter_bf16, nIfm*nOfm*kh*kw );
  libxsmm_truncate_convert_f32_bf16( naive_filter_wu, naive_filter_wu_bf16, nIfm*nOfm*kh*kw );

  printf("\n");
  printf("##########################################\n");
  printf("#     Setting Up    (custom-Storage)     #\n");
  printf("##########################################\n");

  /* setup LIBXSMM handle */
  conv_desc.N = nImg;
  conv_desc.C = nIfm;
  conv_desc.H = ifh;
  conv_desc.W = ifw;
  conv_desc.K = nOfm;
  conv_desc.R = kh;
  conv_desc.S = kw;
  conv_desc.u = stride_h;
  conv_desc.v = stride_w;
  conv_desc.pad_h = pad_h;
  conv_desc.pad_w = pad_w;
  conv_desc.pad_h_in = pad_h_in;
  conv_desc.pad_w_in = pad_w_in;
  conv_desc.pad_h_out = pad_h_out;
  conv_desc.pad_w_out = pad_w_out;
  conv_desc.threads = nThreads;
  conv_desc.algo = LIBXSMM_DNN_CONV_ALGO_DIRECT;
  conv_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  conv_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
#if defined(USE_BWD_NO_FILTER_TRANSPOSE_OVERWRITE)
  conv_desc.options = LIBXSMM_DNN_CONV_OPTION_BWD_NO_FILTER_TRANSPOSE_OVERWRITE;
#elif defined(USE_OVERWRITE)
  conv_desc.options = LIBXSMM_DNN_CONV_OPTION_OVERWRITE;
#endif
  conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
  conv_desc.datatype_in = LIBXSMM_DNN_DATATYPE_BF16;
  conv_desc.datatype_out = LIBXSMM_DNN_DATATYPE_BF16;

  libxsmm_handle = libxsmm_dnn_create_conv_layer( conv_desc, &status );
  CHKERR_LIBXSMM_DNN( status );

  /* setup LIBXSMM buffers and filter */
  libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_input  = libxsmm_dnn_link_tensor( libxsmm_layout,  input_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
  printf("inner activation blocking: %u\n", libxsmm_layout->dim_size[0] );
  libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

  libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dinput = libxsmm_dnn_link_tensor( libxsmm_layout, dinput_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

  libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_OUTPUT, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_output  = libxsmm_dnn_link_tensor( libxsmm_layout,  output_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

  libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_doutput = libxsmm_dnn_link_tensor( libxsmm_layout, doutput_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

  libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_filter  = libxsmm_dnn_link_tensor( libxsmm_layout,  filter_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

  libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER_TRANS, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_filter_tr  = libxsmm_dnn_link_tensor( libxsmm_layout, filtertr_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

  libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dfilter  = libxsmm_dnn_link_tensor( libxsmm_layout,  dfilter_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );


  /* copy in data to LIBXSMM format */
  /* we can also use the layout functions and set the data on our
     own external to the library, @TODO, we plan to add an example here */
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_input, (void*)naive_input_bf16, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_doutput, (void*)naive_output_bp_bf16, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_filter, (void*)naive_filter_bf16, LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_zero_tensor( libxsmm_output ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_zero_tensor( libxsmm_dfilter ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_zero_tensor( libxsmm_dinput ) );

  /* bind buffers and filter to handle */
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_input, LIBXSMM_DNN_REGULAR_INPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_dinput,     LIBXSMM_DNN_GRADIENT_INPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_output, LIBXSMM_DNN_REGULAR_OUTPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_doutput,    LIBXSMM_DNN_GRADIENT_OUTPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_filter, LIBXSMM_DNN_REGULAR_FILTER ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_dfilter, LIBXSMM_DNN_GRADIENT_FILTER ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_filter_tr,  LIBXSMM_DNN_REGULAR_FILTER_TRANS ) );

  /* let's allocate and bind scratch */
  scratch_size = libxsmm_dnn_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );
  CHKERR_LIBXSMM_DNN( status );
  scratch = libxsmm_aligned_malloc( scratch_size, 2097152 );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch ) );
  /* set scratch to bogus to make sure that libxsmm takes care of zeroing internally */
  init_buf( scratch, scratch_size/4, 0, 0 );

  if ((type == 'A' || type == 'F') && LIBXSMM_NEQ(0, check)) {
    printf("##############################################\n");
    printf("#  Check Correctness - FWD (custom-Storage)  #\n");
    printf("##############################################\n");
    /* run LIBXSMM convolutions */
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
    }
    /* copy out data */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_output, (void*)naive_libxsmm_output, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    libxsmm_convert_bf16_f32( naive_libxsmm_output, naive_libxsmm_output_f32, nImg*nOfm*ofhp*ofwp );

    /* compare */
    libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, nImg*nOfm*ofhp*ofwp, 1, naive_output_fp, naive_libxsmm_output_f32, 0, 0);
    printf("L1 reference  : %.25f\n", norms_fwd.l1_ref);
    printf("L1 test       : %.25f\n", norms_fwd.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
    printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
    libxsmm_matdiff_reduce(&diff, &norms_fwd);
  }

  if ((type == 'A' || type == 'B') && (nIfm > 3) && LIBXSMM_NEQ(0, check)) {
    printf("##############################################\n");
    printf("#  Check Correctness - BWD (custom-Storage)  #\n");
    printf("##############################################\n");
#if defined(USE_BWD_NO_FILTER_TRANSPOSE_OVERWRITE)
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_trans_reg_bf16_filter( libxsmm_handle ) );
#endif
    /* run LIBXSMM convolutions */
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid ) );
    }

    /* copy out data */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_dinput, (void*)naive_libxsmm_input, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    libxsmm_convert_bf16_f32( naive_libxsmm_input, naive_libxsmm_input_f32, nImg*nIfm*ifhp*ifwp );

    /* compare */
    libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, nImg*nIfm*ifhp*ifwp, 1, naive_input_bp, naive_libxsmm_input_f32, 0, 0);
    printf("L1 reference  : %.25f\n", norms_bwd.l1_ref);
    printf("L1 test       : %.25f\n", norms_bwd.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
    printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
    libxsmm_matdiff_reduce(&diff, &norms_bwd);
  }

  if ((type == 'A' || type == 'U') && LIBXSMM_NEQ(0, check)) {
    printf("##############################################\n");
    printf("#  Check Correctness - UPD (custom-Storage)  #\n");
    printf("##############################################\n");
    /* run LIBXSMM convolutions */
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid ) );
    }

    /* copy out data */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_dfilter, (void*)naive_libxsmm_filter, LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );
    libxsmm_convert_bf16_f32( naive_libxsmm_filter, naive_libxsmm_filter_f32, nOfm*nIfm*kh*kw);

    /* compare */
    libxsmm_matdiff(&norms_upd, LIBXSMM_DATATYPE_F32, nOfm*nIfm*kh*kw, 1, naive_filter_wu, naive_libxsmm_filter_f32, 0, 0);
    printf("L1 reference  : %.25f\n", norms_upd.l1_ref);
    printf("L1 test       : %.25f\n", norms_upd.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_upd.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_upd.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_upd.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_upd.linf_rel);
    printf("Check-norm    : %.24f\n", norms_upd.normf_rel);
    libxsmm_matdiff_reduce(&diff, &norms_upd);
  }

  if (type == 'A' || type == 'F') {
    printf("##########################################\n");
    printf("#   Performance - FWD (custom-Storage)   #\n");
    printf("##########################################\n");
    /* run LIBXSMM convolution for performance */
    l_start = libxsmm_timer_tick();
    for (i = 0; i < iters; ++i) {
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid );
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    lpOps = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

    printf("GOP  = %.5g\n", lpOps*1e-9/(double)iters);
    printf("fp time = %.5g\n", ((double)(l_total/iters)));
    printf("GOPS  = %.5g\n", (lpOps*1e-9)/l_total);

    printf("PERFDUMP,FP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
        ifw, ifh, kw, kh, stride, padw, padh, ((double)(l_total/iters)), (lpOps*1e-9)/l_total, norms_fwd.l1_ref, norms_fwd.l1_tst,
        norms_fwd.l2_abs, norms_fwd.l2_rel, norms_fwd.linf_abs, norms_fwd.linf_rel, norms_fwd.normf_rel);
  }

  if ( (type == 'A') || ((type == 'B') && (nIfm > 3)) ) {
    printf("##########################################\n");
    printf("#   Performance - BWD (custom-Storage)   #\n");
    printf("##########################################\n");
    /* run LIBXSMM convolution for performance */
    l_start = libxsmm_timer_tick();
    for (i = 0; i < iters; ++i) {
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid );
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    lpOps = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

    printf("GOP  = %.5g\n", lpOps*1e-9/(double)iters);
    printf("fp time = %.5g\n", ((double)(l_total/iters)));
    printf("GOPS  = %.5g\n", (lpOps*1e-9)/l_total);

    printf("PERFDUMP,BP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
        ifw, ifh, kw, kh, stride, padw, padh, ((double)(l_total/iters)), (lpOps*1e-9)/l_total, norms_bwd.l1_ref, norms_bwd.l1_tst,
        norms_bwd.l2_abs, norms_bwd.l2_rel, norms_bwd.linf_abs, norms_bwd.linf_rel, norms_bwd.normf_rel);
  }

  if (type == 'A' || type == 'U') {
    printf("##########################################\n");
    printf("#   Performance - UPD (custom-Storage)   #\n");
    printf("##########################################\n");
    /* run LIBXSMM convolution for performance */
    l_start = libxsmm_timer_tick();
    for (i = 0; i < iters; ++i) {
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid );
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    lpOps = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

    printf("GOP  = %.5g\n", lpOps*1e-9/(double)iters);
    printf("fp time = %.5g\n", ((double)(l_total/iters)));
    printf("GOPS  = %.5g\n", (lpOps*1e-9)/l_total);

    printf("PERFDUMP,WU,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
        ifw, ifh, kw, kh, stride, padw, padh, ((double)(l_total/iters)), (lpOps*1e-9)/l_total, norms_upd.l1_ref, norms_upd.l1_tst,
        norms_upd.l2_abs, norms_upd.l2_rel, norms_upd.linf_abs, norms_upd.linf_rel, norms_upd.normf_rel);
  }

  /* clean-up */
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL ) );
  libxsmm_free(scratch);
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_input ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_output ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_filter ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dinput ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_doutput ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dfilter ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_conv_layer( libxsmm_handle ) );

  /* deallocate data */
  libxsmm_free( naive_input );
  libxsmm_free( naive_input_save );
  libxsmm_free( naive_input_tmp );
  libxsmm_free( naive_output );
  libxsmm_free( naive_output_fp );
  libxsmm_free( naive_output_save );
  libxsmm_free( naive_output_bp );
  libxsmm_free( naive_input_bp );
  libxsmm_free( naive_filter );
  libxsmm_free( naive_filter_wu );
  libxsmm_free( naive_input_bf16 );
  libxsmm_free( naive_input_bp_bf16 );
  libxsmm_free( naive_output_bf16 );
  libxsmm_free( naive_output_bp_bf16 );
  libxsmm_free( naive_filter_bf16 );
  libxsmm_free( naive_filter_wu_bf16 );
  libxsmm_free( naive_libxsmm_output );
  libxsmm_free( naive_libxsmm_output_f32 );
  libxsmm_free( naive_libxsmm_input_f32 );
  libxsmm_free( naive_libxsmm_filter_f32 );
  libxsmm_free( naive_libxsmm_input );
  libxsmm_free( naive_libxsmm_filter );
  libxsmm_free( input_libxsmm );
  libxsmm_free( filter_libxsmm );
  libxsmm_free( output_libxsmm );
  libxsmm_free( dinput_libxsmm );
  libxsmm_free( doutput_libxsmm );
  libxsmm_free( dfilter_libxsmm );

  { const char *const env_check_scale = getenv("CHECK_SCALE");
    const double check_scale = LIBXSMM_ABS(0 == env_check_scale ? 100.0 : atof(env_check_scale));
    if (LIBXSMM_NEQ(0, check) && (check < 100.0 * check_scale * diff.normf_rel) && (global_status == LIBXSMM_DNN_SUCCESS)) {
      fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
      exit(EXIT_FAILURE);
    }
  }

  /* some empty lines at the end */
  printf("\n\n\n");

  return global_status;
}

