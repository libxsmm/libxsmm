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
# include <omp.h>
#endif

/* include c-based dnn library */
#include "../common/dnn_common.h"

#define CHKERR_LIBXSMM_DNN(A) { const int chkerr_libxsmm_dnn_ = A; if (LIBXSMM_DNN_SUCCESS != chkerr_libxsmm_dnn_) { \
  fprintf(stderr, "%s\n", libxsmm_dnn_get_error(chkerr_libxsmm_dnn_)); global_status = chkerr_libxsmm_dnn_; } \
}

int main(int argc, char* argv[])
{
  float *naive_input, *naive_output, *naive_input_add, *naive_delinput_add, *naive_delinput, *naive_deloutput;
  float *naive_input_pad, *naive_output_pad, *naive_input_add_pad, *naive_delinput_add_pad, *naive_delinput_pad, *naive_deloutput_pad;
  float *naive_libxsmm_output, *naive_libxsmm_delinput, *naive_libxsmm_delinput_add;
  float *naive_beta, *naive_gamma, *naive_delbeta, *naive_delgamma, *naive_expectval, *naive_rcpstddev, *naive_variance;
  float *input_libxsmm, *output_libxsmm, *input_add_libxsmm, *delinput_libxsmm, *deloutput_libxsmm, *delinput_add_libxsmm;
  float *beta_libxsmm, *gamma_libxsmm, *delbeta_libxsmm, *delgamma_libxsmm, *expectval_libxsmm, *rcpstddev_libxsmm, *variance_libxsmm;
  unsigned char* relumask_libxsmm;

  int ifhp, ifwp, ofhp, ofwp, ofh, ofw;
  int stride_h, stride_w;
  naive_fusedbatchnorm_t naive_param;
  void* scratch;
  size_t scratch_size = 0;

  /* some parameters we can overwrite via cli,
     default is some inner layer of overfeat */
  int iters = 10;         /* repetitions of benchmark */
  int ifw = 14;           /* input width, "W" */
  int ifh = 20;           /* input height, "H" */
  int nImg = 32;          /* mini-batch size, "N" */
  int nFm = 256;          /* number of input feature maps, "C" */
  int stride = 1;         /* stride when accessing inputs */
  int pad_h_in = 0;       /* padding mode */
  int pad_w_in = 0;       /* padding mode */
  int pad_h_out = 0;      /* padding mode */
  int pad_w_out = 0;      /* padding mode */
  int norm_type = 0;      /* 0: full batchnorm, 1: batch scaling only */
  int fuse_type = 0;      /* 0: nothing fused, 1: relu fused, 2: elementwise fused, 3: relu and elementwise fused */
  char type = 'A';        /* 'A': ALL, 'F': FP, 'B': BP, 'U', WU */
  char format = 'L';

  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 1 : atof(env_check));

#if defined(_OPENMP)
  int nThreads = omp_get_max_threads(); /* number of threads */
#else
  int nThreads = 1; /* number of threads */
#endif

  unsigned long long l_start, l_end;
  double l_total = 0.0;
  double gb = 0.0;
  double gib = 0.0;
  int i;
  int relu_no_match;

  libxsmm_dnn_fusedbatchnorm_desc fusedbatchnorm_desc;
  libxsmm_dnn_fusedbatchnorm* libxsmm_handle;
  libxsmm_dnn_tensor*  libxsmm_input;
  libxsmm_dnn_tensor*  libxsmm_delinput;
  libxsmm_dnn_tensor*  libxsmm_output;
  libxsmm_dnn_tensor*  libxsmm_deloutput;
  libxsmm_dnn_tensor*  libxsmm_input_add;
  libxsmm_dnn_tensor*  libxsmm_delinput_add;
  libxsmm_dnn_tensor*  libxsmm_beta;
  libxsmm_dnn_tensor*  libxsmm_gamma;
  libxsmm_dnn_tensor*  libxsmm_delbeta;
  libxsmm_dnn_tensor*  libxsmm_delgamma;
  libxsmm_dnn_tensor*  libxsmm_expectval;
  libxsmm_dnn_tensor*  libxsmm_rcpstddev;
  libxsmm_dnn_tensor*  libxsmm_variance;
  libxsmm_dnn_tensor*  libxsmm_relumask;
  libxsmm_dnn_tensor_datalayout* libxsmm_layout;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status = LIBXSMM_DNN_SUCCESS;

  libxsmm_matdiff_info norms_fwd, norms_bwd, diff;
  libxsmm_matdiff_clear(&norms_fwd);
  libxsmm_matdiff_clear(&norms_bwd);
  libxsmm_matdiff_clear(&diff);

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("Usage: %s iters inpWidth inpHeight nImg nFm pad_w_in pad_h_in pad_w_out pad_h_out stride type format\n", argv[0]);
    return 0;
  }
  libxsmm_rng_set_seed(1);

  /* reading new values from cli */
  i = 1;
  if (argc > i) iters      = atoi(argv[i++]);
  if (argc > i) ifw        = atoi(argv[i++]);
  if (argc > i) ifh        = atoi(argv[i++]);
  if (argc > i) nImg       = atoi(argv[i++]);
  if (argc > i) nFm        = atoi(argv[i++]);
  if (argc > i) pad_w_in   = atoi(argv[i++]);
  if (argc > i) pad_h_in   = atoi(argv[i++]);
  if (argc > i) pad_w_out  = atoi(argv[i++]);
  if (argc > i) pad_h_out  = atoi(argv[i++]);
  if (argc > i) stride     = atoi(argv[i++]);
  if (argc > i) norm_type  = atoi(argv[i++]);
  if (argc > i) fuse_type  = atoi(argv[i++]);
  if (argc > i) type       = *(argv[i++]);

  if (type != 'A' && type != 'F' && type != 'B') {
    printf("type needs to be 'A' (All), 'F' (FP only), 'B' (BP only)\n");
    return -1;
  }
  if ((norm_type != 0) && (norm_type != 1)) {
    printf("norm type needs to be 0 or 1\n");
    return -1;
  }
  if ((fuse_type < 0) || (fuse_type > 5)) {
    printf("fuse type needs to be 0, 1, 2, 3, 4 or 5\n");
    return -1;
  }

  stride_w = stride;
  stride_h = stride;

  /* deriving some values for naive code */
  ofh  = ifh/stride_h;
  ofw  = ifw/stride_w;
  ifhp = ifh + 2 * pad_h_in;
  ifwp = ifw + 2 * pad_w_in;
  ofhp = ofh + 2 * pad_h_out;
  ofwp = ofw + 2 * pad_w_out;

  /* set struct for naive convolution */
  naive_param.N = nImg;
  naive_param.C = nFm;
  naive_param.H = ifh;
  naive_param.W = ifw;
  naive_param.stride_h = stride_h;
  naive_param.stride_w = stride_w;
  naive_param.norm_type = norm_type;
  naive_param.fuse_type = fuse_type;

#if defined(__SSE3__)
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
#endif

  /* print some summary */
  printf("##########################################\n");
  printf("#          Setting Up (Common)           #\n");
  printf("##########################################\n");
  printf("PARAMS: W:%d  H:%d  N:%d  C:%d  P:%d  Q:%d  STRIDE:%d\n", ifw, ifh, nImg, nFm, ofh, ofw, stride);
  printf("PARAMS: ITERS:%d", iters); if (LIBXSMM_FEQ(0, check)) printf("  Threads:%d\n", nThreads); else printf("\n");
  printf(" InImg %dx%d Padded (%dx%d)\n", ifh, ifw, ifhp, ifwp);
  printf("OutImg %dx%d Padded (%dx%d)\n", ofh, ofw, ofhp, ofwp);
  printf("SIZE Input  (MB): %10.2f MiB\n", (double)(nImg*nFm*ifhp*ifwp*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Output (MB): %10.2f MiB\n", (double)(nImg*nFm*ofhp*ofwp*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Input   (1): %10.2f MiB\n", (double)(1*nFm*ifhp*ifwp*  sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Output  (1): %10.2f MiB\n", (double)(1*nFm*ofhp*ofwp*  sizeof(float))/(1024.0*1024.0) );
#if defined(USE_OVERWRITE)
  printf("Using Overwrite Option\n");
#endif

  /* allocate data */
  naive_input                = (float*)libxsmm_aligned_malloc( nImg*nFm*ifh *ifw *sizeof(float), 2097152);
  naive_input_add            = (float*)libxsmm_aligned_malloc( nImg*nFm*ifh *ifw *sizeof(float), 2097152);
  naive_delinput             = (float*)libxsmm_aligned_malloc( nImg*nFm*ifh *ifw *sizeof(float), 2097152);
  naive_delinput_add         = (float*)libxsmm_aligned_malloc( nImg*nFm*ifh *ifw *sizeof(float), 2097152);
  naive_output               = (float*)libxsmm_aligned_malloc( nImg*nFm*ofh *ofw *sizeof(float), 2097152);
  naive_deloutput            = (float*)libxsmm_aligned_malloc( nImg*nFm*ofh *ofw *sizeof(float), 2097152);

  naive_input_pad            = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
  naive_input_add_pad        = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
  naive_delinput_pad         = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
  naive_delinput_add_pad     = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
  naive_output_pad           = (float*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(float), 2097152);
  naive_deloutput_pad        = (float*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(float), 2097152);

  naive_libxsmm_output       = (float*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(float), 2097152);
  naive_libxsmm_delinput     = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
  naive_libxsmm_delinput_add = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);

  input_libxsmm              = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
  delinput_libxsmm           = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
  input_add_libxsmm          = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
  delinput_add_libxsmm       = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
  output_libxsmm             = (float*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(float), 2097152);
  deloutput_libxsmm          = (float*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(float), 2097152);

  naive_beta                 = (float*)libxsmm_aligned_malloc( nFm*               sizeof(float), 2097152);
  naive_gamma                = (float*)libxsmm_aligned_malloc( nFm*               sizeof(float), 2097152);
  naive_delbeta              = (float*)libxsmm_aligned_malloc( nFm*               sizeof(float), 2097152);
  naive_delgamma             = (float*)libxsmm_aligned_malloc( nFm*               sizeof(float), 2097152);
  naive_expectval            = (float*)libxsmm_aligned_malloc( nFm*               sizeof(float), 2097152);
  naive_rcpstddev            = (float*)libxsmm_aligned_malloc( nFm*               sizeof(float), 2097152);
  naive_variance             = (float*)libxsmm_aligned_malloc( nFm*               sizeof(float), 2097152);

  beta_libxsmm               = (float*)libxsmm_aligned_malloc( nFm*               sizeof(float), 2097152);
  gamma_libxsmm              = (float*)libxsmm_aligned_malloc( nFm*               sizeof(float), 2097152);
  delbeta_libxsmm            = (float*)libxsmm_aligned_malloc( nFm*               sizeof(float), 2097152);
  delgamma_libxsmm           = (float*)libxsmm_aligned_malloc( nFm*               sizeof(float), 2097152);
  expectval_libxsmm          = (float*)libxsmm_aligned_malloc( nFm*               sizeof(float), 2097152);
  rcpstddev_libxsmm          = (float*)libxsmm_aligned_malloc( nFm*               sizeof(float), 2097152);
  variance_libxsmm           = (float*)libxsmm_aligned_malloc( nFm*               sizeof(float), 2097152);

  relumask_libxsmm           = (unsigned char*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(unsigned char), 2097152);

  /* initialize data */
  init_buf( naive_input, nImg*nFm*ifh*ifw, 0, 0 );
  copy_internal_nchw( naive_input_pad , naive_input, nImg, nFm, ifh, ifw, pad_h_in, pad_w_in );
  init_buf( naive_delinput, nImg*nFm*ifh*ifw, 0, 0 );
  copy_internal_nchw( naive_delinput_pad, naive_delinput, nImg, nFm, ifh, ifw, pad_h_in, pad_w_in );
  init_buf( naive_input_add, nImg*nFm*ifh*ifw, 0, 0 );
  copy_internal_nchw( naive_input_add_pad, naive_input_add, nImg, nFm, ifh, ifw, pad_h_in, pad_w_in );
  init_buf( naive_delinput_add, nImg*nFm*ifh*ifw, 0, 0 );
  copy_internal_nchw( naive_delinput_add_pad, naive_delinput_add, nImg, nFm, ifh, ifw, pad_h_in, pad_w_in );
  init_buf( naive_output, nImg*nFm*ofh*ofw, 0, 0  );
  copy_internal_nchw( naive_output_pad, naive_output, nImg, nFm, ofh, ofw, pad_h_out, pad_w_out );
  init_buf( naive_deloutput, nImg*nFm*ofh*ofw, 0, 0 );
  copy_internal_nchw( naive_deloutput_pad, naive_deloutput, nImg, nFm, ofh, ofw, pad_h_out, pad_w_out );

  set_zeropad_nchw(naive_input_pad,        nImg, nFm, ifhp, ifwp, pad_h_in,  pad_w_in);
  set_zeropad_nchw(naive_delinput_pad,     nImg, nFm, ifhp, ifwp, pad_h_in,  pad_w_in);
  set_zeropad_nchw(naive_input_add_pad,    nImg, nFm, ifhp, ifwp, pad_h_in,  pad_w_in);
  set_zeropad_nchw(naive_delinput_add_pad, nImg, nFm, ifhp, ifwp, pad_h_in,  pad_w_in);
  set_zeropad_nchw(naive_output_pad,       nImg, nFm, ofhp, ofwp, pad_h_out, pad_w_out);
  set_zeropad_nchw(naive_deloutput_pad,    nImg, nFm, ofhp, ofwp, pad_h_out, pad_w_out);

  init_buf(naive_beta,      nFm, 0, 0);
  init_buf(naive_gamma,     nFm, 0, 0);
  init_buf(naive_delbeta,   nFm, 0, 0);
  init_buf(naive_delgamma,  nFm, 0, 0);
  init_buf(naive_expectval, nFm, 0, 0);
  init_buf(naive_rcpstddev, nFm, 0, 0);
  init_buf(naive_variance,  nFm, 0, 0);
  copy_buf(naive_beta,      beta_libxsmm,      nFm);
  copy_buf(naive_gamma,     gamma_libxsmm,     nFm);
  copy_buf(naive_delbeta,   delbeta_libxsmm,   nFm);
  copy_buf(naive_delgamma,  delgamma_libxsmm,  nFm);
  copy_buf(naive_expectval, expectval_libxsmm, nFm);
  copy_buf(naive_rcpstddev, rcpstddev_libxsmm, nFm);
  copy_buf(naive_variance,  variance_libxsmm, nFm);

  if (LIBXSMM_NEQ(0, check)) {
    printf("##########################################\n");
    printf("#         Computing Reference ...        #\n");
    printf("##########################################\n");
    if (type == 'A' || type == 'F') {
      naive_fusedbatchnorm_fp(&naive_param, naive_input, naive_output, naive_input_add, naive_beta, naive_gamma, naive_expectval, naive_rcpstddev, naive_variance);
    }
    if (type == 'A' || type == 'B') {
      naive_fusedbatchnorm_bp(&naive_param, naive_input, naive_delinput, naive_output, naive_deloutput, naive_delinput_add,
                       naive_beta, naive_delbeta, naive_gamma, naive_delgamma, naive_expectval, naive_rcpstddev);
    }
    printf("##########################################\n");
    printf("#      Computing Reference ... done      #\n");
    printf("##########################################\n");
  }

  if (format == 'A' || format == 'L') {
    printf("\n");
    printf("##########################################\n");
    printf("#      Setting Up  (custom-Storage)      #\n");
    printf("##########################################\n");

    /* setup LIBXSMM handle */
    fusedbatchnorm_desc.partN = nImg;
    fusedbatchnorm_desc.fullN = nImg;
    fusedbatchnorm_desc.C = nFm;
    fusedbatchnorm_desc.H = ifh;
    fusedbatchnorm_desc.W = ifw;
    fusedbatchnorm_desc.u = stride_h;
    fusedbatchnorm_desc.v = stride_w;
    fusedbatchnorm_desc.pad_h_in = pad_h_in;
    fusedbatchnorm_desc.pad_w_in = pad_w_in;
    fusedbatchnorm_desc.pad_h_out = pad_h_out;
    fusedbatchnorm_desc.pad_w_out = pad_w_out;
    fusedbatchnorm_desc.threads = nThreads;
    fusedbatchnorm_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    fusedbatchnorm_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
    fusedbatchnorm_desc.datatype_stats = LIBXSMM_DNN_DATATYPE_F32;
    fusedbatchnorm_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
    fusedbatchnorm_desc.fuse_order = LIBXSMM_DNN_FUSEDBN_ORDER_BN_ELTWISE_RELU;
    if ( norm_type == 0 ) {
      if ( fuse_type == 0 ) {
        fusedbatchnorm_desc.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN;
      } else if ( fuse_type == 1 ) {
        fusedbatchnorm_desc.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN_RELU;
      } else if ( fuse_type == 2 ) {
        fusedbatchnorm_desc.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN_ELTWISE;
      } else if ( fuse_type == 3 ) {
        fusedbatchnorm_desc.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN_ELTWISE_RELU;
      } else if ( fuse_type == 4 ) {
        fusedbatchnorm_desc.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN_RELU_WITH_MASK;
      } else if ( fuse_type == 5 ) {
        fusedbatchnorm_desc.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN_ELTWISE_RELU_WITH_MASK;
      } else {
        /* shouldn't happen */
        return -1;
      }
    } else {
      if ( fuse_type == 0 ) {
        fusedbatchnorm_desc.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE;
      } else if ( fuse_type == 1 ) {
        fusedbatchnorm_desc.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_RELU;
      } else if ( fuse_type == 2 ) {
        fusedbatchnorm_desc.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE;
      } else if ( fuse_type == 3 ) {
        fusedbatchnorm_desc.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE_RELU;
      } else if ( fuse_type == 4 ) {
        fusedbatchnorm_desc.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_RELU_WITH_MASK;
      } else if ( fuse_type == 5 ) {
        fusedbatchnorm_desc.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE_RELU_WITH_MASK;
      } else {
        /* shouldn't happen */
        return -1;
      }
    }

    libxsmm_handle = libxsmm_dnn_create_fusedbatchnorm( fusedbatchnorm_desc, &status );
    CHKERR_LIBXSMM_DNN( status );

    /* setup LIBXSMM buffers */
    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    printf("inner activation blocking: %i\n", libxsmm_layout->dim_size[0] );
    libxsmm_input  = libxsmm_dnn_link_tensor( libxsmm_layout, input_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_delinput  = libxsmm_dnn_link_tensor( libxsmm_layout, delinput_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT_ADD, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_input_add  = libxsmm_dnn_link_tensor( libxsmm_layout, input_add_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT_ADD, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_delinput_add  = libxsmm_dnn_link_tensor( libxsmm_layout, delinput_add_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_output  = libxsmm_dnn_link_tensor( libxsmm_layout, output_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_deloutput  = libxsmm_dnn_link_tensor( libxsmm_layout, deloutput_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_BETA, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_beta  = libxsmm_dnn_link_tensor( libxsmm_layout, beta_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_CHANNEL_BETA, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_delbeta  = libxsmm_dnn_link_tensor( libxsmm_layout, delbeta_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_gamma  = libxsmm_dnn_link_tensor( libxsmm_layout, gamma_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_delgamma  = libxsmm_dnn_link_tensor( libxsmm_layout, delgamma_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_CHANNEL_EXPECTVAL, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_expectval  = libxsmm_dnn_link_tensor( libxsmm_layout, expectval_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_CHANNEL_RCPSTDDEV, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_rcpstddev  = libxsmm_dnn_link_tensor( libxsmm_layout, rcpstddev_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_CHANNEL_VARIANCE, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_variance  = libxsmm_dnn_link_tensor( libxsmm_layout, variance_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RELU_MASK, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_relumask  = libxsmm_dnn_link_tensor( libxsmm_layout, relumask_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    /* copy in data to LIBXSMM format */
    /* we can also use the layout functions and set the data on our
       own external to the library */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_input,        (void*)naive_input_pad,        LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_output,       (void*)naive_output_pad,       LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_input_add,    (void*)naive_input_add_pad,    LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_delinput,     (void*)naive_delinput_pad,     LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_deloutput,    (void*)naive_deloutput_pad,    LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_delinput_add, (void*)naive_delinput_add_pad, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );

    /* bind buffers and filter to handle */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle, libxsmm_input,        LIBXSMM_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle, libxsmm_delinput,     LIBXSMM_DNN_GRADIENT_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle, libxsmm_output,       LIBXSMM_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle, libxsmm_deloutput,    LIBXSMM_DNN_GRADIENT_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle, libxsmm_input_add,    LIBXSMM_DNN_REGULAR_INPUT_ADD ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle, libxsmm_delinput_add, LIBXSMM_DNN_GRADIENT_INPUT_ADD ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle, libxsmm_beta,         LIBXSMM_DNN_REGULAR_CHANNEL_BETA ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle, libxsmm_gamma,        LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle, libxsmm_delbeta,      LIBXSMM_DNN_GRADIENT_CHANNEL_BETA ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle, libxsmm_delgamma,     LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle, libxsmm_expectval,    LIBXSMM_DNN_CHANNEL_EXPECTVAL ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle, libxsmm_rcpstddev,    LIBXSMM_DNN_CHANNEL_RCPSTDDEV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle, libxsmm_variance,     LIBXSMM_DNN_CHANNEL_VARIANCE ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle, libxsmm_relumask,     LIBXSMM_DNN_RELU_MASK ) );

    /* let's allocate and bind scratch */
    scratch_size = libxsmm_dnn_fusedbatchnorm_get_scratch_size( libxsmm_handle, &status );
    CHKERR_LIBXSMM_DNN( status );
    scratch = libxsmm_aligned_scratch( scratch_size, 2097152 );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_scratch( libxsmm_handle, scratch ) );
    /* set scratch to bogus to make sure that libxsmm takes care of zeroing internally */
    init_buf( (float*)scratch, scratch_size/4, 0, 0 );

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
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
      }
      /* copy out data */
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_output, (void*)naive_libxsmm_output, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
      copy_internal_nchw( naive_output_pad, naive_output, nImg, nFm, ofh, ofw, pad_h_out, pad_w_out);

      /* compare */
      printf("rcpstddev:\n");
      libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, nFm, 1, naive_rcpstddev, rcpstddev_libxsmm, 0, 0);
      printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_fwd);
      printf("variance:\n");
      libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, nFm, 1, naive_variance, variance_libxsmm, 0, 0);
      printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_fwd);
      printf("expected value:\n");
      libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, nFm, 1, naive_expectval, expectval_libxsmm, 0, 0);
      printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_fwd);
      printf("output:\n");
      libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, nImg*nFm*ofhp*ofwp, 1, naive_output_pad, naive_libxsmm_output, 0, 0);
      printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_fwd);

      /* let's check ReLU positions */
      relu_no_match = 0;
      for ( i = 0; i < nImg*nFm*ofhp*ofwp; ++i ) {
        if ( (naive_output_pad[i] == 0.0f && naive_libxsmm_output[i] != 0.0f) ||
             (naive_output_pad[i] != 0.0f && naive_libxsmm_output[i] == 0.0f)    ) {
          relu_no_match++;
        }
      }
      printf("ReLU mismatch count: %i\n", relu_no_match );
    }

    if ( (type == 'A' || type == 'B') && LIBXSMM_NEQ(0, check) ) {
      printf("##########################################\n");
      printf("#   Correctness - BWD (custom-Storage)   #\n");
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
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid ) );
      }

      /* copy out data */
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_delinput,     (void*)naive_libxsmm_delinput,     LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_delinput_add, (void*)naive_libxsmm_delinput_add, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
      copy_internal_nchw( naive_delinput_pad, naive_delinput, nImg, nFm, ifh, ifw, pad_h_in, pad_w_in);
      copy_internal_nchw( naive_delinput_add_pad, naive_delinput_add, nImg, nFm, ifh, ifw, pad_h_in, pad_w_in);

      /* compare */
      printf("delinput_add:\n");
      libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, nImg*nFm*ifhp*ifwp, 1, naive_delinput_add_pad, naive_libxsmm_delinput_add, 0, 0);
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_bwd);
      printf("delbeta:\n");
      libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, nFm, 1, naive_delbeta, delbeta_libxsmm, 0, 0);
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_bwd);
      printf("delgamma:\n");
      libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, nFm, 1, naive_delgamma, delgamma_libxsmm, 0, 0);
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_bwd);
      printf("delinput:\n");
      libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, nImg*nFm*ifhp*ifwp, 1, naive_delinput_pad, naive_libxsmm_delinput, 0, 0);
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_bwd);
    }

    if (type == 'A' || type == 'F') {
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
        for (i = 0; i < iters; ++i) {
          libxsmm_dnn_fusedbatchnorm_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);

      gb = ((double)nImg*(double)nFm*(((double)ifh*(double)ifw) + ((double)ofh*(double)ofw))*(double)sizeof(float)*(double)iters) / (1000*1000*1000);
      gib = ((double)nImg*(double)nFm*(((double)ifh*(double)ifw) + ((double)ofh*(double)ofw))*(double)sizeof(float)*(double)iters) / (1024*1024*1024);

      printf("GB  = %.5g\n", gb/(double)iters);
      printf("GiB  = %.5g\n", gib/(double)iters);
      printf("fp time = %.5g\n", ((double)(l_total/iters)));
      printf("GB/s  = %.5g\n", gb/l_total);
      printf("GiB/s  = %.5g\n", gib/l_total);

      printf("PERFDUMP,FP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nFm,
        ifw, ifh, stride, pad_w_in, pad_h_in, pad_w_out, pad_h_out, ((double)(l_total/iters)), gb/l_total, gib/l_total, norms_fwd.l1_ref, norms_fwd.l1_tst,
        norms_fwd.l2_abs, norms_fwd.l2_rel, norms_fwd.linf_abs, norms_fwd.linf_rel, norms_fwd.normf_rel);
    }

    if ( (type == 'A' || type == 'B') ) {
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
        for (i = 0; i < iters; ++i) {
          libxsmm_dnn_fusedbatchnorm_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);

      gb = (2.0*(double)nImg*(double)nFm*(((double)ifh*(double)ifw) + (2.0*(double)ofh*(double)ofw))*(double)sizeof(float)*(double)iters) / (1000*1000*1000);
      gib = (2.0*(double)nImg*(double)nFm*(((double)ifh*(double)ifw) + (2.0*(double)ofh*(double)ofw))*(double)sizeof(float)*(double)iters) / (1024*1024*1024);

      printf("GB  = %.5g\n", gb/(double)iters);
      printf("GiB  = %.5g\n", gib/(double)iters);
      printf("fp time = %.5g\n", ((double)(l_total/iters)));
      printf("GB/s  = %.5g\n", gb/l_total);
      printf("GiB/s  = %.5g\n", gib/l_total);

      printf("PERFDUMP,BP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nFm,
        ifw, ifh, stride, pad_w_in, pad_h_in, pad_w_out, pad_h_out, ((double)(l_total/iters)), gb/l_total, gib/l_total, norms_bwd.l1_ref, norms_bwd.l1_tst,
        norms_bwd.l2_abs, norms_bwd.l2_rel, norms_bwd.linf_abs, norms_bwd.linf_rel, norms_bwd.normf_rel);
    }

    /* clean-up */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_release_scratch( libxsmm_handle ) );
    libxsmm_free(scratch);
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT_ADD ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT_ADD ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_BETA ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_CHANNEL_BETA ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_release_tensor( libxsmm_handle, LIBXSMM_DNN_CHANNEL_EXPECTVAL ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_release_tensor( libxsmm_handle, LIBXSMM_DNN_CHANNEL_RCPSTDDEV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_release_tensor( libxsmm_handle, LIBXSMM_DNN_CHANNEL_VARIANCE ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_release_tensor( libxsmm_handle, LIBXSMM_DNN_RELU_MASK) );

    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_input ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_delinput ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_output ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_deloutput ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_input_add ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_delinput_add ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_beta ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_delbeta ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_gamma ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_delgamma ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_expectval ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_rcpstddev ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_variance ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_relumask ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_fusedbatchnorm( libxsmm_handle ) );
  }

  /* deallocate data */
  libxsmm_free(naive_input);
  libxsmm_free(naive_input_add);
  libxsmm_free(naive_output);
  libxsmm_free(naive_delinput);
  libxsmm_free(naive_delinput_add);
  libxsmm_free(naive_deloutput);
  libxsmm_free(naive_input_pad);
  libxsmm_free(naive_input_add_pad);
  libxsmm_free(naive_output_pad);
  libxsmm_free(naive_delinput_pad);
  libxsmm_free(naive_delinput_add_pad);
  libxsmm_free(naive_deloutput_pad);
  libxsmm_free(naive_beta);
  libxsmm_free(naive_gamma);
  libxsmm_free(naive_delbeta);
  libxsmm_free(naive_delgamma);
  libxsmm_free(naive_expectval);
  libxsmm_free(naive_rcpstddev);
  libxsmm_free(naive_variance);
  libxsmm_free(naive_libxsmm_output);
  libxsmm_free(naive_libxsmm_delinput);
  libxsmm_free(naive_libxsmm_delinput_add);
  libxsmm_free(input_libxsmm);
  libxsmm_free(input_add_libxsmm);
  libxsmm_free(output_libxsmm);
  libxsmm_free(delinput_libxsmm);
  libxsmm_free(delinput_add_libxsmm);
  libxsmm_free(deloutput_libxsmm);
  libxsmm_free(beta_libxsmm);
  libxsmm_free(gamma_libxsmm);
  libxsmm_free(delbeta_libxsmm);
  libxsmm_free(delgamma_libxsmm);
  libxsmm_free(expectval_libxsmm);
  libxsmm_free(rcpstddev_libxsmm);
  libxsmm_free(variance_libxsmm);
  libxsmm_free(relumask_libxsmm);

  { const char *const env_check_scale = getenv("CHECK_SCALE");
    const double check_scale = LIBXSMM_ABS(0 == env_check_scale ? 1.0 : atof(env_check_scale));
    if (LIBXSMM_NEQ(0, check) && (check < 100.0 * check_scale * diff.normf_rel) && (global_status == LIBXSMM_DNN_SUCCESS)) {
      fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
      exit(EXIT_FAILURE);
    }
  }

  /* some empty lines at the end */
  printf("\n\n\n");

  return global_status;
}

