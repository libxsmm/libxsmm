/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Evangelos Georganas (Intel Corp.)
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
  float *naive_input,     *naive_output,     *naive_delinput,     *naive_deloutput;
  int *naive_mask;
  float *naive_input_pad, *naive_output_pad, *naive_delinput_pad, *naive_deloutput_pad;
  float *naive_libxsmm_output, *naive_libxsmm_delinput;
  float *input_libxsmm, *output_libxsmm, *delinput_libxsmm, *deloutput_libxsmm;
  int *mask_libxsmm;
  libxsmm_bfloat16 *naive_input_pad_bf16, *naive_output_pad_bf16, *naive_delinput_pad_bf16, *naive_deloutput_pad_bf16;
  libxsmm_bfloat16 *naive_libxsmm_output_bf16, *naive_libxsmm_delinput_bf16;
  libxsmm_bfloat16 *input_libxsmm_bf16, *output_libxsmm_bf16, *delinput_libxsmm_bf16, *deloutput_libxsmm_bf16;
  libxsmm_dnn_pooling_fwd_config fwd_cfg;
  libxsmm_dnn_pooling_bwd_config bwd_cfg;
  libxsmm_dnn_pooling_type pool_type_cfg;

  libxsmm_blasint ifhp, ifwp, ofhp, ofwp, ofh, ofw;
  libxsmm_blasint stride_h, stride_w;
  naive_pooling_t naive_param;
  void* scratch;
  size_t scratch_size = 0;
  libxsmm_datatype in_dt;
  libxsmm_datatype out_dt;
  libxsmm_datatype comp_dt;

  /* some parameters we can overwrite via cli,
     default is some inner layer of overfeat */
  libxsmm_blasint iters = 10;         /* repetitions of benchmark */
  libxsmm_blasint ifw = 14;           /* input width, "W" */
  libxsmm_blasint ifh = 20;           /* input height, "H" */
  libxsmm_blasint nImg = 32;          /* mini-batch size, "N" */
  libxsmm_blasint nFm = 256;          /* number of input feature maps, "C" */
  libxsmm_blasint stride = 1;         /* stride when accessing inputs */
  libxsmm_blasint kh = 2;             /* kernel size height */
  libxsmm_blasint kw = 2;             /* kernel size width */
  libxsmm_blasint pad_h = 0;          /* pad in h direction */
  libxsmm_blasint pad_w = 0;          /* pad in w direction */
  libxsmm_blasint pad_h_in = 0;       /* padding mode */
  libxsmm_blasint pad_w_in = 0;       /* padding mode */
  libxsmm_blasint pad_h_out = 0;      /* padding mode */
  libxsmm_blasint pad_w_out = 0;      /* padding mode */
  libxsmm_blasint pool_type = 0;      /* max pooling */
  char type = 'A';        /* 'A': ALL, 'F': FP, 'B': BP */
  char format = 'L';
  libxsmm_blasint bc = 64;
  libxsmm_blasint skip_mask_comp = 0;
  int prec_bf16 = 0;

  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 1 : atof(env_check));

#if defined(_OPENMP)
  libxsmm_blasint nThreads = (libxsmm_blasint)omp_get_max_threads(); /* number of threads */
#else
  libxsmm_blasint nThreads = 1; /* number of threads */
#endif

  unsigned long long l_start, l_end;
  double l_total = 0.0;
  double gb = 0.0;
  double gib = 0.0;
  libxsmm_blasint i;

  libxsmm_matdiff_info norms_fwd, norms_bwd, diff;
  libxsmm_matdiff_clear(&norms_fwd);
  libxsmm_matdiff_clear(&norms_bwd);
  libxsmm_matdiff_clear(&diff);

  naive_input = NULL;
  naive_output = NULL;
  naive_delinput = NULL;
  naive_deloutput = NULL;
  naive_input_pad = NULL;
  naive_output_pad = NULL;
  naive_delinput_pad = NULL;
  naive_deloutput_pad = NULL;
  naive_libxsmm_output = NULL;
  naive_libxsmm_delinput = NULL;
  naive_mask = NULL;
  mask_libxsmm = NULL;
  input_libxsmm = NULL;
  output_libxsmm = NULL;
  delinput_libxsmm = NULL;
  deloutput_libxsmm = NULL;
  naive_input_pad_bf16 = NULL;
  naive_output_pad_bf16 = NULL;
  naive_delinput_pad_bf16 = NULL;
  naive_deloutput_pad_bf16 = NULL;
  naive_libxsmm_output = NULL;
  naive_libxsmm_delinput = NULL;
  input_libxsmm_bf16 = NULL;
  output_libxsmm_bf16 = NULL;
  delinput_libxsmm_bf16 = NULL;
  deloutput_libxsmm_bf16 = NULL;
  naive_libxsmm_output_bf16 = NULL;
  naive_libxsmm_delinput_bf16 = NULL;

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("Usage: %s iters inpWidth inpHeight nImg nFm pad_w_in pad_h_in pad_w_out pad_h_out stride type prec_bf16 skip_mask\n", argv[0]);
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
  if (argc > i) kw         = atoi(argv[i++]);
  if (argc > i) kh         = atoi(argv[i++]);
  if (argc > i) pad_w      = atoi(argv[i++]);
  if (argc > i) pad_h      = atoi(argv[i++]);
  if (argc > i) pad_w_in   = atoi(argv[i++]);
  if (argc > i) pad_h_in   = atoi(argv[i++]);
  if (argc > i) pad_w_out  = atoi(argv[i++]);
  if (argc > i) pad_h_out  = atoi(argv[i++]);
  if (argc > i) stride     = atoi(argv[i++]);
  if (argc > i) pool_type  = atoi(argv[i++]);
  if (argc > i) type       = *(argv[i++]);
  if (argc > i) prec_bf16  = atoi(argv[i++]);
  if (argc > i) skip_mask_comp = atoi(argv[i++]);

  if (type != 'A' && type != 'F' && type != 'B') {
    printf("type needs to be 'A' (All), 'F' (FP only), 'B' (BP only)\n");
    return 0;
  }

  if (pool_type != 0 && pool_type != 1 ) {
    printf("pool_type needs to be '0' (max), '1' (avg)\n");
    return 0;
  }

  stride_w = stride;
  stride_h = stride;

  /* deriving some values for naive code */
  ofh = (ifh + 2 * pad_h - kh)/stride_h + 1;
  ofw = (ifw + 2 * pad_w - kw)/stride_w + 1;
  ifhp = ifh + 2 * pad_h_in;
  ifwp = ifw + 2 * pad_w_in;
  ofhp = ofh + 2 * pad_h_out;
  ofwp = ofw + 2 * pad_w_out;

  /* set struct for naive convolution */
  naive_param.N = nImg;
  naive_param.C = nFm;
  naive_param.H = ifh;
  naive_param.W = ifw;
  naive_param.R = kh;
  naive_param.S = kw;
  naive_param.pad_h = pad_h;
  naive_param.pad_w = pad_w;
  naive_param.stride_h = stride_h;
  naive_param.stride_w = stride_w;
  naive_param.type = pool_type;

#if defined(__SSE3__)
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
#endif

  if ( prec_bf16 == 0 ) {
    in_dt = LIBXSMM_DATATYPE_F32;
    out_dt = LIBXSMM_DATATYPE_F32;
    comp_dt = LIBXSMM_DATATYPE_F32;
  } else {
    in_dt = LIBXSMM_DATATYPE_BF16;
    out_dt = LIBXSMM_DATATYPE_BF16;
    comp_dt = LIBXSMM_DATATYPE_F32;
  }

  /* print some summary */
  printf("##########################################\n");
  printf("#          Setting Up (Common)           #\n");
  printf("##########################################\n");
  printf("PARAMS: W:%d  H:%d  N:%d  C:%d  P:%d  Q:%d  STRIDE:%d\n", ifw, ifh, nImg, nFm, ofh, ofw, stride);
  printf("PARAMS: ITERS:%d", iters); if (LIBXSMM_FEQ(0, check)) printf("  Threads:%d\n", nThreads); else printf("\n");
  printf(" InImg %dx%d Padded (%dx%d)\n", ifh, ifw, ifhp, ifwp);
  printf("OutImg %dx%d Padded (%dx%d)\n", ofh, ofw, ofhp, ofwp);
  if ( pool_type == 0 ) {
    printf("Pooling-Type: Max\n");
  } else if ( pool_type == 1 ) {
    printf("Pooling-Type: Avg\n");
  } else {
    printf("Pooling-Type: UNKNOWN!\n");
    return 0;
  }
  printf("SIZE Input  (MB): %10.2f MiB\n", (double)(nImg*nFm*ifhp*ifwp*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Output (MB): %10.2f MiB\n", (double)(nImg*nFm*ofhp*ofwp*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Input   (1): %10.2f MiB\n", (double)(1*nFm*ifhp*ifwp*  sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Output  (1): %10.2f MiB\n", (double)(1*nFm*ofhp*ofwp*  sizeof(float))/(1024.0*1024.0) );
#if defined(USE_OVERWRITE)
  printf("Using Overwrite Option\n");
#endif

  /* allocate data */
  naive_input                = (float*)libxsmm_aligned_malloc( nImg*nFm*ifh *ifw *sizeof(float), 2097152);
  naive_input_pad            = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
  naive_delinput             = (float*)libxsmm_aligned_malloc( nImg*nFm*ifh *ifw *sizeof(float), 2097152);
  naive_delinput_pad         = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
  naive_mask                 = (int*)  libxsmm_aligned_malloc( nImg*nFm*ofh *ofw *sizeof(int),   2097152);
  naive_output               = (float*)libxsmm_aligned_malloc( nImg*nFm*ofh *ofw *sizeof(float), 2097152);
  naive_output_pad           = (float*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(float), 2097152);
  naive_deloutput            = (float*)libxsmm_aligned_malloc( nImg*nFm*ofh *ofw *sizeof(float), 2097152);
  naive_deloutput_pad        = (float*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(float), 2097152);

  mask_libxsmm               = (int*)  libxsmm_aligned_malloc( nImg*nFm*ofh *ofw *sizeof(int),   2097152);
  naive_libxsmm_output       = (float*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(float), 2097152);
  naive_libxsmm_delinput     = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
  if ( prec_bf16 == 0 ) {
    input_libxsmm              = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
    delinput_libxsmm           = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
    output_libxsmm             = (float*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(float), 2097152);
    deloutput_libxsmm          = (float*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(float), 2097152);
  } else {
    naive_input_pad_bf16       = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(libxsmm_bfloat16), 2097152);
    naive_delinput_pad_bf16    = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(libxsmm_bfloat16), 2097152);
    naive_output_pad_bf16      = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(libxsmm_bfloat16), 2097152);
    naive_deloutput_pad_bf16   = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(libxsmm_bfloat16), 2097152);
    naive_libxsmm_output_bf16    = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(libxsmm_bfloat16), 2097152);
    naive_libxsmm_delinput_bf16  = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(libxsmm_bfloat16), 2097152);

    input_libxsmm_bf16         = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(libxsmm_bfloat16), 2097152);
    delinput_libxsmm_bf16      = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(libxsmm_bfloat16), 2097152);
    output_libxsmm_bf16        = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(libxsmm_bfloat16), 2097152);
    deloutput_libxsmm_bf16     = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(libxsmm_bfloat16), 2097152);
  }

  /* initialize data */
  init_buf(naive_input,          nImg*nFm*ifh*ifw, 0, 0);
  copy_internal_nchw( naive_input_pad , naive_input, nImg, nFm, ifh, ifw, pad_h_in, pad_w_in);
  init_buf(naive_delinput,          nImg*nFm*ifh*ifw, 0, 0);
  copy_internal_nchw( naive_delinput_pad , naive_delinput, nImg, nFm, ifh, ifw, pad_h_in, pad_w_in);
  init_buf(naive_output,          nImg*nFm*ofh*ofw, 0, 0);
  copy_internal_nchw( naive_output_pad , naive_output, nImg, nFm, ofh, ofw, pad_h_out, pad_w_out);
  init_buf(naive_deloutput,          nImg*nFm*ofh*ofw, 0, 0);
  copy_internal_nchw( naive_deloutput_pad , naive_deloutput, nImg, nFm, ofh, ofw, pad_h_out, pad_w_out);

  set_zeropad_nchw(naive_input_pad,   nImg, nFm, ifhp, ifwp, pad_h_in,  pad_w_in);
  set_zeropad_nchw(naive_delinput_pad, nImg, nFm, ifhp, ifwp, pad_h_in,  pad_w_in);
  set_zeropad_nchw(naive_output_pad,   nImg, nFm, ofhp, ofwp, pad_h_out, pad_w_out);
  set_zeropad_nchw(naive_deloutput_pad, nImg, nFm, ofhp, ofwp, pad_h_out, pad_w_out);

  zero_buf_int32(naive_mask,      nImg*nFm*ofh*ofw);
  zero_buf_int32(mask_libxsmm,    nImg*nFm*ofh*ofw);

  if ( prec_bf16 > 0 ) {
    libxsmm_rne_convert_fp32_bf16( naive_input_pad,     naive_input_pad_bf16,     nImg*nFm*ifhp*ifwp );
    libxsmm_rne_convert_fp32_bf16( naive_delinput_pad,  naive_delinput_pad_bf16,  nImg*nFm*ifhp*ifwp );
    libxsmm_rne_convert_fp32_bf16( naive_output_pad,    naive_output_pad_bf16,    nImg*nFm*ofhp*ofwp );
    libxsmm_rne_convert_fp32_bf16( naive_deloutput_pad, naive_deloutput_pad_bf16, nImg*nFm*ofhp*ofwp );
    zero_buf_bf16(input_libxsmm_bf16,     nImg*nFm*ifhp*ifwp);
    zero_buf_bf16(delinput_libxsmm_bf16,  nImg*nFm*ifhp*ifwp);
    zero_buf_bf16(output_libxsmm_bf16,    nImg*nFm*ofhp*ofwp);
    zero_buf_bf16(deloutput_libxsmm_bf16, nImg*nFm*ofhp*ofwp);
  } else {
    zero_buf(input_libxsmm,     nImg*nFm*ifhp*ifwp);
    zero_buf(delinput_libxsmm,  nImg*nFm*ifhp*ifwp);
    zero_buf(output_libxsmm,    nImg*nFm*ofhp*ofwp);
    zero_buf(deloutput_libxsmm, nImg*nFm*ofhp*ofwp);
  }

  if (LIBXSMM_NEQ(0, check)) {
    printf("##########################################\n");
    printf("#         Computing Reference ...        #\n");
    printf("##########################################\n");
    if (type == 'A' || type == 'F') {
      naive_pooling_fp(&naive_param, naive_input, naive_output, naive_mask);
    }
    if (type == 'A' || type == 'B') {
      naive_pooling_bp(&naive_param, naive_delinput, naive_deloutput, naive_mask);
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

    if ( pool_type == 0 ) {
      if (skip_mask_comp == 0) {
        pool_type_cfg = LIBXSMM_DNN_POOLING_TYPE_MAX;
      } else {
        pool_type_cfg = LIBXSMM_DNN_POOLING_TYPE_MAX_NOMASK;
      }
    } else if ( pool_type == 1 ) {
      pool_type_cfg = LIBXSMM_DNN_POOLING_TYPE_AVG;
    } else {
      return 0;
    }

    /* setup LIBXSMM handle */
    fwd_cfg = setup_libxsmm_dnn_pooling_fwd( nImg, nFm, ifh, ifw, kh, kw, stride_h, stride_w,
                                    pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out, pad_w_out,
                                    bc, nThreads, pool_type_cfg,
                                    in_dt, out_dt, comp_dt );

    /* setup LIBXSMM handle */
    bwd_cfg = setup_libxsmm_dnn_pooling_bwd( nImg, nFm, ifh, ifw, kh, kw, stride_h, stride_w,
                                    pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out, pad_w_out,
                                    bc, nThreads, pool_type_cfg,
                                    in_dt, out_dt, comp_dt );

    /* let's allocate and bind scratch */
    scratch_size = LIBXSMM_MAX( fwd_cfg.scratch_size, bwd_cfg.scratch_size );
    scratch = libxsmm_aligned_malloc( scratch_size, 2097152 );
    /* set scratch to bogus to make sure that libxsmm takes care of zeroing internally */
    init_buf( (float*)scratch, scratch_size/4, 0, 0 );

    /* copy tensor into the right format */
    if ( prec_bf16 == 0 ) {
      tensor_copy_NCHW_to_NCHWc(     naive_input_pad,     input_libxsmm, nImg, nFm, ifhp, ifwp, bc );
      tensor_copy_NCHW_to_NCHWc(    naive_output_pad,    output_libxsmm, nImg, nFm, ofhp, ofwp, bc );
      tensor_copy_NCHW_to_NCHWc(  naive_delinput_pad,  delinput_libxsmm, nImg, nFm, ifhp, ifwp, bc );
      tensor_copy_NCHW_to_NCHWc( naive_deloutput_pad, deloutput_libxsmm, nImg, nFm, ofhp, ofwp, bc );
    } else {
      tensor_copy_NCHW_to_NCHWc_bf16(     naive_input_pad_bf16,     input_libxsmm_bf16, nImg, nFm, ifhp, ifwp, bc );
      tensor_copy_NCHW_to_NCHWc_bf16(    naive_output_pad_bf16,    output_libxsmm_bf16, nImg, nFm, ofhp, ofwp, bc );
      tensor_copy_NCHW_to_NCHWc_bf16(  naive_delinput_pad_bf16,  delinput_libxsmm_bf16, nImg, nFm, ifhp, ifwp, bc );
      tensor_copy_NCHW_to_NCHWc_bf16( naive_deloutput_pad_bf16, deloutput_libxsmm_bf16, nImg, nFm, ofhp, ofwp, bc );
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
        if ( prec_bf16 == 0 ) {
          libxsmm_dnn_pooling_fwd_exec_f32( fwd_cfg, input_libxsmm, output_libxsmm, mask_libxsmm,
                                   0, tid, scratch );
        } else {
          libxsmm_dnn_pooling_fwd_exec_bf16( fwd_cfg, input_libxsmm_bf16, output_libxsmm_bf16, mask_libxsmm,
                                    0, tid, scratch );
        }
      }
      /* copy out data */
      if ( prec_bf16 == 0 ) {
        tensor_copy_NCHWc_to_NCHW( output_libxsmm, naive_libxsmm_output, nImg, nFm, ofhp, ofwp, bc );
      } else {
        tensor_copy_NCHWc_to_NCHW_bf16( output_libxsmm_bf16, naive_libxsmm_output_bf16, nImg, nFm, ofhp, ofwp, bc );
        libxsmm_convert_bf16_f32( naive_libxsmm_output_bf16, naive_libxsmm_output, nImg*nFm*ofhp*ofwp );
      }
      copy_internal_nchw( naive_output_pad, naive_output, nImg, nFm, ofh, ofw, pad_h_out, pad_w_out);

      /* compare */
      libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, nImg*nFm*ofhp*ofwp, 1, naive_output_pad, naive_libxsmm_output, 0, 0);
      printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_fwd);
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
        if ( prec_bf16 == 0 ) {
          libxsmm_dnn_pooling_bwd_exec_f32( bwd_cfg, delinput_libxsmm, deloutput_libxsmm, mask_libxsmm,
                                   0, tid, scratch );
        } else {
          libxsmm_dnn_pooling_bwd_exec_bf16( bwd_cfg, delinput_libxsmm_bf16, deloutput_libxsmm_bf16, mask_libxsmm,
                                    0, tid, scratch );
        }
      }

      /* copy out data */
      if ( prec_bf16 == 0 ) {
        tensor_copy_NCHWc_to_NCHW( delinput_libxsmm, naive_libxsmm_delinput, nImg, nFm, ifhp, ifwp, bc );
      } else {
        tensor_copy_NCHWc_to_NCHW_bf16( delinput_libxsmm_bf16, naive_libxsmm_delinput_bf16, nImg, nFm, ifhp, ifwp, bc );
        libxsmm_convert_bf16_f32( naive_libxsmm_delinput_bf16, naive_libxsmm_delinput, nImg*nFm*ifhp*ifwp );
      }
      copy_internal_nchw( naive_delinput_pad, naive_delinput, nImg, nFm, ifh, ifw, pad_h_in, pad_w_in);

      /* compare */
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
          if ( prec_bf16 == 0 ) {
            libxsmm_dnn_pooling_fwd_exec_f32( fwd_cfg, input_libxsmm, output_libxsmm, mask_libxsmm,
                                     0, tid, scratch );
          } else {
            libxsmm_dnn_pooling_fwd_exec_bf16( fwd_cfg, input_libxsmm_bf16, output_libxsmm_bf16, mask_libxsmm,
                                      0, tid, scratch );
          }
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);

      gb = ((double)nImg*(double)nFm*(((double)ifh*(double)ifw) + (2.0*(double)ofh*(double)ofw))*(double)sizeof(float)*(double)iters) / (1000*1000*1000);
      gib = ((double)nImg*(double)nFm*(((double)ifh*(double)ifw) + (2.0*(double)ofh*(double)ofw))*(double)sizeof(float)*(double)iters) / (1024*1024*1024);

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
          if ( prec_bf16 == 0 ) {
            libxsmm_dnn_pooling_bwd_exec_f32( bwd_cfg, delinput_libxsmm, deloutput_libxsmm, mask_libxsmm,
                                     0, tid, scratch );
          } else {
            libxsmm_dnn_pooling_bwd_exec_bf16( bwd_cfg, delinput_libxsmm_bf16, deloutput_libxsmm_bf16, mask_libxsmm,
                                      0, tid, scratch );
          }
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);

      gb = ((double)nImg*(double)nFm*(((double)ifh*(double)ifw) + (2.0*(double)ofh*(double)ofw))*(double)sizeof(float)*(double)iters) / (1000*1000*1000);
      gib = ((double)nImg*(double)nFm*(((double)ifh*(double)ifw) + (2.0*(double)ofh*(double)ofw))*(double)sizeof(float)*(double)iters) / (1024*1024*1024);

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
    libxsmm_free( scratch );
  }

  /* deallocate data */
  libxsmm_free(naive_input);
  libxsmm_free(naive_input_pad);
  libxsmm_free(naive_mask);
  libxsmm_free(naive_output);
  libxsmm_free(naive_output_pad);
  libxsmm_free(naive_delinput);
  libxsmm_free(naive_delinput_pad);
  libxsmm_free(naive_deloutput);
  libxsmm_free(naive_deloutput_pad);
  libxsmm_free(naive_libxsmm_output);
  libxsmm_free(naive_libxsmm_delinput);
  libxsmm_free(mask_libxsmm);
  if ( prec_bf16 == 0 ) {
    libxsmm_free(input_libxsmm);
    libxsmm_free(output_libxsmm);
    libxsmm_free(delinput_libxsmm);
    libxsmm_free(deloutput_libxsmm);
  } else {
    libxsmm_free(naive_libxsmm_output_bf16);
    libxsmm_free(naive_libxsmm_delinput_bf16);
    libxsmm_free(naive_input_pad_bf16);
    libxsmm_free(naive_output_pad_bf16);
    libxsmm_free(naive_delinput_pad_bf16);
    libxsmm_free(naive_deloutput_pad_bf16);
    libxsmm_free(input_libxsmm_bf16);
    libxsmm_free(output_libxsmm_bf16);
    libxsmm_free(delinput_libxsmm_bf16);
    libxsmm_free(deloutput_libxsmm_bf16);
  }

  { const char *const env_check_scale = getenv("CHECK_SCALE");
    const double check_scale = LIBXSMM_ABS(0 == env_check_scale ? 1.0 : atof(env_check_scale));
    if (LIBXSMM_NEQ(0, check) && (check < 100.0 * check_scale * diff.normf_rel)) {
      fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
      exit(EXIT_FAILURE);
    }
  }

  /* some empty lines at the end */
  printf("\n\n\n");

  return 0;
}

