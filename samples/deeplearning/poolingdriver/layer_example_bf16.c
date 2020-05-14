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
  float *naive_input,     *naive_output,     *naive_delinput,     *naive_deloutput;
  float *naive_input_pad, *naive_output_pad, *naive_delinput_pad, *naive_deloutput_pad;
  libxsmm_bfloat16 *naive_input_pad_bf16, *naive_output_pad_bf16, *naive_delinput_pad_bf16, *naive_deloutput_pad_bf16;
  libxsmm_bfloat16 *naive_libxsmm_output, *naive_libxsmm_delinput;
  float *naive_libxsmm_output_f32, *naive_libxsmm_delinput_f32;
  int   *naive_mask, *naive_libxsmm_mask;
  libxsmm_bfloat16 *input_libxsmm, *output_libxsmm, *delinput_libxsmm, *deloutput_libxsmm;
  int   *mask_libxsmm;

  int ifhp, ifwp, ofhp, ofwp, ofh, ofw;
  int stride_h, stride_w;
  naive_pooling_t naive_param;
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
  int kh = 2;             /* kernel size height */
  int kw = 2;             /* kernel size width */
  int pad_h = 0;          /* pad in h direction */
  int pad_w = 0;          /* pad in w direction */
  int pad_h_in = 0;       /* padding mode */
  int pad_w_in = 0;       /* padding mode */
  int pad_h_out = 0;      /* padding mode */
  int pad_w_out = 0;      /* padding mode */
  int pool_type = 0;      /* max pooling */
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

  libxsmm_dnn_pooling_desc pooling_desc;
  libxsmm_dnn_pooling* libxsmm_handle;
  libxsmm_dnn_tensor*  libxsmm_input;
  libxsmm_dnn_tensor*  libxsmm_delinput;
  libxsmm_dnn_tensor*  libxsmm_output;
  libxsmm_dnn_tensor*  libxsmm_deloutput;
  libxsmm_dnn_tensor*  libxsmm_mask;
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
  naive_input_pad            = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
  naive_delinput             = (float*)libxsmm_aligned_malloc( nImg*nFm*ifh *ifw *sizeof(float), 2097152);
  naive_delinput_pad         = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);
  naive_mask                 = (int*  )libxsmm_aligned_malloc( nImg*nFm*ofh *ofw *sizeof(float), 2097152);
  naive_output               = (float*)libxsmm_aligned_malloc( nImg*nFm*ofh *ofw *sizeof(float), 2097152);
  naive_output_pad           = (float*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(float), 2097152);
  naive_deloutput            = (float*)libxsmm_aligned_malloc( nImg*nFm*ofh *ofw *sizeof(float), 2097152);
  naive_deloutput_pad        = (float*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(float), 2097152);

  naive_input_pad_bf16       = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(libxsmm_bfloat16), 2097152);
  naive_delinput_pad_bf16    = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(libxsmm_bfloat16), 2097152);
  naive_output_pad_bf16      = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(libxsmm_bfloat16), 2097152);
  naive_deloutput_pad_bf16   = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(libxsmm_bfloat16), 2097152);

  naive_libxsmm_output       = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(libxsmm_bfloat16), 2097152);
  naive_libxsmm_delinput     = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(libxsmm_bfloat16), 2097152);
  naive_libxsmm_output_f32   = (float*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(float), 2097152);
  naive_libxsmm_delinput_f32 = (float*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(float), 2097152);

  naive_libxsmm_mask         = (int*  )libxsmm_aligned_malloc( nImg*nFm*ofh *ofw *sizeof(float), 2097152);

  input_libxsmm              = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(libxsmm_bfloat16), 2097152);
  delinput_libxsmm           = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nFm*ifhp*ifwp*sizeof(libxsmm_bfloat16), 2097152);
  output_libxsmm             = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(libxsmm_bfloat16), 2097152);
  deloutput_libxsmm          = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nFm*ofhp*ofwp*sizeof(libxsmm_bfloat16), 2097152);
  mask_libxsmm               = (int*  )libxsmm_aligned_malloc( nImg*nFm*ofh *ofw *sizeof(float), 2097152);

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

  libxsmm_rne_convert_fp32_bf16( naive_input_pad,     naive_input_pad_bf16,     nImg*nFm*ifhp*ifwp );
  libxsmm_rne_convert_fp32_bf16( naive_delinput_pad,  naive_delinput_pad_bf16,  nImg*nFm*ifhp*ifwp );
  libxsmm_rne_convert_fp32_bf16( naive_output_pad,    naive_output_pad_bf16,    nImg*nFm*ofhp*ofwp );
  libxsmm_rne_convert_fp32_bf16( naive_deloutput_pad, naive_deloutput_pad_bf16, nImg*nFm*ofhp*ofwp );

  zero_buf_int32(naive_mask,      nImg*nFm*ofh*ofw);
  zero_buf_int32(mask_libxsmm,    nImg*nFm*ofh*ofw);

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

    /* setup LIBXSMM handle */
    pooling_desc.N = nImg;
    pooling_desc.C = nFm;
    pooling_desc.H = ifh;
    pooling_desc.W = ifw;
    pooling_desc.u = stride_h;
    pooling_desc.v = stride_w;
    pooling_desc.R = kh;
    pooling_desc.S = kw;
    pooling_desc.pad_h = pad_h;
    pooling_desc.pad_w = pad_w;
    pooling_desc.pad_h_in = pad_h_in;
    pooling_desc.pad_w_in = pad_w_in;
    pooling_desc.pad_h_out = pad_h_out;
    pooling_desc.pad_w_out = pad_w_out;
    pooling_desc.threads = nThreads;
    pooling_desc.datatype_in = LIBXSMM_DNN_DATATYPE_BF16;
    pooling_desc.datatype_out = LIBXSMM_DNN_DATATYPE_BF16;
    pooling_desc.datatype_mask = LIBXSMM_DNN_DATATYPE_I32;
    pooling_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
    if ( pool_type == 0 ) {
      pooling_desc.pooling_type = LIBXSMM_DNN_POOLING_MAX;
    } else if ( pool_type == 1 ) {
      pooling_desc.pooling_type = LIBXSMM_DNN_POOLING_AVG;
    } else {
      return 0;
    }

    libxsmm_handle = libxsmm_dnn_create_pooling( pooling_desc, &status );
    CHKERR_LIBXSMM_DNN( status );

    /* setup LIBXSMM buffers */
    libxsmm_layout = libxsmm_dnn_pooling_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_input  = libxsmm_dnn_link_tensor( libxsmm_layout, input_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    printf("inner activation blocking: %i\n", libxsmm_layout->dim_size[0] );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_pooling_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_delinput  = libxsmm_dnn_link_tensor( libxsmm_layout, delinput_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_pooling_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_output  = libxsmm_dnn_link_tensor( libxsmm_layout, output_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_pooling_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_deloutput  = libxsmm_dnn_link_tensor( libxsmm_layout, deloutput_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_pooling_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_POOLING_MASK, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_mask  = libxsmm_dnn_link_tensor( libxsmm_layout, mask_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    /* copy in data to LIBXSMM format */
    /* we can also use the layout functions and set the data on our
       own external to the library */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_input,     (void*)naive_input_pad_bf16,     LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_output,    (void*)naive_output_pad_bf16,    LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_delinput,  (void*)naive_delinput_pad_bf16,  LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_deloutput, (void*)naive_deloutput_pad_bf16, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );

    /* bind buffers and filter to handle */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_bind_tensor( libxsmm_handle, libxsmm_input,     LIBXSMM_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_bind_tensor( libxsmm_handle, libxsmm_delinput,  LIBXSMM_DNN_GRADIENT_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_bind_tensor( libxsmm_handle, libxsmm_output,    LIBXSMM_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_bind_tensor( libxsmm_handle, libxsmm_deloutput, LIBXSMM_DNN_GRADIENT_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_bind_tensor( libxsmm_handle, libxsmm_mask  ,    LIBXSMM_DNN_POOLING_MASK ) );

    /* let's allocate and bind scratch */
    scratch_size = libxsmm_dnn_pooling_get_scratch_size( libxsmm_handle, &status );
    CHKERR_LIBXSMM_DNN( status );
    scratch = libxsmm_aligned_scratch( scratch_size, 2097152 );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_bind_scratch( libxsmm_handle, scratch ) );
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
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
      }
      /* copy out data */
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_output, (void*)naive_libxsmm_output, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
      libxsmm_convert_bf16_f32( naive_libxsmm_output, naive_libxsmm_output_f32, nImg*nFm*ofhp*ofwp );
      copy_internal_nchw( naive_output_pad, naive_output, nImg, nFm, ofh, ofw, pad_h_out, pad_w_out);

      /* compare */
      libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, nImg*nFm*ofhp*ofwp, 1, naive_output_pad, naive_libxsmm_output_f32, 0, 0);
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
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid ) );
      }

      /* copy out data */
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_delinput,     (void*)naive_libxsmm_delinput,     LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
      libxsmm_convert_bf16_f32( naive_libxsmm_delinput,     naive_libxsmm_delinput_f32,     nImg*nFm*ifhp*ifwp );
      copy_internal_nchw( naive_delinput_pad, naive_delinput, nImg, nFm, ifh, ifw, pad_h_in, pad_w_in);

      /* compare */
      libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, nImg*nFm*ifhp*ifwp, 1, naive_delinput_pad, naive_libxsmm_delinput_f32, 0, 0);
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
          libxsmm_dnn_pooling_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid );
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
          libxsmm_dnn_pooling_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid );
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
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_release_scratch( libxsmm_handle ) );
    libxsmm_free(scratch);
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_release_tensor( libxsmm_handle, LIBXSMM_DNN_POOLING_MASK ) );

    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_input ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_delinput ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_output ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_deloutput ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_mask ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_pooling( libxsmm_handle ) );
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
  libxsmm_free(naive_libxsmm_mask);
  libxsmm_free(input_libxsmm);
  libxsmm_free(mask_libxsmm);
  libxsmm_free(output_libxsmm);
  libxsmm_free(delinput_libxsmm);
  libxsmm_free(deloutput_libxsmm);
  libxsmm_free(naive_libxsmm_output_f32);
  libxsmm_free(naive_libxsmm_delinput_f32);
  libxsmm_free(naive_input_pad_bf16);
  libxsmm_free(naive_output_pad_bf16);
  libxsmm_free(naive_delinput_pad_bf16);
  libxsmm_free(naive_deloutput_pad_bf16);

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

