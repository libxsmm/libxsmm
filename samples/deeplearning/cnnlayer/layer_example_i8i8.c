/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Evangelos Georganas, Hans Pabst,
   Dhiraj Kalamkar, Ankush Mandal (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

#define USE_OVERWRITE

/* include c-based dnn library */
#include "../common/dnn_common.h"

#define CHKERR_LIBXSMM_DNN(A) { const int chkerr_libxsmm_dnn_ = A; if (LIBXSMM_DNN_SUCCESS != chkerr_libxsmm_dnn_) { \
  fprintf(stderr, "%s\n", libxsmm_dnn_get_error(chkerr_libxsmm_dnn_)); global_status = chkerr_libxsmm_dnn_; } \
}

int main(int argc, char* argv[])
{

  float *naive_input_fp, *naive_output_fp, *naive_filter_fp, *naive_libxsmm_output_fp, *dq_naive_input, *dq_naive_filter;
  char *naive_filter_i8, *naive_output_i8, *naive_libxsmm_output, *filter_libxsmm, *output_libxsmm;
  unsigned char *naive_input_i8, *input_libxsmm;


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
  libxsmm_dnn_tensor_datalayout* libxsmm_layout;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status = LIBXSMM_DNN_SUCCESS;

  libxsmm_matdiff_info norms_fwd, diff, norms_quant;
  libxsmm_matdiff_clear(&norms_fwd);
  libxsmm_matdiff_clear(&diff);
  libxsmm_matdiff_clear(&norms_quant);

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("Usage: %s iters inpWidth inpHeight nImg nIfm nOfm kw kh pad stride type padding_mode\n", argv[0]);
    return 0;
  }
  srand(1);

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

  if (type != 'A' && type != 'F' && type != 'B' && type != 'U') {
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

  /* print some summary */
  printf("##########################################\n");
  printf("#    Setting Up Common    #\n");
  printf("##########################################\n");
  printf("PARAMS: W:%d  H:%d  N:%d  C:%d  K:%d  R:%d  S:%d  P:%d  Q:%d  STRIDE:%d\n", ifw, ifh, nImg, nIfm, nOfm, kw, kh, ofh, ofw, stride);
  printf("PARAMS: ITERS:%d", iters); if (LIBXSMM_FEQ(0, check)) printf("  Threads:%d\n", nThreads); else printf("\n");
  printf(" InImg %dx%d Padded (%dx%d)\n", ifh, ifw, ifhp, ifwp);
  printf("OutImg %dx%d Padded (%dx%d)\n", ofh, ofw, ofhp, ofwp);
  printf("SIZE Input  (MB): %10.2f MiB\n", (double)(nImg*nIfm*ifhp*ifwp*sizeof(unsigned char))/(1024.0*1024.0) );
  printf("SIZE Output (MB): %10.2f MiB\n", (double)(nImg*nOfm*ofhp*ofwp*sizeof(char))/(1024.0*1024.0) );
  printf("SIZE Input   (1): %10.2f MiB\n", (double)(1*nIfm*ifhp*ifwp*   sizeof(unsigned char))/(1024.0*1024.0) );
  printf("SIZE Output  (1): %10.2f MiB\n", (double)(1*nOfm*ofhp*ofwp*   sizeof(char))/(1024.0*1024.0) );
  printf("SIZE Weight     : %10.2f MiB\n", (double)(nIfm*nOfm*kw*kh*    sizeof(char))/(1024.0*1024.0) );

  /* allocate data */
  naive_input_fp        = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  naive_output_fp       = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float),   2097152);
  naive_filter_fp       = (float*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);

  naive_input_i8        = (unsigned char*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(unsigned char), 2097152);
  naive_output_i8       = (char*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(char),   2097152);
  naive_filter_i8       = (char*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(char), 2097152);

  naive_libxsmm_output      = (char*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(char),   2097152);
  naive_libxsmm_output_fp   = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float),   2097152);
  input_libxsmm         = (unsigned char*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(unsigned char), 2097152);
  filter_libxsmm        = (char*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(char), 2097152);
  output_libxsmm        = (char*) libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(char), 2097152);

  dq_naive_input        = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  dq_naive_filter       = (float*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);

  /* initialize data */
  if (padding_mode == 0 ) {
    init_buf_range(naive_input_fp, nImg*nIfm*ifhp*ifwp, 0.0, 1.0);
  } else {
    float *naive_input_tmp = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
    init_buf_range(naive_input_tmp, nImg*nIfm*ifh*ifw, 0.0, 1.0);
    copy_internal_nchw( naive_input_fp , naive_input_tmp, nImg, nIfm, ifh, ifw, pad_h, pad_w);
    libxsmm_free(naive_input_tmp);
  }
  init_buf_range(naive_filter_fp, nOfm*nIfm*kh*kw, -1.0, 1.0);
  zero_buf_int8(output_libxsmm,      nImg*nOfm*ofhp*ofwp);

  if (LIBXSMM_NEQ(0, check)) {
    printf("##########################################\n");
    printf("#         Computing Reference ...        #\n");
    printf("##########################################\n");
    /* run naive convolutions */
    if (type == 'A' || type == 'F') {
      zero_buf(naive_output_fp,    nImg*nOfm*ofhp*ofwp);
      naive_conv_fp(&naive_param, naive_input_fp, naive_output_fp, naive_filter_fp, NULL);
    }
    printf("##########################################\n");
    printf("#      Computing Reference ... done      #\n");
    printf("##########################################\n");
  }

  /* Quantize input and filter  */
  unsigned char filter_scf, input_scf, output_scf;
  quantize_buffer_uchar(naive_input_fp, naive_input_i8, nImg*nIfm*ifhp*ifwp, 2, &input_scf);
  quantize_buffer_char(naive_filter_fp, naive_filter_i8, nOfm*nIfm*kh*kw, 2, &filter_scf);
  quantize_buffer_char(naive_output_fp, naive_output_i8, nImg*nOfm*ofhp*ofwp, 2, &output_scf);

  /* dequantize to check quantization error */
  libxsmm_dnn_dequantize_int8( (char*)naive_input_i8,  dq_naive_input,  nImg*nIfm*ifhp*ifwp, input_scf);
  libxsmm_dnn_dequantize_int8( (char*)naive_filter_i8, dq_naive_filter, nIfm*nOfm*kw*kh,     filter_scf);

#if 0
  /* norms quantization */
  libxsmm_matdiff(&norms_quant, LIBXSMM_DATATYPE_F32, nImg*nIfm*ifhp*ifwp, 1, naive_input_fp, dq_naive_input, 0, 0);
  printf("Input Quantization:\n");
  printf("L1 reference  : %.25g\n", norms_quant.l1_ref);
  printf("L1 test       : %.25g\n", norms_quant.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_quant.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_quant.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_quant.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_quant.linf_rel);
  printf("Check-norm    : %.24f\n", norms_quant.normf_rel);
  libxsmm_matdiff_clear(&norms_quant);

  libxsmm_matdiff(&norms_quant, LIBXSMM_DATATYPE_F32, nIfm*nOfm*kw*kh, 1, naive_filter_fp, dq_naive_filter, 0, 0);
  printf("Filter Quantization:\n");
  printf("L1 reference  : %.25g\n", norms_quant.l1_ref);
  printf("L1 test       : %.25g\n", norms_quant.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_quant.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_quant.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_quant.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_quant.linf_rel);
  printf("Check-norm    : %.24f\n", norms_quant.normf_rel);
  libxsmm_matdiff_clear(&norms_quant);  printf("\n");
#endif

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
  conv_desc.options = LIBXSMM_DNN_CONV_OPTION_OVERWRITE;
  conv_desc.datatype_in = LIBXSMM_DNN_DATATYPE_I8;
  conv_desc.datatype_out = LIBXSMM_DNN_DATATYPE_I8;

  libxsmm_handle = libxsmm_dnn_create_conv_layer( conv_desc, &status );
  CHKERR_LIBXSMM_DNN( status );

  /* setup LIBXSMM buffers and filter */
  libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_input  = libxsmm_dnn_link_tensor( libxsmm_layout,  input_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

  libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_OUTPUT, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_output  = libxsmm_dnn_link_tensor( libxsmm_layout,  output_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

  libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_FILTER, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_filter  = libxsmm_dnn_link_tensor( libxsmm_layout,  filter_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

  /* copy in data to LIBXSMM format */
  /* we can also use the layout functions and set the data on our
     own external to the library, @TODO, we plan to add an example here */
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_input, (void*)naive_input_i8, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_zero_tensor( libxsmm_output ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_filter, (void*)naive_filter_i8, LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );

  /* bind buffers and filter to handle */
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_input, LIBXSMM_DNN_REGULAR_INPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_output, LIBXSMM_DNN_REGULAR_OUTPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_filter, LIBXSMM_DNN_REGULAR_FILTER ) );

  /* set scaling factors into tensors */
  libxsmm_dnn_set_qtensor_scf( libxsmm_input,  input_scf );
  libxsmm_dnn_set_qtensor_scf( libxsmm_filter, filter_scf );
  libxsmm_dnn_set_qtensor_scf( libxsmm_output, output_scf );

  /* let's allocate and bind scratch */
  scratch_size = libxsmm_dnn_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );
  CHKERR_LIBXSMM_DNN( status );
  scratch = libxsmm_aligned_malloc( scratch_size, 2097152 );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch ) );

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

    /* Dequantize result to check correctness  */
    libxsmm_dnn_dequantize_int8( (char*)naive_libxsmm_output, naive_libxsmm_output_fp, nImg*nOfm*ofhp*ofwp, output_scf);

    /* compare */
    libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, nImg*nOfm*ofhp*ofwp, 1, naive_output_fp, naive_libxsmm_output_fp, 0, 0);
    printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
    printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
    printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
    libxsmm_matdiff_reduce(&diff, &norms_fwd);
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

  /* clean-up */
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL ) );
  libxsmm_free(scratch);
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_input ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_output ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_filter ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_conv_layer( libxsmm_handle ) );

  /* deallocate data */
  libxsmm_free(naive_input_fp);
  libxsmm_free(naive_filter_fp);
  libxsmm_free(naive_output_fp);
  libxsmm_free(naive_libxsmm_output_fp);
  libxsmm_free(dq_naive_input);
  libxsmm_free(dq_naive_filter);
  libxsmm_free(naive_filter_i8);
  libxsmm_free(naive_output_i8);
  libxsmm_free(naive_libxsmm_output);
  libxsmm_free(input_libxsmm);
  libxsmm_free(naive_input_i8);
  libxsmm_free(output_libxsmm);
  libxsmm_free(filter_libxsmm);

  { const char *const env_check_scale = getenv("CHECK_SCALE");
    const double check_scale = LIBXSMM_ABS(0 == env_check_scale ? 0.01 : atof(env_check_scale));
    if (LIBXSMM_NEQ(0, check) && (check < 100.0 * check_scale * diff.normf_rel) && (global_status == LIBXSMM_DNN_SUCCESS)) {
      fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
      exit(EXIT_FAILURE);
    }
  }

  /* some empty lines at the end */
  printf("\n\n\n");

  return LIBXSMM_DNN_SUCCESS;
}

