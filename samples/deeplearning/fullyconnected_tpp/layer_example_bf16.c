/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <libxsmm_sync.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

/* include c-based dnn library */
#include "../common/dnn_common.h"
#include "../op_lib_tpp/fullyconnected_tpp.h"

int main(int argc, char* argv[])
{
  float *naive_input, *naive_output, *naive_filter, *naive_delinput, *naive_deloutput, *naive_delfilter, *naive_bias, *naive_delbias;
  libxsmm_bfloat16 *naive_input_bf16, *naive_filter_bf16, *naive_output_bf16, *naive_delinput_bf16, *naive_delfilter_bf16, *naive_deloutput_bf16, *naive_bias_bf16, *naive_delbias_bf16;
  float *naive_libxsmm_output_f32, *naive_libxsmm_delinput_f32, *naive_libxsmm_delfilter_f32, *naive_libxsmm_delbias_f32;
  libxsmm_bfloat16 *naive_libxsmm_output_bf16, *naive_libxsmm_delinput_bf16, *naive_libxsmm_delfilter_bf16, *naive_libxsmm_delbias_bf16;
  libxsmm_bfloat16 *input_libxsmm, *filter_libxsmm, *delinput_libxsmm, *delfilter_libxsmm, *output_libxsmm, *deloutput_libxsmm, *bias_libxsmm, *delbias_libxsmm;
  unsigned char *relumask_libxsmm;
  my_fc_eltw_fuse my_fuse;
  my_fc_fwd_config my_fc_fwd;
  my_fc_bwd_config my_fc_bwd;
  naive_fullyconnected_t naive_param;
  void* scratch = 0;

  /* some parameters we can overwrite via cli,
     default is some inner layer of overfeat */
  int iters = 10;         /* repetitions of benchmark */
  int nImg = 256;          /* mini-batch size, "N" */
  int nIFm = 1024;          /* number of input feature maps, "C" */
  int nOFm = 1024;          /* number of input feature maps, "C" */
  int fuse_type = 0;      /* 0: nothing fused, 1: relu fused, 2: elementwise fused, 3: relu and elementwise fused */
  char type = 'A';        /* 'A': ALL, 'F': FP, 'B': BP, 'U', WU */
  int bn = 32;
  int bk = 32;
  int bc = 32;

  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 1 : atof(env_check));

#if defined(_OPENMP)
  int nThreads = omp_get_max_threads(); /* number of threads */
#else
  int nThreads = 1; /* number of threads */
#endif

  unsigned long long l_start, l_end;
  double l_total = 0.0;
  double gflop = 0.0;
  int i;

  libxsmm_matdiff_info norms_fwd, norms_bwd, norms_upd, diff;
  libxsmm_matdiff_clear(&norms_fwd);
  libxsmm_matdiff_clear(&norms_bwd);
  libxsmm_matdiff_clear(&norms_upd);
  libxsmm_matdiff_clear(&diff);

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("Usage: %s iters nImg nIFm nOFm fuse_type type bn bk bc\n", argv[0]);
    return 0;
  }
  libxsmm_rng_set_seed(1);

  /* reading new values from cli */
  i = 1;
  if (argc > i) iters      = atoi(argv[i++]);
  if (argc > i) nImg       = atoi(argv[i++]);
  if (argc > i) nIFm       = atoi(argv[i++]);
  if (argc > i) nOFm       = atoi(argv[i++]);
  if (argc > i) fuse_type  = atoi(argv[i++]);
  if (argc > i) type       = *(argv[i++]);
  if (argc > i) bn         = atoi(argv[i++]);
  if (argc > i) bk         = atoi(argv[i++]);
  if (argc > i) bc         = atoi(argv[i++]);

  if (type != 'A' && type != 'F' && type != 'B' && type != 'U' && type != 'M') {
    printf("type needs to be 'A' (All), 'F' (FP only), 'B' (BP only), 'U' (UP only). 'M' (BPUP-fused only)\n");
    return -1;
  }
  if ( (fuse_type < 0) || (fuse_type > 5) ) {
    printf("fuse type needs to be 0 (None), 1 (Bias), 2 (ReLU,mask), 3 (Bias+ReLU,mask), 4 (ReLU), 5 (Bias+ReLU)\n");
    return -1;
  }

  /* set struct for naive convolution */
  naive_param.N = nImg;
  naive_param.C = nIFm;
  naive_param.K = nOFm;
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
  printf("PARAMS: N:%d  C:%d  K:%d\n", nImg, nIFm, nOFm);
  printf("PARAMS: ITERS:%d", iters); if (LIBXSMM_FEQ(0, check)) printf("  Threads:%d\n", nThreads); else printf("\n");
  printf("SIZE Input  (MB): %10.2f MiB\n", (double)(nImg*nIFm*sizeof(libxsmm_bfloat16))/(1024.0*1024.0) );
  printf("SIZE Output (MB): %10.2f MiB\n", (double)(nImg*nOFm*sizeof(libxsmm_bfloat16))/(1024.0*1024.0) );
  printf("SIZE Input   (1): %10.2f MiB\n", (double)(1*nIFm*   sizeof(libxsmm_bfloat16))/(1024.0*1024.0) );
  printf("SIZE Output  (1): %10.2f MiB\n", (double)(1*nOFm*   sizeof(libxsmm_bfloat16))/(1024.0*1024.0) );
  printf("SIZE Filter     : %10.2f MiB\n", (double)(nIFm*nOFm*sizeof(libxsmm_bfloat16))/(1024.0*1024.0) );

  /* allocate data */
  naive_input                 = (float*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(float), 2097152);
  naive_delinput              = (float*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(float), 2097152);
  naive_output                = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);
  naive_deloutput             = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);
  naive_filter                = (float*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(float), 2097152);
  naive_delfilter             = (float*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(float), 2097152);
  naive_bias                  = (float*)libxsmm_aligned_malloc( nOFm     *sizeof(float), 2097152);
  naive_delbias               = (float*)libxsmm_aligned_malloc( nOFm     *sizeof(float), 2097152);

  naive_input_bf16            = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_delinput_bf16         = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_output_bf16           = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_deloutput_bf16        = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_filter_bf16           = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_delfilter_bf16        = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_bias_bf16             = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nOFm     *sizeof(libxsmm_bfloat16), 2097152);
  naive_delbias_bf16          = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nOFm     *sizeof(libxsmm_bfloat16), 2097152);

  naive_libxsmm_output_bf16   = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_libxsmm_delinput_bf16 = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_libxsmm_delfilter_bf16= (libxsmm_bfloat16*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_libxsmm_delbias_bf16  = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nOFm     *sizeof(libxsmm_bfloat16), 2097152);
  naive_libxsmm_output_f32    = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);
  naive_libxsmm_delinput_f32  = (float*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(float), 2097152);
  naive_libxsmm_delfilter_f32 = (float*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(float), 2097152);
  naive_libxsmm_delbias_f32   = (float*)libxsmm_aligned_malloc( nOFm     *sizeof(float), 2097152);

  input_libxsmm               = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(libxsmm_bfloat16), 2097152);
  delinput_libxsmm            = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(libxsmm_bfloat16), 2097152);
  output_libxsmm              = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  deloutput_libxsmm           = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  filter_libxsmm              = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  delfilter_libxsmm           = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  bias_libxsmm                =  (libxsmm_bfloat16*)libxsmm_aligned_malloc( nOFm     *sizeof(libxsmm_bfloat16), 2097152);
  delbias_libxsmm             =  (libxsmm_bfloat16*)libxsmm_aligned_malloc( nOFm     *sizeof(libxsmm_bfloat16), 2097152);
  relumask_libxsmm            =  (unsigned char*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(unsigned char), 2097152);

  /* initialize data */
  init_buf( naive_input,     nImg*nIFm, 0, 0 );
  init_buf( naive_delinput,  nImg*nIFm, 0, 0 );
  init_buf( naive_output,    nImg*nOFm, 0, 0 );
  init_buf( naive_deloutput, nImg*nOFm, 0, 0 );
  init_buf( naive_filter,    nIFm*nOFm, 0, 0 );
  init_buf( naive_delfilter, nIFm*nOFm, 0, 0 );
  init_buf( naive_bias,      nOFm,      0, 0 );
  init_buf( naive_delbias,   nOFm,      0, 0 );

  libxsmm_rne_convert_fp32_bf16( naive_input,     naive_input_bf16,     nImg*nIFm );
  libxsmm_rne_convert_fp32_bf16( naive_delinput,  naive_delinput_bf16,  nImg*nIFm );
  libxsmm_rne_convert_fp32_bf16( naive_output,    naive_output_bf16,    nImg*nOFm );
  libxsmm_rne_convert_fp32_bf16( naive_deloutput, naive_deloutput_bf16, nImg*nOFm );
  libxsmm_rne_convert_fp32_bf16( naive_filter,    naive_filter_bf16,    nIFm*nOFm );
  libxsmm_rne_convert_fp32_bf16( naive_delfilter, naive_delfilter_bf16, nIFm*nOFm );
  libxsmm_rne_convert_fp32_bf16( naive_bias,      naive_bias_bf16,      nOFm );
  libxsmm_rne_convert_fp32_bf16( naive_delbias,   naive_delbias_bf16,   nOFm );

  if (LIBXSMM_NEQ(0, check)) {
    printf("##########################################\n");
    printf("#         Computing Reference ...        #\n");
    printf("##########################################\n");
    if (type == 'A' || type == 'F') {
      naive_fullyconnected_fused_fp(&naive_param, naive_input, naive_output, naive_filter, naive_bias);
    }
    if (type == 'A' || type == 'B' || type == 'M') {
      naive_fullyconnected_fused_bp(&naive_param, naive_delinput, naive_deloutput, naive_filter, naive_delbias, naive_output);
    }
    if (type == 'A' || type == 'U' || type == 'M') {
      naive_fullyconnected_wu(&naive_param, naive_input, naive_deloutput, naive_delfilter);
    }
    printf("##########################################\n");
    printf("#      Computing Reference ... done      #\n");
    printf("##########################################\n");
  }

  printf("\n");
  printf("##########################################\n");
  printf("#      Setting Up  (custom-Storage)      #\n");
  printf("##########################################\n");

  if (nImg % bn != 0) {
    bn = nImg;
  }
  if (nIFm % bc != 0) {
    bc = nIFm;
  }
  if (nOFm % bk != 0) {
    bk = nOFm;
  }

  if ( fuse_type == 0 ) {
    my_fuse = MY_FC_ELTW_FUSE_NONE;
  } else if ( fuse_type == 1 ) {
    my_fuse = MY_FC_ELTW_FUSE_BIAS;
  } else if ( fuse_type == 2 ) {
    my_fuse = MY_FC_ELTW_FUSE_RELU_WITH_MASK;
  } else if ( fuse_type == 3 ) {
    my_fuse = MY_FC_ELTW_FUSE_BIAS_RELU_WITH_MASK;
  } else if ( fuse_type == 4 ) {
    my_fuse = MY_FC_ELTW_FUSE_RELU;
  } else if ( fuse_type == 5 ) {
    my_fuse = MY_FC_ELTW_FUSE_BIAS_RELU;
  } else {
    /* cannot happen */
  }

  /* scratch memory size */
  size_t alloc_size = 0;

  if (type == 'A' || type == 'F') {
    my_fc_fwd = setup_my_fc_fwd(nImg, nIFm, nOFm, bn, bc, bk, nThreads, my_fuse, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16);

    alloc_size = my_fc_fwd.scratch_size;
  }
  if (type == 'A' || type == 'B' || type == 'U' || type == 'M') {
    my_fc_bwd = setup_my_fc_bwd(nImg, nIFm, nOFm, bn, bc, bk, nThreads, my_fuse, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16);

    alloc_size = LIBXSMM_MAX( my_fc_fwd.scratch_size, my_fc_bwd.scratch_size);
  }

  /* we can also use the layout functions and set the data on our
     own external to the library */
  matrix_copy_NC_to_NCNC_bf16( naive_input_bf16,     input_libxsmm,     1, nImg, nIFm, bn, bc );
  matrix_copy_NC_to_NCNC_bf16( naive_delinput_bf16,  delinput_libxsmm,  1, nImg, nIFm, bn, bc );
  matrix_copy_NC_to_NCNC_bf16( naive_output_bf16,    output_libxsmm,    1, nImg, nOFm, bn, bk );
  matrix_copy_NC_to_NCNC_bf16( naive_deloutput_bf16, deloutput_libxsmm, 1, nImg, nOFm, bn, bk );
  matrix_copy_KC_to_KCCK_bf16( naive_filter_bf16,    filter_libxsmm      , nIFm, nOFm, bc, bk );
  matrix_copy_KC_to_KCCK_bf16( naive_delfilter_bf16, delfilter_libxsmm   , nIFm, nOFm, bc, bk );
  matrix_copy_NC_to_NCNC_bf16( naive_bias_bf16,    bias_libxsmm,    1, 1, nOFm, 1, nOFm );
  matrix_copy_NC_to_NCNC_bf16( naive_delbias_bf16, delbias_libxsmm, 1, 1, nOFm, 1, nOFm );

  /* let's allocate and bind scratch */
  if ( my_fc_fwd.scratch_size > 0 || my_fc_bwd.scratch_size > 0 ) {
    scratch = libxsmm_aligned_malloc( alloc_size, 2097152 );
    init_buf( (float*)(scratch), (alloc_size)/4, 0, 0 );
  }

  if ((type == 'A' || type == 'F') && LIBXSMM_NEQ(0, check)) {
    printf("##########################################\n");
    printf("#   Correctness - FWD (custom-Storage)   #\n");
    printf("##########################################\n");

#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      my_fc_fwd_exec_bf16( my_fc_fwd, filter_libxsmm, input_libxsmm, output_libxsmm,
                           bias_libxsmm, relumask_libxsmm, 0, tid, scratch );
    }

    /* copy out data */
    matrix_copy_NCNC_to_NC_bf16( output_libxsmm, naive_libxsmm_output_bf16, 1, nImg, nOFm, bn, bk );
    libxsmm_convert_bf16_f32( naive_libxsmm_output_bf16, naive_libxsmm_output_f32, nImg*nOFm );

    /* compare */
    libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, nImg*nOFm, 1, naive_output, naive_libxsmm_output_f32, 0, 0);
    printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
    printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
    printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
    libxsmm_matdiff_reduce(&diff, &norms_fwd);
  }

  if ( (type == 'A' || type == 'M') && LIBXSMM_NEQ(0, check) ) {
    printf("##########################################\n");
    printf("# Correctness - BWDUPD (custom-Storage)  #\n");
    printf("##########################################\n");

#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      my_fc_bwd_exec_bf16( my_fc_bwd, filter_libxsmm, delinput_libxsmm, deloutput_libxsmm, delfilter_libxsmm,
                           input_libxsmm, delbias_libxsmm, relumask_libxsmm, MY_FC_PASS_BWD, 0, tid, scratch );
    }

    /* copy out data */
    matrix_copy_NCNC_to_NC_bf16( delinput_libxsmm, naive_libxsmm_delinput_bf16, 1, nImg, nIFm, bn, bc );
    libxsmm_convert_bf16_f32( naive_libxsmm_delinput_bf16, naive_libxsmm_delinput_f32, nImg*nIFm );
    /* copy out data */
    matrix_copy_KCCK_to_KC_bf16( delfilter_libxsmm, naive_libxsmm_delfilter_bf16, nIFm, nOFm, bc, bk );
    libxsmm_convert_bf16_f32( naive_libxsmm_delfilter_bf16, naive_libxsmm_delfilter_f32, nIFm*nOFm );

    /* compare */
    libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, nImg*nIFm, 1, naive_delinput, naive_libxsmm_delinput_f32, 0, 0);
    printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
    printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
    printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
    libxsmm_matdiff_reduce(&diff, &norms_bwd);
    if ( (fuse_type == 1) || (fuse_type == 3) ) {
      matrix_copy_NCNC_to_NC_bf16( delbias_libxsmm, naive_libxsmm_delbias_bf16, 1, 1, nOFm, 1, nOFm );
      libxsmm_convert_bf16_f32( naive_libxsmm_delbias_bf16, naive_libxsmm_delbias_f32, nOFm );
      libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, nOFm, 1, naive_delbias, naive_libxsmm_delbias_f32, 0, 0);
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_bwd);
    }

    libxsmm_matdiff(&norms_upd, LIBXSMM_DATATYPE_F32, nIFm*nOFm, 1, naive_delfilter, naive_libxsmm_delfilter_f32, 0, 0);
    printf("L1 reference  : %.25g\n", norms_upd.l1_ref);
    printf("L1 test       : %.25g\n", norms_upd.l1_tst);
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
    l_start = libxsmm_timer_tick();
#if defined(_OPENMP)
#   pragma omp parallel private(i)
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      for (i = 0; i < iters; ++i) {
        my_fc_fwd_exec_bf16( my_fc_fwd, filter_libxsmm, input_libxsmm, output_libxsmm,
                             bias_libxsmm, relumask_libxsmm, 0, tid, scratch );
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);

    gflop = (2.0*(double)nImg*(double)nIFm*(double)nOFm*(double)iters) / (1000*1000*1000);

    printf("GFLOP  = %.5g\n", gflop/(double)iters);
    printf("fp time = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS  = %.5g\n", gflop/l_total);

    printf("PERFDUMP,FP,%s,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIFm,
        nOFm, ((double)(l_total/iters)), gflop/l_total, norms_fwd.l1_ref, norms_fwd.l1_tst,
        norms_fwd.l2_abs, norms_fwd.l2_rel, norms_fwd.linf_abs, norms_fwd.linf_rel, norms_fwd.normf_rel);
  }

  if (type == 'A' || type == 'M') {
    printf("##########################################\n");
    printf("# Performance - BWDUPD (custom-Storage)  #\n");
    printf("##########################################\n");
    l_start = libxsmm_timer_tick();
#if defined(_OPENMP)
#   pragma omp parallel private(i)
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      for (i = 0; i < iters; ++i) {
        my_fc_bwd_exec_bf16( my_fc_bwd, filter_libxsmm, delinput_libxsmm, deloutput_libxsmm, delfilter_libxsmm,
                             input_libxsmm, delbias_libxsmm, relumask_libxsmm, MY_FC_PASS_BWD, 0, tid, scratch );
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);

    gflop = (4.0*(double)nImg*(double)nIFm*(double)nOFm*(double)iters) / (1000*1000*1000);

    printf("GFLOP  = %.5g\n", gflop/(double)iters);
    printf("fp time = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS  = %.5g\n", gflop/l_total);

    printf("PERFDUMP,UP,%s,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIFm,
        nOFm, ((double)(l_total/iters)), gflop/l_total, norms_upd.l1_ref, norms_upd.l1_tst,
        norms_upd.l2_abs, norms_upd.l2_rel, norms_upd.linf_abs, norms_upd.linf_rel, norms_upd.normf_rel);
  }

  /* deallocate data */
  if ( scratch != NULL ) {
    libxsmm_free(scratch);
  }
  libxsmm_free(naive_input);
  libxsmm_free(naive_output);
  libxsmm_free(naive_delinput);
  libxsmm_free(naive_deloutput);
  libxsmm_free(naive_filter);
  libxsmm_free(naive_delfilter);
  libxsmm_free(naive_input_bf16);
  libxsmm_free(naive_delinput_bf16);
  libxsmm_free(naive_output_bf16);
  libxsmm_free(naive_deloutput_bf16);
  libxsmm_free(naive_filter_bf16);
  libxsmm_free(naive_delfilter_bf16);
  libxsmm_free(naive_libxsmm_output_bf16);
  libxsmm_free(naive_libxsmm_delinput_bf16);
  libxsmm_free(naive_libxsmm_delfilter_bf16);
  libxsmm_free(naive_libxsmm_output_f32);
  libxsmm_free(naive_libxsmm_delinput_f32);
  libxsmm_free(naive_libxsmm_delfilter_f32);
  libxsmm_free(input_libxsmm);
  libxsmm_free(output_libxsmm);
  libxsmm_free(delinput_libxsmm);
  libxsmm_free(deloutput_libxsmm);
  libxsmm_free(filter_libxsmm);
  libxsmm_free(delfilter_libxsmm);
  libxsmm_free(naive_bias);
  libxsmm_free(naive_delbias);
  libxsmm_free(naive_bias_bf16);
  libxsmm_free(naive_delbias_bf16);
  libxsmm_free(naive_libxsmm_delbias_bf16);
  libxsmm_free(naive_libxsmm_delbias_f32);
  libxsmm_free(relumask_libxsmm);
  libxsmm_free(bias_libxsmm);
  libxsmm_free(delbias_libxsmm);

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

