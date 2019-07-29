/******************************************************************************
** Copyright (c) 2016-2019, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
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
  float *naive_input0, *naive_output0, *naive_filter0, *naive_output1, *naive_filter1, *naive_output2, *naive_filter2, *naive_output3, *naive_filter3;
  float *naive_libxsmm_output0, *naive_libxsmm_output1, *naive_libxsmm_output2, *naive_libxsmm_output3;
  float *input_libxsmm0, *output_libxsmm0, *filter_libxsmm0, *output_libxsmm1, *filter_libxsmm1, *output_libxsmm2, *filter_libxsmm2, *output_libxsmm3, *filter_libxsmm3;

  naive_fullyconnected_t naive_param;
  void* scratch;
  size_t scratch_size = 0;

  /* some parameters we can overwrite via cli,
     default is some inner layer of overfeat */
  int iters = 10;         /* repetitions of benchmark */
  int nImg = 32;          /* mini-batch size, "N" */
  int nIFm = 256;          /* number of input feature maps, "C" */
  int nOFm = 256;          /* number of input feature maps, "C" */
  int fuse_type = 0;      /* 0: nothing fused, 1: relu fused, 2: elementwise fused, 3: relu and elementwise fused */
  char type = 'A';        /* 'A': ALL, 'F': FP, 'B': BP, 'U', WU */
  char format = 'B';
  int bn = 64;
  int bk = 64;
  int bc = 64;

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

  libxsmm_dnn_fullyconnected_desc fullyconnected_desc;
  libxsmm_dnn_fullyconnected* libxsmm_handle0;
  libxsmm_dnn_fullyconnected* libxsmm_handle1;
  libxsmm_dnn_fullyconnected* libxsmm_handle2;
  libxsmm_dnn_fullyconnected* libxsmm_handle3;
  libxsmm_dnn_tensor*  libxsmm_input0;
  libxsmm_dnn_tensor*  libxsmm_output0;
  libxsmm_dnn_tensor*  libxsmm_filter0;
  libxsmm_dnn_tensor*  libxsmm_output1;
  libxsmm_dnn_tensor*  libxsmm_filter1;
  libxsmm_dnn_tensor*  libxsmm_output2;
  libxsmm_dnn_tensor*  libxsmm_filter2;
  libxsmm_dnn_tensor*  libxsmm_output3;
  libxsmm_dnn_tensor*  libxsmm_filter3;
  libxsmm_dnn_tensor_datalayout* libxsmm_layout;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status = LIBXSMM_DNN_SUCCESS;

  libxsmm_matdiff_info norms_fwd, norms_bwd, norms_upd, diff;
  libxsmm_matdiff_clear(&norms_fwd);
  libxsmm_matdiff_clear(&norms_bwd);
  libxsmm_matdiff_clear(&norms_upd);
  libxsmm_matdiff_clear(&diff);

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("Usage: %s iters nImg nIFm nOFm fuse_type type format\n", argv[0]);
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
  if (argc > i) format     = *(argv[i++]);
  if (argc > i) bn         = atoi(argv[i++]);
  if (argc > i) bk         = atoi(argv[i++]);
  if (argc > i) bc         = atoi(argv[i++]);

  if (type != 'A' && type != 'F' && type != 'B' && type != 'U') {
    printf("type needs to be 'A' (All), 'F' (FP only), 'B' (BP only), 'U' (UP only)\n");
    return -1;
  }
  if ( fuse_type != 0 ) {
    printf("fuse type needs to be 0\n");
    return -1;
  }
  if (format != 'B') {
    printf("format needs to be 'B' (for locked NCNC KCCK)\n");
    return -1;
  }
  if ( nIFm != nOFm ) {
    printf("nIFm needs to be equal to nOFm\n");
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
  printf("SIZE Input  (MB): %10.2f MiB\n", (double)(nImg*nIFm*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Output (MB): %10.2f MiB\n", (double)(nImg*nOFm*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Input   (1): %10.2f MiB\n", (double)(1*nIFm*   sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Output  (1): %10.2f MiB\n", (double)(1*nOFm*   sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Filter     : %10.2f MiB\n", (double)(nIFm*nOFm*sizeof(float))/(1024.0*1024.0) );

  /* allocate data */
  naive_input0                = (float*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(float), 2097152);
  naive_output0               = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);
  naive_filter0               = (float*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(float), 2097152);
  naive_output1               = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);
  naive_filter1               = (float*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(float), 2097152);
  naive_output2               = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);
  naive_filter2               = (float*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(float), 2097152);
  naive_output3               = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);
  naive_filter3               = (float*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(float), 2097152);

  naive_libxsmm_output0       = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);
  naive_libxsmm_output1       = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);
  naive_libxsmm_output2       = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);
  naive_libxsmm_output3       = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);

  input_libxsmm0              = (float*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(float), 2097152);
  output_libxsmm0             = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);
  filter_libxsmm0             = (float*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(float), 2097152);
  output_libxsmm1             = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);
  filter_libxsmm1             = (float*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(float), 2097152);
  output_libxsmm2             = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);
  filter_libxsmm2             = (float*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(float), 2097152);
  output_libxsmm3             = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);
  filter_libxsmm3             = (float*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(float), 2097152);

  /* initialize data */
  init_buf( naive_input0,     nImg*nIFm, 0, 0 );
  init_buf( naive_output0,    nImg*nOFm, 0, 0 );
  init_buf( naive_filter0,    nIFm*nOFm, 0, 0 );
  init_buf( naive_output1,    nImg*nOFm, 0, 0 );
  init_buf( naive_filter1,    nIFm*nOFm, 0, 0 );
  init_buf( naive_output2,    nImg*nOFm, 0, 0 );
  init_buf( naive_filter2,    nIFm*nOFm, 0, 0 );
  init_buf( naive_output3,    nImg*nOFm, 0, 0 );
  init_buf( naive_filter3,    nIFm*nOFm, 0, 0 );

  if (LIBXSMM_NEQ(0, check)) {
    printf("##########################################\n");
    printf("#         Computing Reference ...        #\n");
    printf("##########################################\n");
    if (type == 'A' || type == 'F') {
      naive_fullyconnected_fp(&naive_param, naive_input0,  naive_output0, naive_filter0);
      naive_fullyconnected_fp(&naive_param, naive_output0, naive_output1, naive_filter1);
      naive_fullyconnected_fp(&naive_param, naive_output1, naive_output2, naive_filter2);
      naive_fullyconnected_fp(&naive_param, naive_output2, naive_output3, naive_filter3);
    }
    printf("##########################################\n");
    printf("#      Computing Reference ... done      #\n");
    printf("##########################################\n");
  }

  if (format == 'B') {
    printf("\n");
    printf("##########################################\n");
    printf("#      Setting Up  (custom-Storage)      #\n");
    printf("##########################################\n");

    /* setup LIBXSMM handle */
    fullyconnected_desc.N = nImg;
    fullyconnected_desc.C = nIFm;
    fullyconnected_desc.K = nOFm;
    fullyconnected_desc.bn = bn;
    fullyconnected_desc.bk = bk;
    fullyconnected_desc.bc = bc;
    fullyconnected_desc.threads = nThreads;
    fullyconnected_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    fullyconnected_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
    fullyconnected_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_NCPACKED;
    fullyconnected_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_CKPACKED;
    fullyconnected_desc.fuse_ops = LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE;

    libxsmm_handle0 = libxsmm_dnn_create_fullyconnected( fullyconnected_desc, &status );
    libxsmm_handle1 = libxsmm_dnn_create_fullyconnected( fullyconnected_desc, &status );
    libxsmm_handle2 = libxsmm_dnn_create_fullyconnected( fullyconnected_desc, &status );
    libxsmm_handle3 = libxsmm_dnn_create_fullyconnected( fullyconnected_desc, &status );
    CHKERR_LIBXSMM_DNN( status );

    /* setup LIBXSMM buffers */
    libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_handle0, LIBXSMM_DNN_REGULAR_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_input0  = libxsmm_dnn_link_tensor( libxsmm_layout, input_libxsmm0, &status ); CHKERR_LIBXSMM_DNN( status );
    printf("inner activation blocking: %i\n", libxsmm_layout->dim_size[0] );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_handle0, LIBXSMM_DNN_REGULAR_OUTPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_output0  = libxsmm_dnn_link_tensor( libxsmm_layout, output_libxsmm0, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_output1  = libxsmm_dnn_link_tensor( libxsmm_layout, output_libxsmm1, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_output2  = libxsmm_dnn_link_tensor( libxsmm_layout, output_libxsmm2, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_output3  = libxsmm_dnn_link_tensor( libxsmm_layout, output_libxsmm3, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_handle0, LIBXSMM_DNN_REGULAR_FILTER, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_filter0  = libxsmm_dnn_link_tensor( libxsmm_layout, filter_libxsmm0, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_filter1  = libxsmm_dnn_link_tensor( libxsmm_layout, filter_libxsmm1, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_filter2  = libxsmm_dnn_link_tensor( libxsmm_layout, filter_libxsmm2, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_filter3  = libxsmm_dnn_link_tensor( libxsmm_layout, filter_libxsmm3, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    /* copy in data to LIBXSMM format */
    /* we can also use the layout functions and set the data on our
       own external to the library */
    matrix_copy_NC_to_NCNC( naive_input0,     input_libxsmm0,     1, nImg, nIFm, bn, bc );
    matrix_copy_NC_to_NCNC( naive_output0,    output_libxsmm0,    1, nImg, nOFm, bn, bk );
    matrix_copy_KC_to_KCCK( naive_filter0,    filter_libxsmm0      , nIFm, nOFm, bc, bk );
    matrix_copy_NC_to_NCNC( naive_output1,    output_libxsmm1,    1, nImg, nOFm, bn, bk );
    matrix_copy_KC_to_KCCK( naive_filter1,    filter_libxsmm1      , nIFm, nOFm, bc, bk );
    matrix_copy_NC_to_NCNC( naive_output2,    output_libxsmm2,    1, nImg, nOFm, bn, bk );
    matrix_copy_KC_to_KCCK( naive_filter2,    filter_libxsmm2      , nIFm, nOFm, bc, bk );
    matrix_copy_NC_to_NCNC( naive_output3,    output_libxsmm3,    1, nImg, nOFm, bn, bk );
    matrix_copy_KC_to_KCCK( naive_filter3,    filter_libxsmm3      , nIFm, nOFm, bc, bk );

    /* bind buffers and filter to handle */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle0, libxsmm_input0,        LIBXSMM_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle0, libxsmm_output0,       LIBXSMM_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle0, libxsmm_filter0,       LIBXSMM_DNN_REGULAR_FILTER ) );

    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle1, libxsmm_output0,        LIBXSMM_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle1, libxsmm_output1,       LIBXSMM_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle1, libxsmm_filter1,       LIBXSMM_DNN_REGULAR_FILTER ) );

    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle2, libxsmm_output1,        LIBXSMM_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle2, libxsmm_output2,       LIBXSMM_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle2, libxsmm_filter2,       LIBXSMM_DNN_REGULAR_FILTER ) );

    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle3, libxsmm_output2,        LIBXSMM_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle3, libxsmm_output3,       LIBXSMM_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle3, libxsmm_filter3,       LIBXSMM_DNN_REGULAR_FILTER ) );

    /* let's allocate and bind scratch */
    scratch_size = libxsmm_dnn_fullyconnected_get_scratch_size( libxsmm_handle0, &status );
    CHKERR_LIBXSMM_DNN( status );
    scratch = libxsmm_aligned_scratch( scratch_size, 2097152 );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_scratch( libxsmm_handle0, scratch ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_scratch( libxsmm_handle1, scratch ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_scratch( libxsmm_handle2, scratch ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_scratch( libxsmm_handle3, scratch ) );
    /* set scratch to bogus to make sure that libxsmm takes care of zeroing internally */
    init_buf( (float*)scratch, scratch_size/4, 0, 0 );

    if ((type == 'A' || type == 'F') && LIBXSMM_NEQ(0, check)) {
      printf("##########################################\n");
      printf("#   Correctness - FWD (custom-Storage)   #\n");
      printf("##########################################\n");

#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_execute_st( libxsmm_handle0, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_execute_st( libxsmm_handle1, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_execute_st( libxsmm_handle2, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_execute_st( libxsmm_handle3, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
      }

      /* copy out data */
      if ( format == 'L' ) {
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_output0, (void*)naive_libxsmm_output0, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_output1, (void*)naive_libxsmm_output1, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_output2, (void*)naive_libxsmm_output2, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_output3, (void*)naive_libxsmm_output3, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
      } else {
        matrix_copy_NCNC_to_NC( output_libxsmm0, naive_libxsmm_output0, 1, nImg, nOFm, bn, bk );
        matrix_copy_NCNC_to_NC( output_libxsmm1, naive_libxsmm_output1, 1, nImg, nOFm, bn, bk );
        matrix_copy_NCNC_to_NC( output_libxsmm2, naive_libxsmm_output2, 1, nImg, nOFm, bn, bk );
        matrix_copy_NCNC_to_NC( output_libxsmm3, naive_libxsmm_output3, 1, nImg, nOFm, bn, bk );
      }

      /* compare */
      libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, nImg*nOFm, 1, naive_output0, naive_libxsmm_output0, 0, 0);
      printf("Layer 0:\n");
      printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_fwd);
      libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, nImg*nOFm, 1, naive_output1, naive_libxsmm_output1, 0, 0);
      printf("Layer 1:\n");
      printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_fwd);
      libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, nImg*nOFm, 1, naive_output2, naive_libxsmm_output2, 0, 0);
      printf("Layer 2:\n");
      printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_fwd);
      libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, nImg*nOFm, 1, naive_output3, naive_libxsmm_output3, 0, 0);
      printf("Layer 3:\n");
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
          libxsmm_dnn_fullyconnected_execute_st( libxsmm_handle0, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid );
          libxsmm_dnn_fullyconnected_execute_st( libxsmm_handle1, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid );
          libxsmm_dnn_fullyconnected_execute_st( libxsmm_handle2, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid );
          libxsmm_dnn_fullyconnected_execute_st( libxsmm_handle3, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);

      gflop = (4.0*2.0*(double)nImg*(double)nIFm*(double)nOFm*(double)iters) / (1000*1000*1000);

      printf("GFLOP  = %.5g\n", gflop/(double)iters);
      printf("fp time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", gflop/l_total);

      printf("PERFDUMP,FP,%s,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIFm,
          nOFm, ((double)(l_total/iters)), gflop/l_total, norms_fwd.l1_ref, norms_fwd.l1_tst,
          norms_fwd.l2_abs, norms_fwd.l2_rel, norms_fwd.linf_abs, norms_fwd.linf_rel, norms_fwd.normf_rel);
    }

    /* clean-up */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_scratch( libxsmm_handle0 ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_scratch( libxsmm_handle1 ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_scratch( libxsmm_handle2 ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_scratch( libxsmm_handle3 ) );
    libxsmm_free(scratch);
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_handle0, LIBXSMM_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_handle0, LIBXSMM_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_handle0, LIBXSMM_DNN_REGULAR_FILTER ) );

    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_handle1, LIBXSMM_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_handle1, LIBXSMM_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_handle1, LIBXSMM_DNN_REGULAR_FILTER ) );

    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_handle2, LIBXSMM_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_handle2, LIBXSMM_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_handle2, LIBXSMM_DNN_REGULAR_FILTER ) );

    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_handle3, LIBXSMM_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_handle3, LIBXSMM_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_handle3, LIBXSMM_DNN_REGULAR_FILTER ) );

    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_input0 ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_output0 ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_filter0 ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_output1 ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_filter1 ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_output2 ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_filter2 ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_output3 ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_filter3 ) );

    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_fullyconnected( libxsmm_handle0 ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_fullyconnected( libxsmm_handle1 ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_fullyconnected( libxsmm_handle2 ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_fullyconnected( libxsmm_handle3 ) );
  }

  /* deallocate data */
  libxsmm_free(naive_input0);
  libxsmm_free(naive_output0);
  libxsmm_free(naive_filter0);
  libxsmm_free(naive_output1);
  libxsmm_free(naive_filter1);
  libxsmm_free(naive_output2);
  libxsmm_free(naive_filter2);
  libxsmm_free(naive_output3);
  libxsmm_free(naive_filter3);

  libxsmm_free(naive_libxsmm_output0);
  libxsmm_free(naive_libxsmm_output1);
  libxsmm_free(naive_libxsmm_output2);
  libxsmm_free(naive_libxsmm_output3);

  libxsmm_free(input_libxsmm0);
  libxsmm_free(output_libxsmm0);
  libxsmm_free(filter_libxsmm0);
  libxsmm_free(output_libxsmm1);
  libxsmm_free(filter_libxsmm1);
  libxsmm_free(output_libxsmm2);
  libxsmm_free(filter_libxsmm2);
  libxsmm_free(output_libxsmm3);
  libxsmm_free(filter_libxsmm3);

  { const char *const env_check_scale = getenv("CHECK_SCALE");
    const double check_scale = LIBXSMM_ABS(0 == env_check_scale ? 1.0 : atof(env_check_scale));
    if (LIBXSMM_NEQ(0, check) && (check < 100.0 * check_scale * diff.normf_rel) && (global_status == LIBXSMM_DNN_SUCCESS)) {
      fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
      exit(EXIT_FAILURE);
    }
  }

  /* some empty lines at the end */
  printf("\n\n\n");

  return EXIT_SUCCESS;
}

