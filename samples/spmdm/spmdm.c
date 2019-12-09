/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Nadathur Satish (Intel Corp.)
******************************************************************************/

/* NOTE: This code currently ignores alpha input to the matrix multiply */
#include <libxsmm.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

#if !defined(USE_BFLOAT) && 0
# define USE_BFLOAT
typedef libxsmm_bfloat16 REAL_TYPE;
#else
typedef float REAL_TYPE;
#endif


LIBXSMM_INLINE void spmdm_check_c(const libxsmm_spmdm_handle* handle, float* test, float* gold)
{
  double max_error = 0.0;
  double src_norm = 0.0;
  double dst_norm = 0.0;
  size_t l;

  for (l = 0; l < (size_t)handle->m * (size_t)handle->n; ++l) {
    const double dstval = (double)test[l];
    const double srcval = (double)gold[l];
    const double local_error = fabs(dstval - srcval);
    if (local_error > max_error) {
      max_error = local_error;
    }
    /*if (local_error > 1e-3) printf("(%d,%d) : gold: %f, computed: %f\n", l / handle->n, l % handle->n, srcval, dstval);*/
    src_norm += srcval;
    dst_norm += dstval;
  }

  printf(" max error: %f, sum BLAS: %f, sum LIBXSMM: %f \n", max_error, src_norm, dst_norm );
}


LIBXSMM_INLINE
void spmdm_exec_fp32( const libxsmm_spmdm_handle* handle,
                      const char transA,
                      const char transB,
                      const float* alpha,
                      const float* A,
                      const float* B,
                      const char transC,
                      const float* beta,
                      float* C,
                      libxsmm_CSR_sparseslice* A_sparse) {
  int num_createSparseSlice_blocks = libxsmm_spmdm_get_num_createSparseSlice_blocks(handle);
  int num_compute_blocks = libxsmm_spmdm_get_num_compute_blocks(handle);
  int i;

# if defined(_OPENMP)
# pragma omp parallel private(i)
# endif
  {
# if defined(_OPENMP)
    const int nthreads = omp_get_num_threads();
    const int tid = omp_get_thread_num();
# else
    const int nthreads = 1;
    const int tid = 0;
# endif
# if defined(_OPENMP)
#   pragma omp for
# endif
    for (i = 0; i < num_createSparseSlice_blocks; ++i) {
      libxsmm_spmdm_createSparseSlice_fp32_thread(handle, transA, A, A_sparse, i, tid, nthreads);
    }
# if defined(_OPENMP)
#   pragma omp for
# endif
    for (i = 0; i < num_compute_blocks; ++i) {
      libxsmm_spmdm_compute_fp32_thread(handle, transA, transB, alpha, A_sparse, B, transC, beta, C, i, tid, nthreads);
    }
  }
}


LIBXSMM_INLINE
void spmdm_exec_bfloat16( const libxsmm_spmdm_handle* handle,
                          const char transA,
                          const char transB,
                          const libxsmm_bfloat16* alpha,
                          const libxsmm_bfloat16* A,
                          const libxsmm_bfloat16* B,
                          const char transC,
                          const libxsmm_bfloat16* beta,
                          float* C,
                          libxsmm_CSR_sparseslice* A_sparse) {
  int num_createSparseSlice_blocks = libxsmm_spmdm_get_num_createSparseSlice_blocks(handle);
  int num_compute_blocks = libxsmm_spmdm_get_num_compute_blocks(handle);
  int i;

# if defined(_OPENMP)
# pragma omp parallel private(i)
# endif
  {
# if defined(_OPENMP)
    const int nthreads = omp_get_num_threads();
    const int tid = omp_get_thread_num();
# else
    const int nthreads = 1;
    const int tid = 0;
# endif
# if defined(_OPENMP)
#   pragma omp for
# endif
    for (i = 0; i < num_createSparseSlice_blocks; ++i) {
      libxsmm_spmdm_createSparseSlice_bfloat16_thread(handle, transA, A, A_sparse, i, tid, nthreads);
    }
# if defined(_OPENMP)
#   pragma omp for
# endif
    for (i = 0; i < num_compute_blocks; ++i) {
      libxsmm_spmdm_compute_bfloat16_thread(handle, transA, transB, alpha, A_sparse, B, transC, beta, C, i, tid, nthreads);
    }
  }
}


int main(int argc, char *argv[])
{
  REAL_TYPE *A_gold, *B_gold, *A_gold2, *B_gold2;
  float *C_gold, *C0_gold, *C, *C2;

  int M, N, K;
  REAL_TYPE alpha, beta;
  int reps;

  libxsmm_spmdm_handle handle, handle2;
  libxsmm_CSR_sparseslice *A_sparse, *A_sparse2;
  int max_threads;

  /* Step 1: Read in args */
  libxsmm_timer_tickint start, end;
  double flops, duration;
  char transA, transB, transC;
  int i, j, k;
  size_t l;

  /* Step 1: Initialize handle */
  M = 0; N = 0; K = 0; alpha = (REAL_TYPE)1.0; beta = (REAL_TYPE)0.0; reps = 0; transA = 'N'; transB = 'N';

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("\nUsage: %s [M] [N] [K] [transA] [transB] [reps]\n\n", argv[0]);
    return EXIT_SUCCESS;
  }

  /* defaults */
  M = 2048;
  N = 2048;
  K = 2048;
  transA = 'N';
  transB = 'N';
  transC = 'N';
  reps = 100;

  /* reading new values from cli */
  i = 1;
  if (argc > i) M = atoi(argv[i++]);
  if (argc > i) N = atoi(argv[i++]);
  if (argc > i) K = atoi(argv[i++]);
  if (argc > i) { transA = argv[i][0]; i++; }
  if (argc > i) { transB = argv[i][0]; i++; }
  if (argc > i) { transC = argv[i][0]; i++; }
  if (argc > i) reps = atoi(argv[i++]);

  /* Step 2: allocate data */
  A_gold  = (REAL_TYPE*)libxsmm_aligned_malloc( M*K*sizeof(REAL_TYPE), 64 );
  B_gold  = (REAL_TYPE*)libxsmm_aligned_malloc( K*N*sizeof(REAL_TYPE), 64 );
  C_gold  = (float*)libxsmm_aligned_malloc( M*N*sizeof(float), 64 );
  C0_gold = (float*)libxsmm_aligned_malloc( M*N*sizeof(float), 64 );
  C       = (float*)libxsmm_aligned_malloc( M*N*sizeof(float), 64 );

  /* Step 3: init data */
  libxsmm_rng_set_seed(1);
  for (l = 0; l < (size_t)M * (size_t)K; ++l) {
    const double r64 = libxsmm_rng_f64();
    const float r32 = (float)r64;
#ifdef USE_BFLOAT
    const int r = *(const int*)(&r32);
    const libxsmm_bfloat16 val = (r >> 16);
#else
    const float val = r32;
#endif
    if (r64 > 0.85) A_gold[l] = val;
    else              A_gold[l] = (REAL_TYPE)0.0;
  }

  for (l = 0; l < (size_t)K * (size_t)N; ++l) {
    const double r64 = libxsmm_rng_f64();
    const float r32 = (float)r64;
#ifdef USE_BFLOAT
    const int r = *(const int*)(&r32);
    const libxsmm_bfloat16 val = (r >> 16);
#else
    const float val = r32;
#endif
    B_gold[l] = val;
  }
  for (l = 0; l < (size_t)M * (size_t)N; ++l) {
    C0_gold[l] = (float)libxsmm_rng_f64();
    C_gold[l] = C0_gold[l];
  }
  for (l = 0; l < (size_t)M * (size_t)N; ++l) {
    C[l] = (float)C0_gold[l];
  }
  flops = (double)M * (double)N * (double)K * 2.0;

  /*----------------------------------------------------------------------------------------------------------------------*/
  /* Step 4: Initialize LIBXSMM for these sizes - allocates handle and temporary space for the sparse data structure for A */
# if defined(_OPENMP)
  max_threads = omp_get_max_threads();
# else
  max_threads = 1;
# endif

  start = libxsmm_timer_tick();
  libxsmm_spmdm_init(M, N, K, max_threads, &handle, &A_sparse);
  end = libxsmm_timer_tick();
  printf("Time for handle init = %f\n", libxsmm_timer_duration(start, end));

  printf(" running with: M=%i, N=%i, K=%i, bm=%i, bn=%i, bk=%i, mb=%i, nb=%i, kb=%i, reps=%i -- forward pass\n", M, N, K, handle.bm, handle.bn, handle.bk, handle.mb, handle.nb, handle.kb, reps );
  /* The overall function that takes in matrix inputs in dense format, does the conversion of A to sparse format and does the matrix multiply */
  /* Currently ignores alpha */
  /* TODO: fix alpha input */
# ifdef USE_BFLOAT
  spmdm_exec_bfloat16(&handle, transA, transB, &alpha, A_gold, B_gold, transC, &beta, C, A_sparse);
# else
  spmdm_exec_fp32(&handle, transA, transB, &alpha, A_gold, B_gold, transC, &beta, C, A_sparse);
# endif

  /* Checks */

  /* Compute a "gold" answer sequentially */
#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(k);
# pragma omp parallel for private(i, j, k) LIBXSMM_OPENMP_COLLAPSE(2)
#endif
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      float sum = 0.0;
      float Cval;
      for (k = 0; k < K; ++k) {
#       ifdef USE_BFLOAT
        libxsmm_bfloat16 Atmp = A_gold[i*K+k];
        int Atmp_int  = Atmp; Atmp_int <<= 16;
        float Aval = *(float *)&Atmp_int;
        libxsmm_bfloat16 Btmp = B_gold[k*N+j];
        int Btmp_int  = Btmp; Btmp_int <<= 16;
        float Bval = *(float *)&Btmp_int;
#       else
        float Aval = A_gold[i*K + k];
        float Bval = B_gold[k*N + j];
#       endif
        sum += Aval * Bval;
      }
      Cval = sum;
      C_gold[i*N + j] = Cval + beta*C_gold[i*N + j];
    }
  }
  /* LIBXSMM_FSYMBOL(sgemm)(&trans, &trans, &N, &M, &K, &alpha, B_gold, &N, A_gold, &K, &beta, C_gold, &N); */

  /* Compute the max difference between gold and computed results. */
  spmdm_check_c( &handle, C, C_gold );

  /* Timing loop starts */
  start = libxsmm_timer_tick();
  for (i = 0; i < reps; ++i) {
#   ifdef USE_BFLOAT
    spmdm_exec_bfloat16( &handle, transA, transB, &alpha, A_gold, B_gold, transC, &beta, C, A_sparse);
#   else
    spmdm_exec_fp32( &handle, transA, transB, &alpha, A_gold, B_gold, transC, &beta, C, A_sparse);
#   endif
  }
  end = libxsmm_timer_tick();
  duration = libxsmm_timer_duration(start, end);
  printf("Time = %f Time/rep = %f, TFlops/s = %f\n", duration, duration*1.0/reps, flops/1000./1000./1000./1000./duration*reps);
  libxsmm_spmdm_destroy(&handle);

  /*----------------------------------------------------------------------------------------------------------------------*/
  /* Step 5: Initialize libxsmm for transpose A - allocates handle and temporary space for the sparse data structure for A */
  transA = 'T'; transB = 'N'; transC = 'T';
  libxsmm_spmdm_init(M, N, K, max_threads, &handle2, &A_sparse2);
  printf(" running with: M=%i, N=%i, K=%i, bm=%i, bn=%i, bk=%i, mb=%i, nb=%i, kb=%i, reps=%i, transA = Y, transC = Y -- weight update\n", handle2.m, handle2.n, handle2.k, handle2.bm, handle2.bn, handle2.bk, handle2.mb, handle2.nb, handle2.kb, reps );
  A_gold2 = (REAL_TYPE*)libxsmm_aligned_malloc( M*K*sizeof(REAL_TYPE), 64 );
  C2 = (float*)libxsmm_aligned_malloc( M*N*sizeof(float), 64 );

  for (i = 0; i < M; ++i) {
    for (j = 0; j < K; ++j) {
      A_gold2[j*M + i] = A_gold[i*K + j];
    }
  }
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      C[j*M + i] = (float)C0_gold[i*N + j];
    }
  }
  /* The overall function that takes in matrix inputs in dense format, does the conversion of A to sparse format and does the matrix multiply */
  /* Currently ignores alpha */
  /* TODO: fix alpha inputs */
# ifdef USE_BFLOAT
  spmdm_exec_bfloat16( &handle2, transA, transB, &alpha, A_gold2, B_gold, transC, &beta, C, A_sparse2);
# else
  spmdm_exec_fp32( &handle2, transA, transB, &alpha, A_gold2, B_gold, transC, &beta, C, A_sparse2);
# endif

  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      C2[i*N + j] = C[j*M + i];
    }
  }
  /* Checks */
  spmdm_check_c( &handle2, C2, C_gold);

  /* Timing loop starts */
  start = libxsmm_timer_tick();
  for (i = 0; i < reps; ++i) {
#   ifdef USE_BFLOAT
    spmdm_exec_bfloat16( &handle2, transA, transB, &alpha, A_gold2, B_gold, transC, &beta, C, A_sparse2);
#   else
    spmdm_exec_fp32( &handle2, transA, transB, &alpha, A_gold2, B_gold, transC, &beta, C, A_sparse2);
#   endif
  }
  end = libxsmm_timer_tick();
  duration = libxsmm_timer_duration(start, end);
  printf("Time = %f Time/rep = %f, TFlops/s = %f\n", duration, duration*1.0/reps, flops/1000./1000./1000./1000./duration*reps);

  /*----------------------------------------------------------------------------------------------------------------------*/
  /* Step 6: Test transpose B */
  transA = 'N'; transB = 'T'; transC = 'N';
  printf(" running with: M=%i, N=%i, K=%i, bm=%i, bn=%i, bk=%i, mb=%i, nb=%i, kb=%i, reps=%i, transB = Y -- backprop\n", handle2.m, handle2.n, handle2.k, handle2.bm, handle2.bn, handle2.bk, handle2.mb, handle2.nb, handle2.kb, reps );
  B_gold2 = (REAL_TYPE*)libxsmm_aligned_malloc( K*N*sizeof(REAL_TYPE), 64 );

  for (i = 0; i < K; ++i) {
    for (j = 0; j < N; ++j) {
      B_gold2[j*K + i] = B_gold[i*N + j];
    }
  }
  for (l = 0; l < (size_t)M * (size_t)N; ++l) {
    C[l] = (float)C0_gold[l];
  }
  /* The overall function that takes in matrix inputs in dense format, does the conversion of A to sparse format and does the matrix multiply */
  /* Currently ignores alpha */
  /* TODO: fix alpha inputs */
# ifdef USE_BFLOAT
  spmdm_exec_bfloat16( &handle2, transA, transB, &alpha, A_gold, B_gold2, transC, &beta, C, A_sparse2);
# else
  spmdm_exec_fp32( &handle2, transA, transB, &alpha, A_gold, B_gold2, transC, &beta, C, A_sparse2);
# endif

  /* Checks */
  spmdm_check_c( &handle2, C, C_gold);

  /* Timing loop starts */
  start = libxsmm_timer_tick();
  for (i = 0; i < reps; ++i) {
#   ifdef USE_BFLOAT
    spmdm_exec_bfloat16( &handle2, transA, transB, &alpha, A_gold, B_gold2, transC, &beta, C, A_sparse2);
#   else
    spmdm_exec_fp32( &handle2, transA, transB, &alpha, A_gold, B_gold2, transC, &beta, C, A_sparse2);
#   endif
  }
  end = libxsmm_timer_tick();
  duration = libxsmm_timer_duration(start, end);
  printf("Time = %f Time/rep = %f, TFlops/s = %f\n", duration, duration*1.0/reps, flops/1000./1000./1000./1000./duration*reps);
  libxsmm_spmdm_destroy(&handle2);

  libxsmm_free(A_gold);
  libxsmm_free(B_gold);
  libxsmm_free(C_gold);
  libxsmm_free(C);
  libxsmm_free(C2);
  libxsmm_free(C0_gold);
  libxsmm_free(B_gold2);
  libxsmm_free(A_gold2);

  return EXIT_SUCCESS;
}

