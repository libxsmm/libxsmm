/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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
#if defined(_WIN32) || defined(__CYGWIN__)
/* note: this does not reproduce 48-bit RNG quality */
# define drand48() ((double)rand() / RAND_MAX)
# define srand48 srand
#endif

/* #define USE_BFLOAT */
#ifdef USE_BFLOAT
typedef uint16_t real;
#else
typedef float real;
#endif

LIBXSMM_INLINE
void spmdm_check_c( const libxsmm_spmdm_handle* handle,
                    float* test,
                    float* gold) {
  double max_error = 0.0;
  double src_norm = 0.0;
  double dst_norm = 0.0;
  size_t l;

  for ( l = 0; l < (size_t)handle->m * (size_t)handle->n; l++ ) {
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
# pragma omp parallel
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
    for ( i = 0; i < num_createSparseSlice_blocks; i++ ) {
      libxsmm_spmdm_createSparseSlice_fp32_thread( handle, transA, A, A_sparse, i, tid, nthreads);
    }
# if defined(_OPENMP)
#   pragma omp for
# endif
    for ( i = 0; i < num_compute_blocks; i++ ) {
      libxsmm_spmdm_compute_fp32_thread( handle, transA, transB, alpha, A_sparse, B, transC, beta, C, i, tid, nthreads);
    }
  }
}

LIBXSMM_INLINE
void spmdm_exec_bfloat16( const libxsmm_spmdm_handle* handle,
                          const char transA,
                          const char transB,
                          const uint16_t* alpha,
                          const uint16_t* A,
                          const uint16_t* B,
                          const char transC,
                          const uint16_t* beta,
                          float* C,
                          libxsmm_CSR_sparseslice* A_sparse) {
  int num_createSparseSlice_blocks = libxsmm_spmdm_get_num_createSparseSlice_blocks(handle);
  int num_compute_blocks = libxsmm_spmdm_get_num_compute_blocks(handle);

  int i;
# if defined(_OPENMP)
# pragma omp parallel
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
    for ( i = 0; i < num_createSparseSlice_blocks; i++ ) {
      libxsmm_spmdm_createSparseSlice_bfloat16_thread( handle, transA, A, A_sparse, i, tid, nthreads);
    }
# if defined(_OPENMP)
#   pragma omp for
# endif
    for ( i = 0; i < num_compute_blocks; i++ ) {
      libxsmm_spmdm_compute_bfloat16_thread( handle, transA, transB, alpha, A_sparse, B, transC, beta, C, i, tid, nthreads);
    }
  }
}

int main(int argc, char *argv[])
{
  real *A_gold, *B_gold, *A_gold2, *B_gold2;
  float *C_gold, *C0_gold, *C, *C2;

  int M, N, K;
  real alpha, beta;
  int reps;

  libxsmm_spmdm_handle handle, handle2;
  libxsmm_CSR_sparseslice *A_sparse, *A_sparse2;
  int max_threads;

  /* Step 1: Read in args */
  unsigned long long start, end;
  double flops, duration;
  char transA, transB, transC;
  int i, j, k;
  size_t l;

  /* Step 1: Initalize handle */
  M = 0; N = 0; K = 0; alpha = (real)1.0; beta = (real)0.0;   reps = 0; transA = 'N'; transB = 'N';

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("\nUsage: ./block_gemm [M] [N] [K] [transA] [transB] [reps]\n\n");
    return 0;
  }

  /* defaults */
  M  = 2048;
  N = 2048;
  K = 2048;
  transA = 'N';
  transB = 'N';
  transC = 'N';
  reps = 100;

  /* reading new values from cli */
  i = 1;
  if (argc > i) M      = atoi(argv[i++]);
  if (argc > i) N      = atoi(argv[i++]);
  if (argc > i) K      = atoi(argv[i++]);
  if (argc > i) { transA = argv[i][0]; i++; }
  if (argc > i) { transB = argv[i][0]; i++; }
  if (argc > i) { transC = argv[i][0]; i++; }
  if (argc > i) reps   = atoi(argv[i++]);

  /* Step 2: allocate data */
  A_gold = (real*)libxsmm_aligned_malloc( M*K*sizeof(real), 64 );
  B_gold = (real*)libxsmm_aligned_malloc( K*N*sizeof(real), 64 );
  C_gold = (float*)libxsmm_aligned_malloc( M*N*sizeof(float), 64 );
  C0_gold = (float*)libxsmm_aligned_malloc( M*N*sizeof(float), 64 );
  C      = (float*)libxsmm_aligned_malloc( M*N*sizeof(float), 64 );

  /* Step 3: init data */
  srand48(1);
  for ( l = 0; l < (size_t)M * (size_t)K; l++ ) {
    double random = drand48();
    #ifdef USE_BFLOAT
    float  random_f = (float)random;
    int    random_int = *(int *)(&random_f);
    uint16_t val = (random_int>>16);
    #else
    float  val = (float)random;
    #endif
    if (random > 0.85) A_gold[l] = val;
    else              A_gold[l] = (real)0.0;
  }

  for ( l = 0; l < (size_t)K * (size_t)N; l++ ) {
    double random = drand48();
    #ifdef USE_BFLOAT
    float  random_f = (float)random;
    int    random_int = *(int *)(&random_f);
    uint16_t val = (random_int>>16);
    #else
    float  val = (float)random;
    #endif
    B_gold[l] = val;
  }
  for ( l = 0; l < (size_t)M * (size_t)N; l++ ) {
    C0_gold[l] = (float)drand48();
    C_gold[l] = C0_gold[l];
  }
  for ( l = 0; l < (size_t)M * (size_t)N; l++ ) {
    C[l]      = (float)C0_gold[l];
  }
  flops = (double)M * (double)N * (double)K * 2.0;

  /*----------------------------------------------------------------------------------------------------------------------*/
  /* Step 4: Initialize libxsmm for these sizes - allocates handle and temporary space for the sparse data structure for A */
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
  spmdm_exec_bfloat16( &handle, transA, transB, &alpha, A_gold, B_gold, transC, &beta, C, A_sparse);
# else
  spmdm_exec_fp32( &handle, transA, transB, &alpha, A_gold, B_gold, transC, &beta, C, A_sparse);
# endif

  /* Checks */

  /* Compute a "gold" answer sequentially - we can also use MKL; not using MKL now due to difficulty for bfloat16 */
#if defined(_OPENMP)
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2)
#endif
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      float sum = 0.0;
      float Cval;
      for (k = 0; k < K; k++) {
#       ifdef USE_BFLOAT
        uint16_t Atmp = A_gold[i*K + k];
        int Atmp_int  = Atmp; Atmp_int <<= 16;
        float Aval = *(float *)&Atmp_int;
        uint16_t Btmp = B_gold[k*N + j];
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
  for ( i = 0; i < reps; i++) {
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
  A_gold2 = (real*)libxsmm_aligned_malloc( M*K*sizeof(real), 64 );
  C2 = (float*)libxsmm_aligned_malloc( M*N*sizeof(float), 64 );

  for (i = 0; i < M; i++) {
    for (j = 0; j < K; j++) {
      A_gold2[j*M + i] = A_gold[i*K + j];
    }
  }
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
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

  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      C2[i*N + j] = C[j*M + i];
    }
  }
  /* Checks */
  spmdm_check_c( &handle2, C2, C_gold);

  /* Timing loop starts */
  start = libxsmm_timer_tick();
  for ( i = 0; i < reps; i++) {
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
  /* Step 6: Test transpose B  */
  transA = 'N'; transB = 'T'; transC = 'N';
  printf(" running with: M=%i, N=%i, K=%i, bm=%i, bn=%i, bk=%i, mb=%i, nb=%i, kb=%i, reps=%i, transB = Y -- backprop\n", handle2.m, handle2.n, handle2.k, handle2.bm, handle2.bn, handle2.bk, handle2.mb, handle2.nb, handle2.kb, reps );
  B_gold2 = (real*)libxsmm_aligned_malloc( K*N*sizeof(real), 64 );

  for (i = 0; i < K; i++) {
    for (j = 0; j < N; j++) {
      B_gold2[j*K + i] = B_gold[i*N + j];
    }
  }
  for ( l = 0; l < (size_t)M * (size_t)N; l++ ) {
    C[l]      = (float)C0_gold[l];
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
  for ( i = 0; i < reps; i++) {
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

  return 0;
}

