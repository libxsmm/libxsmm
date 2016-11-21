/******************************************************************************
 * ** Copyright (c) 2016, Intel Corporation                                     **
 * ** All rights reserved.                                                      **
 * **                                                                           **
 * ** Redistribution and use in source and binary forms, with or without        **
 * ** modification, are permitted provided that the following conditions        **
 * ** are met:                                                                  **
 * ** 1. Redistributions of source code must retain the above copyright         **
 * **    notice, this list of conditions and the following disclaimer.          **
 * ** 2. Redistributions in binary form must reproduce the above copyright      **
 * **    notice, this list of conditions and the following disclaimer in the    **
 * **    documentation and/or other materials provided with the distribution.   **
 * ** 3. Neither the name of the copyright holder nor the names of its          **
 * **    contributors may be used to endorse or promote products derived        **
 * **    from this software without specific prior written permission.          **
 * **                                                                           **
 * ** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
 * ** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
 * ** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
 * ** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
 * ** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
 * ** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
 * ** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
 * ** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
 * ** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
 * ** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
 * ** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
 * ******************************************************************************/
/* Nadathur Satish (Intel Corp.)
 * ******************************************************************************/

/* NOTE: This code currently ignores alpha, beta and trans inputs to the matrix multiply */
#include <libxsmm_spmdm.h>
#include <libxsmm.h>
#include <libxsmm_intrinsics_x86.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>


#if defined(_WIN32) || defined(__CYGWIN__)
/* note: this does not reproduce 48-bit RNG quality */
# define drand48() ((double)rand() / RAND_MAX)
# define srand48 srand
#endif

//#define USE_BFLOAT
#ifdef USE_BFLOAT
typedef uint16_t real;
#else
typedef float real;
#endif

void libxsmm_spmdm_check_c( const libxsmm_spmdm_handle* handle,
                               real* test,
                               real* gold) {
  int mb, nb, bm, bn;
  double max_error = 0.0;
  double src_norm = 0.0;
  double dst_norm = 0.0;
  size_t l;

  for ( l = 0; l < (size_t)handle->m * (size_t)handle->n; l++ ) {
    const double dstval = (double)test[l];
    const double srcval = (double)gold[l];
    const double local_error = fabs(dstval - srcval);
    //if(local_error > 0.01) printf("l: %lld, gold: %lf actual: %lf local_error: %lf\n", l, srcval, dstval, local_error);
    if (local_error > max_error) {
      max_error = local_error;
    }
    src_norm += srcval;
    dst_norm += dstval;
  }

  printf(" max error: %f, sum BLAS: %f, sum LIBXSMM: %f \n", max_error, src_norm, dst_norm );
}

void libxsmm_spmdm_exec_fp32( const libxsmm_spmdm_handle* handle,
                            const char transA,
                            const float* alpha,
                            const float* A,
                            const float* B,
                            const float* beta,
                            float* C,
                            libxsmm_CSR_sparseslice* A_sparse) {

  int m_blocks = handle->mb;
  int n_blocks = handle->nb;
  int k_blocks = handle->kb;
  int mb, nb, kb;
  if( transA == 'Y') {
  }
  else {
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int nthreads = omp_get_num_threads();
      #pragma omp for collapse(2)
      for ( kb = 0; kb < k_blocks; kb++ ) {
        for ( mb = 0; mb < m_blocks; mb++ ) {
          libxsmm_spmdm_createSparseSlice_fp32_notrans_thread( handle, A, transA, A_sparse, mb, kb, tid, nthreads);
        }
      }
      #if 1
      int num_m_blocks = 1;
      #pragma omp for collapse(2)
      for (mb= 0; mb < m_blocks; mb += num_m_blocks) {
        for ( nb = 0; nb < n_blocks; nb++ ) {
          libxsmm_spmdm_compute_fp32_thread( handle, alpha, A_sparse, B, beta, C, mb, num_m_blocks, nb, tid, nthreads);
        }
      }
      #endif
    }
  }

}

void libxsmm_spmdm_exec_bfloat16( const libxsmm_spmdm_handle* handle,
                            const char transA,
                            const uint16_t* alpha,
                            const uint16_t* A,
                            const uint16_t* B,
                            const uint16_t* beta,
                            uint16_t* C,
                            libxsmm_CSR_sparseslice* A_sparse
				) {

  int m_blocks = handle->mb;
  int n_blocks = handle->nb;
  int k_blocks = handle->kb;
  int mb, nb, kb;
  if( transA == 'Y') {
  }
  else {
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int nthreads = omp_get_num_threads();
      #pragma omp for collapse(2)
      for ( kb = 0; kb < k_blocks; kb++ ) {
        for ( mb = 0; mb < m_blocks; mb++ ) {
          libxsmm_spmdm_createSparseSlice_bfloat16_notrans_thread( handle, A, transA, A_sparse, mb, kb, tid, nthreads);
        }
      }
      #if 1
      int num_m_blocks = 1;
      #pragma omp for collapse(2)
      for (mb= 0; mb < m_blocks; mb += num_m_blocks) {
        for ( nb = 0; nb < n_blocks; nb++ ) {
          libxsmm_spmdm_compute_bfloat16_thread( handle, alpha, A_sparse, B, beta, C, mb, num_m_blocks, nb, tid, nthreads);
        }
      }
      #endif
    }
  }
}

int main(int argc, char **argv)
{
  real *A_gold, *B_gold, *C_gold, *C;

  int M, N, K;
  real alpha, beta;
  int reps;
  double start, end, flops; 
  char trans;
  int i, j, k;

  /* Step 1: Read in args */
  M = 0; N = 0; K = 0; alpha = (real)1.0; beta = (real)1.0;   reps = 0; trans = 'N';

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("\nUsage: ./block_gemm [M] [N] [K] [bm] [bn] [bk] [reps]\n\n");
    return 0;
  }

  /* defaults */
  M  = 2048;
  N = 2048;
  K = 2048;
  reps = 100;

  /* reading new values from cli */
  i = 1;
  if (argc > i) M      = atoi(argv[i++]);
  if (argc > i) N      = atoi(argv[i++]);
  if (argc > i) K      = atoi(argv[i++]);
  if (argc > i) reps          = atoi(argv[i++]);

  /* Step 2: allocate data */
  A_gold = (real*)libxsmm_aligned_malloc( M*K*sizeof(real), 2097152 );
  B_gold = (real*)libxsmm_aligned_malloc( K*N*sizeof(real), 2097152 );
  C_gold = (real*)libxsmm_aligned_malloc( M*N*sizeof(real), 2097152 );
  C      = (real*)libxsmm_aligned_malloc( M*N*sizeof(real), 2097152 );

  /* Step 3: init data */
  srand48(1);
  size_t l;
  for ( l = 0; l < (size_t)M * (size_t)N; l++ ) {
    double random = drand48();
    #ifdef USE_BFLOAT 
    float  random_f = (float)random;
    int    random_int = *(int *)(&random_f);
    uint16_t val = (random_int>>16);
    #else
    float  val = (float)random;
    #endif 
    if(random > 0.85) A_gold[l] = val;
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
    C_gold[l] = (real)0.0;
  }
  for ( l = 0; l < (size_t)M * (size_t)N; l++ ) {
    C[l]      = (real)0.0;
  }
  flops = (double)M * (double)N * (double)K * 2.0;

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /* Step 4: Initialize libxsmm for these sizes - allocates handle and temporary space for the sparse data structure for A */
  libxsmm_spmdm_handle handle;
  libxsmm_CSR_sparseslice* A_sparse;
  libxsmm_spmdm_init(M, N, K, trans, &handle, &A_sparse);
  printf(" running with: M=%i, N=%i, K=%i, bm=%i, bn=%i, bk=%i, mb=%i, nb=%i, kb=%i, reps=%i\n", M, N, K, handle.bm, handle.bn, handle.bk, handle.mb, handle.nb, handle.kb, reps );

  /* The overall function that takes in matrix inputs in dense format, does the conversion of A to sparse format and does the matrix multiply */
  /* Currently ignores alpha, beta and trans */
  /* TODO: fix alpha, beta and trans inputs */
  #ifdef USE_BFLOAT 
  libxsmm_spmdm_exec_bfloat16( &handle, trans, &alpha, A_gold, B_gold, &beta, C, A_sparse);
  #else
  libxsmm_spmdm_exec_fp32( &handle, trans, &alpha, A_gold, B_gold, &beta, C, A_sparse);
  #endif

  /* Checks */
  /* Has A been correctly converted? */
  //libxsmm_test_a ( &handle, A_gold, trans, A_sparse);

  /* Compute a "gold" answer sequentially - we can also use MKL; not using MKL now due to difficulty for bfloat16 */
  #pragma omp parallel for collapse(2)
  for(i = 0; i < M; i++) {
    for(j = 0; j < N; j++) {
      float sum = 0.0;
      for(k = 0; k < K; k++) {
        #ifdef USE_BFLOAT
        uint16_t Atmp = A_gold[i*K + k];
        int Atmp_int  = Atmp; Atmp_int <<= 16;
        float Aval = *(float *)&Atmp_int;
        uint16_t Btmp = B_gold[k*N + j];
        int Btmp_int  = Btmp; Btmp_int <<= 16;
        float Bval = *(float *)&Btmp_int;
        #else
        float Aval = A_gold[i*K + k];
        float Bval = B_gold[k*N + j];
        #endif 
        sum += Aval * Bval;
      }
      #ifdef USE_BFLOAT
      int v = *(int *)(&sum);
      uint16_t Cval = (v >> 16);
      #else
      float Cval = sum;
      #endif
      C_gold[i*N + j] += Cval;
    }
  }
  //LIBXSMM_FSYMBOL(sgemm)(&trans, &trans, &N, &M, &K, &alpha, B_gold, &N, A_gold, &K, &beta, C_gold, &N);
  
  /* Compute the max difference between gold and computed results. */
  libxsmm_spmdm_check_c( &handle, C, C_gold );
 
  /* Timing loop starts */ 
  start = omp_get_wtime();  
  for( i = 0; i < reps; i++) {
    #ifdef USE_BFLOAT 
    libxsmm_spmdm_exec_bfloat16( &handle, trans, &alpha, A_gold, B_gold, &beta, C, A_sparse);
    #else
    libxsmm_spmdm_exec_fp32( &handle, trans, &alpha, A_gold, B_gold, &beta, C, A_sparse);
    #endif
  }
  end = omp_get_wtime();  
  printf("Time = %lf Time/rep = %lf, TFlops/s = %lf\n", (end - start), (end - start)*1.0/reps, flops/1000./1000./1000./1000./(end-start)*reps);
}

