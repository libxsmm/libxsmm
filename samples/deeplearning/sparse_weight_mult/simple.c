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
#include <stdio.h>
#include <stdlib.h>
#include <libxsmm.h>

int main(int argc, char* argv[]) {
  unsigned int N =     ( argc == 6 ) ? atoi(argv[1]) : 64;
  unsigned int C =     ( argc == 6 ) ? atoi(argv[2]) : 512;
  unsigned int K =     ( argc == 6 ) ? atoi(argv[3]) : 32;
  double sparse_frac = ( argc == 6 ) ? atof(argv[4]) : 0.90;
  unsigned int REPS  = ( argc == 6 ) ? atoi(argv[5]) : 1;

  unsigned int* l_rowptr = NULL;
  unsigned int* l_colidx = NULL;
  float* l_a_de = (float*)libxsmm_aligned_malloc(sizeof(float) * C * K, 64);
  float* l_a_sp_csr = NULL;
  float* l_b = (float*)libxsmm_aligned_malloc(sizeof(float) * N * C, 64);
  float* l_c = (float*)libxsmm_aligned_malloc(sizeof(float) * N * K, 64);
  float* l_c_gold = (float*)libxsmm_aligned_malloc(sizeof(float) * N * K, 64);
  float* l_c_asm_csr = (float*)libxsmm_aligned_malloc(sizeof(float) * N * K, 64);
  float l_max_error = 0.0;
  unsigned int l_k, l_n;
  unsigned int l_i, l_j, l_jj;

  LIBXSMM_VLA_DECL(2, float, l_p_a_de, l_a_de, K);
  LIBXSMM_VLA_DECL(3, float, l_p_b, l_b, N/16, 16);
  LIBXSMM_VLA_DECL(3, float, l_p_c_asm_csr, l_c_asm_csr, N/16, 16);
  LIBXSMM_VLA_DECL(3, float, l_p_c_gold, l_c_gold, N/16, 16);

  unsigned long long l_start, l_end;
  double l_total;
  unsigned int NB, nb;
  unsigned int nnz = 0;

  if (argc != 6 && argc != 1) {
    fprintf( stderr, "arguments failure\n" );
    return -1;
  }

  if ( N % 64 != 0 ) {
    fprintf( stderr, "N needs to be disable by 64\n" );
    return -1;
  }

  NB = N / 16;
  nb = 16;

  /* touch B */
  for ( l_i = 0; l_i < C; l_i++) {
    for ( l_j = 0; l_j < NB; l_j++) {
      for ( l_k = 0; l_k < nb; l_k++ ) {
        LIBXSMM_VLA_ACCESS(3, l_p_b, l_i, l_j, l_k, NB, nb) = (float)libxsmm_rng_f64();
      }
    }
  }

  /* touch dense A */
  for ( l_i = 0; l_i < K; l_i++ ) {
    for ( l_j = 0; l_j < C; l_j++ ) {
      double tmp = libxsmm_rng_f64();
      if ( tmp < sparse_frac ) {
        tmp = (double)0;
      } else {
        nnz++;
      }
      LIBXSMM_VLA_ACCESS(2, l_p_a_de, l_i, l_j, C) = tmp;
    }
  }
  printf("we just generated a %i x %i matrix with %i NZ entries\n", K, C, nnz);

  /* touch C */
  for ( l_i = 0; l_i < K; l_i++) {
    for ( l_j = 0; l_j < NB; l_j++) {
      for ( l_k = 0; l_k < nb; l_k++ ) {
        LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_i, l_j, l_k, NB, nb) = (float)0.0;
        LIBXSMM_VLA_ACCESS(3, l_p_c_asm_csr,  l_i, l_j, l_k, NB, nb) = (float)0.0;
      }
    }
  }

  /* create A, csr */
  l_rowptr   = (unsigned int*) libxsmm_aligned_malloc( (K+1)*sizeof(unsigned int), 64 );
  l_colidx   = (unsigned int*) libxsmm_aligned_malloc( nnz*sizeof(unsigned int),   64 );
  l_a_sp_csr = (float*       ) libxsmm_aligned_malloc( nnz*sizeof(float),          64 );
  l_k = 0;
  l_rowptr[K] = nnz;
  for ( l_i = 0; l_i < K; l_i++ ) {
    l_rowptr[l_i] = l_k;
    for ( l_j = 0; l_j < C; l_j++ ) {
      if ( LIBXSMM_VLA_ACCESS(2, l_p_a_de, l_i, l_j, C) != 0.0 ) {
        l_colidx[l_k] = l_j;
        l_a_sp_csr[l_k] = LIBXSMM_VLA_ACCESS(2, l_p_a_de, l_i, l_j, C);
        l_k++;
      }
    }
  }

  /* dense routine */
  l_start = libxsmm_timer_tick();
#if 1
  for ( l_n = 0; l_n < REPS; l_n++) {
    #pragma omp parallel for private(l_j, l_jj, l_i, l_k)
    for ( l_j = 0; l_j < K; l_j++) {
      for ( l_jj = 0; l_jj < C; l_jj++) {
        for ( l_i = 0; l_i < NB; l_i++) {
          LIBXSMM_PRAGMA_SIMD
          for (l_k = 0; l_k < nb; l_k++) {
            LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_j, l_i, l_k, NB, nb)
              +=   LIBXSMM_VLA_ACCESS(3, l_p_b, l_jj, l_i, l_k, NB, nb)
                 * l_a_de[(l_j*C)+l_jj];
          }
        }
      }
    }
  }
#endif
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("%fs for dense\n", l_total);
  printf("%f GFLOPS for dense\n", ((double)((double)REPS * (double)N * (double)C * (double)K) * 2.0) / (l_total * 1.0e9));

  /* sparse routine */
  l_start = libxsmm_timer_tick();
  for ( l_n = 0; l_n < REPS; l_n++) {
    for ( l_i = 0; l_i < N; l_i+= 64 ) {
      #pragma omp parallel for private(l_j,l_k)
      for ( l_k = 0; l_k < K; l_k++) {
#if 0
#if 1
        __m512 c0 = _mm512_loadu_ps( &l_c_asm_csr[(l_k*N)+l_i   ] );
        __m512 c1 = _mm512_loadu_ps( &l_c_asm_csr[(l_k*N)+l_i+16] );
        __m512 c2 = _mm512_loadu_ps( &l_c_asm_csr[(l_k*N)+l_i+32] );
        __m512 c3 = _mm512_loadu_ps( &l_c_asm_csr[(l_k*N)+l_i+48] );
#else
        __m256 c0 = _mm256_loadu_ps( &l_c_asm_csr[(l_k*N)+l_i   ] );
        __m256 c1 = _mm256_loadu_ps( &l_c_asm_csr[(l_k*N)+l_i+ 8] );
        __m256 c2 = _mm256_loadu_ps( &l_c_asm_csr[(l_k*N)+l_i+16] );
        __m256 c3 = _mm256_loadu_ps( &l_c_asm_csr[(l_k*N)+l_i+24] );
        __m256 c4 = _mm256_loadu_ps( &l_c_asm_csr[(l_k*N)+l_i+32] );
        __m256 c5 = _mm256_loadu_ps( &l_c_asm_csr[(l_k*N)+l_i+40] );
        __m256 c6 = _mm256_loadu_ps( &l_c_asm_csr[(l_k*N)+l_i+48] );
        __m256 c7 = _mm256_loadu_ps( &l_c_asm_csr[(l_k*N)+l_i+56] );
#endif
#endif
        for ( l_j = 0; l_j < l_rowptr[l_k+1] - l_rowptr[l_k]; l_j++) {
#if 1
          unsigned int l_ii;
          LIBXSMM_PRAGMA_SIMD
          for ( l_ii = 0; l_ii < 64; l_ii++ ) {
            l_c_asm_csr[(l_k*N)+l_i+l_ii] += l_a_sp_csr[l_rowptr[l_k]+l_j] * l_b[(l_colidx[l_rowptr[l_k]+l_j]*N)+l_i+l_ii];
          }
#else
#if 1
          c0 = _mm512_fmadd_ps( _mm512_set1_ps( l_a_sp_csr[l_rowptr[l_k] + l_j] ), _mm512_loadu_ps( &l_b[(l_colidx[l_rowptr[l_k] + l_j]*N) + l_i   ] ), c0 );
          c1 = _mm512_fmadd_ps( _mm512_set1_ps( l_a_sp_csr[l_rowptr[l_k] + l_j] ), _mm512_loadu_ps( &l_b[(l_colidx[l_rowptr[l_k] + l_j]*N) + l_i+16] ), c1 );
          c2 = _mm512_fmadd_ps( _mm512_set1_ps( l_a_sp_csr[l_rowptr[l_k] + l_j] ), _mm512_loadu_ps( &l_b[(l_colidx[l_rowptr[l_k] + l_j]*N) + l_i+32] ), c2 );
          c3 = _mm512_fmadd_ps( _mm512_set1_ps( l_a_sp_csr[l_rowptr[l_k] + l_j] ), _mm512_loadu_ps( &l_b[(l_colidx[l_rowptr[l_k] + l_j]*N) + l_i+48] ), c3 );
#else
          c0 = _mm256_fmadd_ps( _mm256_set1_ps( l_a_sp_csr[l_rowptr[l_k] + l_j] ), _mm256_loadu_ps( &l_b[(l_colidx[l_rowptr[l_k] + l_j]*N) + l_i   ] ), c0 );
          c1 = _mm256_fmadd_ps( _mm256_set1_ps( l_a_sp_csr[l_rowptr[l_k] + l_j] ), _mm256_loadu_ps( &l_b[(l_colidx[l_rowptr[l_k] + l_j]*N) + l_i+ 8] ), c1 );
          c2 = _mm256_fmadd_ps( _mm256_set1_ps( l_a_sp_csr[l_rowptr[l_k] + l_j] ), _mm256_loadu_ps( &l_b[(l_colidx[l_rowptr[l_k] + l_j]*N) + l_i+16] ), c2 );
          c3 = _mm256_fmadd_ps( _mm256_set1_ps( l_a_sp_csr[l_rowptr[l_k] + l_j] ), _mm256_loadu_ps( &l_b[(l_colidx[l_rowptr[l_k] + l_j]*N) + l_i+24] ), c3 );
          c4 = _mm256_fmadd_ps( _mm256_set1_ps( l_a_sp_csr[l_rowptr[l_k] + l_j] ), _mm256_loadu_ps( &l_b[(l_colidx[l_rowptr[l_k] + l_j]*N) + l_i+32] ), c4 );
          c5 = _mm256_fmadd_ps( _mm256_set1_ps( l_a_sp_csr[l_rowptr[l_k] + l_j] ), _mm256_loadu_ps( &l_b[(l_colidx[l_rowptr[l_k] + l_j]*N) + l_i+40] ), c5 );
          c6 = _mm256_fmadd_ps( _mm256_set1_ps( l_a_sp_csr[l_rowptr[l_k] + l_j] ), _mm256_loadu_ps( &l_b[(l_colidx[l_rowptr[l_k] + l_j]*N) + l_i+48] ), c6 );
          c7 = _mm256_fmadd_ps( _mm256_set1_ps( l_a_sp_csr[l_rowptr[l_k] + l_j] ), _mm256_loadu_ps( &l_b[(l_colidx[l_rowptr[l_k] + l_j]*N) + l_i+56] ), c7 );
#endif
#endif
#if 0
          _mm_prefetch( &l_b[(l_colidx[l_rowptr[l_k] + l_j]*N) + l_i+ 64], _MM_HINT_T1 );
          _mm_prefetch( &l_b[(l_colidx[l_rowptr[l_k] + l_j]*N) + l_i+ 80], _MM_HINT_T1 );
          _mm_prefetch( &l_b[(l_colidx[l_rowptr[l_k] + l_j]*N) + l_i+ 96], _MM_HINT_T1 );
          _mm_prefetch( &l_b[(l_colidx[l_rowptr[l_k] + l_j]*N) + l_i+112], _MM_HINT_T1 );
#endif
        }
#if 0
#if 1
        _mm512_storeu_ps( &l_c_asm_csr[(l_k*N)+l_i]   , c0 );
        _mm512_storeu_ps( &l_c_asm_csr[(l_k*N)+l_i+16], c1 );
        _mm512_storeu_ps( &l_c_asm_csr[(l_k*N)+l_i+32], c2 );
        _mm512_storeu_ps( &l_c_asm_csr[(l_k*N)+l_i+48], c3 );
#else
        _mm256_storeu_ps( &l_c_asm_csr[(l_k*N)+l_i]   , c0 );
        _mm256_storeu_ps( &l_c_asm_csr[(l_k*N)+l_i+ 8], c1 );
        _mm256_storeu_ps( &l_c_asm_csr[(l_k*N)+l_i+16], c2 );
        _mm256_storeu_ps( &l_c_asm_csr[(l_k*N)+l_i+24], c3 );
        _mm256_storeu_ps( &l_c_asm_csr[(l_k*N)+l_i+32], c4 );
        _mm256_storeu_ps( &l_c_asm_csr[(l_k*N)+l_i+40], c5 );
        _mm256_storeu_ps( &l_c_asm_csr[(l_k*N)+l_i+48], c6 );
        _mm256_storeu_ps( &l_c_asm_csr[(l_k*N)+l_i+56], c7 );
#endif
#endif
      }
    }
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("%fs for sparse (asm, csr)\n", l_total);
  printf("%f GFLOPS for sparse (asm, csr)\n", ((double)((double)REPS * (double)N * (double)C * (double)K) * 2.0) / (l_total * 1.0e9));

  /* check for errors */
  l_max_error = (float)0.0;
  for ( l_i = 0; l_i < NB; l_i++) {
    for ( l_j = 0; l_j < K; l_j++) {
      for ( l_k = 0; l_k < nb; l_k++ ) {
        if (fabs( LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_j, l_i, l_k, NB, nb)
                    - LIBXSMM_VLA_ACCESS(3, l_p_c_asm_csr, l_j, l_i, l_k, NB, nb) ) > l_max_error ) {
          l_max_error = (float)fabs( LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_j, l_i, l_k, NB, nb)
                                       -LIBXSMM_VLA_ACCESS(3, l_p_c_asm_csr, l_j, l_i, l_k, NB, nb) );
        }
      }
    }
  }
  printf("max error (csr): %f\n", l_max_error);

  /* free */
  libxsmm_free( l_a_de );
  libxsmm_free( l_b );
  libxsmm_free( l_c_gold );
  libxsmm_free( l_c_asm_csr );

  libxsmm_free( l_a_sp_csr );
  libxsmm_free( l_rowptr );
  libxsmm_free( l_colidx );

  return 0;
}

