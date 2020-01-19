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
  int N =             ( argc == 6 ) ? atoi(argv[1]) : 64;
  int C =             ( argc == 6 ) ? atoi(argv[2]) : 512;
  int K =             ( argc == 6 ) ? atoi(argv[3]) : 32;
  unsigned int SPAR = ( argc == 6 ) ? atoi(argv[4]) : 90;
  unsigned int REPS = ( argc == 6 ) ? atoi(argv[5]) : 1;

  const libxsmm_gemm_prefetch_type prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  const int flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  const float alpha = 1, beta = 1;

  unsigned int* l_colptr = NULL;
  unsigned int* l_rowidx = NULL;
  float* l_b_de = (float*)libxsmm_aligned_malloc(sizeof(float) * C * K, 64);
  float* l_b_sp = NULL;
  float* l_a = (float*)libxsmm_aligned_malloc(sizeof(float) * N * C, 64);
  float* l_c = (float*)libxsmm_aligned_malloc(sizeof(float) * N * K, 64);
  float* l_c_gold = (float*)libxsmm_aligned_malloc(sizeof(float) * N * K, 64);
  float* l_c_asm = (float*)libxsmm_aligned_malloc(sizeof(float) * N * K, 64);
  float l_max_error = 0.0;
  unsigned int l_k, l_n;
  int l_i, l_j, l_jj;

  LIBXSMM_VLA_DECL(2, float, l_p_b_de, l_b_de, C);
  LIBXSMM_VLA_DECL(3, float, l_p_a, l_a, C, 16);
  LIBXSMM_VLA_DECL(3, float, l_p_c_asm, l_c_asm, K, 16);
  LIBXSMM_VLA_DECL(3, float, l_p_c_gold, l_c_gold, K, 16);

  libxsmm_descriptor_blob l_xgemm_blob;
  const libxsmm_gemm_descriptor* l_xgemm_desc = 0;
  LIBXSMM_MMFUNCTION_TYPE(float) mykernel = NULL;

  unsigned long long l_start, l_end;
  double l_total;
  int NB, nb;
  int nnz = 0;
  double sparse_frac = ((double)SPAR/(double)100.0);

  if (argc != 6 && argc != 1) {
    fprintf( stderr, "arguments failure\n" );
    return -1;
  }

  if ( N % 16 != 0 ) {
    fprintf( stderr, "N needs to be disable by 16\n" );
    return -1;
  }

  NB = N / 16;
  nb = 16;

  /* touch A */
  for ( l_i = 0; l_i < NB; l_i++) {
    for ( l_j = 0; l_j < C; l_j++) {
      for ( l_k = 0; l_k < nb; l_k++ ) {
        LIBXSMM_VLA_ACCESS(3, l_p_a, l_i, l_j, l_k, C, nb) = (float)libxsmm_rng_f64();
      }
    }
  }

  /* touch dense B */
  for ( l_i = 0; l_i < K; l_i++ ) {
    for ( l_j = 0; l_j < C; l_j++ ) {
      double tmp = libxsmm_rng_f64();
      if ( tmp < sparse_frac ) {
        tmp = (double)0;
      } else {
        nnz++;
      }
      LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, C) = tmp;
    }
  }
  printf("we just generated a %i x %i matrix with %i NZ entries\n", K, C, nnz);

  /* touch C */
  for ( l_i = 0; l_i < NB; l_i++) {
    for ( l_j = 0; l_j < K; l_j++) {
      for ( l_k = 0; l_k < nb; l_k++ ) {
        LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_i, l_j, l_k, K, nb) = (float)0.0;
        LIBXSMM_VLA_ACCESS(3, l_p_c_asm,  l_i, l_j, l_k, K, nb) = (float)0.0;
      }
    }
  }

  /* create B, csc */
  l_colptr = (unsigned int*) libxsmm_aligned_malloc( (K+1)*sizeof(unsigned int), 64 );
  l_rowidx = (unsigned int*) libxsmm_aligned_malloc( nnz*sizeof(unsigned int),   64 );
  l_b_sp   = (float*       ) libxsmm_aligned_malloc( nnz*sizeof(float),          64 );
  l_k = 0;
  l_colptr[K] = nnz;
  for ( l_i = 0; l_i < K; l_i++ ) {
    l_colptr[l_i] = l_k;
    for ( l_j = 0; l_j < C; l_j++ ) {
      if ( LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, C) != 0.0 ) {
        l_rowidx[l_k] = l_j;
        l_b_sp[l_k] = LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, C);
        l_k++;
      }
    }
  }

  /* dense routine */
  l_start = libxsmm_timer_tick();
#if 1
  for ( l_n = 0; l_n < REPS; l_n++) {
    for ( l_i = 0; l_i < NB; l_i++) {
      for ( l_j = 0; l_j < K; l_j++) {
        for ( l_jj = 0; l_jj < C; l_jj++) {
          LIBXSMM_PRAGMA_SIMD
          for (l_k = 0; l_k < nb; l_k++) {
            LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_i, l_j, l_k, K, nb)
              +=   LIBXSMM_VLA_ACCESS(3, l_p_a, l_i, l_jj, l_k, C, nb)
                 * l_b_de[(l_j*C)+l_jj];
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

  l_xgemm_desc = libxsmm_gemm_descriptor_dinit(&l_xgemm_blob, LIBXSMM_GEMM_PRECISION(float),
    NB, K, C, C, 0, K, alpha, beta, flags, prefetch);

  /* sparse routine */
  mykernel = libxsmm_create_xcsc_soa(l_xgemm_desc, l_colptr, l_rowidx, (const void*)l_b_sp).smm;

  l_start = libxsmm_timer_tick();
  for ( l_n = 0; l_n < REPS; l_n++) {
    mykernel( l_a, l_b_sp, l_c_asm );
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("%fs for sparse (asm)\n", l_total);
  printf("%f GFLOPS for sparse (asm)\n", ((double)((double)REPS * (double)N * (double)C * (double)K) * 2.0) / (l_total * 1.0e9));

  /* check for errors */
  l_max_error = (float)0.0;
  for ( l_i = 0; l_i < NB; l_i++) {
    for ( l_j = 0; l_j < K; l_j++) {
      for ( l_k = 0; l_k < nb; l_k++ ) {
        if (fabs( LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_i, l_j, l_k, K, nb)
                    - LIBXSMM_VLA_ACCESS(3, l_p_c_asm, l_i, l_j, l_k, K, nb) ) > l_max_error ) {
          l_max_error = (float)fabs( LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_i, l_j, l_k, K, nb)
                                       -LIBXSMM_VLA_ACCESS(3, l_p_c_asm, l_i, l_j, l_k, K, nb) );
        }
      }
    }
  }
  printf("max error: %f\n", l_max_error);

  printf("PERFDUMP,%u,%i,%i,%i,%u,%f,%f,%f\n", REPS, N, C, K, nnz, l_max_error, l_total, ((double)((double)REPS * (double)N * (double)C * (double)K) * 2.0) / (l_total * 1.0e9) );

  /* free */
  libxsmm_free( l_b_de );
  libxsmm_free( l_a );
  libxsmm_free( l_c_gold );
  libxsmm_free( l_c_asm );

  libxsmm_free( l_b_sp );
  libxsmm_free( l_colptr );
  libxsmm_free( l_rowidx );

  return 0;
}

