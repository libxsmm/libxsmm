/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <utils/libxsmm_utils.h>
#include <libxsmm.h>


int main(int argc, char* argv[]) {
  libxsmm_blasint N =     ( argc > 1 ) ? atoi(argv[1]) : 64;
  libxsmm_blasint C =     ( argc > 2 ) ? atoi(argv[2]) : 512;
  libxsmm_blasint K =     ( argc > 3 ) ? atoi(argv[3]) : 32;
  libxsmm_blasint nb =    ( argc > 4 ) ? atoi(argv[4]) : 16;
  double sparse_frac = ( argc > 5 ) ? atof(argv[5]) : 0.90;
  unsigned int REPS  = ( argc > 6 ) ? atoi(argv[6]) : 1;

  unsigned int* l_rowptr = NULL;
  unsigned int* l_colidx = NULL;
  float* l_a_de = (float*)libxsmm_aligned_malloc(sizeof(float) * C * K, 64);
  float* l_a_sp_csr = NULL;
  float* l_b = (float*)libxsmm_aligned_malloc(sizeof(float) * N * C, 64);
  float* l_c_gold = (float*)libxsmm_aligned_malloc(sizeof(float) * N * K, 64);
  float* l_c_asm_csr = (float*)libxsmm_aligned_malloc(sizeof(float) * N * K, 64);
  float l_max_error = 0.0;
  libxsmm_blasint l_k, l_n;
  libxsmm_blasint l_i, l_j, l_jj;
  libxsmm_blasint NB = N / nb;

  LIBXSMM_VLA_DECL(2, float, l_p_a_de, l_a_de, C);
  LIBXSMM_VLA_DECL(3, float, l_p_b, l_b, NB, nb);
  LIBXSMM_VLA_DECL(3, float, l_p_c_asm_csr, l_c_asm_csr, NB, nb);
  LIBXSMM_VLA_DECL(3, float, l_p_c_gold, l_c_gold, NB, nb);

  const libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  const libxsmm_bitfield l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
  const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(
      K, NB, C, 0, NB, NB, LIBXSMM_DATATYPE(float),
      LIBXSMM_DATATYPE(float), LIBXSMM_DATATYPE(float), LIBXSMM_DATATYPE(float) );
  libxsmm_gemm_param gemm_param;
  libxsmm_gemmfunction mykernel_csr = NULL;

  unsigned long long l_start, l_end;
  double l_total;
  unsigned int nnz = 0;

  memset( &gemm_param, 0, sizeof(libxsmm_gemm_param) );

  if (argc != 7 && argc != 1) {
    fprintf( stderr, "arguments failure\n" );
    return -1;
  }

  if ( N % nb != 0 ) {
    fprintf( stderr, "N needs to be disable by nb\n" );
    return -1;
  }

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
      float tmp = (float)libxsmm_rng_f64();
      if ( tmp < sparse_frac ) {
        tmp = 0;
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
        LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_i, l_j, l_k, NB, nb) = 0.f;
        LIBXSMM_VLA_ACCESS(3, l_p_c_asm_csr,  l_i, l_j, l_k, NB, nb) = 0.f;
      }
    }
  }

  /* create B, csr */
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
  for ( l_n = 0; l_n < (libxsmm_blasint)REPS; l_n++) {
    for ( l_i = 0; l_i < NB; l_i++) {
      for ( l_j = 0; l_j < K; l_j++) {
        for ( l_jj = 0; l_jj < C; l_jj++) {
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
  mykernel_csr = libxsmm_create_packed_spgemm_csr_v2(gemm_shape, l_flags, l_prefetch_flags, nb,
    l_rowptr, l_colidx, (const void*)l_a_sp_csr);

  gemm_param.a.primary = l_a_sp_csr;
  gemm_param.b.primary = l_b;
  gemm_param.c.primary = l_c_asm_csr;
   l_start = libxsmm_timer_tick();
  for ( l_n = 0; l_n < (libxsmm_blasint)REPS; l_n++) {
    mykernel_csr( &gemm_param );
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("%fs for sparse (asm, csr)\n", l_total);
  printf("%f GFLOPS for sparse (asm, csr)\n", ((double)((double)REPS * (double)N * (double)C * (double)K) * 2.0) / (l_total * 1.0e9));

  /* check for errors */
  l_max_error = 0.f;
  for ( l_i = 0; l_i < NB; l_i++) {
    for ( l_j = 0; l_j < K; l_j++) {
      for ( l_k = 0; l_k < nb; l_k++ ) {
        if (LIBXSMM_FABSF( LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_j, l_i, l_k, NB, nb)
                    - LIBXSMM_VLA_ACCESS(3, l_p_c_asm_csr, l_j, l_i, l_k, NB, nb) ) > l_max_error ) {
          l_max_error = LIBXSMM_FABSF( LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_j, l_i, l_k, NB, nb)
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

