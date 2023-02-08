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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <libxsmm.h>

int main(int argc, char* argv[]) {
  libxsmm_blasint N =     ( argc > 1 ) ? atoi(argv[1]) : 64;
  libxsmm_blasint C =     ( argc > 2 ) ? atoi(argv[2]) : 512;
  libxsmm_blasint K =     ( argc > 3 ) ? atoi(argv[3]) : 32;
  libxsmm_blasint nb =    ( argc > 4 ) ? atoi(argv[4]) : 16;
  double sparse_frac = ( argc > 5 ) ? atof(argv[5]) : 0.90;
  unsigned int REPS  = ( argc > 6 ) ? atoi(argv[6]) : 1;
  unsigned int use_bf16  = ( argc > 7 ) ? atoi(argv[7]) : 0;
  libxsmm_blasint NB = N / nb;

  unsigned int* l_colptr = NULL;
  unsigned int* l_rowidx = NULL;
  unsigned int* l_rowptr = NULL;
  unsigned int* l_colidx = NULL;
  float* l_b_de = (float*)libxsmm_aligned_malloc(sizeof(float) * C * K, 64);
  float* l_b_sp_csc = NULL;
  libxsmm_bfloat16* l_b_sp_csc_bf16 = NULL;
  float* l_b_sp_csr = NULL;
  float* l_a = (float*)libxsmm_aligned_malloc(sizeof(float) * N * C, 64);
  libxsmm_bfloat16* l_a_bf16 = (libxsmm_bfloat16*)libxsmm_aligned_malloc(sizeof(libxsmm_bfloat16) * N * C, 64);
  float* l_c_gold = (float*)libxsmm_aligned_malloc(sizeof(float) * N * K, 64);
  float* l_c_asm_csc = (float*)libxsmm_aligned_malloc(sizeof(float) * N * K, 64);
  libxsmm_bfloat16* l_c_asm_csc_bf16 = (libxsmm_bfloat16*)libxsmm_aligned_malloc(sizeof(libxsmm_bfloat16) * N * K, 64);
  float* l_c_asm_csr = (float*)libxsmm_aligned_malloc(sizeof(float) * N * K, 64);
  float l_max_error = 0.0;
  libxsmm_blasint l_k, l_n;
  libxsmm_blasint l_i, l_j, l_jj;
  libxsmm_datatype dtype = (use_bf16 == 0) ? LIBXSMM_DATATYPE_F32 : LIBXSMM_DATATYPE_BF16;

  LIBXSMM_VLA_DECL(2, float, l_p_b_de, l_b_de, C);
  LIBXSMM_VLA_DECL(3, float, l_p_a, l_a, C, nb);
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, l_p_a_bf16, l_a_bf16, C, nb);
  LIBXSMM_VLA_DECL(3, float, l_p_c_asm_csc, l_c_asm_csc, K, nb);
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, l_p_c_asm_csc_bf16, l_c_asm_csc_bf16, K, nb);
  LIBXSMM_VLA_DECL(3, float, l_p_c_asm_csr, l_c_asm_csr, K, nb);
  LIBXSMM_VLA_DECL(3, float, l_p_c_gold, l_c_gold, K, nb);

  const libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  const libxsmm_bitfield l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
  const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(
      NB, K, C, C, 0, K, dtype,
      dtype, dtype, LIBXSMM_DATATYPE(float) );
  libxsmm_gemm_param gemm_param;
  libxsmm_gemmfunction mykernel_csc = NULL;
  libxsmm_gemmfunction mykernel_csr = NULL;

  unsigned long long l_start, l_end;
  double l_total;
  unsigned int nnz = 0;

  libxsmm_matdiff_info norms_csc, diff;
  libxsmm_matdiff_clear(&norms_csc);
  libxsmm_matdiff_clear(&diff);

  memset( &gemm_param, 0, sizeof(libxsmm_gemm_param) );

  if (argc != 8 && argc != 1) {
    fprintf( stderr, "arguments failure\n" );
    return -1;
  }

  /* touch A */
  for ( l_i = 0; l_i < NB; l_i++) {
    for ( l_j = 0; l_j < C; l_j++) {
      for ( l_k = 0; l_k < nb; l_k++ ) {
        LIBXSMM_VLA_ACCESS(3, l_p_a, l_i, l_j, l_k, C, nb) = (float)libxsmm_rng_f64();
        if (use_bf16 > 0) {
          libxsmm_rne_convert_fp32_bf16( &LIBXSMM_VLA_ACCESS(3, l_p_a, l_i, l_j, l_k, C, nb), &LIBXSMM_VLA_ACCESS(3, l_p_a_bf16, l_i, l_j, l_k, C, nb), 1);
          libxsmm_convert_bf16_f32( &LIBXSMM_VLA_ACCESS(3, l_p_a_bf16, l_i, l_j, l_k, C, nb), &LIBXSMM_VLA_ACCESS(3, l_p_a, l_i, l_j, l_k, C, nb), 1 );
        }
      }
    }
  }

  /* touch dense B */
  for ( l_i = 0; l_i < K; l_i++ ) {
    for ( l_j = 0; l_j < C; l_j++ ) {
      LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, C) = 0;
    }
  }

  if (use_bf16 > 0 || 1) {
    for ( l_i = 0; l_i < K-1; l_i++ ) {
      for ( l_j = 0; l_j < C-3; l_j++ ) {
        if (LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, C) == 0) {
          /* Only in this case triger initialization, otherwise it has been taken care of */
          float tmp = (float)libxsmm_rng_f64();
          if ( tmp < (1.0-(1.0-sparse_frac)/8.0) ) {
            tmp = 0;
            LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, C) = tmp;
          } else {
            LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, C) = tmp;
            LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j+1, C) = tmp;
            LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j+2, C) = tmp;
            LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j+3, C) = tmp;
            LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i+1, l_j, C) = tmp;
            LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i+1, l_j+1, C) = tmp;
            LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i+1, l_j+2, C) = tmp;
            LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i+1, l_j+3, C) = tmp;
          }
        }
      }
    }
  } else {
    for ( l_i = 0; l_i < K; l_i++ ) {
      for ( l_j = 0; l_j < C; l_j++ ) {
        float tmp = (float)libxsmm_rng_f64();
        if ( tmp < sparse_frac ) {
          tmp = 0;
        }
        LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, C) = tmp;
      }
    }
  }

  nnz = 0;
  for ( l_i = 0; l_i < K; l_i++ ) {
    for ( l_j = 0; l_j < C; l_j++ ) {
      if (LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, C) != 0) {
        LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, C) = (float)libxsmm_rng_f64();
        while (LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, C) == 0) {
          LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, C) = (float)libxsmm_rng_f64();
        }
        nnz++;
      }
      //if (l_j == 0) printf("Col %d: ", l_i);
      //printf("%.2g\t", LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, C) );
    }
    //printf("\n");
  }

  printf("we just generated a %i x %i matrix with %i NZ entries\n", K, C, nnz);

  /* touch C */
  for ( l_i = 0; l_i < NB; l_i++) {
    for ( l_j = 0; l_j < K; l_j++) {
      for ( l_k = 0; l_k < nb; l_k++ ) {
        LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_i, l_j, l_k, K, nb) = 0.f;
        LIBXSMM_VLA_ACCESS(3, l_p_c_asm_csc,  l_i, l_j, l_k, K, nb) = 0.f;
        LIBXSMM_VLA_ACCESS(3, l_p_c_asm_csc_bf16,  l_i, l_j, l_k, K, nb) = (libxsmm_bfloat16)0;
        LIBXSMM_VLA_ACCESS(3, l_p_c_asm_csr,  l_i, l_j, l_k, K, nb) = 0.f;
      }
    }
  }

  /* create B, csc */
  l_colptr   = (unsigned int*) libxsmm_aligned_malloc( (K+1)*sizeof(unsigned int), 64 );
  l_rowidx   = (unsigned int*) libxsmm_aligned_malloc( nnz*sizeof(unsigned int),   64 );
  l_b_sp_csc = (float*       ) libxsmm_aligned_malloc( nnz*sizeof(float),          64 );
  l_b_sp_csc_bf16 = (libxsmm_bfloat16*) libxsmm_aligned_malloc( nnz*sizeof(libxsmm_bfloat16),          64 );
  l_k = 0;
  l_colptr[K] = nnz;
  for ( l_i = 0; l_i < K; l_i++ ) {
    l_colptr[l_i] = l_k;
    for ( l_j = 0; l_j < C; l_j++ ) {
      if ( LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, C) != 0.0 ) {
        l_rowidx[l_k] = l_j;
        l_b_sp_csc[l_k] = LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, C);
        if (use_bf16 > 0) {
          libxsmm_rne_convert_fp32_bf16( &l_b_sp_csc[l_k], &l_b_sp_csc_bf16[l_k], 1);
          libxsmm_convert_bf16_f32( &l_b_sp_csc_bf16[l_k], &l_b_sp_csc[l_k], 1 );
          libxsmm_convert_bf16_f32( &l_b_sp_csc_bf16[l_k], &LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, C), 1 );
        }
        l_k++;
      }
    }
  }

  /* create B, csr */
  l_rowptr   = (unsigned int*) libxsmm_aligned_malloc( (C+1)*sizeof(unsigned int), 64 );
  l_colidx   = (unsigned int*) libxsmm_aligned_malloc( nnz*sizeof(unsigned int),   64 );
  l_b_sp_csr = (float*       ) libxsmm_aligned_malloc( nnz*sizeof(float),          64 );
  l_k = 0;
  l_rowptr[C] = nnz;
  for ( l_j = 0; l_j < C; l_j++ ) {
    l_rowptr[l_j] = l_k;
    for ( l_i = 0; l_i < K; l_i++ ) {
      if ( LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, C) != 0.0 ) {
        l_colidx[l_k] = l_i;
        l_b_sp_csr[l_k] = LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, C);
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

  /* sparse routine */
  mykernel_csc = libxsmm_create_packed_spgemm_csc_v2(gemm_shape, l_flags, l_prefetch_flags, nb,
    l_colptr, l_rowidx, (const void*)l_b_sp_csc);
  if (mykernel_csc == NULL) {
    printf("Could not generate CSC kernel!!!\n");
    return 0;
  }
  if (use_bf16 == 0) {
    mykernel_csr = libxsmm_create_packed_spgemm_csr_v2(gemm_shape, l_flags, l_prefetch_flags, nb,
      l_rowptr, l_colidx, (const void*)l_b_sp_csr);
  }

  if (use_bf16 == 0) {
    gemm_param.a.primary = l_a;
    gemm_param.b.primary = l_b_sp_csc;
    gemm_param.c.primary = l_c_asm_csc;
  } else {
    gemm_param.a.primary = l_a_bf16;
    gemm_param.b.primary = l_b_sp_csc_bf16;
    gemm_param.c.primary = l_c_asm_csc_bf16;
  }
  l_start = libxsmm_timer_tick();
  for ( l_n = 0; l_n < (libxsmm_blasint)REPS; l_n++) {
    mykernel_csc( &gemm_param );
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("%fs for sparse (asm, csc)\n", l_total);
  printf("%f GFLOPS for sparse (asm, csc)\n", ((double)((double)REPS * (double)N * (double)C * (double)K) * 2.0) / (l_total * 1.0e9));

  if (use_bf16 == 0) {
    gemm_param.a.primary = l_a;
    gemm_param.b.primary = l_b_sp_csr;
    gemm_param.c.primary = l_c_asm_csr;
    l_start = libxsmm_timer_tick();
    for ( l_n = 0; l_n < (libxsmm_blasint)REPS; l_n++) {
      mykernel_csr( &gemm_param );
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    printf("%fs for sparse (asm, csr)\n", l_total);
    printf("%f GFLOPS for sparse (asm, csr)\n", ((double)((double)REPS * (double)N * (double)C * (double)K) * 2.0) / (l_total * 1.0e9));
  }

  /* check for errors */
  if (use_bf16 > 0) {
    libxsmm_convert_bf16_f32( &LIBXSMM_VLA_ACCESS(3, l_p_c_asm_csc_bf16, 0, 0, 0, K, nb), &LIBXSMM_VLA_ACCESS(3, l_p_c_asm_csc, 0, 0, 0, K, nb), NB * K * nb );
  }

  l_max_error = 0.f;
  for ( l_i = 0; l_i < NB; l_i++) {
    for ( l_j = 0; l_j < K; l_j++) {
      for ( l_k = 0; l_k < nb; l_k++ ) {
        if (LIBXSMM_FABSF( LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_i, l_j, l_k, K, nb)
                    - LIBXSMM_VLA_ACCESS(3, l_p_c_asm_csc, l_i, l_j, l_k, K, nb) ) > l_max_error ) {
          l_max_error = LIBXSMM_FABSF( LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_i, l_j, l_k, K, nb)
                                       -LIBXSMM_VLA_ACCESS(3, l_p_c_asm_csc, l_i, l_j, l_k, K, nb) );
        }
      }
    }
  }
  printf("max error (csc): %f\n", l_max_error);

  /* compare */
  libxsmm_matdiff(&norms_csc, LIBXSMM_DATATYPE_F32, NB * K * nb, 1, l_c_gold, l_c_asm_csc, 0, 0);
  printf("L1 reference  : %.25g\n", norms_csc.l1_ref);
  printf("L1 test       : %.25g\n", norms_csc.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_csc.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_csc.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_csc.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_csc.linf_rel);
  printf("Check-norm    : %.24f\n", libxsmm_matdiff_epsilon(&norms_csc));
  libxsmm_matdiff_reduce(&diff, &norms_csc);

  if (use_bf16 == 0) {
    l_max_error = 0.f;
    for ( l_i = 0; l_i < NB; l_i++) {
      for ( l_j = 0; l_j < K; l_j++) {
        for ( l_k = 0; l_k < nb; l_k++ ) {
          if (LIBXSMM_FABSF( LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_i, l_j, l_k, K, nb)
                      - LIBXSMM_VLA_ACCESS(3, l_p_c_asm_csr, l_i, l_j, l_k, K, nb) ) > l_max_error ) {
            l_max_error = LIBXSMM_FABSF( LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_i, l_j, l_k, K, nb)
                                         -LIBXSMM_VLA_ACCESS(3, l_p_c_asm_csr, l_i, l_j, l_k, K, nb) );
          }
        }
      }
    }
    printf("max error (csr): %f\n", l_max_error);
  }

  /* free */
  libxsmm_free( l_b_de );
  libxsmm_free( l_a );
  libxsmm_free( l_a_bf16 );
  libxsmm_free( l_c_gold );
  libxsmm_free( l_c_asm_csc );
  libxsmm_free( l_c_asm_csc_bf16 );
  libxsmm_free( l_c_asm_csr );

  libxsmm_free( l_b_sp_csc );
  libxsmm_free( l_b_sp_csc_bf16 );
  libxsmm_free( l_colptr );
  libxsmm_free( l_rowidx );

  libxsmm_free( l_b_sp_csr );
  libxsmm_free( l_rowptr );
  libxsmm_free( l_colidx );

  return 0;
}

