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
#include "equation_common.h"

LIBXSMM_INLINE
void eqn_kv_cache_one_f32_gold( const libxsmm_blasint M,
                            const libxsmm_blasint cols,
                            const libxsmm_blasint idxblk,
                            const float* i_kvcache,
                            const float* i_vec_in,
                            float* i_tmp_mat,
                            float* o_vec_out,
                            const long long* i_idx ) {
  libxsmm_blasint i, j;
  /* look up from kv-cache */
  for ( i = 0; i < idxblk; ++i ) {
    for ( j = 0; j < M; ++j ) {
      i_tmp_mat[(i*M)+j] = i_kvcache[(i_idx[i]*M)+j];
    }
  }
  /* matrix multiplication between transpose i_vec_in, i_tmp_mat, i_vec_out */
  for ( i = 0; i < idxblk; ++i ) {
    float tmp = 0.0f;
    for ( j = 0; j < M; ++j ) {
      tmp += i_vec_in[j] * i_tmp_mat[(i*M)+j];
    }
    o_vec_out[i] = tmp;
  }
}

LIBXSMM_INLINE
void eqn_kv_cache_one_f32_tpp( const libxsmm_blasint M,
                           const libxsmm_blasint cols,
                           const libxsmm_blasint idxblk,
                           const float* i_kvcache,
                           const float* i_vec_in,
                           float* i_tmp_mat,
                           float* o_vec_out,
                           const long long* i_idx ) {
  eqn_kv_cache_one_f32_gold( M, cols, idxblk, i_kvcache, i_vec_in, i_tmp_mat, o_vec_out, i_idx );
}

LIBXSMM_INLINE
void eqn_kv_cache_one_f32(libxsmm_blasint cols, libxsmm_blasint M, libxsmm_blasint idxblk, libxsmm_blasint numidx) {
  libxsmm_blasint i, j;
  libxsmm_matdiff_info norms_out;
  float* l_kvcache;
  float* l_vec_in;
  float* l_tmp_mat;
  float* l_vec_out_gold;
  float* l_vec_out_tpp;
  long long* l_idx;
  double check_norm = 0.0;

  l_kvcache      = (float*)     libxsmm_aligned_malloc( sizeof(float)*cols*M,     64);
  l_vec_in       = (float*)     libxsmm_aligned_malloc( sizeof(float)*M,          64);
  l_tmp_mat      = (float*)     libxsmm_aligned_malloc( sizeof(float)*idxblk*M,   64);
  l_vec_out_gold = (float*)     libxsmm_aligned_malloc( sizeof(float)*numidx,     64);
  l_vec_out_tpp  = (float*)     libxsmm_aligned_malloc( sizeof(float)*numidx,     64);
  l_idx          = (long long*) libxsmm_aligned_malloc( sizeof(long long)*numidx, 64);

  /* init kv cache */
  for ( i = 0; i < cols; ++i ) {
    for ( j = 0; j < M; ++j ) {
      l_kvcache[(i*M)+j] = (float)libxsmm_rng_f64();
    }
  }

  /* init transposed vector */
  for ( j = 0; j < M; ++j ) {
    l_vec_in[j] = (float)libxsmm_rng_f64();
  }

  /* init temp gather matix */
  for ( i = 0; i < idxblk; ++i ) {
    for ( j = 0; j < M; ++j ) {
      l_tmp_mat[(i*M)+j] = (float)libxsmm_rng_f64();
    }
  }

  /* init output and lookup idx */
  for ( j = 0; j < numidx; ++j ) {
    l_vec_out_gold[j] = (float)libxsmm_rng_f64();
    l_vec_out_tpp[j]  = (float)libxsmm_rng_f64();
    l_idx[j]          = libxsmm_rng_u32(cols-1);
  }

  /* run gold */
  for ( i = 0; i < numidx; i += idxblk ) {
    eqn_kv_cache_one_f32_gold( M, cols, idxblk, l_kvcache, l_vec_in, l_tmp_mat, l_vec_out_gold+i, l_idx+i );
  }
  /* run tpp */
  for ( i = 0; i < numidx; i += idxblk ) {
    eqn_kv_cache_one_f32_tpp( M, cols, idxblk, l_kvcache, l_vec_in, l_tmp_mat, l_vec_out_tpp+i, l_idx+i );
  }

  printf("##########################################\n");
  printf("#  Correctness Equ. KVCache 1 - Output   #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, numidx, 1, l_vec_out_gold, l_vec_out_tpp, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  check_norm = libxsmm_matdiff_epsilon(&norms_out);
  printf("Check-norm    : %.24f\n\n", check_norm);

  libxsmm_free( l_kvcache );
  libxsmm_free( l_vec_in );
  libxsmm_free( l_tmp_mat );
  libxsmm_free( l_vec_out_gold );
  libxsmm_free( l_vec_out_tpp );
  libxsmm_free( l_idx );
}

int main( int argc, char* argv[] ) {
  int ret = EXIT_SUCCESS;

  libxsmm_blasint cols = 2048;
  libxsmm_blasint M = 256;
  libxsmm_blasint numidx = 512;
  libxsmm_blasint idxblk = 32;
  libxsmm_datatype in_dt = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype out_dt = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype compute_dt = LIBXSMM_DATATYPE_F32;

  libxsmm_init();
  if ( argc > 1 ) cols = atoi(argv[1]);
  if ( argc > 2 ) M = atoi(argv[2]);
  if ( argc > 3 ) numidx = atoi(argv[3]);
  if ( argc > 4 ) idxblk = atoi(argv[4]);

  if ( (in_dt == LIBXSMM_DATATYPE_F32) && (out_dt == LIBXSMM_DATATYPE_F32) && (compute_dt == LIBXSMM_DATATYPE_F32) ) {
    eqn_kv_cache_one_f32(cols, M, idxblk, numidx);
  }

  return ret;
}
