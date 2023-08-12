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
#if defined(__AVX512F__)
#include <libxsmm_intrinsics_x86.h>
#endif

LIBXSMM_INLINE
void eqn_gather_bcstmul_add_f32_gold( const libxsmm_blasint M,
                                      const libxsmm_blasint cols,
                                      const libxsmm_blasint idxblk,
                                      const float* i_gather_dot,
                                      const float* i_vec_in,
                                      float* o_vec_out,
                                      const long long* i_idx ) {
  libxsmm_blasint i, j;
#if __AVX512F__
  if ( M % 16 != 0 ) {
    printf("Using intrinsic version, M mod 16 is required\n");
    exit(-1);
  }

  /* look up from kv-cache */
#if 1
  for ( i = 0; i < idxblk; i += 1 ) {
    __m512 bcst0 = _mm512_set1_ps(i_vec_in[i]);
    for ( j = 0; j < M; j += 16 ) {
      __m512 rego = _mm512_load_ps( &(o_vec_out[j]) );
      __m512 b0   = _mm512_load_ps( &(i_gather_dot[(i_idx[i+0]*M)+j]) );
      rego = _mm512_fmadd_ps( bcst0, b0, rego );
      _mm512_store_ps( &(o_vec_out[j]), rego );
    }
  }
#else
  for ( i = 0; i < idxblk; i += 4 ) {
    __m512 bcst0 = _mm512_set1_ps(i_vec_in[i+0]);
    __m512 bcst1 = _mm512_set1_ps(i_vec_in[i+1]);
    __m512 bcst2 = _mm512_set1_ps(i_vec_in[i+2]);
    __m512 bcst3 = _mm512_set1_ps(i_vec_in[i+3]);
    for ( j = 0; j < M; j += 16 ) {
      __m512 rego = _mm512_load_ps( &(o_vec_out[j]) );
      __m512 b0 = _mm512_load_ps( &(i_gather_dot[(i_idx[i+0]*M)+j]) );
      __m512 b1 = _mm512_load_ps( &(i_gather_dot[(i_idx[i+1]*M)+j]) );
      __m512 b2 = _mm512_load_ps( &(i_gather_dot[(i_idx[i+2]*M)+j]) );
      __m512 b3 = _mm512_load_ps( &(i_gather_dot[(i_idx[i+3]*M)+j]) );
      rego = _mm512_fmadd_ps( bcst0, b0, rego );
      rego = _mm512_fmadd_ps( bcst1, b1, rego );
      rego = _mm512_fmadd_ps( bcst2, b2, rego );
      rego = _mm512_fmadd_ps( bcst3, b3, rego );
      _mm512_store_ps( &(o_vec_out[j]), rego );
    }
  }
#endif
#else
  /* look up from kv-cache, broadcast scalar multiply and reduce over look ups */
  for ( i = 0; i < idxblk; ++i ) {
    for ( j = 0; j < M; ++j ) {
      o_vec_out[j] += i_vec_in[i] * i_gather_dot[(i_idx[i]*M)+j];
    }
  }
#endif
}

LIBXSMM_INLINE
void eqn_gather_bcstmul_add_f32_tpp1( const libxsmm_blasint M,
                                      const libxsmm_blasint cols,
                                      const libxsmm_blasint idxblk,
                                      const float* i_gather_dot,
                                      const float* i_vec_in,
                                      float* o_vec_out,
                                      const long long* i_idx,
                                      libxsmm_meltwfunction_binary i_muladd ) {
  libxsmm_meltw_binary_param l_muladd_param;
  libxsmm_blasint i;

  /* look up from kv-cache */
  for ( i = 0; i < idxblk; ++i ) {
    l_muladd_param.in0.primary = (void*)&(i_gather_dot[(i_idx[i]*M)]);
    l_muladd_param.in1.primary = (void*)&(i_vec_in[i]);
    l_muladd_param.out.primary = (void*)o_vec_out;
    i_muladd( &l_muladd_param );
  }
}

LIBXSMM_INLINE
void eqn_gather_bcstmul_add_f32_tpp2( const libxsmm_blasint M,
                                      const libxsmm_blasint cols,
                                      const libxsmm_blasint idxblk,
                                      float* i_tmp_mat,
                                      float* i_tmp_vec,
                                      const float* i_gather_dot,
                                      const float* i_vec_in,
                                      float* o_vec_out,
                                      const long long* i_idx,
                                      libxsmm_meltwfunction_unary  i_gather_func,
                                      libxsmm_meltwfunction_binary i_mul,
                                      libxsmm_meltwfunction_unary  i_addreduce,
                                      libxsmm_meltwfunction_binary i_add ) {
  libxsmm_meltw_unary_param l_gather_param;
  libxsmm_meltw_binary_param l_mul_param;
  libxsmm_meltw_unary_param l_addreduce_param;
  libxsmm_meltw_binary_param l_add_param;

  l_gather_param.in.primary = (void*)i_gather_dot;
  l_gather_param.in.secondary = (void*)i_idx;
  l_gather_param.out.primary = (void*)i_tmp_mat;

  l_mul_param.in0.primary = (void*)i_tmp_mat;
  l_mul_param.in1.primary = (void*)i_vec_in;
  l_mul_param.out.primary = (void*)i_tmp_mat;

  l_addreduce_param.in.primary = (void*)i_tmp_mat;
  l_addreduce_param.out.primary = (void*)i_tmp_vec;

  l_add_param.in0.primary = (void*)i_tmp_vec;
  l_add_param.in1.primary = (void*)o_vec_out;
  l_add_param.out.primary = (void*)o_vec_out;

  i_gather_func( &l_gather_param );
  i_mul( &l_mul_param );
  i_addreduce( &l_addreduce_param );
  i_add( &l_add_param );
}

LIBXSMM_INLINE
int eqn_gather_bcstmul_add_f32(const libxsmm_blasint cols, const libxsmm_blasint M, const libxsmm_blasint idxblk, const libxsmm_blasint numidx, const libxsmm_blasint iters) {
  int ret = EXIT_SUCCESS;
  libxsmm_blasint i, j;
  libxsmm_matdiff_info norms_out;
  float* l_gather_dot;
  float* l_vec_in;
  float* l_tmp_mat;
  float* l_tmp_vec;
  float* l_vec_out_gold;
  float* l_vec_out_tpp1;
  float* l_vec_out_tpp2;
  long long* l_idx;
  double check_norm = 0.0;
  double pass_norm = 0.00001;
  double l_bytes = ((M * numidx * 4) + (numidx * 4) + (M * 4));
  libxsmm_timer_tickint l_start;
  double l_runtime;

  /* binary muladd TPP */
  libxsmm_meltwfunction_binary l_muladd = NULL;
  libxsmm_meltw_binary_shape   l_muladd_shape = libxsmm_create_meltw_binary_shape( M, 1, M, 1, M, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );

  /* gather + binary mul + addreduce + add TPP */
  libxsmm_meltwfunction_unary  l_gather = NULL;
  libxsmm_meltw_unary_shape    l_gather_shape = libxsmm_create_meltw_unary_shape( M, idxblk, M, M, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
  libxsmm_meltwfunction_binary l_mul = NULL;
  libxsmm_meltw_binary_shape   l_mul_shape = libxsmm_create_meltw_binary_shape( M, idxblk, M, 1, M, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
  libxsmm_meltwfunction_unary  l_addreduce = NULL;
  libxsmm_meltw_unary_shape    l_addreduce_shape = libxsmm_create_meltw_unary_shape( M, idxblk, M, M, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
  libxsmm_meltwfunction_binary l_add = NULL;
  libxsmm_meltw_binary_shape   l_add_shape = libxsmm_create_meltw_binary_shape( M, 1, M, M, M, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );

  l_gather_dot   = (float*)     libxsmm_aligned_malloc( sizeof(float)*cols*M,     64);
  l_vec_in       = (float*)     libxsmm_aligned_malloc( sizeof(float)*cols,       64);
  l_tmp_mat      = (float*)     libxsmm_aligned_malloc( sizeof(float)*idxblk*M,   64);
  l_vec_out_gold = (float*)     libxsmm_aligned_malloc( sizeof(float)*M,          64);
  l_tmp_vec      = (float*)     libxsmm_aligned_malloc( sizeof(float)*M,          64);
  l_vec_out_tpp1 = (float*)     libxsmm_aligned_malloc( sizeof(float)*M,          64);
  l_vec_out_tpp2 = (float*)     libxsmm_aligned_malloc( sizeof(float)*M,          64);
  l_idx          = (long long*) libxsmm_aligned_malloc( sizeof(long long)*numidx, 64);

  /* init kv cache */
  for ( i = 0; i < cols; ++i ) {
    for ( j = 0; j < M; ++j ) {
      l_gather_dot[(i*M)+j] = (float)libxsmm_rng_f64();
    }
  }

  /* init transposed vector */
  for ( j = 0; j < cols; ++j ) {
    l_vec_in[j] = (float)libxsmm_rng_f64();
  }

  /* init temp gather matix */
  for ( i = 0; i < idxblk; ++i ) {
    for ( j = 0; j < M; ++j ) {
      l_tmp_mat[(i*M)+j] = (float)libxsmm_rng_f64();
    }
  }

  /* init output */
  for ( j = 0; j < M; ++j ) {
    l_tmp_vec[j]      = 0.0f;
    l_vec_out_gold[j] = 0.0f;
    l_vec_out_tpp1[j] = 0.0f;
    l_vec_out_tpp2[j] = 0.0f;
  }

  /* init lookup idx */
  for ( j = 0; j < numidx; ++j ) {
    l_idx[j] = libxsmm_rng_u32(cols-1);
  }

  /* first TPP implementation we just run a binary muladd in a loop */
  l_muladd = libxsmm_dispatch_meltw_binary_v2( LIBXSMM_MELTW_TYPE_BINARY_MULADD, l_muladd_shape, LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1 );

  if ( l_muladd == NULL ) {
    printf("JIT failed for muladd, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(-1);
  }

  /* second TPP implementation we run gather + binary mul + addreduce + add */
  l_gather = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_GATHER, l_gather_shape, LIBXSMM_MELTW_FLAG_UNARY_GS_COLS | LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES );
  l_mul = libxsmm_dispatch_meltw_binary_v2( LIBXSMM_MELTW_TYPE_BINARY_MUL, l_mul_shape, LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1 );
  l_addreduce = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_addreduce_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS );
  l_add = libxsmm_dispatch_meltw_binary_v2( LIBXSMM_MELTW_TYPE_BINARY_ADD, l_add_shape, LIBXSMM_MELTW_FLAG_BINARY_NONE );
  if ( (l_gather == NULL) || (l_mul == NULL) || (l_addreduce == NULL) || (l_add == NULL) ) {
    printf("JIT failed for gather+muladd+addreduce+add, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(-1);
  }

  /* run gold */
  for ( i = 0; i < numidx; i += idxblk ) {
    eqn_gather_bcstmul_add_f32_gold( M, cols, idxblk, l_gather_dot, l_vec_in+i, l_vec_out_gold, l_idx+i );
  }
  /* run tpp */
  for ( i = 0; i < numidx; i += idxblk ) {
    eqn_gather_bcstmul_add_f32_tpp1( M, cols, idxblk, l_gather_dot, l_vec_in+i, l_vec_out_tpp1, l_idx+i, l_muladd );
  }
  for ( i = 0; i < numidx; i += idxblk ) {
    eqn_gather_bcstmul_add_f32_tpp2( M, cols, idxblk, l_tmp_mat, l_tmp_vec, l_gather_dot, l_vec_in+i, l_vec_out_tpp2, l_idx+i, l_gather, l_mul, l_addreduce, l_add );
  }

  printf("##########################################\n");
  printf("#   Equ. gather+bcstmul+add - Output     #\n");
  printf("##########################################\n");
  printf("muladd\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, M, 1, l_vec_out_gold, l_vec_out_tpp1, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  check_norm = libxsmm_matdiff_epsilon(&norms_out);
  printf("Check-norm    : %.24f\n\n", check_norm);
  if ( check_norm > pass_norm ) {
    ret = EXIT_FAILURE;
  }
  printf("gather + binary mul + addreduce + add\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, M, 1, l_vec_out_gold, l_vec_out_tpp2, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  check_norm = libxsmm_matdiff_epsilon(&norms_out);
  printf("Check-norm    : %.24f\n\n", check_norm);
  if ( check_norm > pass_norm ) {
    ret = EXIT_FAILURE;
  }

  l_start = libxsmm_timer_tick();
  for ( j = 0; j < iters; ++j ) {
    for ( i = 0; i < numidx; i += idxblk ) {
      eqn_gather_bcstmul_add_f32_gold( M, cols, idxblk, l_gather_dot, l_vec_in+i, l_vec_out_gold, l_idx+i );
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  printf("Compiler Optimized\nRuntime: %f; GiB/s: %f\n", l_runtime, (l_bytes/(1024.0*1024.0*1024.0))/(l_runtime/(double)iters));

  l_start = libxsmm_timer_tick();
  for ( j = 0; j < iters; ++j ) {
    for ( i = 0; i < numidx; i += idxblk ) {
      eqn_gather_bcstmul_add_f32_tpp1( M, cols, idxblk, l_gather_dot, l_vec_in+i, l_vec_out_tpp1, l_idx+i, l_muladd );
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  printf("TPP muladd\nRuntime: %f; GiB/s: %f\n", l_runtime, (l_bytes/(1024.0*1024.0*1024.0))/(l_runtime/(double)iters));

  l_start = libxsmm_timer_tick();
  for ( j = 0; j < iters; ++j ) {
    for ( i = 0; i < numidx; i += idxblk ) {
      eqn_gather_bcstmul_add_f32_tpp2( M, cols, idxblk, l_tmp_mat, l_tmp_vec, l_gather_dot, l_vec_in+i, l_vec_out_tpp2, l_idx+i, l_gather, l_mul, l_addreduce, l_add );
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  printf("TPP gather + binary mul + addreduce + add\nRuntime: %f; GiB/s: %f\n", l_runtime, (l_bytes/(1024.0*1024.0*1024.0))/(l_runtime/(double)iters));

  libxsmm_free( l_gather_dot );
  libxsmm_free( l_vec_in );
  libxsmm_free( l_tmp_mat );
  libxsmm_free( l_tmp_vec );
  libxsmm_free( l_vec_out_gold );
  libxsmm_free( l_vec_out_tpp1 );
  libxsmm_free( l_vec_out_tpp2 );
  libxsmm_free( l_idx );

  return ret;
}

int main( int argc, char* argv[] ) {
  int ret = EXIT_SUCCESS;

  libxsmm_blasint cols = 2048;
  libxsmm_blasint M = 256;
  libxsmm_blasint numidx = 512;
  libxsmm_blasint idxblk = 4;
  libxsmm_blasint iters = 1000000;
  libxsmm_datatype in_dt = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype out_dt = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype compute_dt = LIBXSMM_DATATYPE_F32;

  libxsmm_init();
  if ( argc > 1 ) cols = atoi(argv[1]);
  if ( argc > 2 ) M = atoi(argv[2]);
  if ( argc > 3 ) numidx = atoi(argv[3]);
  if ( argc > 4 ) idxblk = atoi(argv[4]);
  if ( argc > 5 ) iters = atoi(argv[5]);

  if ( numidx % idxblk !=0 ) {
    printf("idxblk ust devide numidx!\n");
    ret = EXIT_FAILURE;
  }

  if ( (in_dt == LIBXSMM_DATATYPE_F32) && (out_dt == LIBXSMM_DATATYPE_F32) && (compute_dt == LIBXSMM_DATATYPE_F32) ) {
    ret = eqn_gather_bcstmul_add_f32(cols, M, idxblk, numidx, iters);
  }

  return ret;
}
