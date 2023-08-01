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
#if defined(__x86_64__)
#include <x86intrin.h>
#endif

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
#ifdef __AVX512F__
  if ( M % 16 != 0 ) {
    printf("Using intrinsic version, M mod 16 is required\n");
    exit(-1);
  }

  /* look up from kv-cache */
#if 0
  for ( i = 0; i < idxblk; i += 1 ) {
    __m512 reg0 = _mm512_setzero_ps();
    for ( j = 0; j < M; j += 16 ) {
      __m512 a = _mm512_load_ps( &(i_vec_in[j]) );
      __m512 b0 = _mm512_load_ps( &(i_kvcache[(i_idx[i+0]*M)+j]) );

      reg0 = _mm512_fmadd_ps( a, b0, reg0 );
    }
    o_vec_out[i+0] = _mm512_reduce_add_ps( reg0 );
  }
#else
  for ( i = 0; i < idxblk; i += 4 ) {
    __m512 reg0 = _mm512_setzero_ps();
    __m512 reg1 = _mm512_setzero_ps();
    __m512 reg2 = _mm512_setzero_ps();
    __m512 reg3 = _mm512_setzero_ps();
    for ( j = 0; j < M; j += 16 ) {
      __m512 a = _mm512_load_ps( &(i_vec_in[j]) );
      __m512 b0 = _mm512_load_ps( &(i_kvcache[(i_idx[i+0]*M)+j]) );
      __m512 b1 = _mm512_load_ps( &(i_kvcache[(i_idx[i+1]*M)+j]) );
      __m512 b2 = _mm512_load_ps( &(i_kvcache[(i_idx[i+2]*M)+j]) );
      __m512 b3 = _mm512_load_ps( &(i_kvcache[(i_idx[i+3]*M)+j]) );
      reg0 = _mm512_fmadd_ps( a, b0, reg0 );
      reg1 = _mm512_fmadd_ps( a, b1, reg1 );
      reg2 = _mm512_fmadd_ps( a, b2, reg2 );
      reg3 = _mm512_fmadd_ps( a, b3, reg3 );
    }
    o_vec_out[i+0] = _mm512_reduce_add_ps( reg0 );
    o_vec_out[i+1] = _mm512_reduce_add_ps( reg1 );
    o_vec_out[i+2] = _mm512_reduce_add_ps( reg2 );
    o_vec_out[i+3] = _mm512_reduce_add_ps( reg3 );
  }
#endif
#else
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
#endif
}

LIBXSMM_INLINE
void eqn_kv_cache_one_f32_tpp1( const libxsmm_blasint M,
                           const libxsmm_blasint cols,
                           const libxsmm_blasint idxblk,
                           const float* i_kvcache,
                           const float* i_vec_in,
                           float* i_tmp_mat,
                           float* o_vec_out,
                           const long long* i_idx,
                           libxsmm_meltwfunction_binary i_mul,
                           libxsmm_meltwfunction_unary  i_addreduce ) {
  libxsmm_meltw_binary_param l_mul_param;
  libxsmm_meltw_unary_param l_addreduce_param;
  libxsmm_blasint i;

  /* look up from kv-cache */
  l_mul_param.in0.primary = (void*)i_vec_in;
  for ( i = 0; i < idxblk; ++i ) {
    l_mul_param.in1.primary = (void*)&(i_kvcache[(i_idx[i]*M)]);
    l_mul_param.out.primary = (void*)i_tmp_mat;
    i_mul( &l_mul_param );

    l_addreduce_param.in.primary = (void*)i_tmp_mat;
    l_addreduce_param.out.primary = (void*)&(o_vec_out[i]);
    i_addreduce( &l_addreduce_param );
  }
}

LIBXSMM_INLINE
void eqn_kv_cache_one_f32_tpp2( const libxsmm_blasint M,
                           const libxsmm_blasint cols,
                           const libxsmm_blasint idxblk,
                           const float* i_kvcache,
                           const float* i_vec_in,
                           float* i_tmp_mat,
                           float* o_vec_out,
                           const long long* i_idx,
                           libxsmm_matrix_eqn_function i_eqn ) {
  libxsmm_matrix_eqn_param l_eqn_param;
  libxsmm_matrix_arg l_arg_array[2];
  libxsmm_blasint i;

  /* look up from kv-cache */
  l_arg_array[0].primary = (void*)i_vec_in;
  for ( i = 0; i < idxblk; ++i ) {
    l_arg_array[1].primary = (void*)&(i_kvcache[(i_idx[i]*M)]);
    l_eqn_param.inputs = l_arg_array;
    l_eqn_param.output.primary = (void*)&(o_vec_out[i]);

    i_eqn( &l_eqn_param );
  }
}

LIBXSMM_INLINE
void eqn_kv_cache_one_f32_tpp3( const libxsmm_blasint M,
                           const libxsmm_blasint cols,
                           const libxsmm_blasint idxblk,
                           const float* i_kvcache,
                           const float* i_vec_in,
                           float* i_tmp_mat,
                           float* o_vec_out,
                           const long long* i_idx,
                           libxsmm_meltwfunction_unary i_gather_func,
                           libxsmm_xmmfunction         i_gemm_func ) {
  libxsmm_meltw_unary_param l_gather_param;
  libxsmm_gemm_param l_gemm_param;

  l_gather_param.in.primary = (void*)i_kvcache;
  l_gather_param.in.secondary = (void*)i_idx;
  l_gather_param.out.primary = (void*)i_tmp_mat;

  l_gemm_param.a.primary = (void*)i_vec_in;
  l_gemm_param.b.primary = (void*)i_tmp_mat;
  l_gemm_param.c.primary = (void*)o_vec_out;

  i_gather_func( &l_gather_param );
  i_gemm_func.gemm( &l_gemm_param );
}

LIBXSMM_INLINE
void eqn_kv_cache_one_f32_tpp4( const libxsmm_blasint M,
                           const libxsmm_blasint cols,
                           const libxsmm_blasint idxblk,
                           const float* i_kvcache,
                           const float* i_vec_in,
                           float* i_tmp_mat,
                           float* o_vec_out,
                           const long long* i_idx,
                           libxsmm_meltwfunction_unary  i_gather_func,
                           libxsmm_meltwfunction_binary i_mul,
                           libxsmm_meltwfunction_unary  i_addreduce ) {
  libxsmm_meltw_unary_param l_gather_param;
  libxsmm_meltw_binary_param l_mul_param;
  libxsmm_meltw_unary_param l_addreduce_param;

  l_gather_param.in.primary = (void*)i_kvcache;
  l_gather_param.in.secondary = (void*)i_idx;
  l_gather_param.out.primary = (void*)i_tmp_mat;

  l_mul_param.in0.primary = (void*)i_tmp_mat;
  l_mul_param.in1.primary = (void*)i_vec_in;
  l_mul_param.out.primary = (void*)i_tmp_mat;

  l_addreduce_param.in.primary = (void*)i_tmp_mat;
  l_addreduce_param.out.primary = (void*)o_vec_out;

  i_gather_func( &l_gather_param );
  i_mul( &l_mul_param );
  i_addreduce( &l_addreduce_param );
}

LIBXSMM_INLINE
int eqn_kv_cache_one_f32(const libxsmm_blasint cols, const libxsmm_blasint M, const libxsmm_blasint idxblk, const libxsmm_blasint numidx, const libxsmm_blasint iters) {
  int ret = EXIT_SUCCESS;
  libxsmm_blasint i, j;
  libxsmm_matdiff_info norms_out;
  float* l_kvcache;
  float* l_vec_in;
  float* l_tmp_mat;
  float* l_vec_out_gold;
  float* l_vec_out_tpp1;
  float* l_vec_out_tpp2;
  float* l_vec_out_tpp3;
  float* l_vec_out_tpp4;
  long long* l_idx;
  double check_norm = 0.0;
  double pass_norm = 0.00001;
  double l_bytes = ((M * numidx * 4) + (numidx * (8 + 4)) + (M * 4));
  libxsmm_timer_tickint l_start;
  double l_runtime;
  libxsmm_blasint idxblk_gemm = idxblk/4;
  libxsmm_blasint idxblk_mulred = idxblk;
  /* binary mul + reduce add TPP */
  libxsmm_meltwfunction_binary l_mul = NULL;
  libxsmm_meltw_binary_shape   l_mul_shape = libxsmm_create_meltw_binary_shape( M, 1, M, M, M, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
  libxsmm_meltwfunction_unary  l_addreduce = NULL;
  libxsmm_meltw_unary_shape    l_addreduce_shape = libxsmm_create_meltw_unary_shape( M, 1, M, 1, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
  /* equation for mul + reduce add TPP */
  libxsmm_blasint l_eqn_0_idx = 0;
  libxsmm_meqn_arg_shape l_eqn_0_shape_out;
  libxsmm_matrix_eqn_function l_eqn_0 = NULL;
  /* gather + GEMM TPP */
  libxsmm_meltwfunction_unary l_gather = NULL;
  libxsmm_meltw_unary_shape   l_gahter_shape = libxsmm_create_meltw_unary_shape( M, idxblk_gemm, M, M, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
  libxsmm_xmmfunction l_gather_gemm;
  libxsmm_gemm_shape  l_gather_gemm_shape = libxsmm_create_gemm_shape( 1, idxblk_gemm, M, 1, M, 1, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
  libxsmm_bitfield l_gather_gemm_flags = LIBXSMM_GEMM_FLAGS('N', 'N') | LIBXSMM_GEMM_FLAG_BETA_0;
  libxsmm_bitfield l_gather_gemm_prefetch_flags = 0;
  /* gather + mul + reduce */
  libxsmm_meltwfunction_unary  l_gather_2 = NULL;
  libxsmm_meltw_unary_shape    l_gahter_2_shape = libxsmm_create_meltw_unary_shape( M, idxblk_mulred, M, M, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
  libxsmm_meltwfunction_binary l_mul_2 = NULL;
  libxsmm_meltw_binary_shape   l_mul_2_shape = libxsmm_create_meltw_binary_shape( M, idxblk_mulred, M, M, M, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
  libxsmm_meltwfunction_unary  l_addreduce_2 = NULL;
  libxsmm_meltw_unary_shape    l_addreduce_2_shape = libxsmm_create_meltw_unary_shape( M, idxblk_mulred, M, 1, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );

  l_kvcache      = (float*)     libxsmm_aligned_malloc( sizeof(float)*cols*M,     64);
  l_vec_in       = (float*)     libxsmm_aligned_malloc( sizeof(float)*M,          64);
  l_tmp_mat      = (float*)     libxsmm_aligned_malloc( sizeof(float)*idxblk*M,   64);
  l_vec_out_gold = (float*)     libxsmm_aligned_malloc( sizeof(float)*numidx,     64);
  l_vec_out_tpp1 = (float*)     libxsmm_aligned_malloc( sizeof(float)*numidx,     64);
  l_vec_out_tpp2 = (float*)     libxsmm_aligned_malloc( sizeof(float)*numidx,     64);
  l_vec_out_tpp3 = (float*)     libxsmm_aligned_malloc( sizeof(float)*numidx,     64);
  l_vec_out_tpp4 = (float*)     libxsmm_aligned_malloc( sizeof(float)*numidx,     64);
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
    l_vec_out_tpp1[j] = (float)libxsmm_rng_f64();
    l_vec_out_tpp2[j] = (float)libxsmm_rng_f64();
    l_vec_out_tpp3[j] = (float)libxsmm_rng_f64();
    l_vec_out_tpp4[j] = (float)libxsmm_rng_f64();
    l_idx[j]          = libxsmm_rng_u32(cols-1);
  }

  /* first TPP implementation we just run a reduce muladd */
  l_mul = libxsmm_dispatch_meltw_binary_v2( LIBXSMM_MELTW_TYPE_BINARY_MUL, l_mul_shape, LIBXSMM_MELTW_FLAG_BINARY_NONE );
  l_addreduce = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_addreduce_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS );

  /* second TPP implementation equation for reduce muladd */
  l_eqn_0_idx = libxsmm_matrix_eqn_create();
  libxsmm_matrix_eqn_push_back_unary_op( l_eqn_0_idx, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD,
                                         LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( l_eqn_0_idx, LIBXSMM_MELTW_TYPE_BINARY_MUL,
                                          LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( l_eqn_0_idx, M, 1, M, 0, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( l_eqn_0_idx, M, 1, M, 1, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_tree_print( l_eqn_0_idx );
  libxsmm_matrix_eqn_rpn_print( l_eqn_0_idx );
  l_eqn_0_shape_out = libxsmm_create_meqn_arg_shape( M, 1, M, LIBXSMM_DATATYPE_F32 );
  l_eqn_0 = libxsmm_dispatch_matrix_eqn_v2( l_eqn_0_idx, l_eqn_0_shape_out );

  /* third TPP implementation we run gather and than matmul */
  l_gather = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_GATHER, l_gahter_shape, LIBXSMM_MELTW_FLAG_UNARY_GS_COLS | LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES );
  l_gather_gemm.gemm = libxsmm_dispatch_gemm_v2( l_gather_gemm_shape, l_gather_gemm_flags, l_gather_gemm_prefetch_flags );

  /* forth TPP implementation we run gather and reduce muladd */
  l_gather_2 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_GATHER, l_gahter_2_shape, LIBXSMM_MELTW_FLAG_UNARY_GS_COLS | LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES );
  l_mul_2 = libxsmm_dispatch_meltw_binary_v2( LIBXSMM_MELTW_TYPE_BINARY_MUL, l_mul_2_shape, LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1 );
  l_addreduce_2 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_addreduce_2_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS );

  if ( (l_gather == NULL) || (l_gather_gemm.xmm == NULL) ) {
    printf("JIT failed for gather+gemm, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(-1);
  }

  /* run gold */
  for ( i = 0; i < numidx; i += idxblk ) {
    eqn_kv_cache_one_f32_gold( M, cols, idxblk, l_kvcache, l_vec_in, l_tmp_mat, l_vec_out_gold+i, l_idx+i );
  }
  /* run tpp */
  for ( i = 0; i < numidx; i += idxblk ) {
    eqn_kv_cache_one_f32_tpp1( M, cols, idxblk, l_kvcache, l_vec_in, l_tmp_mat, l_vec_out_tpp1+i, l_idx+i, l_mul, l_addreduce );
  }
  for ( i = 0; i < numidx; i += idxblk ) {
    eqn_kv_cache_one_f32_tpp2( M, cols, idxblk, l_kvcache, l_vec_in, l_tmp_mat, l_vec_out_tpp2+i, l_idx+i, l_eqn_0 );
  }
  for ( i = 0; i < numidx; i += idxblk_gemm ) {
    eqn_kv_cache_one_f32_tpp3( M, cols, idxblk_gemm, l_kvcache, l_vec_in, l_tmp_mat, l_vec_out_tpp3+i, l_idx+i, l_gather, l_gather_gemm );
  }
  for ( i = 0; i < numidx; i += idxblk_mulred ) {
    eqn_kv_cache_one_f32_tpp4( M, cols, idxblk, l_kvcache, l_vec_in, l_tmp_mat, l_vec_out_tpp4+i, l_idx+i, l_gather_2, l_mul_2, l_addreduce_2 );
  }

  printf("##########################################\n");
  printf("#  Correctness Equ. KVCache 1 - Output   #\n");
  printf("##########################################\n");
  printf("mul + reduce\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, numidx, 1, l_vec_out_gold, l_vec_out_tpp1, 0, 0);
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
  printf("equation mul + reduce\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, numidx, 1, l_vec_out_gold, l_vec_out_tpp2, 0, 0);
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
  printf("gather + gemm\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, numidx, 1, l_vec_out_gold, l_vec_out_tpp3, 0, 0);
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
  printf("gather + mul + reduce\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, numidx, 1, l_vec_out_gold, l_vec_out_tpp4, 0, 0);
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
      eqn_kv_cache_one_f32_gold( M, cols, idxblk, l_kvcache, l_vec_in, l_tmp_mat, l_vec_out_gold+i, l_idx+i );
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  printf("Compiler Optimized\nRuntime: %f; GiB/s: %f\n", l_runtime, (l_bytes/(1024.0*1024.0*1024.0))/(l_runtime/(double)iters));

  l_start = libxsmm_timer_tick();
  for ( j = 0; j < iters; ++j ) {
    for ( i = 0; i < numidx; i += idxblk ) {
      eqn_kv_cache_one_f32_tpp1( M, cols, idxblk, l_kvcache, l_vec_in, l_tmp_mat, l_vec_out_tpp1+i, l_idx+i, l_mul, l_addreduce );
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  printf("TPP mul + reduce\nRuntime: %f; GiB/s: %f\n", l_runtime, (l_bytes/(1024.0*1024.0*1024.0))/(l_runtime/(double)iters));

  l_start = libxsmm_timer_tick();
  for ( j = 0; j < iters; ++j ) {
    for ( i = 0; i < numidx; i += idxblk ) {
      eqn_kv_cache_one_f32_tpp2( M, cols, idxblk, l_kvcache, l_vec_in, l_tmp_mat, l_vec_out_tpp2+i, l_idx+i, l_eqn_0 );
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  printf("TPP equation mul + reduce\nRuntime: %f; GiB/s: %f\n", l_runtime, (l_bytes/(1024.0*1024.0*1024.0))/(l_runtime/(double)iters));

  l_start = libxsmm_timer_tick();
  for ( j = 0; j < iters; ++j ) {
    for ( i = 0; i < numidx; i += idxblk_gemm ) {
      eqn_kv_cache_one_f32_tpp3( M, cols, idxblk_gemm, l_kvcache, l_vec_in, l_tmp_mat, l_vec_out_tpp3+i, l_idx+i, l_gather, l_gather_gemm );
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  printf("TPP Gather GEMM\nRuntime: %f; GiB/s: %f\n", l_runtime, (l_bytes/(1024.0*1024.0*1024.0))/(l_runtime/(double)iters));

  l_start = libxsmm_timer_tick();
  for ( j = 0; j < iters; ++j ) {
    for ( i = 0; i < numidx; i += idxblk_mulred ) {
      eqn_kv_cache_one_f32_tpp4( M, cols, idxblk_gemm, l_kvcache, l_vec_in, l_tmp_mat, l_vec_out_tpp4+i, l_idx+i, l_gather_2, l_mul_2, l_addreduce_2 );
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  printf("TPP gather + mul + reduce\nRuntime: %f; GiB/s: %f\n", l_runtime, (l_bytes/(1024.0*1024.0*1024.0))/(l_runtime/(double)iters));

  libxsmm_free( l_kvcache );
  libxsmm_free( l_vec_in );
  libxsmm_free( l_tmp_mat );
  libxsmm_free( l_vec_out_gold );
  libxsmm_free( l_vec_out_tpp1 );
  libxsmm_free( l_vec_out_tpp2 );
  libxsmm_free( l_vec_out_tpp3 );
  libxsmm_free( l_vec_out_tpp4 );
  libxsmm_free( l_idx );

  return ret;
}

int main( int argc, char* argv[] ) {
  int ret = EXIT_SUCCESS;

  libxsmm_blasint cols = 2048;
  libxsmm_blasint M = 256;
  libxsmm_blasint numidx = 512;
  libxsmm_blasint idxblk = 32;
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
    ret = eqn_kv_cache_one_f32(cols, M, idxblk, numidx, iters);
  }

  return ret;
}
