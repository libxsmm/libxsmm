/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <libxsmm_intrinsics_x86.h>

#define ALIGNDOWN(N, A) ((N) & ~((A)-1))

#define EPS 1.19209290e-03F


void reference_unpack(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, float *in, libxsmm_bfloat16 *out_lo, libxsmm_bfloat16 *out_hi) {
  libxsmm_blasint i, j;
  for (j = 0; j < N; j++) {
    for (i = 0; i < M; i++) {
      union libxsmm_bfloat16_hp bf16_hp;
      bf16_hp.f = in[j * ld + i];
      out_lo[j * ld + i] = bf16_hp.i[0];
      out_hi[j * ld + i] = bf16_hp.i[1];
    }
  }
}

void reference_pack(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, float *out, libxsmm_bfloat16 *in_lo, libxsmm_bfloat16 *in_hi) {
  libxsmm_blasint i, j;
  for (j = 0; j < N; j++) {
    for (i = 0; i < M; i++) {
      union libxsmm_bfloat16_hp bf16_hp;
      bf16_hp.i[0] = in_lo[j * ld + i];
      bf16_hp.i[1] = in_hi[j * ld + i];
      out[j * ld + i] = bf16_hp.f;
    }
  }
}

void reference_equation(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, libxsmm_bfloat16 *dwt, float lr, libxsmm_bfloat16 *out_lo, libxsmm_bfloat16 *out_hi) {
  libxsmm_blasint i, j;
  for (j = 0; j < N; j++) {
    for (i = 0; i < M; i++) {
      union libxsmm_bfloat16_hp bf16_hp;
      union libxsmm_bfloat16_hp bf16_wt;
      bf16_wt.i[0] = 0;
      bf16_wt.i[1] = dwt[j * ld + i];
      bf16_hp.i[0] = out_lo[j * ld + i];
      bf16_hp.i[1] = out_hi[j * ld + i];
      bf16_hp.f = bf16_wt.f * lr + bf16_hp.f;
      out_lo[j * ld + i] = bf16_hp.i[0];
      out_hi[j * ld + i] = bf16_hp.i[1];
    }
  }
}

#if defined(__AVX512F__)
inline __m512 convert_split_bf16_to_fp32(const __m256i src_hi, const __m256i src_lo) {
  __m512i y1 = _mm512_cvtepu16_epi32(src_hi);
  __m512i y2 = _mm512_cvtepu16_epi32(src_lo);
  return _mm512_castsi512_ps(_mm512_add_epi32(_mm512_bslli_epi128(y1, 2), y2));
}

inline __m512 convert_bf16_to_fp32(const __m256i src) {
  __m512i y = _mm512_cvtepu16_epi32(src);
  return _mm512_castsi512_ps(_mm512_bslli_epi128(y, 2));
}

inline __m256i  convert_fp32_to_bf16(const __m512 src) {
  __m512i y = _mm512_bsrli_epi128(_mm512_castps_si512(src), 2);
  return _mm512_cvtepi32_epi16(y);
}

inline void iadd_split_bf16(libxsmm_bfloat16 *inout_hi, libxsmm_bfloat16 *inout_lo, libxsmm_bfloat16 *in, int len, float alpha) {
  __m512 vAlpha = _mm512_set1_ps(alpha);
  int state_mask = ((1 << 16) - 1);
  __m512i vMask = _mm512_set1_epi32(state_mask);
  int i = 0;
  for(; i < len - 15; i += 16) {
    __m512 y1 = convert_split_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(inout_hi+i)), _mm256_loadu_si256((__m256i*)(inout_lo+i)));
    __m512 y2 = convert_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(in+i)));
    y1 = _mm512_fmadd_ps(vAlpha, y2, y1);
    _mm256_storeu_si256((__m256i*)(inout_hi+i), convert_fp32_to_bf16(y1));
    _mm256_storeu_si256((__m256i*)(inout_lo+i), _mm512_cvtepi32_epi16(_mm512_and_si512(_mm512_castps_si512(y1), vMask)));
  }
  if(i < len) {
    int rem = len - i;
    __mmask16 mask = (1 << rem) - 1;
    __m512 y1 = convert_split_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, inout_hi+i), _mm256_maskz_loadu_epi16(mask, inout_lo+i));
    __m512 y2 = convert_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, in+i));
    y1 = _mm512_fmadd_ps(vAlpha, y2, y1);
    _mm256_mask_storeu_epi16(inout_hi+i, mask, convert_fp32_to_bf16(y1));
    _mm256_mask_storeu_epi16(inout_lo+i, mask, _mm512_cvtepi32_epi16(_mm512_and_si512(_mm512_castps_si512(y1), vMask)));
  }
}

void vec_equation(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, libxsmm_bfloat16 *dwt, float lr, libxsmm_bfloat16 *out_lo, libxsmm_bfloat16 *out_hi) {
  libxsmm_blasint i, j;
  for (j = 0; j < N; j++) {
    iadd_split_bf16(&out_hi[j*ld], &out_lo[j*ld], &dwt[j*ld], M, lr);
  }
}
#endif

int main( int argc, char* argv[] ) {
  libxsmm_blasint my_eqn0;
  libxsmm_matrix_eqn_function func0;
  float *wt;
  float *f32_ref_out;
  float *f32_eqn_out;
  libxsmm_bfloat16 *bf16_dwt;
  libxsmm_bfloat16 *wt_lo, *wt_hi;
  libxsmm_bfloat16 *eqn_wt_lo, *eqn_wt_hi;
  long long offset;
  float lr = 0.7f;
  int M = 64;
  int N = 64;
  int ld = 64;
  /*unsigned int correct = 1;*/
  libxsmm_matrix_eqn_param eqn_param;
  libxsmm_matrix_arg arg_array[4];
  libxsmm_matdiff_info norms_out;
  int i, j, it = 0;
  unsigned long long l_start, l_end;
  double l_total = 0, l_total2 = 0;
  int iters = 100;

  if ( argc > 1 ) M = atoi(argv[1]);
  if ( argc > 2 ) N = atoi(argv[2]);
  if ( argc > 3 ) ld = atoi(argv[3]);
  if ( argc > 4 ) iters = atoi(argv[4]);

  wt = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  wt_lo = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld,   64);
  wt_hi = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld,   64);
  eqn_wt_lo = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld,   64);
  eqn_wt_hi = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld,   64);
  bf16_dwt = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld,   64);
  f32_ref_out = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  f32_eqn_out = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);

  libxsmm_init();
  libxsmm_matdiff_clear(&norms_out);

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ld; ++j ) {
      wt[(i*ld)+j] = (float)libxsmm_rng_f64();
    }
  }

  reference_unpack(M, N, ld, wt, wt_lo, wt_hi);
  memcpy(eqn_wt_lo, wt_lo, ld * N * sizeof (libxsmm_bfloat16));
  memcpy(eqn_wt_hi, wt_hi, ld * N * sizeof (libxsmm_bfloat16));

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ld; ++j ) {
      wt[(i*ld)+j] = (float)libxsmm_rng_f64();
    }
  }
  libxsmm_rne_convert_fp32_bf16( wt, bf16_dwt, ld * N );

  /* Split sgd via equation  */
  my_eqn0 = libxsmm_matrix_eqn_create();
  libxsmm_matrix_eqn_push_back_unary_op( my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_UNPACK_TO_BLOCKS, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn0, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, M, N, ld, 3, 0, LIBXSMM_DATATYPE_BF16 ); /* This is the "gradient" weights   */
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32 );   /* This is the scalar learning rate */
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_PACK, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, M, N, ld, 0, 0, LIBXSMM_DATATYPE_I16 );  /* This is the tensor with lo bits  */
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, M, N, ld, 1, 0, LIBXSMM_DATATYPE_I16 );  /* This is the tensor with hi bits  */
  func0 = libxsmm_dispatch_matrix_eqn( M, N, &ld, LIBXSMM_DATATYPE_I16, my_eqn0 );

  arg_array[0].primary = (void*)eqn_wt_lo;
  arg_array[1].primary = (void*)eqn_wt_hi;
  arg_array[2].primary = (void*)&lr;
  arg_array[3].primary = (void*)bf16_dwt;
  eqn_param.inputs = arg_array;
  eqn_param.output.primary = (void*)eqn_wt_lo;
  offset = (long long) ((char*)eqn_wt_hi - (char*)eqn_wt_lo);
  eqn_param.output.secondary = (void*)offset;
  func0(&eqn_param);

  /* Run reference split sgd  */
  reference_equation(M, N, ld, bf16_dwt, lr, wt_lo, wt_hi);

  reference_pack(M, N, ld, f32_ref_out, wt_lo, wt_hi);
  reference_pack(M, N, ld, f32_eqn_out, eqn_wt_lo, eqn_wt_hi);

  /* compare */
  printf("##########################\n");
  printf("# Correctness Split SGD  #\n");
  printf("##########################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, ld * N, 1, f32_ref_out, f32_eqn_out, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

  reference_equation(M, N, ld, bf16_dwt, lr, wt_lo, wt_hi);
  l_start = libxsmm_timer_tick();
  for (it = 0; it < iters; it++) {
    reference_equation(M, N, ld, bf16_dwt, lr, wt_lo, wt_hi);
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("Compiler equation time  = %.5g\n", ((double)(l_total)));

  func0(&eqn_param);
  l_start = libxsmm_timer_tick();
  for (it = 0; it < iters; it++) {
    func0(&eqn_param);
  }
  l_end = libxsmm_timer_tick();
  l_total2 = libxsmm_timer_duration(l_start, l_end);
  printf("JITed TPP equation time = %.5g\n", ((double)(l_total2)));
  printf("Speedup over compiler is %.5g\n", l_total/l_total2);

#if defined(__AVX512F__)
  vec_equation(M, N, ld, bf16_dwt, lr, wt_lo, wt_hi);
  l_start = libxsmm_timer_tick();
  for (it = 0; it < iters; it++) {
    vec_equation(M, N, ld, bf16_dwt, lr, wt_lo, wt_hi);
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("Vectorized equation time  = %.5g\n", ((double)(l_total)));
  printf("Speedup over vectorized code is %.5g\n", l_total/l_total2);
#endif

  libxsmm_free(wt);
  libxsmm_free(wt_lo);
  libxsmm_free(wt_hi);
  libxsmm_free(eqn_wt_lo);
  libxsmm_free(eqn_wt_hi);
  libxsmm_free(bf16_dwt);
  libxsmm_free(f32_ref_out);
  libxsmm_free(f32_eqn_out);

  return 0;
}

