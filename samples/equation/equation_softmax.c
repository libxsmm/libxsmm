/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <libxsmm_intrinsics_x86.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define FWD_SMAX 1
#define BWD_SMAX 2
#define FWD_BWD_SMAX 3

#if defined(__AVX512F__)
LIBXSMM_INLINE __m512 _mm512_loadu_ps_auto(libxsmm_bfloat16 const* mem_addr) { return LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)mem_addr)); }
LIBXSMM_INLINE __m512 _mm512_maskz_loadu_ps_auto(__mmask16 k, libxsmm_bfloat16 const* mem_addr) { return LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_maskz_loadu_epi16(k, (__m256i*)mem_addr)); }
LIBXSMM_INLINE void _mm512_storeu_ps_auto(libxsmm_bfloat16* mem_addr, __m512 a) { _mm256_storeu_si256((__m256i*)mem_addr, LIBXSMM_INTRINSICS_MM512_CVT_FP32_BF16(a)); }
LIBXSMM_INLINE void _mm512_mask_storeu_ps_auto(libxsmm_bfloat16* mem_addr, __mmask16 k, __m512 a) { _mm256_mask_storeu_epi16((__m256i*)mem_addr, k, LIBXSMM_INTRINSICS_MM512_CVT_FP32_BF16(a)); }
#endif

LIBXSMM_INLINE
float upconvert_bf16(libxsmm_bfloat16 x) {
  libxsmm_bfloat16_f32 bf16_hp;
  bf16_hp.i[1] = x;
  bf16_hp.i[0] = 0;
  return bf16_hp.f;
}

LIBXSMM_INLINE
void vectorized_softmax_fwd_bf16(long S1, long S2, long S3, libxsmm_bfloat16 *pinp, libxsmm_bfloat16 *pout) {
  int s1, s2, s3;
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, inp, pinp, S2, S3);
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, out, pout, S2, S3);
#if defined(__AVX512F__)
  for (s2 = 0; s2 < S2; s2++) {
    float tmp[S1][S3];
    float max = upconvert_bf16(LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3));
    float sum = 0.0;
    __m512 vmax = _mm512_set1_ps(max);
    __m512 vsum = _mm512_setzero_ps();

    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < LIBXSMM_LO2(S3, 16); s3+=16) {
        vmax = _mm512_max_ps(_mm512_loadu_ps_auto(&LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)), vmax);
      }
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        vmax = _mm512_mask_max_ps(vmax, mask, _mm512_maskz_loadu_ps_auto(mask, &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)), vmax);
      }
    }
    max = _mm512_reduce_max_ps(vmax);
    vmax = _mm512_set1_ps(max);
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < LIBXSMM_LO2(S3, 16); s3+=16) {
        __m512 vz = LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS(_mm512_sub_ps(_mm512_loadu_ps_auto(&LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)), vmax));
        _mm512_storeu_ps(&tmp[s1][s3], vz);
        vsum = _mm512_add_ps(vsum, vz);
      }
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        __m512 vz = LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS(_mm512_sub_ps(_mm512_maskz_loadu_ps_auto(mask, &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)), vmax));
        _mm512_mask_storeu_ps(&tmp[s1][s3], mask, vz);
        vsum = _mm512_mask_add_ps(vsum, mask, vsum, vz);
      }
    }
    sum = _mm512_reduce_add_ps(vsum);
    sum = 1.0 / sum;
    vsum = _mm512_set1_ps(sum);
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < LIBXSMM_LO2(S3, 16); s3+=16) {
        _mm512_storeu_ps_auto(&LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3), _mm512_mul_ps(vsum, _mm512_loadu_ps(&tmp[s1][s3])));
      }
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        _mm512_mask_storeu_ps_auto(&LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3), mask, _mm512_mul_ps(vsum, _mm512_maskz_loadu_ps(mask, &tmp[s1][s3])));
      }
    }
  }
#else
  for (s2 = 0; s2 < S2; s2++) {
    float buf[4096];
    LIBXSMM_VLA_DECL(2, float, tmp, buf, S3);
    float max = upconvert_bf16(LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3));
    float sum = 0.0;
    assert((sizeof(*buf) * S1 * S3) <= sizeof(buf));
    for ( s1 = 0; s1 < S1; s1++) {
      for ( s3 = 0; s3 < S3; s3++) {
        float cur = upconvert_bf16(LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
        if (max < cur) max = cur;
      }
    }
    for ( s1 = 0; s1 < S1; s1++) {
      for ( s3 = 0; s3 < S3; s3++) {
        float cur = upconvert_bf16(LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
        float z = LIBXSMM_EXPF(cur - max);
        LIBXSMM_VLA_ACCESS(2, tmp, s1, s3, S3) = z;
        sum += z;
      }
    }
    sum = (float)(1.0 / sum);
    for ( s1 = 0; s1 < S1; s1++) {
      for ( s3 = 0; s3 < S3; s3++) {
        float cur = LIBXSMM_VLA_ACCESS(2, tmp, s1, s3, S3) * sum;
        libxsmm_rne_convert_fp32_bf16( &cur, &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3), 1 );
      }
    }
  }
#endif
}

LIBXSMM_INLINE
void vectorized_softmax_bwd_bf16(long S1, long S2, long S3, float *pgradinp, float *pgradout, libxsmm_bfloat16 *pout) {
  int s1, s2, s3;
  LIBXSMM_VLA_DECL(3, float, ginp, pgradinp, S2, S3);
  LIBXSMM_VLA_DECL(3, float, gout, pgradout, S2, S3);
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, out, pout, S2, S3);
#if defined(__AVX512F__)
  for (s2 = 0; s2 < S2; s2++) {
    float sum = 0.0;
    __m512 vsum = _mm512_setzero_ps();
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < LIBXSMM_LO2(S3, 16); s3+=16) {
        __m512 vgo = _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3));
        __m512 vo = _mm512_loadu_ps_auto(&LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3));
        vsum = _mm512_fmadd_ps(vgo, vo, vsum);
      }
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        __m512 vgo = _mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3));
        __m512 vo = _mm512_maskz_loadu_ps_auto(mask, &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3));
        vsum = _mm512_fmadd_ps(vgo, vo, vsum);
      }
    }
    sum = _mm512_reduce_add_ps(vsum);
    vsum = _mm512_set1_ps(sum);
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < LIBXSMM_LO2(S3, 16); s3+=16) {
        __m512 tmp = _mm512_sub_ps(_mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3)), vsum);
        _mm512_storeu_ps(&LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3), _mm512_mul_ps(_mm512_loadu_ps_auto(&LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3)), tmp));
      }
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        __m512 tmp = _mm512_sub_ps(_mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3)), vsum);
        _mm512_mask_storeu_ps(&LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3), mask, _mm512_mul_ps(_mm512_maskz_loadu_ps_auto(mask, &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3)), tmp));
      }
    }
  }
#else
  for (s2 = 0; s2 < S2; s2++) {
    float sum = 0.0;
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < S3; s3++) {
        sum += LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3) * upconvert_bf16(LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3));
      }
    }
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < S3; s3++) {
        LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3) = upconvert_bf16(LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3)) * (LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3) - sum);
      }
    }
  }
#endif
}

LIBXSMM_INLINE
void vectorized_softmax_fwd(long S1, long S2, long S3, float *pinp, float *pout) {
  int s1, s2, s3;
  LIBXSMM_VLA_DECL(3, float, inp, pinp, S2, S3);
  LIBXSMM_VLA_DECL(3, float, out, pout, S2, S3);
#if defined(__AVX512F__)
  for (s2 = 0; s2 < S2; s2++) {
    float tmp[S1][S3];
    float max = LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3);
    float sum = 0.0;
    __m512 vmax = _mm512_set1_ps(max);
    __m512 vsum = _mm512_setzero_ps();

    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < LIBXSMM_LO2(S3, 16); s3+=16) {
        vmax = _mm512_max_ps(_mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)), vmax);
      }
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        vmax = _mm512_mask_max_ps(vmax, mask, _mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)), vmax);
      }
    }
    max = _mm512_reduce_max_ps(vmax);
    vmax = _mm512_set1_ps(max);
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < LIBXSMM_LO2(S3, 16); s3+=16) {
        __m512 vz = LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS(_mm512_sub_ps(_mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)), vmax));
        _mm512_storeu_ps(&tmp[s1][s3], vz);
        vsum = _mm512_add_ps(vsum, vz);
      }
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        __m512 vz = LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS(_mm512_sub_ps(_mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)), vmax));
        _mm512_mask_storeu_ps(&tmp[s1][s3], mask, vz);
        vsum = _mm512_mask_add_ps(vsum, mask, vsum, vz);
      }
    }
    sum = _mm512_reduce_add_ps(vsum);
    sum = 1.0 / sum;
    vsum = _mm512_set1_ps(sum);
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < LIBXSMM_LO2(S3, 16); s3+=16) {
        _mm512_storeu_ps(&LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3), _mm512_mul_ps(vsum, _mm512_loadu_ps(&tmp[s1][s3])));
      }
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        _mm512_mask_storeu_ps(&LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3), mask, _mm512_mul_ps(vsum, _mm512_maskz_loadu_ps(mask, &tmp[s1][s3])));
      }
    }
  }
#else
  for (s2 = 0; s2 < S2; s2++) {
    float buf[4096];
    LIBXSMM_VLA_DECL(2, float, tmp, buf, S3);
    float max = LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3);
    float sum = 0.0;
    assert((sizeof(*buf)* S1* S3) <= sizeof(buf));
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < S3; s3++) {
        if (max < LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)) max = LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3);
      }
    }
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < S3; s3++) {
        float z = LIBXSMM_EXPF(LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) - max);
        LIBXSMM_VLA_ACCESS(2, tmp, s1, s3, S3) = z;
        sum += z;
      }
    }
    sum = (float)(1.0 / sum);
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < S3; s3++) {
        LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3) = LIBXSMM_VLA_ACCESS(2, tmp, s1, s3, S3) * sum;
      }
    }
  }
#endif
}

LIBXSMM_INLINE
void vectorized_softmax_bwd(long S1, long S2, long S3, float *pgradinp, float *pgradout, float *pout) {
  int s1, s2, s3;
  LIBXSMM_VLA_DECL(3, float, ginp, pgradinp, S2, S3);
  LIBXSMM_VLA_DECL(3, float, gout, pgradout, S2, S3);
  LIBXSMM_VLA_DECL(3, float, out, pout, S2, S3);
#if defined(__AVX512F__)
  for (s2 = 0; s2 < S2; s2++) {
    float sum = 0.0;
    __m512 vsum = _mm512_setzero_ps();
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < LIBXSMM_LO2(S3, 16); s3+=16) {
        __m512 vgo = _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3));
        __m512 vo = _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3));
        vsum = _mm512_fmadd_ps(vgo, vo, vsum);
      }
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        __m512 vgo = _mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3));
        __m512 vo = _mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3));
        vsum = _mm512_fmadd_ps(vgo, vo, vsum);
      }
    }
    sum = _mm512_reduce_add_ps(vsum);
    vsum = _mm512_set1_ps(sum);
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < LIBXSMM_LO2(S3, 16); s3+=16) {
        __m512 tmp = _mm512_sub_ps(_mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3)), vsum);
        _mm512_storeu_ps(&LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3), _mm512_mul_ps(_mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3)), tmp));
      }
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        __m512 tmp = _mm512_sub_ps(_mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3)), vsum);
        _mm512_mask_storeu_ps(&LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3), mask, _mm512_mul_ps(_mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3)), tmp));
      }
    }
  }
#else
  for (s2 = 0; s2 < S2; s2++) {
    float sum = 0.0;
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < S3; s3++) {
        sum += LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3) * LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3);
      }
    }
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < S3; s3++) {
        LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3) = LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3) * (LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3) - sum);
      }
    }
  }
#endif
}

LIBXSMM_INLINE
void tpp_softmax_fwd(long S1, long S2, long S3, float *pinp, float *pout, libxsmm_matrix_eqn_function func0, libxsmm_matrix_eqn_function func1) {
  int s2;
  LIBXSMM_ALIGNED(float tmp[4096], 64);
  libxsmm_matrix_eqn_param eqn_param;
  LIBXSMM_VLA_DECL(3, float, inp, pinp, S2, S3);
  LIBXSMM_VLA_DECL(3, float, out, pout, S2, S3);
  libxsmm_matrix_arg arg_array[1];
  eqn_param.inputs = arg_array;
  assert((sizeof(*tmp) * S1 * S3) <= sizeof(tmp));
  for (s2 = 0; s2 < S2; s2++) {
    arg_array[0].primary = &LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3);
    eqn_param.output.primary = tmp;
    func0(&eqn_param);
    arg_array[0].primary = tmp;
    eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, out, 0, s2, 0, S2, S3);
    func1(&eqn_param);
  }
}

LIBXSMM_INLINE
void tpp_softmax_fwd_bf16(long S1, long S2, long S3, libxsmm_bfloat16 *pinp, libxsmm_bfloat16 *pout, libxsmm_matrix_eqn_function func0, libxsmm_matrix_eqn_function func1) {
  int s2;
  LIBXSMM_ALIGNED(float tmp[4096], 64);
  libxsmm_matrix_eqn_param eqn_param;
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, inp, pinp, S2, S3);
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, out, pout, S2, S3);
  libxsmm_matrix_arg  arg_array[1];
  eqn_param.inputs = arg_array;
  assert((sizeof(*tmp) * S1 * S3) <= sizeof(tmp));
  for (s2 = 0; s2 < S2; s2++) {
    arg_array[0].primary = &LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3);
    eqn_param.output.primary = tmp;
    func0(&eqn_param);
    arg_array[0].primary = tmp;
    eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, out, 0, s2, 0, S2, S3);
    func1(&eqn_param);
  }
}

#if 1
LIBXSMM_INLINE
void tpp_softmax_bwd(long S1, long S2, long S3, float *pgradinp, float *pgradout, float *pout, libxsmm_matrix_eqn_function func0, libxsmm_matrix_eqn_function func1) {
  int s2;
  LIBXSMM_ALIGNED(float tmp[4096], 64);
  libxsmm_matrix_eqn_param eqn_param;
  LIBXSMM_VLA_DECL(3, float, ginp, pgradinp, S2, S3);
  LIBXSMM_VLA_DECL(3, float, gout, pgradout, S2, S3);
  LIBXSMM_VLA_DECL(3, float, out, pout, S2, S3);
  libxsmm_matrix_arg arg_array[2];
  eqn_param.inputs = arg_array;
  assert((sizeof(*tmp) * S1 * S3) <= sizeof(tmp));
  for (s2 = 0; s2 < S2; s2++) {
    arg_array[0].primary = &LIBXSMM_VLA_ACCESS(3, gout, 0, s2, 0, S2, S3);
    arg_array[1].primary = &LIBXSMM_VLA_ACCESS(3, out, 0, s2, 0, S2, S3);
    eqn_param.output.primary = tmp;
    func0(&eqn_param);
    arg_array[0].primary = tmp;
    eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, ginp, 0, s2, 0, S2, S3);
    func1(&eqn_param);
  }
}

LIBXSMM_INLINE
void tpp_softmax_bwd_bf16(long S1, long S2, long S3, float *pgradinp, float *pgradout, libxsmm_bfloat16 *pout, libxsmm_matrix_eqn_function func0, libxsmm_matrix_eqn_function func1) {
  int s2;
  LIBXSMM_ALIGNED(float tmp[4096], 64);
  libxsmm_matrix_eqn_param eqn_param;
  LIBXSMM_VLA_DECL(3, float, ginp, pgradinp, S2, S3);
  LIBXSMM_VLA_DECL(3, float, gout, pgradout, S2, S3);
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, out, pout, S2, S3);
  libxsmm_matrix_arg arg_array[2];
  eqn_param.inputs = arg_array;
  assert((sizeof(*tmp) * S1 * S3) <= sizeof(tmp));
  for (s2 = 0; s2 < S2; s2++) {
    arg_array[0].primary = &LIBXSMM_VLA_ACCESS(3, gout, 0, s2, 0, S2, S3);
    arg_array[1].primary = &LIBXSMM_VLA_ACCESS(3, out, 0, s2, 0, S2, S3);
    eqn_param.output.primary = tmp;
    func0(&eqn_param);
    arg_array[0].primary = tmp;
    eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, ginp, 0, s2, 0, S2, S3);
    func1(&eqn_param);
  }
}
#else
LIBXSMM_INLINE
void tpp_softmax_bwd(long S1, long S2, long S3, float *pgradinp, float *pgradout, float *pout, libxsmm_matrix_eqn_function func0, libxsmm_matrix_eqn_function funcfoo) {
  int s1, s2, s3;
  libxsmm_matrix_eqn_param eqn_param;
  LIBXSMM_VLA_DECL(3, float, ginp, pgradinp, S2, S3);
  LIBXSMM_VLA_DECL(3, float, gout, pgradout, S2, S3);
  LIBXSMM_VLA_DECL(3, float, out, pout, S2, S3);
  libxsmm_matrix_arg arg_array[2];
  eqn_param.inputs = arg_array;
  for (s2 = 0; s2 < S2; s2++) {
    arg_array[0].primary = &LIBXSMM_VLA_ACCESS(3, gout, 0, s2, 0, S2, S3);
    arg_array[1].primary = &LIBXSMM_VLA_ACCESS(3, out, 0, s2, 0, S2, S3);
    eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, ginp, 0, s2, 0, S2, S3);
    func0(&eqn_param);
  }
}

LIBXSMM_INLINE
void tpp_softmax_bwd_bf16(long S1, long S2, long S3, float *pgradinp, float *pgradout, libxsmm_bfloat16 *pout, libxsmm_matrix_eqn_function func0, libxsmm_matrix_eqn_function funcfoo) {
  int s1, s2, s3;
  libxsmm_matrix_eqn_param eqn_param;
  LIBXSMM_VLA_DECL(3, float, ginp, pgradinp, S2, S3);
  LIBXSMM_VLA_DECL(3, float, gout, pgradout, S2, S3);
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, out, pout, S2, S3);
  libxsmm_matrix_arg arg_array[2];
  eqn_param.inputs = arg_array;
  for (s2 = 0; s2 < S2; s2++) {
    arg_array[0].primary = &LIBXSMM_VLA_ACCESS(3, gout, 0, s2, 0, S2, S3);
    arg_array[1].primary = &LIBXSMM_VLA_ACCESS(3, out, 0, s2, 0, S2, S3);
    eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, ginp, 0, s2, 0, S2, S3);
    func0(&eqn_param);
  }
}
#endif

int main( int argc, char* argv[] ) {
  int ret = EXIT_SUCCESS;
  double error_bound = 0.00001;
  libxsmm_blasint my_eqn0, my_eqn1, my_eqn2, my_eqn3;
  libxsmm_matrix_eqn_function func0, func1, func2, func3;
  libxsmm_blasint i, it, ld, tmp_ld;
  unsigned long long l_start, l_end;
  double l_total = 0, l_total2 = 0;
  double t_vec = 0, t_tpp = 0;
  libxsmm_matdiff_info norms_out;
  float *inp, *out, *eqn_out, *gout, *cache_fl, sum = 0.0;
  libxsmm_bfloat16 *bf16_inp, *bf16_out, *bf16_eqn_out;
  int S1 = 64;
  int S2 = 64;
  int S3 = 64;
  int iters = 100;
  int datatype_mode = 0;
  int pass = FWD_BWD_SMAX;
  libxsmm_datatype  in_dt = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype  out_dt = LIBXSMM_DATATYPE_F32;
  libxsmm_meqn_arg_shape arg_shape_out;

  if ( argc > 1 ) S1 = atoi(argv[1]);
  if ( argc > 2 ) S2 = atoi(argv[2]);
  if ( argc > 3 ) S3 = atoi(argv[3]);
  if ( argc > 4 ) datatype_mode = atoi(argv[4]);
  if ( argc > 5 ) pass = atoi(argv[5]);
  if ( argc > 6 ) iters = atoi(argv[6]);

  if (datatype_mode == 0) {
    in_dt = LIBXSMM_DATATYPE_F32;
    out_dt = LIBXSMM_DATATYPE_F32;
  } else if (datatype_mode == 1) {
    in_dt = LIBXSMM_DATATYPE_BF16;
    out_dt = LIBXSMM_DATATYPE_BF16;
  } else {
    printf("ERROR: Supporting only FP32 and BF16 precisions...\n");
  }

  inp = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*S2*S3,   2097152);
  out = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*S2*S3,   2097152);
  gout = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*S2*S3,   2097152);
  eqn_out  = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*S2*S3,   2097152);
  cache_fl  = (float*) libxsmm_aligned_malloc( sizeof(float)*1024*1024,   2097152);

  bf16_inp = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*S1*S2*S3,   2097152);
  bf16_out = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*S1*S2*S3,   2097152);
  bf16_eqn_out  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*S1*S2*S3,   2097152);

  libxsmm_init();
  libxsmm_matdiff_clear(&norms_out);

  /* Initializing arrays */
  for ( i = 0; i < S1*S2*S3; ++i ) {
    inp[i] = (float)libxsmm_rng_f64();
    out[i] = (float)libxsmm_rng_f64();
    gout[i] = (float)libxsmm_rng_f64();
    eqn_out[i] = out[i];
    libxsmm_rne_convert_fp32_bf16( &inp[i], &bf16_inp[i], 1 );
    libxsmm_rne_convert_fp32_bf16( &out[i], &bf16_out[i], 1 );
    libxsmm_rne_convert_fp32_bf16( &eqn_out[i], &bf16_eqn_out[i], 1 );
  }
  for (i = 0; i < 1024 * 1024; i++ ) {
    cache_fl[i] = (float)libxsmm_rng_f64();
  }

  /* Create MatEq for fwd softmax */
  if ((pass & FWD_SMAX) > 0) {
    tmp_ld = S3;
    ld = S2*S3;
    my_eqn0 = libxsmm_matrix_eqn_create();
    libxsmm_matrix_eqn_push_back_unary_op( my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_EXP, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_DATATYPE_F32 );
    libxsmm_matrix_eqn_push_back_binary_op( my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_SUB, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1, LIBXSMM_DATATYPE_F32 );
    libxsmm_matrix_eqn_push_back_arg( my_eqn0, S3, S1, ld, 0, 0, in_dt );
    libxsmm_matrix_eqn_push_back_unary_op( my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS, LIBXSMM_DATATYPE_F32 );
    libxsmm_matrix_eqn_push_back_unary_op( my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32 );
    libxsmm_matrix_eqn_push_back_arg( my_eqn0, S3, S1, ld, 0, 0, in_dt );
    /*libxsmm_matrix_eqn_tree_print( my_eqn0 );*/
    arg_shape_out = libxsmm_create_meqn_arg_shape( S3, S1, tmp_ld, LIBXSMM_DATATYPE_F32 );
    func0 = libxsmm_dispatch_matrix_eqn_v2( my_eqn0, arg_shape_out );

    my_eqn1 = libxsmm_matrix_eqn_create();
    libxsmm_matrix_eqn_push_back_binary_op( my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1, LIBXSMM_DATATYPE_F32 );
    libxsmm_matrix_eqn_push_back_arg( my_eqn1, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32 );
    libxsmm_matrix_eqn_push_back_unary_op( my_eqn1, LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_DATATYPE_F32 );
    libxsmm_matrix_eqn_push_back_unary_op( my_eqn1, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS, LIBXSMM_DATATYPE_F32 );
    libxsmm_matrix_eqn_push_back_unary_op( my_eqn1, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32 );
    libxsmm_matrix_eqn_push_back_arg( my_eqn1, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32 );
    /*libxsmm_matrix_eqn_tree_print( my_eqn1 );*/
    arg_shape_out = libxsmm_create_meqn_arg_shape( S3, S1, ld, out_dt );
    func1 = libxsmm_dispatch_matrix_eqn_v2( my_eqn1, arg_shape_out );

    if (datatype_mode == 0) {
      vectorized_softmax_fwd(S1, S2, S3, inp, out);
      tpp_softmax_fwd(S1, S2, S3, inp, eqn_out, func0, func1);
    } else if (datatype_mode == 1) {
      vectorized_softmax_fwd_bf16(S1, S2, S3, bf16_inp, bf16_out);
      tpp_softmax_fwd_bf16(S1, S2, S3, bf16_inp, bf16_eqn_out, func0, func1);
      for ( i = 0; i < S1*S2*S3; ++i ) {
        out[i] = upconvert_bf16(bf16_out[i]);
        eqn_out[i] = upconvert_bf16(bf16_eqn_out[i]);
      }
    }

    /* compare */
    printf("##########################################\n");
    if (datatype_mode == 0) {
      printf("# Correctness FP32 FWD Softmax - Output  #\n");
    } else {
      printf("# Correctness BF16 FWD Softmax - Output  #\n");
    }
    printf("##########################################\n");
    libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, S1*S2*S3, 1, out, eqn_out, 0, 0);
    printf("L1 reference  : %.25g\n", norms_out.l1_ref);
    printf("L1 test       : %.25g\n", norms_out.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

    if ( norms_out.normf_rel > error_bound ) {
      ret = EXIT_FAILURE;
    }

    if (iters > 0) {
      if (datatype_mode == 0) {
        for (i = 0; i < 1024 * 1024; i++ ) {
          sum += cache_fl[i];
        }
        vectorized_softmax_fwd(S1, S2, S3, inp, out);
        l_start = libxsmm_timer_tick();
        for (it = 0; it < iters; it++) {
          vectorized_softmax_fwd(S1, S2, S3, inp, out);
        }
        l_end = libxsmm_timer_tick();
        l_total = libxsmm_timer_duration(l_start, l_end);
        printf("Intrinsics softmax time FWD  = %.5g\n", ((double)(l_total)));
        for (i = 0; i < 1024 * 1024; i++ ) {
          sum += cache_fl[i] + (float)l_total;
        }
        tpp_softmax_fwd(S1, S2, S3, inp, eqn_out, func0, func1);
        l_start = libxsmm_timer_tick();
        for (it = 0; it < iters; it++) {
          tpp_softmax_fwd(S1, S2, S3, inp, eqn_out, func0, func1);
        }
        l_end = libxsmm_timer_tick();
        l_total2 = libxsmm_timer_duration(l_start, l_end);
        printf("TPP softmax time FWD  = %.5g\n", ((double)(l_total2)));
        printf("Speedup FWD is %.5g\n", l_total/l_total2);
      } else if (datatype_mode == 1) {
        for (i = 0; i < 1024 * 1024; i++ ) {
          sum += cache_fl[i];
        }
        vectorized_softmax_fwd_bf16(S1, S2, S3, bf16_inp, bf16_out);
        l_start = libxsmm_timer_tick();
        for (it = 0; it < iters; it++) {
          vectorized_softmax_fwd_bf16(S1, S2, S3, bf16_inp, bf16_out);
        }
        l_end = libxsmm_timer_tick();
        l_total = libxsmm_timer_duration(l_start, l_end);
        printf("Intrinsics softmax time FWD = %.5g\n", ((double)(l_total)));
        for (i = 0; i < 1024 * 1024; i++ ) {
          sum += cache_fl[i] + (float)l_total;
        }
        tpp_softmax_fwd_bf16(S1, S2, S3, bf16_inp, bf16_eqn_out, func0, func1);
        l_start = libxsmm_timer_tick();
        for (it = 0; it < iters; it++) {
          tpp_softmax_fwd_bf16(S1, S2, S3, bf16_inp, bf16_eqn_out, func0, func1);
        }
        l_end = libxsmm_timer_tick();
        l_total2 = libxsmm_timer_duration(l_start, l_end);
        printf("TPP softmax time FWD  = %.5g\n", ((double)(l_total2)));
        printf("Speedup FWD is %.5g\n", l_total/l_total2);
      }

      t_tpp = l_total2;
      t_vec = l_total;
    }
  }

  /* Create MatEq for bwd softmax */
#if 1
  if ((pass & BWD_SMAX) > 0) {
    tmp_ld = S3;
    ld = S2*S3;

    my_eqn2 = libxsmm_matrix_eqn_create();
    libxsmm_matrix_eqn_push_back_binary_op( my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
    libxsmm_matrix_eqn_push_back_arg( my_eqn2, S3, S1, ld, 0, 0, LIBXSMM_DATATYPE_F32 );
    libxsmm_matrix_eqn_push_back_arg( my_eqn2, S3, S1, ld, 1, 0, in_dt );
    arg_shape_out = libxsmm_create_meqn_arg_shape( S3, S1, tmp_ld, LIBXSMM_DATATYPE_F32 );
    func2 = libxsmm_dispatch_matrix_eqn_v2( my_eqn2, arg_shape_out );
#if 0
    my_eqn3 = libxsmm_matrix_eqn_create();
    libxsmm_matrix_eqn_push_back_binary_op( my_eqn3, LIBXSMM_MELTW_TYPE_BINARY_SUB, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
    libxsmm_matrix_eqn_push_back_arg( my_eqn3, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32 );
    libxsmm_matrix_eqn_push_back_binary_op( my_eqn3, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0, LIBXSMM_DATATYPE_F32 );
    libxsmm_matrix_eqn_push_back_unary_op( my_eqn3, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS, LIBXSMM_DATATYPE_F32 );
    libxsmm_matrix_eqn_push_back_unary_op( my_eqn3, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32 );
    libxsmm_matrix_eqn_push_back_arg( my_eqn3, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32 );
    libxsmm_matrix_eqn_push_back_arg( my_eqn3, S3, S1, ld, 1, 0, in_dt );
    arg_shape_out = libxsmm_create_meqn_arg_shape( S3, S1, ld, LIBXSMM_DATATYPE_F32 );
    func3 = libxsmm_dispatch_matrix_eqn_v2( my_eqn3, arg_shape_out );
#else
    my_eqn3 = libxsmm_matrix_eqn_create();
    libxsmm_matrix_eqn_push_back_ternary_op( my_eqn3, LIBXSMM_MELTW_TYPE_TERNARY_NMULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_0 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32);
    libxsmm_matrix_eqn_push_back_unary_op( my_eqn3, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS, LIBXSMM_DATATYPE_F32 );
    libxsmm_matrix_eqn_push_back_unary_op( my_eqn3, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32 );
    libxsmm_matrix_eqn_push_back_arg( my_eqn3, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32 );
    libxsmm_matrix_eqn_push_back_arg( my_eqn3, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32 );
    libxsmm_matrix_eqn_push_back_arg( my_eqn3, S3, S1, ld, 1, 0, in_dt );
    arg_shape_out = libxsmm_create_meqn_arg_shape( S3, S1, ld, LIBXSMM_DATATYPE_F32 );
    func3 = libxsmm_dispatch_matrix_eqn_v2( my_eqn3, arg_shape_out );
#endif
#else
    ld = S2*S3;
    my_eqn2 = libxsmm_matrix_eqn_create();
    libxsmm_matrix_eqn_push_back_ternary_op( my_eqn2, LIBXSMM_MELTW_TYPE_TERNARY_NMULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_0 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
    libxsmm_matrix_eqn_push_back_binary_op( my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
    libxsmm_matrix_eqn_push_back_arg( my_eqn2, S3, S1, ld, 0, 0, LIBXSMM_DATATYPE_F32 );
    libxsmm_matrix_eqn_push_back_arg( my_eqn2, S3, S1, ld, 1, 0, in_dt );
    libxsmm_matrix_eqn_push_back_binary_op( my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
    libxsmm_matrix_eqn_push_back_arg( my_eqn2, S3, S1, ld, 0, 0, LIBXSMM_DATATYPE_F32 );
    libxsmm_matrix_eqn_push_back_arg( my_eqn2, S3, S1, ld, 1, 0, in_dt );
    libxsmm_matrix_eqn_push_back_arg( my_eqn2, S3, S1, ld, 1, 0, in_dt );
    func2 = libxsmm_dispatch_matrix_eqn( S3, S1, &ld, LIBXSMM_DATATYPE_F32, my_eqn2 );
#endif

    if (datatype_mode == 0) {
      vectorized_softmax_bwd(S1, S2, S3, out, inp, gout);
      tpp_softmax_bwd(S1, S2, S3, eqn_out, inp, gout, func2, func3);
    } else if (datatype_mode == 1) {
      vectorized_softmax_bwd_bf16(S1, S2, S3, out, inp, bf16_out);
      tpp_softmax_bwd_bf16(S1, S2, S3, eqn_out, inp, bf16_out, func2, func3);
    }

    /* compare */
    printf("##########################################\n");
    if (datatype_mode == 0) {
      printf("# Correctness FP32 BWD Softmax - Output  #\n");
    } else {
      printf("# Correctness BF16 BWD Softmax - Output  #\n");
    }
    printf("##########################################\n");
    libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, S1*S2*S3, 1, out, eqn_out, 0, 0);
    printf("L1 reference  : %.25g\n", norms_out.l1_ref);
    printf("L1 test       : %.25g\n", norms_out.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

    if ( norms_out.normf_rel > error_bound ) {
      ret = EXIT_FAILURE;
    }

    if (iters > 0 ) {
      if (datatype_mode == 0) {
        for (i = 0; i < 1024 * 1024; i++ ) {
          sum += cache_fl[i];
        }
        vectorized_softmax_bwd(S1, S2, S3, out, inp, gout);
        l_start = libxsmm_timer_tick();
        for (it = 0; it < iters; it++) {
          vectorized_softmax_bwd(S1, S2, S3, out, inp, gout);
        }
        l_end = libxsmm_timer_tick();
        l_total = libxsmm_timer_duration(l_start, l_end);
        printf("Intrinsics softmax time BWD  = %.5g\n", ((double)(l_total)));
        for (i = 0; i < 1024 * 1024; i++ ) {
          sum += cache_fl[i] + (float)l_total;
        }
        tpp_softmax_bwd(S1, S2, S3, eqn_out, inp, gout, func2, func3);
        l_start = libxsmm_timer_tick();
        for (it = 0; it < iters; it++) {
          tpp_softmax_bwd(S1, S2, S3, eqn_out, inp, gout, func2, func3);
        }
        l_end = libxsmm_timer_tick();
        l_total2 = libxsmm_timer_duration(l_start, l_end);
        printf("TPP softmax time BWD  = %.5g\n", ((double)(l_total2)));
        printf("Speedup BWD is %.5g\n", l_total/l_total2);
      } else if (datatype_mode == 1) {
        for (i = 0; i < 1024 * 1024; i++ ) {
          sum += cache_fl[i];
        }
        vectorized_softmax_bwd_bf16(S1, S2, S3, out, inp, bf16_out);
        l_start = libxsmm_timer_tick();
        for (it = 0; it < iters; it++) {
          vectorized_softmax_bwd_bf16(S1, S2, S3, out, inp, bf16_out);
        }
        l_end = libxsmm_timer_tick();
        l_total = libxsmm_timer_duration(l_start, l_end);
        printf("Intrinsics softmax time BWD = %.5g\n", ((double)(l_total)));
        for (i = 0; i < 1024 * 1024; i++ ) {
          sum += cache_fl[i] + (float)l_total;
        }
        tpp_softmax_bwd_bf16(S1, S2, S3, eqn_out, inp, bf16_out, func2, func3);
        l_start = libxsmm_timer_tick();
        for (it = 0; it < iters; it++) {
          tpp_softmax_bwd_bf16(S1, S2, S3, eqn_out, inp, bf16_out, func2, func3);
        }
        l_end = libxsmm_timer_tick();
        l_total2 = libxsmm_timer_duration(l_start, l_end);
        printf("TPP softmax time BWD  = %.5g\n", ((double)(l_total2)));
        printf("Speedup BWD is %.5g\n", l_total/l_total2);
      }
      /* printf("Running sum is %.5f\n", sum); */

      t_tpp += l_total2;
      t_vec += l_total;
    }
  }

  if (iters > 0) {
    printf("\n\n=================================\n");
    printf("Total Speedup via TPP Matrix equation is %.5g\n", t_vec/t_tpp);
    printf("=================================\n");
  }
  libxsmm_free(inp);
  libxsmm_free(out);
  libxsmm_free(gout);
  libxsmm_free(eqn_out);
  libxsmm_free(bf16_inp);
  libxsmm_free(bf16_out);
  libxsmm_free(bf16_eqn_out);
  libxsmm_free(cache_fl);

  return ret;
}
