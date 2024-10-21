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
#include "generator_mateltwise_common.h"
#include "generator_common.h"
#include "generator_mateltwise_reference_impl.h"

LIBXSMM_API_INTERN libxsmm_float16 my_libxsmm_convert_f32_to_f16(float in)
{
  unsigned int f32_bias = 127;
  unsigned int f16_bias = 15;
  libxsmm_float_uint hybrid_in = { 0 };
  libxsmm_float16 res = 0;
  unsigned int s, e, m, e_f32, m_f32;
  unsigned int fixup;
  hybrid_in.f = in;

  /* DAZ */
  hybrid_in.u = ((hybrid_in.u & 0x7f800000) == 0x0
    ? (hybrid_in.u & 0x80000000)
    : (hybrid_in.u & 0xffffffff));

  s = (hybrid_in.u & 0x80000000) >> 16;
  e_f32 = (hybrid_in.u & 0x7f800000) >> 23;
  m_f32 = (hybrid_in.u & 0x007fffff);

  /* special value */
  if (e_f32 == 0xff) {
    e = 0x1f;
    m = (m_f32 == 0 ? 0 : ((m_f32 >> 13) | 0x200));
    /* overflow */
  }
  else if (e_f32 > (f32_bias + f16_bias)) {
    e = 0x1f;
    m = 0x0;
    /* smaller than denormal f16 */
  }
  else if (e_f32 < f32_bias - f16_bias - 10) {
    e = 0x0;
    m = 0x0;
    /* denormal */
  }
  else if (e_f32 <= f32_bias - f16_bias) {
    /* RNE */
#if 1
    /* denormalized mantissa */
    m = m_f32 | 0x00800000;
    /* additionally subnormal shift */
    m = m >> ((f32_bias - f16_bias) + 1 - e_f32);
    /* preserve sticky bit (some sticky bits are lost when denormalizing) */
    m |= (((m_f32 & 0x1fff) + 0x1fff) >> 13);
    /* RNE Round */
    fixup = (m >> 13) & 0x1;
    m = m + 0x000000fff + fixup;
    m = m >> 13;
    e = 0x0;
#else
    /* RAZ */
    m = (m_f32 | 0x00800000) >> 12;
    m = (m >> ((f32_bias - f16_bias) + 2 - e_f32)) + ((m >> ((f32_bias - f16_bias) + 1 - e_f32)) & 1);
    e = 0x0;
#endif
    /* normal */
  }
  else {
#if 1
    /* RNE round */
    fixup = (m_f32 >> 13) & 0x1;
    hybrid_in.u = hybrid_in.u + 0x000000fff + fixup;
    e = (hybrid_in.u & 0x7f800000) >> 23;
    m = (hybrid_in.u & 0x007fffff);
    LIBXSMM_ASSERT(e >= (f32_bias - f16_bias));
    e -= (f32_bias - f16_bias);
    m = m >> 13;
#else
    /* RAZ */
    hybrid_in.u = hybrid_in.u + 0x00001000;
    e = (hybrid_in.u & 0x7f800000) >> 23;
    m = (hybrid_in.u & 0x007fffff);
    LIBXSMM_ASSERT(e >= (f32_bias - f16_bias));
    e -= (f32_bias - f16_bias);
    m = m >> 13;
#endif
  }

  /* set result to 0 */
  res = 0x0;
  /* set exponent and mantissa */
  res |= e << 10;
  res |= m;
  /* sign it */
  res |= s;

  return res;
}

LIBXSMM_API_INTERN float my_libxsmm_convert_f16_to_f32(libxsmm_float16 in)
{
  unsigned int f32_bias = 127;
  unsigned int f16_bias = 15;
  unsigned int s = (in & 0x8000) << 16;
  unsigned int e = (in & 0x7c00) >> 10;
  unsigned int m = (in & 0x03ff);
  unsigned int e_norm = e + (f32_bias - f16_bias);
  libxsmm_float_uint res = { 0 };

  /* convert denormal fp16 number into a normal fp32 number */
  if ((e == 0) && (m != 0)) {
    unsigned int lz_cnt = 9;
    lz_cnt = (m > 0x1 ? 8 : lz_cnt);
    lz_cnt = (m > 0x3 ? 7 : lz_cnt);
    lz_cnt = (m > 0x7 ? 6 : lz_cnt);
    lz_cnt = (m > 0xf ? 5 : lz_cnt);
    lz_cnt = (m > 0x1f ? 4 : lz_cnt);
    lz_cnt = (m > 0x3f ? 3 : lz_cnt);
    lz_cnt = (m > 0x7f ? 2 : lz_cnt);
    lz_cnt = (m > 0xff ? 1 : lz_cnt);
    lz_cnt = (m > 0x1ff ? 0 : lz_cnt);
    LIBXSMM_ASSERT(e_norm >= lz_cnt);
    e_norm -= lz_cnt;
    m = (m << (lz_cnt + 1)) & 0x03ff;
  }
  else if ((e == 0) && (m == 0)) {
    e_norm = 0;
  }
  else if (e == 0x1f) {
    e_norm = 0xff;
    m |= (m == 0 ? 0 : 0x0200); /* making first mantissa bit 1 */
  }

  /* set result to 0 */
  res.u = 0x0;
  /* set exponent and mantissa */
  res.u |= (e_norm << 23);
  res.u |= (m << 13);
  /* sign it */
  res.u |= s;

  return res.f;
}

LIBXSMM_API_INTERN void my_libxsmm_lsfr_i32(unsigned int* rng_state, unsigned int* prng_out, const unsigned int seed_idx) {
  unsigned int rng_num = 0;
  const unsigned int state_ld = 16;

  unsigned int state_0 = rng_state[seed_idx + (0 * state_ld)];
  unsigned int state_1 = rng_state[seed_idx + (1 * state_ld)];
  unsigned int state_2 = rng_state[seed_idx + (2 * state_ld)];
  unsigned int state_3 = rng_state[seed_idx + (3 * state_ld)];

  unsigned int tmp_0, tmp_1;
  tmp_0 = (state_0 + state_3) << 7;
  tmp_1 = (state_0 + state_3) >> 25;
  rng_num = (tmp_0 | tmp_1) + state_0;
  prng_out[0] = rng_num;
  tmp_0 = state_1 << 9;
  state_2 = state_2 ^ state_0;
  state_3 = state_3 ^ state_1;
  state_1 = state_1 ^ state_2;
  state_0 = state_0 ^ state_3;
  state_2 = state_2 ^ tmp_0;
  tmp_0 = state_3 << 11;
  tmp_1 = state_3 >> 21;
  state_3 = tmp_0 | tmp_1;
  rng_state[seed_idx + (0 * state_ld)] = state_0;
  rng_state[seed_idx + (1 * state_ld)] = state_1;
  rng_state[seed_idx + (2 * state_ld)] = state_2;
  rng_state[seed_idx + (3 * state_ld)] = state_3;
}

LIBXSMM_API_INTERN void my_libxsmm_stochastic_convert_fp32_bf8(const float* in, libxsmm_bfloat8* out, unsigned int len, void *rng_state, unsigned int start_seed_idx) {
  unsigned int i = 0;
  unsigned int j = 0;

  /* truncate buffer to bf8 */
  for ( i = 0; i < len; i+=16 ) {
    unsigned int j_length = ( i + 16 < len ) ? 16 : len - i;
    for (j=0; j < j_length; j++) {
      libxsmm_float16_ushort hybrid_in = { 0 };
      libxsmm_bfloat8 res;
      unsigned short fixup;
      unsigned int vrng;
      unsigned short rand;

      hybrid_in.f = my_libxsmm_convert_f32_to_f16( in[i+j] );
      my_libxsmm_lsfr_i32((unsigned int*)rng_state, &vrng, (start_seed_idx + j) % 16);
      rand = (unsigned short)((vrng >> 24) & 0xff);

      /* RNE fixup */
      fixup = (unsigned short)((hybrid_in.u >> 8) & 1);

      /* we do not round inf and NaN */
      hybrid_in.u = (unsigned short)(((hybrid_in.u & 0x7c00) == 0x7c00)
        ? ( ((hybrid_in.u & 0x03ff) == 0x0) ? hybrid_in.u : hybrid_in.u | 0x0200 )
        /* we only stochastically round normal numbers, RNE for subnormal */
        : ( ((hybrid_in.u & 0x7c00) == 0x0000) ? hybrid_in.u + 0x007f + fixup : hybrid_in.u + rand ) );

      /* shift right */
      res = (libxsmm_bfloat8)(hybrid_in.u >> 8);

      out[i+j] = res;
    }
  }
}

LIBXSMM_API_INTERN float my_libxsmm_convert_bf8_to_f32(libxsmm_bfloat8 in)
{
  const unsigned short inus = (unsigned short)in;
  const unsigned short tmp = (unsigned short)(inus << 8);
  return my_libxsmm_convert_f16_to_f32(tmp);
}


LIBXSMM_API_INTERN float my_libxsmm_convert_hf8_to_f32(libxsmm_hfloat8 in)
{
  const unsigned int f32_bias = 127, f8_bias = 7;
  const unsigned int s = (in & 0x80 ) << 24;
  const unsigned int e = (in & 0x78 ) >> 3;
  unsigned int m = (in & 0x07 );
  unsigned int e_norm = e + (f32_bias - f8_bias);
  libxsmm_float_uint res;
  /* convert denormal fp8 number into a normal fp32 number */
  if ( (e == 0) && (m != 0) ) {
    unsigned int lz_cnt = 2;
    lz_cnt = (m > 0x1 ? 1 : lz_cnt);
    lz_cnt = (m > 0x3 ? 0 : lz_cnt);
    LIBXSMM_ASSERT(e_norm >= lz_cnt);
    e_norm -= lz_cnt;
    m = (m << (lz_cnt+1)) & 0x07;
  } else if (e == 0 && m == 0) {
    e_norm = 0;
  } else if (e == 0xf && m == 0x7) {
    e_norm = 0xff;
    m = 0x4; /* making first mantissa bit 1 */
  }
  /* set result to 0 */
  res.u = 0x0;
  /* set exponent and mantissa */
  res.u |= (e_norm << 23);
  res.u |= (m << 20);
  /* sign it */
  res.u |= s;
  return res.f;
}


LIBXSMM_API_INTERN float my_libxsmm_convert_bf16_to_f32(libxsmm_bfloat16 in)
{
  libxsmm_float_uint hybrid_in = { 0 };
  hybrid_in.u = in;
  /* DAZ */
  hybrid_in.u = ((hybrid_in.u & 0x7f80) == 0x0
    ? (unsigned short)(hybrid_in.u & 0x8000)
    : hybrid_in.u);
  hybrid_in.u = hybrid_in.u << 16;
  return hybrid_in.f;
}

LIBXSMM_API_INTERN libxsmm_bfloat16 my_libxsmm_convert_f32_to_bf16_truncate(float in)
{
  libxsmm_float_uint hybrid_in = { 0 };
  libxsmm_bfloat16 res;
  hybrid_in.f = in;
  /* DAZ */
  hybrid_in.u = ((hybrid_in.u & 0x7f800000) == 0x0
    ? (hybrid_in.u & 0x80000000)
    : hybrid_in.u);
  /* we do not round inf and NaN */
  hybrid_in.u = ((hybrid_in.u & 0x7f800000) == 0x7f800000
    ? ((hybrid_in.u & 0x007fffff) == 0x0 ? hybrid_in.u : (hybrid_in.u | 0x00400000))
    : hybrid_in.u);
  /* shift right */
  res = (unsigned short)(hybrid_in.u >> 16);
  return res;
}


LIBXSMM_API_INTERN libxsmm_bfloat16 my_libxsmm_convert_f32_to_bf16_rnaz(float in)
{
  libxsmm_float_uint hybrid_in = { 0 };
  libxsmm_bfloat16 res;
  hybrid_in.f = in;
  /* DAZ */
  hybrid_in.u = ((hybrid_in.u & 0x7f800000) == 0x0
    ? (hybrid_in.u & 0x80000000)
    : hybrid_in.u);
  /* we do not round inf and NaN */
  hybrid_in.u = ((hybrid_in.u & 0x7f800000) == 0x7f800000
    ? ((hybrid_in.u & 0x007fffff) == 0x0 ? hybrid_in.u : (hybrid_in.u | 0x00400000))
    : (hybrid_in.u + 0x00008000));
  /* shift right */
  res = (unsigned short)(hybrid_in.u >> 16);
  return res;
}


LIBXSMM_API_INTERN libxsmm_bfloat16 my_libxsmm_convert_f32_to_bf16_rne(float in)
{
  libxsmm_float_uint hybrid_in = { 0 };
  libxsmm_bfloat16 res;
  unsigned int fixup;
  hybrid_in.f = in;
  /* DAZ */
  hybrid_in.u = ((hybrid_in.u & 0x7f800000) == 0x0
    ? (hybrid_in.u & 0x80000000)
    : hybrid_in.u);
  /* RNE round */
  fixup = (hybrid_in.u >> 16) & 1;
  /* we do not round inf and NaN */
  hybrid_in.u = ((hybrid_in.u & 0x7f800000) == 0x7f800000
    ? ((hybrid_in.u & 0x007fffff) == 0x0 ? hybrid_in.u : (hybrid_in.u | 0x00400000))
    : (hybrid_in.u + 0x00007fff + fixup));
  /* shift right */
  res = (unsigned short)(hybrid_in.u >> 16);
  return res;
}


LIBXSMM_API_INTERN libxsmm_bfloat8 my_libxsmm_convert_f32_to_bf8_stochastic(float in, unsigned int seed)
{
  /* initial downcast */
  libxsmm_float16 f16 = my_libxsmm_convert_f32_to_f16(in);
  /* do not round inf and NaN */
  if ((f16 & 0x7c00) == 0x7c00) {
    f16 = (unsigned short)(((f16 & 0x03ff) == 0x0) ? f16 : (f16 | 0x0200));
  }
  else if ((f16 & 0x7c00) != 0x0000) { /* only round normal numbers */
#if 1
    const unsigned short stochastic = (unsigned short)LIBXSMM_MOD2(seed, 0xff + 1);
#else
    const unsigned short stochastic = (unsigned short)((seed >> 24) & 0xff);
#endif
    f16 = (unsigned short)(f16 + stochastic);
  }
  else { /* RNE for subnormal */
    const unsigned short fixup = (unsigned short)((f16 >> 8) & 1);
    f16 = (unsigned short)(f16 + 0x007f + fixup);
  }
  /* create the bf8 value by shifting out the lower 8 bits */
  return (unsigned char)(f16 >> 8);
}


LIBXSMM_API_INTERN libxsmm_bfloat8 my_libxsmm_convert_f32_to_bf8_rne(float in)
{
  libxsmm_float16_ushort hybrid_in = { 0 };
  libxsmm_bfloat8 res;
  unsigned int fixup;
  hybrid_in.f = my_libxsmm_convert_f32_to_f16(in);
  /* RNE round */
  fixup = (hybrid_in.u >> 8) & 1;
  /* we do not round inf and NaN */
  hybrid_in.u = (unsigned short)(((hybrid_in.u & 0x7c00) == 0x7c00)
    ? ((hybrid_in.u & 0x03ff) == 0x0 ? hybrid_in.u : (hybrid_in.u | 0x0200))
    : (hybrid_in.u + 0x007f + fixup));
  /* shift right */
  res = (libxsmm_bfloat8)(hybrid_in.u >> 8);
  return res;
}


LIBXSMM_API libxsmm_hfloat8 my_libxsmm_convert_f16_to_hf8_rne(libxsmm_float16 in)
{
  unsigned int f16_bias = 15;
  unsigned int f8_bias = 7;
  libxsmm_hfloat8 res = 0;
  unsigned short s, e, m, e_f16, m_f16;
  unsigned int fixup;

  s = (in & 0x8000) >> 8;
  e_f16 = (in & 0x7c00) >> 10;
  m_f16 = (in & 0x03ff);

  /* special value --> make it NaN */
  if (e_f16 == 0x1f) {
    e = 0xf;
    m = 0x7;
    /* overflow --> make it NaN */
  }
  else if ((e_f16 >  (f16_bias - f8_bias + 15)) ||
          ((e_f16 == (f16_bias - f8_bias + 15)) && (m_f16 > 0x0340)))
  {
    e = 0xf;
    m = 0x7;
    /* smaller than denormal f8 + eps */
  }
  else if (e_f16 < f16_bias - f8_bias - 3) {
    e = 0x0;
    m = 0x0;
    /* denormal */
  }
  else if (e_f16 <= f16_bias - f8_bias) {
    /* RNE */
    /* denormalized mantissa */
    m = m_f16 | 0x0400;
    /* additionally subnormal shift */
    m = m >> ((f16_bias - f8_bias) + 1 - e_f16);
    /* preserve sticky bit (some sticky bits are lost when denormalizing) */
    m |= (((m_f16 & 0x007f) + 0x007f) >> 7);
    /* RNE Round */
    fixup = (m >> 7) & 0x1;
    m = m + LIBXSMM_CAST_USHORT(0x003f + fixup);
    m = m >> 7;
    e = 0x0;
    /* normal */
  }
  else {
    /* RNE round */
    fixup = (m_f16 >> 7) & 0x1;
    in = in + LIBXSMM_CAST_USHORT(0x003f + fixup);
    e = (in & 0x7c00) >> 10;
    m = (in & 0x03ff);
    LIBXSMM_ASSERT(e >= LIBXSMM_CAST_USHORT(f16_bias - f8_bias));
    e -= LIBXSMM_CAST_USHORT(f16_bias - f8_bias);
    m = m >> 7;
  }

  /* set result to 0 */
  res = 0x0;
  /* set exponent and mantissa */
  res |= e << 3;
  res |= m;
  /* sign it */
  res |= s;

  return res;
}


LIBXSMM_API_INTERN libxsmm_hfloat8 my_libxsmm_convert_f32_to_hf8_rne(float in)
{
  const libxsmm_float16 itm = my_libxsmm_convert_f32_to_f16(in);
  return my_libxsmm_convert_f16_to_hf8_rne(itm);
}

LIBXSMM_API_INTERN
float libxsmm_fsigmoid(float x) {
  return (LIBXSMM_TANHF(x/2.0f) + 1.0f)/2.0f;
}

LIBXSMM_API_INTERN
float libxsmm_fsigmoid_inv(float x) {
  return libxsmm_fsigmoid(x) * (1.0f-libxsmm_fsigmoid(x));
}

LIBXSMM_API_INTERN
float libxsmm_tanh_inv(float x) {
  return 1.0f-LIBXSMM_TANHF(x)*LIBXSMM_TANHF(x);
}

LIBXSMM_API_INTERN
float libxsmm_gelu(float x) {
  return (LIBXSMM_ERFF(x/LIBXSMM_SQRTF(2.0f)) + 1.0f)*0.5f*x;
}

LIBXSMM_API_INTERN
float libxsmm_gelu_inv(float x) {
  return (0.5f + 0.5f * LIBXSMM_ERFF(x/LIBXSMM_SQRTF(2.0f)) + x/(LIBXSMM_SQRTF(2.0f*M_PI)) * LIBXSMM_EXPF(-0.5f*x*x) );
}

LIBXSMM_API_INTERN
void libxsmm_lsfr_Xwide( unsigned int* rng_state, float* prng_out, const unsigned int width ) {
  union { unsigned int i; float f; } rng_num = { 0 };
  const unsigned int state_ld = 16;
  const float one = 1.0f;
  unsigned int w;

  for ( w = 0 ; w < width; ++w ) {
    unsigned int state_0 = rng_state[w + (0 * state_ld)];
    unsigned int state_1 = rng_state[w + (1 * state_ld)];
    unsigned int state_2 = rng_state[w + (2 * state_ld)];
    unsigned int state_3 = rng_state[w + (3 * state_ld)];
    unsigned int tmp_0, tmp_1;
    rng_num.i = state_3 + state_0;
    rng_num.i = rng_num.i >> 9;
    rng_num.i = 0x3f800000 | rng_num.i;
    prng_out[w] = rng_num.f - one;
    tmp_0 = state_1 << 9;
    state_2 = state_2 ^ state_0;
    state_3 = state_3 ^ state_1;
    state_1 = state_1 ^ state_2;
    state_0 = state_0 ^ state_3;
    state_2 = state_2 ^ tmp_0;
    tmp_0 = state_3 << 11;
    tmp_1 = state_3 >> 21;
    state_3 = tmp_0 | tmp_1;
    rng_state[w + (0 * state_ld)] = state_0;
    rng_state[w + (1 * state_ld)] = state_1;
    rng_state[w + (2 * state_ld)] = state_2;
    rng_state[w + (3 * state_ld)] = state_3;
  }
}

LIBXSMM_API_INTERN
float libxsmm_fp32_unary_compute(float in, libxsmm_meltw_unary_type op) {
  float res = 0;
  if ( op == LIBXSMM_MELTW_TYPE_UNARY_IDENTITY || op == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR || op == LIBXSMM_MELTW_TYPE_UNARY_DUMP) {
    res = in;
  } else if ( op == LIBXSMM_MELTW_TYPE_UNARY_NEGATE) {
    res = -1.0f * in;
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_X2) {
    res = in * in;
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_XOR) {
    res = 0;
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_TANH) {
    res = LIBXSMM_TANHF(in);
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID) {
    res = libxsmm_fsigmoid(in);
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_GELU) {
    res = libxsmm_gelu(in);
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV) {
    res = libxsmm_gelu_inv(in);
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_TANH_INV) {
    res = libxsmm_tanh_inv(in);
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV) {
    res = libxsmm_fsigmoid_inv(in);
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_SQRT) {
    res = LIBXSMM_SQRTF(in);
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_INC) {
    res = in + 1.0f;
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL) {
    res = 1.0f/in;
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT) {
    res = 1.0f/LIBXSMM_SQRTF(in);
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_EXP) {
    res = LIBXSMM_EXPF(in);
  } else {
    printf("Invalid OP\n");
    exit(-1);
  }
  return res;
}

LIBXSMM_API_INTERN
double libxsmm_fp64_unary_compute(double in, libxsmm_meltw_unary_type op) {
  double res = 0;
  if ( op == LIBXSMM_MELTW_TYPE_UNARY_IDENTITY) {
    res = in;
  } else if ( op == LIBXSMM_MELTW_TYPE_UNARY_NEGATE) {
    res = -1.0 * in;
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_X2) {
    res = in * in;
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_XOR) {
    res = 0;
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_SQRT) {
    res = sqrt(in);
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_INC) {
    res = in + 1.0;
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL) {
    res = 1.0/in;
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT) {
    res = 1.0/sqrt(in);
  } else {
    printf("Invalid OP\n");
    exit(-1);
  }
  return res;
}

LIBXSMM_API_INTERN
int libxsmm_is_cmp_op(libxsmm_meltw_binary_type op) {
  int result = 0;
  if (op == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_GT || op == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_GE || op == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_LT || op == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_LE || op == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_EQ || op == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_NE) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INTERN
void libxsmm_set_bit(unsigned char *bit_matrix, libxsmm_blasint i, libxsmm_blasint j, libxsmm_blasint ld) {
  libxsmm_blasint byte_pos = i/8;
  libxsmm_blasint pos_in_byte = i%8;
  unsigned char byte_to_write = 1;
  byte_to_write = (unsigned char)(byte_to_write << pos_in_byte);
  bit_matrix[byte_pos + j * (ld/8)] |= byte_to_write;
}

LIBXSMM_API_INTERN
void libxsmm_zero_bit(unsigned char *bit_matrix, libxsmm_blasint i, libxsmm_blasint j, libxsmm_blasint ld) {
  libxsmm_blasint byte_pos = i/8;
  libxsmm_blasint pos_in_byte = i%8;
  unsigned char and_mask = 1;
  and_mask = (unsigned char)(and_mask << pos_in_byte);
  and_mask = ~and_mask;
  bit_matrix[byte_pos + j * (ld/8)] = bit_matrix[byte_pos + j * (ld/8)] & and_mask;
}

LIBXSMM_API_INTERN
unsigned char libxsmm_extract_bit(const char *bit_matrix, libxsmm_blasint i, libxsmm_blasint j, libxsmm_blasint ld) {
  unsigned char result = 0;
  libxsmm_blasint byte_load = i/8;
  libxsmm_blasint pos_in_byte = i%8;
  char byte_loaded = bit_matrix[byte_load + j * (ld/8)];
  result = ((unsigned char)(byte_loaded << (7-pos_in_byte))) >> 7;
  result = (result == 0) ? 0 : 1;
  return result;
}

LIBXSMM_API_INTERN
float libxsmm_fp32_binary_compute(float in0, float in1, float out, libxsmm_meltw_binary_type op) {
  float res = out;
  if ( op == LIBXSMM_MELTW_TYPE_BINARY_ADD) {
    res = in0 + in1;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_SUB) {
    res = in0 - in1;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_MUL) {
    res = in0 * in1;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_DIV) {
    res = in0 / in1;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_MULADD) {
    res += in0 * in1;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_MAX) {
    res = (in0 > in1) ? in0 : in1;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_MIN) {
    res = (in0 > in1) ? in1 : in0;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_GT) {
    res = (in0 > in1) ? 1 : 0;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_GE) {
    res = (in0 >= in1 ) ? 1 : 0;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_LT) {
    res = (in0 < in1) ? 1 : 0;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_LE) {
    res = (in0 <= in1) ? 1 : 0;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_EQ) {
    res = (in0 == in1) ? 1 : 0;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_NE) {
    res = (in0 != in1) ? 1 : 0;
  } else {
    printf("Invalid OP\n");
    exit(-1);
  }
  return res;
}

LIBXSMM_API_INTERN
double libxsmm_fp64_binary_compute(double in0, double in1, double out, libxsmm_meltw_binary_type op) {
  double res = out;
  if ( op == LIBXSMM_MELTW_TYPE_BINARY_ADD) {
    res = in0 + in1;
  } else  if ( op == LIBXSMM_MELTW_TYPE_BINARY_SUB) {
    res = in0 - in1;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_MUL) {
    res = in0 * in1;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_DIV) {
    res = in0 / in1;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_MULADD) {
    res += in0 * in1;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_MAX) {
    res = (in0 > in1) ? in0 : in1;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_MIN) {
    res = (in0 > in1) ? in1 : in0;
  } else {
    printf("Invalid OP\n");
    exit(-1);
  }
  return res;
}

LIBXSMM_API_INTERN
libxsmm_blasint libxsmm_elementwise_get_index(libxsmm_blasint i, libxsmm_blasint j, libxsmm_blasint ld, const libxsmm_meltw_descriptor *i_mateltwise_desc, unsigned int operand_id) {
  libxsmm_blasint result = 0;
  unsigned int bcast_row = (((operand_id == 0) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW) > 0)) ||
                            ((operand_id == 0) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0) > 0)) ||
                            ((operand_id == 0) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_0) > 0)) ||
                            ((operand_id == 1) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1) > 0)) ||
                            ((operand_id == 1) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_1) > 0)) ||
                            ((operand_id == 2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_2) > 0))) ? 1 : 0;
  unsigned int bcast_col = (((operand_id == 0) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) > 0)) ||
                            ((operand_id == 0) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL) > 0)) ||
                            ((operand_id == 0) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0) > 0)) ||
                            ((operand_id == 0) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0) > 0)) ||
                            ((operand_id == 1) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1) > 0)) ||
                            ((operand_id == 1) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1) > 0)) ||
                            ((operand_id == 2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2) > 0))) ? 1 : 0;
  unsigned int bcast_sca = (((operand_id == 0) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR) > 0)) ||
                            ((operand_id == 0) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0) > 0)) ||
                            ((operand_id == 0) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_0) > 0)) ||
                            ((operand_id == 1) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1) > 0)) ||
                            ((operand_id == 1) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1) > 0)) ||
                            ((operand_id == 2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2) > 0))) ? 1 : 0;
  if (bcast_row > 0) {
    result = j * ld;
  } else if (bcast_col > 0) {
    result = i;
  } else if (bcast_sca > 0) {
    result = 0;
  } else {
    result = i + j * ld;
  }
  return result;
}

LIBXSMM_API_INTERN
float libxsmm_elementwise_get_float_value(const void *in, libxsmm_blasint i, libxsmm_blasint j, libxsmm_blasint ld, libxsmm_datatype dtype_in, const libxsmm_meltw_descriptor *i_mateltwise_desc, unsigned int operand_id) {
  float result = 0.0;
  if ( dtype_in == LIBXSMM_DATATYPE_F32 ) {
    const float* f_in = (const float*)in;
    result = f_in[libxsmm_elementwise_get_index(i, j, ld, i_mateltwise_desc, operand_id)];
  } else if ( dtype_in == LIBXSMM_DATATYPE_BF16 ) {
    const libxsmm_bfloat16* bf16_in = (const libxsmm_bfloat16*)in;
    result = my_libxsmm_convert_bf16_to_f32(bf16_in[libxsmm_elementwise_get_index(i, j, ld, i_mateltwise_desc, operand_id)]);
  } else if ( dtype_in == LIBXSMM_DATATYPE_F16 ) {
    const libxsmm_float16* f16_in = (const libxsmm_float16*)in;
    result = my_libxsmm_convert_f16_to_f32(f16_in[libxsmm_elementwise_get_index(i, j, ld, i_mateltwise_desc, operand_id)]);
  } else if ( dtype_in == LIBXSMM_DATATYPE_BF8 ) {
    const libxsmm_bfloat8* bf8_in = (const libxsmm_bfloat8*)in;
    result = my_libxsmm_convert_bf8_to_f32(bf8_in[libxsmm_elementwise_get_index(i, j, ld, i_mateltwise_desc, operand_id)]);
  } else if ( dtype_in == LIBXSMM_DATATYPE_HF8 ) {
    const libxsmm_hfloat8* hf8_in = (const libxsmm_hfloat8*)in;
    result = my_libxsmm_convert_hf8_to_f32(hf8_in[libxsmm_elementwise_get_index(i, j, ld, i_mateltwise_desc, operand_id)]);
  } else {
    /* should not happen */
  }
  return result;
}

LIBXSMM_API_INTERN
void libxsmm_elementwise_store_value(void *out, void* out_value_ptr, libxsmm_blasint i, libxsmm_blasint j, libxsmm_blasint ldo, unsigned int use_stoch, libxsmm_datatype dtype_out, void *rng_state, unsigned int seed_idx) {
  float out_value = *((float*)out_value_ptr);
  if ( dtype_out == LIBXSMM_DATATYPE_F32 ) {
    float* f_out = (float*)out;
    f_out[(j*ldo) + i] = out_value;
  } else if ( dtype_out == LIBXSMM_DATATYPE_BF16 ) {
    libxsmm_bfloat16* bf16_out = (libxsmm_bfloat16*)out;
    bf16_out[(j*ldo) + i] = my_libxsmm_convert_f32_to_bf16_rne(out_value);
  } else if ( dtype_out == LIBXSMM_DATATYPE_F16 ) {
    libxsmm_float16* f16_out = (libxsmm_float16*)out;
    f16_out[(j*ldo) + i] = my_libxsmm_convert_f32_to_f16(out_value);
  } else if ( dtype_out == LIBXSMM_DATATYPE_BF8 ) {
    libxsmm_bfloat8* bf8_out = (libxsmm_bfloat8*)out;
    if (use_stoch > 0) {
      my_libxsmm_stochastic_convert_fp32_bf8(&out_value, &(bf8_out[(j*ldo) + i]), 1, rng_state, seed_idx);
    } else {
      bf8_out[(j*ldo) + i] = my_libxsmm_convert_f32_to_bf8_rne(out_value);
    }
  } else if ( dtype_out == LIBXSMM_DATATYPE_HF8 ) {
    libxsmm_hfloat8* hf8_out = (libxsmm_hfloat8*)out;
    hf8_out[(j*ldo) + i] = my_libxsmm_convert_f32_to_hf8_rne(out_value);
  } else {
    /* should not happen */
  }
  return;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_is_reduce_op(const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  unsigned int result = 0;
  if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X2_OP_ADD ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MIN ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_ADD ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_MAX ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_MIN ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ABSMAX  ) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_is_transform_op(const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  unsigned int result = 0;
  if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2_TO_VNNI2T ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI4T ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI8_TO_VNNI8T ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2 ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2T ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI8_TO_NORM ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI8T_TO_NORM ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4T_TO_NORM ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2T_TO_NORM ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4T ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8T ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4_PAD ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4 ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8_PAD ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8 ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_NORM ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI2 ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD2 ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD2 ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2 ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD4 ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD4 ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD4 ) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INTERN
void libxsmm_ref_transpose(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j;
  const void *in = (const void*)param->in.primary;
  void* out = (void*)param->out.primary;
  const libxsmm_blasint M = i_mateltwise_desc->m;
  const libxsmm_blasint N = i_mateltwise_desc->n;
  const libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  const libxsmm_blasint ldo = i_mateltwise_desc->ldo;
  const libxsmm_datatype dtype = (libxsmm_datatype)libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0);

  if ( (dtype == LIBXSMM_DATATYPE_F64) || (dtype == LIBXSMM_DATATYPE_I64) ) {
    const double*  in_data = (const double*)in;
    double*       out_data = (double*)      out;
    for ( i = 0; i < (libxsmm_blasint)N; ++i ) {
      for ( j = 0; j < (libxsmm_blasint)M; ++j ) {
        out_data[(j*(libxsmm_blasint)ldo)+i] = in_data[(i*(libxsmm_blasint)ldi)+j];
      }
    }
  } else if ( (dtype == LIBXSMM_DATATYPE_F32) || (dtype == LIBXSMM_DATATYPE_I32)) {
    const float*  in_data = (const float*)in;
    float*       out_data = (float*)      out;
    for ( i = 0; i < (libxsmm_blasint)N; ++i ) {
      for ( j = 0; j < (libxsmm_blasint)M; ++j ) {
        out_data[(j*(libxsmm_blasint)ldo)+i] = in_data[(i*(libxsmm_blasint)ldi)+j];
      }
    }
  } else if ( (dtype == LIBXSMM_DATATYPE_BF16) || (dtype == LIBXSMM_DATATYPE_F16) || (dtype == LIBXSMM_DATATYPE_I16) ) {
    const unsigned short*  in_data = (const unsigned short*)in;
    unsigned short*       out_data = (unsigned short*)      out;
    for ( i = 0; i < (libxsmm_blasint)N; ++i ) {
      for ( j = 0; j < (libxsmm_blasint)M; ++j ) {
        out_data[(j*(libxsmm_blasint)ldo)+i] = in_data[(i*(libxsmm_blasint)ldi)+j];
      }
    }
  } else if ( (dtype == LIBXSMM_DATATYPE_I8) || (dtype == LIBXSMM_DATATYPE_BF8) || (dtype == LIBXSMM_DATATYPE_HF8) ) {
    const unsigned char*  in_data = (const unsigned char*)in;
    unsigned char*       out_data = (unsigned char*)      out;
    for ( i = 0; i < (libxsmm_blasint)N; ++i ) {
      for ( j = 0; j < (libxsmm_blasint)M; ++j ) {
        out_data[(j*(libxsmm_blasint)ldo)+i] = in_data[(i*(libxsmm_blasint)ldi)+j];
      }
    }
  } else {
    /* should not happen */
  }
}

LIBXSMM_API_INTERN
void libxsmm_ref_vnni2_to_vnni2T_16bit(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j, j2, i2;
  unsigned short *in = (unsigned short*)param->in.primary;
  unsigned short *out = (unsigned short*)param->out.primary;
  const libxsmm_blasint M = i_mateltwise_desc->m;
  const libxsmm_blasint N = i_mateltwise_desc->n;
  const libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  const libxsmm_blasint ldo = i_mateltwise_desc->ldo;
  for ( j = 0; j < M/2; ++j ) {
    for ( i = 0; i < N/2 ; ++i ) {
      for ( j2 = 0; j2 < 2; ++j2 ) {
        for ( i2 = 0; i2 < 2; ++i2 ) {
          out[j*ldo*2+j2+(i*2+i2)*2] = in[i*ldi*2+i2+(j*2+j2)*2];
        }
      }
    }
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_vnni4_to_vnni4T_16bit(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j, j2, i2;
  unsigned short *in = (unsigned short*)param->in.primary;
  unsigned short *out = (unsigned short*)param->out.primary;
  const libxsmm_blasint M = i_mateltwise_desc->m;
  const libxsmm_blasint N = i_mateltwise_desc->n;
  const libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  const libxsmm_blasint ldo = i_mateltwise_desc->ldo;
  /* to vnni */
  for ( j = 0; j < M/4; ++j ) {
    for ( i = 0; i < N/4 ; ++i ) {
      for ( j2 = 0; j2 < 4; ++j2 ) {
        for ( i2 = 0; i2 < 4; ++i2 ) {
          out[j*ldo*4+j2+(i*4+i2)*4] = in[i*ldi*4+i2+(j*4+j2)*4];
        }
      }
    }
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_vnni4_to_vnni4T_08bit(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j, j2, i2;
  unsigned char *in = (unsigned char*)param->in.primary;
  unsigned char *out = (unsigned char*)param->out.primary;
  const libxsmm_blasint M = i_mateltwise_desc->m;
  const libxsmm_blasint N = i_mateltwise_desc->n;
  const libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  const libxsmm_blasint ldo = i_mateltwise_desc->ldo;
  /* to vnni */
  for ( j = 0; j < M/4; ++j ) {
    for ( i = 0; i < N/4 ; ++i ) {
      for ( j2 = 0; j2 < 4; ++j2 ) {
        for ( i2 = 0; i2 < 4; ++i2 ) {
          out[j*ldo*4+j2+(i*4+i2)*4] = in[i*ldi*4+i2+(j*4+j2)*4];
        }
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_ref_vnni8_to_vnni8T_16bit(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j, j2, i2;
  unsigned short *in = (unsigned short*)param->in.primary;
  unsigned short *out = (unsigned short*)param->out.primary;
  const libxsmm_blasint M = i_mateltwise_desc->m;
  const libxsmm_blasint N = i_mateltwise_desc->n;
  const libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  const libxsmm_blasint ldo = i_mateltwise_desc->ldo;
  /* to vnni */
  for ( j = 0; j < M/8; ++j ) {
    for ( i = 0; i < N/8 ; ++i ) {
      for ( j2 = 0; j2 < 8; ++j2 ) {
        for ( i2 = 0; i2 < 8; ++i2 ) {
          out[j*ldo*8+j2+(i*8+i2)*8] = in[i*ldi*8+i2+(j*8+j2)*8];
        }
      }
    }
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_vnni8_to_vnni8T_08bit(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j, j2, i2;
  unsigned char *in = (unsigned char*)param->in.primary;
  unsigned char *out = (unsigned char*)param->out.primary;
  const libxsmm_blasint M = i_mateltwise_desc->m;
  const libxsmm_blasint N = i_mateltwise_desc->n;
  const libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  const libxsmm_blasint ldo = i_mateltwise_desc->ldo;
  /* to vnni */
  for ( j = 0; j < M/8; ++j ) {
    for ( i = 0; i < N/8 ; ++i ) {
      for ( j2 = 0; j2 < 8; ++j2 ) {
        for ( i2 = 0; i2 < 8; ++i2 ) {
          out[j*ldo*8+j2+(i*8+i2)*8] = in[i*ldi*8+i2+(j*8+j2)*8];
        }
      }
    }
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_norm_to_vnni2_16bit(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j, j2;
  unsigned short *in = (unsigned short*)param->in.primary;
  unsigned short *out = (unsigned short*)param->out.primary;
  const libxsmm_blasint M = i_mateltwise_desc->m;
  const libxsmm_blasint N = i_mateltwise_desc->n;
  const libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  const libxsmm_blasint ldo = i_mateltwise_desc->ldo;
  libxsmm_blasint Nn = N + (N%2);

  for ( i = 0; i < ldo*Nn; ++i ) {
    out[i] = 0;
  }
  /* to vnni */
  for ( j = 0; j < Nn/2; ++j ) {
    for ( i = 0; i < M ; ++i ) {
      for ( j2 = 0; j2 < 2; ++j2 ) {
        out[(j*ldo*2)+(i*2)+j2] = in[(((j*2)+j2)*ldi)+i];
      }
    }
  }

  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_norm_to_vnni2T_16bit(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j, i2;
  unsigned short *in = (unsigned short*)param->in.primary;
  unsigned short *out = (unsigned short*)param->out.primary;
  const libxsmm_blasint M = i_mateltwise_desc->m;
  const libxsmm_blasint N = i_mateltwise_desc->n;
  const libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  const libxsmm_blasint ldo = i_mateltwise_desc->ldo;

  /* to vnni */
  for ( i = 0; i < M/2; ++i ) {
    for ( j = 0; j < N ; ++j ) {
      for ( i2 = 0; i2 < 2; ++i2 ) {
        out[(i*ldo*2)+(j*2)+i2] = in[(j*ldi)+(i*2+i2)];
      }
    }
  }

  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_vnni8T_to_norm_16bit(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j, i2;
  unsigned short *in = (unsigned short*)param->in.primary;
  unsigned short *out = (unsigned short*)param->out.primary;
  const libxsmm_blasint M = i_mateltwise_desc->n;
  const libxsmm_blasint N = i_mateltwise_desc->m;
  const libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  const libxsmm_blasint ldo = i_mateltwise_desc->ldo;
  /* to vnni */
  for ( i = 0; i < M/8; ++i ) {
    for ( j = 0; j < N ; ++j ) {
      for ( i2 = 0; i2 < 8; ++i2 ) {
        out[(j*ldo)+(i*8)+i2] = in[(i*ldi*8)+(j*8+i2)];
      }
    }
  }

  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_vnni4T_to_norm_16bit(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j, i2;
  unsigned short *in = (unsigned short*)param->in.primary;
  unsigned short *out = (unsigned short*)param->out.primary;
  const libxsmm_blasint M = i_mateltwise_desc->n;
  const libxsmm_blasint N = i_mateltwise_desc->m;
  const libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  const libxsmm_blasint ldo = i_mateltwise_desc->ldo;

  /* to vnni */
  for ( i = 0; i < M/4; ++i ) {
    for ( j = 0; j < N ; ++j ) {
      for ( i2 = 0; i2 < 4; ++i2 ) {
        out[(j*ldo)+(i*4)+i2] = in[(i*ldi*4)+(j*4+i2)];
      }
    }
  }

  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_vnni2T_to_norm_16bit(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j, i2;
  unsigned short *in = (unsigned short*)param->in.primary;
  unsigned short *out = (unsigned short*)param->out.primary;
  const libxsmm_blasint M = i_mateltwise_desc->n;
  const libxsmm_blasint N = i_mateltwise_desc->m;
  const libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  const libxsmm_blasint ldo = i_mateltwise_desc->ldo;

  /* to vnni */
  for ( i = 0; i < M/2; ++i ) {
    for ( j = 0; j < N ; ++j ) {
      for ( i2 = 0; i2 < 2; ++i2 ) {
        out[(j*ldo)+(i*2)+i2] = in[(i*ldi*2)+(j*2+i2)];
      }
    }
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_norm_to_vnni4T_16bit(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j, i2;
  unsigned short *in = (unsigned short*)param->in.primary;
  unsigned short *out = (unsigned short*)param->out.primary;
  const libxsmm_blasint M = i_mateltwise_desc->m;
  const libxsmm_blasint N = i_mateltwise_desc->n;
  const libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  const libxsmm_blasint ldo = i_mateltwise_desc->ldo;

  /* to vnni */
  for ( i = 0; i < M/4; ++i ) {
    for ( j = 0; j < N ; ++j ) {
      for ( i2 = 0; i2 < 4; ++i2 ) {
        out[(i*ldo*4)+(j*4)+i2] = in[(j*ldi)+(i*4+i2)];
      }
    }
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_norm_to_vnni8T_16bit(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j, i2;
  unsigned short *in = (unsigned short*)param->in.primary;
  unsigned short *out = (unsigned short*)param->out.primary;
  const libxsmm_blasint M = i_mateltwise_desc->m;
  const libxsmm_blasint N = i_mateltwise_desc->n;
  const libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  const libxsmm_blasint ldo = i_mateltwise_desc->ldo;

  /* to vnni */
  for ( i = 0; i < M/8; ++i ) {
    for ( j = 0; j < N ; ++j ) {
      for ( i2 = 0; i2 < 8; ++i2 ) {
        out[(i*ldo*8)+(j*8)+i2] = in[(j*ldi)+(i*8+i2)];
      }
    }
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_norm_to_vnni4_16bit(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j, j2;
  unsigned short *in = (unsigned short*)param->in.primary;
  unsigned short *out = (unsigned short*)param->out.primary;
  const libxsmm_blasint M = i_mateltwise_desc->m;
  const libxsmm_blasint N = i_mateltwise_desc->n;
  const libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  const libxsmm_blasint ldo = i_mateltwise_desc->ldo;
  libxsmm_blasint Nn = ((N%4) == 0) ? N : LIBXSMM_UP(N, 4);

  for ( i = 0; i < ldo*Nn; ++i ) {
    out[i] = 0;
  }
  /* to vnni */
  for ( j = 0; j < Nn/4; ++j ) {
    for ( i = 0; i < M ; ++i ) {
      for ( j2 = 0; j2 < 4; ++j2 ) {
        out[(j*ldo*4)+(i*4)+j2] = in[(((j*4)+j2)*ldi)+i];
      }
    }
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_norm_to_vnni8_16bit(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j, j2;
  unsigned short *in = (unsigned short*)param->in.primary;
  unsigned short *out = (unsigned short*)param->out.primary;
  const libxsmm_blasint M = i_mateltwise_desc->m;
  const libxsmm_blasint N = i_mateltwise_desc->n;
  const libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  const libxsmm_blasint ldo = i_mateltwise_desc->ldo;
  libxsmm_blasint Nn = ((N%8) == 0) ? N : LIBXSMM_UP(N, 8);

  for ( i = 0; i < ldo*Nn; ++i ) {
    out[i] = 0;
  }
  /* to vnni */
  for ( j = 0; j < Nn/8; ++j ) {
    for ( i = 0; i < M ; ++i ) {
      for ( j2 = 0; j2 < 8; ++j2 ) {
        out[(j*ldo*8)+(i*8)+j2] = in[(((j*8)+j2)*ldi)+i];
      }
    }
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_norm_to_vnni4_08bit(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j, j2;
  unsigned char *in = (unsigned char*)param->in.primary;
  unsigned char *out = (unsigned char*)param->out.primary;
  const libxsmm_blasint M = i_mateltwise_desc->m;
  const libxsmm_blasint N = i_mateltwise_desc->n;
  const libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  const libxsmm_blasint ldo = i_mateltwise_desc->ldo;
  libxsmm_blasint Nn = ((N%4) == 0) ? N : LIBXSMM_UP(N, 4);

  for ( i = 0; i < ldo*Nn; ++i ) {
    out[i] = 0;
  }
  /* to vnni */
  for ( j = 0; j < Nn/4; ++j ) {
    for ( i = 0; i < M ; ++i ) {
      for ( j2 = 0; j2 < 4; ++j2 ) {
        out[(j*ldo*4)+(i*4)+j2] = in[(((j*4)+j2)*ldi)+i];
      }
    }
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_norm_to_vnni8_08bit(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j, j2;
  unsigned char *in = (unsigned char*)param->in.primary;
  unsigned char *out = (unsigned char*)param->out.primary;
  const libxsmm_blasint M = i_mateltwise_desc->m;
  const libxsmm_blasint N = i_mateltwise_desc->n;
  const libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  const libxsmm_blasint ldo = i_mateltwise_desc->ldo;
  libxsmm_blasint Nn = ((N%8) == 0) ? N : LIBXSMM_UP(N, 8);

  for ( i = 0; i < ldo*Nn; ++i ) {
    out[i] = 0;
  }
  /* to vnni */
  for ( j = 0; j < Nn/8; ++j ) {
    for ( i = 0; i < M ; ++i ) {
      for ( j2 = 0; j2 < 8; ++j2 ) {
        out[(j*ldo*8)+(i*8)+j2] = in[(((j*8)+j2)*ldi)+i];
      }
    }
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_vnni4_to_norm_08bit(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j;
  unsigned char *in = (unsigned char*)param->in.primary;
  unsigned char *out = (unsigned char*)param->out.primary;
  const libxsmm_blasint M = i_mateltwise_desc->m;
  const libxsmm_blasint N = i_mateltwise_desc->n;
  const libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  const libxsmm_blasint ldo = i_mateltwise_desc->ldo;

  /* to vnni */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      out[(i*ldo)+j] = in[((i/4)*ldi*4)+j*4+(i%4)];
    }
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_vnni4_to_vnni2_08bit(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j;
  unsigned char *in = (unsigned char*)param->in.primary;
  unsigned char *out = (unsigned char*)param->out.primary;
  const libxsmm_blasint M = i_mateltwise_desc->m;
  const libxsmm_blasint N = i_mateltwise_desc->n;
  const libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  const libxsmm_blasint ldo = i_mateltwise_desc->ldo;

  /* to vnni */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      out[((i/2)*ldo*2)+j*2+(i%2)] = in[((i/4)*ldi*4)+j*4+(i%4)];
    }
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_norm_padn_mod2_16bit(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j;
  unsigned short *in = (unsigned short*)param->in.primary;
  unsigned short *out = (unsigned short*)param->out.primary;
  const libxsmm_blasint M = i_mateltwise_desc->m;
  const libxsmm_blasint N = i_mateltwise_desc->n;
  const libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  const libxsmm_blasint ldo = i_mateltwise_desc->ldo;
  libxsmm_blasint Nn = ((N%2) == 0) ? N : N+1;

  for ( i = 0; i < ldo*Nn; ++i ) {
    out[i] = 0;
  }
  /* to vnni */
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M ; ++i ) {
      out[(j*ldo)+i] = in[(j*ldi)+i];
    }
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_norm_padm_mod2_16bit(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j;
  unsigned short *in = (unsigned short*)param->in.primary;
  unsigned short *out = (unsigned short*)param->out.primary;
  const libxsmm_blasint M = i_mateltwise_desc->m;
  const libxsmm_blasint N = i_mateltwise_desc->n;
  const libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  const libxsmm_blasint ldo = i_mateltwise_desc->ldo;

  for ( i = 0; i < ldo*N; ++i ) {
    out[i] = 0;
  }

  /* to vnni */
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M ; ++i ) {
      out[(j*ldo)+i] = in[(j*ldi)+i];
    }
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_norm_padnm_mod2_16bit(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j;
  unsigned short *in = (unsigned short*)param->in.primary;
  unsigned short *out = (unsigned short*)param->out.primary;
  const libxsmm_blasint M = i_mateltwise_desc->m;
  const libxsmm_blasint N = i_mateltwise_desc->n;
  const libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  const libxsmm_blasint ldo = i_mateltwise_desc->ldo;
  libxsmm_blasint Nn = ((N%2) == 0) ? N : N+1;

  for ( i = 0; i < ldo*Nn; ++i ) {
    out[i] = 0;
  }

  /* to vnni */
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M ; ++i ) {
      out[(j*ldo)+i] = in[(j*ldi)+i];
    }
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_norm_padn_mod4_08bit(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j;
  unsigned char *in = (unsigned char*)param->in.primary;
  unsigned char *out = (unsigned char*)param->out.primary;
  const libxsmm_blasint M = i_mateltwise_desc->m;
  const libxsmm_blasint N = i_mateltwise_desc->n;
  const libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  const libxsmm_blasint ldo = i_mateltwise_desc->ldo;
  libxsmm_blasint Nn = ((N%4) == 0) ? N : LIBXSMM_UP(N, 4);

  for ( i = 0; i < ldo*Nn; ++i ) {
    out[i] = 0;
  }

  /* to vnni */
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M ; ++i ) {
      out[(j*ldo)+i] = in[(j*ldi)+i];
    }
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_norm_padm_mod4_08bit(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j;
  unsigned char *in = (unsigned char*)param->in.primary;
  unsigned char *out = (unsigned char*)param->out.primary;
  const libxsmm_blasint M = i_mateltwise_desc->m;
  const libxsmm_blasint N = i_mateltwise_desc->n;
  const libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  const libxsmm_blasint ldo = i_mateltwise_desc->ldo;

  for ( i = 0; i < ldo*N; ++i ) {
    out[i] = 0;
  }

  /* to vnni */
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M ; ++i ) {
      out[(j*ldo)+i] = in[(j*ldi)+i];
    }
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_norm_padnm_mod4_08bit(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j;
  unsigned char *in = (unsigned char*)param->in.primary;
  unsigned char *out = (unsigned char*)param->out.primary;
  const libxsmm_blasint M = i_mateltwise_desc->m;
  const libxsmm_blasint N = i_mateltwise_desc->n;
  const libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  const libxsmm_blasint ldo = i_mateltwise_desc->ldo;
  libxsmm_blasint Nn = ((N%4) == 0) ? N : LIBXSMM_UP(N, 4);

  for ( i = 0; i < ldo*Nn; ++i ) {
    out[i] = 0;
  }

  /* to vnni */
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M ; ++i ) {
      out[(j*ldo)+i] = in[(j*ldi)+i];
    }
  }

  return;
}

LIBXSMM_API_INTERN
void libxsmm_elementwise_transform_kernel(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_datatype dtype = (libxsmm_datatype)libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0);
  if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT ) {
    libxsmm_ref_transpose(param, i_mateltwise_desc);
  } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2_TO_VNNI2T &&
      ( dtype == LIBXSMM_DATATYPE_BF16 || dtype == LIBXSMM_DATATYPE_F16 || dtype == LIBXSMM_DATATYPE_I16 ) ) {
    libxsmm_ref_vnni2_to_vnni2T_16bit(param, i_mateltwise_desc);
  } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI4T &&
      ( dtype == LIBXSMM_DATATYPE_I8 || dtype == LIBXSMM_DATATYPE_BF8 || dtype == LIBXSMM_DATATYPE_HF8 ) ) {
    libxsmm_ref_vnni4_to_vnni4T_08bit(param, i_mateltwise_desc);
  } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI8_TO_VNNI8T &&
      ( dtype == LIBXSMM_DATATYPE_I16 || dtype == LIBXSMM_DATATYPE_BF16 || dtype == LIBXSMM_DATATYPE_F16 ) ) {
    libxsmm_ref_vnni8_to_vnni8T_16bit(param, i_mateltwise_desc);
  } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI8_TO_VNNI8T &&
      ( dtype == LIBXSMM_DATATYPE_I8 || dtype == LIBXSMM_DATATYPE_BF8 || dtype == LIBXSMM_DATATYPE_HF8 ) ) {
    libxsmm_ref_vnni8_to_vnni8T_08bit(param, i_mateltwise_desc);
  } else if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD ||
               i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2) &&
      ( dtype == LIBXSMM_DATATYPE_I16 || dtype == LIBXSMM_DATATYPE_BF16 || dtype == LIBXSMM_DATATYPE_F16 ) ) {
    libxsmm_ref_norm_to_vnni2_16bit(param, i_mateltwise_desc);
  }   else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2T &&
       ( dtype == LIBXSMM_DATATYPE_BF16 || dtype == LIBXSMM_DATATYPE_F16 || dtype == LIBXSMM_DATATYPE_I16 ) ) {
    libxsmm_ref_norm_to_vnni2T_16bit(param, i_mateltwise_desc);
  } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI4T &&
      ( dtype == LIBXSMM_DATATYPE_BF16 || dtype == LIBXSMM_DATATYPE_F16 || dtype == LIBXSMM_DATATYPE_I16) ) {
    libxsmm_ref_vnni4_to_vnni4T_16bit(param, i_mateltwise_desc);
  } else if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD ||
               i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2) &&
      ( dtype == LIBXSMM_DATATYPE_I16 || dtype == LIBXSMM_DATATYPE_BF16 || dtype == LIBXSMM_DATATYPE_F16 ) ) {
    libxsmm_ref_norm_to_vnni2_16bit(param, i_mateltwise_desc);
  } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2T &&
      ( dtype == LIBXSMM_DATATYPE_I16 || dtype == LIBXSMM_DATATYPE_BF16 || dtype == LIBXSMM_DATATYPE_F16 ) ) {
    libxsmm_ref_norm_to_vnni2T_16bit(param, i_mateltwise_desc);
  } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI8T_TO_NORM &&
      ( dtype == LIBXSMM_DATATYPE_I16 || dtype == LIBXSMM_DATATYPE_BF16 || dtype == LIBXSMM_DATATYPE_F16 ) ) {
    libxsmm_ref_vnni8T_to_norm_16bit(param, i_mateltwise_desc);
  } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4T_TO_NORM &&
      ( dtype == LIBXSMM_DATATYPE_I16 || dtype == LIBXSMM_DATATYPE_BF16 || dtype == LIBXSMM_DATATYPE_F16 ) ) {
    libxsmm_ref_vnni4T_to_norm_16bit(param, i_mateltwise_desc);
  } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2T_TO_NORM &&
      ( dtype == LIBXSMM_DATATYPE_I16 || dtype == LIBXSMM_DATATYPE_BF16 || dtype == LIBXSMM_DATATYPE_F16 ) ) {
    libxsmm_ref_vnni2T_to_norm_16bit(param, i_mateltwise_desc);
  } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4T &&
      ( dtype == LIBXSMM_DATATYPE_I16 || dtype == LIBXSMM_DATATYPE_BF16 || dtype == LIBXSMM_DATATYPE_F16 ) ) {
    libxsmm_ref_norm_to_vnni4T_16bit(param, i_mateltwise_desc);
  } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8T &&
      ( dtype == LIBXSMM_DATATYPE_I16 || dtype == LIBXSMM_DATATYPE_BF16 || dtype == LIBXSMM_DATATYPE_F16 ) ) {
    libxsmm_ref_norm_to_vnni8T_16bit(param, i_mateltwise_desc);
  } else if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4_PAD ||
               i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4)&&
      ( dtype == LIBXSMM_DATATYPE_I8 || dtype == LIBXSMM_DATATYPE_BF8 || dtype == LIBXSMM_DATATYPE_HF8 ) ) {
    libxsmm_ref_norm_to_vnni4_08bit(param, i_mateltwise_desc);
  } else if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8_PAD ||
               i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8) &&
      ( dtype == LIBXSMM_DATATYPE_I8 || dtype == LIBXSMM_DATATYPE_BF8 || dtype == LIBXSMM_DATATYPE_HF8 ) ) {
    libxsmm_ref_norm_to_vnni8_08bit(param, i_mateltwise_desc);
  } else if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4_PAD ||
                i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4) &&
      ( dtype == LIBXSMM_DATATYPE_BF16 || dtype == LIBXSMM_DATATYPE_F16 || dtype == LIBXSMM_DATATYPE_I16 ) ) {
    libxsmm_ref_norm_to_vnni4_16bit(param, i_mateltwise_desc);
  } else if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8_PAD ||
               i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8) &&
      ( dtype == LIBXSMM_DATATYPE_BF16 || dtype == LIBXSMM_DATATYPE_F16 || dtype == LIBXSMM_DATATYPE_I16 ) ) {
    libxsmm_ref_norm_to_vnni8_16bit(param, i_mateltwise_desc);
  } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI4T &&
      ( dtype == LIBXSMM_DATATYPE_BF16 || dtype == LIBXSMM_DATATYPE_F16 || dtype == LIBXSMM_DATATYPE_I16 ) ) {
    libxsmm_ref_vnni4_to_vnni4T_16bit(param, i_mateltwise_desc);
  } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI2 &&
      ( dtype == LIBXSMM_DATATYPE_I8 || dtype == LIBXSMM_DATATYPE_BF8 || dtype == LIBXSMM_DATATYPE_HF8 ) ) {
    libxsmm_ref_vnni4_to_vnni2_08bit(param, i_mateltwise_desc);
  } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD2 &&
      ( dtype == LIBXSMM_DATATYPE_BF16 || dtype == LIBXSMM_DATATYPE_F16 || dtype == LIBXSMM_DATATYPE_I16 ) ) {
    libxsmm_ref_norm_padn_mod2_16bit(param, i_mateltwise_desc);
  } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD2 &&
      ( dtype == LIBXSMM_DATATYPE_BF16 || dtype == LIBXSMM_DATATYPE_F16 || dtype == LIBXSMM_DATATYPE_I16 ) ) {
    libxsmm_ref_norm_padm_mod2_16bit(param, i_mateltwise_desc);
  } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2 &&
    ( dtype == LIBXSMM_DATATYPE_BF16 || dtype == LIBXSMM_DATATYPE_F16 || dtype == LIBXSMM_DATATYPE_I16 ) ) {
    libxsmm_ref_norm_padnm_mod2_16bit(param, i_mateltwise_desc);
  } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD4 &&
      ( dtype == LIBXSMM_DATATYPE_I8 || dtype == LIBXSMM_DATATYPE_BF8 || dtype == LIBXSMM_DATATYPE_HF8 ) ) {
    libxsmm_ref_norm_padn_mod4_08bit(param, i_mateltwise_desc);
  } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD4 &&
      ( dtype == LIBXSMM_DATATYPE_I8 || dtype == LIBXSMM_DATATYPE_BF8 || dtype == LIBXSMM_DATATYPE_HF8 ) ) {
    libxsmm_ref_norm_padm_mod4_08bit(param, i_mateltwise_desc);
  } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD4 &&
      ( dtype == LIBXSMM_DATATYPE_I8 || dtype == LIBXSMM_DATATYPE_BF8 || dtype == LIBXSMM_DATATYPE_HF8 ) ) {
    libxsmm_ref_norm_padnm_mod4_08bit(param, i_mateltwise_desc);
  } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_NORM &&
      ( dtype == LIBXSMM_DATATYPE_I8 || dtype == LIBXSMM_DATATYPE_BF8 || dtype == LIBXSMM_DATATYPE_HF8 ) ) {
    libxsmm_ref_vnni4_to_norm_08bit(param, i_mateltwise_desc);
  } else {
    /* Should not happen  */
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_elementwise_reduce_kernel(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint m = i_mateltwise_desc->m;
  libxsmm_blasint n = i_mateltwise_desc->n;
  libxsmm_blasint ld_in = i_mateltwise_desc->ldi;
  libxsmm_blasint ld_out = i_mateltwise_desc->ldo;
  libxsmm_datatype dtype_in = (libxsmm_datatype)libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0);
  libxsmm_datatype dtype_out = (libxsmm_datatype)libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT);
  unsigned int reduce_rows = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS) > 0) ? 1 : 0;
  int result_size = (reduce_rows == 1) ? n : ld_out;
  void *in = (void*)param->in.primary;
  void *result_reduce_elts = (void*)param->out.primary;
  void *result_reduce_elts_sq = (char*)result_reduce_elts + result_size * LIBXSMM_TYPESIZE(dtype_out);
  void *cols_ind_array = (void*)param->in.secondary;
  void *ref_argop_off = (void*)param->out.secondary;
  libxsmm_blasint i = 0, j = 0, jj = 0, n_cols = 0;
  unsigned int *col_idx_32bit = (unsigned int*) cols_ind_array;
  unsigned long long *col_idx_64bit = (unsigned long long*) cols_ind_array;
  unsigned int *argop_idx_32bit = (unsigned int*) ref_argop_off;
  unsigned long long *argop_idx_64bit = (unsigned long long*) ref_argop_off;
  unsigned int reduce_on_output = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_INIT_ACC) > 0) ? 1 : 0 ;
  unsigned int reduce_op = 0;
  unsigned int record_idx = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_RECORD_ARGOP) > 0) ? 1 : 0 ;
  unsigned long long n_cols_idx = 0;
  unsigned int index_tsize = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES) > 0) ? 4 : 8;
  unsigned int reduce_elts = 0, reduce_elts_sq = 0;
  /* Configuration of redice kernel */
  if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD) {
    reduce_op = 0;
    reduce_elts = 1;
    reduce_elts_sq = 1;
  }
  if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD) {
    reduce_op = 0;
    reduce_elts = 1;
    reduce_elts_sq = 0;
  }
  if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X2_OP_ADD) {
    reduce_op = 0;
    reduce_elts = 0;
    reduce_elts_sq = 1;
  }
  if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX) {
    reduce_op = 1;
    reduce_elts = 1;
    reduce_elts_sq = 0;
  }
  if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ABSMAX) {
    reduce_op = 3;
    reduce_elts = 1;
    reduce_elts_sq = 0;
  }
  if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MIN) {
    reduce_op = 2;
    reduce_elts = 1;
    reduce_elts_sq = 0;
  }
  if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_ADD) {
    reduce_op = 0;
    reduce_elts = 1;
    reduce_elts_sq = 0;
    n_cols_idx = *((unsigned long long*)(param->in.tertiary));
  }
  if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_MAX) {
    reduce_op = 1;
    reduce_elts = 1;
    reduce_elts_sq = 0;
    n_cols_idx = *((unsigned long long*)(param->in.tertiary));
  }
  if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_MIN) {
    reduce_op = 2;
    reduce_elts = 1;
    reduce_elts_sq = 0;
    n_cols_idx = *((unsigned long long*)(param->in.tertiary));
  }
  if (reduce_elts == 0 && reduce_elts_sq == 1) {
    result_reduce_elts_sq = (void*)param->out.primary;
  }

  if (dtype_in == LIBXSMM_DATATYPE_F64 && dtype_out == LIBXSMM_DATATYPE_F64) {
    double *in_f64 = (double*)in;
    double *out_elts_f64 = (double*)result_reduce_elts;
    double *out_elts_sq_f64 = (double*)result_reduce_elts_sq;
    double *tmp_result_elts = NULL, *tmp_result_elts_squared = NULL;
    tmp_result_elts = (double*) malloc( sizeof(double)*result_size );
    tmp_result_elts_squared = (double*) malloc( sizeof(double)*result_size );
    if (reduce_op == 0) {
      /* Calculate reference results... */
      if (reduce_rows == 1) {
        for (j = 0; j < n; j++) {
          if (reduce_elts) tmp_result_elts[j] = 0.0;
          if (reduce_elts_sq) tmp_result_elts_squared[j] = 0.0;
          for (i = 0; i < m; i++) {
            double in_val = in_f64[i + j * ld_in];
            if (reduce_elts) tmp_result_elts[j] += in_val;
            if (reduce_elts_sq) tmp_result_elts_squared[j] += in_val * in_val;
          }
        }
        if (reduce_on_output > 0) {
          for (j = 0; j < n; j++) {
            if (reduce_elts) {
              double out_val = out_elts_f64[j];
              tmp_result_elts[j] += out_val;
            }
            if (reduce_elts_sq) {
              double out_val = out_elts_sq_f64[j];
              tmp_result_elts_squared[j] += out_val;
            }
          }
        }
      } else {
        if (n_cols_idx == 0) {
          /* In this case we reduce columns */
          for (i = 0; i < m; i++) {
            if (reduce_elts) tmp_result_elts[i] = 0.0;
            if (reduce_elts_sq) tmp_result_elts_squared[i] = 0.0;
            for (j = 0; j < n; j++) {
              double in_val = in_f64[i + j * ld_in];
              if (reduce_elts) tmp_result_elts[i] += in_val;
              if (reduce_elts_sq) tmp_result_elts_squared[i] += in_val * in_val;
            }
          }
          if (reduce_on_output > 0) {
            for (i = 0; i < m; i++) {
              if (reduce_elts) {
                double out_val = out_elts_f64[i];
                tmp_result_elts[i] += out_val;
              }
              if (reduce_elts_sq) {
                double out_val = out_elts_sq_f64[i];
                tmp_result_elts_squared[i] += out_val;
              }
            }
          }
        } else {
          for (i = 0; i < m; i++) {
            tmp_result_elts[i] = 0.0;
            for (jj = 0; jj < (libxsmm_blasint)n_cols_idx; jj++) {
              double in_val;
              j = (libxsmm_blasint) ((index_tsize == 4) ? col_idx_32bit[jj] : col_idx_64bit[jj]);
              in_val  = in_f64[i + j * ld_in];
              tmp_result_elts[i] += in_val;
            }
          }
        }
      }
    } else {
      if (reduce_rows == 1) {
        for (j = 0; j < n; j++) {
          tmp_result_elts[j] = in_f64[j * ld_in];
          for (i = 0; i < m; i++) {
            double in_val = in_f64[i + j * ld_in];;
            tmp_result_elts[j] = (reduce_op == 1) ? LIBXSMM_MAX( tmp_result_elts[j], in_val )
                                                  : (reduce_op == 3) ?  LIBXSMM_MAX( LIBXSMM_ABS(tmp_result_elts[j]), LIBXSMM_ABS(in_val) )
                                                                     :  LIBXSMM_MIN( tmp_result_elts[j], in_val );
          }
        }
      } else {
        if (n_cols_idx == 0) {
          n_cols = n;
        } else {
          n_cols = n_cols_idx;
        }
        if (reduce_op == 1 || reduce_op == 3) {
          for (i = 0; i < m; i++) {
            tmp_result_elts[i] = (reduce_op == 1) ? -FLT_MAX : 0;
            for (jj = 0; jj < n_cols; jj++) {
              double in_val;
              if (n_cols_idx == 0) {
                j = jj;
              } else {
                j = (libxsmm_blasint) ((index_tsize == 4) ? col_idx_32bit[jj] : col_idx_64bit[jj]);
              }
              in_val = in_f64[i + j * ld_in];
              if (reduce_op == 3) in_val = LIBXSMM_ABS(in_val);
              if (record_idx > 0) {
                if (in_val >= tmp_result_elts[i] ) {
                  tmp_result_elts[i] = in_val;
                  if (index_tsize == 4) {
                    argop_idx_32bit[i] = (unsigned int)j;
                  } else {
                    argop_idx_64bit[i] = (unsigned long long)j;
                  }
                }
              } else {
                tmp_result_elts[i] = LIBXSMM_MAX( in_val, tmp_result_elts[i]);
              }
            }
          }
        } else if (reduce_op == 2) {
          for (i = 0; i < m; i++) {
            tmp_result_elts[i] = FLT_MAX;
            for (jj = 0; jj < n_cols; jj++) {
              double in_val;
              if (n_cols_idx == 0) {
                j = jj;
              } else {
                j = (libxsmm_blasint) ((index_tsize == 4) ? col_idx_32bit[jj] : col_idx_64bit[jj]);
              }
              in_val = in_f64[i + j * ld_in];
              if (record_idx > 0) {
                if (in_val <= tmp_result_elts[i] ) {
                  tmp_result_elts[i] = in_val;
                  if (index_tsize == 4) {
                    argop_idx_32bit[i] = (unsigned int)j;
                  } else {
                    argop_idx_64bit[i] = (unsigned long long)j;
                  }
                }
              } else {
                tmp_result_elts[i] = LIBXSMM_MIN( in_val, tmp_result_elts[i]);
              }
            }
          }
        } else {
          printf("Should not happen\n");
          for (i = 0; i < result_size; i++) {
            if (reduce_elts) tmp_result_elts[i] = 0.0;
            if (reduce_elts_sq) out_elts_sq_f64[i] = 0.0;
          }
        }
      }
    }

    /* Now store the tmp result to output  */
    for (i = 0; i < result_size; i++) {
      if (reduce_elts) out_elts_f64[i] = tmp_result_elts[i];
      if (reduce_elts_sq) out_elts_sq_f64[i] = out_elts_sq_f64[i];
    }
    free (tmp_result_elts);
    free (tmp_result_elts_squared);
  } else {
    float *tmp_result_elts = NULL, *tmp_result_elts_squared = NULL;
    tmp_result_elts = (float*) malloc( sizeof(float)*result_size );
    tmp_result_elts_squared = (float*) malloc( sizeof(float)*result_size );
    if (reduce_op == 0) {
      /* Calculate reference results... */
      if (reduce_rows == 1) {
        for (j = 0; j < n; j++) {
          if (reduce_elts) tmp_result_elts[j] = 0.0;
          if (reduce_elts_sq) tmp_result_elts_squared[j] = 0.0;
          for (i = 0; i < m; i++) {
            float in_val = libxsmm_elementwise_get_float_value(in, i, j, ld_in, dtype_in, i_mateltwise_desc, 3);
            if (reduce_elts) tmp_result_elts[j] += in_val;
            if (reduce_elts_sq) tmp_result_elts_squared[j] += in_val * in_val;
          }
        }
        if (reduce_on_output > 0) {
          for (j = 0; j < n; j++) {
            if (reduce_elts) {
              float out_val = libxsmm_elementwise_get_float_value(result_reduce_elts, j, 0, ld_out, dtype_out, i_mateltwise_desc, 3);
              tmp_result_elts[j] += out_val;
            }
            if (reduce_elts_sq) {
              float out_val = libxsmm_elementwise_get_float_value(result_reduce_elts_sq, j, 0, ld_out, dtype_out, i_mateltwise_desc, 3);
              tmp_result_elts_squared[j] += out_val;
            }
          }
        }
      } else {
        if (n_cols_idx == 0) {
          /* In this case we reduce columns */
          for (i = 0; i < m; i++) {
            if (reduce_elts) tmp_result_elts[i] = 0.0;
            if (reduce_elts_sq) tmp_result_elts_squared[i] = 0.0;
            for (j = 0; j < n; j++) {
              float in_val = libxsmm_elementwise_get_float_value(in, i, j, ld_in, dtype_in, i_mateltwise_desc, 3);
              if (reduce_elts) tmp_result_elts[i] += in_val;
              if (reduce_elts_sq) tmp_result_elts_squared[i] += in_val * in_val;
            }
          }
          if (reduce_on_output > 0) {
            for (i = 0; i < m; i++) {
              if (reduce_elts) {
                float out_val = libxsmm_elementwise_get_float_value(result_reduce_elts, i, 0, ld_out, dtype_out, i_mateltwise_desc, 3);
                tmp_result_elts[i] += out_val;
              }
              if (reduce_elts_sq) {
                float out_val = libxsmm_elementwise_get_float_value(result_reduce_elts_sq, i, 0, ld_out, dtype_out, i_mateltwise_desc, 3);
                tmp_result_elts_squared[i] += out_val;
              }
            }
          }
        } else {
          for (i = 0; i < m; i++) {
            tmp_result_elts[i] = 0.0;
            for (jj = 0; jj < (libxsmm_blasint)n_cols_idx; jj++) {
              float in_val;
              j = (libxsmm_blasint) ((index_tsize == 4) ? col_idx_32bit[jj] : col_idx_64bit[jj]);
              in_val  = libxsmm_elementwise_get_float_value(in, i, j, ld_in, dtype_in, i_mateltwise_desc, 3);
              tmp_result_elts[i] += in_val;
            }
          }
        }
      }
    } else {
      if (reduce_rows == 1) {
        for (j = 0; j < n; j++) {
          tmp_result_elts[j] = libxsmm_elementwise_get_float_value(in, 0, j, ld_in, dtype_in, i_mateltwise_desc, 3);
          for (i = 0; i < m; i++) {
            float in_val  = libxsmm_elementwise_get_float_value(in, i, j, ld_in, dtype_in, i_mateltwise_desc, 3);
            tmp_result_elts[j] = (reduce_op == 1) ? LIBXSMM_MAX( tmp_result_elts[j], in_val )
                                                  : (reduce_op == 3) ?  LIBXSMM_MAX( LIBXSMM_ABS(tmp_result_elts[j]), LIBXSMM_ABS(in_val) )
                                                                     :  LIBXSMM_MIN( tmp_result_elts[j], in_val );
          }
        }
      } else {
        if (n_cols_idx == 0) {
          n_cols = n;
        } else {
          n_cols = n_cols_idx;
        }
        if (reduce_op == 1 || reduce_op == 3) {
          for (i = 0; i < m; i++) {
            tmp_result_elts[i] = (reduce_op == 1) ? -FLT_MAX : 0;
            for (jj = 0; jj < n_cols; jj++) {
              float in_val;
              if (n_cols_idx == 0) {
                j = jj;
              } else {
                j = (libxsmm_blasint) ((index_tsize == 4) ? col_idx_32bit[jj] : col_idx_64bit[jj]);
              }
              in_val = libxsmm_elementwise_get_float_value(in, i, j, ld_in, dtype_in, i_mateltwise_desc, 3);
              if (reduce_op == 3) in_val = LIBXSMM_ABS(in_val);
              if (record_idx > 0) {
                if (in_val >= tmp_result_elts[i] ) {
                  tmp_result_elts[i] = in_val;
                  if (index_tsize == 4) {
                    argop_idx_32bit[i] = (unsigned int)j;
                  } else {
                    argop_idx_64bit[i] = (unsigned long long)j;
                  }
                }
              } else {
                tmp_result_elts[i] = LIBXSMM_MAX( in_val, tmp_result_elts[i]);
              }
            }
          }
        }
        if (reduce_op == 2) {
          for (i = 0; i < m; i++) {
            tmp_result_elts[i] = FLT_MAX;
            for (jj = 0; jj < n_cols; jj++) {
              float in_val;
              if (n_cols_idx == 0) {
                j = jj;
              } else {
                j = (libxsmm_blasint) ((index_tsize == 4) ? col_idx_32bit[jj] : col_idx_64bit[jj]);
              }
              in_val = libxsmm_elementwise_get_float_value(in, i, j, ld_in, dtype_in, i_mateltwise_desc, 3);
              if (record_idx > 0) {
                if (in_val <= tmp_result_elts[i] ) {
                  tmp_result_elts[i] = in_val;
                  if (index_tsize == 4) {
                    argop_idx_32bit[i] = (unsigned int)j;
                  } else {
                    argop_idx_64bit[i] = (unsigned long long)j;
                  }
                }
              } else {
                tmp_result_elts[i] = LIBXSMM_MIN( in_val, tmp_result_elts[i]);
              }
            }
          }
        }
      }
    }

    /* Now store the tmp result to output  */
    for (i = 0; i < result_size; i++) {
      if (reduce_elts) libxsmm_elementwise_store_value(result_reduce_elts, (void*)&tmp_result_elts[i], i, 0, ld_out, 0, dtype_out, NULL, 0);
      if (reduce_elts_sq) libxsmm_elementwise_store_value(result_reduce_elts_sq, (void*)&tmp_result_elts_squared[i], i, 0, ld_out, 0, dtype_out, NULL, 0);
    }
    free (tmp_result_elts);
    free (tmp_result_elts_squared);
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_elementwise_gather_scatter_kernel(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint m = i_mateltwise_desc->m;
  libxsmm_blasint n = i_mateltwise_desc->n;
  libxsmm_blasint inp_ld = i_mateltwise_desc->ldi;
  libxsmm_blasint out_ld = i_mateltwise_desc->ldo;
  libxsmm_datatype dtype_in =  (libxsmm_datatype)libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0);
  libxsmm_blasint i, j, ind, ind2;
  unsigned int use_32bit_dtype = (dtype_in == LIBXSMM_DATATYPE_I32 || dtype_in == LIBXSMM_DATATYPE_F32) ? 1 : 0;
  unsigned int use_16bit_dtype = (dtype_in == LIBXSMM_DATATYPE_I16 || dtype_in == LIBXSMM_DATATYPE_F16 || dtype_in == LIBXSMM_DATATYPE_BF16) ? 1 : 0;
  unsigned int use_8bit_dtype = (dtype_in == LIBXSMM_DATATYPE_I8 || dtype_in == LIBXSMM_DATATYPE_BF8 || dtype_in == LIBXSMM_DATATYPE_HF8) ? 1 : 0;
  unsigned int use_64bit_index = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES) > 0) ? 1 : 0;
  unsigned long long *ind_array_64bit = (unsigned long long*) ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GATHER) ? param->in.secondary : param->out.secondary);
  unsigned int *ind_array_32bit = (unsigned int*) ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GATHER) ? param->in.secondary : param->out.secondary);
  float *sinp = (float*) param->in.primary;
  float *sout = (float*) param->out.primary;
  unsigned short *binp = (unsigned short*) param->in.primary;
  unsigned short *bout = (unsigned short*) param->out.primary;
  unsigned char *cinp = (unsigned char*) param->in.primary;
  unsigned char *cout = (unsigned char*) param->out.primary;
  libxsmm_blasint inp_m = 0, inp_n = 0, out_m = 0, out_n = 0;

  if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GATHER) {
    if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_COLS) > 0) {
      out_m = m;
      out_n = n;
    } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_ROWS) > 0) {
      out_m = m;
      out_n = n;
    } else {
      out_m = m;
      out_n = n;
    }
  } else {
    if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_COLS) > 0) {
      inp_n = n;
      out_m = m;
    } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_ROWS) > 0) {
      inp_m = m;
      inp_n = n;
    } else  {
      inp_m = m;
      inp_n = n;
    }
  }

  if (use_32bit_dtype == 1) {
    if (use_64bit_index == 1) {
      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GATHER) {
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_COLS) > 0) {
          for (ind = 0; ind < out_n; ind++) {
            j = (libxsmm_blasint)ind_array_64bit[ind];
            for (i = 0; i < out_m; i++) {
              sout[i + ind * out_ld] = sinp[i + j * inp_ld];
            }
          }
        } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_ROWS) > 0) {
          for (ind = 0; ind < out_m; ind++) {
            i = (libxsmm_blasint)ind_array_64bit[ind];
            for (j = 0; j < out_n; j++) {
              sout[ind + j * out_ld] = sinp[i + j * inp_ld];
            }
          }
        } else {
          for (ind2 = 0; ind2 < out_n; ind2++) {
            for (ind = 0; ind < out_m; ind++) {
              i = (libxsmm_blasint)ind_array_64bit[ind + ind2 * out_m];
              sout[ind + ind2 * out_ld] = sinp[i];
            }
          }
        }
      } else {
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_COLS) > 0) {
          for (ind = 0; ind < inp_n; ind++) {
            j = (libxsmm_blasint)ind_array_64bit[ind];
            for (i = 0; i < out_m; i++) {
              sout[i + j * out_ld] = sinp[i + ind * inp_ld];
            }
          }
        } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_ROWS) > 0) {
          for (ind = 0; ind < inp_m; ind++) {
            i = (libxsmm_blasint)ind_array_64bit[ind];
            for (j = 0; j < inp_n; j++) {
              sout[i + j * out_ld] = sinp[ind + j * inp_ld];
            }
          }
        } else  {
          for (ind2 = 0; ind2 < inp_n; ind2++) {
            for (ind = 0; ind < inp_m; ind++) {
              i = (libxsmm_blasint)ind_array_64bit[ind + ind2 * inp_m];
              sout[i] = sinp[ind + ind2 * inp_ld ];
            }
          }
        }
      }
    } else {
      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GATHER) {
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_COLS) > 0) {
          for (ind = 0; ind < out_n; ind++) {
            j = ind_array_32bit[ind];
            for (i = 0; i < out_m; i++) {
              sout[i + ind * out_ld] = sinp[i + j * inp_ld];
            }
          }
        } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_ROWS) > 0) {
          for (ind = 0; ind < out_m; ind++) {
            i = ind_array_32bit[ind];
            for (j = 0; j < out_n; j++) {
              sout[ind + j * out_ld] = sinp[i + j * inp_ld];
            }
          }
        } else  {
          for (ind2 = 0; ind2 < out_n; ind2++) {
            for (ind = 0; ind < out_m; ind++) {
              i = ind_array_32bit[ind + ind2 * out_m];
              sout[ind + ind2 * out_ld] = sinp[i];
            }
          }
        }
      } else {
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_COLS) > 0) {
          for (ind = 0; ind < inp_n; ind++) {
            j = ind_array_32bit[ind];
            for (i = 0; i < out_m; i++) {
              sout[i + j * out_ld] = sinp[i + ind * inp_ld];
            }
          }
        } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_ROWS) > 0) {
          for (ind = 0; ind < inp_m; ind++) {
            i = ind_array_32bit[ind];
            for (j = 0; j < inp_n; j++) {
              sout[i + j * out_ld] = sinp[ind + j * inp_ld];
            }
          }
        } else  {
          for (ind2 = 0; ind2 < inp_n; ind2++) {
            for (ind = 0; ind < inp_m; ind++) {
              i = ind_array_32bit[ind + ind2 * inp_m];
              sout[i] = sinp[ind + ind2 * inp_ld ];
            }
          }
        }
      }
    }
  } else if (use_16bit_dtype == 1) {
    if (use_64bit_index == 1) {
      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GATHER) {
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_COLS) > 0) {
          for (ind = 0; ind < out_n; ind++) {
            j = (libxsmm_blasint)ind_array_64bit[ind];
            for (i = 0; i < out_m; i++) {
              bout[i + ind * out_ld] = binp[i + j * inp_ld];
            }
          }
        } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_ROWS) > 0) {
          for (ind = 0; ind < out_m; ind++) {
            i = (libxsmm_blasint)ind_array_64bit[ind];
            for (j = 0; j < out_n; j++) {
              bout[ind + j * out_ld] = binp[i + j * inp_ld];
            }
          }
        } else {
          for (ind2 = 0; ind2 < out_n; ind2++) {
            for (ind = 0; ind < out_m; ind++) {
              i = (libxsmm_blasint)ind_array_64bit[ind + ind2 * out_m];
              bout[ind + ind2 * out_ld] = binp[i];
            }
          }
        }
      } else {
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_COLS) > 0) {
          for (ind = 0; ind < inp_n; ind++) {
            j = (libxsmm_blasint)ind_array_64bit[ind];
            for (i = 0; i < out_m; i++) {
              bout[i + j * out_ld] = binp[i + ind * inp_ld];
            }
          }
        } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_ROWS) > 0) {
          for (ind = 0; ind < inp_m; ind++) {
            i = (libxsmm_blasint)ind_array_64bit[ind];
            for (j = 0; j < inp_n; j++) {
              bout[i + j * out_ld] = binp[ind + j * inp_ld];
            }
          }
        } else  {
          for (ind2 = 0; ind2 < inp_n; ind2++) {
            for (ind = 0; ind < inp_m; ind++) {
              i = (libxsmm_blasint)ind_array_64bit[ind + ind2 * inp_m];
              bout[i] = binp[ind + ind2 * inp_ld ];
            }
          }
        }
      }
    } else {
      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GATHER) {
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_COLS) > 0) {
          for (ind = 0; ind < out_n; ind++) {
            j = ind_array_32bit[ind];
            for (i = 0; i < out_m; i++) {
              bout[i + ind * out_ld] = binp[i + j * inp_ld];
            }
          }
        } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_ROWS) > 0) {
          for (ind = 0; ind < out_m; ind++) {
            i = ind_array_32bit[ind];
            for (j = 0; j < out_n; j++) {
              bout[ind + j * out_ld] = binp[i + j * inp_ld];
            }
          }
        } else  {
          for (ind2 = 0; ind2 < out_n; ind2++) {
            for (ind = 0; ind < out_m; ind++) {
              i = ind_array_32bit[ind + ind2 * out_m];
              bout[ind + ind2 * out_ld] = binp[i];
            }
          }
        }
      } else {
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_COLS) > 0) {
          for (ind = 0; ind < inp_n; ind++) {
            j = ind_array_32bit[ind];
            for (i = 0; i < out_m; i++) {
              bout[i + j * out_ld] = binp[i + ind * inp_ld];
            }
          }
        } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_ROWS) > 0) {
          for (ind = 0; ind < inp_m; ind++) {
            i = ind_array_32bit[ind];
            for (j = 0; j < inp_n; j++) {
              bout[i + j * out_ld] = binp[ind + j * inp_ld];
            }
          }
        } else  {
          for (ind2 = 0; ind2 < inp_n; ind2++) {
            for (ind = 0; ind < inp_m; ind++) {
              i = ind_array_32bit[ind + ind2 * inp_m];
              bout[i] = binp[ind + ind2 * inp_ld ];
            }
          }
        }
      }
    }
  } else if (use_8bit_dtype == 1) {
    if (use_64bit_index == 1) {
      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GATHER) {
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_COLS) > 0) {
          for (ind = 0; ind < out_n; ind++) {
            j = (libxsmm_blasint)ind_array_64bit[ind];
            for (i = 0; i < out_m; i++) {
              cout[i + ind * out_ld] = cinp[i + j * inp_ld];
            }
          }
        } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_ROWS) > 0) {
          for (ind = 0; ind < out_m; ind++) {
            i = (libxsmm_blasint)ind_array_64bit[ind];
            for (j = 0; j < out_n; j++) {
              cout[ind + j * out_ld] = cinp[i + j * inp_ld];
            }
          }
        } else {
          for (ind2 = 0; ind2 < out_n; ind2++) {
            for (ind = 0; ind < out_m; ind++) {
              i = (libxsmm_blasint)ind_array_64bit[ind + ind2 * out_m];
              cout[ind + ind2 * out_ld] = cinp[i];
            }
          }
        }
      } else {
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_COLS) > 0) {
          for (ind = 0; ind < inp_n; ind++) {
            j = (libxsmm_blasint)ind_array_64bit[ind];
            for (i = 0; i < out_m; i++) {
              cout[i + j * out_ld] = cinp[i + ind * inp_ld];
            }
          }
        } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_ROWS) > 0) {
          for (ind = 0; ind < inp_m; ind++) {
            i = (libxsmm_blasint)ind_array_64bit[ind];
            for (j = 0; j < inp_n; j++) {
              cout[i + j * out_ld] = cinp[ind + j * inp_ld];
            }
          }
        } else  {
          for (ind2 = 0; ind2 < inp_n; ind2++) {
            for (ind = 0; ind < inp_m; ind++) {
              i = (libxsmm_blasint)ind_array_64bit[ind + ind2 * inp_m];
              cout[i] = cinp[ind + ind2 * inp_ld ];
            }
          }
        }
      }
    } else {
      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GATHER) {
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_COLS) > 0) {
          for (ind = 0; ind < out_n; ind++) {
            j = ind_array_32bit[ind];
            for (i = 0; i < out_m; i++) {
              cout[i + ind * out_ld] = cinp[i + j * inp_ld];
            }
          }
        } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_ROWS) > 0) {
          for (ind = 0; ind < out_m; ind++) {
            i = ind_array_32bit[ind];
            for (j = 0; j < out_n; j++) {
              cout[ind + j * out_ld] = cinp[i + j * inp_ld];
            }
          }
        } else  {
          for (ind2 = 0; ind2 < out_n; ind2++) {
            for (ind = 0; ind < out_m; ind++) {
              i = ind_array_32bit[ind + ind2 * out_m];
              cout[ind + ind2 * out_ld] = cinp[i];
            }
          }
        }
      } else {
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_COLS) > 0) {
          for (ind = 0; ind < inp_n; ind++) {
            j = ind_array_32bit[ind];
            for (i = 0; i < out_m; i++) {
              cout[i + j * out_ld] = cinp[i + ind * inp_ld];
            }
          }
        } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_GS_ROWS) > 0) {
          for (ind = 0; ind < inp_m; ind++) {
            i = ind_array_32bit[ind];
            for (j = 0; j < inp_n; j++) {
              cout[i + j * out_ld] = cinp[ind + j * inp_ld];
            }
          }
        } else  {
          for (ind2 = 0; ind2 < inp_n; ind2++) {
            for (ind = 0; ind < inp_m; ind++) {
              i = ind_array_32bit[ind + ind2 * inp_m];
              cout[i] = cinp[ind + ind2 * inp_ld ];
            }
          }
        }
      }
    }
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_reference_unary_elementwise(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  if (libxsmm_is_reduce_op(i_mateltwise_desc) > 0) {
    libxsmm_elementwise_reduce_kernel( param, i_mateltwise_desc);
  } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GATHER || i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_SCATTER) {
    libxsmm_elementwise_gather_scatter_kernel( param, i_mateltwise_desc);
  } else if (libxsmm_is_transform_op(i_mateltwise_desc) > 0) {
    libxsmm_elementwise_transform_kernel( param, i_mateltwise_desc);
  } else {
    libxsmm_blasint i, j;
    libxsmm_blasint M = i_mateltwise_desc->m;
    libxsmm_blasint N = (libxsmm_blasint) (( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) ? *((unsigned long long*)(param->op.primary)) : i_mateltwise_desc->n);
    libxsmm_blasint ldi = i_mateltwise_desc->ldi;
    libxsmm_blasint ldo = i_mateltwise_desc->ldo;
    libxsmm_datatype dtype_in = (libxsmm_datatype)libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0);
    libxsmm_datatype dtype_out = (libxsmm_datatype)libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT);
    libxsmm_datatype dtype_comp = (libxsmm_datatype)libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP);
    libxsmm_bitfield flags =  i_mateltwise_desc->flags;
    void *rng_state = (void*) param->op.secondary;
    void *in = (void*)param->in.primary;
    void *out = (void*)param->out.primary;
    void *out_dump = (void*)param->out.secondary;
    unsigned int seed_idx = 0;

    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_TO_SCALAR_OP_ADD) {
      float acc_f32 = 0.0f;
      double acc_f64 = 0.0;
      for ( j = 0; j < N; ++j ) {
        for ( i = 0; i < M; ++i ) {
          if (dtype_in == LIBXSMM_DATATYPE_F64 && dtype_out == LIBXSMM_DATATYPE_F64 && dtype_comp == LIBXSMM_DATATYPE_F64) {
            double *in_double = (double*)in;
            double in_val_double = in_double[libxsmm_elementwise_get_index(i, j, ldi, i_mateltwise_desc, 0)];
            acc_f64 += in_val_double;
          } else {
            float in_val  = libxsmm_elementwise_get_float_value(in, i, j, ldi, dtype_in, i_mateltwise_desc, 0);
            acc_f32 += in_val;
          }
        }
      }
      if (dtype_in == LIBXSMM_DATATYPE_F64 && dtype_out == LIBXSMM_DATATYPE_F64 && dtype_comp == LIBXSMM_DATATYPE_F64) {
        double *out_f64 = (double*)out;
        out_f64[0] = acc_f64;
      } else {
        libxsmm_elementwise_store_value(out, (void*)&acc_f32, 0, 0, ldo, 0, dtype_out, NULL, 0);
      }
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD_NCNC_FORMAT ) {
      libxsmm_blasint bc = i_mateltwise_desc->m;
      libxsmm_blasint bn = i_mateltwise_desc->n;
      libxsmm_blasint C = i_mateltwise_desc->ldi;
      libxsmm_blasint ic = 0, iC = 0, iN = 0, i_n = 0;
      N = i_mateltwise_desc->ldo;
      for (iC = 0; iC < C/bc; iC++) {
        for (ic = 0; ic < bc; ic++) {
          libxsmm_blasint c = iC*bc+ic;
          float tmp = 0.0f;
          for (iN = 0; iN < N/bn; iN++) {
            for (i_n = 0; i_n < bn; i_n++) {
              libxsmm_blasint offset = iN * C * bn + iC * bn * bc + i_n * bc + ic;
              float in_val  = libxsmm_elementwise_get_float_value(in, offset, 0, offset, dtype_in, i_mateltwise_desc, 0);
              tmp += in_val;
            }
          }
          libxsmm_elementwise_store_value(out, (void*)&tmp, c, 0, ldo, 0, dtype_out, NULL, 0);
        }
      }
    } else if (  i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU ||
          i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU ||
          i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU ) {
      unsigned int bitm = ( (flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) ? 1 : 0;
      libxsmm_blasint mask_ld = (bitm == 0) ? ldo : LIBXSMM_UPDIV(ldo, 16)*16;
      float alpha = (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) ? 1.0f : *((float*)(param->op.primary)) ;
      unsigned char *relu_mask = (unsigned char*) param->out.secondary;
      for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
          float in_val  = libxsmm_elementwise_get_float_value(in, i, j, ldi, dtype_in, i_mateltwise_desc, 0);
          float out_val = 0.0;
          if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) {
            out_val = ( in_val <= 0.0f ) ? 0.0f : in_val;
          } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) {
            out_val = ( in_val <= 0.0f ) ? alpha*in_val : in_val;
          } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU) {
            out_val = ( in_val <= 0.0f ) ? alpha*(LIBXSMM_EXPF(in_val)-1.0f) : in_val;
          } else {
            /* Should not happen  */
          }
          libxsmm_elementwise_store_value(out, (void*)&out_val, i, j, ldo, 0, dtype_out, NULL, 0);
          if (bitm > 0) {
            if ( ( in_val <= 0.0f ) ) {
              libxsmm_zero_bit((unsigned char*)relu_mask, i, j, mask_ld);
            } else {
              libxsmm_set_bit((unsigned char*)relu_mask, i, j, mask_ld);
            }
          }
        }
      }
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV ||
                i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV ||
                i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV ) {
      unsigned int bitm = ( (flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) ? 1 : 0;
      libxsmm_blasint mask_ld = (bitm == 0) ? ldi : LIBXSMM_UPDIV(ldi, 16)*16;
      float alpha = (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) ? 1.0f : *((float*)(param->op.primary)) ;
      unsigned char *relu_mask = (unsigned char*) param->in.secondary;
      void *out_fwd = (void*) param->in.secondary;
      for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
          float in_val  = libxsmm_elementwise_get_float_value(in, i, j, ldi, dtype_in, i_mateltwise_desc, 0);
          float out_val = 0.0;
          if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) {
            unsigned char bit_val = libxsmm_extract_bit((const char*)relu_mask, i, j, mask_ld);
            out_val = ( bit_val == 0 ) ? 0.0f : in_val;
          } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) {
            unsigned char bit_val = libxsmm_extract_bit((const char*)relu_mask, i, j, mask_ld);
            out_val = ( bit_val == 0 ) ? alpha*in_val : in_val;
          } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) {
            float out_fwd_val = libxsmm_elementwise_get_float_value(out_fwd, i, j, ldi, dtype_in, i_mateltwise_desc, 0);
            out_val = ( out_fwd_val > 0 ) ? in_val : in_val * (out_fwd_val + alpha);
          } else {
            /* Should not happen  */
          }
          libxsmm_elementwise_store_value(out, (void*)&out_val, i, j, ldo, 0, dtype_out, NULL, 0);
        }
      }
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_QUANT ) {
      unsigned int skip_scf_cvt = ( (flags & LIBXSMM_MELTW_FLAG_UNARY_NO_SCF_QUANT) > 0 ) ? 1 : 0;
      unsigned int signed_sat = ( (flags & LIBXSMM_MELTW_FLAG_UNARY_SIGN_SAT_QUANT) > 0 ) ? 1 : 0;
      float *in_ptr = (float*)in;
      float scf_quant = (skip_scf_cvt > 0) ? 1.0f : *((float*)(param->in.secondary));
      if ( (dtype_in == LIBXSMM_DATATYPE_F32) && (dtype_out == LIBXSMM_DATATYPE_I8) ) {
        char *char_data = (char*)out;
        for ( j = 0; j < N; ++j ) {
          for ( i = 0; i < M; ++i ) {
            float in_f = in_ptr[libxsmm_elementwise_get_index(i, j, ldi, i_mateltwise_desc, 0)];
            if (signed_sat > 0) {
              float tmp = LIBXSMM_NEARBYINTF( in_f * scf_quant );
              if (tmp < -128) {
                tmp = -128.0;
              }
              if (tmp > 127) {
                tmp = 127.0;
              }
              char_data[(j*ldo)+i] = (char) tmp;
            } else {
              char_data[(j*ldo)+i] = (char) (0x000000ff & ((int)LIBXSMM_NEARBYINTF( in_f * scf_quant )));
            }
          }
        }
      } else if ( (dtype_in == LIBXSMM_DATATYPE_F32) && (dtype_out == LIBXSMM_DATATYPE_I16) ) {
        short *short_data = (short*)out;
        for ( j = 0; j < N; ++j ) {
          for ( i = 0; i < M; ++i ) {
            float in_f = in_ptr[libxsmm_elementwise_get_index(i, j, ldi, i_mateltwise_desc, 0)];
            if (signed_sat > 0) {
              float tmp = LIBXSMM_NEARBYINTF( in_f * scf_quant );
              if (tmp < -32768) {
                tmp = -32768.0;
              }
              if (tmp > 32767) {
                tmp = 32767.0;
              }
              short_data[(j*ldo)+i] = (short) tmp;
            } else {
              short_data[(j*ldo)+i] = (short) (0x0000ffff & ((int)LIBXSMM_NEARBYINTF( in_f * scf_quant )));
            }
          }
        }
      } else if ( (dtype_in == LIBXSMM_DATATYPE_F32) && (dtype_out == LIBXSMM_DATATYPE_I32) ) {
        int *int_data = (int*)out;
        for ( j = 0; j < N; ++j ) {
          for ( i = 0; i < M; ++i ) {
            float in_f = in_ptr[libxsmm_elementwise_get_index(i, j, ldi, i_mateltwise_desc, 0)];
            int_data[(j*ldo)+i] = LIBXSMM_NEARBYINTF( in_f * scf_quant );
          }
        }
      } else {
        /* Should not happen  */
      }
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DEQUANT ) {
      unsigned int skip_scf_cvt = ( (flags & LIBXSMM_MELTW_FLAG_UNARY_NO_SCF_QUANT) > 0 ) ? 1 : 0;
      float *out_data = (float*)out;
      float scf_dequant = (skip_scf_cvt > 0) ? 1.0f : *((float*)(param->in.secondary));
      if ( (dtype_out == LIBXSMM_DATATYPE_F32) && (dtype_in == LIBXSMM_DATATYPE_I8) ) {
        char *char_data = (char*)in;
        for ( j = 0; j < N; ++j ) {
          for ( i = 0; i < M; ++i ) {
            char in_val = char_data[libxsmm_elementwise_get_index(i, j, ldi, i_mateltwise_desc, 0)];
            out_data[(j*ldo)+i] = ((float)in_val)* scf_dequant;
          }
        }
      } else if ( (dtype_out == LIBXSMM_DATATYPE_F32) && (dtype_in == LIBXSMM_DATATYPE_I16) ) {
        short *short_data = (short*)in;
        for ( j = 0; j < N; ++j ) {
          for ( i = 0; i < M; ++i ) {
            short in_val = short_data[libxsmm_elementwise_get_index(i, j, ldi, i_mateltwise_desc, 0)];
            out_data[(j*ldo)+i] = ((float)in_val)* scf_dequant;
          }
        }
      } else if ( (dtype_out == LIBXSMM_DATATYPE_F32) && (dtype_in == LIBXSMM_DATATYPE_I32) ) {
        int *int_data = (int*)in;
        for ( j = 0; j < N; ++j ) {
          for ( i = 0; i < M; ++i ) {
            int in_val = int_data[libxsmm_elementwise_get_index(i, j, ldi, i_mateltwise_desc, 0)];
            out_data[(j*ldo)+i] = ((float)in_val)* scf_dequant;
          }
        }
      } else {
        /* Should not happen  */
      }
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT ) {
      unsigned int bitm = ( (flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) ? 1 : 0;
      libxsmm_blasint mask_ld = (bitm == 0) ? ldo : LIBXSMM_UPDIV(ldo, 16)*16;
      float vrng[16];
      float p = *((float*)(param->op.primary));
      float pn = 1 - p;
      float pi = 1/pn;
      libxsmm_blasint jj;
      libxsmm_blasint w = libxsmm_cpuid_vlen32(libxsmm_target_archid);
      unsigned char *dropout_mask = (unsigned char*) param->out.secondary;
      for (j = 0; j < 16; j++) {
        vrng[j] = 0.0;
      }
      for (j = 0; j < N; j++) {
        for (i = 0; i < (libxsmm_blasint)LIBXSMM_LO2(M, w); i+=w) {
          libxsmm_lsfr_Xwide( (unsigned int*)rng_state, vrng, w );
          for ( jj = 0; jj < w; ++jj ) {
            float in_val  = libxsmm_elementwise_get_float_value(in, i+jj, j, ldi, dtype_in, i_mateltwise_desc, 0);
            float out_val = ( vrng[jj] < pn ) ? pi * in_val : 0.0f;
            libxsmm_elementwise_store_value(out, (void*)&out_val, i+jj, j, ldo, 0, dtype_out, NULL, 0);
            if (bitm > 0) {
              if ( vrng[jj] < pn ) {
                libxsmm_set_bit((unsigned char*)dropout_mask, i+jj, j, mask_ld);
              } else {
                libxsmm_zero_bit((unsigned char*)dropout_mask, i+jj, j, mask_ld);
              }
            }
          }
        }
        if (i < M) {
          libxsmm_lsfr_Xwide( (unsigned int*)rng_state, vrng, w );
          jj = 0;
          for ( ; i < M; ++i ) {
            float in_val  = libxsmm_elementwise_get_float_value(in, i, j, ldi, dtype_in, i_mateltwise_desc, 0);
            float out_val = ( vrng[jj] < pn ) ? pi * in_val : 0.0f;
            libxsmm_elementwise_store_value(out, (void*)&out_val, i, j, ldo, 0, dtype_out, NULL, 0);
            if (bitm > 0) {
              if ( vrng[jj] < pn ) {
                libxsmm_set_bit((unsigned char*)dropout_mask, i, j, mask_ld);
              } else {
                libxsmm_zero_bit((unsigned char*)dropout_mask, i, j, mask_ld);
              }
            }
            jj++;
          }
        }
      }
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV ) {
      unsigned int bitm = ( (flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) ? 1 : 0;
      libxsmm_blasint mask_ld = (bitm == 0) ? ldi : LIBXSMM_UPDIV(ldi, 16)*16;
      float p = *((float*)(param->op.primary));
      float pn = 1.0f - p;
      float pi = 1.0f/pn;
      unsigned char *dropout_mask = (unsigned char*) param->in.secondary;
      for ( j = 0; j < N; ++j ) {
        for ( i = 0; i < M; ++i ) {
          float in_val  = libxsmm_elementwise_get_float_value(in, i, j, ldi, dtype_in, i_mateltwise_desc, 0) * pi;
          unsigned char bit_val = libxsmm_extract_bit((const char*)dropout_mask, i, j, mask_ld);
          float out_val = (bit_val > 0) ? in_val : 0.0f;
          libxsmm_elementwise_store_value(out, (void*)&out_val, i, j, ldo, 0, dtype_out, NULL, 0);
        }
      }
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_UNZIP ) {
      /* Special case for unzip TPP */
      float *_in = (float*)in;
      unsigned long long offset = *((unsigned long long*)(param->out.secondary));
      libxsmm_bfloat16 *out_lo = (libxsmm_bfloat16*)out;
      libxsmm_bfloat16 *out_hi = (libxsmm_bfloat16*)((char*)out + offset);
      for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
          libxsmm_bfloat16_f32 bf16_hp;
          bf16_hp.f = _in[libxsmm_elementwise_get_index(i, j, ldi, i_mateltwise_desc, 0)];
          out_lo[j * ldo + i] = bf16_hp.i[0];
          out_hi[j * ldo + i] = bf16_hp.i[1];
        }
      }
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X2 ||
               i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3) {
      unsigned long long *strides = (unsigned long long*)(param->out.secondary);
      /* Special case for decompose TPP */
      for ( j = 0; j < N; ++j ) {
        for ( i = 0; i < M; ++i ) {
          float in_value = 0.0f;
          libxsmm_bfloat16 out1_value = 0, out2_value = 0, out3_value = 0;
          float ftmp = 0.0f, ftmp2 = 0.0f;
          libxsmm_bfloat16_f32 tmp;
          const float* f_in = (const float*)in;
          libxsmm_bfloat16* bf16_out = (libxsmm_bfloat16*)out;
          in_value = f_in[libxsmm_elementwise_get_index(i, j, ldi, i_mateltwise_desc, 0)];
          tmp.f = in_value;
          tmp.i[0] = 0;
          out1_value = tmp.i[1];
          ftmp = in_value - tmp.f;
          if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3 ) {
            tmp.f = ftmp;
            tmp.i[0] = 0;
            out2_value = tmp.i[1];
            ftmp2 = ftmp - tmp.f;
            out3_value = my_libxsmm_convert_f32_to_bf16_rne(ftmp2);
          } else {
            out2_value = my_libxsmm_convert_f32_to_bf16_rne(ftmp);
          }
          bf16_out[(j*ldo) + i                   ] = out1_value;
          bf16_out[(j*ldo) + i + (strides[0]/2)  ] = out2_value;
          if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3) {
            bf16_out[(j*ldo) + i + (strides[1]/2)] = out3_value;
          }
        }
      }
    } else {
      for ( j = 0; j < N; ++j ) {
        for ( i = 0; i < M; ++i ) {
          if (dtype_in == LIBXSMM_DATATYPE_F64 && dtype_out == LIBXSMM_DATATYPE_F64 && dtype_comp == LIBXSMM_DATATYPE_F64) {
            double *in_double = (double*)in;
            double *out_double = (double*)out;
            double in_val_double = in_double[libxsmm_elementwise_get_index(i, j, ldi, i_mateltwise_desc, 0)];
            out_double[i + j * ldo] = libxsmm_fp64_unary_compute(in_val_double, (libxsmm_meltw_unary_type)i_mateltwise_desc->param);
            if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP) {
              double* out1 = (double*)out_dump;
              out1[i + j * ldo] =  out_double[i + j * ldo];
            }
          } else {
            float in_val  = libxsmm_elementwise_get_float_value(in, i, j, ldi, dtype_in, i_mateltwise_desc, 0);
            float out_val = libxsmm_fp32_unary_compute(in_val, (libxsmm_meltw_unary_type)i_mateltwise_desc->param);
            libxsmm_elementwise_store_value(out, (void*)&out_val, i, j, ldo, ((flags & LIBXSMM_MELTW_FLAG_UNARY_STOCHASTIC_ROUND) > 0 ), dtype_out, rng_state, seed_idx);
            seed_idx++;
            if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP) {
              if (((flags & LIBXSMM_MELTW_FLAG_UNARY_STOCHASTIC_ROUND) > 0) && (dtype_out == LIBXSMM_DATATYPE_BF8)) {
                char* out0 = (char*)out;
                char* out1 = (char*)out_dump;
                out1[i + j * ldo] = out0[i + j * ldo];
              } else {
                libxsmm_elementwise_store_value(out_dump, (void*)&out_val, i, j, ldo, 0, dtype_out, NULL, 0);
              }
            }
          }
        }
      }
    }
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_reference_binary_elementwise(libxsmm_meltw_binary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j;
  libxsmm_blasint M = i_mateltwise_desc->m;
  libxsmm_blasint N = i_mateltwise_desc->n;
  libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  libxsmm_blasint ldi1 = i_mateltwise_desc->ldi2;
  libxsmm_blasint ldo = i_mateltwise_desc->ldo;
  libxsmm_datatype dtype_in = (libxsmm_datatype)libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0);
  libxsmm_datatype dtype_in1 = (libxsmm_datatype)libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1);
  libxsmm_datatype dtype_out = (libxsmm_datatype)libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT);
  libxsmm_datatype dtype_comp = (libxsmm_datatype)libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP);
  libxsmm_bitfield flags =  i_mateltwise_desc->flags;
  void *rng_state = (void*) param->op.secondary;
  void *in = (void*)param->in0.primary;
  void *in1 = (void*)param->in1.primary;
  void *out = (void*)param->out.primary;
  unsigned int seed_idx = 0;

  if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD) {
    float acc_f32 = 0.0f;
    double acc_f64 = 0.0;
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        if (dtype_in == LIBXSMM_DATATYPE_F64 && dtype_in1 == LIBXSMM_DATATYPE_F64 && dtype_out == LIBXSMM_DATATYPE_F64 && dtype_comp == LIBXSMM_DATATYPE_F64) {
          double *in_double = (double*)in;
          double *in1_double = (double*)in1;
          double in_val_double = in_double[libxsmm_elementwise_get_index(i, j, ldi, i_mateltwise_desc, 0)];
          double in1_val_double = in1_double[libxsmm_elementwise_get_index(i, j, ldi1, i_mateltwise_desc, 1)];
          acc_f64 += in_val_double * in1_val_double;
        } else {
          float in_val  = libxsmm_elementwise_get_float_value(in, i, j, ldi, dtype_in, i_mateltwise_desc, 0);
          float in1_val  = libxsmm_elementwise_get_float_value(in1, i, j, ldi1, dtype_in1, i_mateltwise_desc, 1);
          acc_f32 += in_val * in1_val;
        }
      }
    }
    if (dtype_in == LIBXSMM_DATATYPE_F64 && dtype_in1 == LIBXSMM_DATATYPE_F64 && dtype_out == LIBXSMM_DATATYPE_F64 && dtype_comp == LIBXSMM_DATATYPE_F64) {
      double *out_f64 = (double*)out;
      out_f64[0] = acc_f64;
    } else {
      libxsmm_elementwise_store_value(out, (void*)&acc_f32, 0, 0, ldo, 0, dtype_out, NULL, 0);
    }
  } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_ZIP ) {
    /* Special case for zip TPP */
    float *out_res = (float*)out;
    libxsmm_bfloat16 *in_lo = (libxsmm_bfloat16*)in;
    libxsmm_bfloat16 *in_hi = (libxsmm_bfloat16*)in1;
    for (j = 0; j < N; j++) {
      for (i = 0; i < M; i++) {
        libxsmm_bfloat16_f32 bf16_hp;
        bf16_hp.i[0] = in_lo[libxsmm_elementwise_get_index(i, j, ldi, i_mateltwise_desc, 0)];
        bf16_hp.i[1] = in_hi[libxsmm_elementwise_get_index(i, j, ldi1, i_mateltwise_desc, 1)];
        out_res[j * ldo + i] = bf16_hp.f;
      }
    }
  } else {
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        if (dtype_in == LIBXSMM_DATATYPE_F64 && dtype_in1 == LIBXSMM_DATATYPE_F64 && dtype_out == LIBXSMM_DATATYPE_F64 && dtype_comp == LIBXSMM_DATATYPE_F64) {
          double *in_double = (double*)in;
          double *in1_double = (double*)in1;
          double *out_double = (double*)out;
          double in_val_double = in_double[libxsmm_elementwise_get_index(i, j, ldi, i_mateltwise_desc, 0)];
          double in1_val_double = in1_double[libxsmm_elementwise_get_index(i, j, ldi1, i_mateltwise_desc, 1)];
          out_double[i + j * ldo] = libxsmm_fp64_binary_compute(in_val_double, in1_val_double, out_double[i + j * ldo], (libxsmm_meltw_binary_type)i_mateltwise_desc->param);
        } else {
          float in_val  = libxsmm_elementwise_get_float_value(in, i, j, ldi, dtype_in, i_mateltwise_desc, 0);
          float in1_val  = libxsmm_elementwise_get_float_value(in1, i, j, ldi1, dtype_in1, i_mateltwise_desc, 1);
          if (libxsmm_is_cmp_op((libxsmm_meltw_binary_type)i_mateltwise_desc->param) > 0) {
            unsigned int l_ldo = LIBXSMM_UPDIV(ldo, 16)*16;
            float out_value = libxsmm_fp32_binary_compute(in_val, in1_val, 0.0,(libxsmm_meltw_binary_type)i_mateltwise_desc->param);
            unsigned char result_bit = (out_value > 0.1) ? 1 : 0;
            if (result_bit > 0) {
              libxsmm_set_bit((unsigned char*)out, i, j, l_ldo);
            } else {
              libxsmm_zero_bit((unsigned char*)out, i, j, l_ldo);
            }
          } else {
            float out_in  = libxsmm_elementwise_get_float_value(out, i, j, ldo, dtype_out, i_mateltwise_desc, 3);
            float out_val = libxsmm_fp32_binary_compute(in_val, in1_val, out_in, (libxsmm_meltw_binary_type)i_mateltwise_desc->param);
            libxsmm_elementwise_store_value(out, (void*)&out_val, i, j, ldo, ((flags & LIBXSMM_MELTW_FLAG_BINARY_STOCHASTIC_ROUND) > 0 ), dtype_out, rng_state, seed_idx);
          }
          seed_idx++;
        }
      }
    }
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_reference_ternary_elementwise(libxsmm_meltw_ternary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j;
  libxsmm_blasint M = i_mateltwise_desc->m;
  libxsmm_blasint N = i_mateltwise_desc->n;
  libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  libxsmm_blasint ldi1 = i_mateltwise_desc->ldi2;
  libxsmm_blasint ldi2 = i_mateltwise_desc->ldi3;
  libxsmm_blasint ldo = i_mateltwise_desc->ldo;
  libxsmm_datatype dtype_in = (libxsmm_datatype)libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0);
  libxsmm_datatype dtype_in1 = (libxsmm_datatype)libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1);
  libxsmm_datatype dtype_in2 = (libxsmm_datatype)libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN2);
  libxsmm_datatype dtype_out = (libxsmm_datatype)libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT);
  libxsmm_datatype dtype_comp = (libxsmm_datatype)libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP);
  libxsmm_bitfield flags =  i_mateltwise_desc->flags;
  void *rng_state = (void*) param->op.secondary;
  void *in = (void*)param->in0.primary;
  void *in1 = (void*)param->in1.primary;
  void *in2 = (void*)param->in2.primary;
  void *out = (void*)param->out.primary;
  unsigned int seed_idx = 0;

  if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_TERNARY_SELECT) {
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        libxsmm_blasint l_ld2 = LIBXSMM_UPDIV(ldi2, 16)*16;
        if (dtype_in == LIBXSMM_DATATYPE_F64 && dtype_in1 == LIBXSMM_DATATYPE_F64 && dtype_in2 == LIBXSMM_DATATYPE_F64 && dtype_out == LIBXSMM_DATATYPE_F64 && dtype_comp == LIBXSMM_DATATYPE_F64) {
          double *in_double = (double*)in;
          double *in1_double = (double*)in1;
          double *out_double = (double*)out;
          double in_val_double = in_double[libxsmm_elementwise_get_index(i, j, ldi, i_mateltwise_desc, 0)];
          double in1_val_double = in1_double[libxsmm_elementwise_get_index(i, j, ldi1, i_mateltwise_desc, 1)];
          unsigned char bit_val = libxsmm_extract_bit((const char*)in2, i, j, l_ld2);
          double out_value = (bit_val == 0) ? in_val_double : in1_val_double;
          out_double[i + j * ldo] = out_value;
        } else {
          float in_val  = libxsmm_elementwise_get_float_value(in, i, j, ldi, dtype_in, i_mateltwise_desc, 0);
          float in1_val  = libxsmm_elementwise_get_float_value(in1, i, j, ldi1, dtype_in1, i_mateltwise_desc, 1);
          unsigned char bit_val = libxsmm_extract_bit((const char*)in2, i, j, l_ld2);
          float out_value = (bit_val == 0) ? in_val : in1_val;
          libxsmm_elementwise_store_value(out, (void*)&out_value, i, j, ldo, ((flags & LIBXSMM_MELTW_FLAG_TERNARY_STOCHASTIC_ROUND) > 0 ), dtype_out, rng_state, seed_idx);
          seed_idx++;
        }
      }
    }
  } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_TERNARY_MULADD || i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_TERNARY_NMULADD) {
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        float out_tmp = 0.0f;
        float in_val  = libxsmm_elementwise_get_float_value(in, i, j, ldi, dtype_in, i_mateltwise_desc, 0);
        float in1_val  = libxsmm_elementwise_get_float_value(in1, i, j, ldi1, dtype_in1, i_mateltwise_desc, 1);
        float in2_val  = libxsmm_elementwise_get_float_value(in2, i, j, ldi2, dtype_in2, i_mateltwise_desc, 2);
        if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_TERNARY_MULADD) {
          out_tmp = in2_val + in_val * in1_val;
        } else {
          out_tmp = in1_val - in_val * in2_val;
        }
        libxsmm_elementwise_store_value(out, (void*)&out_tmp, i, j, ldo, ((flags & LIBXSMM_MELTW_FLAG_TERNARY_STOCHASTIC_ROUND) > 0 ), dtype_out, rng_state, seed_idx);
        seed_idx++;
      }
    }
  } else {

  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_reference_elementwise(void *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
    libxsmm_reference_unary_elementwise((libxsmm_meltw_unary_param*)param, i_mateltwise_desc);
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
    libxsmm_reference_binary_elementwise((libxsmm_meltw_binary_param*)param, i_mateltwise_desc);
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
    libxsmm_reference_ternary_elementwise((libxsmm_meltw_ternary_param*)param, i_mateltwise_desc);
  } else {
    /* Should not happen  */
  }
  return;
}
