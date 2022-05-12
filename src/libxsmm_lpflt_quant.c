/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm_lpflt_quant.h>
#include "libxsmm_main.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

LIBXSMM_API_INLINE float libxsmm_internal_get_max( float* in_buffer, int length ) {
  float absmax_value = LIBXSMM_ABS(in_buffer[0]);
  int i = 0;
#ifdef _OPENMP
  LIBXSMM_OMP_VAR(i);
# pragma omp parallel private(i)
  {
    float my_absmax_value = absmax_value;
#   pragma omp for
    for (i = 0; i < length; ++i ) {
      if (LIBXSMM_ABS(in_buffer[i]) > my_absmax_value) {
        my_absmax_value = LIBXSMM_ABS(in_buffer[i]);
      }
    }
#   pragma omp critical
    {
      if (my_absmax_value > absmax_value) {
        absmax_value = my_absmax_value;
      }
    }
  }
#else
  for (i = 1; i < length; ++i ) {
    if (LIBXSMM_ABS(in_buffer[i]) > absmax_value) {
      absmax_value = LIBXSMM_ABS(in_buffer[i]);
    }
  }
#endif

  return absmax_value;
}


LIBXSMM_API_INLINE unsigned char libxsmm_internal_get_max_exp( float* in_buffer, int length ) {
  libxsmm_float_uint val_exp;
  unsigned char max_exp = 0;

  /* bit-wise conversion to int */
  val_exp.f = libxsmm_internal_get_max( in_buffer, length );
  /* shift by mantissa to the right and convert to char */
  max_exp = (unsigned char)((val_exp.u & LIBXSMM_MASK_ABS_F32) >> LIBXSMM_MANT_SZ_F32);

  return max_exp;
}


LIBXSMM_API_INLINE short libxsmm_internal_quantize_scalar_no_scf( float input, unsigned char max_exp, unsigned char add_shift, int round_mode ) {
  libxsmm_float_uint value;
  unsigned int qvalue = 0;
  unsigned int mant = 0;
  unsigned int sign = 0;
  unsigned char rhs = 0;
  unsigned char exp_off = 0;

  /* init libxsmm */
  LIBXSMM_INIT

  /* in case of zero we don't need to do anything */
  if (LIBXSMM_FEQ(input, 0)) {
    qvalue = 0;
  } else {
    /* let's get a float copy to work on */
    /* vinp = LIBXSMM_INTRINSICS_MM512_LOAD_PS( in_buffer[i] ); */
    value.f = input;
    /* let's compute the offset of the current exp at pos i from max offset, we need to mask the sign bit though */
    /*__m512i vexp     = _mm512_cvtps_epi32(_mm512_getexp_ps (vinp));
      __m512i vexp_off = _mm512_sub_epi32(maxexpf, vexp);*/
    exp_off = (unsigned char)(max_exp - ((value.u & LIBXSMM_MASK_ABS_F32) >> LIBXSMM_MANT_SZ_F32));
    /* cut out mantissa and set leading bit */
    /*__m512i mmask = _mm512_set1_epi32(LIBXSMM_MASK_MANT_F32);
      __m512i vmant = _mm512_or_epi32(_mm512_set1_epi32(0x1 << LIBXSMM_MANT_SZ_F32), _mm512_and_epi32( _mm512_castps_si512( vinp ), mmask));*/
    mant = ((0x1 << LIBXSMM_MANT_SZ_F32) | (value.u & LIBXSMM_MASK_MANT_F32));
    /* extract sign */
    /* __mmask16 smask =  _mm512_cmplt_ps_mask (inp, _mm512_set1_ps(0)); */
    sign = ((value.u & LIBXSMM_MASK_SIGN_F32) >> (LIBXSMM_SZ_F32-1));
    /* calculate rhs, be aware of the now explicit leading bit, @TODO add DFP8/4 */
    rhs = (unsigned char)((LIBXSMM_MANT_SZ_F32+1) - LIBXSMM_MANT_DFP16 + exp_off + add_shift);
    /* some safety, to generate 0 when we fall off quant region, @TODO issue a LIBXSMM WARNING: that we shifted out the entire mantissa */
    if (rhs > (LIBXSMM_MANT_SZ_F32+1)) {
      rhs = (LIBXSMM_MANT_SZ_F32+1);
    }
    /* finally shift the value into the region we need, this is now a 15-add_rhs bit number for the max value in in_buffer */
    qvalue = (mant >> rhs);
    /* handle sign, 2 complement */
    if ( (sign > 0) && (qvalue > 0) ) {
      qvalue = (~qvalue + 1);
    }

    if (round_mode == LIBXSMM_QUANT_BIAS_ROUND) {
      /* biased rounding towards next bigger number */
      /* first let's determine in the original number if we need a bias rounding, @TODO need fix for F64 */
      int bias_needed = (mant & (0x3 << (rhs-2)));
      /* apply bias */
      if (bias_needed > 0) {
        qvalue++;
      }
    } else if (round_mode == LIBXSMM_QUANT_NEAREST_ROUND) {
      int nearest_needed = (mant & (0x1 << (rhs-1)));
      /* apply rounding */
      if ((nearest_needed > 0) && (rhs > 1)) {
        qvalue++;
      }
    } else if (round_mode == LIBXSMM_QUANT_STOCH_ROUND) {
      /* stochastic rounding, as implemented in the IBM paper from 2015, @TODO, fix F64 and DFP8 */
      const float eps = LIBXSMM_RES_DFP16;
      /* coverity[dont_call] */
      const float r = (float)rand();
      libxsmm_float_uint fvalue;
      float p, q;
      /* masking all bits which will be shifted out */
      fvalue.u = value.u & ((LIBXSMM_MASK_FULL_F32) << rhs);
      /* drawing a random number */
      p = r/((float)RAND_MAX);
      q = (input - fvalue.f)/eps;
      /* apply rounding if needed */
      if ((p + q) > 0.5f) {
        ++qvalue;
      }
    } else {
      /* do nothing about rounding, just chop */
    }
  }

  return (short)qvalue;
}


/* @TODO make this routine aware of any int type */
LIBXSMM_API void libxsmm_quantize_i16( float* in_buffer, short* out_buffer, int length, unsigned char add_shift, unsigned char* scf, int round_mode ) {
  int i = 0;

  /* init libxsmm */
  LIBXSMM_INIT

  /* in case we are using FP-Mul based quantization we use a different path for now
     @TODO let's unify the paths by using the similar vectorization for both */
  if ( round_mode == LIBXSMM_QUANT_FPHW_ROUND ) {
    const float max_value = libxsmm_internal_get_max( in_buffer, length );
    int maxexp = 0;
    /* take return value of LIBXSMM_FREXPF to mute static analysis issue */
    float scfq = LIBXSMM_FREXPF(max_value, &maxexp);
    maxexp -= (15/*LIBXSMM_MANT_DFP16?*/ - add_shift);
    scfq = libxsmm_sexp2_i8i(-maxexp);

#if (LIBXSMM_X86_AVX512 <= LIBXSMM_STATIC_TARGET_ARCH)
    if ( length % 16 == 0 ) {
      __m512 vscfq = _mm512_set1_ps(scfq);
#ifdef _OPENMP
#     pragma omp parallel for private(i)
#endif
      for (i = 0; i < length; i+=16 ) {
        _mm256_stream_si256( (__m256i *)&(out_buffer[i]), LIBXSMM_INTRINSICS_MM512_QUANTIZE_NEAR_PS_EPI16( &(in_buffer[i]), vscfq ) );
      }
    } else {
#endif
#ifdef _OPENMP
#     pragma omp parallel for private(i)
#endif
      for (i = 0; i < length; ++i ) {
        out_buffer[i] = (short)LIBXSMM_ROUNDF(in_buffer[i] * scfq);
      }
#if (LIBXSMM_X86_AVX512 <= LIBXSMM_STATIC_TARGET_ARCH)
    }
#endif
    /* @TODO, we need to potentially fix this unsigned char problem */
#if !defined(NDEBUG) /* library code is expected to be mute */
    if (maxexp > 0) {
      fprintf(stderr, "error quant fil\n");
    }
#endif
    *scf = (unsigned char)(-maxexp);
  } else {
    /* get max exponent */
    unsigned char max_exp = libxsmm_internal_get_max_exp( in_buffer, length );

    /* if we go for stochastic rounding, let's initialize random seed */
    if ( round_mode == LIBXSMM_QUANT_STOCH_ROUND ) {
      srand(libxsmm_timer_tick() % ((unsigned int)-1));
    }

#ifdef _OPENMP
#   pragma omp parallel for private(i)
#endif
    for (i = 0; i < length; ++i ) {
      out_buffer[i] = libxsmm_internal_quantize_scalar_no_scf( in_buffer[i], max_exp, add_shift, round_mode );
    }

    *scf = (unsigned char)(14 - add_shift - (max_exp - 127));
  }
}


LIBXSMM_API void libxsmm_dequantize_i16( short* in_buffer, float* out_buffer, int length, unsigned char scf ) {
  const float val_exp = libxsmm_sexp2_i8i(-scf);
  int i = 0;

#ifdef _OPENMP
# pragma omp parallel for private(i)
#endif
  for ( i = 0; i < length; ++i ) {
    out_buffer[i] = ((float)in_buffer[i])*val_exp;
  }
}

LIBXSMM_API void libxsmm_truncate_convert_f32_bf16(const float* in, libxsmm_bfloat16* out, unsigned int length) {
  unsigned int i = 0;

  /* truncate buffer to bf16 */
  for ( i = 0; i < length; ++i ) {
    libxsmm_bfloat16_f32 t;

    t.f = in[i];
    out[i] = t.i[1];
  }
}

LIBXSMM_API void libxsmm_rnaz_convert_fp32_bf16(const float* in, libxsmm_bfloat16* out, unsigned int len) {
  unsigned int i = 0;

  /* truncate buffer to bf16 */
  for ( i = 0; i < len; ++i ) {
    unsigned int int_round = 0;
    unsigned int do_round = 1;

    int_round = *((unsigned int*)&(in[i]));

    /* we don't round NaN and inf */
    if ( (int_round & 0x7f800000) == 0x7f800000 ) {
      do_round = 0;
    }

    /* perform round nearest tie away from zero */
    if ( do_round != 0 ) {
      int_round = int_round + 0x00008000;
    }

    /* create the bf16 value by shifting out the lower 16bits */
    int_round = int_round >> 16;

    out[i] = (libxsmm_bfloat16)int_round;
  }
}

LIBXSMM_API void libxsmm_rne_convert_fp32_bf16(const float* in, libxsmm_bfloat16* out, unsigned int len) {
  unsigned int i = 0;

  /* truncate buffer to bf16 */
  for ( i = 0; i < len; ++i ) {
    unsigned int int_round = 0;
    unsigned int do_round = 1;

    int_round = *((unsigned int*)&(in[i]));

    /* we don't round NaN and inf */
    if ( (int_round & 0x7f800000) == 0x7f800000 ) {
      do_round = 0;
    }

    /* perform round nearest tie even */
    if ( do_round != 0 ) {
      unsigned int fixup = (int_round >> 16) & 1;
      int_round = int_round + 0x00007fff + fixup;
    }

    /* create the bf16 value by shifting out the lower 16bits */
    int_round = int_round >> 16;

    out[i] = (unsigned short)int_round;
  }
}

LIBXSMM_API void libxsmm_convert_bf16_f32(const libxsmm_bfloat16* in, float* out, unsigned int length) {
  unsigned int i = 0;

  /* up-convert is super simple */
  for ( i = 0; i < length; ++i ) {
    libxsmm_bfloat16_f32 t;

    t.i[1] = in[i];
    t.i[0] = 0;
    out[i] = t.f;
  }
}

LIBXSMM_API float libxsmm_convert_f16_to_f32( libxsmm_float16 in ) {
  unsigned int f32_bias = 127;
  unsigned int f16_bias = 15;
  unsigned int s = ( in & 0x8000 ) << 16;
  unsigned int e = ( in & 0x7c00 ) >> 10;
  unsigned int m = ( in & 0x03ff );
  unsigned int e_norm = e + (f32_bias - f16_bias);
  libxsmm_float_uint res;

  /* convert denormal fp16 number into a normal fp32 number */
  if ( (e == 0) && (m != 0) ) {
    unsigned int lz_cnt = 9;
    lz_cnt = ( m >   0x1 ) ? 8 : lz_cnt;
    lz_cnt = ( m >   0x3 ) ? 7 : lz_cnt;
    lz_cnt = ( m >   0x7 ) ? 6 : lz_cnt;
    lz_cnt = ( m >   0xf ) ? 5 : lz_cnt;
    lz_cnt = ( m >  0x1f ) ? 4 : lz_cnt;
    lz_cnt = ( m >  0x3f ) ? 3 : lz_cnt;
    lz_cnt = ( m >  0x7f ) ? 2 : lz_cnt;
    lz_cnt = ( m >  0xff ) ? 1 : lz_cnt;
    lz_cnt = ( m > 0x1ff ) ? 0 : lz_cnt;
    e_norm -= lz_cnt;
    m = (m << (lz_cnt+1)) & 0x03ff;
  } else if ( (e == 0) && (m == 0) ) {
    e_norm = 0;
  } else if ( e == 0x1f ) {
    e_norm = 0xff;
    m |= ( m == 0 ) ? 0 : 0x0200; /* making first mantissa bit 1 */
  }

  /* set result to 0 */
  res.u = 0x0;
  /* set exp and mant */
  res.u |= (e_norm << 23);
  res.u |= (m << 13);
  /* sign it */
  res.u |= s;

  return res.f;
}

LIBXSMM_API libxsmm_float16 libxsmm_convert_f32_to_f16( float in ) {
  unsigned int f32_bias = 127;
  unsigned int f16_bias = 15;
  libxsmm_float_uint hybrid_in;
  libxsmm_float16 res = 0;
  unsigned int s, e, m, e_f32, m_f32;
  unsigned int fixup;
  hybrid_in.f = in;

  /* DAZ */
  hybrid_in.u = ( (hybrid_in.u & 0x7f800000) == 0x0 ) ? ( hybrid_in.u & 0x80000000 ) : ( hybrid_in.u & 0xffffffff );

  s = ( hybrid_in.u & 0x80000000 ) >> 16;
  e_f32 = ( hybrid_in.u & 0x7f800000 ) >> 23;
  m_f32 = ( hybrid_in.u & 0x007fffff );

  /* special value */
  if ( e_f32 == 0xff ) {
    e = 0x1f;
    m = (m_f32 == 0) ? 0 : (m_f32 >> 13) | 0x200;
  /* overflow */
  } else if ( e_f32 > (f32_bias + f16_bias) ) {
    e = 0x1f;
    m = 0x0;
  /* smaller than denormal f16 */
  } else if ( e_f32 < f32_bias - f16_bias - 10 ) {
    e = 0x0;
    m = 0x0;
  /* denormal */
  } else if ( e_f32 <= f32_bias - f16_bias ) {
    /* RNE */
#if 1
    /* denormalized mantissa */
    m = m_f32 | 0x00800000;
    /* addtionally subnormal shift */
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
    m = (m >> ((f32_bias - f16_bias) + 2 - e_f32)) + ((m >> ((f32_bias - f16_bias) + 1 - e_f32)) & 1) ;
    e = 0x0;
#endif
  /* normal */
  } else {
#if 1
    /* RNE round */
    fixup = (m_f32 >> 13) & 0x1;
    hybrid_in.u = hybrid_in.u + 0x000000fff + fixup;
    e = ( hybrid_in.u & 0x7f800000 ) >> 23;
    m = ( hybrid_in.u & 0x007fffff );
    e -= (f32_bias - f16_bias);
    m = m >> 13;
#else
    /* RAZ */
    hybrid_in.u = hybrid_in.u + 0x00001000;
    e = ( hybrid_in.u & 0x7f800000 ) >> 23;
    m = ( hybrid_in.u & 0x007fffff );
    e -= (f32_bias - f16_bias);
    m = m >> 13;
#endif
  }

  /* set result to 0 */
  res = 0x0;
  /* set exp and mant */
  res |= e << 10;
  res |= m;
  /* sign it */
  res |= s;

  return res;
}

LIBXSMM_API libxsmm_bfloat8 libxsmm_convert_f32_to_bf8( float in ) {
  unsigned int f32_bias = 127;
  unsigned int bf8_bias = 15;
  libxsmm_float_uint hybrid_in;
  unsigned char res = 0;
  unsigned int s, e, m, e_f32, m_f32;
  unsigned int fixup;
  hybrid_in.f = in;

  /* DAZ */
  hybrid_in.u = ( (hybrid_in.u & 0x7f800000) == 0x0 ) ? ( hybrid_in.u & 0x80000000 ) : ( hybrid_in.u & 0xffffffff );

  s = ( hybrid_in.u & 0x80000000 ) >> 24;
  e_f32 = ( hybrid_in.u & 0x7f800000 ) >> 23;
  m_f32 = ( hybrid_in.u & 0x007fffff );

  /* special value */
  if ( e_f32 == 0xff ) {
    e = 0x1f;
    m = (m_f32 == 0) ? 0 : (m_f32 >> 21) | 0x20;
  /* overflow */
  } else if ( e_f32 > (f32_bias + bf8_bias) ) {
    e = 0x1f;
    m = 0x0;
  /* smaller than denormal f16 */
  } else if ( e_f32 < f32_bias - bf8_bias - 2 ) {
    e = 0x0;
    m = 0x0;
  /* denormal */
  } else if ( e_f32 <= f32_bias - bf8_bias ) {
    /* RNE */
    /* denormalized mantissa */
    m = m_f32 | 0x00800000;
    /* addtionally subnormal shift */
    m = m >> ((f32_bias - bf8_bias) + 1 - e_f32);
    /* preserve sticky bit (some sticky bits are lost when denormalizing) */
    m |= (((m_f32 & 0x1fff) + 0x1fff) >> 21);
    /* RNE Round */
    fixup = (m >> 21) & 0x1;
    m = m + 0x000fffff + fixup;
    m = m >> 21;
    e = 0x0;
  /* normal */
  } else {
    /* RNE round */
    fixup = (m_f32 >> 21) & 0x1;
    hybrid_in.u = hybrid_in.u + 0x000fffff + fixup;
    e = ( hybrid_in.u & 0x7f800000 ) >> 23;
    m = ( hybrid_in.u & 0x007fffff );
    e -= (f32_bias - bf8_bias);
    m = m >> 21;
  }

  /* set result to 0 */
  res = 0x0;
  /* set exp and mant */
  res |= e << 2;
  res |= m;
  /* sign it */
  res |= s;

  return res;
}

LIBXSMM_API void libxsmm_rne_convert_fp32_bf8(const float* in, libxsmm_bfloat8* out, unsigned int length) {
  unsigned int i = 0;

  /* truncate buffer to bf8 */
  for ( i = 0; i < length; ++i ) {
#if 10
    out[i] = libxsmm_convert_f32_to_bf8( in[i] );
#else
    unsigned short short_round = 0;
    unsigned int do_round = 1;

    short_round = libxsmm_convert_f32_to_f16( in[i] );

    /* we don't round NaN and inf */
    if ( (short_round & 0x7c00) == 0x7c00 ) {
      do_round = 0;
    }

    /* perform round nearest tie even */
    if ( do_round != 0 ) {
      unsigned short fixup = (short_round >> 8) & 1;
      short_round = short_round + 0x007f + fixup;
    }

    /* create the bf8 value by shifting out the lower 16bits */
    short_round = short_round >> 8;

    out[i] = (unsigned char)short_round;
#endif
  }
}

LIBXSMM_API void libxsmm_convert_bf8_f32(const libxsmm_bfloat8* in, float* out, unsigned int length) {
  unsigned int i = 0;

  for ( i = 0; i < length; ++i ) {
    unsigned short tmp = ((unsigned short)in[i]) << 8;
    out[i] = libxsmm_convert_f16_to_f32( tmp );
  }
}

LIBXSMM_API void libxsmm_stochastic_convert_fp32_bf8(const float* in, libxsmm_bfloat8* out, unsigned int len) {
  unsigned int i = 0;

  unsigned short rand = (unsigned short)libxsmm_rng_u32(1);
  /* truncate buffer to bf8 */
  for ( i = 0; i < len; ++i ) {
    unsigned short short_round = 0;
    unsigned int do_round = 1;

    short_round = libxsmm_convert_f32_to_f16( in[i] );

    /* we don't round NaN and inf */
    if ( (short_round & 0x7c00) == 0x7c00 ) {
      do_round = 0;
    }
    /* perform round nearest tie even */
    if ( do_round != 0) {
      if ( (short_round & 0x7c00) == 0x0000) {
        unsigned short fixup = (short_round >> 8) & 1;
        short_round = short_round + 0x007f + fixup;
      } else {
        short_round = short_round + (rand & 0x00ff);
      }
    }
    /* create the bf8 value by shifting out the lower 16bits */
    short_round = short_round >> 8;
    out[i] = (unsigned char)short_round;
  }
}

