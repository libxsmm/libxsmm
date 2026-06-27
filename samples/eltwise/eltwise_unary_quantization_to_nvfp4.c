/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Test driver for BF16 -> NVFP4 (E2M1, blocksize=16, E4M3 scale) unary
 * quantization TPP.
******************************************************************************/
#include "eltwise_common.h"
#include <math.h>

/* ========================================================================== */
/* Inline golden-reference: FP32 -> NVFP4 (E2M1, blocksize=16, E4M3 scale)    */
/* ========================================================================== */

typedef union { float f; unsigned int u; } gold_f32u;

/*
 * Quantize |x| to a 3-bit unsigned E2M1 code (0x0..0x7) using
 * round-to-nearest-even (RNE)
 */
static unsigned char gold_encode_e2m1_abs(float absval)
{
  if (absval != absval) return 0x7;       /* NaN  -> saturate              */
  if (absval >  5.0f)   return 0x7;       /* (5.0, inf] -> sat to 6.0      */
  if (absval >= 3.5f)   return 0x6;       /* [3.5, 5.0] -> 4.0  (RNE tie)  */
  if (absval >  2.5f)   return 0x5;       /* (2.5, 3.5) -> 3.0             */
  if (absval >= 1.75f)  return 0x4;       /* [1.75, 2.5] -> 2.0 (RNE tie)  */
  if (absval >  1.25f)  return 0x3;       /* (1.25, 1.75) -> 1.5           */
  if (absval >= 0.75f)  return 0x2;       /* [0.75, 1.25] -> 1.0 (RNE tie) */
  if (absval >  0.25f)  return 0x1;       /* (0.25, 0.75) -> 0.5           */
  return 0x0;                             /* [0.0, 0.25]  -> 0.0 (RNE tie) */
}

/*
 * Convert an E4M3 (HF8) byte to float.
 * E4M3 layout: 1 sign, 4 exponent (bias 7), 3 mantissa bits.
 * Special: exponent=0xF mantissa!=0 => NaN, exponent=0xF mantissa=0 => max normal (no Inf).
 */
static float gold_hf8_to_float(unsigned char hf8)
{
  unsigned int sign = (hf8 >> 7) & 1u;
  unsigned int exp  = (hf8 >> 3) & 0xFu;
  unsigned int mant = hf8 & 0x7u;
  gold_f32u r;

  if (exp == 0 && mant == 0) {
    r.u = sign << 31;
    return r.f;
  }
  if (exp == 0) {
    /* subnormal: value = (-1)^s * 2^(1-7) * (0.mant) = (-1)^s * 2^(-6) * mant/8 */
    float val = (float)mant / 8.0f * (1.0f / 64.0f);
    return sign ? -val : val;
  }
  if (exp == 0xF && mant != 0) {
    /* NaN */
    r.u = 0x7FC00000u | (sign << 31);
    return r.f;
  }
  /* Normal (including exp=0xF, mant=0 which is max normal 448, not inf) */
  {
    int unbiased = (int)exp - 7;
    float val = (1.0f + (float)mant / 8.0f);
    float result = val;
    if (unbiased >= 0) {
      result = val * (float)(1 << unbiased);
    } else {
      result = val / (float)(1 << (-unbiased));
    }
    return sign ? -result : result;
  }
}

/*
 * Convert float to E4M3 (HF8) byte using round-to-nearest-even.
 * E4M3: bias=7, no infinity, exp=0xF mant=0 is max normal (448).
 */
static unsigned char gold_float_to_hf8(float val)
{
  gold_f32u u;
  unsigned int sign, f_exp, f_mant;
  int unbiased;
  unsigned int e4m3_exp, e4m3_mant;

  u.f = val;
  sign = (u.u >> 31) & 1u;
  f_exp  = (u.u >> 23) & 0xFFu;
  f_mant = u.u & 0x7FFFFFu;

  /* NaN */
  if (f_exp == 0xFF && f_mant != 0) {
    return (unsigned char)((sign << 7) | 0x7F);  /* E4M3 NaN */
  }
  /* Inf or too large -> clamp to max normal (448) */
  if (f_exp == 0xFF || LIBXSMM_FABSF(val) > 448.0f) {
    return (unsigned char)((sign << 7) | 0x78);  /* exp=0xF, mant=0 => 448 */
  }
  /* Zero */
  if (f_exp == 0 && f_mant == 0) {
    return (unsigned char)(sign << 7);
  }

  /* Get unbiased exponent */
  if (f_exp == 0) {
    /* FP32 subnormal - very tiny, flush to zero for E4M3 */
    return (unsigned char)(sign << 7);
  }
  unbiased = (int)f_exp - 127;

  /* E4M3 range: exp 0x1..0xE => unbiased -6..+7, exp 0xF mant=0 => +8 (special max) */
  if (unbiased > 8) {
    return (unsigned char)((sign << 7) | 0x78); /* clamp to 448 */
  }
  if (unbiased < -9) {
    return (unsigned char)(sign << 7); /* too small, flush to zero */
  }

  /* Normal E4M3 encoding with RNE */
  if (unbiased >= -6) {
    e4m3_exp = (unsigned int)(unbiased + 7);
    /* Round 23-bit mantissa to 3 bits with RNE */
    {
      unsigned int round_bit = (f_mant >> 19) & 1u;
      unsigned int sticky = (f_mant & 0x7FFFFu) ? 1u : 0u;
      unsigned int trunc_mant = f_mant >> 20;
      if (round_bit && (sticky || (trunc_mant & 1u))) {
        trunc_mant++;
      }
      if (trunc_mant >= 8) {
        trunc_mant = 0;
        e4m3_exp++;
      }
      e4m3_mant = trunc_mant;
    }
    /* Check for overflow into special max */
    if (e4m3_exp >= 0xF) {
      if (e4m3_exp == 0xF && e4m3_mant == 0) {
        return (unsigned char)((sign << 7) | 0x78); /* 448 */
      }
      return (unsigned char)((sign << 7) | 0x78); /* clamp */
    }
    return (unsigned char)((sign << 7) | (e4m3_exp << 3) | e4m3_mant);
  } else {
    /* Subnormal E4M3: unbiased < -6, so e4m3_exp = 0 */
    /* value = 2^(-6) * mant/8 */
    int shift = -6 - unbiased; /* how many extra right-shifts */
    unsigned int full_mant = (1u << 3) | ((f_mant >> 20) & 0x7u); /* 1.mmm in 4 bits */
    unsigned int round_bit, sticky, trunc_mant;
    if (shift >= 4) {
      return (unsigned char)(sign << 7); /* too small */
    }
    /* Shift right by 'shift' with RNE */
    trunc_mant = full_mant >> shift;
    round_bit = (full_mant >> (shift - 1)) & 1u;
    sticky = (full_mant & ((1u << (shift - 1)) - 1u)) ? 1u : 0u;
    /* Also include original sticky bits from fp32 mantissa */
    if (f_mant & 0xFFFFFu) sticky = 1u;
    if (round_bit && (sticky || (trunc_mant & 1u))) {
      trunc_mant++;
    }
    if (trunc_mant >= 8) {
      /* Promoted to normal */
      return (unsigned char)((sign << 7) | (1u << 3) | 0u);
    }
    return (unsigned char)((sign << 7) | (trunc_mant & 0x7u));
  }
}

/*
 * Convert a block of 16 FP32 values to NVFP4 (E2M1, blocksize=16).
 *
 *   in[16]       - 16 float inputs
 *   out_data[8]  - packed E2M1 nibbles: out_data[i] = (elem[2i+1] << 4) | elem[2i]
 *   *out_scale   - E4M3 (HF8) shared scale byte
 */
static void gold_fp32_to_nvfp4_block(
  const float      *in,
  unsigned char    *out_data,
  unsigned char    *out_scale)
{
  float amax = 0.0f;
  int i;
  float scale_f;
  unsigned char scale_hf8;

  /* 1. Max absolute value */
  for (i = 0; i < 16; i++) {
    float a = LIBXSMM_FABSF(in[i]);
    if (a > amax || a != a) amax = a;
  }

  /* 2. Compute E4M3 scale: scale = amax / 6.0 (6.0 is max representable E2M1 value)
   * Then quantize scale to E4M3, and reconstruct the float scale from the E4M3 byte. */
  if (amax == 0.0f) {
    scale_hf8 = 0;
    scale_f = 0.0f;
  } else {
    /* Use BF16 math: multiply amax by reciprocal of 6 in BF16 (0x3E2A) */
    libxsmm_bfloat16_f32 rcp6;
    libxsmm_bfloat16 amax_bf16;
    float raw_scale;
    rcp6.i[0] = 0;
    rcp6.i[1] = 0x3E2A; /* 1/6 in BF16 */
    /* Convert amax to BF16 using RNE */
    libxsmm_rne_convert_fp32_bf16(&amax, &amax_bf16, 1);
    { libxsmm_bfloat16_f32 t; t.i[0] = 0; t.i[1] = amax_bf16; amax = t.f; }
    /* Multiply in BF16 precision: convert product back to BF16 using RNE */
    raw_scale = amax * rcp6.f;
    {
      libxsmm_bfloat16 raw_scale_bf16;
      libxsmm_rne_convert_fp32_bf16(&raw_scale, &raw_scale_bf16, 1);
      { libxsmm_bfloat16_f32 t; t.i[0] = 0; t.i[1] = raw_scale_bf16; raw_scale = t.f; }
    }
    scale_hf8 = gold_float_to_hf8(raw_scale);
    scale_f = gold_hf8_to_float(scale_hf8);
  }

  *out_scale = scale_hf8;

  /* 3. If scale is zero, all outputs are zero */
  if (scale_f == 0.0f) {
    memset(out_data, 0, 8);
    return;
  }

  /* 4. Compute BF16 reciprocal of scale, multiply each element, encode E2M1 */
  {
    libxsmm_bfloat16 scale_bf16;
    float scale_bf16_f, rcp_f;
    libxsmm_bfloat16 rcp_bf16;

    /* Convert scale_f to BF16 using RNE */
    libxsmm_rne_convert_fp32_bf16(&scale_f, &scale_bf16, 1);
    { libxsmm_bfloat16_f32 t; t.i[0] = 0; t.i[1] = scale_bf16; scale_bf16_f = t.f; }
    /* Compute reciprocal and convert to BF16 using RNE */
    rcp_f = 1.0f / scale_bf16_f;
    libxsmm_rne_convert_fp32_bf16(&rcp_f, &rcp_bf16, 1);
    { libxsmm_bfloat16_f32 t; t.i[0] = 0; t.i[1] = rcp_bf16; rcp_f = t.f; }

    for (i = 0; i < 8; i++) {
      float v0 = in[2*i]     * rcp_f;
      float v1 = in[2*i + 1] * rcp_f;
      libxsmm_bfloat16 bf16_v0, bf16_v1;
      gold_f32u u0, u1;
      unsigned char s0, s1, lo, hi;
      /* Convert products to BF16 using RNE */
      libxsmm_rne_convert_fp32_bf16(&v0, &bf16_v0, 1);
      libxsmm_rne_convert_fp32_bf16(&v1, &bf16_v1, 1);
      { libxsmm_bfloat16_f32 t0, t1;
        t0.i[0] = 0; t0.i[1] = bf16_v0; v0 = t0.f;
        t1.i[0] = 0; t1.i[1] = bf16_v1; v1 = t1.f;
      }
      u0.f = in[2*i];     s0 = (u0.u >> 31) ? 0x8u : 0u;
      u1.f = in[2*i + 1]; s1 = (u1.u >> 31) ? 0x8u : 0u;
      lo = s0 | gold_encode_e2m1_abs(LIBXSMM_FABSF(v0));
      hi = s1 | gold_encode_e2m1_abs(LIBXSMM_FABSF(v1));
      out_data[i] = (unsigned char)((hi << 4) | lo);
    }
  }
}

unsigned int is_reference_kernel = 0;
libxsmm_kernel_info info;

LIBXSMM_INLINE
int test_bf16_to_nvfp4( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  libxsmm_bfloat16 *in_bf16;
  float *in_f32;                        /* fp32 copy for the golden reference */
  unsigned char *nvfp4_data;            /* packed NVFP4X2 output from TPP     */
  unsigned char *nvfp4_data_gold;       /* packed NVFP4X2 output from ref     */
  unsigned char *scales;                /* E4M3 (HF8) scales from TPP  (M/16 x N)  */
  unsigned char *scales_gold;           /* E4M3 (HF8) scales from ref               */
  libxsmm_blasint i, j, b;
  unsigned int s;
  int ret = EXIT_SUCCESS;
  libxsmm_blasint num_blocks_m = M / 16; /* number of 16-element blocks per row */
  libxsmm_blasint ldo_scales = ldo / 16;  /* scales row stride matches kernel layout (ldo/block); only the first num_blocks_m blocks per row are valid */
  libxsmm_meltwfunction_unary unary_kernel_quant;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_shape unary_shape;

  if ( M > ldi ) {
    fprintf( stderr, "test_bf16_to_nvfp4: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if ( M % 16 != 0 ) {
    fprintf( stderr, "test_bf16_to_nvfp4: M must be a multiple of 16 (NVFP4 block size)\n");
    exit(-1);
  }
  if ( (M == 0) || (N == 0) ) {
    return ret;
  }

  if ( ldo < M ) {
    fprintf( stderr, "test_bf16_to_nvfp4: ldo must be >= M \n");
    exit(-1);
  }

  in_bf16         = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16) * N * ldi, 64 );
  in_f32          = (float*)            libxsmm_aligned_malloc( sizeof(float) * N * ldi, 64 );
  nvfp4_data      = (unsigned char*)    libxsmm_aligned_malloc( sizeof(unsigned char) * N * (ldo / 2), 64 );
  nvfp4_data_gold = (unsigned char*)    libxsmm_aligned_malloc( sizeof(unsigned char) * N * (ldo / 2), 64 );
  scales          = (unsigned char*)    libxsmm_aligned_malloc( sizeof(unsigned char) * N * ldo_scales, 64 );
  scales_gold     = (unsigned char*)    libxsmm_aligned_malloc( sizeof(unsigned char) * N * ldo_scales, 64 );

  /* Initialise input as BF16 random values and create FP32 mirror */
  init_random_matrix( LIBXSMM_DATATYPE_BF16, in_bf16, 1, ldi, N, 1 );
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      libxsmm_bfloat16_f32 tmp;
      tmp.i[0] = 0;
      tmp.i[1] = in_bf16[(i * ldi) + j];
      in_f32[(i * ldi) + j] = tmp.f;
    }
  }

  {
  libxsmm_blasint ldo_bytes = ldo / 2;  /* NVFP4X2: 2 values per byte, ldo is in elements */

  /* Zero outputs */
  memset( nvfp4_data,      0, sizeof(unsigned char) * N * ldo_bytes );
  memset( nvfp4_data_gold, 0, sizeof(unsigned char) * N * ldo_bytes );
  memset( scales,          0, sizeof(unsigned char) * N * ldo_scales );
  memset( scales_gold,     0, sizeof(unsigned char) * N * ldo_scales );

  /* --- Golden reference --- */
  for ( i = 0; i < N; ++i ) {
    for ( b = 0; b < num_blocks_m; ++b ) {
      float block_f32[16];
      unsigned char block_out[8];
      unsigned char block_scale;
      /* Gather the 16-element block from the row */
      for ( j = 0; j < 16; ++j ) {
        block_f32[j] = in_f32[(i * ldi) + b * 16 + j];
      }
      gold_fp32_to_nvfp4_block( block_f32, block_out, &block_scale );
      /* Store packed nibbles: 8 bytes per block */
      for ( j = 0; j < 8; ++j ) {
        nvfp4_data_gold[(i * ldo_bytes) + b * 8 + j] = block_out[j];
      }
      scales_gold[(i * ldo_scales) + b] = block_scale;
    }
  }

  /* --- JIT-ed TPP kernel --- */
  unary_shape.m = M;
  unary_shape.n = N;
  unary_shape.ldi = ldi;
  unary_shape.ldo = ldo;
  unary_shape.in0_type  = LIBXSMM_DATATYPE_BF16;
  unary_shape.out_type  = LIBXSMM_DATATYPE_NVFP4X2;
  unary_shape.comp_type = LIBXSMM_DATATYPE_BF16;

  unary_kernel_quant = libxsmm_dispatch_meltw_unary(
    LIBXSMM_MELTW_TYPE_UNARY_QUANT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );

  if ( unary_kernel_quant == NULL ) {
    fprintf( stderr, "JIT for BF16->NVFP4 QUANT TPP failed. Bailing...!\n" );
    exit(-1);
  }
  libxsmm_get_kernel_info( (const void*)unary_kernel_quant, &info );
  is_reference_kernel = info.is_reference_kernel;

  unary_param.in.primary    = (void*)in_bf16;
  unary_param.out.primary   = (void*)nvfp4_data;
  unary_param.out.secondary = (void*)scales;
  unary_kernel_quant( &unary_param );

  /* --- Compare packed NVFP4 data --- */
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M / 2; ++j ) {
      if ( nvfp4_data_gold[(i * ldo_bytes) + j] != nvfp4_data[(i * ldo_bytes) + j] ) {
        libxsmm_blasint elem0 = j * 2;
        libxsmm_blasint blk = elem0 / 16;
        printf( "NVFP4 data error at row=%lld, byte=%lld: gold=0x%02x, got=0x%02x\n",
                (long long)i, (long long)j,
                (unsigned int)nvfp4_data_gold[(i * ldo_bytes) + j],
                (unsigned int)nvfp4_data[(i * ldo_bytes) + j] );
        printf( "  scale_gold=0x%02x (f=%f), scale_jit=0x%02x (f=%f)\n",
                (unsigned int)scales_gold[(i * ldo_scales) + blk],
                gold_hf8_to_float(scales_gold[(i * ldo_scales) + blk]),
                (unsigned int)scales[(i * ldo_scales) + blk],
                gold_hf8_to_float(scales[(i * ldo_scales) + blk]) );
        s = 1;
      }
    }
  }
  if ( s == 0 ) {
    printf( "SUCCESS BF16 -> NVFP4X2 packed data\n" );
  } else {
    printf( "FAILURE BF16 -> NVFP4X2 packed data\n" );
    ret = EXIT_FAILURE;
  }

  /* --- Compare scales (E4M3 / HF8) --- */
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( b = 0; b < num_blocks_m; ++b ) {
      if ( scales_gold[(i * ldo_scales) + b] != scales[(i * ldo_scales) + b] ) {
        printf( "Scale error at row=%lld, block=%lld: gold=0x%02x (f=%f), got=0x%02x (f=%f)\n",
                (long long)i, (long long)b,
                (unsigned int)scales_gold[(i * ldo_scales) + b],
                gold_hf8_to_float(scales_gold[(i * ldo_scales) + b]),
                (unsigned int)scales[(i * ldo_scales) + b],
                gold_hf8_to_float(scales[(i * ldo_scales) + b]) );
        s = 1;
      }
    }
  }
  if ( s == 0 ) {
    printf( "SUCCESS BF16 -> NVFP4 scales (E4M3/HF8)\n" );
  } else {
    printf( "FAILURE BF16 -> NVFP4 scales (E4M3/HF8)\n" );
    ret = EXIT_FAILURE;
  }

  libxsmm_free( in_bf16 );
  libxsmm_free( in_f32 );
  libxsmm_free( nvfp4_data );
  libxsmm_free( nvfp4_data_gold );
  libxsmm_free( scales );
  libxsmm_free( scales_gold );
  }

  return ret;
}

int main( int argc, char* argv[] ) {
  libxsmm_blasint M;
  libxsmm_blasint N;
  libxsmm_blasint ldi;
  libxsmm_blasint ldo;
  int ret = EXIT_FAILURE;

  if ( argc != 5 ) {
    printf( " Error! Usage: %s [M] [N] [ldi] [ldo]\n", argv[0] );
    printf( "   M   - number of columns (must be multiple of 16)\n" );
    printf( "   N   - number of rows\n" );
    printf( "   ldi - leading dimension of BF16 input  (>= M)\n" );
    printf( "   ldo - leading dimension of NVFP4X2 output in bytes (>= M/2)\n" );
    exit(-1);
  }

  M   = atoi( argv[1] );
  N   = atoi( argv[2] );
  ldi = atoi( argv[3] );
  ldo = atoi( argv[4] );

  printf( "Testing BF16 -> NVFP4 quant - M=%lld, N=%lld, LDI=%lld, LDO=%lld\n",
          (long long)M, (long long)N, (long long)ldi, (long long)ldo );
  ret = test_bf16_to_nvfp4( M, N, ldi, ldo );

  ret = (ret == EXIT_SUCCESS) ? libxsmm_return_success_code( is_reference_kernel ) : ret;
  return ret;
}
