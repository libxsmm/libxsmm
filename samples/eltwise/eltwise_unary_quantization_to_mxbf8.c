/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Test driver for BF16 -> MXBF8 (E5M2, blocksize=32) unary quantization TPP.
******************************************************************************/
#include "eltwise_common.h"
#include <math.h>

typedef union { float f; unsigned int u; } gold_f32u;
/*
 * Convert a block of 32 FP32 values to MXBF8 (E5M2 / BF8).
 *
 *   in[32]       - 32 float inputs
 *   out_data[32] - BF8 (E5M2) output bytes, one per element
 *   *out_scale   - E8M0 shared scale (biased exponent byte)
 */
static void gold_fp32_to_mxbf8_block(
  const float      *in,
  unsigned char    *out_data,
  unsigned char    *out_scale)
{
  gold_f32u mx;
  float amax = 0.0f;
  int   shared_exp, scale_mant, i;
  float scale;
  unsigned int is_inf_or_nan = 0;

  /* 1. Max absolute value (propagates NaN like the MX reference) */
  for (i = 0; i < 32; i++) {
    float a = LIBXSMM_FABSF(in[i]);
    if (a > amax || a != a) amax = a;
  }

  /* 2. Shared biased exponent, offset by elem_emax = 15 (for E5M2) */
  mx.f = amax;
  shared_exp = (int)((mx.u >> 23) & 0xFFu);
  is_inf_or_nan = (shared_exp == 0xFF);
  shared_exp -= 15;
  /* saturate to max scale if any input is inf/NaN */
  if (is_inf_or_nan) shared_exp = 0xFF;
  if (shared_exp < 0) shared_exp = 0;

  *out_scale = (unsigned char)shared_exp;

  /* 3. Construct the shared scale as a float from the E8M0 exponent */
  scale_mant = (shared_exp == 0 || is_inf_or_nan > 0) ? (1 << 22) : 0;
  mx.u = ((unsigned int)shared_exp << 23) | (unsigned int)scale_mant;
  scale = mx.f;

  /* Implementation dependent when shared exp is Inf/NaN:
     All scaled entries are Max Normal BF8 (E5M2) = 0x7B */
  if (is_inf_or_nan) {
    memset(out_data, 0x7B, 32);
    return;
  }

  /* 4. Scale each element by dividing by the shared scale, convert to BF8 (E5M2) */
  for (i = 0; i < 32; i++) {
    float v = in[i] / scale;
    libxsmm_bfloat8 bf8_v;
    libxsmm_rne_convert_fp32_bf8(&v, &bf8_v, 1);
    out_data[i] = (unsigned char)bf8_v;
  }
}

unsigned int is_reference_kernel = 0;
libxsmm_kernel_info info;

LIBXSMM_INLINE
int test_bf16_to_mxbf8( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  libxsmm_bfloat16 *in_bf16;
  float *in_f32;                        /* fp32 copy for the golden reference */
  unsigned char *mxbf8_data;            /* MXBF8 output from TPP              */
  unsigned char *mxbf8_data_gold;       /* MXBF8 output from ref              */
  unsigned char *scales;                /* E8M0 scales from TPP  (M/32 x N)  */
  unsigned char *scales_gold;           /* E8M0 scales from ref               */
  libxsmm_blasint i, j, b;
  unsigned int s;
  int ret = EXIT_SUCCESS;
  libxsmm_blasint num_blocks_m = M / 32; /* number of 32-element blocks per column */
  libxsmm_blasint ldo_scales = ldo / 32;  /* scales row stride matches kernel layout (ldo/block); only the first num_blocks_m blocks per row are valid */
  libxsmm_meltwfunction_unary unary_kernel_quant;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_shape unary_shape;

  if ( M > ldi ) {
    fprintf( stderr, "test_bf16_to_mxbf8: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if ( M % 32 != 0 ) {
    fprintf( stderr, "test_bf16_to_mxbf8: M must be a multiple of 32 (MX block size)\n");
    exit(-1);
  }
  if ( (M == 0) || (N == 0) ) {
    return ret;
  }

  /* ldo for MXBF8: 1 byte per element, so ldo >= M */
  if ( ldo < M ) {
    fprintf( stderr, "test_bf16_to_mxbf8: ldo must be >= M (MXBF8 bytes per column)\n");
    exit(-1);
  }

  in_bf16         = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16) * N * ldi, 64 );
  in_f32          = (float*)            libxsmm_aligned_malloc( sizeof(float) * N * ldi, 64 );
  mxbf8_data      = (unsigned char*)    libxsmm_aligned_malloc( sizeof(unsigned char) * N * ldo, 64 );
  mxbf8_data_gold = (unsigned char*)    libxsmm_aligned_malloc( sizeof(unsigned char) * N * ldo, 64 );
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

  /* Zero outputs */
  memset( mxbf8_data,      0, sizeof(unsigned char) * N * ldo );
  memset( mxbf8_data_gold, 0, sizeof(unsigned char) * N * ldo );
  memset( scales,          0, sizeof(unsigned char) * N * ldo_scales );
  memset( scales_gold,     0, sizeof(unsigned char) * N * ldo_scales );

  /* --- Golden reference --- */
  for ( i = 0; i < N; ++i ) {
    for ( b = 0; b < num_blocks_m; ++b ) {
      float block_f32[32];
      unsigned char block_out[32];
      unsigned char block_scale;
      /* Gather the 32-element block from the column */
      for ( j = 0; j < 32; ++j ) {
        block_f32[j] = in_f32[(i * ldi) + b * 32 + j];
      }
      gold_fp32_to_mxbf8_block( block_f32, block_out, &block_scale );
      /* Store BF8 bytes: 32 bytes per block */
      for ( j = 0; j < 32; ++j ) {
        mxbf8_data_gold[(i * ldo) + b * 32 + j] = block_out[j];
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
  unary_shape.out_type  = LIBXSMM_DATATYPE_MXBF8;
  unary_shape.comp_type = LIBXSMM_DATATYPE_BF16;

  unary_kernel_quant = libxsmm_dispatch_meltw_unary(
    LIBXSMM_MELTW_TYPE_UNARY_QUANT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );

  if ( unary_kernel_quant == NULL ) {
    fprintf( stderr, "JIT for BF16->MXBF8 QUANT TPP failed. Bailing...!\n" );
    exit(-1);
  }
  libxsmm_get_kernel_info( (const void*)(uintptr_t)unary_kernel_quant, &info );
  is_reference_kernel = info.is_reference_kernel;

  unary_param.in.primary    = (void*)in_bf16;
  unary_param.out.primary   = (void*)mxbf8_data;
  unary_param.out.secondary = (void*)scales;
  unary_kernel_quant( &unary_param );

  /* --- Compare MXBF8 data --- */
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if ( mxbf8_data_gold[(i * ldo) + j] != mxbf8_data[(i * ldo) + j] ) {
        libxsmm_blasint blk = j / 32;
        printf( "MXBF8 data error at row=%lld, col=%lld: gold=0x%02x, got=0x%02x\n",
                (long long)i, (long long)j,
                (unsigned int)mxbf8_data_gold[(i * ldo) + j],
                (unsigned int)mxbf8_data[(i * ldo) + j] );
        printf( "  input BF16[%lld]=0x%04x (f32=%f), scale_gold=0x%02x, scale_jit=0x%02x\n",
                (long long)j, (unsigned int)in_bf16[(i * ldi) + j], in_f32[(i * ldi) + j],
                (unsigned int)scales_gold[(i * ldo_scales) + blk],
                (unsigned int)scales[(i * ldo_scales) + blk] );
        s = 1;
      }
    }
  }
  if ( s == 0 ) {
    printf( "SUCCESS BF16 -> MXBF8 data\n" );
  } else {
    printf( "FAILURE BF16 -> MXBF8 data\n" );
    ret = EXIT_FAILURE;
  }

  /* --- Compare scales --- */
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( b = 0; b < num_blocks_m; ++b ) {
      if ( scales_gold[(i * ldo_scales) + b] != scales[(i * ldo_scales) + b] ) {
        printf( "Scale error at row=%lld, block=%lld: gold=0x%02x, got=0x%02x\n",
                (long long)i, (long long)b,
                (unsigned int)scales_gold[(i * ldo_scales) + b],
                (unsigned int)scales[(i * ldo_scales) + b] );
        s = 1;
      }
    }
  }
  if ( s == 0 ) {
    printf( "SUCCESS BF16 -> MXBF8 scales (E8M0)\n" );
  } else {
    printf( "FAILURE BF16 -> MXBF8 scales (E8M0)\n" );
    ret = EXIT_FAILURE;
  }

  libxsmm_free( in_bf16 );
  libxsmm_free( in_f32 );
  libxsmm_free( mxbf8_data );
  libxsmm_free( mxbf8_data_gold );
  libxsmm_free( scales );
  libxsmm_free( scales_gold );

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
    printf( "   M   - number of columns (must be multiple of 32)\n" );
    printf( "   N   - number of rows\n" );
    printf( "   ldi - leading dimension of BF16 input  (>= M)\n" );
    printf( "   ldo - leading dimension of MXBF8 output in bytes (>= M)\n" );
    exit(-1);
  }

  M   = atoi( argv[1] );
  N   = atoi( argv[2] );
  ldi = atoi( argv[3] );
  ldo = atoi( argv[4] );

  printf( "Testing BF16 -> MXBF8 quant - M=%lld, N=%lld, LDI=%lld, LDO=%lld\n",
          (long long)M, (long long)N, (long long)ldi, (long long)ldo );
  ret = test_bf16_to_mxbf8( M, N, ldi, ldo );

  ret = (ret == EXIT_SUCCESS) ? libxsmm_return_success_code( is_reference_kernel ) : ret;
  return ret;
}
