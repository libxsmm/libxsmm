/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef __AVX512BW__
#include <libxsmm_intrinsics_x86.h>
#endif

#ifdef __AVX512BW__
void dropout_fwd_gold(unsigned int M, float *in, float *out, unsigned short *dropout_mask, void* rng_state, float p) {
  unsigned int i;
  float pn = 1 - p;
  __m512 vp = _mm512_set1_ps(pn);
  __m512 vpi = _mm512_set1_ps(1.0/pn);
  for (i = 0; i < M - 15; i+=16) {
    __m512 rnd = LIBXSMM_INTRINSICS_MM512_RNG_EXTSTATE_PS(rng_state);
    __m512 vin = _mm512_loadu_ps(in+i);
    __mmask16 dmsk = _mm512_cmplt_ps_mask(rnd, vp);
    __m512 vout = _mm512_maskz_mul_ps(dmsk, vin, vpi);
    _mm512_storeu_ps(out+i, vout);
    dropout_mask[i/16] = dmsk;
  }
  if (i < M) {
    int rem = M - i;
    __mmask16 mask = (1 << rem) - 1;
    __m512 rnd = LIBXSMM_INTRINSICS_MM512_RNG_EXTSTATE_PS(rng_state);
    __m512 vin = _mm512_maskz_loadu_ps(mask, in+i);
    __mmask16 dmsk = _mm512_cmplt_ps_mask(rnd, vp);
    __m512 vout = _mm512_maskz_mul_ps(dmsk, vin, vpi);
    _mm512_mask_storeu_ps(out+i, mask, vout);
    dropout_mask[i/16] = dmsk & mask;
  }
}

void dropout_bwd_gold(unsigned int M, float *in, float *out, unsigned short *dropout_mask, float p) {
  unsigned int i = 0;
  float pn = 1 - p;
  __m512 vpi = _mm512_set1_ps(1.0/pn);
  for (i = 0; i < M - 15; i+=16) {
    __m512 vin = _mm512_loadu_ps(in+i);
    __mmask16 dmsk = dropout_mask[i/16];
    __m512 vout = _mm512_maskz_mul_ps(dmsk, vin, vpi);
    _mm512_storeu_ps(out+i, vout);
  }
  if (i < M) {
    int rem = M - i;
    __mmask16 mask = (1 << rem) - 1;
    __m512 vin = _mm512_maskz_loadu_ps(mask, in+i);
    __mmask16 dmsk = dropout_mask[i/16];
    __m512 vout = _mm512_maskz_mul_ps(dmsk, vin, vpi);
    _mm512_mask_storeu_ps(out+i, mask, vout);
  }
}
#endif

void test_dropout_f32_fwd( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  float *in;
  float *out, *out_gold;
  unsigned char *mask, *mask_gold;
  unsigned int *rng_state, *rng_state_gold;
  unsigned int i, j;
  unsigned int s;
  float p = 0.3f;
  libxsmm_meltw_dropout_param dropout_param;
  libxsmm_meltw_dropout_flags dropout_flags;

  if ( M > ldi ) {
    fprintf( stderr, "test_dropout_f32_fwd: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_dropout_f32_fwd: ldo needs to be equal to or bigger than N\n");
    exit(-1);
  }

  in        = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi,   64);
  out       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo,   64);
  out_gold  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo,   64);
  mask      = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*ldo/8, 64);
  mask_gold = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*ldo/8, 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      in[(i*ldi)+j] = (double)(((i*ldi)+j)%4096);
    }
  }

  /* init out */
  for ( i = 0; i < N*ldo; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < N*ldo; ++i ) {
    out_gold[i] = 0;
  }
  for ( i = 0; i < N*ldo/8; ++i ) {
    mask[i] = 0;
  }
  for ( i = 0; i < N*ldo/8; ++i ) {
    mask_gold[i] = 0;
  }

  rng_state = libxsmm_rng_create_avx512_extstate( 555 );
  rng_state_gold = libxsmm_rng_create_avx512_extstate( 555 );

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    dropout_fwd_gold( M, &in[(i*ldi)], &out_gold[(i*ldo)], (unsigned short*)&mask_gold[(i*ldo)/8], rng_state_gold, p );
  }

  /* use jited tranpose */
  dropout_param.in_ptr  = (void*)in;
  dropout_param.out_ptr = (void*)out;
  dropout_param.mask_ptr = (void*)mask;
  dropout_param.prob_ptr = (void*)&p;
  dropout_param.rng_state = (void*)rng_state;
  dropout_flags = LIBXSMM_MELTW_FLAG_DROPOUT_FWD;
  libxsmm_meltwfunction_dropout dropout_kernel = libxsmm_dispatch_meltw_dropout(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, dropout_flags);
  if ( dropout_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
    exit(-1);
  }
  dropout_kernel( &dropout_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if ( out_gold[(i*ldo)+j] != out[(i*ldo)+j] ) {
        printf("error at possition i=%i, j=%i, %f, %f\n", i, j, out[(i*ldo)+j], out_gold[(i*ldo)+j]);
        s = 1;
      }
#if 0
      else {
        printf("correct at possition i=%i, j=%i, %f, %f\n", i, j, out[(i*ldo)+j], out_gold[(i*ldo)+j]);
      }
#endif
    }
  }
  if ( s == 0 ) {
    printf("SUCCESS output\n");
  } else {
    printf("FAILURE output\n");
  }
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M/8; ++j ) {
      if ( mask_gold[(i*ldo)+j] != mask[(i*ldo)+j] ) {
        printf("error at possition i=%i, j=%i, %u, %u\n", i, j, mask[(i*ldo)+j], mask_gold[(i*ldo)+j]);
        s = 1;
      }
#if 0
      else {
        printf("correct at possition i=%i, j=%i, %u, %u\n", i, j, mask[(i*ldo)+j], mask_gold[(i*ldo)+j]);
      }
#endif
    }
  }
  if ( s == 0 ) {
    printf("SUCCESS mask\n");
  } else {
    printf("FAILURE mask\n");
  }

  libxsmm_rng_destroy_avx512_extstate( rng_state );
  libxsmm_rng_destroy_avx512_extstate( rng_state_gold );

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
  libxsmm_free( mask );
  libxsmm_free( mask_gold );
}

int main( int argc, char* argv[] ) {
  libxsmm_blasint dtype;
  char op;
  libxsmm_blasint M;
  libxsmm_blasint N;
  libxsmm_blasint ldi;
  libxsmm_blasint ldo;

  if ( argc != 7 ) {
    printf(" Error! Usage: %s [F/B] [4/2] [M] [N] [ldi] [ldo]\n", argv[0] );
    exit(-1);
  }

  op  = *(argv[1]);
  dtype = atoi(argv[2]);
  M     = atoi(argv[3]);
  N     = atoi(argv[4]);
  ldi   = atoi(argv[5]);
  ldo   = atoi(argv[6]);

  if ( op == 'F' && dtype == 4 ) {
    printf("Testing F32 forward dropout\n");
    test_dropout_f32_fwd( M, N, ldi, ldo );
  } else {
    printf(" Not implemented case! Usage: %s [F/B] [4/2] [M] [N] [ldi] [ldo]\n", argv[0] );
    exit(-1);
  }
}
