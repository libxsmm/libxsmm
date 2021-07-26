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
#include <float.h>

int test_float_to_int8_to_float( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  float *in;
  char *char_data;
  char *char_data_gold;
  float *f32_char_data;
  float *f32_char_data_gold;
  unsigned int i, j;
  unsigned int s;
  int ret = EXIT_SUCCESS;
  float max_value = FLT_MIN;
  libxsmm_meltw_unary_param unary_param;
  int maxexp = 0;
  float scf_quant = 0.0;
  float scf_dequant = 0.0;

  if ( M > ldi ) {
    fprintf( stderr, "test_float_to_int8_to_float: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }

  in                 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi, 64);
  char_data          = (char*)  libxsmm_aligned_malloc( sizeof(char)*N*ldi, 64);
  char_data_gold     = (char*)  libxsmm_aligned_malloc( sizeof(char)*N*ldi, 64);
  f32_char_data      = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi, 64);
  f32_char_data_gold = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi, 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      in[(i*ldi)+j] = (float)(((i*ldi)+j)%4096);
      max_value = ( max_value < in[(i*ldi)+j] ) ? in[(i*ldi)+j] : max_value;
    }
  }

  /* compute scaling factor */
  /* take return value of LIBXSMM_FREXPF to mute static analysis issue */
  LIBXSMM_FREXPF(max_value, &maxexp);
  maxexp -= 23;
  scf_quant = libxsmm_sexp2_i8i(-maxexp);

  /* run quantization */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      char_data_gold[(i*ldi)+j] = (char)LIBXSMM_ROUNDF( in[(i*ldi)+j] * scf_quant );
    }
  }

  /* run dequantization */
  scf_dequant = libxsmm_sexp2_i8i(maxexp);
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      f32_char_data_gold[(i*ldi)+j] = ((float)char_data_gold[(i*ldi)+j]) * scf_dequant ;
    }
  }

  /* use jited quantization */
  unary_param.in.primary  = (void*)in;
  unary_param.in.secondary  = (void*)&scf_quant;
  unary_param.out.primary = (void*)char_data;
  libxsmm_meltwfunction_unary unary_kernel_quant = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_I8, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
  if ( unary_kernel_quant == NULL ) {
    fprintf( stderr, "JIT for IDENTITY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel_quant( &unary_param );

  /* use jited quantization */
  unary_param.in.primary  = (void*)char_data;
  unary_param.in.secondary  = (void*)&scf_dequant;
  unary_param.out.primary = (void*)f32_char_data;
  libxsmm_meltwfunction_unary unary_kernel_dequant = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_I8, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
  if ( unary_kernel_dequant == NULL ) {
    fprintf( stderr, "JIT for IDENTITY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel_dequant( &unary_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < M; ++i ) {
    for ( j = 0; j < N; ++j ) {
      if ( char_data_gold[(i*ldo)+j] != char_data[(i*ldo)+j] ) {
        printf("error at possition i=%i, j=%i, %i, %i\n", i, j, char_data_gold[(i*ldo)+j], char_data[(i*ldo)+j]);
        s = 1;
      }
    }
  }
  if ( s == 0 ) {
    printf("SUCCESS quant 32bit -> int8\n");
  } else {
    printf("FAILURE quant 32bit -> int8\n");
    ret = EXIT_FAILURE;
  }
  s = 0;
  for ( i = 0; i < M; ++i ) {
    for ( j = 0; j < N; ++j ) {
      if ( f32_char_data_gold[(i*ldo)+j] != f32_char_data[(i*ldo)+j] ) {
        printf("error at possition i=%i, j=%i, %f, %f\n", i, j, f32_char_data_gold[(i*ldo)+j], f32_char_data[(i*ldo)+j]);
        s = 1;
      }
    }
  }
  if ( s == 0 ) {
    printf("SUCCESS quant int8 -> 32bit\n");
  } else {
    printf("FAILURE quant int8 -> 32bit\n");
    ret = EXIT_FAILURE;
  }

  libxsmm_free( char_data_gold );
  libxsmm_free( char_data );
  libxsmm_free( f32_char_data_gold );
  libxsmm_free( f32_char_data );
  libxsmm_free( in );

  return ret;
}

int main( int argc, char* argv[] ) {
  libxsmm_blasint dtype_one;
  libxsmm_blasint dtype_two;
  libxsmm_blasint M;
  libxsmm_blasint N;
  libxsmm_blasint ldi;
  libxsmm_blasint ldo;
  int ret = EXIT_FAILURE;

  if ( argc != 7 ) {
    printf(" Error! Usage: %s [8/4/2/1] [8/4/2/1] [M] [N] [ldi] [ldo]\n", argv[0] );
    exit(-1);
  }

  dtype_one = atoi(argv[1]);
  dtype_two = atoi(argv[2]);
  M         = atoi(argv[3]);
  N         = atoi(argv[4]);
  ldi       = atoi(argv[5]);
  ldo       = atoi(argv[6]);

  if ( dtype_one == 4 && dtype_two == 1 ) {
    printf("Testing FP32 <-> int quant\n");
    ret = test_float_to_int8_to_float( M, N, ldi, ldo );
  } else {
    printf(" Not implemented case! Usage: %s [8/4/2/1] [8/4/2/1] [M] [N] [ldi] [ldo]\n", argv[0] );
    exit(-1);
  }

  return ret;
}
