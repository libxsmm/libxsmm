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
#include "eltwise_common.h"

#if 0
#define PRINT_VERBOSE
#endif

unsigned int is_reference_kernel = 0;
libxsmm_kernel_info info;

LIBXSMM_INLINE
int test_float_to_int8_to_float( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, unsigned int skip_scf_cvt, unsigned int signed_sat ) {
  float *in;
  char *char_data;
  char *char_data_gold;
  float *f32_char_data;
  float *f32_char_data_gold;
  libxsmm_blasint i, j;
  unsigned int s;
  int ret = EXIT_SUCCESS;
  float max_value = FLT_MIN;
  libxsmm_meltwfunction_unary unary_kernel_dequant;
  libxsmm_meltwfunction_unary unary_kernel_quant;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_shape unary_shape;
  int maxexp = 0;
  float scf_quant = 0.0;
  float scf_dequant = 0.0;

  if ( M > ldi ) {
    fprintf( stderr, "test_float_to_int8_to_float: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if ( (M == 0) || (N == 0) ) {
    return ret;
  }

  in                 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi, 64);
  char_data          = (char*)  libxsmm_aligned_malloc( sizeof(char)*N*ldo, 64);
  char_data_gold     = (char*)  libxsmm_aligned_malloc( sizeof(char)*N*ldo, 64);
  f32_char_data      = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi, 64);
  f32_char_data_gold = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi, 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      in[(i*ldi)+j] = (float)(((i*ldi)+j)%64);
      max_value = ( max_value < in[(i*ldi)+j] ) ? in[(i*ldi)+j] : max_value;
#ifdef PRINT_VERBOSE
      printf("%f ", in[(i*ldi)+j]);
#endif
    }
#ifdef PRINT_VERBOSE
    printf("\n");
#endif
  }

  /* compute scaling factor */
  /* take return value of LIBXSMM_FREXPF to mute static analysis issue */
  LIBXSMM_ELIDE_RESULT(float, LIBXSMM_FREXPF(max_value, &maxexp));
  /* devide by 128 as we want to scale into the range of -128 to 127 */
  maxexp = LIBXSMM_MAX(-128, maxexp-6);
  /* create floating point scale */
  scf_quant = libxsmm_sexp2_i8i(-maxexp);

  if (skip_scf_cvt > 0) {
    scf_quant = 1.0;
  }

  /* run quantization */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if (signed_sat > 0) {
        float tmp = LIBXSMM_NEARBYINTF( in[(i*ldi)+j] * scf_quant );
        if (tmp < -128) {
          tmp = -128.0;
        }
        if (tmp > 127) {
          tmp = 127.0;
        }
        char_data_gold[(i*ldo)+j] = (char) tmp;
      } else {
        char_data_gold[(i*ldo)+j] = (char) (0x000000ff & ((int)LIBXSMM_NEARBYINTF( in[(i*ldi)+j] * scf_quant )));
      }
    }
  }

  /* run dequantization */
  scf_dequant = libxsmm_sexp2_i8i(maxexp);

  if (skip_scf_cvt > 0) {
    scf_dequant = 1.0;
  }

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      f32_char_data_gold[(i*ldi)+j] = ((float)char_data_gold[(i*ldo)+j]) * scf_dequant ;
    }
  }

  unary_shape.m = M;
  unary_shape.n = N;
  unary_shape.ldi = ldi;
  unary_shape.ldo = ldo;
  unary_shape.in0_type = LIBXSMM_DATATYPE_F32;
  unary_shape.out_type = LIBXSMM_DATATYPE_I8;
  unary_shape.comp_type = LIBXSMM_DATATYPE_F32;

  /* use jited quantization */
  unary_param.in.primary  = (void*)in;
  if (skip_scf_cvt == 0) {
    unary_param.in.secondary  = (void*)&scf_quant;
  }
  unary_param.out.primary = (void*)char_data;
  unary_kernel_quant = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_QUANT, unary_shape, (skip_scf_cvt > 0) ? ( (signed_sat > 0) ? LIBXSMM_MELTW_FLAG_UNARY_NO_SCF_QUANT | LIBXSMM_MELTW_FLAG_UNARY_SIGN_SAT_QUANT : LIBXSMM_MELTW_FLAG_UNARY_NO_SCF_QUANT) : ((signed_sat > 0) ? LIBXSMM_MELTW_FLAG_UNARY_SIGN_SAT_QUANT : LIBXSMM_MELTW_FLAG_UNARY_NONE) );
  libxsmm_get_kernel_info((const void*) unary_kernel_quant, &info);
  is_reference_kernel = info.is_reference_kernel;
  if ( unary_kernel_quant == NULL ) {
    fprintf( stderr, "JIT for IDENTITY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel_quant( &unary_param );

  unary_shape.m = M;
  unary_shape.n = N;
  unary_shape.ldi = ldo;
  unary_shape.ldo = ldi;
  unary_shape.in0_type = LIBXSMM_DATATYPE_I8;
  unary_shape.out_type = LIBXSMM_DATATYPE_F32;
  unary_shape.comp_type = LIBXSMM_DATATYPE_F32;

  /* use jited quantization */
  unary_param.in.primary  = (void*)char_data;
  if (skip_scf_cvt == 0) {
    unary_param.in.secondary  = (void*)&scf_dequant;
  }
  unary_param.out.primary = (void*)f32_char_data;
  unary_kernel_dequant = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_DEQUANT, unary_shape, (skip_scf_cvt > 0) ? LIBXSMM_MELTW_FLAG_UNARY_NO_SCF_QUANT : LIBXSMM_MELTW_FLAG_UNARY_NONE );
  libxsmm_get_kernel_info((const void*) unary_kernel_dequant, &info);
  is_reference_kernel = info.is_reference_kernel;
  if ( unary_kernel_dequant == NULL ) {
    fprintf( stderr, "JIT for IDENTITY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel_dequant( &unary_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if ( char_data_gold[(i*ldo)+j] != char_data[(i*ldo)+j] ) {
        printf("error at position i=%i, j=%i, %i, %i\n", i, j, char_data_gold[(i*ldo)+j], char_data[(i*ldo)+j]);
        s = 1;
      }
#ifdef PRINT_VERBOSE
      printf("%i ", char_data_gold[(i*ldo)+j]);
#endif
    }
#ifdef PRINT_VERBOSE
    printf("\n");
#endif
  }
  if ( s == 0 ) {
    printf("SUCCESS unary quant FP32 -> int8\n");
  } else {
    printf("FAILURE unary quant FP32 -> int8\n");
    ret = EXIT_FAILURE;
  }
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if ( f32_char_data_gold[(i*ldi)+j] != f32_char_data[(i*ldi)+j] ) {
        printf("error at position i=%i, j=%i, %f, %f\n", i, j, f32_char_data_gold[(i*ldi)+j], f32_char_data[(i*ldi)+j]);
        s = 1;
      }
#ifdef PRINT_VERBOSE
      printf("%f ", f32_char_data_gold[(i*ldi)+j]);
#endif
    }
#ifdef PRINT_VERBOSE
    printf("\n");
#endif
  }
  if ( s == 0 ) {
    printf("SUCCESS unary quant int8 -> FP32\n");
  } else {
    printf("FAILURE unary quant int8 -> FP32\n");
    ret = EXIT_FAILURE;
  }

  libxsmm_free( char_data_gold );
  libxsmm_free( char_data );
  libxsmm_free( f32_char_data_gold );
  libxsmm_free( f32_char_data );
  libxsmm_free( in );

  return ret;
}

LIBXSMM_INLINE
int test_float_to_int16_to_float( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo,  unsigned int skip_scf_cvt, unsigned int signed_sat ) {
  float *in;
  short *short_data;
  short *short_data_gold;
  float *f32_short_data;
  float *f32_short_data_gold;
  libxsmm_blasint i, j;
  unsigned int s;
  int ret = EXIT_SUCCESS;
  float max_value = FLT_MIN;
  libxsmm_meltwfunction_unary unary_kernel_dequant;
  libxsmm_meltwfunction_unary unary_kernel_quant;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_shape unary_shape;
  int maxexp = 0;
  float scf_quant = 0.0;
  float scf_dequant = 0.0;

  if ( M > ldi ) {
    fprintf( stderr, "test_float_to_int8_to_float: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if ( (M == 0) || (N == 0) ) {
    return ret;
  }

  in                  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi, 64);
  short_data          = (short*) libxsmm_aligned_malloc( sizeof(short)*N*ldo, 64);
  short_data_gold     = (short*) libxsmm_aligned_malloc( sizeof(short)*N*ldo, 64);
  f32_short_data      = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi, 64);
  f32_short_data_gold = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi, 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      in[(i*ldi)+j] = (float)(((i*ldi)+j)%64);
      max_value = ( max_value < in[(i*ldi)+j] ) ? in[(i*ldi)+j] : max_value;
#ifdef PRINT_VERBOSE
      printf("%f ", in[(i*ldi)+j]);
#endif
    }
#ifdef PRINT_VERBOSE
    printf("\n");
#endif
  }

  /* compute scaling factor */
  /* take return value of LIBXSMM_FREXPF to mute static analysis issue */
  LIBXSMM_ELIDE_RESULT(float, LIBXSMM_FREXPF(max_value, &maxexp));
  maxexp = LIBXSMM_MAX(-128, maxexp-14);
  scf_quant = libxsmm_sexp2_i8i(-maxexp);

  if (skip_scf_cvt > 0) {
    scf_quant = 1.0;
  }

  /* run quantization */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if (signed_sat > 0) {
        float tmp = LIBXSMM_NEARBYINTF( in[(i*ldi)+j] * scf_quant );
        if (tmp < -32768) {
          tmp = -32768.0;
        }
        if (tmp > 32767) {
          tmp = 32767.0;
        }
        short_data_gold[(i*ldo)+j] = (short) tmp;
      } else {
        short_data_gold[(i*ldo)+j] = (short) ( 0x0000ffff & ((int)LIBXSMM_NEARBYINTF( in[(i*ldi)+j] * scf_quant )));
      }
    }
  }

  /* run dequantization */
  scf_dequant = libxsmm_sexp2_i8i(maxexp);

  if (skip_scf_cvt > 0) {
    scf_dequant = 1.0;
  }
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      f32_short_data_gold[(i*ldi)+j] = ((float)short_data_gold[(i*ldo)+j]) * scf_dequant ;
    }
  }

  unary_shape.m = M;
  unary_shape.n = N;
  unary_shape.ldi = ldi;
  unary_shape.ldo = ldo;
  unary_shape.in0_type = LIBXSMM_DATATYPE_F32;
  unary_shape.out_type = LIBXSMM_DATATYPE_I16;
  unary_shape.comp_type = LIBXSMM_DATATYPE_F32;

  /* use jited quantization */
  unary_param.in.primary  = (void*)in;
  if (skip_scf_cvt == 0) {
    unary_param.in.secondary  = (void*)&scf_quant;
  }
  unary_param.out.primary = (void*)short_data;
  unary_kernel_quant = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_QUANT, unary_shape, (skip_scf_cvt > 0) ? ( (signed_sat > 0) ? LIBXSMM_MELTW_FLAG_UNARY_NO_SCF_QUANT | LIBXSMM_MELTW_FLAG_UNARY_SIGN_SAT_QUANT : LIBXSMM_MELTW_FLAG_UNARY_NO_SCF_QUANT) : ((signed_sat > 0) ? LIBXSMM_MELTW_FLAG_UNARY_SIGN_SAT_QUANT : LIBXSMM_MELTW_FLAG_UNARY_NONE) );
  libxsmm_get_kernel_info((const void*) unary_kernel_quant, &info);
  is_reference_kernel = info.is_reference_kernel;
  if ( unary_kernel_quant == NULL ) {
    fprintf( stderr, "JIT for IDENTITY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel_quant( &unary_param );

  unary_shape.m = M;
  unary_shape.n = N;
  unary_shape.ldi = ldo;
  unary_shape.ldo = ldi;
  unary_shape.in0_type = LIBXSMM_DATATYPE_I16;
  unary_shape.out_type = LIBXSMM_DATATYPE_F32;
  unary_shape.comp_type = LIBXSMM_DATATYPE_F32;

  /* use jited quantization */
  unary_param.in.primary  = (void*)short_data;
  if (skip_scf_cvt == 0) {
    unary_param.in.secondary  = (void*)&scf_dequant;
  }
  unary_param.out.primary = (void*)f32_short_data;
  unary_kernel_dequant = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_DEQUANT, unary_shape,  (skip_scf_cvt > 0) ? LIBXSMM_MELTW_FLAG_UNARY_NO_SCF_QUANT : LIBXSMM_MELTW_FLAG_UNARY_NONE );
  libxsmm_get_kernel_info((const void*) unary_kernel_dequant, &info);
  is_reference_kernel = info.is_reference_kernel;
  if ( unary_kernel_dequant == NULL ) {
    fprintf( stderr, "JIT for IDENTITY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel_dequant( &unary_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if ( short_data_gold[(i*ldo)+j] != short_data[(i*ldo)+j] ) {
        printf("error at position i=%i, j=%i, %i, %i\n", i, j, short_data_gold[(i*ldo)+j], short_data[(i*ldo)+j]);
        s = 1;
      }
#ifdef PRINT_VERBOSE
      printf("%i ", short_data_gold[(i*ldo)+j]);
#endif
    }
#ifdef PRINT_VERBOSE
    printf("\n");
#endif
  }
  if ( s == 0 ) {
    printf("SUCCESS unary quant FP32 -> int16\n");
  } else {
    printf("FAILURE unary quant FP32 -> int16\n");
    ret = EXIT_FAILURE;
  }
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if ( f32_short_data_gold[(i*ldi)+j] != f32_short_data[(i*ldi)+j] ) {
        printf("error at position i=%i, j=%i, %f, %f\n", i, j, f32_short_data_gold[(i*ldi)+j], f32_short_data[(i*ldi)+j]);
        s = 1;
      }
#ifdef PRINT_VERBOSE
      printf("%f ", f32_short_data_gold[(i*ldi)+j]);
#endif
    }
#ifdef PRINT_VERBOSE
    printf("\n");
#endif
  }
  if ( s == 0 ) {
    printf("SUCCESS unary quant int16 -> FP32\n");
  } else {
    printf("FAILURE unary quant int16 -> FP32\n");
    ret = EXIT_FAILURE;
  }

  libxsmm_free( short_data_gold );
  libxsmm_free( short_data );
  libxsmm_free( f32_short_data_gold );
  libxsmm_free( f32_short_data );
  libxsmm_free( in );

  return ret;
}

LIBXSMM_INLINE
int test_float_to_int32_to_float( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, unsigned int skip_scf_cvt ) {
  float *in;
  int *int_data;
  int *int_data_gold;
  float *f32_int_data;
  float *f32_int_data_gold;
  libxsmm_blasint i, j;
  unsigned int s;
  int ret = EXIT_SUCCESS;
  float max_value = FLT_MIN;
  libxsmm_meltwfunction_unary unary_kernel_dequant;
  libxsmm_meltwfunction_unary unary_kernel_quant;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_shape unary_shape;
  int maxexp = 0;
  float scf_quant = 0.0;
  float scf_dequant = 0.0;

  if ( M > ldi ) {
    fprintf( stderr, "test_float_to_int8_to_float: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if ( (M == 0) || (N == 0) ) {
    return ret;
  }

  in                = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi, 64);
  int_data          = (int*)   libxsmm_aligned_malloc( sizeof(int)*N*ldo,   64);
  int_data_gold     = (int*)   libxsmm_aligned_malloc( sizeof(int)*N*ldo,   64);
  f32_int_data      = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi, 64);
  f32_int_data_gold = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi, 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      in[(i*ldi)+j] = (float)(((i*ldi)+j)%64);
      max_value = ( max_value < in[(i*ldi)+j] ) ? in[(i*ldi)+j] : max_value;
#ifdef PRINT_VERBOSE
      printf("%f ", in[(i*ldi)+j]);
#endif
    }
#ifdef PRINT_VERBOSE
    printf("\n");
#endif
  }

  /* compute scaling factor */
  /* take return value of LIBXSMM_FREXPF to mute static analysis issue */
  LIBXSMM_ELIDE_RESULT(float, LIBXSMM_FREXPF(max_value, &maxexp));
  maxexp = LIBXSMM_MAX(-128, maxexp-30);
  scf_quant = libxsmm_sexp2_i8i(-maxexp);

  if (skip_scf_cvt > 0) {
    scf_quant = 1.0;
  }

  /* run quantization */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      int_data_gold[(i*ldo)+j] = (int)LIBXSMM_NEARBYINTF( in[(i*ldi)+j] * scf_quant );
    }
  }

  /* run dequantization */
  scf_dequant = libxsmm_sexp2_i8i(maxexp);

  if (skip_scf_cvt > 0) {
    scf_dequant = 1.0;
  }

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      f32_int_data_gold[(i*ldi)+j] = ((float)int_data_gold[(i*ldo)+j]) * scf_dequant ;
    }
  }

  unary_shape.m = M;
  unary_shape.n = N;
  unary_shape.ldi = ldi;
  unary_shape.ldo = ldo;
  unary_shape.in0_type = LIBXSMM_DATATYPE_F32;
  unary_shape.out_type = LIBXSMM_DATATYPE_I32;
  unary_shape.comp_type = LIBXSMM_DATATYPE_F32;

  /* use jited quantization */
  unary_param.in.primary  = (void*)in;
  if (skip_scf_cvt == 0) {
    unary_param.in.secondary  = (void*)&scf_quant;
  }
  unary_param.out.primary = (void*)int_data;
  unary_kernel_quant = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_QUANT, unary_shape, (skip_scf_cvt > 0) ? LIBXSMM_MELTW_FLAG_UNARY_NO_SCF_QUANT : LIBXSMM_MELTW_FLAG_UNARY_NONE  );
  libxsmm_get_kernel_info((const void*) unary_kernel_quant, &info);
  is_reference_kernel = info.is_reference_kernel;
  if ( unary_kernel_quant == NULL ) {
    fprintf( stderr, "JIT for IDENTITY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel_quant( &unary_param );

  unary_shape.m = M;
  unary_shape.n = N;
  unary_shape.ldi = ldo;
  unary_shape.ldo = ldi;
  unary_shape.in0_type = LIBXSMM_DATATYPE_I32;
  unary_shape.out_type = LIBXSMM_DATATYPE_F32;
  unary_shape.comp_type = LIBXSMM_DATATYPE_F32;

  /* use jited quantization */
  unary_param.in.primary  = (void*)int_data;
  if (skip_scf_cvt == 0) {
    unary_param.in.secondary  = (void*)&scf_dequant;
  }
  unary_param.out.primary = (void*)f32_int_data;
  unary_kernel_dequant = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_DEQUANT, unary_shape, (skip_scf_cvt > 0) ? LIBXSMM_MELTW_FLAG_UNARY_NO_SCF_QUANT : LIBXSMM_MELTW_FLAG_UNARY_NONE  );
  libxsmm_get_kernel_info((const void*) unary_kernel_dequant, &info);
  is_reference_kernel = info.is_reference_kernel;
  if ( unary_kernel_dequant == NULL ) {
    fprintf( stderr, "JIT for IDENTITY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel_dequant( &unary_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if ( int_data_gold[(i*ldo)+j] != int_data[(i*ldo)+j] ) {
        printf("error at position i=%i, j=%i, %i, %i\n", i, j, int_data_gold[(i*ldo)+j], int_data[(i*ldo)+j]);
        s = 1;
      }
#ifdef PRINT_VERBOSE
      printf("%i ", int_data_gold[(i*ldo)+j]);
#endif
    }
#ifdef PRINT_VERBOSE
    printf("\n");
#endif
  }
  if ( s == 0 ) {
    printf("SUCCESS unary quant FP32 -> int32\n");
  } else {
    printf("FAILURE unary quant FP32 -> int32\n");
    ret = EXIT_FAILURE;
  }
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if ( f32_int_data_gold[(i*ldi)+j] != f32_int_data[(i*ldi)+j] ) {
        printf("error at position i=%i, j=%i, %f, %f\n", i, j, f32_int_data_gold[(i*ldi)+j], f32_int_data[(i*ldi)+j]);
        s = 1;
      }
#ifdef PRINT_VERBOSE
      printf("%f ", f32_int_data_gold[(i*ldi)+j]);
#endif
    }
#ifdef PRINT_VERBOSE
    printf("\n");
#endif
  }
  if ( s == 0 ) {
    printf("SUCCESS unary quant int32 -> FP32\n");
  } else {
    printf("FAILURE unary quant int32 -> FP32\n");
    ret = EXIT_FAILURE;
  }

  libxsmm_free( int_data_gold );
  libxsmm_free( int_data );
  libxsmm_free( f32_int_data_gold );
  libxsmm_free( f32_int_data );
  libxsmm_free( in );

  return ret;
}

int main( int argc, char* argv[] ) {
  char* dt_in = NULL;
  char* dt_out = NULL;
  libxsmm_datatype dtype_in;
  libxsmm_datatype dtype_out;
  /*libxsmm_datatype dtype_comp = LIBXSMM_DATATYPE_F32;*/
  libxsmm_blasint M;
  libxsmm_blasint N;
  libxsmm_blasint ldi;
  libxsmm_blasint ldo;
  unsigned int skip_scf_cvt = 0;
  unsigned int signed_sat = 0;

  int ret = EXIT_FAILURE;

  if ( argc != 9 ) {
    printf(" Error! Usage: %s [F32] [I8/I16/I32] [M] [N] [ldi] [ldo] [skip_scf_cvt] [signed_sat]\n", argv[0] );
    exit(-1);
  }

  dt_in     = argv[1];
  dt_out    = argv[2];
  M         = atoi(argv[3]);
  N         = atoi(argv[4]);
  ldi       = atoi(argv[5]);
  ldo       = atoi(argv[6]);
  skip_scf_cvt  = atoi(argv[7]);
  signed_sat  = atoi(argv[8]);

  dtype_in  = char_to_libxsmm_datatype( dt_in );
  dtype_out = char_to_libxsmm_datatype( dt_out );

  if ( (dtype_in == LIBXSMM_DATATYPE_F32) && (dtype_out == LIBXSMM_DATATYPE_I8) ) {
    printf("Testing FP32 <-> int8 quant - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_float_to_int8_to_float( M, N, ldi, ldo, skip_scf_cvt, signed_sat );
  } else if ( (dtype_in == LIBXSMM_DATATYPE_F32) && (dtype_out == LIBXSMM_DATATYPE_I16) ) {
    printf("Testing FP32 <-> int16 quant - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_float_to_int16_to_float( M, N, ldi, ldo, skip_scf_cvt, signed_sat );
  } else if ( (dtype_in == LIBXSMM_DATATYPE_F32) && (dtype_out == LIBXSMM_DATATYPE_I32) ) {
    printf("Testing FP32 <-> int32 quant - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_float_to_int32_to_float( M, N, ldi, ldo, skip_scf_cvt);
  } else {
    printf(" Case not implemented! Usage: %s [F32] [I8/I16/I32] [M] [N] [ldi] [ldo] [skip_scf_cvt] [signed_sat]\n", argv[0] );
    exit(-1);
  }

  ret = (ret == EXIT_SUCCESS) ? libxsmm_return_success_code(is_reference_kernel) : ret;
  return ret;
}
