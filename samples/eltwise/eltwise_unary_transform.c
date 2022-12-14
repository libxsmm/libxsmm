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
#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "eltwise_common.h"

LIBXSMM_INLINE
void ref_transpose( const void* in, void* out, const libxsmm_blasint M, const libxsmm_blasint N, const libxsmm_blasint ldi, const libxsmm_blasint ldo, const libxsmm_datatype dtype ) {
  size_t i, j;

  if ( (dtype == LIBXSMM_DATATYPE_F64) || (dtype == LIBXSMM_DATATYPE_I64) ) {
    const double*  in_data = (const double*)in;
    double*       out_data = (double*)      out;
    for ( i = 0; i < (size_t)N; ++i ) {
      for ( j = 0; j < (size_t)M; ++j ) {
        out_data[(j*(size_t)ldo)+i] = in_data[(i*(size_t)ldi)+j];
      }
    }
  } else if ( (dtype == LIBXSMM_DATATYPE_F32) || (dtype == LIBXSMM_DATATYPE_I32)) {
    const float*  in_data = (const float*)in;
    float*       out_data = (float*)      out;
    for ( i = 0; i < (size_t)N; ++i ) {
      for ( j = 0; j < (size_t)M; ++j ) {
        out_data[(j*(size_t)ldo)+i] = in_data[(i*(size_t)ldi)+j];
      }
    }
  } else if ( (dtype == LIBXSMM_DATATYPE_BF16) || (dtype == LIBXSMM_DATATYPE_F16) || (dtype == LIBXSMM_DATATYPE_I16) ) {
    const unsigned short*  in_data = (const unsigned short*)in;
    unsigned short*       out_data = (unsigned short*)      out;
    for ( i = 0; i < (size_t)N; ++i ) {
      for ( j = 0; j < (size_t)M; ++j ) {
        out_data[(j*(size_t)ldo)+i] = in_data[(i*(size_t)ldi)+j];
      }
    }
  } else if ( (dtype == LIBXSMM_DATATYPE_I8) || (dtype == LIBXSMM_DATATYPE_BF8) ) {
    const unsigned char*  in_data = (const unsigned char*)in;
    unsigned char*       out_data = (unsigned char*)      out;
    for ( i = 0; i < (size_t)N; ++i ) {
      for ( j = 0; j < (size_t)M; ++j ) {
        out_data[(j*(size_t)ldo)+i] = in_data[(i*(size_t)ldi)+j];
      }
    }
  } else {
    /* should not happen */
  }
}

LIBXSMM_INLINE
int test_normal_to_normalT( const libxsmm_blasint M, const libxsmm_blasint N, const libxsmm_blasint ldi, const libxsmm_blasint ldo, const libxsmm_datatype dtype ) {
  const libxsmm_meltw_unary_shape unary_shape = libxsmm_create_meltw_unary_shape( M, N, ldi, ldo, dtype, dtype, dtype );
  libxsmm_meltw_unary_param unary_param /*= { 0 }*/;
  libxsmm_meltwfunction_unary unary_kernel;
  libxsmm_matdiff_info norms_out;
  char *in;
  char *out, *out_gold;
  int ret = EXIT_SUCCESS;

  if ( M > ldi ) {
    fprintf( stderr, "test_normal_to_normalT: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if ( N > ldo ) {
    fprintf( stderr, "test_normal_to_normalT: ldo needs to be equal to or bigger than N\n");
    exit(-1);
  }

  in       = (char*) libxsmm_aligned_malloc((size_t)LIBXSMM_TYPESIZE(dtype)*N*ldi, 64);
  out      = (char*) libxsmm_aligned_malloc((size_t)LIBXSMM_TYPESIZE(dtype)*M*ldo, 64);
  out_gold = (char*) libxsmm_aligned_malloc((size_t)LIBXSMM_TYPESIZE(dtype)*M*ldo, 64);

  init_random_matrix( dtype, in,       1, ldi, N, 0 );
  init_zero_matrix(   dtype, out,      1, ldo, M );
  init_zero_matrix(   dtype, out_gold, 1, ldo, M );

  /* run reference */
  ref_transpose( in, out_gold, M, N, ldi, ldo, dtype );

  /* use jited transpose */
  unary_param.in.primary  = (void*)in;
  unary_param.out.primary = (void*)out;
  unary_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for NORM_TO_NORMT TPP. Bailing...!\n");
    exit(LIBXSMM_EXIT_ERROR);
  }
  /* run TPP */
  unary_kernel( &unary_param );

  /* check result */
  norms_out = check_matrix( dtype, out_gold, out, ldo, N, M );

  if ( norms_out.linf_abs == 0 ) {
    printf("SUCCESS unary transpose LIBXSMM_TYPESIZE: %i\n", LIBXSMM_TYPESIZE(dtype));
  } else {
    printf("FAILURE unary transpose LIBXSMM_TYPESIZE: %i\n", LIBXSMM_TYPESIZE(dtype));
    ret = EXIT_FAILURE;
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );

  return ret;
}

LIBXSMM_INLINE
int test_vnni2_to_vnni2T_16bit( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  unsigned short *in, *in_vnni;
  unsigned short *out, *out_gold, *out_vnni;
  libxsmm_blasint i, j, j2;
  unsigned int s;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param /*= { 0 }*/;
  libxsmm_meltw_unary_shape unary_shape /*= { 0 }*/;
  libxsmm_meltwfunction_unary unary_kernel;

  if ( M > ldi ) {
    fprintf( stderr, "test_vnni_to_vnniT_16bit: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if ( N > ldo ) {
    fprintf( stderr, "test_vnni_to_vnniT_16bit: ldo needs to be equal to or bigger than N\n");
    exit(-1);
  }

  in       = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*ldi*N, 64);
  in_vnni  = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*ldi*N, 64);
  out      = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*M*ldo, 64);
  out_gold = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*M*ldo, 64);
  out_vnni = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*M*ldo, 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      in[(i*ldi)+j] = (unsigned short)(((i*ldi)+j)%112);
    }
  }
  /* to vnni */
  for ( j = 0; j < N/2; ++j ) {
    for ( i = 0; i < ldi ; ++i ) {
      for ( j2 = 0; j2 < 2; ++j2 ) {
        in_vnni[(j*ldi*2)+(i*2)+j2] = in[(((j*2)+j2)*ldi)+i];
      }
    }
  }

  /* init out */
  for ( i = 0; i < M*ldo; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < M*ldo; ++i ) {
    out_gold[i] = 0;
    out_vnni[i] = 0;
  }

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      out_gold[(j*ldo)+i] = in[(i*ldi)+j];
    }
  }

  /* to vnni */
  for ( j = 0; j < M/2; ++j ) {
    for ( i = 0; i < N ; ++i ) {
      for ( j2 = 0; j2 < 2; ++j2 ) {
        out_vnni[(j*ldo*2)+(i*2)+j2] = out_gold[(((j*2)+j2)*ldo)+i];
      }
    }
  }

  unary_shape.m = M;
  unary_shape.n = N;
  unary_shape.ldi = ldi;
  unary_shape.ldo = ldo;
  unary_shape.in0_type = LIBXSMM_DATATYPE_BF16;
  unary_shape.out_type = LIBXSMM_DATATYPE_BF16;
  unary_shape.comp_type = LIBXSMM_DATATYPE_BF16;

  /* use jited transpose */
  /* to vnni */
#if 0
  for ( j = 0; j < N/2; ++j ) {
    for ( i = 0; i < ldi ; ++i ) {
      printf("( ");
      for ( j2 = 0; j2 < 2; ++j2 ) {
        printf("%i ", in_vnni[(j*ldi*2)+(i*2)+j2]);
      }
      printf(") ");
    }
    printf("\n");
  }
#endif
  unary_param.in.primary  = (void*)in_vnni;
  unary_param.out.primary = (void*)out;
  unary_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2_TO_VNNI2T, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for VNNI2_TO_VNNI2T TPP. Bailing...!\n");
    exit(LIBXSMM_EXIT_ERROR);
  }
  unary_kernel( &unary_param );
#if 0
  for ( j = 0; j < N/2; ++j ) {
    for ( i = 0; i < ldi ; ++i ) {
      printf("( ");
      for ( j2 = 0; j2 < 2; ++j2 ) {
        printf("%i ", out[(j*ldi*2)+(i*2)+j2]);
      }
      printf(") ");
    }
    printf("\n");
  }
#endif

  /* compare result */
  s = 0;
  for ( i = 0; i < M; ++i ) {
    for ( j = 0; j < N; ++j ) {
      if ( out_vnni[(i*ldo)+j] != out[(i*ldo)+j] ) {
        printf("error at possition i=%i, j=%i\n", i, j);
        s = 1;
      }
    }
  }
  if ( s == 0 ) {
    printf("SUCCESS unary VNNI2 transpose 16bit\n");
  } else {
    printf("FAILURE unary VNNI2 transpose 16bit\n");
    ret = EXIT_FAILURE;
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( out_vnni );
  libxsmm_free( in );
  libxsmm_free( in_vnni );

  return ret;
}

LIBXSMM_INLINE
int test_vnni4_to_vnni4T_16bit( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  unsigned short *in, *in_vnni;
  unsigned short *out, *out_gold, *out_vnni;
  libxsmm_blasint i, j, j2;
  unsigned int s;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param /*= { 0 }*/;
  libxsmm_meltw_unary_shape unary_shape /*= { 0 }*/;
  libxsmm_meltwfunction_unary unary_kernel;

  if ( M > ldi ) {
    fprintf( stderr, "test_vnni_to_vnniT_16bit: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if ( N > ldo ) {
    fprintf( stderr, "test_vnni_to_vnniT_16bit: ldo needs to be equal to or bigger than N\n");
    exit(-1);
  }

  in       = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*ldi*N, 64);
  in_vnni  = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*ldi*N, 64);
  out      = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*M*ldo, 64);
  out_gold = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*M*ldo, 64);
  out_vnni = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*M*ldo, 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      in[(i*ldi)+j] = (unsigned char)(((i*ldi)+j)%112);
    }
  }
  /* to vnni */
  for ( j = 0; j < N/4; ++j ) {
    for ( i = 0; i < ldi ; ++i ) {
      for ( j2 = 0; j2 < 4; ++j2 ) {
        in_vnni[(j*ldi*4)+(i*4)+j2] = in[(((j*4)+j2)*ldi)+i];
      }
    }
  }

  /* init out */
  for ( i = 0; i < M*ldo; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < M*ldo; ++i ) {
    out_gold[i] = 0;
    out_vnni[i] = 0;
  }

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      out_gold[(j*ldo)+i] = in[(i*ldi)+j];
    }
  }

  /* to vnni */
  for ( j = 0; j < M/4; ++j ) {
    for ( i = 0; i < N ; ++i ) {
      for ( j2 = 0; j2 < 4; ++j2 ) {
        out_vnni[(j*ldo*4)+(i*4)+j2] = out_gold[(((j*4)+j2)*ldo)+i];
      }
    }
  }

  unary_shape.m = M;
  unary_shape.n = N;
  unary_shape.ldi = ldi;
  unary_shape.ldo = ldo;
  unary_shape.in0_type = LIBXSMM_DATATYPE_I16;
  unary_shape.out_type = LIBXSMM_DATATYPE_I16;
  unary_shape.comp_type = LIBXSMM_DATATYPE_I16;

  /* use jited tranpose */
  unary_param.in.primary  = (void*)in_vnni;
  unary_param.out.primary = (void*)out;
  unary_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI4T, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for VNNI4_TO_VNNI4T TPP. Bailing...!\n");
    exit(LIBXSMM_EXIT_ERROR);
  }
  unary_kernel( &unary_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < M; ++i ) {
    for ( j = 0; j < N; ++j ) {
      if ( out_vnni[(i*ldo)+j] != out[(i*ldo)+j] ) {
        printf("error at possition i=%i, j=%i\n", i, j);
        s = 1;
      }
    }
  }
  if ( s == 0 ) {
    printf("SUCCESS unary VNNI4 transpose 16bit\n");
  } else {
    printf("FAILURE unary VNNI4 transpose 16bit\n");
    ret = EXIT_FAILURE;
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( out_vnni );
  libxsmm_free( in );
  libxsmm_free( in_vnni );

  return ret;
}

LIBXSMM_INLINE
int test_vnni4_to_vnni4T_08bit( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  unsigned char *in, *in_vnni;
  unsigned char *out, *out_gold, *out_vnni;
  libxsmm_blasint i, j, j2;
  unsigned int s;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param /*= { 0 }*/;
  libxsmm_meltw_unary_shape unary_shape /*= { 0 }*/;
  libxsmm_meltwfunction_unary unary_kernel;

  if ( M > ldi ) {
    fprintf( stderr, "test_vnni_to_vnniT_08bit: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if ( N > ldo ) {
    fprintf( stderr, "test_vnni_to_vnniT_08bit: ldo needs to be equal to or bigger than N\n");
    exit(-1);
  }

  in       = (unsigned char*)libxsmm_aligned_malloc( sizeof(unsigned char)*ldi*N, 64);
  in_vnni  = (unsigned char*)libxsmm_aligned_malloc( sizeof(unsigned char)*ldi*N, 64);
  out      = (unsigned char*)libxsmm_aligned_malloc( sizeof(unsigned char)*M*ldo, 64);
  out_gold = (unsigned char*)libxsmm_aligned_malloc( sizeof(unsigned char)*M*ldo, 64);
  out_vnni = (unsigned char*)libxsmm_aligned_malloc( sizeof(unsigned char)*M*ldo, 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      in[(i*ldi)+j] = (unsigned char)(((i*ldi)+j)%112);
    }
  }
  /* to vnni */
  for ( j = 0; j < N/4; ++j ) {
    for ( i = 0; i < ldi ; ++i ) {
      for ( j2 = 0; j2 < 4; ++j2 ) {
        in_vnni[(j*ldi*4)+(i*4)+j2] = in[(((j*4)+j2)*ldi)+i];
      }
    }
  }

  /* init out */
  for ( i = 0; i < M*ldo; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < M*ldo; ++i ) {
    out_gold[i] = 0;
    out_vnni[i] = 0;
  }

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      out_gold[(j*ldo)+i] = in[(i*ldi)+j];
    }
  }

  /* to vnni */
  for ( j = 0; j < M/4; ++j ) {
    for ( i = 0; i < N ; ++i ) {
      for ( j2 = 0; j2 < 4; ++j2 ) {
        out_vnni[(j*ldo*4)+(i*4)+j2] = out_gold[(((j*4)+j2)*ldo)+i];
      }
    }
  }

  unary_shape.m = M;
  unary_shape.n = N;
  unary_shape.ldi = ldi;
  unary_shape.ldo = ldo;
  unary_shape.in0_type = LIBXSMM_DATATYPE_I8;
  unary_shape.out_type = LIBXSMM_DATATYPE_I8;
  unary_shape.comp_type = LIBXSMM_DATATYPE_I8;

  /* use jited transpose */
  unary_param.in.primary  = (void*)in_vnni;
  unary_param.out.primary = (void*)out;
  unary_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI4T, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for VNNI4_TO_VNNI4T TPP. Bailing...!\n");
    exit(LIBXSMM_EXIT_ERROR);
  }
  unary_kernel( &unary_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < M; ++i ) {
    for ( j = 0; j < N; ++j ) {
      if ( out_vnni[(i*ldo)+j] != out[(i*ldo)+j] ) {
        printf("error at possition i=%i, j=%i\n", i, j);
        s = 1;
      }
    }
  }
  if ( s == 0 ) {
    printf("SUCCESS unary VNNI4 transpose 08bit\n");
  } else {
    printf("FAILURE unary VNNI4 transpose 08bit\n");
    ret = EXIT_FAILURE;
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( out_vnni );
  libxsmm_free( in );
  libxsmm_free( in_vnni );

  return ret;
}

LIBXSMM_INLINE
int test_norm_to_vnni2_16bit( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  unsigned short *in;
  unsigned short *out, *out_gold;
  libxsmm_blasint i, j, j2;
  unsigned int s;
  int ret = EXIT_SUCCESS;
  libxsmm_blasint Nn = N + (N%2);

  libxsmm_meltw_unary_param unary_param /*= { 0 }*/;
  libxsmm_meltw_unary_shape unary_shape /*= { 0 }*/;
  libxsmm_meltw_unary_type  unary_type;
  libxsmm_meltwfunction_unary unary_kernel;

  if ( M > ldi ) {
    fprintf( stderr, "test_norm_to_vnni2_16bit: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if ( M > ldo ) {
    fprintf( stderr, "test_norm_to_vnni2_16bit: ldo needs to be equal to or bigger than M\n");
    exit(-1);
  }

  in       = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*ldi*Nn, 64);
  out      = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*ldo*Nn, 64);
  out_gold = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*ldo*Nn, 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      in[(i*ldi)+j] = (unsigned short)(((i*ldi)+j)%112);
    }
  }
  for ( i = N; i < Nn; ++i ) {
    for ( j = 0; j < M; ++j ) {
      in[(i*ldi)+j] = 0;
    }
  }

  /* init out */
  for ( i = 0; i < ldo*Nn; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < ldo*Nn; ++i ) {
    out_gold[i] = 0;
  }

  /* to vnni */
  for ( j = 0; j < Nn/2; ++j ) {
    for ( i = 0; i < M ; ++i ) {
      for ( j2 = 0; j2 < 2; ++j2 ) {
        out_gold[(j*ldo*2)+(i*2)+j2] = in[(((j*2)+j2)*ldi)+i];
      }
    }
  }

  unary_shape.m = M;
  unary_shape.n = N;
  unary_shape.ldi = ldi;
  unary_shape.ldo = ldo;
  unary_shape.in0_type = LIBXSMM_DATATYPE_BF16;
  unary_shape.out_type = LIBXSMM_DATATYPE_BF16;
  unary_shape.comp_type = LIBXSMM_DATATYPE_BF16;

  /* use jited transpose */
  unary_param.in.primary  = (void*)in;
  unary_param.out.primary = (void*)out;
  if ( N % 2 == 1 ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD;
  } else {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2;
  }
  unary_kernel = libxsmm_dispatch_meltw_unary_v2( unary_type, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for NORM_TO_VNNI2 TPP. Bailing...!\n");
    exit(LIBXSMM_EXIT_ERROR);
  }
  unary_kernel( &unary_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < Nn; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if ( out_gold[(i*ldo)+j] != out[(i*ldo)+j] ) {
        printf("error at possition i=%i, j=%i, %i %i\n", i, j, out_gold[(i*ldo)+j], out[(i*ldo)+j]);
        s = 1;
      }
    }
  }
  if ( s == 0 ) {
    printf("SUCCESS unary VNNI2 transform 16bit\n");
  } else {
    printf("FAILURE unary VNNI2 transform 16bit\n");
    ret = EXIT_FAILURE;
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );

  return ret;
}

LIBXSMM_INLINE
int test_norm_to_vnni4T_16bit( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  unsigned short *in;
  unsigned short *out, *out_gold;
  libxsmm_blasint i, j, i2;
  unsigned int s;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param /*= { 0 }*/;
  libxsmm_meltw_unary_shape unary_shape /*= { 0 }*/;
  libxsmm_meltw_unary_type  unary_type;
  libxsmm_meltwfunction_unary unary_kernel;

  if ( M % 4 != 0 || ldi % 4 != 0 ) {
    fprintf( stderr, "test_norm_to_vnni4T_16bit: M mod 4 AND ldi mod 4 must be 0\n");
    exit(-1);
  }
  if ( M > ldi ) {
    fprintf( stderr, "test_norm_to_vnni4T_16bit: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if ( N > ldo ) {
    fprintf( stderr, "test_norm_to_vnni4T_16bit: ldo needs to be equal to or bigger than N\n");
    exit(-1);
  }

  in       = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*ldi*N, 64);
  out      = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*ldo*M, 64);
  out_gold = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*ldo*M, 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      in[(i*ldi)+j] = (unsigned short)(((i*ldi)+j)%112);
    }
  }

  /* init out */
  for ( i = 0; i < ldo*M; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < ldo*M; ++i ) {
    out_gold[i] = 0;
  }

  /* to vnni4T */
  for ( i = 0; i < M/4; ++i ) {
    for ( j = 0; j < N ; ++j ) {
      for ( i2 = 0; i2 < 4; ++i2 ) {
        out_gold[(i*ldo*4)+(j*4)+i2] = in[(j*ldi)+(i*4+i2)];
      }
    }
  }

  unary_shape.m = M;
  unary_shape.n = N;
  unary_shape.ldi = ldi;
  unary_shape.ldo = ldo;
  unary_shape.in0_type = LIBXSMM_DATATYPE_I16;
  unary_shape.out_type = LIBXSMM_DATATYPE_I16;
  unary_shape.comp_type = LIBXSMM_DATATYPE_I16;

  /* use jited tranpose */
  unary_param.in.primary  = (void*)in;
  unary_param.out.primary = (void*)out;
  unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4T;
  unary_kernel = libxsmm_dispatch_meltw_unary_v2( unary_type, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for NORM_TO_VNNI4T TPP. Bailing...!\n");
    exit(LIBXSMM_EXIT_ERROR);
  }
  unary_kernel( &unary_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < M/4; ++i ) {
    for ( j = 0; j < N ; ++j ) {
      for ( i2 = 0; i2 < 4; ++i2 ) {
        if ( out_gold[(i*ldo*4)+(j*4)+i2] != out[(i*ldo*4)+(j*4)+i2] ) {
          printf("error at position OUT[%i][%i][%i]: %i %i\n", i, j, i2, out_gold[(i*ldo*4)+(j*4)+i2], out[(i*ldo*4)+(j*4)+i2]);
          s = 1;
        }
      }
    }
  }

  if ( s == 0 ) {
    printf("SUCCESS unary norm to VNNI4T transform 16bit\n");
  } else {
    printf("FAILURE unary norm to VNNI4T transform 16bit\n");
    ret = EXIT_FAILURE;
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );

  return ret;
}

LIBXSMM_INLINE
int test_norm_to_vnni4_16bit( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  unsigned short *in;
  unsigned short *out, *out_gold;
  libxsmm_blasint i, j, j2;
  unsigned int s;
  int ret = EXIT_SUCCESS;
  libxsmm_blasint Nn = ((N%4) == 0) ? N : LIBXSMM_UP(N, 4);

  libxsmm_meltw_unary_param unary_param /*= { 0 }*/;
  libxsmm_meltw_unary_shape unary_shape /*= { 0 }*/;
  libxsmm_meltw_unary_type  unary_type;
  libxsmm_meltwfunction_unary unary_kernel;

  if ( M > ldi ) {
    fprintf( stderr, "test_norm_to_vnni4_16bit: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if ( M > ldo ) {
    fprintf( stderr, "test_norm_to_vnni4_16bit: ldo needs to be equal to or bigger than M\n");
    exit(-1);
  }

  in       = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*ldi*Nn, 64);
  out      = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*ldo*Nn, 64);
  out_gold = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*ldo*Nn, 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      in[(i*ldi)+j] = (unsigned short)(((i*ldi)+j)%112);
    }
  }
  for ( i = N; i < Nn; ++i ) {
    for ( j = 0; j < M; ++j ) {
      in[(i*ldi)+j] = 0;
    }
  }

  /* init out */
  for ( i = 0; i < ldo*Nn; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < ldo*Nn; ++i ) {
    out_gold[i] = 0;
  }

  /* to vnni */
  for ( j = 0; j < Nn/4; ++j ) {
    for ( i = 0; i < M ; ++i ) {
      for ( j2 = 0; j2 < 4; ++j2 ) {
        out_gold[(j*ldo*4)+(i*4)+j2] = in[(((j*4)+j2)*ldi)+i];
      }
    }
  }

  unary_shape.m = M;
  unary_shape.n = N;
  unary_shape.ldi = ldi;
  unary_shape.ldo = ldo;
  unary_shape.in0_type = LIBXSMM_DATATYPE_I16;
  unary_shape.out_type = LIBXSMM_DATATYPE_I16;
  unary_shape.comp_type = LIBXSMM_DATATYPE_I16;

  /* use jited tranpose */
  unary_param.in.primary  = (void*)in;
  unary_param.out.primary = (void*)out;
  if ( N % 4 != 0 ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4_PAD;
  } else {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4;
  }
  unary_kernel = libxsmm_dispatch_meltw_unary_v2( unary_type, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for NORM_TO_VNNI4 TPP. Bailing...!\n");
    exit(LIBXSMM_EXIT_ERROR);
  }
  unary_kernel( &unary_param );

  /* compare result */
  s = 0;
  for ( j = 0; j < Nn/4; ++j ) {
    for ( i = 0; i < M; ++i ) {
      for (j2 = 0; j2 < 4; ++j2){
        if ( out_gold[(j*ldo*4)+(i*4)+j2] != out[(j*ldo*4)+(i*4)+j2] ) {
          printf("error at position OUT[%i][%i][%i]: %i %i\n", j, i, j2, out_gold[(j*ldo*4)+(i*4)+j2], out[(j*ldo*4)+(i*4)+j2]);
          s = 1;
        }
      }
    }
  }
  if ( s == 0 ) {
    printf("SUCCESS unary VNNI4 transform 16bit\n");
  } else {
    printf("FAILURE unary VNNI4 transform 16bit\n");
    ret = EXIT_FAILURE;
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );

  return ret;
}

LIBXSMM_INLINE
int test_norm_to_vnni4_08bit( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  unsigned char *in;
  unsigned char *out, *out_gold;
  libxsmm_blasint i, j, j2;
  unsigned int s;
  int ret = EXIT_SUCCESS;
  libxsmm_blasint Nn = ((N%4) == 0) ? N : LIBXSMM_UP(N, 4);

  libxsmm_meltw_unary_param unary_param /*= { 0 }*/;
  libxsmm_meltw_unary_shape unary_shape /*= { 0 }*/;
  libxsmm_meltw_unary_type  unary_type;
  libxsmm_meltwfunction_unary unary_kernel;

  if ( M > ldi ) {
    fprintf( stderr, "test_norm_to_vnni_16bit: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if ( M > ldo ) {
    fprintf( stderr, "test_norm_to_vnni_16bit: ldo needs to be equal to or bigger than M\n");
    exit(-1);
  }

  in       = (unsigned char*)libxsmm_aligned_malloc( sizeof(unsigned char)*ldi*Nn, 64);
  out      = (unsigned char*)libxsmm_aligned_malloc( sizeof(unsigned char)*ldo*Nn, 64);
  out_gold = (unsigned char*)libxsmm_aligned_malloc( sizeof(unsigned char)*ldo*Nn, 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      in[(i*ldi)+j] = (unsigned char)(((i*ldi)+j)%112);
    }
  }
  for ( i = N; i < Nn; ++i ) {
    for ( j = 0; j < M; ++j ) {
      in[(i*ldi)+j] = 0;
    }
  }

  /* init out */
  for ( i = 0; i < ldo*Nn; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < ldo*Nn; ++i ) {
    out_gold[i] = 0;
  }

  /* to vnni */
  for ( j = 0; j < Nn/4; ++j ) {
    for ( i = 0; i < M ; ++i ) {
      for ( j2 = 0; j2 < 4; ++j2 ) {
        out_gold[(j*ldo*4)+(i*4)+j2] = in[(((j*4)+j2)*ldi)+i];
      }
    }
  }

  unary_shape.m = M;
  unary_shape.n = N;
  unary_shape.ldi = ldi;
  unary_shape.ldo = ldo;
  unary_shape.in0_type = LIBXSMM_DATATYPE_I8;
  unary_shape.out_type = LIBXSMM_DATATYPE_I8;
  unary_shape.comp_type = LIBXSMM_DATATYPE_I8;

  /* use jited transpose */
  unary_param.in.primary  = (void*)in;
  unary_param.out.primary = (void*)out;
  if ( N % 4 != 0 ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4_PAD;
  } else {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4;
  }
  unary_kernel = libxsmm_dispatch_meltw_unary_v2( unary_type, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for NORM_TO_VNNI4 TPP. Bailing...!\n");
    exit(LIBXSMM_EXIT_ERROR);
  }
  unary_kernel( &unary_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < Nn; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if ( out_gold[(i*ldo)+j] != out[(i*ldo)+j] ) {
        printf("error at possition i=%i, j=%i, %i %i\n", i, j, out_gold[(i*ldo)+j], out[(i*ldo)+j]);
        s = 1;
      }
    }
  }
  if ( s == 0 ) {
    printf("SUCCESS unary VNNI4 transform 08bit\n");
  } else {
    printf("FAILURE unary VNNI4 transform 08bit\n");
    ret = EXIT_FAILURE;
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );

  return ret;
}

LIBXSMM_INLINE
int test_vnni4_to_norm_08bit( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  unsigned char *in;
  unsigned char *out, *out_gold;
  libxsmm_blasint i, j;
  unsigned int s;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param /*= { 0 }*/;
  libxsmm_meltw_unary_shape unary_shape /*= { 0 }*/;
  libxsmm_meltwfunction_unary unary_kernel;

  if ( M > ldi ) {
    fprintf( stderr, "test_vnni4_to_norm_08bit: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if ( M > ldo ) {
    fprintf( stderr, "test_vnni4_to_norm_08bit: ldo needs to be equal to or bigger than M\n");
    exit(-1);
  }

  in       = (unsigned char*)libxsmm_aligned_malloc( sizeof(unsigned char)*N*ldi, 64);
  out      = (unsigned char*)libxsmm_aligned_malloc( sizeof(unsigned char)*N*ldo, 64);
  out_gold = (unsigned char*)libxsmm_aligned_malloc( sizeof(unsigned char)*N*ldo, 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      in[(i*ldi)+j] = (unsigned char)(((i*ldi)+j)%112);
    }
  }

  /* init out */
  for ( i = 0; i < N*ldo; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < N*ldo; ++i ) {
    out_gold[i] = 0;
  }

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      out_gold[(i*ldo)+j] = in[((i/4)*ldi*4)+j*4+(i%4)];
    }
  }

  unary_shape.m = M;
  unary_shape.n = N;
  unary_shape.ldi = ldi;
  unary_shape.ldo = ldo;
  unary_shape.in0_type = LIBXSMM_DATATYPE_I8;
  unary_shape.out_type = LIBXSMM_DATATYPE_I8;
  unary_shape.comp_type = LIBXSMM_DATATYPE_I8;

  /* use jited transpose */
  unary_param.in.primary  = (void*)in;
  unary_param.out.primary = (void*)out;
  unary_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_NORM, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );

  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for VNNI4_TO_NORM TPP. Bailing...!\n");
    exit(LIBXSMM_EXIT_ERROR);
  }
  unary_kernel( &unary_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < M; ++i ) {
    for ( j = 0; j < N; ++j ) {
      if ( out_gold[(j*ldo)+i] != out[(j*ldo)+i] ) {
        printf("error at position i=%i, j=%i\n", i, j);
        s = 1;
      }
    }
  }
  if ( s == 0 ) {
    printf("SUCCESS unary VNNI4 to norm 08bit\n");
  } else {
    printf("FAILURE unary VNNI4 to norm 08bit\n");
    ret = EXIT_FAILURE;
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );

  return ret;
}

LIBXSMM_INLINE
int test_vnni4_to_vnni2_08bit( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  unsigned char *in;
  unsigned char *out, *out_gold;
  libxsmm_blasint i, j;
  unsigned int s;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param /*= { 0 }*/;
  libxsmm_meltw_unary_shape unary_shape /*= { 0 }*/;
  libxsmm_meltwfunction_unary unary_kernel;

  if ( M > ldi ) {
    fprintf( stderr, "test_vnni4_to_vnni2_08bit: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if ( M > ldo ) {
    fprintf( stderr, "test_vnni4_to_vnni2_08bit: ldo needs to be equal to or bigger than M\n");
    exit(-1);
  }

  in       = (unsigned char*)libxsmm_aligned_malloc( sizeof(unsigned char)*N*ldi, 64);
  out      = (unsigned char*)libxsmm_aligned_malloc( sizeof(unsigned char)*N*ldo, 64);
  out_gold = (unsigned char*)libxsmm_aligned_malloc( sizeof(unsigned char)*N*ldo, 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      in[(i*ldi)+j] = (unsigned char)(((i*ldi)+j)%112);
    }
  }

  /* init out */
  for ( i = 0; i < N*ldo; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < N*ldo; ++i ) {
    out_gold[i] = 0;
  }

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      out_gold[((i/2)*ldo*2)+j*2+(i%2)] = in[((i/4)*ldi*4)+j*4+(i%4)];
    }
  }

  unary_shape.m = M;
  unary_shape.n = N;
  unary_shape.ldi = ldi;
  unary_shape.ldo = ldo;
  unary_shape.in0_type = LIBXSMM_DATATYPE_I8;
  unary_shape.out_type = LIBXSMM_DATATYPE_I8;
  unary_shape.comp_type = LIBXSMM_DATATYPE_I8;

  /* use jited transpose */
  unary_param.in.primary  = (void*)in;
  unary_param.out.primary = (void*)out;
  unary_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI2, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );

  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for VNNI4_TO_VNNI2 TPP. Bailing...!\n");
    exit(LIBXSMM_EXIT_ERROR);
  }
  unary_kernel( &unary_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < M; ++i ) {
    for ( j = 0; j < N; ++j ) {
      if ( out_gold[(j*ldo)+i] != out[(j*ldo)+i] ) {
        printf("error at position i=%i, j=%i\n", i, j);
        s = 1;
      }
    }
  }
  if ( s == 0 ) {
    printf("SUCCESS unary VNNI4 to VNNI2 08bit\n");
  } else {
    printf("FAILURE unary VNNI4 to VNNI2 08bit\n");
    ret = EXIT_FAILURE;
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );

  return ret;
}

LIBXSMM_INLINE
int test_norm_padn_mod2_16bit( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  unsigned short *in;
  unsigned short *out, *out_gold;
  libxsmm_blasint i, j;
  unsigned int s;
  int ret = EXIT_SUCCESS;
  libxsmm_blasint Nn = ((N%2) == 0) ? N : N+1;

  libxsmm_meltw_unary_param unary_param /*= { 0 }*/;
  libxsmm_meltw_unary_shape unary_shape /*= { 0 }*/;
  libxsmm_meltw_unary_type  unary_type;
  libxsmm_meltwfunction_unary unary_kernel;

  if ( M > ldi ) {
    fprintf( stderr, "test_norm_padn_mod2_16bit: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if ( M > ldo ) {
    fprintf( stderr, "test_norm_padn_mod2_16bit: ldo needs to be equal to or bigger than M\n");
    exit(-1);
  }

  in       = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*ldi*N,  64);
  out      = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*ldo*Nn, 64);
  out_gold = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*ldo*Nn, 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      in[(i*ldi)+j] = (unsigned short)(((i*ldi)+j)%112);
    }
  }

  /* init out */
  for ( i = 0; i < ldo*Nn; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < ldo*Nn; ++i ) {
    out_gold[i] = 0;
  }

  /* to vnni */
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M ; ++i ) {
      out_gold[(j*ldo)+i] = in[(j*ldi)+i];
    }
  }

  unary_shape.m = M;
  unary_shape.n = N;
  unary_shape.ldi = ldi;
  unary_shape.ldo = ldo;
  unary_shape.in0_type = LIBXSMM_DATATYPE_BF16;
  unary_shape.out_type = LIBXSMM_DATATYPE_BF16;
  unary_shape.comp_type = LIBXSMM_DATATYPE_BF16;

  /* use jited transpose */
  unary_param.in.primary  = (void*)in;
  unary_param.out.primary = (void*)out;
  unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD2;
  unary_kernel = libxsmm_dispatch_meltw_unary_v2( unary_type, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for PADN_MOD2 TPP. Bailing...!\n");
    exit(LIBXSMM_EXIT_ERROR);
  }
  unary_kernel( &unary_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < Nn; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if ( out_gold[(i*ldo)+j] != out[(i*ldo)+j] ) {
        printf("error at possition i=%i, j=%i, %i %i\n", i, j, out_gold[(i*ldo)+j], out[(i*ldo)+j]);
        s = 1;
      }
    }
  }
  if ( s == 0 ) {
    printf("SUCCESS unary mod2 padn 16bit\n");
  } else {
    printf("FAILURE unary mod2 padn 16bit\n");
    ret = EXIT_FAILURE;
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );

  return ret;
}

LIBXSMM_INLINE
int test_norm_padm_mod2_16bit( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  unsigned short *in;
  unsigned short *out, *out_gold;
  libxsmm_blasint i, j;
  unsigned int s;
  int ret = EXIT_SUCCESS;
  libxsmm_blasint Mn = ((M%2) == 0) ? M : M+1;

  libxsmm_meltw_unary_param unary_param /*= { 0 }*/;
  libxsmm_meltw_unary_shape unary_shape /*= { 0 }*/;
  libxsmm_meltw_unary_type  unary_type;
  libxsmm_meltwfunction_unary unary_kernel;

  if ( M > ldi ) {
    fprintf( stderr, "test_norm_padm_mod2_16bit: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if ( Mn > ldo ) {
    fprintf( stderr, "test_norm_padm_mod2_16bit: ldo needs to be equal to or bigger than M\n");
    exit(-1);
  }

  in       = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*ldi*N, 64);
  out      = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*ldo*N, 64);
  out_gold = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*ldo*N, 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      in[(i*ldi)+j] = (unsigned short)(((i*ldi)+j)%112);
    }
  }

  /* init out */
  for ( i = 0; i < ldo*N; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < ldo*N; ++i ) {
    out_gold[i] = 0;
  }

  /* to vnni */
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M ; ++i ) {
      out_gold[(j*ldo)+i] = in[(j*ldi)+i];
    }
  }

  unary_shape.m = M;
  unary_shape.n = N;
  unary_shape.ldi = ldi;
  unary_shape.ldo = ldo;
  unary_shape.in0_type = LIBXSMM_DATATYPE_BF16;
  unary_shape.out_type = LIBXSMM_DATATYPE_BF16;
  unary_shape.comp_type = LIBXSMM_DATATYPE_BF16;

  /* use jited transpose */
  unary_param.in.primary  = (void*)in;
  unary_param.out.primary = (void*)out;
  unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD2;
  unary_kernel = libxsmm_dispatch_meltw_unary_v2( unary_type, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for PADM_MOD2 TPP. Bailing...!\n");
    exit(LIBXSMM_EXIT_ERROR);
  }
  unary_kernel( &unary_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < Mn; ++j ) {
      if ( out_gold[(i*ldo)+j] != out[(i*ldo)+j] ) {
        printf("error at possition i=%i, j=%i, %i %i\n", i, j, out_gold[(i*ldo)+j], out[(i*ldo)+j]);
        s = 1;
      }
    }
  }
  if ( s == 0 ) {
    printf("SUCCESS unary mod2 padm 16bit\n");
  } else {
    printf("FAILURE unary mod2 padm 16bit\n");
    ret = EXIT_FAILURE;
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );

  return ret;
}

LIBXSMM_INLINE
int test_norm_padnm_mod2_16bit( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  unsigned short *in;
  unsigned short *out, *out_gold;
  libxsmm_blasint i, j;
  unsigned int s;
  int ret = EXIT_SUCCESS;
  libxsmm_blasint Nn = ((N%2) == 0) ? N : N+1;
  libxsmm_blasint Mn = ((M%2) == 0) ? M : M+1;

  libxsmm_meltw_unary_param unary_param /*= { 0 }*/;
  libxsmm_meltw_unary_shape unary_shape /*= { 0 }*/;
  libxsmm_meltw_unary_type  unary_type;
  libxsmm_meltwfunction_unary unary_kernel;

  if ( M > ldi ) {
    fprintf( stderr, "test_norm_padm_mod2_16bit: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if ( Mn > ldo ) {
    fprintf( stderr, "test_norm_padm_mod2_16bit: ldo needs to be equal to or bigger than M\n");
    exit(-1);
  }

  in       = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*ldi*N,  64);
  out      = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*ldo*Nn, 64);
  out_gold = (unsigned short*)libxsmm_aligned_malloc( sizeof(unsigned short)*ldo*Nn, 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      in[(i*ldi)+j] = (unsigned short)(((i*ldi)+j)%112);
    }
  }

  /* init out */
  for ( i = 0; i < ldo*Nn; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < ldo*Nn; ++i ) {
    out_gold[i] = 0;
  }

  /* to vnni */
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M ; ++i ) {
      out_gold[(j*ldo)+i] = in[(j*ldi)+i];
    }
  }

  unary_shape.m = M;
  unary_shape.n = N;
  unary_shape.ldi = ldi;
  unary_shape.ldo = ldo;
  unary_shape.in0_type = LIBXSMM_DATATYPE_BF16;
  unary_shape.out_type = LIBXSMM_DATATYPE_BF16;
  unary_shape.comp_type = LIBXSMM_DATATYPE_BF16;

  /* use jited transpose */
  unary_param.in.primary  = (void*)in;
  unary_param.out.primary = (void*)out;
  unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2;
  unary_kernel = libxsmm_dispatch_meltw_unary_v2( unary_type, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for PADNM_MOD2 TPP. Bailing...!\n");
    exit(LIBXSMM_EXIT_ERROR);
  }
  unary_kernel( &unary_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < Nn; ++i ) {
    for ( j = 0; j < Mn; ++j ) {
      if ( out_gold[(i*ldo)+j] != out[(i*ldo)+j] ) {
        printf("error at possition i=%i, j=%i, %i %i\n", i, j, out_gold[(i*ldo)+j], out[(i*ldo)+j]);
        s = 1;
      }
    }
  }
  if ( s == 0 ) {
    printf("SUCCESS unary mod2 padnm 16bit\n");
  } else {
    printf("FAILURE unary mod2 padnm 16bit\n");
    ret = EXIT_FAILURE;
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );

  return ret;
}

LIBXSMM_INLINE
int test_norm_padn_mod4_08bit( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  unsigned char *in;
  unsigned char *out, *out_gold;
  libxsmm_blasint i, j;
  unsigned int s;
  int ret = EXIT_SUCCESS;
  libxsmm_blasint Nn = ((N%4) == 0) ? N : LIBXSMM_UP(N, 4);

  libxsmm_meltw_unary_param unary_param /*= { 0 }*/;
  libxsmm_meltw_unary_shape unary_shape /*= { 0 }*/;
  libxsmm_meltw_unary_type  unary_type;
  libxsmm_meltwfunction_unary unary_kernel;

  if ( M > ldi ) {
    fprintf( stderr, "test_norm_padn_mod4_08bit: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if ( M > ldo ) {
    fprintf( stderr, "test_norm_padn_mod4_08bit: ldo needs to be equal to or bigger than M\n");
    exit(-1);
  }

  in       = (unsigned char*)libxsmm_aligned_malloc( sizeof(unsigned char)*ldi*N,  64);
  out      = (unsigned char*)libxsmm_aligned_malloc( sizeof(unsigned char)*ldo*Nn, 64);
  out_gold = (unsigned char*)libxsmm_aligned_malloc( sizeof(unsigned char)*ldo*Nn, 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      in[(i*ldi)+j] = (unsigned char)(((i*ldi)+j)%112);
    }
  }

  /* init out */
  for ( i = 0; i < ldo*Nn; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < ldo*Nn; ++i ) {
    out_gold[i] = 0;
  }

  /* to vnni */
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M ; ++i ) {
      out_gold[(j*ldo)+i] = in[(j*ldi)+i];
    }
  }

  unary_shape.m = M;
  unary_shape.n = N;
  unary_shape.ldi = ldi;
  unary_shape.ldo = ldo;
  unary_shape.in0_type = LIBXSMM_DATATYPE_I8;
  unary_shape.out_type = LIBXSMM_DATATYPE_I8;
  unary_shape.comp_type = LIBXSMM_DATATYPE_I8;

  /* use jited transpose */
  unary_param.in.primary  = (void*)in;
  unary_param.out.primary = (void*)out;
  unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD4;
  unary_kernel = libxsmm_dispatch_meltw_unary_v2( unary_type, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for PADN_MOD2 TPP. Bailing...!\n");
    exit(LIBXSMM_EXIT_ERROR);
  }
  unary_kernel( &unary_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < Nn; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if ( out_gold[(i*ldo)+j] != out[(i*ldo)+j] ) {
        printf("error at possition i=%i, j=%i, %i %i\n", i, j, out_gold[(i*ldo)+j], out[(i*ldo)+j]);
        s = 1;
      }
    }
  }
  if ( s == 0 ) {
    printf("SUCCESS unary mod4 padn 08bit\n");
  } else {
    printf("FAILURE unary mod4 padn 08bit\n");
    ret = EXIT_FAILURE;
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );

  return ret;
}

LIBXSMM_INLINE
int test_norm_padm_mod4_08bit( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  unsigned char *in;
  unsigned char *out, *out_gold;
  libxsmm_blasint i, j;
  unsigned int s;
  int ret = EXIT_SUCCESS;
  libxsmm_blasint Mn = ((M%4) == 0) ? M : LIBXSMM_UP(M, 4);

  libxsmm_meltw_unary_param unary_param /*= { 0 }*/;
  libxsmm_meltw_unary_shape unary_shape /*= { 0 }*/;
  libxsmm_meltw_unary_type  unary_type;
  libxsmm_meltwfunction_unary unary_kernel;

  if ( M > ldi ) {
    fprintf( stderr, "test_norm_padm_mod4_08bit: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if ( Mn > ldo ) {
    fprintf( stderr, "test_norm_padm_mod4_08bit: ldo needs to be equal to or bigger than M\n");
    exit(-1);
  }

  in       = (unsigned char*)libxsmm_aligned_malloc( sizeof(unsigned char)*ldi*N, 64);
  out      = (unsigned char*)libxsmm_aligned_malloc( sizeof(unsigned char)*ldo*N, 64);
  out_gold = (unsigned char*)libxsmm_aligned_malloc( sizeof(unsigned char)*ldo*N, 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      in[(i*ldi)+j] = (unsigned char)(((i*ldi)+j)%112);
    }
  }

  /* init out */
  for ( i = 0; i < ldo*N; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < ldo*N; ++i ) {
    out_gold[i] = 0;
  }

  /* to vnni */
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M ; ++i ) {
      out_gold[(j*ldo)+i] = in[(j*ldi)+i];
    }
  }

  unary_shape.m = M;
  unary_shape.n = N;
  unary_shape.ldi = ldi;
  unary_shape.ldo = ldo;
  unary_shape.in0_type = LIBXSMM_DATATYPE_I8;
  unary_shape.out_type = LIBXSMM_DATATYPE_I8;
  unary_shape.comp_type = LIBXSMM_DATATYPE_I8;

  /* use jited transpose */
  unary_param.in.primary  = (void*)in;
  unary_param.out.primary = (void*)out;
  unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD4;
  unary_kernel = libxsmm_dispatch_meltw_unary_v2( unary_type, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for PADM_MOD4 TPP. Bailing...!\n");
    exit(LIBXSMM_EXIT_ERROR);
  }
  unary_kernel( &unary_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < Mn; ++j ) {
      if ( out_gold[(i*ldo)+j] != out[(i*ldo)+j] ) {
        printf("error at possition i=%i, j=%i, %i %i\n", i, j, out_gold[(i*ldo)+j], out[(i*ldo)+j]);
        s = 1;
      }
    }
  }
  if ( s == 0 ) {
    printf("SUCCESS unary mod4 padm 08bit\n");
  } else {
    printf("FAILURE unary mod4 padm 08bit\n");
    ret = EXIT_FAILURE;
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );

  return ret;
}

LIBXSMM_INLINE
int test_norm_padnm_mod4_08bit( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  unsigned char *in;
  unsigned char *out, *out_gold;
  libxsmm_blasint i, j;
  unsigned int s;
  int ret = EXIT_SUCCESS;
  libxsmm_blasint Nn = ((N%4) == 0) ? N : LIBXSMM_UP(N, 4);
  libxsmm_blasint Mn = ((M%4) == 0) ? M : LIBXSMM_UP(M, 4);

  libxsmm_meltw_unary_param unary_param /*= { 0 }*/;
  libxsmm_meltw_unary_shape unary_shape /*= { 0 }*/;
  libxsmm_meltw_unary_type  unary_type;
  libxsmm_meltwfunction_unary unary_kernel;

  if ( M > ldi ) {
    fprintf( stderr, "test_norm_padm_mod4_08bit: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if ( Mn > ldo ) {
    fprintf( stderr, "test_norm_padm_mod4_08bit: ldo needs to be equal to or bigger than M\n");
    exit(-1);
  }

  in       = (unsigned char*)libxsmm_aligned_malloc( sizeof(unsigned char)*ldi*N,  64);
  out      = (unsigned char*)libxsmm_aligned_malloc( sizeof(unsigned char)*ldo*Nn, 64);
  out_gold = (unsigned char*)libxsmm_aligned_malloc( sizeof(unsigned char)*ldo*Nn, 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      in[(i*ldi)+j] = (unsigned char)(((i*ldi)+j)%112);
    }
  }

  /* init out */
  for ( i = 0; i < ldo*Nn; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < ldo*Nn; ++i ) {
    out_gold[i] = 0;
  }

  /* to vnni */
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M ; ++i ) {
      out_gold[(j*ldo)+i] = in[(j*ldi)+i];
    }
  }

  unary_shape.m = M;
  unary_shape.n = N;
  unary_shape.ldi = ldi;
  unary_shape.ldo = ldo;
  unary_shape.in0_type = LIBXSMM_DATATYPE_I8;
  unary_shape.out_type = LIBXSMM_DATATYPE_I8;
  unary_shape.comp_type = LIBXSMM_DATATYPE_I8;

  /* use jited transpose */
  unary_param.in.primary  = (void*)in;
  unary_param.out.primary = (void*)out;
  unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD4;
  unary_kernel = libxsmm_dispatch_meltw_unary_v2( unary_type, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for PADNM_MOD4 TPP. Bailing...!\n");
    exit(LIBXSMM_EXIT_ERROR);
  }
  unary_kernel( &unary_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < Nn; ++i ) {
    for ( j = 0; j < Mn; ++j ) {
      if ( out_gold[(i*ldo)+j] != out[(i*ldo)+j] ) {
        printf("error at possition i=%i, j=%i, %i %i\n", i, j, out_gold[(i*ldo)+j], out[(i*ldo)+j]);
        s = 1;
      }
    }
  }
  if ( s == 0 ) {
    printf("SUCCESS unary mod4 padnm 08bit\n");
  } else {
    printf("FAILURE unary mod4 padnm 08bit\n");
    ret = EXIT_FAILURE;
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );

  return ret;
}

int main( int argc, char* argv[] ) {
  char* dt = NULL;
  libxsmm_blasint op;
  libxsmm_blasint M;
  libxsmm_blasint N;
  libxsmm_blasint ldi;
  libxsmm_blasint ldo;
  libxsmm_datatype dtype;

  int ret = EXIT_FAILURE;

  if ( argc != 7 ) {
    printf(" Error! Usage: %s [A/T/V/R/X/Y/Z/B/I] [F64/I64/F32/I32/BF16/F16/I16/BF8/I8] [M] [N] [ldi] [ldo]\n", argv[0] );
    exit(-1);
  }

  op  = *(argv[1]);
  dt    = argv[2];
  M     = atoi(argv[3]);
  N     = atoi(argv[4]);
  ldi   = atoi(argv[5]);
  ldo   = atoi(argv[6]);

  dtype = char_to_libxsmm_datatype( dt );
  if ( dtype == LIBXSMM_DATATYPE_UNSUPPORTED ) {
    printf(" Error! Usage: %s [A/T/V/R/X/Y/Z/B/I] [F64/I64/F32/I32/BF16/F16/I16/BF8/I8] [M] [N] [ldi] [ldo]\n", argv[0] );
    exit(-1);
  }

  if ( op == 'T' ) {
    printf("Testing %s Norm to Norm Transpose - M=%i, N=%i, LDI=%i, LDO=%i \n", libxsmm_get_typename(dtype), M, N, ldi, ldo);
    ret = test_normal_to_normalT( M, N, ldi, ldo, dtype );
  } else if ( op == 'R' && ( dtype == LIBXSMM_DATATYPE_BF16 || dtype == LIBXSMM_DATATYPE_F16 || dtype == LIBXSMM_DATATYPE_I16 ) ) {
    printf("Testing 16bit VNNI2 to VNNI2 Transpose - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_vnni2_to_vnni2T_16bit( M, N, ldi, ldo );
  } else if ( op == 'R' && ( dtype == LIBXSMM_DATATYPE_I8 || dtype == LIBXSMM_DATATYPE_BF8 ) ) {
    printf("Testing 08bit VNNI to VNNI4 Transpose - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_vnni4_to_vnni4T_08bit( M, N, ldi, ldo );
  } else if ( op == 'I' && ( dtype == LIBXSMM_DATATYPE_I16 || dtype == LIBXSMM_DATATYPE_BF16 ) ) {
    printf("Testing 16bit VNNI to VNNI4 Transpose - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_vnni4_to_vnni4T_16bit( M, N, ldi, ldo );
  } else if ( op == 'V' && ( dtype == LIBXSMM_DATATYPE_BF16 || dtype == LIBXSMM_DATATYPE_F16 || dtype == LIBXSMM_DATATYPE_I16 ) ) {
    printf("Testing 16bit NORM to VNNI2 Reformat - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_norm_to_vnni2_16bit( M, N, ldi, ldo );
  } else if ( op == 'V' && ( dtype == LIBXSMM_DATATYPE_I8 || dtype == LIBXSMM_DATATYPE_BF8 ) ) {
    printf("Testing 08bit NORM to VNNI4 Reformat - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_norm_to_vnni4_08bit( M, N, ldi, ldo );
  } else if ( op == 'A' && ( dtype == LIBXSMM_DATATYPE_I16 || dtype == LIBXSMM_DATATYPE_BF16 ) ) {
    printf("Testing 16bit NORM to VNNI4T Reformat - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_norm_to_vnni4T_16bit( M, N, ldi, ldo );
  } else if ( op == 'B' && ( dtype == LIBXSMM_DATATYPE_I16 || dtype == LIBXSMM_DATATYPE_BF16 ) ) {
    printf("Testing 16bit NORM to VNNI4 Reformat - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_norm_to_vnni4_16bit( M, N, ldi, ldo );
  } else if ( op == 'N' && ( dtype == LIBXSMM_DATATYPE_I8 || dtype == LIBXSMM_DATATYPE_BF8 ) ) {
    printf("Testing 08bit VNNI4 to NORM Reformat - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_vnni4_to_norm_08bit( M, N, ldi, ldo );
  } else if ( op == 'W' && ( dtype == LIBXSMM_DATATYPE_I8 || dtype == LIBXSMM_DATATYPE_BF8 ) ) {
    printf("Testing 08bit VNNI4 to VNNI2 Reformat - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_vnni4_to_vnni2_08bit( M, N, ldi, ldo );
  } else if ( op == 'X' && ( dtype == LIBXSMM_DATATYPE_BF16 || dtype == LIBXSMM_DATATYPE_F16 || dtype == LIBXSMM_DATATYPE_I16 ) ) {
    printf("Testing 16bit NORM PADN Mod2 Reformat - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_norm_padn_mod2_16bit( M, N, ldi, ldo );
  } else if ( op == 'Y' && ( dtype == LIBXSMM_DATATYPE_BF16 || dtype == LIBXSMM_DATATYPE_F16 || dtype == LIBXSMM_DATATYPE_I16 ) ) {
    printf("Testing 16bit NORM PADM Mod2 Reformat - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_norm_padm_mod2_16bit( M, N, ldi, ldo );
  } else if ( op == 'Z' && ( dtype == LIBXSMM_DATATYPE_BF16 || dtype == LIBXSMM_DATATYPE_F16 || dtype == LIBXSMM_DATATYPE_I16 ) ) {
    printf("Testing 16bit NORM PADNM Mod2 Reformat - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_norm_padnm_mod2_16bit( M, N, ldi, ldo );
  } else if ( op == 'X' && ( dtype == LIBXSMM_DATATYPE_I8 || dtype == LIBXSMM_DATATYPE_BF8 ) ) {
    printf("Testing 08bit NORM PADN Mod4 Reformat - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_norm_padn_mod4_08bit( M, N, ldi, ldo );
  } else if ( op == 'Y' && ( dtype == LIBXSMM_DATATYPE_I8 || dtype == LIBXSMM_DATATYPE_BF8 ) ) {
    printf("Testing 08bit NORM PADM Mod4 Reformat - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_norm_padm_mod4_08bit( M, N, ldi, ldo );
  } else if ( op == 'Z' && ( dtype == LIBXSMM_DATATYPE_I8 || dtype == LIBXSMM_DATATYPE_BF8 ) ) {
    printf("Testing 08bit NORM PADNM Mod4 Reformat - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_norm_padnm_mod4_08bit( M, N, ldi, ldo );
   } else {
    printf(" Case not implemented! Usage: %s [A/T/V/R/N/W/X/Y/Z/B/I] [F64/I64/F32/I32/BF16/F16/I16/BF8/I8] [M] [N] [ldi] [ldo]\n", argv[0] );
    exit(-1);
  }

  return ret;
}
