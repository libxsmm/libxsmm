/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
*               Friedrich Schiller University Jena - All rights reserved.     *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.), Antonio Noack (FSU Jena)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "eltwise_common.h"

void relu_fwd_gold(const libxsmm_blasint M, const libxsmm_blasint N, const libxsmm_blasint ldi, const libxsmm_blasint ldo, const libxsmm_blasint ldo_mask, const void *in, void *out, const float alpha, unsigned char *out_mask, const unsigned char type, const libxsmm_datatype dtype_in, const libxsmm_datatype dtype_out, const libxsmm_datatype dtype_comp) {
  libxsmm_blasint i, j;

  if ( (dtype_in == LIBXSMM_DATATYPE_F32) && (dtype_out == LIBXSMM_DATATYPE_F32) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    const float* f_in = (const float*)in;
    float* f_out = (float*)out;
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        if ( type == 0 ) {
          f_out[(j*ldo) + i] = ( f_in[(j*ldi) + i] < 0.0f ) ? 0.0f : f_in[(j*ldi) + i];
        } else if ( type == 1 ) {
          f_out[(j*ldo) + i] = ( f_in[(j*ldi) + i] < 0.0f ) ? alpha*f_in[(j*ldi) + i] : f_in[(j*ldi) + i];
        } else if ( type == 2 ) {
          f_out[(j*ldo) + i] = ( f_in[(j*ldi) + i] < 0.0f ) ? alpha * (expf(f_in[(j*ldi) + i])-1.0f) : f_in[(j*ldi) + i];
        }
        if ( type != 2) {
          out_mask[(j*ldo_mask) + i/8] |= (unsigned char)(( f_in[(j*ldi) + i] < 0.0f ) ? 0x0 : (1 << (i%8)) );
        }
      }
    }
  } else if ( (dtype_in == LIBXSMM_DATATYPE_BF16) && (dtype_out == LIBXSMM_DATATYPE_BF16) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    const libxsmm_bfloat16* bf_in = (const libxsmm_bfloat16*)in;
    libxsmm_bfloat16* bf_out = (libxsmm_bfloat16*)out;
    float in_value, out_value;
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        libxsmm_convert_bf16_f32( &(bf_in[(j*ldi) + i]), &in_value, 1 );
        if ( type == 0 ) {
          out_value = ( in_value < 0.0f ) ? 0.0f : in_value;
        } else if ( type == 1 ) {
          out_value = ( in_value < 0.0f ) ? alpha*in_value : in_value;
        } else if ( type == 2 ) {
          out_value = ( in_value < 0.0f ) ? alpha*(expf(in_value)-1.0f) : in_value;
        }
        if ( type != 2) {
          out_mask[(j*ldo_mask) + i/8] |= (unsigned char)(( in_value < 0.0f ) ? 0x0 : (1 << (i%8)) );
        }
        libxsmm_rne_convert_fp32_bf16(&out_value, &(bf_out[(j*ldo) + i]), 1);
      }
    }
  } else if ( (dtype_in == LIBXSMM_DATATYPE_F32) && (dtype_out == LIBXSMM_DATATYPE_BF16) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    const float* f_in = (const float*)in;
    libxsmm_bfloat16* bf_out = (libxsmm_bfloat16*)out;
    float out_value;
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        if ( type == 0 ) {
          out_value = ( f_in[(j*ldi) + i] < 0.0f ) ? 0.0f : f_in[(j*ldi) + i];
        } else if ( type == 1 ) {
          out_value = ( f_in[(j*ldi) + i] < 0.0f ) ? alpha*f_in[(j*ldi) + i] : f_in[(j*ldi) + i];
        } else if ( type == 2 ) {
          out_value = ( f_in[(j*ldi) + i] < 0.0f ) ? alpha * (expf(f_in[(j*ldi) + i])-1.0f) : f_in[(j*ldi) + i];
        }
        if ( type != 2) {
          out_mask[(j*ldo_mask) + i/8] |= (unsigned char)(( f_in[(j*ldi) + i] < 0.0f ) ? 0x0 : (1 << (i%8)) );
        }
        libxsmm_rne_convert_fp32_bf16(&out_value, &(bf_out[(j*ldo) + i]), 1);
      }
    }
  } else if ( (dtype_in == LIBXSMM_DATATYPE_BF16) && (dtype_out == LIBXSMM_DATATYPE_F32) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    const libxsmm_bfloat16* bf_in = (const libxsmm_bfloat16*)in;
    float* f_out = (float*)out;
    float in_value;
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        libxsmm_convert_bf16_f32( &(bf_in[(j*ldi) + i]), &in_value, 1 );
        if ( type == 0 ) {
          f_out[(j*ldo) + i] = ( in_value < 0.0f ) ? 0.0f : in_value;
        } else if ( type == 1 ) {
          f_out[(j*ldo) + i] = ( in_value < 0.0f ) ? alpha*in_value : in_value;
        } else if ( type == 2 ) {
          f_out[(j*ldo) + i] = ( in_value < 0.0f ) ? alpha*(expf(in_value)-1.0f) : in_value;
        }
        if ( type != 2) {
          out_mask[(j*ldo_mask) + i/8] |= (unsigned char)(( in_value < 0.0f ) ? 0x0 : (1 << (i%8)) );
        }
      }
    }
  } else {
    /* shouldn't happen */
  }
}

void relu_bwd_gold(const libxsmm_blasint M, const libxsmm_blasint N, const libxsmm_blasint ldi, const libxsmm_blasint ldo, const void *in, void *out, float alpha, const void *out_fwd, const unsigned char *mask, const unsigned char type, const libxsmm_datatype dtype_in, const libxsmm_datatype dtype_out, const libxsmm_datatype dtype_comp) {
  size_t i, j;

  if ( (dtype_in == LIBXSMM_DATATYPE_F32) && (dtype_out == LIBXSMM_DATATYPE_F32) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    const float* f_in = (const float*)in;
    float* f_out = (float*)out;
    const float* f_out_fwd = (float*)out_fwd;
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        if ( type == 0 ) {
          f_out[(j*ldo) + i] = ( mask[(j*ldi) + i] == 0 ) ? f_in[(j*ldi) + i] : 0.0f;
        } else if ( type == 1 ) {
          f_out[(j*ldo) + i] = ( mask[(j*ldi) + i] == 0 ) ? f_in[(j*ldi) + i] : alpha*f_in[(j*ldi) + i];
        } else if ( type == 2 ) {
          f_out[(j*ldo) + i] = ( f_out_fwd[(j*ldi) + i] > 0 ) ? f_in[(j*ldi) + i] : f_in[(j*ldi) + i] * (f_out_fwd[(j*ldi) + i] + alpha) ;
        }
      }
    }
  } else if ( (dtype_in == LIBXSMM_DATATYPE_BF16) && (dtype_out == LIBXSMM_DATATYPE_BF16) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    const libxsmm_bfloat16* bf_in = (const libxsmm_bfloat16*)in;
    libxsmm_bfloat16* bf_out = (libxsmm_bfloat16*)out;
    const libxsmm_bfloat16* bf_out_fwd = (libxsmm_bfloat16*)out_fwd;
    float in_value, out_value, out_fwd_value;
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        libxsmm_convert_bf16_f32( &(bf_in[(j*ldi) + i]), &in_value, 1 );
        if ( type == 0 ) {
          out_value = ( mask[(j*ldi) + i] == 0 ) ? in_value : 0.0f;
        } else if ( type == 1 ) {
          out_value = ( mask[(j*ldi) + i] == 0 ) ? in_value : alpha*in_value;
        } else if ( type == 2 ) {
          libxsmm_convert_bf16_f32( &(bf_out_fwd[(j*ldi) + i]), &out_fwd_value, 1 );
          out_value = ( out_fwd_value > 0 ) ? in_value : in_value * (out_fwd_value + alpha) ;
        }
        libxsmm_rne_convert_fp32_bf16(&out_value, &(bf_out[(j*ldo) + i]), 1);
      }
    }
  } else if ( (dtype_in == LIBXSMM_DATATYPE_F32) && (dtype_out == LIBXSMM_DATATYPE_BF16) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    const float* f_in = (const float*)in;
    libxsmm_bfloat16* bf_out = (libxsmm_bfloat16*)out;
    const float* f_out_fwd = (float*)out_fwd;
    float out_value;
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        if ( type == 0 ) {
          out_value = ( mask[(j*ldi) + i] == 0 ) ? f_in[(j*ldi) + i] : 0.0f;
        } else if ( type == 1 ) {
          out_value = ( mask[(j*ldi) + i] == 0 ) ? f_in[(j*ldi) + i] : alpha*f_in[(j*ldi) + i];
        } else if ( type == 2 ) {
          out_value = ( f_out_fwd[(j*ldi) + i] > 0 ) ? f_in[(j*ldi) + i] : f_in[(j*ldi) + i] * (f_out_fwd[(j*ldi) + i] + alpha) ;
        }
        libxsmm_rne_convert_fp32_bf16(&out_value, &(bf_out[(j*ldo) + i]), 1);
      }
    }
  } else if ( (dtype_in == LIBXSMM_DATATYPE_BF16) && (dtype_out == LIBXSMM_DATATYPE_F32) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    const libxsmm_bfloat16* bf_in = (const libxsmm_bfloat16*)in;
    float* f_out = (float*)out;
    const libxsmm_bfloat16* bf_out_fwd = (libxsmm_bfloat16*)out_fwd;
    float in_value, out_fwd_value;
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        libxsmm_convert_bf16_f32( &(bf_in[(j*ldi) + i]), &in_value, 1 );
        if ( type == 0 ) {
          f_out[(j*ldo) + i] = ( mask[(j*ldi) + i] == 0 ) ? in_value : 0.0f;
        } else if ( type == 1 ) {
          f_out[(j*ldo) + i] = ( mask[(j*ldi) + i] == 0 ) ? in_value : alpha*in_value;
        } else if ( type == 2 ) {
          libxsmm_convert_bf16_f32( &(bf_out_fwd[(j*ldi) + i]), &out_fwd_value, 1 );
          f_out[(j*ldo) + i] = ( out_fwd_value > 0 ) ? in_value : in_value * (out_fwd_value + alpha) ;
        }
      }
    }
  } else {
    /* shouldn't happen */
  }
}

int test_relu_fwd( const libxsmm_blasint bitm, const libxsmm_blasint M, const libxsmm_blasint N, const libxsmm_blasint ldi, const libxsmm_blasint ldo, const unsigned char type,
                   const libxsmm_datatype dtype_in, const libxsmm_datatype dtype_out, const libxsmm_datatype dtype_comp ) {
  char *in;
  char *out, *out_gold;
  unsigned char *mask, *mask_gold;
  libxsmm_blasint i, j;
  unsigned int s;
  float alpha = 0.1f;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param = { 0 };
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_matdiff_info norms_out;
  libxsmm_meltw_unary_shape unary_shape = libxsmm_create_meltw_unary_shape( M, N, ldi, ldo, dtype_in, dtype_out, dtype_comp );
  libxsmm_meltwfunction_unary unary_kernel;
  libxsmm_blasint mask_ld = (bitm == 0) ? ldo : ((ldo+15)-((ldo+15)%16))/8;
  libxsmm_meltw_unary_type unary_type;

  if ( M > ldi ) {
    fprintf( stderr, "test_relu_fwd %i %i %i: ldi needs to be equal to or bigger than M\n", dtype_in, dtype_out, dtype_comp);
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_relu_fwd %i %i %i: ldo needs to be equal to or bigger than N\n", dtype_in, dtype_out, dtype_comp);
    exit(-1);
  }

  libxsmm_rng_set_seed(1);

  in        = (char*) libxsmm_aligned_malloc((size_t)LIBXSMM_TYPESIZE(dtype_in)*N*ldi, 64);
  out       = (char*) libxsmm_aligned_malloc((size_t)LIBXSMM_TYPESIZE(dtype_out)*N*ldo, 64);
  out_gold  = (char*) libxsmm_aligned_malloc((size_t)LIBXSMM_TYPESIZE(dtype_out)*N*ldo, 64);
  mask      = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);
  mask_gold = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);

  /* init in */
  init_random_matrix( dtype_in,  in,       1, ldi, N, 1 );
  init_zero_matrix(   dtype_out, out,      1, ldo, N );
  init_zero_matrix(   dtype_out, out_gold, 1, ldo, N );
  init_zero_matrix(   LIBXSMM_DATATYPE_I8, mask,      1, mask_ld, N );
  init_zero_matrix(   LIBXSMM_DATATYPE_I8, mask_gold, 1, mask_ld, N );

  /* compute out_gold */
  relu_fwd_gold( M, N, ldi, ldo, mask_ld, in, out_gold, alpha, mask_gold, type, dtype_in, dtype_out, dtype_comp );

  unary_param.op.primary = (void*)(&alpha);
  unary_param.in.primary = (void*)in;
  unary_param.out.primary = (void*)out;
  unary_param.out.secondary = (bitm == 0) ? NULL : (void*)mask;
  unary_flags = (bitm == 0) ? LIBXSMM_MELTW_FLAG_UNARY_NONE : LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
  if ( type == 0 ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_RELU;
  } else if ( type == 1 ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU;
  } else if ( type == 2 ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_ELU;
  } else {
    fprintf( stderr, "JIT for UNARY TPP. Bailing...!\n");
    exit(-1);
  }

  unary_kernel = libxsmm_dispatch_meltw_unary_v2( unary_type, unary_shape, unary_flags );

  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for UNARY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel( &unary_param );

  /* compare result */
  norms_out = check_matrix( dtype_out, out_gold, out, ldo, M, N );
  printf("##########################################\n");
  printf("#   Correctness  - Output                #\n");
  printf("##########################################\n");
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

  if ( (dtype_in == LIBXSMM_DATATYPE_F32) && (dtype_out == LIBXSMM_DATATYPE_F32) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    if ( norms_out.normf_rel > 0.00001 ) {
      ret = EXIT_FAILURE;
    }
  } else {
    if ( norms_out.normf_rel > 0.007 ) {
      ret = EXIT_FAILURE;
    }
  }

  if ( type != 2 ) {
    s = 0;
    if ( bitm != 0 ) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < M/8; ++j ) {
          if ( mask_gold[(i*mask_ld)+j] != mask[(i*mask_ld)+j] ) {
            printf("error at possition i=%i, j=%i, %u, %u\n", i, j*8, mask[(i*mask_ld)+j], mask_gold[(i*mask_ld)+j]);
            s = 1;
          }
#if 0
        else {
          printf("correct at possition i=%i, j=%i, %u, %u\n", i, j*8, mask[(i*mask_ld)+j], mask_gold[(i*mask_ld)+j]);
        }
#endif
        }
      }
      if ( s == 0 ) {
        printf("SUCCESS mask\n");
      } else {
        printf("FAILURE mask\n");
        ret = EXIT_FAILURE;
      }
    }
  }

  benchmark_unary(unary_type, unary_shape, unary_flags, unary_param);

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
  libxsmm_free( mask );
  libxsmm_free( mask_gold );

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS unary relu fwd %i %i %i\n", dtype_in, dtype_out, dtype_comp);
  } else {
    printf("FAILURE unary relu fwd %i %i %i\n", dtype_in, dtype_out, dtype_comp);
  }

  return ret;
}

int test_relu_bwd( const libxsmm_blasint M, const libxsmm_blasint N, const libxsmm_blasint ldi, const libxsmm_blasint ldo, const unsigned char type,
                   const libxsmm_datatype dtype_in, const libxsmm_datatype dtype_out, const libxsmm_datatype dtype_comp ) {
  char *in;
  char *out, *out_gold;
  char *out_fwd;
  unsigned char *mask_bit;
  unsigned char *mask_gold;
  float alpha = 0.1f;
  int ret = EXIT_SUCCESS;
  size_t i,j;
  libxsmm_meltw_unary_param unary_param = { 0 };
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_matdiff_info norms_out;
  libxsmm_meltw_unary_shape unary_shape = libxsmm_create_meltw_unary_shape( M, N, ldi, ldo, dtype_in, dtype_out, dtype_comp );
  libxsmm_meltwfunction_unary unary_kernel;
  libxsmm_blasint mask_ld = ((ldi+15)-((ldi+15)%16))/8;
  libxsmm_meltw_unary_type unary_type;

  if ( M > ldi ) {
    fprintf( stderr, "test_relu_bwd %i %i %i: ldi needs to be equal to or bigger than M\n", dtype_in, dtype_out, dtype_comp);
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_relu_bwd %i %i %i: ldo needs to be equal to or bigger than N\n", dtype_in, dtype_out, dtype_comp);
    exit(-1);
  }

  libxsmm_rng_set_seed(1);

  in        = (char*) libxsmm_aligned_malloc( LIBXSMM_TYPESIZE(dtype_in)*N*ldi,   64);
  out_fwd   = (char*) libxsmm_aligned_malloc( LIBXSMM_TYPESIZE(dtype_in)*N*ldi,   64);
  out       = (char*) libxsmm_aligned_malloc( LIBXSMM_TYPESIZE(dtype_out)*N*ldo,   64);
  out_gold  = (char*) libxsmm_aligned_malloc( LIBXSMM_TYPESIZE(dtype_out)*N*ldo,   64);
  mask_bit  = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);
  mask_gold = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*ldi, 64);

  /* init in, out */
  init_random_matrix( dtype_in,  in,       1, ldi, N, 1 );
  init_random_matrix( dtype_in,  out_fwd,  1, ldi, N, 1 );
  init_zero_matrix(   dtype_out, out,      1, ldo, N );
  init_zero_matrix(   dtype_out, out_gold, 1, ldo, N );

  /* mask fake init, will be fwd IRL */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < mask_ld; ++j ) {
      mask_bit[(i*mask_ld)+j] = 0xaa;
    }
  }
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      mask_gold[(i*ldi)+j] = ( j % 2 == 1 ) ? 0 : 1;
    }
  }

  /* compute out_gold */
  relu_bwd_gold( M, N, ldi, ldo, in, out_gold, alpha, out_fwd, mask_gold, type, dtype_in, dtype_out, dtype_comp );

  /* use jited relu */
  unary_param.op.primary    = (void*)(&alpha);
  unary_param.in.primary    = (void*)in;
  unary_param.in.secondary  = ( type == 2 ) ? (void*)out_fwd : (void*)mask_bit;
  unary_param.out.primary   = (void*)out;

  if ( type == 0 ) {
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_RELU_INV;
  } else if ( type == 1 ) {
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV;
  } else if ( type == 2 ) {
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_ELU_INV;
  } else {
    fprintf( stderr, "JIT for UNARY TPP. Bailing...!\n");
    exit(-1);
  }

  unary_kernel = libxsmm_dispatch_meltw_unary_v2( unary_type, unary_shape, unary_flags );

  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for UNARY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel( &unary_param );

  /* compare result */
  norms_out = check_matrix( dtype_out, out_gold, out, ldo, M, N );
  printf("##########################################\n");
  printf("#   Correctness  - Output                #\n");
  printf("##########################################\n");
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

  if ( (dtype_in == LIBXSMM_DATATYPE_F32) && (dtype_out == LIBXSMM_DATATYPE_F32) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    if ( norms_out.normf_rel > 0.00001 ) {
      ret = EXIT_FAILURE;
    }
  } else {
    if ( norms_out.normf_rel > 0.007 ) {
      ret = EXIT_FAILURE;
    }
  }

  libxsmm_free( out_fwd );
  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
  libxsmm_free( mask_bit );
  libxsmm_free( mask_gold );

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS unary relu bwd %i %i %i\n", dtype_in, dtype_out, dtype_comp);
  } else {
    printf("FAILURE unary relu bwd %i %i %i\n", dtype_in, dtype_out, dtype_comp);
  }

  return ret;
}

int main( int argc, char* argv[] ) {
  libxsmm_blasint dtype_in;
  libxsmm_blasint dtype_out;
  char op;
  char type;
  libxsmm_blasint bitm;
  libxsmm_blasint M;
  libxsmm_blasint N;
  libxsmm_blasint ldi;
  libxsmm_blasint ldo;
  unsigned char itype;
  int ret = EXIT_FAILURE;

  if ( argc != 10 ) {
    printf(" Error! Usage: %s [D/L/E] [F/B] [bitmask: 0/1] [prec_in: 4/2] [prec_out: 4/2] [M] [N] [ldi] [ldo]\n", argv[0] );
    exit(-1);
  }

  type      = *(argv[1]);
  op        = *(argv[2]);
  bitm      = atoi(argv[3]);
  dtype_in  = atoi(argv[4]);
  dtype_out = atoi(argv[5]);
  M         = atoi(argv[6]);
  N         = atoi(argv[7]);
  ldi       = atoi(argv[8]);
  ldo       = atoi(argv[9]);

  if (  op == 'B' && bitm == 0 && (type == 'D' || type == 'L') ) {
    printf("Backward needs masks!\n");
    return ret;
  }


  if ( type == 'D' ) {
    itype = 0;
    printf("Testing ReLU ");
  } else if ( type == 'L' ) {
    itype = 1;
    printf("Testing Leaky ReLU ");
  } else if ( type == 'E' ) {
    itype = 2;
#if 0
    bitm = 0;
    printf("Testing ELU (disabling bitmask support) ");
#else
    printf("Testing ELU ");
#endif
  } else {
    itype = 0;
    printf("Testing ReLU ");
  }

  if ( op == 'F' && dtype_in == 4 && dtype_out == 4  ) {
    printf("F32 F32 forward - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_relu_fwd( bitm, M, N, ldi, ldo, itype, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
  } else if ( op == 'F' && dtype_in == 2  && dtype_out == 2 ) {
    printf("BF16 BF16 forward - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_relu_fwd( bitm, M, N, ldi, ldo, itype, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32 );
  } else if ( op == 'F' && dtype_in == 4  && dtype_out == 2 ) {
    printf("F32 BF16 forward - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_relu_fwd( bitm, M, N, ldi, ldo, itype, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32 );
  } else if ( op == 'F' && dtype_in == 2  && dtype_out == 4 ) {
    printf("BF16 F32 forward - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_relu_fwd( bitm, M, N, ldi, ldo, itype, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
  } else if ( op == 'B' && dtype_in == 4 && dtype_out == 4 ) {
    printf("F32 F32 backward - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_relu_bwd( M, N, ldi, ldo, itype, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
  } else if ( op == 'B' && dtype_in == 2 && dtype_out == 2 ) {
    printf("BF16 BF16 backward - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_relu_bwd( M, N, ldi, ldo, itype, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32 );
  } else if ( op == 'B' && dtype_in == 4 && dtype_out == 2 ) {
    printf("F32 BF16 backward - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_relu_bwd( M, N, ldi, ldo, itype, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32 );
  } else if ( op == 'B' && dtype_in == 2 && dtype_out == 4 ) {
    printf("BF16 F32 backward - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_relu_bwd( M, N, ldi, ldo, itype, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
  } else {
    printf(" Not implemented case! Usage: %s [D/L/E] [F/B] [bitmask: 0/1] [prec_in: 4/2] [prec_out: 4/2] [M] [N] [ldi] [ldo]\n", argv[0] );
    exit(-1);
  }

  return ret;
}
