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

#include "eltwise_perf_tester.h"

#define TYPE_RELU 0
#define TYPE_LEAKY_RELU 1
#define TYPE_ELU 2

float upconvert_bf16(libxsmm_bfloat16 x) {
  union libxsmm_bfloat16_hp bf16_hp;
  bf16_hp.i[1] = x;
  bf16_hp.i[0] = 0;
  return bf16_hp.f;
}

void relu_fwd_f32_f32_gold(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, libxsmm_blasint ldo_mask, float *in, float *out, float alpha, unsigned char *out_mask, unsigned char type) {
  libxsmm_blasint i, j;
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M; ++i ) {
      if ( type == TYPE_RELU ) {
        out[(j*ldo) + i] = ( in[(j*ldi) + i] < 0.0f ) ? 0.0f : in[(j*ldi) + i];
      } else if ( type == TYPE_LEAKY_RELU ) {
        out[(j*ldo) + i] = ( in[(j*ldi) + i] < 0.0f ) ? alpha*in[(j*ldi) + i] : in[(j*ldi) + i];
      } else if ( type == TYPE_ELU ) {
        out[(j*ldo) + i] = ( in[(j*ldi) + i] < 0.0f ) ? alpha * (expf(in[(j*ldi) + i])-1.0) : in[(j*ldi) + i];
      }
      if ( type != 2) {
        out_mask[(j*ldo_mask) + i/8] |= (unsigned char)(( in[(j*ldi) + i] < 0.0f ) ? 0x0 : (1 << (i%8)) );
      }
    }
  }
}

void relu_fwd_bf16_bf16_gold(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, libxsmm_blasint ldo_mask, libxsmm_bfloat16 *in, libxsmm_bfloat16 *out, float alpha, unsigned char *out_mask, unsigned char type) {
  libxsmm_blasint i, j;
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M; ++i ) {
      if ( type == TYPE_RELU ) {
        out[(j*ldo) + i] = ( (in[(j*ldi) + i] & 0x8000) == 0x8000 ) ? 0 : in[(j*ldi) + i];
      } else if ( type == TYPE_LEAKY_RELU ) {
        union libxsmm_bfloat16_hp bf16_hp;
        union libxsmm_bfloat16_hp bf16_hp_out;
        bf16_hp.i[0] = 0;
        bf16_hp.i[1] = in[(j*ldi) + i];
        bf16_hp_out.f = ( (in[(j*ldi) + i] & 0x8000) == 0x8000 ) ? alpha*bf16_hp.f : bf16_hp.f;
        out[(j*ldo) + i] = bf16_hp_out.i[1];
      } else if ( type == TYPE_ELU ) {
        float in_f;
        libxsmm_bfloat16 res;
        union libxsmm_bfloat16_hp bf16_hp;
        bf16_hp.i[1] = in[(j*ldi) + i];
        bf16_hp.i[0] = 0;
        in_f = bf16_hp.f;
        in_f = alpha * (expf(in_f)-1.0);
        libxsmm_rne_convert_fp32_bf16( &in_f, &res, 1 );
        out[(j*ldo) + i] = ( (in[(j*ldi) + i] & 0x8000) == 0x8000 ) ? res : in[(j*ldi) + i];
      }
      if ( type != 2) {
        out_mask[(j*ldo_mask) + i/8] |= (unsigned char)(( (in[(j*ldi) + i] & 0x8000) == 0x8000 ) ? 0x0 : (1 << (i%8)) );
      }
    }
  }
}

void relu_fwd_f32_bf16_gold(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, libxsmm_blasint ldo_mask, float *in, libxsmm_bfloat16 *out, float alpha, unsigned char *out_mask, unsigned char type) {
  libxsmm_blasint i, j;
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M; ++i ) {
      union libxsmm_bfloat16_hp bf16_hp;
      bf16_hp.f = in[(j*ldi) + i];
      if ( type == TYPE_RELU ) {
        out[(j*ldo) + i] = ( in[(j*ldi) + i] < 0.0f ) ? 0 : bf16_hp.i[1];
      } else if ( type == TYPE_LEAKY_RELU ) {
        bf16_hp.f = ( in[(j*ldi) + i] < 0.0f ) ? alpha*bf16_hp.f : bf16_hp.f;
        out[(j*ldo) + i] = bf16_hp.i[1];
      } else if ( type == TYPE_ELU ) {
        float res;
        libxsmm_bfloat16 res_bf16;
        res = ( in[(j*ldi) + i] < 0.0f ) ? alpha * (expf(in[(j*ldi) + i])-1.0) : in[(j*ldi) + i] ;
        libxsmm_rne_convert_fp32_bf16( &res, &res_bf16, 1 );
        out[(j*ldo) + i] = res_bf16;
      }
      if ( type != 2) {
        out_mask[(j*ldo_mask) + i/8] |= (unsigned char)(( in[(j*ldi) + i] < 0.0f ) ? 0x0 : (1 << (i%8)) );
      }
    }
  }
}

void relu_fwd_bf16_f32_gold(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, libxsmm_blasint ldo_mask, libxsmm_bfloat16 *in, float *out, float alpha, unsigned char *out_mask, unsigned char type) {
  libxsmm_blasint i, j;
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M; ++i ) {
      union libxsmm_bfloat16_hp bf16_hp;
      bf16_hp.i[1] = in[(j*ldi) + i];
      bf16_hp.i[0] = 0;
      if ( type == TYPE_RELU ) {
        out[(j*ldo) + i] = ( bf16_hp.f < 0.0f ) ? 0 : bf16_hp.f;
      } else if ( type == TYPE_LEAKY_RELU ) {
        out[(j*ldo) + i] = ( bf16_hp.f < 0.0f ) ? alpha*bf16_hp.f : bf16_hp.f;
      } else if ( type == TYPE_ELU ) {
        float in_f;
        in_f = bf16_hp.f;
        in_f = alpha * (expf(in_f)-1.0);
        out[(j*ldo) + i] = ( bf16_hp.f < 0.0f ) ? in_f : bf16_hp.f;
      }
      if ( type != 2) {
        out_mask[(j*ldo_mask) + i/8] |= (unsigned char)(( bf16_hp.f < 0.0f ) ? 0x0 : (1 << (i%8)) );
      }
    }
  }
}

void relu_bwd_f32_f32_gold(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, float *in, float *out, float alpha, float *out_fwd, unsigned char *mask, unsigned char type) {
  libxsmm_blasint i, j;
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M; ++i ) {
      if ( type == TYPE_RELU ) {
        out[(j*ldo) + i] = ( mask[(j*ldi) + i] == 0 ) ? in[(j*ldi) + i] : 0.0f;
      } else if ( type == TYPE_LEAKY_RELU ) {
        out[(j*ldo) + i] = ( mask[(j*ldi) + i] == 0 ) ? in[(j*ldi) + i] : alpha*in[(j*ldi) + i];
      } else if ( type == TYPE_ELU ) {
        out[(j*ldo) + i] = ( out_fwd[(j*ldi) + i] > 0 ) ? in[(j*ldi) + i] : in[(j*ldi) + i] * (out_fwd[(j*ldi) + i] + alpha) ;
      }
    }
  }
}

void relu_bwd_bf16_bf16_gold(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, libxsmm_bfloat16 *in, libxsmm_bfloat16 *out, float alpha, libxsmm_bfloat16 *out_fwd, unsigned char *mask, unsigned char type) {
  libxsmm_blasint i, j;
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M; ++i ) {
      if ( type == TYPE_RELU ) {
        out[(j*ldo) + i] = ( mask[(j*ldi) + i] == 0 ) ? in[(j*ldi) + i] : 0;
      } else if ( type == TYPE_LEAKY_RELU ) {
        union libxsmm_bfloat16_hp bf16_hp;
        union libxsmm_bfloat16_hp bf16_hp_out;
        bf16_hp.i[0] = 0;
        bf16_hp.i[1] = in[(j*ldi) + i];
        bf16_hp_out.f = ( mask[(j*ldi) + i] == 0 ) ? bf16_hp.f : alpha*bf16_hp.f;
        out[(j*ldo) + i] = bf16_hp_out.i[1];
      } else if ( type == TYPE_ELU ) {
        union libxsmm_bfloat16_hp bf16_hp;
        libxsmm_bfloat16 res_bf16;
        float res;
        float comp;
        bf16_hp.i[1] = out_fwd[(j*ldi) + i];
        bf16_hp.i[0] = 0;
        comp =  bf16_hp.f;
        res =  bf16_hp.f + alpha;
        bf16_hp.i[1] = in[(j*ldi) + i];
        res = res * bf16_hp.f;
        libxsmm_rne_convert_fp32_bf16( &res, &res_bf16, 1 );

        out[(j*ldo) + i] = ( comp > 0.0 ) ? in[(j*ldi) + i] : res_bf16;
      }
    }
  }
}

void relu_bwd_f32_bf16_gold(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, float *in, libxsmm_bfloat16 *out, float alpha, float *out_fwd, unsigned char *mask, unsigned char type) {
  libxsmm_blasint i, j;
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M; ++i ) {
      union libxsmm_bfloat16_hp bf16_hp;
      bf16_hp.f = in[(j*ldi) + i];
      if ( type == TYPE_RELU ) {
        out[(j*ldo) + i] = ( mask[(j*ldi) + i] == 0 ) ? bf16_hp.i[1] : 0;
      } else if ( type == TYPE_LEAKY_RELU ) {
        union libxsmm_bfloat16_hp bf16_hp_two;
        bf16_hp_two.f = alpha*in[(j*ldi) + i];
        out[(j*ldo) + i] = ( mask[(j*ldi) + i] == 0 ) ? bf16_hp.i[1] : bf16_hp_two.i[1];
      } else if ( type == TYPE_ELU ) {
        libxsmm_bfloat16 res_bf16;
        float res;
        if ( out_fwd[(j*ldi) + i] > 0 ) {
         libxsmm_rne_convert_fp32_bf16( &in[(j*ldi) + i], &res_bf16, 1 );
        } else {
          res = in[(j*ldi) + i] * (out_fwd[(j*ldi) + i] + alpha);
          libxsmm_rne_convert_fp32_bf16( &res, &res_bf16, 1 );
        }
        out[(j*ldo) + i] =  res_bf16;
      }
    }
  }
}

void relu_bwd_bf16_f32_gold(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, libxsmm_bfloat16 *in, float *out, float alpha, libxsmm_bfloat16 *out_fwd, unsigned char *mask, unsigned char type) {
  libxsmm_blasint i, j;
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M; ++i ) {
      if ( type == TYPE_RELU ) {
        union libxsmm_bfloat16_hp bf16_hp;
        bf16_hp.i[1] = ( mask[(j*ldi) + i] == 0 ) ? in[(j*ldi) + i] : 0;
        bf16_hp.i[0] = 0;
        out[(j*ldo) + i] = bf16_hp.f;
      } else if ( type == TYPE_LEAKY_RELU ) {
        union libxsmm_bfloat16_hp bf16_hp;
        bf16_hp.i[1] = in[(j*ldi) + i];
        bf16_hp.i[0] = 0;
        out[(j*ldo) + i] = ( mask[(j*ldi) + i] == 0 ) ? bf16_hp.f : alpha*bf16_hp.f;
      } else if ( type == TYPE_ELU ) {
        union libxsmm_bfloat16_hp bf16_hp;
        float res;
        float comp;
        bf16_hp.i[1] = out_fwd[(j*ldi) + i];
        bf16_hp.i[0] = 0;
        comp = bf16_hp.f;
        res =  bf16_hp.f + alpha;
        bf16_hp.i[1] = in[(j*ldi) + i];
        res = res * bf16_hp.f;

        out[(j*ldo) + i] = ( comp > 0 ) ? bf16_hp.f : res;
      }
    }
  }
}

int test_relu_f32_f32_fwd( libxsmm_blasint bitm, libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, unsigned char type ) {
  float *in;
  float *out, *out_gold;
  unsigned char *mask, *mask_gold;
  unsigned int i, j;
  unsigned int s;
  float alpha = 0.1f;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_matdiff_info norms_out;
  libxsmm_meltwfunction_unary unary_kernel;
  libxsmm_blasint mask_ld = (bitm == 0) ? ldo : LIBXSMM_UPDIV(ldi, 16)*2;/* ldi/8, rounding up to even numbers */
  int bandwidthPerIteration, flopsPerIteration;

  if ( M > ldi ) {
    fprintf( stderr, "test_relu_f32_f32_fwd: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_relu_f32_f32_fwd: ldo needs to be equal to or bigger than N\n");
    exit(-1);
  }

  in        = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi,   64);
  out       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo,   64);
  out_gold  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo,   64);
  mask      = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);
  mask_gold = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);

  BENCHMARK_INIT();

  libxsmm_rng_set_seed(1);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      in[(i*ldi)+j] = (float) (0.05 - libxsmm_rng_f64()/10.0);
    }
  }

  /* init out */
  for ( i = 0; i < N*ldo; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < N*ldo; ++i ) {
    out_gold[i] = 0;
  }
  for ( i = 0; i < N*mask_ld; ++i ) {
    mask[i] = 0;
  }
  for ( i = 0; i < N*mask_ld; ++i ) {
    mask_gold[i] = 0;
  }

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    relu_fwd_f32_f32_gold( M, 1, ldi, ldo, mask_ld, &in[(i*ldi)], &out_gold[(i*ldo)], alpha, &mask_gold[(i*mask_ld)], type );
  }

  unary_param.op.primary = (void*)(&alpha);
  unary_param.in.primary = (void*)in;
  unary_param.out.primary = (void*)out;
  unary_param.out.secondary = (bitm == 0) ? NULL : (void*)mask;
  unary_flags = (bitm == 0) ? LIBXSMM_MELTW_FLAG_UNARY_NONE : LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
  if ( type == TYPE_RELU ) {
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_RELU);
  } else if ( type == TYPE_LEAKY_RELU ) {
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU);
  } else if ( type == TYPE_ELU ) {
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_ELU);
  } else {
    unary_kernel = 0;
  }
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for UNARY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel( &unary_param );

  /* compare result */
  libxsmm_matdiff_clear(&norms_out);
  printf("##########################################\n");
  printf("#   Correctness  - Output                #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, ldo*N, 1, out_gold, out, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n", norms_out.normf_rel);

  flopsPerIteration = 1; /* could be adjusted */
  bandwidthPerIteration = sizeof(float) * 2; /* load + store */
  BENCHMARK_RUN(unary_kernel(&unary_param), bandwidthPerIteration, flopsPerIteration);

  if ( norms_out.normf_rel > 0.00001 ) {
    ret = EXIT_FAILURE;
  }

  if ( type != TYPE_ELU ) {
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

  BENCHMARK_FINALIZE();

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
  libxsmm_free( mask );
  libxsmm_free( mask_gold );

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS unary relu fwd fp32 fp32\n");
  } else {
    printf("FAILURE unary relu fwd fp32 fp32\n");
  }

  return ret;
}

int test_relu_bf16_bf16_fwd( libxsmm_blasint bitm, libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, unsigned char type ) {
  libxsmm_bfloat16 *in;
  libxsmm_bfloat16 *out, *out_gold;
  float *f32out, *f32out_gold;
  unsigned char *mask, *mask_gold;
  unsigned int i, j;
  unsigned int s;
  float alpha = 0.1f;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_matdiff_info norms_out;
  libxsmm_meltwfunction_unary unary_kernel;
  union libxsmm_bfloat16_hp bf16_hp;
  libxsmm_blasint mask_ld = (bitm == 0) ? ldo : ((ldo+15)-((ldo+15)%16))/8;

  if ( M > ldi ) {
    fprintf( stderr, "test_relu_bf16_bf16_fwd: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_relu_bf16_bf16_fwd: ldo needs to be equal to or bigger than N\n");
    exit(-1);
  }

  libxsmm_rng_set_seed(1);

  in        = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldi,   64);
  out       = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo,   64);
  out_gold  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo,   64);
  f32out       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo,   64);
  f32out_gold  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo,   64);
  mask      = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);
  mask_gold = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      bf16_hp.f = (float)(0.05 - libxsmm_rng_f64()/10.0);
      in[(i*ldi)+j] = bf16_hp.i[1];
    }
  }

  /* init out */
  for ( i = 0; i < N*ldo; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < N*ldo; ++i ) {
    out_gold[i] = 0;
  }
  for ( i = 0; i < N*mask_ld; ++i ) {
    mask[i] = 0;
  }
  for ( i = 0; i < N*mask_ld; ++i ) {
    mask_gold[i] = 0;
  }

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    relu_fwd_bf16_bf16_gold( M, 1, ldi, ldo, mask_ld, &in[(i*ldi)], &out_gold[(i*ldo)], alpha, &mask_gold[(i*mask_ld)], type );
  }

  unary_param.op.primary = (void*)(&alpha);
  unary_param.in.primary = (void*)in;
  unary_param.out.primary = (void*)out;
  unary_param.out.secondary = (bitm == 0) ? NULL : (void*)mask;
  unary_flags = (bitm == 0) ? LIBXSMM_MELTW_FLAG_UNARY_NONE : LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
  if ( type == TYPE_RELU ) {
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_RELU);
  } else if ( type == TYPE_LEAKY_RELU ) {
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU);
  } else if ( type == TYPE_ELU ) {
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_ELU);
  } else {
    unary_kernel = 0;
  }
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for UNARY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel( &unary_param );

  /* compare result */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldo; ++j ) {
      f32out_gold[(i*ldo)+j] = upconvert_bf16(out_gold[(i*ldo)+j]);
      f32out[(i*ldo)+j] = upconvert_bf16(out[(i*ldo)+j]);
    }
  }

  libxsmm_matdiff_clear(&norms_out);
  printf("##########################################\n");
  printf("#   Correctness  - Output                #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, ldo*N, 1, f32out_gold, f32out, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n", norms_out.normf_rel);

  if ( norms_out.normf_rel > 0.007 ) {
    ret = EXIT_FAILURE;
  }

  if ( type != 2 ) {
    s = 0;
    if ( s == 0 ) {
      printf("SUCCESS output\n");
    } else {
      printf("FAILURE output\n");
      ret = EXIT_FAILURE;
    }
    if ( bitm != 0 ) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < M/8; ++j ) {
          if ( mask_gold[(i*mask_ld)+j] != mask[(i*mask_ld)+j] ) {
            printf("error at possition i=%i, j=%i, %u, %u\n", i, j, mask[(i*mask_ld)+j], mask_gold[(i*mask_ld)+j]);
            s = 1;
          }
#if 0
          else {
            printf("correct at possition i=%i, j=%i, %u, %u\n", i, j, mask[(i*mask_ld)+j], mask_gold[(i*mask_ld)+j]);
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

  libxsmm_free( f32out_gold );
  libxsmm_free( f32out );
  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
  libxsmm_free( mask );
  libxsmm_free( mask_gold );

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS unary relu fwd bf16 bf16\n");
  } else {
    printf("FAILURE unary relu fwd bf16 bf16\n");
  }

  return ret;
}

int test_relu_f32_bf16_fwd( libxsmm_blasint bitm, libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, unsigned char type ) {
  float *in;
  libxsmm_bfloat16 *out, *out_gold;
  float *f32out, *f32out_gold;
  unsigned char *mask, *mask_gold;
  unsigned int i, j;
  unsigned int s;
  float alpha = 0.1f;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_matdiff_info norms_out;
  libxsmm_meltwfunction_unary unary_kernel;
  union libxsmm_bfloat16_hp bf16_hp;
  libxsmm_blasint mask_ld = (bitm == 0) ? ldo : LIBXSMM_UPDIV(ldi, 16)*2;
  int flopsPerIteration, bandwidthPerIteration;

  if ( M > ldi ) {
    fprintf( stderr, "test_relu_f32_bf16_fwd: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_relu_f32_bf16_fwd: ldo needs to be equal to or bigger than N\n");
    exit(-1);
  }

  in        = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi,   64);
  out       = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo,   64);
  out_gold  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo,   64);
  f32out       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo,   64);
  f32out_gold  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo,   64);
  mask      = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);
  mask_gold = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);

  BENCHMARK_INIT();

  libxsmm_rng_set_seed(1);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      bf16_hp.f = (float)(0.05 - libxsmm_rng_f64()/10.0);
      bf16_hp.i[0] = 0;
      in[(i*ldi)+j] = bf16_hp.f;
    }
  }

  /* init out */
  for ( i = 0; i < N*ldo; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < N*ldo; ++i ) {
    out_gold[i] = 0;
  }
  for ( i = 0; i < N*mask_ld; ++i ) {
    mask[i] = 0;
  }
  for ( i = 0; i < N*mask_ld; ++i ) {
    mask_gold[i] = 0;
  }

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    relu_fwd_f32_bf16_gold( M, 1, ldi, ldo, mask_ld, &in[(i*ldi)], &out_gold[(i*ldo)], alpha, &mask_gold[(i*mask_ld)], type );
  }

  unary_param.op.primary = (void*)(&alpha);
  unary_param.in.primary = (void*)in;
  unary_param.out.primary = (void*)out;
  unary_param.out.secondary = (bitm == 0) ? NULL : (void*)mask;
  unary_flags = (bitm == 0) ? LIBXSMM_MELTW_FLAG_UNARY_NONE : LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
  if ( type == TYPE_RELU ) {
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_RELU);
  } else if ( type == TYPE_LEAKY_RELU ) {
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU);
  } else if ( type == TYPE_ELU ) {
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_ELU);
  } else {
    unary_kernel = 0;
  }
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for UNARY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel( &unary_param );

  /* compare result */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldo; ++j ) {
      f32out_gold[(i*ldo)+j] = upconvert_bf16(out_gold[(i*ldo)+j]);
      f32out[(i*ldo)+j] = upconvert_bf16(out[(i*ldo)+j]);
    }
  }

  libxsmm_matdiff_clear(&norms_out);
  printf("##########################################\n");
  printf("#   Correctness  - Output                #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, ldi*N, 1, f32out_gold, f32out, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n", norms_out.normf_rel);

  flopsPerIteration = 1;/* could be adjusted */
  bandwidthPerIteration = sizeof(float) + sizeof(libxsmm_bfloat16);
  BENCHMARK_RUN(unary_kernel(&unary_param), bandwidthPerIteration, flopsPerIteration);

  if ( norms_out.normf_rel > 0.007 ) {
    ret = EXIT_FAILURE;
  }

  if ( type != 2 ) {
    s = 0;
    if ( bitm != 0 ) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < M/8; ++j ) {
          if ( mask_gold[(i*mask_ld)+j] != mask[(i*mask_ld)+j] ) {
            printf("error at possition i=%i, j=%i, %u, %u\n", i, j, mask[(i*mask_ld)+j], mask_gold[(i*mask_ld)+j]);
            s = 1;
          }
#if 0
          else {
            printf("correct at possition i=%i, j=%i, %u, %u\n", i, j, mask[(i*mask_ld)+j], mask_gold[(i*mask_ld)+j]);
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

  BENCHMARK_FINALIZE();

  libxsmm_free( f32out_gold );
  libxsmm_free( f32out );
  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
  libxsmm_free( mask );
  libxsmm_free( mask_gold );

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS unary relu fwd fp32 bf16\n");
  } else {
    printf("FAILURE unary relu fwd fp32 bf16\n");
  }

  return ret;
}

int test_relu_bf16_f32_fwd( libxsmm_blasint bitm, libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, unsigned char type ) {
  libxsmm_bfloat16 *in;
  float *out, *out_gold;
  unsigned char *mask, *mask_gold;
  unsigned int i, j;
  unsigned int s;
  float alpha = 0.1f;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_matdiff_info norms_out;
  libxsmm_meltwfunction_unary unary_kernel;
  union libxsmm_bfloat16_hp bf16_hp;
  libxsmm_blasint mask_ld = (bitm == 0) ? ldo : LIBXSMM_UPDIV(ldi, 16)*2;
  int flopsPerIteration, bandwidthPerIteration;

  if ( M > ldi ) {
    fprintf( stderr, "test_relu_bf16_f32_fwd: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_relu_bf16_f32_fwd: ldo needs to be equal to or bigger than N\n");
    exit(-1);
  }

  in        = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldi, 64);
  out       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo, 64);
  out_gold  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo, 64);
  mask      = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);
  mask_gold = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);

  BENCHMARK_INIT();

  libxsmm_rng_set_seed(1);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      bf16_hp.f = (float)(0.05 - libxsmm_rng_f64()/10.0);
      in[(i*ldi)+j] = bf16_hp.i[1];
    }
  }

  /* init out */
  for ( i = 0; i < N*ldo; ++i ) {
    out[i] = 0.0f;
  }
  for ( i = 0; i < N*ldo; ++i ) {
    out_gold[i] = 0.0f;
  }
  for ( i = 0; i < N*mask_ld; ++i ) {
    mask[i] = 0;
  }
  for ( i = 0; i < N*mask_ld; ++i ) {
    mask_gold[i] = 0;
  }

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    relu_fwd_bf16_f32_gold( M, 1, ldi, ldo, mask_ld, &in[(i*ldi)], &out_gold[(i*ldo)], alpha, &mask_gold[(i*mask_ld)], type );
  }

  /* use jited relu */
  unary_param.op.primary = (void*)(&alpha);
  unary_param.in.primary = (void*)in;
  unary_param.out.primary = (void*)out;
  unary_param.out.secondary = (bitm == 0) ? NULL : (void*)mask;
  unary_flags = (bitm == 0) ? LIBXSMM_MELTW_FLAG_UNARY_NONE : LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
  if ( type == TYPE_RELU ) {
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_RELU);
  } else if ( type == TYPE_LEAKY_RELU ) {
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU);
  } else if ( type == TYPE_ELU ) {
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_ELU);
  } else {
    unary_kernel = 0;
  }
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for UNARY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel( &unary_param );

  /* compare result */
  libxsmm_matdiff_clear(&norms_out);
  printf("##########################################\n");
  printf("#   Correctness  - Output                #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, ldi*N, 1, out_gold, out, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n", norms_out.normf_rel);

  flopsPerIteration = 1; /* could be adjusted depending on the type */
  bandwidthPerIteration = sizeof(libxsmm_bfloat16) + sizeof(float);/* load + store */
  BENCHMARK_RUN(unary_kernel(&unary_param), bandwidthPerIteration, flopsPerIteration);

  if ( norms_out.normf_rel > 0.007 ) {
    ret = EXIT_FAILURE;
  }

  if ( type != TYPE_ELU ) {
    s = 0;
    if ( bitm != 0 ) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < M/8; ++j ) {
          if ( mask_gold[(i*mask_ld)+j] != mask[(i*mask_ld)+j] ) {
            printf("error at possition i=%i, j=%i, %u, %u\n", i, j, mask[(i*mask_ld)+j], mask_gold[(i*mask_ld)+j]);
            s = 1;
          }
#if 0
          else {
            printf("correct at possition i=%i, j=%i, %u, %u\n", i, j, mask[(i*mask_ld)+j], mask_gold[(i*mask_ld)+j]);
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

  BENCHMARK_FINALIZE();

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
  libxsmm_free( mask );
  libxsmm_free( mask_gold );

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS unary relu fwd bf16 fp32\n");
  } else {
    printf("FAILURE unary relu fwd bf16 fp32\n");
  }

  return ret;
}

int test_relu_f32_f32_bwd( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, unsigned char type ) {
  float *in;
  float *out, *out_gold;
  float *out_fwd;
  unsigned char *mask_bit;
  unsigned char *mask_gold;
  unsigned int i, j;
  float alpha = 0.1f;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_matdiff_info norms_out;
  libxsmm_meltwfunction_unary unary_kernel;
  libxsmm_blasint mask_ld = LIBXSMM_UPDIV(ldi, 16)*2;
  int flopsPerIteration, bandwidthPerIteration;

  if ( M > ldi ) {
    fprintf( stderr, "test_relu_f32_f32_bwd: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_relu_f32_f32_bwd: ldo needs to be equal to or bigger than N\n");
    exit(-1);
  }

  in        = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi, 64);
  out_fwd   = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi, 64);
  out       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo, 64);
  out_gold  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo, 64);
  mask_bit  = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);
  mask_gold = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*ldi, 64);

  BENCHMARK_INIT();

  libxsmm_rng_set_seed(1);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      in[(i*ldi)+j]      = (float)(0.05 - libxsmm_rng_f64()/10.0);
      out_fwd[(i*ldi)+j] = (float)(0.05 - libxsmm_rng_f64()/10.0);
    }
  }

  /* init out */
  for ( i = 0; i < N*ldo; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < N*ldo; ++i ) {
    out_gold[i] = 0;
  }
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
  for ( i = 0; i < N; ++i ) {
    relu_bwd_f32_f32_gold( M, 1, ldi, ldo, &in[(i*ldi)], &out_gold[(i*ldo)], alpha, &out_fwd[(i*ldi)], &mask_gold[(i*ldi)], type );
  }

  /* use jited relu */
  unary_param.op.primary   = (void*)(&alpha);
  unary_param.in.primary   = (void*)in;
  unary_param.in.secondary = ( type == TYPE_ELU ) ? (void*)out_fwd : (void*)mask_bit;
  unary_param.out.primary  = (void*)out;
  if ( type == TYPE_RELU ) {
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_RELU_INV);
  } else if ( type == TYPE_LEAKY_RELU ) {
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV);
  } else if ( type == TYPE_ELU ) {
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_ELU_INV);
  } else {
    unary_kernel = 0;
  }
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for UNARY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel( &unary_param );

  /* compare result */
  libxsmm_matdiff_clear(&norms_out);
  printf("##########################################\n");
  printf("#   Correctness  - Output                #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, ldo*N, 1, out_gold, out, 0, 0);

  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n", norms_out.normf_rel);

  flopsPerIteration = 1;
  bandwidthPerIteration = sizeof(float) * 2;
  BENCHMARK_RUN(unary_kernel(&unary_param), bandwidthPerIteration, flopsPerIteration);

  if ( norms_out.normf_rel > 0.00001 ) {
    ret = EXIT_FAILURE;
  }

  BENCHMARK_FINALIZE();

  libxsmm_free( out_fwd );
  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
  libxsmm_free( mask_bit );
  libxsmm_free( mask_gold );

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS unary relu bwd fp32 fp32\n");
  } else {
    printf("FAILURE unary relu bwd fp32 fp32\n");
  }

  return ret;
}

int test_relu_bf16_bf16_bwd( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, unsigned char type ) {
  libxsmm_bfloat16 *in, *out_fwd;
  libxsmm_bfloat16 *out, *out_gold;
  float *f32out, *f32out_gold;
  unsigned char *mask_bit;
  unsigned char *mask_gold;
  unsigned int i, j;
  float alpha = 0.1f;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_matdiff_info norms_out;
  libxsmm_meltwfunction_unary unary_kernel;
  union libxsmm_bfloat16_hp bf16_hp;
  libxsmm_blasint mask_ld = LIBXSMM_UPDIV(ldi, 16)*2;
  int flopsPerIteration, bandwidthPerIteration;

  if ( M > ldi ) {
    fprintf( stderr, "test_relu_bf16_bf16_bwd: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_relu_bf16_bf16_bwd: ldo needs to be equal to or bigger than N\n");
    exit(-1);
  }

  in        = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldi, 64);
  out_fwd   = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldi, 64);
  out       = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo, 64);
  out_gold  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo, 64);
  f32out       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo, 64);
  f32out_gold  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo, 64);
  mask_bit  = (unsigned char*)  libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);
  mask_gold = (unsigned char*)  libxsmm_aligned_malloc( sizeof(unsigned char)*N*ldi, 64);

  BENCHMARK_INIT();

  libxsmm_rng_set_seed(1);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      bf16_hp.f = (float)(0.05 - libxsmm_rng_f64()/10.0);
      in[(i*ldi)+j] = bf16_hp.i[1];
      bf16_hp.f = (float)(0.05 - libxsmm_rng_f64()/10.0);
      out_fwd[(i*ldi)+j] = bf16_hp.i[1];
    }
  }

  /* init out */
  for ( i = 0; i < N*ldo; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < N*ldo; ++i ) {
    out_gold[i] = 0;
  }
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
  for ( i = 0; i < N; ++i ) {
    relu_bwd_bf16_bf16_gold( M, 1, ldi, ldo, &in[(i*ldi)], &out_gold[(i*ldo)], alpha, &out_fwd[(i*ldi)], &mask_gold[(i*ldi)], type );
  }

  /* use jited relu */
  unary_param.op.primary    = (void*)(&alpha);
  unary_param.in.primary    = (void*)in;
  unary_param.in.secondary  = ( type == TYPE_ELU ) ? (void*)out_fwd : (void*)mask_bit;
  unary_param.out.primary   = (void*)out;
  if ( type == TYPE_RELU ) {
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_RELU_INV);
  } else if ( type == TYPE_LEAKY_RELU ) {
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV);
  } else if ( type == TYPE_ELU ) {
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_ELU_INV);
  } else {
    unary_kernel = 0;
  }
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for UNARY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel( &unary_param );

  /* compare result */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldo; ++j ) {
      f32out_gold[(i*ldo)+j] = upconvert_bf16(out_gold[(i*ldo)+j]);
      f32out[(i*ldo)+j] = upconvert_bf16(out[(i*ldo)+j]);
    }
  }

  libxsmm_matdiff_clear(&norms_out);
  printf("##########################################\n");
  printf("#   Correctness  - Output                #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, ldo*N, 1, f32out_gold, f32out, 0, 0);

  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n", norms_out.normf_rel);

  flopsPerIteration = 1;
  bandwidthPerIteration = sizeof(libxsmm_bfloat16) * 2;
  BENCHMARK_RUN(unary_kernel(&unary_param), bandwidthPerIteration, flopsPerIteration);

  if ( norms_out.normf_rel > 0.007 ) {
    ret = EXIT_FAILURE;
  }

  BENCHMARK_FINALIZE();

  libxsmm_free( f32out_gold );
  libxsmm_free( f32out );
  libxsmm_free( out_fwd );
  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
  libxsmm_free( mask_bit );
  libxsmm_free( mask_gold );

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS unary relu bwd bf16 bf16\n");
  } else {
    printf("FAILURE unary relu bwd bf16 bf16\n");
  }

  return ret;
}

int test_relu_f32_bf16_bwd( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, unsigned char type ) {
  float *in, *out_fwd;
  libxsmm_bfloat16 *out, *out_gold;
  float *f32out, *f32out_gold;
  unsigned char *mask_bit;
  unsigned char *mask_gold;
  unsigned int i, j;
  float alpha = 0.1f;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_matdiff_info norms_out;
  libxsmm_meltwfunction_unary unary_kernel;
  union libxsmm_bfloat16_hp bf16_hp;
  libxsmm_blasint mask_ld = LIBXSMM_UPDIV(ldi, 16)*2;
  int flopsPerIteration, bandwidthPerIteration;

  if ( M > ldi ) {
    fprintf( stderr, "test_relu_f32_bf16_bwd: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_relu_f32_bf16_bwd: ldo needs to be equal to or bigger than N\n");
    exit(-1);
  }

  in        = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi, 64);
  out_fwd   = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi, 64);
  out       = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo, 64);
  out_gold  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo, 64);
  f32out       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo, 64);
  f32out_gold  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo, 64);
  mask_bit  = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);
  mask_gold = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*ldi, 64);

  BENCHMARK_INIT();

  libxsmm_rng_set_seed(1);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      bf16_hp.f = (float)(0.05 - libxsmm_rng_f64()/10.0);
      bf16_hp.i[0] = 0;
      in[(i*ldi)+j] = bf16_hp.f;
      bf16_hp.f = (float)(0.05 - libxsmm_rng_f64()/10.0);
      bf16_hp.i[0] = 0;
      out_fwd[(i*ldi)+j] = bf16_hp.f;
    }
  }

  /* init out */
  for ( i = 0; i < N*ldo; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < N*ldo; ++i ) {
    out_gold[i] = 0;
  }
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
  for ( i = 0; i < N; ++i ) {
    relu_bwd_f32_bf16_gold( M, 1, ldi, ldo, &in[(i*ldi)], &out_gold[(i*ldo)], alpha, &out_fwd[(i*ldi)], &mask_gold[(i*ldi)], type );
  }

  /* use jited relu */
  unary_param.op.primary    = (void*)(&alpha);
  unary_param.in.primary    = (void*)in;
  unary_param.in.secondary  = ( type == TYPE_ELU ) ? (void*)out_fwd : (void*)mask_bit;
  unary_param.out.primary   = (void*)out;
  if ( type == TYPE_RELU ) {
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_RELU_INV);
  } else if ( type == TYPE_LEAKY_RELU ) {
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV);
  } else if ( type == TYPE_ELU ) {
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_ELU_INV);
  } else {
    unary_kernel = 0;
  }
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for UNARY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel( &unary_param );

  /* compare result */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldo; ++j ) {
      f32out_gold[(i*ldo)+j] = upconvert_bf16(out_gold[(i*ldo)+j]);
      f32out[(i*ldo)+j] = upconvert_bf16(out[(i*ldo)+j]);
    }
  }

  libxsmm_matdiff_clear(&norms_out);
  printf("##########################################\n");
  printf("#   Correctness  - Output                #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, ldo*N, 1, f32out_gold, f32out, 0, 0);

  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n", norms_out.normf_rel);

  flopsPerIteration = 1;
  bandwidthPerIteration = sizeof(float) + sizeof(libxsmm_bfloat16);
  BENCHMARK_RUN(unary_kernel(&unary_param), bandwidthPerIteration, flopsPerIteration);

  if ( norms_out.normf_rel > 0.007 ) {
    ret = EXIT_FAILURE;
  }

  BENCHMARK_FINALIZE();

  libxsmm_free( f32out_gold );
  libxsmm_free( f32out );
  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( out_fwd );
  libxsmm_free( in );
  libxsmm_free( mask_bit );
  libxsmm_free( mask_gold );

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS unary relu bwd fp32 bf16\n");
  } else {
    printf("FAILURE unary relu bwd fp32 bf16\n");
  }

  return ret;
}

int test_relu_bf16_f32_bwd( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, unsigned char type ) {
  libxsmm_bfloat16 *in, *out_fwd;
  float *out, *out_gold;
  unsigned char *mask_bit;
  unsigned char *mask_gold;
  unsigned int i, j;
  float alpha = 0.1f;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_matdiff_info norms_out;
  libxsmm_meltwfunction_unary unary_kernel;
  union libxsmm_bfloat16_hp bf16_hp;
  libxsmm_blasint mask_ld = LIBXSMM_UPDIV(ldi, 16)*2;
  int flopsPerIteration, bandwidthPerIteration;

  if ( M > ldi ) {
    fprintf( stderr, "test_relu_bf16_f32_bwd: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_relu_bf16_f32_bwd: ldo needs to be equal to or bigger than N\n");
    exit(-1);
  }

  in        = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldi, 64);
  out_fwd   = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldi, 64);
  out       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo, 64);
  out_gold  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo, 64);
  mask_bit  = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);
  mask_gold = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*ldi, 64);

  BENCHMARK_INIT();

  libxsmm_rng_set_seed(1);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      bf16_hp.f = (float)(0.05 - libxsmm_rng_f64()/10.0);
      in[(i*ldi)+j] = bf16_hp.i[1];
      bf16_hp.f = (float)(0.05 - libxsmm_rng_f64()/10.0);
      out_fwd[(i*ldi)+j] = bf16_hp.i[1];
    }
  }

  /* init out */
  for ( i = 0; i < N*ldo; ++i ) {
    out[i] = 0.0f;
  }
  for ( i = 0; i < N*ldo; ++i ) {
    out_gold[i] = 0.0f;
  }
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
  for ( i = 0; i < N; ++i ) {
    relu_bwd_bf16_f32_gold( M, 1, ldi, ldo, &in[(i*ldi)], &out_gold[(i*ldo)], alpha, &out_fwd[(i*ldi)], &mask_gold[(i*ldi)], type );
  }

  /* use jited relu */
  unary_param.op.primary    = (void*)(&alpha);
  unary_param.in.primary    = (void*)in;
  unary_param.in.secondary  = ( type == TYPE_ELU ) ? (void*)out_fwd : (void*)mask_bit;
  unary_param.out.primary   = (void*)out;
  if ( type == TYPE_RELU ) {
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_RELU_INV);
  } else if ( type == TYPE_LEAKY_RELU ) {
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV);
  } else if ( type == TYPE_ELU ) {
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_ELU_INV);
  } else {
    unary_kernel = 0;
  }
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for UNARY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel( &unary_param );

  /* compare result */
  libxsmm_matdiff_clear(&norms_out);
  printf("##########################################\n");
  printf("#   Correctness  - Output                #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, ldo*N, 1, out_gold, out, 0, 0);

  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n", norms_out.normf_rel);

  flopsPerIteration = 1;
  bandwidthPerIteration = sizeof(libxsmm_bfloat16) + sizeof(float);
  BENCHMARK_RUN(unary_kernel(&unary_param), bandwidthPerIteration, flopsPerIteration);

  if ( norms_out.normf_rel > 0.007 ) {
    ret = EXIT_FAILURE;
  }

  BENCHMARK_FINALIZE();

  libxsmm_free( out_fwd );
  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
  libxsmm_free( mask_bit );
  libxsmm_free( mask_gold );

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS unary relu bwd bf16 fp32\n");
  } else {
    printf("FAILURE unary relu bwd bf16 fp32\n");
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
    itype = TYPE_RELU;
    printf("Testing ReLU ");
  } else if ( type == 'L' ) {
    itype = TYPE_LEAKY_RELU;
    printf("Testing Leaky ReLU ");
  } else if ( type == 'E' ) {
    itype = TYPE_ELU;
#if 0
    bitm = 0;
    printf("Testing ELU (disabling bitmask support) ");
#else
    printf("Testing ELU ");
#endif
  } else {
    itype = TYPE_RELU;
    printf("Testing ReLU ");
  }

  if ( op == 'F' && dtype_in == 4 && dtype_out == 4  ) {
    printf("F32 F32 forward - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_relu_f32_f32_fwd( bitm, M, N, ldi, ldo, itype );
  } else if ( op == 'F' && dtype_in == 2  && dtype_out == 2 ) {
    printf("BF16 BF16 forward - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_relu_bf16_bf16_fwd( bitm, M, N, ldi, ldo, itype );
  } else if ( op == 'F' && dtype_in == 4  && dtype_out == 2 ) {
    printf("F32 BF16 forward - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_relu_f32_bf16_fwd( bitm, M, N, ldi, ldo, itype );
  } else if ( op == 'F' && dtype_in == 2  && dtype_out == 4 ) {
    printf("BF16 F32 forward - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_relu_bf16_f32_fwd( bitm, M, N, ldi, ldo, itype );
  } else if ( op == 'B' && dtype_in == 4 && dtype_out == 4 ) {
    printf("F32 F32 backward - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_relu_f32_f32_bwd( M, N, ldi, ldo, itype );
  } else if ( op == 'B' && dtype_in == 2 && dtype_out == 2 ) {
    printf("BF16 BF16 backward - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_relu_bf16_bf16_bwd( M, N, ldi, ldo, itype );
  } else if ( op == 'B' && dtype_in == 4 && dtype_out == 2 ) {
    printf("F32 BF16 backward - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_relu_f32_bf16_bwd( M, N, ldi, ldo, itype );
  } else if ( op == 'B' && dtype_in == 2 && dtype_out == 4 ) {
    printf("BF16 F32 backward - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_relu_bf16_f32_bwd( M, N, ldi, ldo, itype );
  } else {
    printf(" Not implemented case! Usage: %s [D/L/E] [F/B] [bitmask: 0/1] [prec_in: 4/2] [prec_out: 4/2] [M] [N] [ldi] [ldo]\n", argv[0] );
    exit(-1);
  }

  return ret;
}
