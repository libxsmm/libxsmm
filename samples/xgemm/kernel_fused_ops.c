/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
# if defined(__APPLE__) && defined(__arm64__)
#include <pthread.h>
# endif

#define USE_GEMM_EXT_FRONTEND

#define COLBIAS_ADD     1
#define RELU_NOBITMASK  1
#define RELU_BITMASK    2
#define SIGMOID         3

typedef struct fusion_args {
  char *colbias;
  char *relu_bitmask;
} fusion_args;

typedef struct gemm_def {
  libxsmm_datatype in_type;
  libxsmm_datatype out_type;
  libxsmm_datatype comp_type;
  libxsmm_blasint m;
  libxsmm_blasint n;
  libxsmm_blasint k;
  libxsmm_blasint lda;
  libxsmm_blasint ldb;
  libxsmm_blasint ldc;
  double alpha;
  double beta;
  int trans_a;
  int trans_b;
  int vnni_a;
  int vnni_b;
  int vnni_c;
  int unsigned_a;
  int unsigned_b;
  int unsigned_c;
  int aligned_a;
  int aligned_c;
  int prefetch;
  int br_type;
  libxsmm_blasint br_count;
  int br_unroll;
  int tc_config;
  float scf;
  int binary_postop;
  int unary_postop;
} gemm_def;

float fsigmoid(float x) {
  return (LIBXSMM_TANHF(x/2.0f) + 1.0f)/2.0f;
}

void relu_fwd_f32_f32_gold(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, libxsmm_blasint ldo_mask, float *in, float *out, float alpha, unsigned char *out_mask, unsigned char type, libxsmm_blasint use_bitmask) {
  libxsmm_blasint i, j;
  if ( (type != 2) && (use_bitmask > 0)) {
    memset(out_mask, 0, ldo_mask*N);
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        out_mask[(j*ldo_mask) + i/8] |= (unsigned char)(( in[(j*ldi) + i] < 0.0f ) ? 0x0 : (1 << (i%8)) );
      }
    }
  }
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M; ++i ) {
      if ( type == 0 ) {
        out[(j*ldo) + i] = ( in[(j*ldi) + i] < 0.0f ) ? 0.0f : in[(j*ldi) + i];
      } else if ( type == 1 ) {
        out[(j*ldo) + i] = ( in[(j*ldi) + i] < 0.0f ) ? alpha*in[(j*ldi) + i] : in[(j*ldi) + i];
      } else if ( type == 2 ) {
        out[(j*ldo) + i] = ( in[(j*ldi) + i] < 0.0f ) ? alpha * (expf(in[(j*ldi) + i])-1.0) : in[(j*ldi) + i];
      }
    }
  }
}

void relu_fwd_bf16_bf16_gold(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, libxsmm_blasint ldo_mask, libxsmm_bfloat16 *in, libxsmm_bfloat16 *out, float alpha, unsigned char *out_mask, unsigned char type, libxsmm_blasint use_bitmask) {
  libxsmm_blasint i, j;
  if ( (type != 2) && (use_bitmask > 0)) {
    memset(out_mask, 0, ldo_mask*N);
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        out_mask[(j*ldo_mask) + i/8] |= (unsigned char)(( (in[(j*ldi) + i] & 0x8000) == 0x8000 ) ? 0x0 : (1 << (i%8)) );
      }
    }
  }
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M; ++i ) {
      if ( type == 0 ) {
        out[(j*ldo) + i] = ( (in[(j*ldi) + i] & 0x8000) == 0x8000 ) ? 0 : in[(j*ldi) + i];
      } else if ( type == 1 ) {
        union libxsmm_bfloat16_hp bf16_hp;
        union libxsmm_bfloat16_hp bf16_hp_out;
        bf16_hp.i[0] = 0;
        bf16_hp.i[1] = in[(j*ldi) + i];
        bf16_hp_out.f = ( (in[(j*ldi) + i] & 0x8000) == 0x8000 ) ? alpha*bf16_hp.f : bf16_hp.f;
        out[(j*ldo) + i] = bf16_hp_out.i[1];
      } else if ( type == 2 ) {
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
    }
  }
}

void apply_colbias_add(const gemm_def *i_gemm_def, void *l_c_gold, void *l_colbias) {
  unsigned int ldc = i_gemm_def->ldc;
  unsigned int m = i_gemm_def->m;
  unsigned int n = i_gemm_def->n;
  libxsmm_blasint i, j;
  if (i_gemm_def->out_type == LIBXSMM_DATATYPE_F32) {
    float* f_c_gold  = (float*)l_c_gold;
    float* f_colbias = (float*)l_colbias;

    for (j = 0; j < n; j++) {
      for (i = 0; i < m; i++) {
        f_c_gold[i + j * ldc] = f_c_gold[i + j * ldc] + f_colbias[i];
      }
    }
  } else if (i_gemm_def->out_type  == LIBXSMM_DATATYPE_BF16 ) {
    libxsmm_bfloat16* h_c_gold  = (libxsmm_bfloat16*)l_c_gold;
    libxsmm_bfloat16* h_colbias = (libxsmm_bfloat16*)l_colbias;
    for (j = 0; j < n; j++) {
      for (i = 0; i < m; i++) {
        union libxsmm_bfloat16_hp tmp_c;
        union libxsmm_bfloat16_hp tmp_colb;
        float res = 0.0;
        tmp_c.i[0] = 0;
        tmp_c.i[1] = h_c_gold[i + j * ldc];
        tmp_colb.i[0] = 0;
        tmp_colb.i[1] = h_colbias[i];
        res = tmp_c.f + tmp_colb.f;
        libxsmm_rne_convert_fp32_bf16( &res, &h_c_gold[i + j * ldc], 1 );
      }
    }
  }
}

void apply_relu(const gemm_def *i_gemm_def, void *l_c_gold, void *l_relu_bitmask_gold, libxsmm_blasint use_bitmask) {
  unsigned int ldc = i_gemm_def->ldc;
  unsigned int m = i_gemm_def->m;
  unsigned int n = i_gemm_def->n;
  if (i_gemm_def->out_type == LIBXSMM_DATATYPE_F32) {
    float* f_c_gold  = (float*)l_c_gold;
    relu_fwd_f32_f32_gold(m, n, ldc, ldc, ldc/8, f_c_gold, f_c_gold, 0, (unsigned char *)l_relu_bitmask_gold, 0, use_bitmask);
  } else if (i_gemm_def->out_type == LIBXSMM_DATATYPE_BF16) {
    libxsmm_bfloat16* h_c_gold  = (libxsmm_bfloat16*)l_c_gold;
    relu_fwd_bf16_bf16_gold(m, n, ldc, ldc, ldc/8, h_c_gold, h_c_gold, 0, (unsigned char *)l_relu_bitmask_gold, 0, use_bitmask);
  }
}

void apply_sigmoid(const gemm_def *i_gemm_def, void *l_c_gold) {
  unsigned int ldc = i_gemm_def->ldc;
  unsigned int m = i_gemm_def->m;
  unsigned int n = i_gemm_def->n;
  libxsmm_blasint i, j;
  if (i_gemm_def->out_type == LIBXSMM_DATATYPE_F32) {
    float* f_c_gold  = (float*)l_c_gold;
    for (j = 0; j < n; j++) {
      for (i = 0; i < m; i++) {
        f_c_gold[i + j * ldc] = fsigmoid(f_c_gold[i + j * ldc]);
      }
    }
  } else if (i_gemm_def->out_type == LIBXSMM_DATATYPE_BF16) {
    libxsmm_bfloat16* h_c_gold  = (libxsmm_bfloat16*)l_c_gold;
    for (j = 0; j < n; j++) {
      for (i = 0; i < m; i++) {
        union libxsmm_bfloat16_hp tmp_c;
        float res = 0.0;
        tmp_c.i[0] = 0;
        tmp_c.i[1] = h_c_gold[i + j * ldc];
        res = fsigmoid(tmp_c.f);
        libxsmm_rne_convert_fp32_bf16( &res, &h_c_gold[i + j * ldc], 1 );
      }
    }
  }
}

void init_random_matrix( libxsmm_datatype dtype, void* data, libxsmm_blasint br, libxsmm_blasint ld, libxsmm_blasint n ) {
  double* d_data = (double*) data;
  float* f_data = (float*) data;
  libxsmm_bfloat16* bf_data = (libxsmm_bfloat16*) data;
  int* i_data = (int*) data;
  short* s_data = (short*) data;
  char* c_data = (char*) data;
  unsigned int l_r, l_i, l_j;

  for (l_r = 0; l_r < br; l_r++) {
    for (l_i = 0; l_i < ld; l_i++) {
      for (l_j = 0; l_j < n; l_j++) {
        if ( dtype == LIBXSMM_DATATYPE_F64 ) {
          d_data[(l_r * ld * n) + (l_j * ld) + l_i] = libxsmm_rng_f64();
        } else if ( dtype == LIBXSMM_DATATYPE_F32 ) {
          f_data[(l_r * ld * n) + (l_j * ld) + l_i] = (float)(0.05 - libxsmm_rng_f64()/10.0);
        } else if ( dtype == LIBXSMM_DATATYPE_BF16 ) {
          union libxsmm_bfloat16_hp tmp;
          tmp.f = (float)(0.05 - libxsmm_rng_f64()/10.0);
          bf_data[(l_r * ld * n) + (l_j * ld) + l_i] = tmp.i[1];
        } else if ( dtype == LIBXSMM_DATATYPE_I32 ) {
          i_data[(l_r * ld * n) + (l_j * ld) + l_i] = (int)  (libxsmm_rng_f64() * 20.0);
        } else if ( dtype == LIBXSMM_DATATYPE_I16 ) {
          s_data[(l_r * ld * n) + (l_j * ld) + l_i] = (short)(libxsmm_rng_f64() * 20.0);
        } else if ( dtype == LIBXSMM_DATATYPE_I8 ) {
          c_data[(l_r * ld * n) + (l_j * ld) + l_i] = (char) (libxsmm_rng_f64() * 20.0);
        } else {
        }
      }
    }
  }
}

void init_zero_matrix( libxsmm_datatype dtype, void* data, libxsmm_blasint br, libxsmm_blasint ld, libxsmm_blasint n ) {
  char* l_data = (char*) data;
  memset( l_data, 0x0, br*ld*n*LIBXSMM_TYPESIZE(dtype) );
}

void init_garbage_matrix( libxsmm_datatype dtype, void* data, libxsmm_blasint br, libxsmm_blasint ld, libxsmm_blasint n ) {
  char* l_data = (char*) data;
  memset( l_data, 0xdeadbeef, br*ld*n*LIBXSMM_TYPESIZE(dtype) );
}

void ref_matmul( gemm_def* i_gemm_def, void* a, void* b, void* c ) {
  unsigned int l_r, l_j, l_i, l_s, l_k2;
  unsigned int lda = i_gemm_def->lda;
  unsigned int ldb = i_gemm_def->ldb;
  unsigned int ldc = i_gemm_def->ldc;
  unsigned int m = i_gemm_def->m;
  unsigned int n = i_gemm_def->n;
  unsigned int k = i_gemm_def->k;

  if ( (i_gemm_def->in_type   == LIBXSMM_DATATYPE_F64) &&
       (i_gemm_def->out_type  == LIBXSMM_DATATYPE_F64) &&
       (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F64)    ) {
    double* d_a = (double*)a;
    double* d_b = (double*)b;
    double* d_c = (double*)c;

    for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
      for (l_j = 0; l_j < n; l_j++) {
        for (l_i = 0; l_i < m; l_i++) {
          if ( (i_gemm_def->beta == 0) && (l_r == 0) ) {
            d_c[(l_j * ldc) + l_i] = 0.0;
          }
          for (l_s = 0; l_s < k; l_s++) {
            if ( i_gemm_def->trans_b == 0 ) {
              d_c[(l_j * ldc) + l_i] += d_a[(l_r * lda * k) + ((l_s * lda) + l_i)] * d_b[(l_r * ldb * n) + ((l_j * ldb) + l_s)];
            } else {
              d_c[(l_j * ldc) + l_i] += d_a[(l_r * lda * k) + ((l_s * lda) + l_i)] * d_b[(l_r * ldb * k) + ((l_s * ldb) + l_j)];
            }
          }
        }
      }
    }
  } else if ( (i_gemm_def->in_type   == LIBXSMM_DATATYPE_F32) &&
              (i_gemm_def->out_type  == LIBXSMM_DATATYPE_F32) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F32)    ) {
    float* f_a = (float*)a;
    float* f_b = (float*)b;
    float* f_c = (float*)c;

    for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
      for (l_j = 0; l_j < n; l_j++) {
        for (l_i = 0; l_i < m; l_i++) {
          if ( (i_gemm_def->beta == 0) && (l_r == 0) ) {
            f_c[(l_j * ldc) + l_i] = 0.0;
          }
          for (l_s = 0; l_s < k; l_s++) {
            if ( i_gemm_def->trans_b == 0 ) {
              f_c[(l_j * ldc) + l_i] += f_a[(l_r * lda * k) + ((l_s * lda) + l_i)] * f_b[(l_r * ldb * n) + ((l_j * ldb) + l_s)];
            } else {
              f_c[(l_j * ldc) + l_i] += f_a[(l_r * lda * k) + ((l_s * lda) + l_i)] * f_b[(l_r * ldb * k) + ((l_s * ldb) + l_j)];
            }
          }
        }
      }
    }
  } else if ( (i_gemm_def->in_type   == LIBXSMM_DATATYPE_I16) &&
              (i_gemm_def->out_type  == LIBXSMM_DATATYPE_I32) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_I32)    ) {
    short* s_a = (short*)a;
    short* s_b = (short*)b;
    int*   i_c = (int*)c;
    int l_k_block = 2;

    for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
      for (l_j = 0; l_j < n; l_j++) {
        for (l_i = 0; l_i < m; l_i++) {
          if ( (i_gemm_def->beta == 0) && (l_r == 0) ) {
            i_c[(l_j * ldc) + l_i] = 0;
          }
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              i_c[(l_j * ldc) + l_i] += s_a[(l_r * lda * k) + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2] *
                                        s_b[(l_r * ldb * n) + (l_j * ldb) + (l_s*l_k_block) + l_k2];
            }
          }
        }
      }
    }
  } else if ( (i_gemm_def->in_type   == LIBXSMM_DATATYPE_I8)  &&
              (i_gemm_def->out_type  == LIBXSMM_DATATYPE_I32)  &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_I32) &&
              (i_gemm_def->unsigned_a == 1) && (i_gemm_def->unsigned_b == 0) ) {
    unsigned char* c_a = (unsigned char*)a;
    char*          c_b = (char*)b;
    int*           i_c = (int*)c;
    int l_k_block = 4;

    for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
      for (l_j = 0; l_j < n; l_j++) {
        for (l_i = 0; l_i < m; l_i++) {
          if ( (i_gemm_def->beta == 0) && (l_r == 0) ) {
            i_c[(l_j * ldc) + l_i] = 0;
          }
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              i_c[(l_j * ldc) + l_i] += c_a[(l_r * lda * k) + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2] *
                                        c_b[(l_r * ldb * n) + (l_j * ldb) + (l_s*l_k_block) + l_k2];
            }
          }
        }
      }
    }
  } else if ( (i_gemm_def->in_type   == LIBXSMM_DATATYPE_I8)  &&
              (i_gemm_def->out_type  == LIBXSMM_DATATYPE_I32)  &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_I32) &&
              (i_gemm_def->unsigned_a == 0) && (i_gemm_def->unsigned_b == 1) ) {
    char*          c_a = (char*)a;
    unsigned char* c_b = (unsigned char*)b;
    int*           i_c = (int*)c;
    int l_k_block = 4;

    for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
      for (l_j = 0; l_j < n; l_j++) {
        for (l_i = 0; l_i < m; l_i++) {
          if ( (i_gemm_def->beta == 0) && (l_r == 0) ) {
            i_c[(l_j * ldc) + l_i] = 0;
          }
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              i_c[(l_j * ldc) + l_i] += c_a[(l_r * lda * k) + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2] *
                                        c_b[(l_r * ldb * n) + (l_j * ldb) + (l_s*l_k_block) + l_k2];
            }
          }
        }
      }
    }
  } else if ( (i_gemm_def->in_type   == LIBXSMM_DATATYPE_I8)  &&
              (i_gemm_def->out_type  == LIBXSMM_DATATYPE_I8)  &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_I32) &&
              (i_gemm_def->unsigned_a == 0) && (i_gemm_def->unsigned_b == 1) && (i_gemm_def->unsigned_c == 1) ) {
    char*          c_a = (char*)a;
    unsigned char* c_b = (unsigned char*)b;
    unsigned char* c_c = (unsigned char*)c;
    int l_k_block = 4;

    for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
      for (l_j = 0; l_j < n; l_j++) {
        for (l_i = 0; l_i < m; l_i++) {
          int tmp;
          float ftmp;
          if ( (i_gemm_def->beta == 0) && (l_r == 0) ) {
            tmp = 0;
          } else {
            tmp = (int)c_c[(l_j * ldc) + l_i];
          }
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              tmp += c_a[(l_r * lda * k) + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2] *
                     c_b[(l_r * ldb * n) + (l_j * ldb) + (l_s*l_k_block) + l_k2];
            }
          }
          ftmp = (float)tmp;
          ftmp *= i_gemm_def->scf;
          c_c[(l_j * ldc) + l_i] = (unsigned char)ftmp;
        }
      }
    }
  } else if ( (i_gemm_def->in_type   == LIBXSMM_DATATYPE_BF16) &&
              (i_gemm_def->out_type  == LIBXSMM_DATATYPE_F32)  &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F32)     ) {
    libxsmm_bfloat16* h_a = (libxsmm_bfloat16*)a;
    libxsmm_bfloat16* h_b = (libxsmm_bfloat16*)b;
    float*            f_c = (float*)c;
    int l_k_block = ( i_gemm_def->vnni_a != 0) ?  2 : 1;

    for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
      for (l_j = 0; l_j < n; l_j++) {
        for (l_i = 0; l_i < m; l_i++) {
          if ( (i_gemm_def->beta == 0) && (l_r == 0) ) {
            f_c[(l_j * ldc) + l_i] = 0.0f;
          }
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              union libxsmm_bfloat16_hp tmp_a_f;
              union libxsmm_bfloat16_hp tmp_b_f;
              tmp_a_f.i[0] = 0;
              tmp_a_f.i[1] = h_a[(l_r * lda * k) + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              tmp_b_f.i[0] = 0;
              tmp_b_f.i[1] = h_b[(l_r * ldb * n) + (l_j * ldb) + (l_s*l_k_block) + l_k2];

              f_c[(l_j * ldc) + l_i] += tmp_a_f.f * tmp_b_f.f;
            }
          }
        }
      }
    }
  } else if ( (i_gemm_def->in_type   == LIBXSMM_DATATYPE_BF16) &&
              (i_gemm_def->out_type  == LIBXSMM_DATATYPE_BF16) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F32)     ) {
    libxsmm_bfloat16* h_a = (libxsmm_bfloat16*)a;
    libxsmm_bfloat16* h_b = (libxsmm_bfloat16*)b;
    libxsmm_bfloat16* h_c = (libxsmm_bfloat16*)c;
    int l_k_block = ( i_gemm_def->vnni_a != 0) ?  2 : 1;
    float acc = 0.0f;
    libxsmm_bfloat16 h_acc;

    for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
      for (l_j = 0; l_j < n; l_j++) {
        for (l_i = 0; l_i < m; l_i++) {
          if ( (i_gemm_def->beta == 0) && (l_r == 0) ) {
            acc = 0.0f;
          } else {
            union libxsmm_bfloat16_hp tmp;
            tmp.i[0] = 0;
            tmp.i[1] = h_c[(l_j * ldc) + l_i];
            acc = tmp.f;
          }
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              union libxsmm_bfloat16_hp tmp_a_f;
              union libxsmm_bfloat16_hp tmp_b_f;
              tmp_a_f.i[0] = 0;
              tmp_a_f.i[1] = h_a[(l_r * lda * k) + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              tmp_b_f.i[0] = 0;
              tmp_b_f.i[1] = h_b[(l_r * ldb * n) + (l_j * ldb) + (l_s*l_k_block) + l_k2];

              acc += tmp_a_f.f * tmp_b_f.f;
            }
          }
          libxsmm_rne_convert_fp32_bf16( &acc, &h_acc, 1 );
          h_c[(l_j * ldc) + l_i] = h_acc;
        }
      }
    }
  }
}

void fused_matmul( gemm_def* i_gemm_def, void* l_a, void* l_b, void* l_c_gold, fusion_args *ref_fusion_arguments ) {

  /* Run matmul */
  ref_matmul( i_gemm_def, l_a, l_b, l_c_gold );

  /* Perform binary postop if requested */
  if (i_gemm_def->binary_postop == COLBIAS_ADD) {
    apply_colbias_add(i_gemm_def, l_c_gold, ref_fusion_arguments->colbias);
  }
  /* Perform unary postop if requested */
  if (i_gemm_def->unary_postop == RELU_NOBITMASK) {
    apply_relu(i_gemm_def, l_c_gold, ref_fusion_arguments->relu_bitmask, 0);
  } else if (i_gemm_def->unary_postop == RELU_BITMASK) {
    apply_relu(i_gemm_def, l_c_gold, ref_fusion_arguments->relu_bitmask, 1);
  } else if (i_gemm_def->unary_postop == SIGMOID) {
    apply_sigmoid(i_gemm_def, l_c_gold);
  }
}

double check_matrix( libxsmm_datatype dtype, void* data_gold, void* data, libxsmm_blasint ld, libxsmm_blasint m, libxsmm_blasint n ) {
  libxsmm_matdiff_info l_diff;
  double max_error = 0.0;

  libxsmm_matdiff_clear(&l_diff);

  if ( dtype == LIBXSMM_DATATYPE_F64 ) {
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F64, m, n, data_gold, data, &ld, &ld);
    max_error = l_diff.linf_abs;
  } else if ( dtype == LIBXSMM_DATATYPE_F32 ) {
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, m, n, data_gold, data, &ld, &ld);
    max_error = l_diff.linf_abs;
  } else if ( dtype == LIBXSMM_DATATYPE_BF16 ) {
    unsigned int l_i, l_j;
    libxsmm_bfloat16* h_data =      (libxsmm_bfloat16*)data;
    libxsmm_bfloat16* h_data_gold = (libxsmm_bfloat16*)data_gold;
    for (l_i = 0; l_i < m; l_i++) {
      for (l_j = 0; l_j < n; l_j++) {
        union libxsmm_bfloat16_hp tmp_c;
        union libxsmm_bfloat16_hp tmp_gold;
        double l_fabs;

        tmp_c.i[1] = h_data[(l_j * ld) + l_i];
        tmp_c.i[0] = 0;
        tmp_gold.i[1] = h_data_gold[(l_j * ld) + l_i];
        tmp_gold.i[0] = 0;
        l_fabs = fabs((double)tmp_gold.f - (double)tmp_c.f);
#if 1
        if (max_error < l_fabs) max_error = l_fabs;
#else
        if (max_error < l_fabs) {
          max_error = l_fabs;
          printf("vals are %.5g and %.5g\n",(double)tmp_gold.f,(double)tmp_c.f);
        }
#endif
      }
    }
  } else if ( dtype == LIBXSMM_DATATYPE_I32 ) {
    unsigned int l_i, l_j;
    int* l_data = (int*)data;
    int* l_data_gold = (int*)data_gold;
    for (l_i = 0; l_i < m; l_i++) {
      for (l_j = 0; l_j < n; l_j++) {
        const double l_fabs = fabs((double)l_data_gold[(l_j * ld) + l_i] - (double)l_data[(l_j * ld) + l_i]);
        if (max_error < l_fabs) max_error = l_fabs;
      }
    }
  } else if ( dtype == LIBXSMM_DATATYPE_I8 ) {
    unsigned int l_i, l_j;
    unsigned char* l_data      = (unsigned char*)data;
    unsigned char* l_data_gold = (unsigned char*)data_gold;
    for (l_i = 0; l_i < m; l_i++) {
      for (l_j = 0; l_j < n; l_j++) {
        const double l_fabs = fabs((double)l_data_gold[(l_j * ld) + l_i] - (double)l_data[(l_j * ld) + l_i]);
        if (max_error < l_fabs) max_error = l_fabs;
      }
    }
  } else {
    max_error = 100.0;
  }

  return max_error;
}

void check_matrix_norms( libxsmm_datatype dtype, void* data_gold, void* data, libxsmm_blasint ld, libxsmm_blasint m, libxsmm_blasint n  ) {
  libxsmm_matdiff_info norms, diff;
  libxsmm_matdiff_clear(&norms);
  libxsmm_matdiff_clear(&diff);
  float *gold_c = (float*) libxsmm_aligned_malloc( ld * n * sizeof(float), 64);
  float *comp_c = (float*) libxsmm_aligned_malloc( ld * n * sizeof(float), 64);

  if (dtype == LIBXSMM_DATATYPE_F32) {
    memcpy(gold_c, data_gold, ld * n * sizeof(float));
    memcpy(comp_c, data     , ld * n * sizeof(float));
  }
  if (dtype == LIBXSMM_DATATYPE_BF16) {
    libxsmm_convert_bf16_f32( data_gold, gold_c, ld*n );
    libxsmm_convert_bf16_f32( data     , comp_c, ld*n );
  }

  /* compare */
  libxsmm_matdiff(&norms, LIBXSMM_DATATYPE_F32, ld*n, 1, gold_c, comp_c, 0, 0);
  printf("\n##########################################\n");
  printf("#       Correctness norm-checking        #\n");
  printf("##########################################\n");
  printf("L1 reference  : %.25g\n", norms.l1_ref);
  printf("L1 test       : %.25g\n", norms.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms.l2_rel);
  printf("Linf abs.error: %.24f\n", norms.linf_abs);
  printf("Linf rel.error: %.24f\n", norms.linf_rel);
  printf("Check-norm    : %.24f\n", norms.normf_rel);
  libxsmm_matdiff_reduce(&diff, &norms);

  libxsmm_free(gold_c);
  libxsmm_free(comp_c);
}

double jit_matmul( const gemm_def*    i_gemm_def,
                   const void*        i_a,
                   const void*        i_b,
                   void*              o_c,
                   void*              o_c_perf,
                   const int          i_reps,
                   const unsigned int i_print_jit_info,
                   fusion_args        *i_fusion_arguments ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
  libxsmm_xmmfunction cfg_tr = { NULL };
  libxsmm_xmmfunction rls_tr = { NULL };
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  libxsmm_gemm_shape l_shape;
  libxsmm_gemm_batch_reduce_config l_brconfig;
  libxsmm_gemm_ext_unary_argops l_argops;
  libxsmm_gemm_ext_binary_postops l_postops;
  libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  libxsmm_bitfield l_prefetch_flags = 0;
#if defined(USE_GEMM_EXT_FRONTEND)
  libxsmm_gemm_ext_param gemm_param;
#else
  libxsmm_gemm_param gemm_param;
#endif
  double l_jittime, l_runtime;
  size_t l_t, l_r;
  char** l_a_addr = (char**)malloc(i_gemm_def->br_count*sizeof(char*));
  char** l_b_addr = (char**)malloc(i_gemm_def->br_count*sizeof(char*));
  unsigned long long* l_a_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  unsigned long long* l_b_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  double l_beta = i_gemm_def->beta;
  unsigned long long l_br = (unsigned long long)i_gemm_def->br_count;
  int l_cfg_flags = 0;
  int l_rls_flags = 0;

  if (0 == i_gemm_def) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  /* setup brgemm offsets */
  if ( i_gemm_def->br_type == 2 ) {
    for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
      l_a_offs[l_r] = l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k * LIBXSMM_TYPESIZE(i_gemm_def->in_type);
      if (i_gemm_def->trans_b == 0) {
        l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * LIBXSMM_TYPESIZE(i_gemm_def->in_type);
      } else {
        l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->k * LIBXSMM_TYPESIZE(i_gemm_def->in_type);
      }
    }
  }

  /* set up the flags */
  if ( i_gemm_def->trans_b != 0 ) {
    l_flags |= LIBXSMM_GEMM_FLAG_TRANS_B;
  }
  if ( i_gemm_def->trans_a != 0 ) {
    fprintf(stderr, "trans_a needs to be 0\n");
    return EXIT_FAILURE;
  }
  if ( i_gemm_def->vnni_a != 0 ) {
    l_flags |= LIBXSMM_GEMM_FLAG_VNNI_A;
  }
  if ( i_gemm_def->unsigned_a != 0 ) {
    l_flags |= LIBXSMM_GEMM_FLAG_A_UNSIGNED;
  }
  if ( i_gemm_def->unsigned_b != 0 ) {
    l_flags |= LIBXSMM_GEMM_FLAG_B_UNSIGNED;
  }

  l_flags |= (0 != i_gemm_def->aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0);
  l_flags |= (0 != i_gemm_def->aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0);
  l_flags |= ( l_beta == 0 ) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0;

  /* setting update GEMM struct */
  l_shape = libxsmm_create_gemm_shape( i_gemm_def->m,  i_gemm_def->n, i_gemm_def->k,
      i_gemm_def->lda, i_gemm_def->ldb, i_gemm_def->ldc,
      i_gemm_def->in_type, i_gemm_def->in_type, i_gemm_def->out_type, i_gemm_def->comp_type );

  /* setting BRGEMM config strucut */
  if (i_gemm_def->br_type == 1) {
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_ADDRESS;
    l_brconfig.br_stride_a_hint = 0;
    l_brconfig.br_stride_b_hint = 0;
    l_brconfig.br_unroll_hint = ( i_gemm_def->br_unroll == 0 ) ? 0 : i_gemm_def->br_count;
  } else if (i_gemm_def->br_type == 2) {
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_OFFSET;
    l_brconfig.br_stride_a_hint = 0;
    l_brconfig.br_stride_b_hint = 0;
    l_brconfig.br_unroll_hint = ( i_gemm_def->br_unroll == 0 ) ? 0 : i_gemm_def->br_count;
  } else if (i_gemm_def->br_type == 3) {
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
    l_brconfig.br_stride_a_hint = i_gemm_def->lda*i_gemm_def->k*LIBXSMM_TYPESIZE(i_gemm_def->in_type);
    l_brconfig.br_stride_b_hint = (i_gemm_def->trans_b == 0) ? i_gemm_def->ldb*i_gemm_def->n*LIBXSMM_TYPESIZE(i_gemm_def->in_type) : i_gemm_def->ldb*i_gemm_def->k*LIBXSMM_TYPESIZE(i_gemm_def->in_type);
    l_brconfig.br_unroll_hint = ( i_gemm_def->br_unroll == 0 ) ? 0 : i_gemm_def->br_count;
  } else {
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_NONE;
    l_brconfig.br_stride_a_hint = 0;
    l_brconfig.br_stride_b_hint = 0;
    l_brconfig.br_unroll_hint = 0;
  }

  /* setting prefetch flags */
  l_prefetch_flags = i_gemm_def->prefetch;

  /* setting ext strcuts to 0 */
  memset( &l_argops, 0, sizeof(libxsmm_gemm_ext_unary_argops) );
  memset( &l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops) );

  if (i_gemm_def->binary_postop == COLBIAS_ADD) {
    l_postops.d_in_type      = i_gemm_def->out_type;
    l_postops.d_binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
    l_postops.d_binary_type  = LIBXSMM_MELTW_TYPE_BINARY_ADD;
    l_postops.ldd            = i_gemm_def->ldc;
  }

  if (i_gemm_def->unary_postop == SIGMOID) {
    l_argops.ldcp = i_gemm_def->ldc;
    l_argops.cp_unary_type  = LIBXSMM_MELTW_TYPE_UNARY_SIGMOID;
  }

  if (i_gemm_def->unary_postop == RELU_NOBITMASK) {
    l_argops.ldcp = i_gemm_def->ldc;
    l_argops.cp_unary_type  = LIBXSMM_MELTW_TYPE_UNARY_RELU;
  }

  if (i_gemm_def->unary_postop == RELU_BITMASK) {
    l_argops.ldcp = i_gemm_def->ldc;
    l_argops.cp_unary_type  = LIBXSMM_MELTW_TYPE_UNARY_RELU;
    l_argops.cp_unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
  }

  l_start = libxsmm_timer_tick();
  if (i_gemm_def->tc_config) {
    l_cfg_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | l_flags;
    l_rls_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | l_flags;
    l_flags |= (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
    cfg_tr.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_cfg_flags, l_prefetch_flags, l_brconfig );
    rls_tr.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_rls_flags, l_prefetch_flags, l_brconfig );
  }
#if defined(USE_GEMM_EXT_FRONTEND)
  l_test_jit.gemm_ext = libxsmm_dispatch_brgemm_ext_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig, l_argops, l_postops );
#else
  l_test_jit.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
#endif
  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  if (l_test_jit.xmm == 0) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(EXIT_FAILURE);
  }

  /* receive kernel information */
  libxsmm_get_mmkernel_info(l_test_jit, &l_info);

  /* run external tileconfig */
  if (i_gemm_def->tc_config) {
    cfg_tr.gemm( NULL );
  }

  /* reset GEMM paramater */
#if defined(USE_GEMM_EXT_FRONTEND)
  memset( &gemm_param, 0, sizeof(libxsmm_gemm_ext_param) );
#else
  memset( &gemm_param, 0, sizeof(libxsmm_gemm_param) );
#endif

  if (i_gemm_def->binary_postop == COLBIAS_ADD) {
    gemm_param.d.primary = (void*)i_fusion_arguments->colbias;
  }

  if (i_gemm_def->unary_postop == RELU_BITMASK) {
    gemm_param.c.secondary  = (void*) i_fusion_arguments->relu_bitmask;
  }

  gemm_param.op.tertiary = &l_br;
  gemm_param.c.primary = (void*)o_c;
  gemm_param.c.tertiary = (void*)(( i_gemm_def->unsigned_c != 0 ) ? &(i_gemm_def->scf) : NULL);
  /* run correctness */
  if (i_gemm_def->br_type == 0) {
    gemm_param.a.primary = (void*)i_a;
    gemm_param.b.primary = (void*)i_b;
    if ( l_info.prefetch != LIBXSMM_GEMM_PREFETCH_NONE ) {
      gemm_param.a.quaternary = (void*)i_a;
      gemm_param.b.quaternary = (void*)i_b;
      gemm_param.c.quaternary = (void*)o_c;
    }
#if defined(USE_GEMM_EXT_FRONTEND)
    l_test_jit.gemm_ext( &gemm_param );
#else
    l_test_jit.gemm( &gemm_param );
#endif
  } else if (i_gemm_def->br_type == 1) {
    gemm_param.a.primary = l_a_addr;
    gemm_param.b.primary = l_b_addr;
    for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
      l_a_addr[l_r] = (char*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k * LIBXSMM_TYPESIZE(i_gemm_def->in_type));
      if (i_gemm_def->trans_b == 0) {
        l_b_addr[l_r] = (char*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * LIBXSMM_TYPESIZE(i_gemm_def->in_type));
      } else {
        l_b_addr[l_r] = (char*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->k * LIBXSMM_TYPESIZE(i_gemm_def->in_type));
      }
    }
#if defined(USE_GEMM_EXT_FRONTEND)
    l_test_jit.gemm_ext( &gemm_param );
#else
    l_test_jit.gemm( &gemm_param );
#endif
  } else if (i_gemm_def->br_type == 2) {
    gemm_param.a.primary = (void*)i_a;
    gemm_param.a.secondary = l_a_offs;
    gemm_param.b.primary = (void*)i_b;
    gemm_param.b.secondary = l_b_offs;
#if defined(USE_GEMM_EXT_FRONTEND)
    l_test_jit.gemm_ext( &gemm_param );
#else
    l_test_jit.gemm( &gemm_param );
#endif
  } else if (i_gemm_def->br_type == 3) {
    gemm_param.a.primary = (void*)i_a;
    gemm_param.b.primary = (void*)i_b;
#if defined(USE_GEMM_EXT_FRONTEND)
    l_test_jit.gemm_ext( &gemm_param );
#else
    l_test_jit.gemm( &gemm_param );
#endif
  }

  /* run performance */
  gemm_param.c.primary = (void*)o_c_perf;
  l_start = libxsmm_timer_tick();
  if (i_gemm_def->br_type == 0) {
    gemm_param.a.primary = (void*)i_a;
    gemm_param.b.primary = (void*)i_b;
    if ( l_info.prefetch != LIBXSMM_GEMM_PREFETCH_NONE ) {
      gemm_param.a.quaternary = (void*)i_a;
      gemm_param.b.quaternary = (void*)i_b;
      gemm_param.c.quaternary = (void*)o_c_perf;
    }
    for (l_t = 0; l_t < i_reps; l_t++) {
#if defined(USE_GEMM_EXT_FRONTEND)
      l_test_jit.gemm_ext( &gemm_param );
#else
      l_test_jit.gemm( &gemm_param );
#endif
    }
  } else if (i_gemm_def->br_type == 1) {
    gemm_param.a.primary = l_a_addr;
    gemm_param.b.primary = l_b_addr;
    for (l_t = 0; l_t < i_reps; l_t++) {
      for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
        l_a_addr[l_r] = (char*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k * LIBXSMM_TYPESIZE(i_gemm_def->in_type));
        if (i_gemm_def->trans_b == 0) {
          l_b_addr[l_r] = (char*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * LIBXSMM_TYPESIZE(i_gemm_def->in_type));
        } else {
          l_b_addr[l_r] = (char*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->k * LIBXSMM_TYPESIZE(i_gemm_def->in_type));
        }
      }
#if defined(USE_GEMM_EXT_FRONTEND)
      l_test_jit.gemm_ext( &gemm_param );
#else
      l_test_jit.gemm( &gemm_param );
#endif
    }
  } else if (i_gemm_def->br_type == 2) {
    gemm_param.a.primary = (void*)i_a;
    gemm_param.a.secondary = l_a_offs;
    gemm_param.b.primary = (void*)i_b;
    gemm_param.b.secondary = l_b_offs;
    for (l_t = 0; l_t < i_reps; l_t++) {
#if defined(USE_GEMM_EXT_FRONTEND)
      l_test_jit.gemm_ext( &gemm_param );
#else
      l_test_jit.gemm( &gemm_param );
#endif
    }
  } else if (i_gemm_def->br_type == 3) {
    gemm_param.a.primary = (void*)i_a;
    gemm_param.b.primary = (void*)i_b;
    for (l_t = 0; l_t < i_reps; l_t++) {
#if defined(USE_GEMM_EXT_FRONTEND)
      l_test_jit.gemm_ext( &gemm_param );
#else
      l_test_jit.gemm( &gemm_param );
#endif
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  /* run external tilerelease */
  if (i_gemm_def->tc_config) {
    rls_tr.gemm( NULL );
  }

  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  free( (void*)l_a_addr );
  free( (void*)l_b_addr );
  free( (void*)l_a_offs );
  free( (void*)l_b_offs );

  return l_runtime;
}

void print_help(void) {
  printf("\n\n");
  printf("1. Usage (dense*dense=dense, correctness and performance):\n");
  printf("    M\n");
  printf("    N\n");
  printf("    K\n");
  printf("    LDA\n");
  printf("    LDB\n");
  printf("    LDC\n");
  printf("    alpha: 1\n");
  printf("    beta: 0 or 1\n");
  printf("    0: unaligned A, otherwise aligned\n");
  printf("    0: unaligned C, otherwise aligned\n");
  printf("    0: A normal, 1: A trans\n");
  printf("    0: B normal, 1: B trans\n");
  printf("    PREFETCH: nopf (none), pfsigonly, BL2viaC, AL2, curAL2, AL2_BL2viaC, curAL2_BL2viaC\n");
  printf("    PRECISION: SP, DP, I16I32, USI8I32, SUI8I32, SUI8UI8, BF16F32, BF16, BF16F32_FLAT, BF16_FLAT\n");
  printf("    BRGEMM: nobr, addrbr, offsbr, strdbr\n");
  printf("    BRsize: 1 - N\n");
  printf("    BRunroll: 0/1\n");
  printf("    #repetitions\n");
  printf("    tile configuration: 1 - external, 0 - internal\n");
  printf("    post_gemm_binary: 0 - none, 1 - colbias_add\n");
  printf("    post_gemm_unary: 0 - none, 1 - relu_nobitmask, 2 - relu_bitmask, 3 - sigmoid \n");
  printf("\n\n");
  printf("2. Usage (dense*dense=dense, performance only option available):\n");
  printf("    filename with space-sperated sizes (M N K LDA LDB LDC)\n");
  printf("    alpha: 1\n");
  printf("    beta: 0 or 1\n");
  printf("    0: unaligned A, otherwise aligned\n");
  printf("    0: unaligned C, otherwise aligned\n");
  printf("    0: A normal, 1: A trans\n");
  printf("    0: B normal, 1: B trans\n");
  printf("    PRECISION: SP, DP, I16I32, USI8I32, SUI8I32, SUI8UI8, BF16F32, BF16, BF16F32_FLAT, BF16_FLAT\n");
  printf("    BRGEMM: nobr, addrbr, offsbr, strdbr\n");
  printf("    BRsize: 1 - N\n");
  printf("    BRunroll: 0/1\n");
  printf("    #repetitions\n");
  printf("    0: no check, otherwise: run check\n");
  printf("    tile configuration: 1 - external, 0 - internal\n");
  printf("    post_gemm_binary: 0 - none, 1 - colbias_add\n");
  printf("    post_gemm_unary: 0 - none, 1 - relu_nobitmask, 2 - relu_bitmask, 3 - sigmoid \n");
  printf("\n\n");
}

int main(int argc, char* argv []) {
  char* l_precision = NULL;
  libxsmm_blasint l_lda = 0, l_ldb = 0, l_ldc = 0;
  libxsmm_blasint l_m = 0, l_n = 0, l_k = 0;
  int l_aligned_a = 0;
  int l_aligned_c = 0;
  int l_trans_a = 0;
  int l_trans_b = 0;
  double l_alpha = 0;
  double l_beta = 0;
  int l_br = 1;
  int l_br_type = 0;
  int l_br_unroll = 0;
  double l_runtime_libxsmm = 0;
  int l_file_input = 0;
  char* l_file_name = NULL;
  FILE *l_file_handle = NULL;
  int l_run_check = 0;
  double l_total_max_error = 0.0;
  int l_tc_config = 0;
  int l_reps;
  int l_binary_postop;
  int l_unary_postop;
  libxsmm_gemm_prefetch_type l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  gemm_def l_gemm_def;

# if defined(__APPLE__) && defined(__arm64__)
#  if 1
  pthread_set_qos_class_self_np( QOS_CLASS_USER_INTERACTIVE, 0 );
#  else
  pthread_set_qos_class_self_np( QOS_CLASS_BACKGROUND, 0 );
#  endif
# endif

  /* check argument count for a valid range */
  if ( argc == 22 ) {
    /* xgemm sizes */
    l_m = atoi(argv[1]);
    l_n = atoi(argv[2]);
    l_k = atoi(argv[3]);
    l_lda = atoi(argv[4]);
    l_ldb = atoi(argv[5]);
    l_ldc = atoi(argv[6]);

    /* some sugar */
    l_alpha = atof(argv[7]);
    l_beta = atof(argv[8]);
    l_aligned_a = atoi(argv[9]);
    l_aligned_c = atoi(argv[10]);
    l_trans_a = atoi(argv[11]);
    l_trans_b = atoi(argv[12]);

    /* arch specific stuff */
    l_precision = argv[14];
    l_br = atoi(argv[16]);
    l_br_unroll = atoi(argv[17]);
    l_reps = atoi(argv[18]);
    l_tc_config = atoi(argv[19]);
    l_binary_postop = atoi(argv[20]);
    l_unary_postop = atoi(argv[21]);

    /* set value of prefetch flag */
    if (strcmp("nopf", argv[13]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
    }
    else if (strcmp("pfsigonly", argv[13]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_SIGONLY;
    }
    else if (strcmp("BL2viaC", argv[13]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_BL2_VIA_C;
    }
    else if (strcmp("curAL2", argv[13]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_AL2_AHEAD;
    }
    else if (strcmp("curAL2_BL2viaC", argv[13]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD;
    }
    else if (strcmp("AL2", argv[13]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_AL2;
    }
    else if (strcmp("AL2_BL2viaC", argv[13]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C;
    }
    else {
      print_help();
      return EXIT_FAILURE;
    }

    if (strcmp("nobr", argv[15]) == 0) {
      l_br_type = 0;
    }
    else if (strcmp("addrbr", argv[15]) == 0) {
      l_br_type = 1;
    }
    else if (strcmp("offsbr", argv[15]) == 0) {
      l_br_type = 2;
    }
    else if (strcmp("strdbr", argv[15]) == 0) {
      l_br_type = 3;
    }
    else {
      print_help();
      return EXIT_FAILURE;
    }

    l_file_input = 0;
    l_run_check = 1;
  } else if ( argc == 17 ) {
    l_file_input = 1;
    l_file_name = argv[1];
    l_alpha = atof(argv[2]);
    l_beta = atof(argv[3]);
    l_aligned_a = atoi(argv[4]);
    l_aligned_c = atoi(argv[5]);
    l_trans_a = atoi(argv[6]);
    l_trans_b = atoi(argv[7]);
    l_precision = argv[8];
    l_br = atoi(argv[10]);
    l_br_unroll = atoi(argv[11]);
    l_tc_config = atoi(argv[14]);
    l_binary_postop = atoi(argv[15]);
    l_unary_postop = atoi(argv[16]);

    if (strcmp("nobr", argv[9]) == 0) {
      l_br_type = 0;
    }
    else if (strcmp("addrbr", argv[9]) == 0) {
      l_br_type = 1;
    }
    else if (strcmp("offsbr", argv[9]) == 0) {
      l_br_type = 2;
    }
    else if (strcmp("strdbr", argv[9]) == 0) {
      l_br_type = 3;
    }
    else {
      print_help();
      return EXIT_FAILURE;
    }
    l_reps = atoi(argv[12]);
    l_run_check = atoi(argv[13]);
    l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  } else {
    print_help();
    return EXIT_FAILURE;
  }

  const char *env_arch = getenv("LIBXSMM_TARGET");
  const int is_env_SPR = (
      env_arch == libxsmm_stristr(env_arch, "spr") ||
      env_arch == libxsmm_stristr(env_arch, "amx"));
  int arch_cpuid = libxsmm_cpuid();

  if ((!is_env_SPR && arch_cpuid < LIBXSMM_X86_AVX512_SPR)
       && (l_tc_config)) {
    printf("Warning: external tile configuration will be ingnored\n");
    l_tc_config = 0;
  }

  l_br = (l_br < 1) ? 1 : l_br;
  l_br = (l_br_type == 0) ? 1 : l_br;
  l_br_unroll = (l_br_type == 0) ? 0 : l_br_unroll;

  /* check alpha */
  if ( LIBXSMM_NEQ(l_alpha, 1.0) ) {
    fprintf(stderr, "JIT: alpha needs to be 1.0!\n");
    exit(EXIT_FAILURE);
  }

  /* check beta */
  if ( LIBXSMM_NEQ(l_beta, 0.0) && LIBXSMM_NEQ(l_beta, 1.0) ) {
    fprintf(stderr, "JIT: beta needs to be 0.0 or 1.0!\n");
    exit(EXIT_FAILURE);
  }

  /* setting static GEMM parameters */
  l_gemm_def.alpha = l_alpha;
  l_gemm_def.beta = l_beta;
  l_gemm_def.trans_a = l_trans_a;
  l_gemm_def.trans_b = l_trans_b;
  l_gemm_def.vnni_a = 0;
  l_gemm_def.vnni_b = 0;
  l_gemm_def.vnni_c = 0;
  l_gemm_def.unsigned_a = 0;
  l_gemm_def.unsigned_b = 0;
  l_gemm_def.unsigned_c = 0;
  l_gemm_def.aligned_a = l_aligned_a;
  l_gemm_def.aligned_c = l_aligned_c;
  l_gemm_def.prefetch = l_prefetch;
  l_gemm_def.br_type = l_br_type;
  l_gemm_def.br_count = l_br;
  l_gemm_def.br_unroll = l_br_unroll;
  l_gemm_def.tc_config = l_tc_config;
  l_gemm_def.scf = 0.0;
  l_gemm_def.binary_postop = l_binary_postop;
  l_gemm_def.unary_postop  = l_unary_postop;

  /* setting precision in GEMM struct */
  if ( (strcmp(l_precision, "DP") == 0) ) {
    l_gemm_def.in_type = LIBXSMM_DATATYPE_F64;
    l_gemm_def.out_type = LIBXSMM_DATATYPE_F64;
    l_gemm_def.comp_type = LIBXSMM_DATATYPE_F64;
  } else if ( (strcmp(l_precision, "SP") == 0) ) {
    l_gemm_def.in_type = LIBXSMM_DATATYPE_F32;
    l_gemm_def.out_type = LIBXSMM_DATATYPE_F32;
    l_gemm_def.comp_type = LIBXSMM_DATATYPE_F32;
  } else if ( (strcmp(l_precision, "I16I32") == 0) ) {
    l_gemm_def.in_type = LIBXSMM_DATATYPE_I16;
    l_gemm_def.out_type = LIBXSMM_DATATYPE_I32;
    l_gemm_def.comp_type = LIBXSMM_DATATYPE_I32;
    l_gemm_def.vnni_a = 1;
    l_gemm_def.trans_a = 0;
    l_gemm_def.trans_b = 0;
  } else if (strcmp(l_precision, "USI8I32") == 0) {
    l_gemm_def.in_type = LIBXSMM_DATATYPE_I8;
    l_gemm_def.out_type = LIBXSMM_DATATYPE_I32;
    l_gemm_def.comp_type = LIBXSMM_DATATYPE_I32;
    l_gemm_def.vnni_a = 1;
    l_gemm_def.trans_a = 0;
    l_gemm_def.trans_b = 0;
    l_gemm_def.unsigned_a = 1;
  } else if (strcmp(l_precision, "SUI8I32") == 0) {
    l_gemm_def.in_type = LIBXSMM_DATATYPE_I8;
    l_gemm_def.out_type = LIBXSMM_DATATYPE_I32;
    l_gemm_def.comp_type = LIBXSMM_DATATYPE_I32;
    l_gemm_def.vnni_a = 1;
    l_gemm_def.trans_a = 0;
    l_gemm_def.trans_b = 0;
    l_gemm_def.unsigned_b = 1;
  } else if (strcmp(l_precision, "SUI8UI8") == 0) {
    l_gemm_def.in_type = LIBXSMM_DATATYPE_I8;
    l_gemm_def.out_type = LIBXSMM_DATATYPE_I32;
    l_gemm_def.comp_type = LIBXSMM_DATATYPE_I32;
    l_gemm_def.vnni_a = 1;
    l_gemm_def.trans_a = 0;
    l_gemm_def.trans_b = 0;
    l_gemm_def.unsigned_b = 1;
    l_gemm_def.unsigned_c = 1;
    l_gemm_def.scf = 1.0f;
  } else if (strcmp(l_precision, "BF16F32") == 0) {
    l_gemm_def.in_type = LIBXSMM_DATATYPE_BF16;
    l_gemm_def.out_type = LIBXSMM_DATATYPE_F32;
    l_gemm_def.comp_type = LIBXSMM_DATATYPE_F32;
    l_gemm_def.vnni_a = 1;
    l_gemm_def.trans_a = 0;
    l_gemm_def.trans_b = 0;
  } else if (strcmp(l_precision, "BF16") == 0) {
    l_gemm_def.in_type = LIBXSMM_DATATYPE_BF16;
    l_gemm_def.out_type = LIBXSMM_DATATYPE_BF16;
    l_gemm_def.comp_type = LIBXSMM_DATATYPE_F32;
    l_gemm_def.vnni_a = 1;
    l_gemm_def.trans_a = 0;
    l_gemm_def.trans_b = 0;
  } else if (strcmp(l_precision, "BF16F32_FLAT") == 0) {
    l_gemm_def.in_type = LIBXSMM_DATATYPE_BF16;
    l_gemm_def.out_type = LIBXSMM_DATATYPE_F32;
    l_gemm_def.comp_type = LIBXSMM_DATATYPE_F32;
  } else if (strcmp(l_precision, "BF16_FLAT") == 0) {
    l_gemm_def.in_type = LIBXSMM_DATATYPE_BF16;
    l_gemm_def.out_type = LIBXSMM_DATATYPE_BF16;
    l_gemm_def.comp_type = LIBXSMM_DATATYPE_F32;
  } else {
    fprintf(stderr, "Unsupported precision %s!\n", l_precision);
    exit(EXIT_FAILURE);
  }

  /* Test if valid precision for fusion ops  */
  if ( !((strcmp(l_precision, "SP") == 0) ||
        (strcmp(l_precision, "BF16F32") == 0) ||
        (strcmp(l_precision, "BF16") == 0) ||
        (strcmp(l_precision, "BF16F32_FLAT") == 0) ||
        (strcmp(l_precision, "BF16_FLAT") == 0)) ) {
    fprintf(stderr, "Unsupported precision for fused xgemm kernel %s!\n", l_precision);
    exit(EXIT_FAILURE);
  }

  if ( l_file_input != 0 ) {
    l_file_handle = fopen( l_file_name, "r" );
  } else {
    if ( l_trans_b == 0 ) {
      printf("------------------------------------------------\n");
      printf("RUNNING (%ix%i) X (%ix%i) = (%ix%i), %s, BR=%i\n", l_m, l_k, l_k, l_n, l_m, l_n, l_precision, l_br);
      printf("------------------------------------------------\n");
    } else {
      printf("------------------------------------------------\n");
      printf("RUNNING (%ix%i) X (%ix%i)^T = (%ix%i), %s, BR=%i\n", l_m, l_k, l_k, l_n, l_m, l_n, l_precision, l_br);
      printf("------------------------------------------------\n");
    }
  }

  unsigned int l_keep_going = 0;
  do {
    double error = 0.0;
    double error_bitmask = 0.0;
    char *l_a, *l_b, *l_c, *l_c_perf, *l_c_gold, *l_colbias, *l_relu_bitmask, *l_relu_bitmask_gold;
    fusion_args fusion_arguments;
    fusion_args ref_fusion_arguments;

    if ( l_file_input != 0 ) {
      char l_line[512];
      if ( fgets( l_line, 512, l_file_handle) == NULL ) {
        l_keep_going = 0;
        break;
      } else {
        l_keep_going = 1;
      }
      if ( 6 != sscanf( l_line, "%i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_lda, &l_ldb, &l_ldc ) ) exit(EXIT_FAILURE);
    }

    l_gemm_def.m = l_m;
    l_gemm_def.n = l_n;
    l_gemm_def.k = l_k;
    l_gemm_def.lda = l_lda;
    l_gemm_def.ldb = l_ldb;
    l_gemm_def.ldc = l_ldc;

    l_a      = (char*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * LIBXSMM_TYPESIZE(l_gemm_def.in_type), 64);
    if (l_gemm_def.trans_b == 0) {
      l_b      = (char*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * (size_t)l_br * LIBXSMM_TYPESIZE(l_gemm_def.in_type), 64);
    } else {
      l_b      = (char*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_k * (size_t)l_br * LIBXSMM_TYPESIZE(l_gemm_def.in_type), 64);
    }
    l_c                 = (char*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * LIBXSMM_TYPESIZE(l_gemm_def.out_type), 64);
    l_c_perf            = (char*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * LIBXSMM_TYPESIZE(l_gemm_def.out_type), 64);
    l_c_gold            = (char*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * LIBXSMM_TYPESIZE(l_gemm_def.out_type), 64);
    l_colbias           = (char*)libxsmm_aligned_malloc((size_t)l_ldc * LIBXSMM_TYPESIZE(l_gemm_def.out_type), 64);
    l_relu_bitmask      = (char*)libxsmm_aligned_malloc(((size_t)l_ldc/8) * (size_t)l_n, 64);
    l_relu_bitmask_gold = (char*)libxsmm_aligned_malloc(((size_t)l_ldc/8) * (size_t)l_n, 64);
    fusion_arguments.colbias      = l_colbias;
    fusion_arguments.relu_bitmask = l_relu_bitmask;
    ref_fusion_arguments.colbias      = l_colbias;
    ref_fusion_arguments.relu_bitmask = l_relu_bitmask_gold;

    init_random_matrix( l_gemm_def.in_type, l_a, l_br, l_lda, l_k );
    if (l_gemm_def.trans_b == 0) {
      init_random_matrix( l_gemm_def.in_type, l_b, l_br, l_ldb, l_n );
    } else {
      init_random_matrix( l_gemm_def.in_type, l_b, l_br, l_ldb, l_k );
    }
    if ( l_beta == 0 ) {
      init_garbage_matrix( l_gemm_def.out_type, l_c,      1, l_ldc, l_n );
      init_garbage_matrix( l_gemm_def.out_type, l_c_perf, 1, l_ldc, l_n );
      init_garbage_matrix( l_gemm_def.out_type, l_c_gold, 1, l_ldc, l_n );
    } else {
      init_zero_matrix( l_gemm_def.out_type, l_c,      1, l_ldc, l_n );
      init_zero_matrix( l_gemm_def.out_type, l_c_perf, 1, l_ldc, l_n );
      init_zero_matrix( l_gemm_def.out_type, l_c_gold, 1, l_ldc, l_n );
    }

    init_random_matrix( LIBXSMM_DATATYPE_I8, l_relu_bitmask, 1, l_ldc/8, l_n );
    init_random_matrix( l_gemm_def.out_type, l_colbias, 1, l_ldc, 1 );
    memcpy(l_relu_bitmask_gold, l_relu_bitmask, (l_ldc/8) * l_n * sizeof(char));

    /* run gold solution */
    fused_matmul( &l_gemm_def, l_a, l_b, l_c_gold, &ref_fusion_arguments );

    /* run LIBXSMM solution */
    l_runtime_libxsmm = jit_matmul( &l_gemm_def, l_a, l_b, l_c, l_c_perf, l_reps, l_file_input, &fusion_arguments);

    /* run compare */
    error = check_matrix( l_gemm_def.out_type, l_c_gold, l_c, l_ldc, l_m, l_n );
    if (l_unary_postop == RELU_BITMASK) {
      error_bitmask = check_matrix( LIBXSMM_DATATYPE_I8, l_relu_bitmask_gold, l_relu_bitmask, l_ldc/8, l_m/8, l_n );
    }

    if ( l_file_input == 0 ) {
      printf("%fs for libxsmm\n", l_runtime_libxsmm);
      printf("%f GFLOPS for libxsmm\n", ((double)((double)l_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
      printf("max. error: %f\n", error);
      if (l_unary_postop == RELU_BITMASK) {
        printf("max. error relu bitmask: %f\n", error_bitmask);
      }

      check_matrix_norms( l_gemm_def.out_type, l_c_gold, l_c, l_ldc, l_m, l_n );

    } else {
      if ( l_run_check == 1 ) {
        printf("%i %i %i %i %i %i %i %i %i %s %f %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)l_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9), error );
        } else {
        printf("%i %i %i %i %i %i %i %i %i %s %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)l_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
      }
    }

    if ( (l_total_max_error < error) && (l_run_check == 1) ) {
      l_total_max_error = error;
    }

    libxsmm_free(l_a);
    libxsmm_free(l_b);
    libxsmm_free(l_c);
    libxsmm_free(l_c_perf);
    libxsmm_free(l_c_gold);
    libxsmm_free(l_colbias);
    libxsmm_free(l_relu_bitmask);
    libxsmm_free(l_relu_bitmask_gold);
  } while ( l_keep_going );

  if ( l_file_input != 0 ) {
    fclose( l_file_handle );
  } else {
    printf("------------------------------------------------\n");
  }

  /* Print total max error */
  printf("\n\n Total Max Error %f\n\n", l_total_max_error );

  if ( l_total_max_error >= 0.00005 && l_br_type == 0) {
    return EXIT_FAILURE;
  } else if ( l_total_max_error >= 0.0005 && l_br_type > 0) {
    return EXIT_FAILURE;
  } else {
    return EXIT_SUCCESS;
  }
}

