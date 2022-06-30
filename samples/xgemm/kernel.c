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
#include <stdio.h>
#include <math.h>
#include <string.h>
# if defined(__APPLE__) && defined(__arm64__)
#include <pthread.h>
# endif

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
} gemm_def;

double get_random_posneg_p5_num() {
  double tmp = libxsmm_rng_f64()-0.5;

  if ( tmp < -0.4 ) {
    tmp = -0.4;
  } else if ( tmp < -0.3 ) {
    tmp = -0.3;
  } else if ( tmp < -0.2 ) {
    tmp = -0.2;
  } else if ( tmp < -0.1 ) {
    tmp = -0.1;
  } else if ( tmp < 0 ) {
    tmp = 0;
  } else if ( tmp < 0.1 ) {
    tmp = 0.1;
  } else if ( tmp < 0.2 ) {
    tmp = 0.2;
  } else if ( tmp < 0.3 ) {
    tmp = 0.3;
  } else if ( tmp < 0.4 ) {
    tmp = 0.4;
  } else if ( tmp < 0.5 ) {
    tmp = 0.5;
  } else {
    tmp = 0.5;
  }

  return tmp;
}

double get_random_pos_p5_num() {
  double tmp = libxsmm_rng_f64();

  if ( tmp < 0.1 ) {
    tmp = 0.1;
  } else if ( tmp < 0.2 ) {
    tmp = 0.2;
  } else if ( tmp < 0.3 ) {
    tmp = 0.3;
  } else if ( tmp < 0.4 ) {
    tmp = 0.4;
  } else if ( tmp < 0.5 ) {
    tmp = 0.5;
  } else if ( tmp < 0.6 ) {
    tmp = 0.6;
  } else if ( tmp < 0.7 ) {
    tmp = 0.7;
  } else if ( tmp < 0.8 ) {
    tmp = 0.8;
  } else if ( tmp < 0.9 ) {
    tmp = 0.9;
  } else if ( tmp < 1.0 ) {
    tmp = 1.0;
  } else {
    tmp = 1.0;
  }

  return tmp;
}

void init_random_matrix( const libxsmm_datatype dtype, void* data, const libxsmm_blasint br, const libxsmm_blasint ld, const libxsmm_blasint n, const libxsmm_blasint pos_val_only ) {
  double* d_data = (double*) data;
  float* f_data = (float*) data;
  libxsmm_bfloat16* bf16_data = (libxsmm_bfloat16*) data;
  libxsmm_bfloat8* bf8_data = (libxsmm_bfloat8*) data;
  int* i_data = (int*) data;
  short* s_data = (short*) data;
  char* sc_data = (char*) data;
  unsigned char* uc_data = (unsigned char*) data;
  libxsmm_blasint l_r, l_i, l_j;

  for (l_r = 0; l_r < br; l_r++) {
    for (l_i = 0; l_i < ld; l_i++) {
      for (l_j = 0; l_j < n; l_j++) {
        if ( dtype == LIBXSMM_DATATYPE_F64 ) {
          d_data[(l_r * ld * n) + (l_j * ld) + l_i] = get_random_posneg_p5_num();
        } else if ( dtype == LIBXSMM_DATATYPE_F32 ) {
          f_data[(l_r * ld * n) + (l_j * ld) + l_i] = (float)get_random_posneg_p5_num();
        } else if ( dtype == LIBXSMM_DATATYPE_BF16 ) {
          libxsmm_bfloat16_f32 tmp /*= { 0 }*/;
          tmp.f = (float)get_random_posneg_p5_num();
          bf16_data[(l_r * ld * n) + (l_j * ld) + l_i] = tmp.i[1];
        } else if ( dtype == LIBXSMM_DATATYPE_BF8 ) {
          union libxsmm_bfloat8_f16 tmp/* = { 0 }*/;
          tmp.hf = libxsmm_convert_f32_to_f16( (float)get_random_posneg_p5_num() );
          bf8_data[(l_r * ld * n) + (l_j * ld) + l_i] = tmp.i[1];
        } else if ( dtype == LIBXSMM_DATATYPE_I32 ) {
          i_data[(l_r * ld * n) + (l_j * ld) + l_i] = (int)  (get_random_posneg_p5_num() * 40.0);
        } else if ( dtype == LIBXSMM_DATATYPE_I16 ) {
          s_data[(l_r * ld * n) + (l_j * ld) + l_i] = (short)(get_random_posneg_p5_num() * 40.0);
        } else if ( dtype == LIBXSMM_DATATYPE_I8 ) {
          if ( pos_val_only != 0 ) {
            uc_data[(l_r * ld * n) + (l_j * ld) + l_i] = (unsigned char) (get_random_pos_p5_num() * 20.0);
          } else {
            sc_data[(l_r * ld * n) + (l_j * ld) + l_i] = (char) (get_random_posneg_p5_num() * 40.0);
          }
        } else {
        }
      }
    }
  }
}

void init_zero_matrix( const libxsmm_datatype dtype, void* data, const libxsmm_blasint br, const libxsmm_blasint ld, const libxsmm_blasint n ) {
  char* l_data = (char*) data;
  memset( l_data, 0x0, (size_t)br*ld*n*LIBXSMM_TYPESIZE(dtype) );
}

void init_garbage_matrix( const libxsmm_datatype dtype, void* data, const libxsmm_blasint br, const libxsmm_blasint ld, const libxsmm_blasint n ) {
  char* l_data = (char*) data;
  memset( l_data, 0xdeadbeef, (size_t)br*ld*n*LIBXSMM_TYPESIZE(dtype) );
}

void ref_matmul( const gemm_def* i_gemm_def, const void* a, const void* b, void* c ) {
  libxsmm_blasint l_r, l_j, l_i, l_s, l_k2;
  libxsmm_blasint lda = i_gemm_def->lda;
  libxsmm_blasint ldb = i_gemm_def->ldb;
  libxsmm_blasint ldc = i_gemm_def->ldc;
  libxsmm_blasint m = i_gemm_def->m;
  libxsmm_blasint n = i_gemm_def->n;
  libxsmm_blasint k = i_gemm_def->k;

  if ( (i_gemm_def->in_type   == LIBXSMM_DATATYPE_F64) &&
       (i_gemm_def->out_type  == LIBXSMM_DATATYPE_F64) &&
       (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F64)    ) {
    double* d_a = (double*)a;
    double* d_b = (double*)b;
    double* d_c = (double*)c;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        if ( i_gemm_def->beta == 0 ) {
          d_c[(l_j * ldc) + l_i] = 0.0;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          for (l_s = 0; l_s < k; l_s++) {
            if (i_gemm_def->trans_b == 0) {
              if (i_gemm_def->trans_a == 0) {
                d_c[(l_j * ldc) + l_i] += d_a[(l_r * lda * k) + (l_s * lda) + l_i] *
                                                   d_b[(l_r * ldb * n) + (l_j * ldb) + l_s];
              } else {
                d_c[(l_j * ldc) + l_i] += d_a[(l_r * lda * m) + (l_i * lda) + l_s] *
                                                   d_b[(l_r * ldb * n) + (l_j * ldb) + l_s];
              } /* if-else l_trans_a */
            } else {
              if (i_gemm_def->trans_a == 0) {
                d_c[(l_j * ldc) + l_i] += d_a[(l_r * lda * k) + (l_s * lda) + l_i] *
                                                   d_b[(l_r * ldb * k) + (l_s * ldb) + l_j];
              } else {
                d_c[(l_j * ldc) + l_i] += d_a[(l_r * lda * m) + (l_i * lda) + l_s] *
                                                   d_b[(l_r * ldb * k) + (l_s * ldb) + l_j];
              } /* if-else l_trans_a */
            } /* if-else l_trans_b */
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

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        if ( i_gemm_def->beta == 0 ) {
          f_c[(l_j * ldc) + l_i] = 0.0f;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          for (l_s = 0; l_s < k; l_s++) {
            if (i_gemm_def->trans_b == 0) {
              if (i_gemm_def->trans_a == 0) {
                f_c[(l_j * ldc) + l_i] += f_a[(l_r * lda * k) + (l_s * lda) + l_i] *
                                                   f_b[(l_r * ldb * n) + (l_j * ldb) + l_s];
              } else {
                f_c[(l_j * ldc) + l_i] += f_a[(l_r * lda * m) + (l_i * lda) + l_s] *
                                                   f_b[(l_r * ldb * n) + (l_j * ldb) + l_s];
              } /* if-else l_trans_a */
            } else {
              if (i_gemm_def->trans_a == 0) {
                f_c[(l_j * ldc) + l_i] += f_a[(l_r * lda * k) + (l_s * lda) + l_i] *
                                                   f_b[(l_r * ldb * k) + (l_s * ldb) + l_j];
              } else {
                f_c[(l_j * ldc) + l_i] += f_a[(l_r * lda * m) + (l_i * lda) + l_s] *
                                                   f_b[(l_r * ldb * k) + (l_s * ldb) + l_j];
              } /* if-else l_trans_a */
            } /* if-else l_trans_b */
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

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        if ( i_gemm_def->beta == 0 ) {
          i_c[(l_j * ldc) + l_i] = 0;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
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

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        if ( i_gemm_def->beta == 0 ) {
          i_c[(l_j * ldc) + l_i] = 0;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
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

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        if ( i_gemm_def->beta == 0 ) {
          i_c[(l_j * ldc) + l_i] = 0;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
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
              (i_gemm_def->out_type  == LIBXSMM_DATATYPE_F32) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_I32) &&
              (i_gemm_def->unsigned_a == 1) && (i_gemm_def->unsigned_b == 0) ) {
    unsigned char* c_a = (unsigned char*)a;
    char* c_b          = (char*)b;
    float*         c_c = (float*)c;
    int l_k_block = 4;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        int tmp = 0;
        float ftmp;
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              tmp += c_a[(l_r * lda * k) + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2] *
                     c_b[(l_r * ldb * n) + (l_j * ldb) + (l_s*l_k_block) + l_k2];
            }
          }
        }
        ftmp = (float)tmp;
        ftmp *= i_gemm_def->scf;
        if ( i_gemm_def->beta == 1 ) {
          ftmp += c_c[(l_j * ldc) + l_i];
        }

        c_c[(l_j * ldc) + l_i] = ftmp;
      }
    }
  } else if ( (i_gemm_def->in_type   == LIBXSMM_DATATYPE_I8)  &&
              (i_gemm_def->out_type  == LIBXSMM_DATATYPE_F32) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_I32) &&
              (i_gemm_def->unsigned_a == 0) && (i_gemm_def->unsigned_b == 1) ) {
    char*          c_a = (char*)a;
    unsigned char* c_b = (unsigned char*)b;
    float*         c_c = (float*)c;
    int l_k_block = 4;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        int tmp = 0;
        float ftmp;
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              tmp += c_a[(l_r * lda * k) + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2] *
                     c_b[(l_r * ldb * n) + (l_j * ldb) + (l_s*l_k_block) + l_k2];
            }
          }
        }
        ftmp = (float)tmp;
        ftmp *= i_gemm_def->scf;
        if ( i_gemm_def->beta == 1 ) {
          ftmp += c_c[(l_j * ldc) + l_i];
        }

        c_c[(l_j * ldc) + l_i] = ftmp;
      }
    }
  } else if ( (i_gemm_def->in_type   == LIBXSMM_DATATYPE_BF16) &&
              (i_gemm_def->out_type  == LIBXSMM_DATATYPE_F32)  &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F32)     ) {
    libxsmm_bfloat16* h_a = (libxsmm_bfloat16*)a;
    libxsmm_bfloat16* h_b = (libxsmm_bfloat16*)b;
    float*            f_c = (float*)c;
    int l_k_block = ( i_gemm_def->vnni_a != 0) ?  2 : 1;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        if ( i_gemm_def->beta == 0 ) {
          f_c[(l_j * ldc) + l_i] = 0.0f;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = l_k_block - 1; l_k2 >= 0; l_k2--) {
              libxsmm_bfloat16_f32 tmp_a_f = { 0 }, tmp_b_f = { 0 };
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

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        if ( i_gemm_def->beta == 0 ) {
          acc = 0.0f;
        } else {
          libxsmm_bfloat16_f32 tmp/* = { 0 }*/;
          tmp.i[0] = 0;
          tmp.i[1] = h_c[(l_j * ldc) + l_i];
          acc = tmp.f;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = l_k_block - 1; l_k2 >= 0; l_k2--) {
              libxsmm_bfloat16_f32 tmp_a_f = { 0 }, tmp_b_f = { 0 };
              tmp_a_f.i[0] = 0;
              tmp_a_f.i[1] = h_a[(l_r * lda * k) + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              tmp_b_f.i[0] = 0;
              tmp_b_f.i[1] = h_b[(l_r * ldb * n) + (l_j * ldb) + (l_s*l_k_block) + l_k2];

              acc += tmp_a_f.f * tmp_b_f.f;
            }
          }
        }
        libxsmm_rne_convert_fp32_bf16( &acc, &h_acc, 1 );
        h_c[(l_j * ldc) + l_i] = h_acc;
      }
    }
  } else if ( (i_gemm_def->in_type   == LIBXSMM_DATATYPE_BF8) &&
              (i_gemm_def->out_type  == LIBXSMM_DATATYPE_F32) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F32)    ) {
    libxsmm_bfloat8* h_a = (libxsmm_bfloat8*)a;
    libxsmm_bfloat8* h_b = (libxsmm_bfloat8*)b;
    float*           f_c = (float*)c;
    int l_k_block = 4;

    for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
      for (l_j = 0; l_j < n; l_j++) {
        for (l_i = 0; l_i < m; l_i++) {
          if ( (i_gemm_def->beta == 0) && (l_r == 0) ) {
            f_c[(l_j * ldc) + l_i] = 0.0f;
          }
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              union libxsmm_bfloat8_f16 tmp_a_hf;
              union libxsmm_bfloat8_f16 tmp_b_hf;
              float tmp_a_f;
              float tmp_b_f;
              tmp_a_hf.i[0] = 0;
              tmp_a_hf.i[1] = h_a[(l_r * lda * k) + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              tmp_b_hf.i[0] = 0;
              tmp_b_hf.i[1] = h_b[(l_r * ldb * n) + (l_j * ldb) + (l_s*l_k_block) + l_k2];
              tmp_a_f = libxsmm_convert_f16_to_f32( tmp_a_hf.hf );
              tmp_b_f = libxsmm_convert_f16_to_f32( tmp_b_hf.hf );

              f_c[(l_j * ldc) + l_i] += tmp_a_f * tmp_b_f;
            }
          }
        }
      }
    }
  } else if ( (i_gemm_def->in_type   == LIBXSMM_DATATYPE_BF8) &&
              (i_gemm_def->out_type  == LIBXSMM_DATATYPE_BF8) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F32)    ) {
    libxsmm_bfloat8* h_a = (libxsmm_bfloat8*)a;
    libxsmm_bfloat8* h_b = (libxsmm_bfloat8*)b;
    libxsmm_bfloat8* h_c = (libxsmm_bfloat8*)c;
    int l_k_block = 4;
    float acc = 0.0f;
    libxsmm_bfloat8 bf8_acc;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        if ( i_gemm_def->beta == 0 ) {
          acc = 0.0f;
        } else {
          union libxsmm_bfloat8_f16 tmp_c_hf;
          tmp_c_hf.i[0] = 0;
          tmp_c_hf.i[1] = h_c[(l_j * ldc) + l_i];
          acc = libxsmm_convert_f16_to_f32( tmp_c_hf.hf );
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              union libxsmm_bfloat8_f16 tmp_a_hf;
              union libxsmm_bfloat8_f16 tmp_b_hf;
              float tmp_a_f;
              float tmp_b_f;
              tmp_a_hf.i[0] = 0;
              tmp_a_hf.i[1] = h_a[(l_r * lda * k) + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              tmp_b_hf.i[0] = 0;
              tmp_b_hf.i[1] = h_b[(l_r * ldb * n) + (l_j * ldb) + (l_s*l_k_block) + l_k2];
              tmp_a_f = libxsmm_convert_f16_to_f32( tmp_a_hf.hf );
              tmp_b_f = libxsmm_convert_f16_to_f32( tmp_b_hf.hf );

              acc += tmp_a_f * tmp_b_f;
            }
          }
        }
        libxsmm_rne_convert_fp32_bf8( &acc, &bf8_acc, 1 );
        h_c[(l_j * ldc) + l_i] = bf8_acc;
      }
    }
  }
}

double check_matrix( const libxsmm_datatype dtype, const void* data_gold, const void* data, const libxsmm_blasint ld, const libxsmm_blasint m, const libxsmm_blasint n ) {
  libxsmm_matdiff_info l_diff;
  double error = 0.0;

  libxsmm_matdiff_clear(&l_diff);
  if ( dtype == LIBXSMM_DATATYPE_F64 ) {
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F64, m, n, data_gold, data, &ld, &ld);
    error = libxsmm_matdiff_epsilon(&l_diff);
  } else if ( dtype == LIBXSMM_DATATYPE_F32 ) {
#if 0
    libxsmm_blasint l_i, l_j;
    float* f_data =      (float*)data;
    float* f_data_gold = (float*)data_gold;
    for (l_i = 0; l_i < m; l_i++) {
      for (l_j = 0; l_j < n; l_j++) {
        printf("gold: %f, computed: %f\n", f_data_gold[(l_j * ld) + l_i], f_data[(l_j * ld) + l_i] );
      }
    }
#endif
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, m, n, data_gold, data, &ld, &ld);
    error = libxsmm_matdiff_epsilon(&l_diff);
  } else if ( dtype == LIBXSMM_DATATYPE_BF16 ) {
    float* data_gold_f = (float*)malloc( sizeof(float) * ld * n );
    float* data_f      = (float*)malloc( sizeof(float) * ld * n );

    libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)data_gold, data_gold_f, ld*n );
    libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)data,      data_f,      ld*n );
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, m, n, data_gold_f, data_f, &ld, &ld);
    error = libxsmm_matdiff_epsilon(&l_diff);

    free( data_f );
    free( data_gold_f ) ;
  } else if ( dtype == LIBXSMM_DATATYPE_BF8 ) {
    float* data_gold_f = malloc( ld * n * sizeof(float) );
    float* data_f      = malloc( ld * n * sizeof(float) );

    libxsmm_convert_bf8_f32( (libxsmm_bfloat8*)data_gold, data_gold_f, ld*n );
    libxsmm_convert_bf8_f32( (libxsmm_bfloat8*)data,      data_f,      ld*n );
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, m, n, data_gold_f, data_f, &ld, &ld);
    error = l_diff.normf_rel;

#if 0
    libxsmm_blasint l_i, l_j;
    float* f_data =      (float*)data;
    float* f_data_gold = (float*)data_gold;
    for (l_i = 0; l_i < m; l_i++) {
      for (l_j = 0; l_j < n; l_j++) {
        printf("gold: %f, computed: %f\n", data_gold_f[(l_j * ld) + l_i], data_f[(l_j * ld) + l_i] );
      }
    }
#endif

    free( data_f );
    free( data_gold_f ) ;
  } else if ( dtype == LIBXSMM_DATATYPE_I32 ) {
    libxsmm_blasint l_i, l_j;
    double* data_gold_f = (double*)malloc( sizeof(double) * ld * n );
    double* data_f      = (double*)malloc( sizeof(double)* ld * n );
    int* l_data = (int*)data;
    int* l_data_gold = (int*)data_gold;

    assert(NULL != data_gold_f && NULL != data_f);
    for (l_i = 0; l_i < m; l_i++) {
      for (l_j = 0; l_j < n; l_j++) {
        data_gold_f[(l_j * ld) + l_i] = (double)l_data_gold[(l_j * ld) + l_i];
        data_f[(l_j * ld) + l_i]      = (double)l_data[(l_j * ld) + l_i];
      }
    }

    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F64, m, n, data_gold_f, data_f, &ld, &ld);
    error = libxsmm_matdiff_epsilon(&l_diff);

    free( data_f );
    free( data_gold_f );
  } else if ( dtype == LIBXSMM_DATATYPE_I8 ) {
    libxsmm_blasint l_i, l_j;
    double* data_gold_f = (double*)malloc( sizeof(double) * ld * n );
    double* data_f      = (double*)malloc( sizeof(double) * ld * n );
    unsigned char* l_data      = (unsigned char*)data;
    unsigned char* l_data_gold = (unsigned char*)data_gold;
    assert(NULL != data_gold_f && NULL != data_f);
    for (l_i = 0; l_i < m; l_i++) {
      for (l_j = 0; l_j < n; l_j++) {
        data_gold_f[(l_j * ld) + l_i] = (double)l_data_gold[(l_j * ld) + l_i];
        data_f[(l_j * ld) + l_i]      = (double)l_data[(l_j * ld) + l_i];
      }
    }

    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F64, m, n, data_gold_f, data_f, &ld, &ld);
    error = libxsmm_matdiff_epsilon(&l_diff);

    free( data_f );
    free( data_gold_f );
  } else {
    error = 100.0;
  }

  printf("\nPrinting Norms:\n");
  printf("L1 reference  : %.25g\n", l_diff.l1_ref);
  printf("L1 test       : %.25g\n", l_diff.l1_tst);
  printf("L2 abs.error  : %.24f\n", l_diff.l2_abs);
  printf("L2 rel.error  : %.24f\n", l_diff.l2_rel);
  printf("Linf abs.error: %.24f\n", l_diff.linf_abs);
  printf("Linf rel.error: %.24f\n", l_diff.linf_rel);
  printf("Check-norm    : %.24f\n", error);
  printf("\n");

  return error;
}

double jit_matmul( const gemm_def*    i_gemm_def,
                   const void*        i_a,
                   const void*        i_b,
                   void*              o_c,
                   void*              o_c_perf,
                   const int          i_reps,
                   const unsigned int i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
  libxsmm_xmmfunction cfg_tr = { NULL };
  libxsmm_xmmfunction rls_tr = { NULL };
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  libxsmm_gemm_shape l_shape;
  libxsmm_gemm_batch_reduce_config l_brconfig = { 0 };
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
  char** l_a_addr = (char**)malloc(sizeof(char*)*i_gemm_def->br_count);
  char** l_b_addr = (char**)malloc(sizeof(char*)*i_gemm_def->br_count);
  long long* l_a_offs = (long long*)malloc(sizeof(long long)*i_gemm_def->br_count);
  long long* l_b_offs = (long long*)malloc(sizeof(long long)*i_gemm_def->br_count);
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
    for ( l_r = 0 ; l_r < (size_t)i_gemm_def->br_count; l_r++ ) {
      l_a_offs[l_r] = l_r * (long long)i_gemm_def->lda * i_gemm_def->k * LIBXSMM_TYPESIZE(i_gemm_def->in_type);
      if (i_gemm_def->trans_b == 0) {
        l_b_offs[l_r] = l_r * (long long)i_gemm_def->ldb * i_gemm_def->n * LIBXSMM_TYPESIZE(i_gemm_def->in_type);
      } else {
        l_b_offs[l_r] = l_r * (long long)i_gemm_def->ldb * i_gemm_def->k * LIBXSMM_TYPESIZE(i_gemm_def->in_type);
      }
    }
  }

  /* set up the flags */
  if ( i_gemm_def->trans_b != 0 ) {
    l_flags |= LIBXSMM_GEMM_FLAG_TRANS_B;
  }
  if ( i_gemm_def->trans_a != 0 ) {
    l_flags |= LIBXSMM_GEMM_FLAG_TRANS_A;
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

  /* setting BRGEMM config struct */
  if (i_gemm_def->br_type == 1) {
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_ADDRESS;
    l_brconfig.br_stride_a_hint = 0;
    l_brconfig.br_stride_b_hint = 0;
    l_brconfig.br_unroll_hint = (unsigned char)(( i_gemm_def->br_unroll == 0 ) ? 0 : i_gemm_def->br_count);
  } else if (i_gemm_def->br_type == 2) {
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_OFFSET;
    l_brconfig.br_stride_a_hint = 0;
    l_brconfig.br_stride_b_hint = 0;
    l_brconfig.br_unroll_hint = (unsigned char)(( i_gemm_def->br_unroll == 0 ) ? 0 : i_gemm_def->br_count);
  } else if (i_gemm_def->br_type == 3) {
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
    l_brconfig.br_stride_a_hint = i_gemm_def->lda*i_gemm_def->k*LIBXSMM_TYPESIZE(i_gemm_def->in_type);
    l_brconfig.br_stride_b_hint = (i_gemm_def->trans_b == 0) ? i_gemm_def->ldb*i_gemm_def->n*LIBXSMM_TYPESIZE(i_gemm_def->in_type) : i_gemm_def->ldb*i_gemm_def->k*LIBXSMM_TYPESIZE(i_gemm_def->in_type);
    l_brconfig.br_unroll_hint = (unsigned char)(( i_gemm_def->br_unroll == 0 ) ? 0 : i_gemm_def->br_count);
  } else {
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_NONE;
    l_brconfig.br_stride_a_hint = 0;
    l_brconfig.br_stride_b_hint = 0;
    l_brconfig.br_unroll_hint = 0;
  }

  /* setting prefetch flags */
  l_prefetch_flags = i_gemm_def->prefetch;

  /* setting ext structs to 0 */
  memset( &l_argops, 0, sizeof(libxsmm_gemm_ext_unary_argops) );
  memset( &l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops) );

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

  /* reset GEMM parameter */
#if defined(USE_GEMM_EXT_FRONTEND)
  memset( &gemm_param, 0, sizeof(libxsmm_gemm_ext_param) );
#else
  memset( &gemm_param, 0, sizeof(libxsmm_gemm_param) );
#endif

  gemm_param.op.tertiary = &l_br;
  gemm_param.c.primary = (void*)o_c;
  gemm_param.c.tertiary = (void*)(( i_gemm_def->in_type == LIBXSMM_DATATYPE_I8 && i_gemm_def->out_type == LIBXSMM_DATATYPE_F32 ) ? &(i_gemm_def->scf) : NULL);
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
    for ( l_r = 0 ; l_r < (size_t)i_gemm_def->br_count; l_r++ ) {
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
    test_jit.gemm_ext( &gemm_param );
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
    for (l_t = 0; l_t < (size_t)i_reps; l_t++) {
#if defined(USE_GEMM_EXT_FRONTEND)
      l_test_jit.gemm_ext( &gemm_param );
#else
      l_test_jit.gemm( &gemm_param );
#endif
    }
  } else if (i_gemm_def->br_type == 1) {
    gemm_param.a.primary = l_a_addr;
    gemm_param.b.primary = l_b_addr;
    assert(NULL != l_a_addr && NULL != l_b_addr);
    for (l_t = 0; l_t < (size_t)i_reps; l_t++) {
      for ( l_r = 0 ; l_r < (size_t)i_gemm_def->br_count; l_r++ ) {
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
    for (l_t = 0; l_t < (size_t)i_reps; l_t++) {
#if defined(USE_GEMM_EXT_FRONTEND)
      l_test_jit.gemm_ext( &gemm_param );
#else
      l_test_jit.gemm( &gemm_param );
#endif
    }
  } else if (i_gemm_def->br_type == 3) {
    gemm_param.a.primary = (void*)i_a;
    gemm_param.b.primary = (void*)i_b;
    for (l_t = 0; l_t < (size_t)i_reps; l_t++) {
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
  printf("    PRECISION: SP, DP, I16I32, USI8I32, SUI8I32, USI8F32, SUI8F32, BF16F32, BF16, BF16F32_FLAT, BF16_FLAT, BF8F32, BF8\n");
  printf("    BRGEMM: nobr, addrbr, offsbr, strdbr\n");
  printf("    BRsize: 1 - N\n");
  printf("    BRunroll: 0/1\n");
  printf("    #repetitions\n");
  printf("    tile configuration: 1 - external, 0 - internal\n");
  printf("\n\n");
  printf("2. Usage (dense*dense=dense, performance only option available):\n");
  printf("    filename with space-sperated sizes (M N K LDA LDB LDC)\n");
  printf("    alpha: 1\n");
  printf("    beta: 0 or 1\n");
  printf("    0: unaligned A, otherwise aligned\n");
  printf("    0: unaligned C, otherwise aligned\n");
  printf("    0: A normal, 1: A trans\n");
  printf("    0: B normal, 1: B trans\n");
  printf("    PRECISION: SP, DP, I16I32, USI8I32, SUI8I32, USI8F32, SUI8F32, BF16F32, BF16, BF16F32_FLAT, BF16_FLAT, BF8F32, BF8\n");
  printf("    BRGEMM: nobr, addrbr, offsbr, strdbr\n");
  printf("    BRsize: 1 - N\n");
  printf("    BRunroll: 0/1\n");
  printf("    #repetitions\n");
  printf("    0: no check, otherwise: run check\n");
  printf("    tile configuration: 1 - external, 0 - internal\n");
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
  libxsmm_gemm_prefetch_type l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  gemm_def l_gemm_def = { 0 };
  int l_n_threads = 1;
  unsigned int l_keep_going = 0;

# if defined(__APPLE__) && defined(__arm64__)
#  if 1
  pthread_set_qos_class_self_np( QOS_CLASS_USER_INTERACTIVE, 0 );
#  else
  pthread_set_qos_class_self_np( QOS_CLASS_BACKGROUND, 0 );
#  endif
# endif

  /* check argument count for a valid range */
  if ( argc == 20 || argc == 19 ) {
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
    if ( argc == 20 ) {
      l_tc_config = atoi(argv[19]);
    } else {
      l_tc_config = 0;
    }

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
  } else if ( argc == 15 || argc == 14 ) {
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
    if ( argc == 15 ) {
      l_tc_config = atoi(argv[14]);
    } else {
      l_tc_config = 0;
    }

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

  { const char* env_arch = getenv("LIBXSMM_TARGET");
    const int is_env_SPR = (
      env_arch == libxsmm_stristr(env_arch, "spr") ||
      env_arch == libxsmm_stristr(env_arch, "amx"));
    int arch_cpuid = libxsmm_cpuid();

    if ((!is_env_SPR && arch_cpuid < LIBXSMM_X86_AVX512_SPR)
      && (l_tc_config)) {
      printf("Warning: external tile configuration will be ingnored\n");
      l_tc_config = 0;
    }
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
  } else if (strcmp(l_precision, "USI8F32") == 0) {
    l_gemm_def.in_type = LIBXSMM_DATATYPE_I8;
    l_gemm_def.out_type = LIBXSMM_DATATYPE_F32;
    l_gemm_def.comp_type = LIBXSMM_DATATYPE_I32;
    l_gemm_def.vnni_a = 1;
    l_gemm_def.trans_a = 0;
    l_gemm_def.trans_b = 0;
    l_gemm_def.unsigned_a = 1;
    l_gemm_def.scf = 1.0f;
  } else if (strcmp(l_precision, "SUI8F32") == 0) {
    l_gemm_def.in_type = LIBXSMM_DATATYPE_I8;
    l_gemm_def.out_type = LIBXSMM_DATATYPE_F32;
    l_gemm_def.comp_type = LIBXSMM_DATATYPE_I32;
    l_gemm_def.vnni_a = 1;
    l_gemm_def.trans_a = 0;
    l_gemm_def.trans_b = 0;
    l_gemm_def.unsigned_b = 1;
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
  } else if (strcmp(l_precision, "BF8F32") == 0) {
    l_gemm_def.in_type = LIBXSMM_DATATYPE_BF8;
    l_gemm_def.out_type = LIBXSMM_DATATYPE_F32;
    l_gemm_def.comp_type = LIBXSMM_DATATYPE_F32;
    l_gemm_def.vnni_a = 1;
    l_gemm_def.trans_a = 0;
    l_gemm_def.trans_b = 0;
  } else if (strcmp(l_precision, "BF8") == 0) {
    l_gemm_def.in_type = LIBXSMM_DATATYPE_BF8;
    l_gemm_def.out_type = LIBXSMM_DATATYPE_BF8;
    l_gemm_def.comp_type = LIBXSMM_DATATYPE_F32;
    l_gemm_def.vnni_a = 1;
    l_gemm_def.trans_a = 0;
    l_gemm_def.trans_b = 0;
  } else {
    fprintf(stderr, "Unsupported precision %s!\n", l_precision);
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

  /* read the number of threads */
#if defined(_OPENMP) && defined(LIBXSMM_PARALLEL_KERNEL_TEST)
# pragma omp parallel
  {
#   pragma omp master
    {
      l_n_threads = omp_get_num_threads();
    }
  }
#else
  l_n_threads = 1;
#endif

  l_keep_going = 0;
  do {
    double error = 0.0;

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

    /* set rng seed */
    libxsmm_rng_set_seed( 555 );

#if defined(_OPENMP) && defined(LIBXSMM_PARALLEL_KERNEL_TEST)
#   pragma omp parallel reduction(+:l_runtime_libxsmm)
#endif
    {
      char *l_a, *l_b, *l_c, *l_c_perf, *l_c_gold;

      if (l_gemm_def.trans_a == 0) {
        l_a      = (char*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * LIBXSMM_TYPESIZE(l_gemm_def.in_type), 64);
      } else {
        l_a      = (char*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_m * (size_t)l_br * LIBXSMM_TYPESIZE(l_gemm_def.in_type), 64);
      }
      if (l_gemm_def.trans_b == 0) {
        l_b      = (char*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * (size_t)l_br * LIBXSMM_TYPESIZE(l_gemm_def.in_type), 64);
      } else {
        l_b      = (char*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_k * (size_t)l_br * LIBXSMM_TYPESIZE(l_gemm_def.in_type), 64);
      }
      l_c      = (char*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * LIBXSMM_TYPESIZE(l_gemm_def.out_type), 64);
      l_c_perf = (char*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * LIBXSMM_TYPESIZE(l_gemm_def.out_type), 64);
      l_c_gold = (char*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * LIBXSMM_TYPESIZE(l_gemm_def.out_type), 64);

      if (l_gemm_def.trans_a == 0) {
        init_random_matrix( l_gemm_def.in_type, l_a, l_br, l_lda, l_k, l_gemm_def.unsigned_a );
      } else {
        init_random_matrix( l_gemm_def.in_type, l_a, l_br, l_lda, l_m, l_gemm_def.unsigned_a );
      }
      if (l_gemm_def.trans_b == 0) {
        init_random_matrix( l_gemm_def.in_type, l_b, l_br, l_ldb, l_n, l_gemm_def.unsigned_b );
      } else {
        init_random_matrix( l_gemm_def.in_type, l_b, l_br, l_ldb, l_k, l_gemm_def.unsigned_b );
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

      /* run gold solution */
#if defined(_OPENMP) && defined(LIBXSMM_PARALLEL_KERNEL_TEST)
#     pragma omp master
#endif
      {
        ref_matmul( &l_gemm_def, l_a, l_b, l_c_gold );
      }

      /* run LIBXSMM solution */
      l_runtime_libxsmm = jit_matmul( &l_gemm_def, l_a, l_b, l_c, l_c_perf, l_reps, l_file_input );

      /* run compare */
#if defined(_OPENMP) && defined(LIBXSMM_PARALLEL_KERNEL_TEST)
#     pragma omp master
#endif
      {
        error = check_matrix( l_gemm_def.out_type, l_c_gold, l_c, l_ldc, l_m, l_n );
      }

      libxsmm_free(l_a);
      libxsmm_free(l_b);
      libxsmm_free(l_c);
      libxsmm_free(l_c_perf);
      libxsmm_free(l_c_gold);
    } /* close parallel region */

    { const char *prefetch = NULL, *br_type = NULL;
      switch (l_prefetch) {
        case LIBXSMM_GEMM_PREFETCH_NONE: prefetch = "nopf"; break;
        case LIBXSMM_GEMM_PREFETCH_SIGONLY: prefetch = "pfsigonly"; break;
        case LIBXSMM_GEMM_PREFETCH_BL2_VIA_C: prefetch = "BL2viaC"; break;
        case LIBXSMM_GEMM_PREFETCH_AL2_AHEAD: prefetch = "curAL2"; break;
        case LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD: prefetch = "curAL2_BL2viaC"; break;
        case LIBXSMM_GEMM_PREFETCH_AL2: prefetch = "AL2"; break;
        case LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C: prefetch = "AL2_BL2viaC"; break;
        default: prefetch = "unknown";
      }
      switch (l_br_type) {
        case 0: br_type = "nobr"; break;
        case 1: br_type = "addrbr"; break;
        case 2: br_type = "offsbr"; break;
        case 3: br_type = "strdbr"; break;
        default: br_type = "unknown";
      }
      assert(NULL != prefetch && NULL != br_type);
      l_runtime_libxsmm /= (double)l_n_threads;
      printf("Command line:\n%s %i %i %i %i %i %i %f %f %i %i %i %i %s %s %s %i %i %i %i\n\n", argv[0],
        l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_alpha, l_beta, l_aligned_a, l_aligned_c, l_trans_a, l_trans_b,
        prefetch, l_precision, br_type, l_br, l_br_unroll, l_reps, l_tc_config);
      printf("%fs for LIBXSMM\n", l_runtime_libxsmm);
      printf("%f GFLOPS\n", ((double)((double)l_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br * (double)l_n_threads) * 2.0) / (l_runtime_libxsmm * 1.0e9));
      printf("max. error: %f\n", error);
    }

    if ( (l_total_max_error < error) && (l_run_check == 1) ) {
      l_total_max_error = error;
    }
  } while ( l_keep_going );

  if ( l_file_input != 0 ) {
    fclose( l_file_handle );
  } else {
    printf("------------------------------------------------\n");
  }

  /* Print total max error */
  printf("\nTotal Max Error %f\n", l_total_max_error);

  if (0.005 <= l_total_max_error && l_gemm_def.out_type == LIBXSMM_DATATYPE_BF16) {
    printf("FAILURE\n");
    return EXIT_FAILURE;
  }
  else if (0.000005 <= l_total_max_error) {
    printf("FAILURE\n");
    return EXIT_FAILURE;
  }
  else {
    printf("SUCCESS\n");
    return EXIT_SUCCESS;
  }
}
