/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include "generator_mateltwise_common.h"
#include "generator_common.h"
#include "generator_gemm_reference_impl.h"
#include "generator_mateltwise_reference_impl.h"

#if 0
typedef struct libxsmm_gemm_def {
  libxsmm_datatype a_type;
  libxsmm_datatype b_type;
  libxsmm_datatype c_type;
  libxsmm_datatype comp_type;
  libxsmm_blasint m;
  libxsmm_blasint n;
  libxsmm_blasint k;
  libxsmm_blasint lda;
  libxsmm_blasint ldb;
  libxsmm_blasint ldc;
  libxsmm_blasint uop_ld;
  libxsmm_blasint bop_ld;
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
  float *scf_f32;
  float *zpt_f32;
  libxsmm_float16 *zpt_f16;
  unsigned char *zpt_u8;
  unsigned char *scf_u8;
  libxsmm_float16 *scf_f16;
  int binary_postop;
  int unary_postop;
  unsigned int is_Ai4Bf16_gemm;
  unsigned int is_Amxfp4Bbf16_gemm;
  unsigned int is_Amxfp4Bfp32_gemm;
  unsigned int is_Ai4Bi8_gemm;
  unsigned int is_Abf8Bbf16_gemm;
  unsigned int is_Abf8Bf16_gemm;
  unsigned int is_Ahf8Bbf16_gemm;
  unsigned int fuse_zpt_sub;
} libxsmm_gemm_def;

typedef struct libxsmm_fusion_args {
  char *colbias;
  char *relu_bitmask;
} libxsmm_fusion_args;

LIBXSMM_API_INTERN
void libxsmm_ref_matmul( const libxsmm_gemm_def* i_gemm_def, const void* a, const void* b, void* c ) {
  libxsmm_blasint l_r, l_j, l_i, l_s, l_k2;
  libxsmm_blasint lda = i_gemm_def->lda;
  libxsmm_blasint ldb = i_gemm_def->ldb;
  libxsmm_blasint ldc = i_gemm_def->ldc;
  libxsmm_blasint m = i_gemm_def->m;
  libxsmm_blasint n = i_gemm_def->n;
  libxsmm_blasint k = i_gemm_def->k;

  if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_F64) &&
       (i_gemm_def->b_type    == LIBXSMM_DATATYPE_F64) &&
       (i_gemm_def->c_type    == LIBXSMM_DATATYPE_F64) &&
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
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_F32) &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_F32) &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_F32) &&
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
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_I16) &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_I16) &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_I32) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_I32)    ) {
    short* s_a = (short*)a;
    short* s_b = (short*)b;
    int*   i_c = (int*)c;
    int l_k_block = libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type);

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
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_I8)  &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_I8)  &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_I32) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_I32) &&
              (i_gemm_def->unsigned_a == 1) && (i_gemm_def->unsigned_b == 1) ) {
    unsigned char* c_a = (unsigned char*)a;
    unsigned char* c_b = (unsigned char*)b;
    int*           i_c = (int*)c;
    int l_k_block = libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type);

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
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_I8)  &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_I8)  &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_I32) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_I32) &&
              (i_gemm_def->unsigned_a == 0) && (i_gemm_def->unsigned_b == 0) ) {
    char* c_a = (char*)a;
    char* c_b = (char*)b;
    int*           i_c = (int*)c;
    int l_k_block = libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type);

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
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_I8)  &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_I8)  &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_I32) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_I32) &&
              (i_gemm_def->unsigned_a == 1) && (i_gemm_def->unsigned_b == 0) ) {
    unsigned char* c_a = (unsigned char*)a;
    char*          c_b = (char*)b;
    int*           i_c = (int*)c;
    int l_k_block = libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type);

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
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_I8)  &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_I8)  &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_I32) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_I32) &&
              (i_gemm_def->unsigned_a == 0) && (i_gemm_def->unsigned_b == 1) ) {
    char*          c_a = (char*)a;
    unsigned char* c_b = (unsigned char*)b;
    int*           i_c = (int*)c;
    int l_k_block = libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type);

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
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_I8)  &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_I8)  &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_F32) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_I32) &&
              (i_gemm_def->unsigned_a == 1) && (i_gemm_def->unsigned_b == 1) ) {
    unsigned char* c_a = (unsigned char*)a;
    unsigned char* c_b = (unsigned char*)b;
    float*         c_c = (float*)c;
    int l_k_block = libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type);

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
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_I8)  &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_I8)  &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_F32) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_I32) &&
              (i_gemm_def->unsigned_a == 0) && (i_gemm_def->unsigned_b == 0) ) {
    char*          c_a = (char*)a;
    char*          c_b = (char*)b;
    float*         c_c = (float*)c;
    int l_k_block = libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type);

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
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_I8)  &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_I8)  &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_F32) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_I32) &&
              (i_gemm_def->unsigned_a == 1) && (i_gemm_def->unsigned_b == 0) ) {
    unsigned char* c_a = (unsigned char*)a;
    char* c_b          = (char*)b;
    float*         c_c = (float*)c;
    int l_k_block = libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type);

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
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_I8)  &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_I8)  &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_F32) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_I32) &&
              (i_gemm_def->unsigned_a == 0) && (i_gemm_def->unsigned_b == 1) ) {
    char*          c_a = (char*)a;
    unsigned char* c_b = (unsigned char*)b;
    float*         c_c = (float*)c;
    int l_k_block = libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type);

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
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_I8)  &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_BF16) &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_BF16 || i_gemm_def->c_type    == LIBXSMM_DATATYPE_F32) ) {
    char* c_a = (char*)a;
    libxsmm_bfloat16* bf16_c = (libxsmm_bfloat16*)c;
    float* f32_c = (float*)c;
    libxsmm_bfloat16* bf16_b = (libxsmm_bfloat16*)b;
    int l_k_block = 1;
    libxsmm_bfloat16 tmp_bf16;
    float up_c;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        float ftmp = 0.0;
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              float a_use, b_use;
              char char_a = c_a[(l_r * lda * k) + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              int int_a = (int) char_a;
              a_use = (float) int_a;
              a_use = a_use * i_gemm_def->scf_f32[l_i];
              libxsmm_rne_convert_fp32_bf16(&a_use, &tmp_bf16, 1);
              libxsmm_convert_bf16_f32( &tmp_bf16, &a_use, 1 );
              tmp_bf16 = bf16_b[(l_r * ldb * n) + (l_j * ldb) + (l_s*l_k_block) + l_k2];
              libxsmm_convert_bf16_f32( &tmp_bf16, &b_use, 1 );
              ftmp += a_use * b_use;
            }
          }
        }
        if ( i_gemm_def->c_type    == LIBXSMM_DATATYPE_BF16 ) {
          if ( i_gemm_def->beta == 1 ) {
            tmp_bf16 = bf16_c[(l_j * ldc) + l_i];
            libxsmm_convert_bf16_f32( &tmp_bf16, &up_c, 1 );
            ftmp += up_c;
          }
          libxsmm_rne_convert_fp32_bf16(&ftmp, &tmp_bf16, 1);
          bf16_c[(l_j * ldc) + l_i] = tmp_bf16;
        } else {
          if ( i_gemm_def->beta == 1 ) {
            ftmp += f32_c[(l_j * ldc) + l_i];
          }
          f32_c[(l_j * ldc) + l_i] = ftmp;
        }
      }
    }
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_BF8)  &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_F16) &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_F16 || i_gemm_def->c_type    == LIBXSMM_DATATYPE_F32) &&
              ((i_gemm_def->comp_type == LIBXSMM_DATATYPE_F16) || (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F32) || (i_gemm_def->comp_type == LIBXSMM_DATATYPE_IMPLICIT))  ) {
    libxsmm_bfloat8* bf8_a = (libxsmm_bfloat8*)a;
    libxsmm_float16* f16_c = (libxsmm_float16*)c;
    float* f32_c = (float*)c;
    libxsmm_float16* f16_b = (libxsmm_float16*)b;
    libxsmm_float16 c_tmp;
    libxsmm_float16 cur_b;
    float up_c;
    int l_k_block = 1;
    const char* env_arch = getenv("LIBXSMM_TARGET");
    const int is_env_SPR = (
      env_arch == libxsmm_stristr(env_arch, "spr") ||
      env_arch == libxsmm_stristr(env_arch, "amx"));
    int arch_cpuid = libxsmm_cpuid(NULL);
    int l_is_gt_spr = (is_env_SPR > 0) ? 1 : ((env_arch == NULL && arch_cpuid >= LIBXSMM_X86_AVX512_SPR) ? 1 : 0);
    int l_use_replacement_fma = (((i_gemm_def->comp_type == LIBXSMM_DATATYPE_F16) || (i_gemm_def->comp_type == LIBXSMM_DATATYPE_IMPLICIT && l_is_gt_spr > 0))) ? 1 : 0;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        float ftmp = 0.0, b_use;
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              union libxsmm_bfloat8_f16 tmp_a_hf;
              float tmp_a_f;
              tmp_a_hf.i[0] = 0;
              tmp_a_hf.i[1] = bf8_a[(l_r * lda * k) + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              tmp_a_f = libxsmm_convert_f16_to_f32( tmp_a_hf.hf );
              if (i_gemm_def->trans_b == 0) {
                cur_b = f16_b[(l_r * ldb * n) + (l_j * ldb) + (l_s*l_k_block) + l_k2];
              } else {
                cur_b = f16_b[(l_r * ldb * k) + (l_s*l_k_block+l_k2) * ldb + l_j];
              }
              libxsmm_convert_f16_f32( &cur_b, &b_use, 1 );
              ftmp += tmp_a_f * b_use;
              if (l_use_replacement_fma > 0) {
                libxsmm_rne_convert_fp32_f16(&ftmp, &c_tmp, 1);
                libxsmm_convert_f16_f32( &c_tmp, &ftmp, 1 );
              }
            }
          }
        }
        if ( i_gemm_def->c_type    == LIBXSMM_DATATYPE_F16 ) {
          if ( i_gemm_def->beta == 1 ) {
            c_tmp = f16_c[(l_j * ldc) + l_i];
            libxsmm_convert_f16_f32( &c_tmp, &up_c, 1 );
            ftmp += up_c;
          }
          libxsmm_rne_convert_fp32_f16(&ftmp, &c_tmp, 1);
          f16_c[(l_j * ldc) + l_i] = c_tmp;
        } else {
          if ( i_gemm_def->beta == 1 ) {
            ftmp += f32_c[(l_j * ldc) + l_i];
          }
          f32_c[(l_j * ldc) + l_i] = ftmp;
        }
      }
    }
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_I8)  &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_F16) &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_F16) &&
              ((i_gemm_def->comp_type == LIBXSMM_DATATYPE_F16) || (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F32) || (i_gemm_def->comp_type == LIBXSMM_DATATYPE_IMPLICIT))  ) {
    char* c_a = (char*)a;
    libxsmm_float16* f16_c = (libxsmm_float16*)c;
    libxsmm_float16* f16_b = (libxsmm_float16*)b;
    libxsmm_float16 c_tmp;
    libxsmm_float16 cur_a, cur_b;
    float up_c;
    int l_k_block = 1;
    const char* env_arch = getenv("LIBXSMM_TARGET");
    const int is_env_SPR = (
      env_arch == libxsmm_stristr(env_arch, "spr") ||
      env_arch == libxsmm_stristr(env_arch, "amx"));
    int arch_cpuid = libxsmm_cpuid(NULL);
    int l_is_gt_spr = (is_env_SPR > 0) ? 1 : ((env_arch == NULL && arch_cpuid >= LIBXSMM_X86_AVX512_SPR) ? 1 : 0);
    int l_use_replacement_fma = (((i_gemm_def->comp_type == LIBXSMM_DATATYPE_F16) || (i_gemm_def->comp_type == LIBXSMM_DATATYPE_IMPLICIT && l_is_gt_spr > 0))) ? 1 : 0;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        float ftmp = 0.0, a_use, b_use;
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              char char_a = c_a[(l_r * lda * k) + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              short short_a = (short) char_a;
              int int_a = (int) char_a;
              if (l_use_replacement_fma > 0) {
                a_use = (float) short_a;
                libxsmm_rne_convert_fp32_f16(&a_use, &cur_a, 1);
                libxsmm_convert_f16_f32( &cur_a, &a_use, 1 );
              } else {
                a_use = (float) int_a;
              }
              if (i_gemm_def->fuse_zpt_sub > 0) {
                a_use = a_use - i_gemm_def->zpt_f32[l_i];
                if (l_use_replacement_fma > 0) {
                  libxsmm_rne_convert_fp32_f16(&a_use, &cur_a, 1);
                  libxsmm_convert_f16_f32( &cur_a, &a_use, 1 );
                }
              }
              a_use = a_use * i_gemm_def->scf_f32[l_i];
              if (l_use_replacement_fma > 0) {
                libxsmm_rne_convert_fp32_f16(&a_use, &c_tmp, 1);
                libxsmm_convert_f16_f32( &c_tmp, &a_use, 1 );
              }

              if (i_gemm_def->trans_b == 0) {
                cur_b = f16_b[(l_r * ldb * n) + (l_j * ldb) + (l_s*l_k_block) + l_k2];
              } else {
                cur_b = f16_b[(l_r * ldb * k) + (l_s*l_k_block+l_k2) * ldb + l_j];
              }
              libxsmm_convert_f16_f32( &cur_b, &b_use, 1 );
              ftmp += a_use * b_use;
              if (l_use_replacement_fma > 0) {
                libxsmm_rne_convert_fp32_f16(&ftmp, &c_tmp, 1);
                libxsmm_convert_f16_f32( &c_tmp, &ftmp, 1 );
              }
            }
          }
        }
        if ( i_gemm_def->beta == 1 ) {
          c_tmp = f16_c[(l_j * ldc) + l_i];
          libxsmm_convert_f16_f32( &c_tmp, &up_c, 1 );
          ftmp += up_c;
        }
        libxsmm_rne_convert_fp32_f16(&ftmp, &c_tmp, 1);
        f16_c[(l_j * ldc) + l_i] = c_tmp;
      }
    }
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_I8)  &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_F16) &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_F32) &&
              ((i_gemm_def->comp_type == LIBXSMM_DATATYPE_F16) || (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F32) || (i_gemm_def->comp_type == LIBXSMM_DATATYPE_IMPLICIT))  ) {
    char* c_a = (char*)a;
    float* f32_c = (float*)c;
    libxsmm_float16* f16_b = (libxsmm_float16*)b;
    libxsmm_float16 c_tmp;
    float c_tmp_f32;
    libxsmm_float16 cur_a, cur_b;
    int l_k_block = 1;
    const char* env_arch = getenv("LIBXSMM_TARGET");
    const int is_env_SPR = (
      env_arch == libxsmm_stristr(env_arch, "spr") ||
      env_arch == libxsmm_stristr(env_arch, "amx"));
    int arch_cpuid = libxsmm_cpuid(NULL);
    int l_is_gt_spr = (is_env_SPR > 0) ? 1 : ((env_arch == NULL && arch_cpuid >= LIBXSMM_X86_AVX512_SPR) ? 1 : 0);
    int l_use_replacement_fma = (((i_gemm_def->comp_type == LIBXSMM_DATATYPE_F16) || (i_gemm_def->comp_type == LIBXSMM_DATATYPE_IMPLICIT && l_is_gt_spr > 0))) ? 1 : 0;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        float ftmp = 0.0, a_use, b_use;
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              char char_a = c_a[(l_r * lda * k) + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              short short_a = (short) char_a;
              int int_a = (int) char_a;
              if (l_use_replacement_fma > 0) {
                a_use = (float) short_a;
                libxsmm_rne_convert_fp32_f16(&a_use, &cur_a, 1);
                libxsmm_convert_f16_f32( &cur_a, &a_use, 1 );
              } else {
                a_use = (float) int_a;
              }
              if (i_gemm_def->fuse_zpt_sub > 0) {
                a_use = a_use - i_gemm_def->zpt_f32[l_i];
                if (l_use_replacement_fma > 0) {
                  libxsmm_rne_convert_fp32_f16(&a_use, &cur_a, 1);
                  libxsmm_convert_f16_f32( &cur_a, &a_use, 1 );
                }
              }
              a_use = a_use * i_gemm_def->scf_f32[l_i];
              if (l_use_replacement_fma > 0) {
                libxsmm_rne_convert_fp32_f16(&a_use, &c_tmp, 1);
                libxsmm_convert_f16_f32( &c_tmp, &a_use, 1 );
              }
              if (i_gemm_def->trans_b == 0) {
                cur_b = f16_b[(l_r * ldb * n) + (l_j * ldb) + (l_s*l_k_block) + l_k2];
              } else {
                cur_b = f16_b[(l_r * ldb * k) + (l_s*l_k_block+l_k2) * ldb + l_j];
              }
              libxsmm_convert_f16_f32( &cur_b, &b_use, 1 );
              ftmp += a_use * b_use;
              if (l_use_replacement_fma > 0) {
                libxsmm_rne_convert_fp32_f16(&ftmp, &c_tmp, 1);
                libxsmm_convert_f16_f32( &c_tmp, &ftmp, 1 );
              }
            }
          }
        }
        if ( i_gemm_def->beta == 1 ) {
          c_tmp_f32 = f32_c[(l_j * ldc) + l_i];
          libxsmm_rne_convert_fp32_f16(&c_tmp_f32, &c_tmp, 1);
          libxsmm_convert_f16_f32( &c_tmp, &c_tmp_f32, 1 );
          ftmp += c_tmp_f32;
        }
        f32_c[(l_j * ldc) + l_i] = ftmp;
      }
    }
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_F16)  &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_F16)  &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_F16) &&
              ((i_gemm_def->comp_type == LIBXSMM_DATATYPE_F16) || (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F32) || (i_gemm_def->comp_type == LIBXSMM_DATATYPE_IMPLICIT))  ) {
    libxsmm_float16* f16_a = (libxsmm_float16*)a;
    libxsmm_float16* f16_c = (libxsmm_float16*)c;
    libxsmm_float16* f16_b = (libxsmm_float16*)b;
    libxsmm_float16 c_tmp;
    libxsmm_float16 cur_a, cur_b;
    float up_c;
    int l_k_block = ( i_gemm_def->vnni_a != 0) ? libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type) : 1;
    const char* env_arch = getenv("LIBXSMM_TARGET");
    const int is_env_SPR = (
      env_arch == libxsmm_stristr(env_arch, "spr") ||
      env_arch == libxsmm_stristr(env_arch, "amx"));
    int arch_cpuid = libxsmm_cpuid(NULL);
    int l_is_gt_spr = (is_env_SPR > 0) ? 1 : ((env_arch == NULL && arch_cpuid >= LIBXSMM_X86_AVX512_SPR) ? 1 : 0);
    int l_use_replacement_fma = (((i_gemm_def->comp_type == LIBXSMM_DATATYPE_F16) || (i_gemm_def->comp_type == LIBXSMM_DATATYPE_IMPLICIT && l_is_gt_spr > 0))) ? 1 : 0;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        float ftmp = 0.0, a_use, b_use;
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              cur_a = f16_a[(l_r * lda * k) + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              libxsmm_convert_f16_f32( &cur_a, &a_use, 1 );
              if (i_gemm_def->trans_b == 0) {
                cur_b = f16_b[(l_r * ldb * n) + (l_j * ldb) + (l_s*l_k_block) + l_k2];
              } else {
                cur_b = f16_b[(l_r * ldb * k) + (l_s*l_k_block+l_k2) * ldb + l_j];
              }
              libxsmm_convert_f16_f32( &cur_b, &b_use, 1 );
              ftmp += a_use * b_use;
              if (l_use_replacement_fma > 0) {
                libxsmm_rne_convert_fp32_f16(&ftmp, &c_tmp, 1);
                libxsmm_convert_f16_f32( &c_tmp, &ftmp, 1 );
              }
            }
          }
        }
        if ( i_gemm_def->beta == 1 ) {
          c_tmp = f16_c[(l_j * ldc) + l_i];
          libxsmm_convert_f16_f32( &c_tmp, &up_c, 1 );
          ftmp += up_c;
        }
        libxsmm_rne_convert_fp32_f16(&ftmp, &c_tmp, 1);
        f16_c[(l_j * ldc) + l_i] = c_tmp;
      }
    }
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_F16)  &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_F16)  &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_F32) &&
              ((i_gemm_def->comp_type == LIBXSMM_DATATYPE_F16) || (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F32) || (i_gemm_def->comp_type == LIBXSMM_DATATYPE_IMPLICIT))  ) {
    libxsmm_float16* f16_a = (libxsmm_float16*)a;
    float* f32_c = (float*)c;
    libxsmm_float16* f16_b = (libxsmm_float16*)b;
    libxsmm_float16 c_tmp;
    float c_tmp_f32;
    libxsmm_float16 cur_a, cur_b;
    int l_k_block = ( i_gemm_def->vnni_a != 0) ? libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type) : 1;
    const char* env_arch = getenv("LIBXSMM_TARGET");
    const int is_env_SPR = (
      env_arch == libxsmm_stristr(env_arch, "spr") ||
      env_arch == libxsmm_stristr(env_arch, "amx"));
    int arch_cpuid = libxsmm_cpuid(NULL);
    int l_is_gt_spr = (is_env_SPR > 0) ? 1 : ((env_arch == NULL && arch_cpuid >= LIBXSMM_X86_AVX512_SPR) ? 1 : 0);
    int l_use_replacement_fma = (((i_gemm_def->comp_type == LIBXSMM_DATATYPE_F16) || (i_gemm_def->comp_type == LIBXSMM_DATATYPE_IMPLICIT && l_is_gt_spr > 0))) ? 1 : 0;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        float ftmp = 0.0, a_use, b_use;
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              cur_a = f16_a[(l_r * lda * k) + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              libxsmm_convert_f16_f32( &cur_a, &a_use, 1 );
              if (i_gemm_def->trans_b == 0) {
                cur_b = f16_b[(l_r * ldb * n) + (l_j * ldb) + (l_s*l_k_block) + l_k2];
              } else {
                cur_b = f16_b[(l_r * ldb * k) + (l_s*l_k_block+l_k2) * ldb + l_j];
              }
              libxsmm_convert_f16_f32( &cur_b, &b_use, 1 );
              ftmp += a_use * b_use;
              if (l_use_replacement_fma > 0) {
                libxsmm_rne_convert_fp32_f16(&ftmp, &c_tmp, 1);
                libxsmm_convert_f16_f32( &c_tmp, &ftmp, 1 );
              }
            }
          }
        }
        if ( i_gemm_def->beta == 1 ) {
          c_tmp_f32 = f32_c[(l_j * ldc) + l_i];
          libxsmm_rne_convert_fp32_f16(&c_tmp_f32, &c_tmp, 1);
          libxsmm_convert_f16_f32( &c_tmp, &c_tmp_f32, 1 );
          ftmp += c_tmp_f32;
        }
        f32_c[(l_j * ldc) + l_i] = ftmp;
      }
    }
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_BF16) &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_BF16) &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_F32)  &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F32)     ) {
    libxsmm_bfloat16* h_a = (libxsmm_bfloat16*)a;
    libxsmm_bfloat16* h_b = (libxsmm_bfloat16*)b;
    float*            f_c = (float*)c;
    int l_k_block = ( i_gemm_def->vnni_a != 0) ? libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type) : 1;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        if ( i_gemm_def->beta == 0 ) {
          f_c[(l_j * ldc) + l_i] = 0.0f;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = l_k_block - 1; l_k2 >= 0; l_k2--) {
              union libxsmm_bfloat16_f32 tmp_a_f;
              union libxsmm_bfloat16_f32 tmp_b_f;
              tmp_a_f.i[0] = 0;
              if ( (i_gemm_def->trans_a == 0) ) {
                tmp_a_f.i[1] = h_a[(l_r * lda * k) + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              } else if ( (i_gemm_def->trans_a > 0) && ( i_gemm_def->vnni_a == 0) ) {
                tmp_a_f.i[1] = h_a[(l_r * lda * m) + (l_i * lda) + (l_s*l_k_block) + l_k2];
              } else {
                /* should happen */
              }
              tmp_b_f.i[0] = 0;
              if ((i_gemm_def->trans_b > 0) && (i_gemm_def->vnni_b > 0)) {
                tmp_b_f.i[1] = h_b[(l_r * ldb * k) + (l_j * l_k_block) + (l_s * (ldb*l_k_block)) + l_k2];
              } else if ( (i_gemm_def->trans_b > 0) && (i_gemm_def->vnni_b == 0) ) {
                tmp_b_f.i[1] = h_b[(l_r * ldb * k) + (((l_s*l_k_block) + l_k2) * ldb) + l_j];
              } else if ( (i_gemm_def->trans_b == 0) && (i_gemm_def->vnni_b == 0 ) ) {
                tmp_b_f.i[1] = h_b[(l_r * ldb * n) + (l_j * ldb) + (l_s*l_k_block) + l_k2];
              } else {
                /* should not happen */
              }
              f_c[(l_j * ldc) + l_i] += tmp_a_f.f * tmp_b_f.f;
            }
          }
        }
      }
    }
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_BF16) &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_BF16) &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_BF16) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F32)     ) {
    libxsmm_bfloat16* h_a = (libxsmm_bfloat16*)a;
    libxsmm_bfloat16* h_b = (libxsmm_bfloat16*)b;
    libxsmm_bfloat16* h_c = (libxsmm_bfloat16*)c;
    int l_k_block = ( i_gemm_def->vnni_a != 0) ? libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type) : 1;
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
              union libxsmm_bfloat16_f32 tmp_a_f;
              union libxsmm_bfloat16_f32 tmp_b_f;
              tmp_a_f.i[0] = 0;
              if ( (i_gemm_def->trans_a == 0) ) {
                tmp_a_f.i[1] = h_a[(l_r * lda * k) + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              } else if ( (i_gemm_def->trans_a > 0) && ( i_gemm_def->vnni_a == 0) ) {
                tmp_a_f.i[1] = h_a[(l_r * lda * m) + (l_i * lda) + (l_s*l_k_block) + l_k2];
              } else {
                /* should happen */
              }
              tmp_b_f.i[0] = 0;
              if ((i_gemm_def->trans_b > 0) && (i_gemm_def->vnni_b > 0)) {
                tmp_b_f.i[1] = h_b[(l_r * ldb * k) + (l_j * l_k_block) + (l_s * (ldb*l_k_block)) + l_k2];
              } else if ( (i_gemm_def->trans_b > 0) && (i_gemm_def->vnni_b == 0) ) {
                tmp_b_f.i[1] = h_b[(l_r * ldb * k) + (((l_s*l_k_block) + l_k2) * ldb) + l_j];
              } else if ( (i_gemm_def->trans_b == 0) && (i_gemm_def->vnni_b == 0 ) ) {
                tmp_b_f.i[1] = h_b[(l_r * ldb * n) + (l_j * ldb) + (l_s*l_k_block) + l_k2];
              } else {
                /* should not happen */
              }
              acc += tmp_a_f.f * tmp_b_f.f;
            }
          }
        }
        libxsmm_rne_convert_fp32_bf16( &acc, &h_acc, 1 );
        h_c[(l_j * ldc) + l_i] = h_acc;
      }
    }
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_BF8) &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_BF8) &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_F32) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F32)    ) {
    libxsmm_bfloat8* h_a = (libxsmm_bfloat8*)a;
    libxsmm_bfloat8* h_b = (libxsmm_bfloat8*)b;
    float*           f_c = (float*)c;
    int l_k_block = ( i_gemm_def->vnni_a != 0) ? libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type) : 1;
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
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_HF8) &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_HF8) &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_F32) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F32)    ) {
    libxsmm_hfloat8* h_a = (libxsmm_hfloat8*)a;
    libxsmm_hfloat8* h_b = (libxsmm_hfloat8*)b;
    float*           f_c = (float*)c;
    int l_k_block = ( i_gemm_def->vnni_a != 0) ? libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type) : 1;
    for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
      for (l_j = 0; l_j < n; l_j++) {
        for (l_i = 0; l_i < m; l_i++) {
          if ( (i_gemm_def->beta == 0) && (l_r == 0) ) {
            f_c[(l_j * ldc) + l_i] = 0.0f;
          }
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              float tmp_a_f, tmp_b_f;
              libxsmm_convert_hf8_f32(&h_a[(l_r * lda * k) + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2], &tmp_a_f, 1);
              libxsmm_convert_hf8_f32(&h_b[(l_r * ldb * n) + (l_j * ldb) + (l_s*l_k_block) + l_k2], &tmp_b_f, 1);
              f_c[(l_j * ldc) + l_i] += tmp_a_f * tmp_b_f;
            }
          }
        }
      }
    }
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_BF8) &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_BF8) &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_BF8) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F32)    ) {
    libxsmm_bfloat8* h_a = (libxsmm_bfloat8*)a;
    libxsmm_bfloat8* h_b = (libxsmm_bfloat8*)b;
    libxsmm_bfloat8* h_c = (libxsmm_bfloat8*)c;
    int l_k_block = ( i_gemm_def->vnni_a != 0) ? libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type) : 1;
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
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_HF8) &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_HF8) &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_HF8) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F32)    ) {
    libxsmm_hfloat8* h_a = (libxsmm_hfloat8*)a;
    libxsmm_hfloat8* h_b = (libxsmm_hfloat8*)b;
    libxsmm_hfloat8* h_c = (libxsmm_hfloat8*)c;
    int l_k_block = ( i_gemm_def->vnni_a != 0) ? libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type) : 1;
    float acc = 0.0f;
    libxsmm_hfloat8 hf8_acc;
    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        if ( i_gemm_def->beta == 0 ) {
          acc = 0.0f;
        } else {
          float tmp_c_f;
          libxsmm_convert_hf8_f32(&h_c[(l_j * ldc) + l_i], &tmp_c_f, 1);
          acc = tmp_c_f;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              float tmp_a_f;
              float tmp_b_f;
              libxsmm_convert_hf8_f32(&h_a[(l_r * lda * k) + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2], &tmp_a_f, 1);
              libxsmm_convert_hf8_f32(&h_b[(l_r * ldb * n) + (l_j * ldb) + (l_s*l_k_block) + l_k2], &tmp_b_f, 1);
              acc += tmp_a_f * tmp_b_f;
            }
          }
        }
        libxsmm_rne_convert_fp32_hf8( &acc, &hf8_acc, 1 );
        h_c[(l_j * ldc) + l_i] = hf8_acc;
      }
    }
  }
}

#endif

LIBXSMM_API_INTERN
void libxsmm_reference_gemm(void *param, const libxsmm_gemm_descriptor *i_xgemm_desc) {
  printf("Hello world\n");
}

