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
#include <libxsmm_math.h>
#include <utils/libxsmm_lpflt_quant.h>

LIBXSMM_API_INTERN
void unpack2bit(char* m0, char* m1, char* m2, char* m3, char packed) {
  if ((packed & 0x3) == 0) {
    *m0 = 0;
  } else if ((packed & 0x3) == 1) {
    *m0 = 1;
  } else if ((packed & 0x3) == 2) {
    *m0 = -1;
  } else if ((packed & 0x3) == 3) {
    *m0 = -1;
  }
  if (((packed >> 2) & 0x3) == 0) {
    *m1 = 0;
  } else if (((packed >> 2) & 0x3) == 1) {
    *m1 = 1;
  } else if (((packed >> 2) & 0x3) == 2) {
    *m1 = -1;
  } else if (((packed >> 2) & 0x3) == 3) {
    *m1 = -1;
  }
  if (((packed >> 4) & 0x3) == 0) {
    *m2 = 0;
  } else if (((packed >> 4) & 0x3) == 1) {
    *m2 = 1;
  } else if (((packed >> 4) & 0x3) == 2) {
    *m2 = -1;
  } else if (((packed >> 4) & 0x3) == 3) {
    *m2 = -1;
  }
  if (((packed >> 6) & 0x3) == 0) {
    *m3 = 0;
  } else if (((packed >> 6) & 0x3) == 1) {
    *m3 = 1;
  } else if (((packed >> 6) & 0x3) == 2) {
    *m3 = -1;
  } else if (((packed >> 6) & 0x3) == 3) {
    *m3 = -1;
  }
  return;
}

LIBXSMM_API_INTERN
float libxsmm_convert_mxfp4_to_float(unsigned char x) {
  float fp4_e2m1_lut[16] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};
  float result = fp4_e2m1_lut[x];
  return result;
}

LIBXSMM_API_INTERN
char libxsmm_convert_mxfp4_to_char(unsigned char x) {
  char fp4_e2m1_lut[16] = {0, 11, 21, 32, 42, 64, 85, 127, 0, (char)-11, (char)-21, (char)-32, (char)-42, (char)-64, (char)-85, (char)-127};
  char result = fp4_e2m1_lut[x];
  return result;
}

LIBXSMM_API_INTERN
unsigned char libxsmm_convert_fp6_e2m3_to_hf8( unsigned char in ) {
  unsigned char lut_f6_e2m3_to_hf8[64] =
  { 0x00, 0x20, 0x28, 0x2c, 0x30, 0x32, 0x34, 0x36, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f,
    0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f,
    0x80, 0xa0, 0xa8, 0xac, 0xb0, 0xb2, 0xb4, 0xb6, 0xb8, 0xb9, 0xba, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf,
    0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xcb, 0xcc, 0xcd, 0xce, 0xcf };
  return lut_f6_e2m3_to_hf8[ (unsigned char)(in & 0x3f) ];
}

LIBXSMM_API_INTERN
unsigned char libxsmm_convert_fp6_e3m2_to_hf8( unsigned char in ) {
  unsigned char lut_f6_e3m2_to_hf8[64] =
  { 0x00, 0x18, 0x20, 0x24, 0x28, 0x2a, 0x2c, 0x2e, 0x30, 0x32, 0x34, 0x36, 0x38, 0x3a, 0x3c, 0x3e,
    0x40, 0x42, 0x44, 0x46, 0x48, 0x4a, 0x4c, 0x4e, 0x50, 0x52, 0x54, 0x56, 0x58, 0x5a, 0x5c, 0x5e,
    0x80, 0x98, 0xa0, 0xa4, 0xa8, 0xaa, 0xac, 0xae, 0xb0, 0xb2, 0xb4, 0xb6, 0xb8, 0xba, 0xbc, 0xbe,
    0xc0, 0xc2, 0xc4, 0xc6, 0xc8, 0xca, 0xcc, 0xce, 0xd0, 0xd2, 0xd4, 0xd6, 0xd8, 0xda, 0xdc, 0xde };
  return lut_f6_e3m2_to_hf8[ (unsigned char)(in & 0x3f) ];
}

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
  float **scf_b_f32_braddr;
  float *scf_b_f32;
  float *zpt_f32;
  libxsmm_float16 *zpt_f16;
  unsigned char **zpt_u8_braddr;
  unsigned char *zpt_u8;
  unsigned char **scf_u8_braddr;
  unsigned char *scf_u8;
  unsigned char *scf_b_u8;
  unsigned char *scf_c_u8;
  libxsmm_float16 *scf_f16;
  int binary_postop;
  int unary_postop;
  unsigned int mxfp4_block_size;
  unsigned int is_Ai4Bf16_gemm;
  unsigned int is_Amxfp4Bbf16_gemm;
  unsigned int is_Amxfp4Bfp32_gemm;
  unsigned int is_Amxfp4Bi8_gemm;
  unsigned int is_Ai4Bi8_gemm;
  unsigned int is_Ai2Bi8_gemm;
  unsigned int is_Ai1Bi8_gemm;
  unsigned int is_Abf8Bbf16_gemm;
  unsigned int is_Abf8Bf16_gemm;
  unsigned int is_Ahf8Bbf16_gemm;
  unsigned int is_spmm;
  unsigned int fuse_zpt_sub;

  /* BRGEMM related fields */
  long long stride_a;
  long long stride_b;
  long long *br_offs_A;
  long long *br_offs_B;
  void **br_addr_A;
  void **br_addr_B;
  unsigned char *decompress_bitmap;

  /* fusion aux variables */
  unsigned int fuse_colbias_add;
  unsigned int fuse_relu;
  unsigned int fuse_relu_bitmask;
  unsigned int fuse_sigmoid;
  unsigned int fuse_vnni_c;
  unsigned int fuse_via_scratch;

  /* c_scratch for fusion */
  void *c_scratch;
  void *c_vnni_scratch;
} libxsmm_gemm_def;

typedef struct libxsmm_fusion_args {
  char *colbias;
  char *relu_bitmask;
} libxsmm_fusion_args;

LIBXSMM_API_INTERN
void libxsmm_calculate_brgemm_offsets(void **a_addr, void **b_addr, long long *offs_a, long long *offs_b, long long l_r, const libxsmm_gemm_def* i_gemm_def) {
  if (i_gemm_def->br_type == 1) {
    *a_addr = i_gemm_def->br_addr_A[l_r];
    *b_addr = i_gemm_def->br_addr_B[l_r];
    *offs_a = 0;
    *offs_b = 0;
  } else if (i_gemm_def->br_type == 2) {
    *offs_a = (i_gemm_def->br_offs_A[l_r])/LIBXSMM_TYPESIZE(i_gemm_def->a_type);
    *offs_b = (i_gemm_def->br_offs_B[l_r])/LIBXSMM_TYPESIZE(i_gemm_def->b_type);
  } else if (i_gemm_def->br_type == 3) {
    *offs_a = i_gemm_def->stride_a * l_r;
    *offs_b = i_gemm_def->stride_b * l_r;
  } else {
    *offs_a = 0;
    *offs_b = 0;
  }
  return;
}

LIBXSMM_API_INTERN
float libxsmm_calculate_mxfp4_scf(libxsmm_blasint i_br, libxsmm_blasint i_k, libxsmm_blasint i_m, const libxsmm_gemm_def* i_gemm_def) {
  float res_f32 = 0;
  unsigned char scale_mxfp4_uchar = 0;
  unsigned int mxfp4_block_size = i_gemm_def->mxfp4_block_size;
  unsigned int scale_mxfp4_u32 = 0;
  libxsmm_float_uint fuint;

  if (i_gemm_def->br_type == 1) {
    scale_mxfp4_uchar = *((unsigned char*)i_gemm_def->scf_u8_braddr[i_br] + i_k * i_gemm_def->lda + i_m);
  } else if (i_gemm_def->br_type == 2) {
    scale_mxfp4_uchar = *((unsigned char*)i_gemm_def->scf_u8 + ((i_gemm_def->br_offs_A[i_br]*2)/mxfp4_block_size) + i_k * i_gemm_def->lda + i_m);
  } else if (i_gemm_def->br_type == 3) {
    scale_mxfp4_uchar = *((unsigned char*)i_gemm_def->scf_u8 + ((i_gemm_def->stride_a*2)/mxfp4_block_size) * i_br + i_k * i_gemm_def->lda + i_m);
  } else {
    scale_mxfp4_uchar = *((unsigned char*)i_gemm_def->scf_u8 + i_k * i_gemm_def->lda + i_m);
  }
  scale_mxfp4_u32 = (unsigned int) scale_mxfp4_uchar;
  scale_mxfp4_u32 = scale_mxfp4_u32 << 23;
  fuint.u = scale_mxfp4_u32;
  res_f32 = fuint.f;
  return res_f32;
}

LIBXSMM_API_INTERN
float libxsmm_calculate_mxfp4_scf_b(libxsmm_blasint i_br, libxsmm_blasint i_k, libxsmm_blasint i_n, const libxsmm_gemm_def* i_gemm_def) {
  float res_f32 = 0;
  unsigned int mxfp4_block_size = i_gemm_def->mxfp4_block_size;
  if (i_gemm_def->br_type == 1) {
    res_f32 = *((float*)i_gemm_def->scf_b_f32_braddr[i_br] + i_n * (i_gemm_def->ldb/mxfp4_block_size) + i_k);
  } else if (i_gemm_def->br_type == 2) {
    res_f32 = *((float*)i_gemm_def->scf_b_f32 + (i_gemm_def->br_offs_B[i_br]/mxfp4_block_size) + i_n * (i_gemm_def->ldb/mxfp4_block_size) + i_k);
  } else if (i_gemm_def->br_type == 3) {
    res_f32 = *((float*)i_gemm_def->scf_b_f32 + (i_gemm_def->stride_b/mxfp4_block_size) * i_br + i_n * (i_gemm_def->ldb/mxfp4_block_size) + i_k);
  } else {
    res_f32 = *((float*)i_gemm_def->scf_b_f32 + i_n * (i_gemm_def->ldb/mxfp4_block_size) + i_k);
  }
  return res_f32;
}

LIBXSMM_API_INTERN
unsigned char libxsmm_calculate_zpt(libxsmm_blasint i_br, libxsmm_blasint i_m, const libxsmm_gemm_def* i_gemm_def) {
  unsigned char res = 0;
  if (i_gemm_def->br_type == 1) {
    res = *((unsigned char*)i_gemm_def->zpt_u8_braddr[i_br] + i_m);
  } else if (i_gemm_def->br_type == 2) {
    res = *((unsigned char*)i_gemm_def->zpt_u8 + ((i_gemm_def->br_offs_A[i_br]*2)/i_gemm_def->k) + i_m);
  } else if (i_gemm_def->br_type == 3) {
    res = *((unsigned char*)i_gemm_def->zpt_u8 + ((i_gemm_def->stride_a*2)/i_gemm_def->k) * i_br + i_m);
  } else {
    res = *((unsigned char*)i_gemm_def->zpt_u8 + i_m);
  }
  return res;
}

LIBXSMM_API_INTERN
void libxsmm_ref_allocate_c_scratch(libxsmm_gemm_def* i_gemm_def) {
  if (i_gemm_def->fuse_colbias_add > 0 || i_gemm_def->fuse_relu > 0 || i_gemm_def->fuse_sigmoid > 0) {
    if (i_gemm_def->c_type != LIBXSMM_DATATYPE_F32) {
      i_gemm_def->fuse_via_scratch = 1;
      i_gemm_def->c_scratch = (void*) malloc((size_t)i_gemm_def->ldc * i_gemm_def->n * 4);
    } else {
      i_gemm_def->fuse_via_scratch = 0;
    }
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_allocate_c_vnni_scratch(libxsmm_gemm_def* i_gemm_def) {
  if (i_gemm_def->fuse_vnni_c > 0) {
    i_gemm_def->c_vnni_scratch = (void*) malloc((size_t)i_gemm_def->ldc * i_gemm_def->n * LIBXSMM_TYPESIZE(i_gemm_def->c_type));
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_deallocate_c_scratch(libxsmm_gemm_def* i_gemm_def) {
  if (i_gemm_def->fuse_via_scratch) {
    free(i_gemm_def->c_scratch);
    i_gemm_def->c_scratch = NULL;
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_deallocate_c_vnni_scratch(libxsmm_gemm_def* i_gemm_def) {
  if (i_gemm_def->fuse_vnni_c > 0) {
    free(i_gemm_def->c_vnni_scratch);
    i_gemm_def->c_vnni_scratch = NULL;
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_apply_preop(libxsmm_gemm_def* i_gemm_def, void *param, const libxsmm_gemm_descriptor *i_xgemm_desc) {
  libxsmm_gemm_ext_param *gemm_param = (libxsmm_gemm_ext_param*)param;
  if (i_gemm_def->fuse_colbias_add > 0) {
    if ( i_gemm_def->beta == 0 ) {
      libxsmm_meltw_unary_param unary_param;
      libxsmm_descriptor_blob blob;
      const libxsmm_meltw_descriptor *const desc = libxsmm_meltw_descriptor_init2(&blob, i_gemm_def->c_type, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, i_xgemm_desc->m, i_xgemm_desc->n, i_xgemm_desc->m, i_xgemm_desc->ldc, 0, 0, (unsigned short)LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL, (unsigned short)LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, LIBXSMM_MELTW_OPERATION_UNARY);
      unary_param.in.primary  = (void*)gemm_param->d.primary;
      unary_param.out.primary = (void*)i_gemm_def->c_scratch;
      libxsmm_reference_unary_elementwise(&unary_param, desc);
      i_gemm_def->beta = 1.0;
      i_gemm_def->c_type = LIBXSMM_DATATYPE_F32;
    } else {
      /* Setup binary desc and call refence kernel*/
      libxsmm_meltw_binary_param binary_param;
      libxsmm_descriptor_blob blob;
      const libxsmm_meltw_descriptor *const desc = libxsmm_meltw_descriptor_init2(&blob, i_gemm_def->c_type, i_gemm_def->c_type, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, i_xgemm_desc->m, i_xgemm_desc->n,
          i_xgemm_desc->m, i_xgemm_desc->ldc, i_xgemm_desc->ldc, 0, (unsigned short)LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0, (unsigned short)LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_OPERATION_BINARY);
      binary_param.in0.primary = (void*)gemm_param->d.primary;
      binary_param.in1.primary = (void*)gemm_param->c.primary;
      binary_param.out.primary = (void*)i_gemm_def->c_scratch;
      libxsmm_reference_binary_elementwise(&binary_param, desc);
      i_gemm_def->c_type = LIBXSMM_DATATYPE_F32;
    }
  } else {
    if (i_gemm_def->fuse_via_scratch > 0) {
      if ( i_gemm_def->beta != 0 ) {
        libxsmm_meltw_unary_param unary_param;
        libxsmm_descriptor_blob blob;
        const libxsmm_meltw_descriptor *const desc = libxsmm_meltw_descriptor_init2(&blob, i_gemm_def->c_type, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, i_xgemm_desc->m, i_xgemm_desc->n, i_xgemm_desc->ldc, i_xgemm_desc->ldc, 0, 0, (unsigned short)LIBXSMM_MELTW_FLAG_UNARY_NONE, (unsigned short)LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, LIBXSMM_MELTW_OPERATION_UNARY);
        unary_param.in.primary  = (void*)gemm_param->c.primary;
        unary_param.out.primary = (void*)i_gemm_def->c_scratch;
        libxsmm_reference_unary_elementwise(&unary_param, desc);
      }
      i_gemm_def->c_type = LIBXSMM_DATATYPE_F32;
    }
  }
  return;
}

LIBXSMM_API_INTERN
void libxsmm_ref_apply_postop(libxsmm_gemm_def* i_gemm_def, void *param, const libxsmm_gemm_descriptor *i_xgemm_desc) {
  libxsmm_gemm_ext_param *gemm_param = (libxsmm_gemm_ext_param*)param;
  if (i_gemm_def->fuse_relu > 0) {
    /* Setup unary desc and call refence kernel*/
    libxsmm_meltw_unary_param unary_param;
    libxsmm_descriptor_blob blob;
    const libxsmm_meltw_descriptor *const desc = libxsmm_meltw_descriptor_init2(&blob, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_F32, (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ), i_xgemm_desc->m, i_xgemm_desc->n,
        i_xgemm_desc->ldc, i_xgemm_desc->ldc, 0, 0, (unsigned short) ((i_gemm_def->fuse_relu_bitmask > 0) ? LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT: LIBXSMM_MELTW_FLAG_UNARY_NONE), (unsigned short)LIBXSMM_MELTW_TYPE_UNARY_RELU, LIBXSMM_MELTW_OPERATION_UNARY);
    unary_param.in.primary  = (void*)i_gemm_def->c_scratch;
    unary_param.out.primary = (void*)gemm_param->c.primary;
    if (i_gemm_def->fuse_relu_bitmask > 0) {
      unary_param.out.secondary = (void*)gemm_param->c.secondary;
    }
    libxsmm_reference_unary_elementwise(&unary_param, desc);
  } else if (i_gemm_def->fuse_sigmoid > 0) {
    /* Setup unary desc and call refence kernel*/
    libxsmm_meltw_unary_param unary_param;
    libxsmm_descriptor_blob blob;
    const libxsmm_meltw_descriptor *const desc = libxsmm_meltw_descriptor_init2(&blob, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_F32, (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ), i_xgemm_desc->m, i_xgemm_desc->n,
        i_xgemm_desc->ldc, i_xgemm_desc->ldc, 0, 0, (unsigned short)LIBXSMM_MELTW_FLAG_UNARY_NONE, (unsigned short)LIBXSMM_MELTW_TYPE_UNARY_SIGMOID, LIBXSMM_MELTW_OPERATION_UNARY);
    unary_param.in.primary  = (void*)i_gemm_def->c_scratch;
    unary_param.out.primary = (void*)gemm_param->c.primary;
    libxsmm_reference_unary_elementwise(&unary_param, desc);
  } else {
    /* Copy to actual C if using scratch  */
    if (i_gemm_def->fuse_via_scratch > 0) {
      libxsmm_meltw_unary_param unary_param;
      libxsmm_descriptor_blob blob;
      const libxsmm_meltw_descriptor *const desc = libxsmm_meltw_descriptor_init2(&blob, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_F32, (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ), i_xgemm_desc->m, i_xgemm_desc->n,
          i_xgemm_desc->ldc, i_xgemm_desc->ldc, 0, 0, (unsigned short)LIBXSMM_MELTW_FLAG_UNARY_NONE, (unsigned short)LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, LIBXSMM_MELTW_OPERATION_UNARY);
      unary_param.in.primary  = (void*)i_gemm_def->c_scratch;
      unary_param.out.primary = (void*)gemm_param->c.primary;
      libxsmm_reference_unary_elementwise(&unary_param, desc);
    }
  }

  return;
}

LIBXSMM_API_INTERN
void libxsmm_setup_gemm_def(libxsmm_gemm_def* i_gemm_def, void *param, const libxsmm_gemm_descriptor *i_xgemm_desc) {
  libxsmm_gemm_def l_gemm_def;
  libxsmm_gemm_ext_param *gemm_param_ext = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI) > 0) ? (libxsmm_gemm_ext_param*)param : NULL;
  libxsmm_gemm_param *gemm_param = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI) > 0) ? NULL :  (libxsmm_gemm_param*)param;
  libxsmm_datatype l_dtype_a = (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype l_dtype_b = (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype l_dtype_c = (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype l_dtype_comp = (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype );
  libxsmm_blasint l_lda = i_xgemm_desc->lda, l_ldb = i_xgemm_desc->ldb, l_ldc = i_xgemm_desc->ldc;
  libxsmm_blasint l_m = i_xgemm_desc->m, l_n = i_xgemm_desc->n, l_k = i_xgemm_desc->k;
  int l_aligned_a = 1;
  int l_aligned_c = 1;
  int l_trans_a = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) > 0) ? 1 : 0;
  int l_trans_b = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0) ? 1 : 0;
  int l_vnni_a = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) ? 1 : 0;
  int l_vnni_b = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0) ? 1 : 0;
  int l_vnni_c = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_C) > 0) ? 1 : 0;
  double l_alpha = 1.0;
  double l_beta = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BETA_0) > 0) ? 0.0 : 1.0;
  int l_br = 0;
  int l_br_type = 0;
  int l_binary_postop = 0;
  int l_unary_postop = 0;
  libxsmm_gemm_prefetch_type l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;

  l_gemm_def.fuse_relu = 0;
  l_gemm_def.fuse_relu_bitmask = 0;
  l_gemm_def.fuse_sigmoid = 0;
  l_gemm_def.fuse_vnni_c = 0;
  l_gemm_def.fuse_colbias_add = 0;

  /* Check for fusions and set proper flags */
  if (i_xgemm_desc->eltw_cp_op == LIBXSMM_MELTW_OPERATION_UNARY) {
    if (i_xgemm_desc->eltw_cp_param == LIBXSMM_MELTW_TYPE_UNARY_RELU) {
      l_gemm_def.fuse_relu = 1;
      if ((i_xgemm_desc->eltw_cp_flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) {
        l_gemm_def.fuse_relu_bitmask = 1;
      } else {
        l_gemm_def.fuse_relu_bitmask = 0;
      }
    }
    if (i_xgemm_desc->eltw_cp_param == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID) {
      l_gemm_def.fuse_sigmoid = 1;
    }
  }

  if (libxsmm_gemm_descriptor_get_meltw_operation(i_xgemm_desc) == LIBXSMM_MELTW_OPERATION_BINARY) {
    if (libxsmm_gemm_descriptor_get_meltw_param(i_xgemm_desc) == LIBXSMM_MELTW_TYPE_BINARY_ADD) {
      if (((i_xgemm_desc->meltw_flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0) > 0 ) ||
          ((i_xgemm_desc->meltw_flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1) > 0 )) {
        l_gemm_def.fuse_colbias_add = 1;
      }
    }
  }

  l_gemm_def.fuse_zpt_sub = 0;
  l_gemm_def.is_Ai4Bf16_gemm = 0;
  l_gemm_def.is_Ai4Bi8_gemm = 0;
  l_gemm_def.is_Ai2Bi8_gemm = 0;
  l_gemm_def.is_Ai1Bi8_gemm = 0;
  l_gemm_def.is_Amxfp4Bbf16_gemm = 0;
  l_gemm_def.is_Amxfp4Bfp32_gemm = 0;
  l_gemm_def.is_Abf8Bbf16_gemm = 0;
  l_gemm_def.is_Abf8Bf16_gemm = 0;
  l_gemm_def.is_Ahf8Bbf16_gemm = 0;
  l_gemm_def.is_Amxfp4Bbf16_gemm = 0;
  l_gemm_def.is_Amxfp4Bfp32_gemm = 0;
  l_gemm_def.is_Amxfp4Bi8_gemm = 0;
  l_gemm_def.unsigned_a = 0;
  l_gemm_def.unsigned_b = 0;
  l_gemm_def.is_spmm = 0;
  l_gemm_def.scf_b_u8 = NULL;
  l_gemm_def.scf_c_u8 = NULL;

  if (LIBXSMM_GEMM_GETENUM_A_UNSIGNED(i_xgemm_desc->datatype)) {
    l_gemm_def.unsigned_a = 1;
  }

  if (LIBXSMM_GEMM_GETENUM_B_UNSIGNED(i_xgemm_desc->datatype)) {
    l_gemm_def.unsigned_b = 1;
  }

  if ((l_dtype_a == LIBXSMM_DATATYPE_MXFP4X2) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_INTLV_A_FORMAT) == 0)) {
    if (l_dtype_b == LIBXSMM_DATATYPE_BF16) {
      l_gemm_def.is_Amxfp4Bbf16_gemm = 1;
    }
    if (l_dtype_b == LIBXSMM_DATATYPE_F32) {
      l_gemm_def.is_Amxfp4Bfp32_gemm = 1;
    }
    l_gemm_def.mxfp4_block_size = 32;
  }

  if ((l_dtype_a == LIBXSMM_DATATYPE_MXFP4X2) && (l_dtype_b == LIBXSMM_DATATYPE_I8) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_INTLV_A_FORMAT) > 0)) {
    l_gemm_def.is_Amxfp4Bi8_gemm = 1;
    l_gemm_def.mxfp4_block_size = 32;
  }

  if ((l_dtype_a == LIBXSMM_DATATYPE_I4X2) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_INTLV_A_FORMAT) == 0) && (l_dtype_b == LIBXSMM_DATATYPE_F16)) {
    l_gemm_def.is_Ai4Bf16_gemm = 1;
    l_gemm_def.fuse_zpt_sub = 1;
  }

  if ((l_dtype_a == LIBXSMM_DATATYPE_I4X2) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_INTLV_A_FORMAT) > 0) && (l_dtype_b == LIBXSMM_DATATYPE_I8)) {
    l_gemm_def.is_Ai4Bi8_gemm = 1;
    l_gemm_def.fuse_zpt_sub = 1;
  }

  if ((l_dtype_a == LIBXSMM_DATATYPE_I2X4) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_INTLV_A_FORMAT) > 0) && (l_dtype_b == LIBXSMM_DATATYPE_I8 || l_dtype_b == LIBXSMM_DATATYPE_U8)) {
    l_gemm_def.is_Ai2Bi8_gemm = 1;
  }

  if ((l_dtype_a == LIBXSMM_DATATYPE_I1X8) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) && (l_dtype_b == LIBXSMM_DATATYPE_I8)) {
    l_gemm_def.is_Ai1Bi8_gemm = 1;
  }

  if (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0) || ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) > 0) || ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0)) {
    l_br = (int) ( (gemm_param_ext == NULL) ? (*(unsigned long long*)gemm_param->op.tertiary) : (*(unsigned long long*)gemm_param_ext->op.tertiary) );
  }

  if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
    l_br_type = 1;
    if (gemm_param_ext == NULL) {
      l_gemm_def.br_addr_A = (void**)gemm_param->a.primary;
      l_gemm_def.br_addr_B = (void**)gemm_param->b.primary;
      if (l_gemm_def.is_Ai4Bi8_gemm > 0) {
        l_gemm_def.zpt_u8_braddr = (unsigned char**)gemm_param->a.quaternary;
      }
    } else {
      l_gemm_def.br_addr_A = (void**)gemm_param_ext->a.primary;
      l_gemm_def.br_addr_B = (void**)gemm_param_ext->b.primary;
      if (l_gemm_def.is_Ai4Bi8_gemm > 0) {
        l_gemm_def.zpt_u8_braddr = (unsigned char**)gemm_param_ext->a.quaternary;
      }
    }
  } else if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) {
    l_br_type = 2;
    if (gemm_param_ext == NULL) {
      l_gemm_def.br_offs_A = (long long *)gemm_param->a.secondary;
      l_gemm_def.br_offs_B = (long long *)gemm_param->b.secondary;
      if (l_gemm_def.is_Ai4Bi8_gemm > 0) {
        l_gemm_def.zpt_u8 = (unsigned char*)gemm_param->a.quaternary;
      }
    } else {
      l_gemm_def.br_offs_A = (long long *)gemm_param_ext->a.secondary;
      l_gemm_def.br_offs_B = (long long *)gemm_param_ext->b.secondary;
      if (l_gemm_def.is_Ai4Bi8_gemm > 0) {
        l_gemm_def.zpt_u8 = (unsigned char*)gemm_param_ext->a.quaternary;
      }
    }
  } else if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) {
    l_br_type = 3;
    l_gemm_def.stride_a = i_xgemm_desc->c1/LIBXSMM_TYPESIZE(l_dtype_a);
    l_gemm_def.stride_b = i_xgemm_desc->c2/LIBXSMM_TYPESIZE(l_dtype_b);
    if (l_gemm_def.is_Ai4Bi8_gemm > 0) {
      if (gemm_param_ext == NULL) {
        l_gemm_def.zpt_u8 = (unsigned char*)gemm_param->a.quaternary;
      } else {
        l_gemm_def.zpt_u8 = (unsigned char*)gemm_param_ext->a.quaternary;
      }
    }
  } else if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) {
    l_br_type = 0;
    l_br = 1;
    l_gemm_def.is_spmm = 1;
  } else {
    l_br_type = 0;
    l_br = 1;
    if (l_gemm_def.is_Ai4Bi8_gemm > 0) {
      if (gemm_param_ext == NULL) {
        l_gemm_def.zpt_u8 = (unsigned char*)gemm_param->a.quaternary;
      } else {
        l_gemm_def.zpt_u8 = (unsigned char*)gemm_param_ext->a.quaternary;
      }
    }
  }

  if (gemm_param_ext == NULL) {
    if (l_gemm_def.is_spmm > 0) {
      l_gemm_def.decompress_bitmap = (unsigned char*)gemm_param->a.secondary;
      if (l_dtype_c != LIBXSMM_DATATYPE_F32) {
        l_gemm_def.c_scratch = (void*)malloc((size_t)l_m*l_n*4);
      } else {
        l_gemm_def.c_scratch = NULL;
      }
    } else {
      if ( ((l_dtype_a == LIBXSMM_DATATYPE_I8) || (l_dtype_a == LIBXSMM_DATATYPE_I4X2))  && (l_dtype_b == LIBXSMM_DATATYPE_F16) &&
               (l_dtype_comp == LIBXSMM_DATATYPE_F16 || l_dtype_comp == LIBXSMM_DATATYPE_F32 || l_dtype_comp == LIBXSMM_DATATYPE_IMPLICIT ) && (l_dtype_c == LIBXSMM_DATATYPE_F16 || l_dtype_c == LIBXSMM_DATATYPE_F32) ) {
        l_gemm_def.zpt_f16 = (libxsmm_float16*)gemm_param->a.quaternary;
        l_gemm_def.scf_f16 = (libxsmm_float16*)gemm_param->a.tertiary;
      }
      if (l_gemm_def.is_Amxfp4Bbf16_gemm > 0 || l_gemm_def.is_Amxfp4Bfp32_gemm > 0 || l_gemm_def.is_Amxfp4Bi8_gemm > 0) {
        l_gemm_def.scf_u8 = (unsigned char*)gemm_param->a.tertiary;
        if (l_br_type == 1) {
          l_gemm_def.scf_u8_braddr = (unsigned char**)gemm_param->a.tertiary;
        }
        if (l_gemm_def.is_Amxfp4Bi8_gemm > 0 ) {
          l_gemm_def.scf_b_f32 = (float*)gemm_param->b.tertiary;
          if (l_br_type == 1) {
            l_gemm_def.scf_b_f32_braddr = (float**)gemm_param->b.tertiary;
          }
        }
      }
      if ( ((l_dtype_a == LIBXSMM_DATATYPE_MXBF8) || (l_dtype_a == LIBXSMM_DATATYPE_MXHF8) ||
            (l_dtype_a == LIBXSMM_DATATYPE_MXBF6) || (l_dtype_a == LIBXSMM_DATATYPE_MXHF6) ||
            (l_dtype_a == LIBXSMM_DATATYPE_MXFP4X2)) && (l_dtype_a == l_dtype_b) &&
           ((l_dtype_c == LIBXSMM_DATATYPE_F32) || (l_dtype_c == l_dtype_a)) ) {
        l_gemm_def.scf_u8   = (unsigned char*)gemm_param->a.tertiary;
        l_gemm_def.scf_b_u8 = (unsigned char*)gemm_param->b.tertiary;
        if (l_dtype_c == l_dtype_a) {
          l_gemm_def.scf_c_u8 = (unsigned char*)gemm_param->c.tertiary;
        }
      }
      if ( (l_dtype_a    == LIBXSMM_DATATYPE_I8)  && (l_dtype_b == LIBXSMM_DATATYPE_BF16) &&
             (l_dtype_c == LIBXSMM_DATATYPE_BF16 || l_dtype_c == LIBXSMM_DATATYPE_F32) ) {
          l_gemm_def.scf_f32 =  (float*)gemm_param->a.tertiary;
      }
      if ( l_dtype_a == LIBXSMM_DATATYPE_I8 && l_dtype_b == LIBXSMM_DATATYPE_I8 && l_dtype_c == LIBXSMM_DATATYPE_F32 ) {
        l_gemm_def.scf =  *((float*)gemm_param->c.tertiary);
      }
    }
  } else {
    if ( ((l_dtype_a == LIBXSMM_DATATYPE_I8) || (l_dtype_a == LIBXSMM_DATATYPE_I4X2))  && (l_dtype_b == LIBXSMM_DATATYPE_F16) &&
             (l_dtype_comp == LIBXSMM_DATATYPE_F16 || l_dtype_comp == LIBXSMM_DATATYPE_F32 || l_dtype_comp == LIBXSMM_DATATYPE_IMPLICIT ) && (l_dtype_c == LIBXSMM_DATATYPE_F16 || l_dtype_c == LIBXSMM_DATATYPE_F32) ) {
      l_gemm_def.zpt_f16 = (libxsmm_float16*)gemm_param_ext->a.quaternary;
      l_gemm_def.scf_f16 = (libxsmm_float16*)gemm_param_ext->a.tertiary;
    }
    if (l_gemm_def.is_Amxfp4Bbf16_gemm > 0 || l_gemm_def.is_Amxfp4Bfp32_gemm > 0 || l_gemm_def.is_Amxfp4Bi8_gemm > 0) {
      l_gemm_def.scf_u8 = (unsigned char*)gemm_param_ext->a.tertiary;
      if (l_br_type == 1) {
        l_gemm_def.scf_u8_braddr = (unsigned char**)gemm_param_ext->a.tertiary;
      }
      if (l_gemm_def.is_Amxfp4Bi8_gemm > 0 ) {
        l_gemm_def.scf_b_f32 = (float*)gemm_param_ext->b.tertiary;
        if (l_br_type == 1) {
          l_gemm_def.scf_b_f32_braddr = (float**)gemm_param_ext->b.tertiary;
        }
      }
    }
    if ( ((l_dtype_a == LIBXSMM_DATATYPE_MXBF8) || (l_dtype_a == LIBXSMM_DATATYPE_MXHF8) ||
          (l_dtype_a == LIBXSMM_DATATYPE_MXBF6) || (l_dtype_a == LIBXSMM_DATATYPE_MXHF6) ||
          (l_dtype_a == LIBXSMM_DATATYPE_MXFP4X2)) && (l_dtype_a == l_dtype_b) &&
         ((l_dtype_c == LIBXSMM_DATATYPE_F32) || (l_dtype_c == l_dtype_a)) ) {
      l_gemm_def.scf_u8   = (unsigned char*)gemm_param_ext->a.tertiary;
      l_gemm_def.scf_b_u8 = (unsigned char*)gemm_param_ext->b.tertiary;
      if (l_dtype_c == l_dtype_a) {
        l_gemm_def.scf_c_u8 = (unsigned char*)gemm_param_ext->c.tertiary;
      }
    }
    if ( (l_dtype_a    == LIBXSMM_DATATYPE_I8)  && (l_dtype_b == LIBXSMM_DATATYPE_BF16) &&
           (l_dtype_c == LIBXSMM_DATATYPE_BF16 || l_dtype_c == LIBXSMM_DATATYPE_F32) ) {
        l_gemm_def.scf_f32 =  (float*)gemm_param_ext->a.tertiary;
    }
    if ( l_dtype_a == LIBXSMM_DATATYPE_I8 && l_dtype_b == LIBXSMM_DATATYPE_I8 && l_dtype_c == LIBXSMM_DATATYPE_F32 ) {
      l_gemm_def.scf =  *((float*)gemm_param_ext->c.tertiary);
    }
  }

  /* setting static GEMM parameters */
  l_gemm_def.m = l_m;
  l_gemm_def.n = l_n;
  l_gemm_def.k = l_k;
  l_gemm_def.lda = l_lda;
  l_gemm_def.ldb = l_ldb;
  l_gemm_def.ldc = l_ldc;
  l_gemm_def.a_type = l_dtype_a;
  l_gemm_def.b_type = l_dtype_b;
  l_gemm_def.comp_type = l_dtype_comp;
  l_gemm_def.c_type = l_dtype_c;
  l_gemm_def.alpha = l_alpha;
  l_gemm_def.beta = l_beta;
  l_gemm_def.trans_a = l_trans_a;
  l_gemm_def.trans_b = l_trans_b;
  l_gemm_def.vnni_a = l_vnni_a;
  l_gemm_def.vnni_b = l_vnni_b;
  l_gemm_def.vnni_c = l_vnni_c;
  l_gemm_def.aligned_a = l_aligned_a;
  l_gemm_def.aligned_c = l_aligned_c;
  l_gemm_def.prefetch = l_prefetch;
  l_gemm_def.br_type = l_br_type;
  l_gemm_def.br_count = l_br;
  l_gemm_def.binary_postop = l_binary_postop;
  l_gemm_def.unary_postop  = l_unary_postop;
  l_gemm_def.fuse_vnni_c = l_vnni_c;

  *i_gemm_def = l_gemm_def;
  return;
}

/* Quantise |x| to a 3-bit unsigned E2M1 code */
LIBXSMM_API_INTERN
unsigned char libxsmm_gemm_ref_encode_e2m1_abs(float absval);
LIBXSMM_API_INTERN
unsigned char libxsmm_gemm_ref_encode_e2m1_abs(float absval) {
  if (absval != absval) return 0x7;
  if (absval >  5.0f)   return 0x7;
  if (absval >= 3.5f)   return 0x6;
  if (absval >  2.5f)   return 0x5;
  if (absval >= 1.75f)  return 0x4;
  if (absval >  1.25f)  return 0x3;
  if (absval >= 0.75f)  return 0x2;
  if (absval >  0.25f)  return 0x1;
  return 0x0;
}

/* Convert a block of 32 FP32 values to MXFP4 (E2M1) with shared E8M0 scale.
   out_data[16] = packed nibbles, *out_scale = E8M0 biased exponent byte. */
LIBXSMM_API_INTERN
void libxsmm_gemm_ref_fp32_to_mxfp4_block(const float* in, unsigned char* out_data, unsigned char* out_scale);
LIBXSMM_API_INTERN
void libxsmm_gemm_ref_fp32_to_mxfp4_block(const float* in, unsigned char* out_data, unsigned char* out_scale) {
  union { float f; unsigned int u; } mx;
  float amax = 0.0f;
  int shared_exp, scale_mant, i;
  float scale;
  float bf16_in[32];

  /* 0. Convert F32 -> BF16 -> F32 to match JIT precision */
  for (i = 0; i < 32; i++) {
    libxsmm_bfloat16 tmp_bf16;
    libxsmm_bfloat16_f32 tmp_u;
    float tmp_f32 = in[i];
    libxsmm_rne_convert_fp32_bf16(&tmp_f32, &tmp_bf16, 1);
    tmp_u.i[0] = 0; tmp_u.i[1] = tmp_bf16;
    bf16_in[i] = tmp_u.f;
  }

  /* 1. Max absolute value */
  for (i = 0; i < 32; i++) {
    float a = LIBXSMM_FABSF(bf16_in[i]);
    if (a > amax || a != a) amax = a;
  }

  /* 2. Shared biased exponent, offset by elem_emax = 2 */
  mx.f = amax;
  shared_exp = (amax == 0.0f) ? 0 : (int)((mx.u >> 23) & 0xFFu);
  shared_exp -= 2;
  if (shared_exp < 0)   shared_exp = 0;
  if (shared_exp > 254) shared_exp = 254;
  *out_scale = (unsigned char)shared_exp;

  /* 3. Construct scale, round to BF16, compute BF16 reciprocal */
  scale_mant = (shared_exp == 0) ? (1 << 22) : 0;
  mx.u = ((unsigned int)shared_exp << 23) | (unsigned int)scale_mant;
  scale = mx.f;

  {
    libxsmm_bfloat16 bf16_scale;
    float scale_f32, rcp_f32;
    libxsmm_bfloat16 bf16_rcp;
    libxsmm_rne_convert_fp32_bf16(&scale, &bf16_scale, 1);
    { libxsmm_bfloat16_f32 ts; ts.i[0] = 0; ts.i[1] = bf16_scale; scale_f32 = ts.f; }
    rcp_f32 = 1.0f / scale_f32;
    libxsmm_rne_convert_fp32_bf16(&rcp_f32, &bf16_rcp, 1);
    { libxsmm_bfloat16_f32 tr; tr.i[0] = 0; tr.i[1] = bf16_rcp; rcp_f32 = tr.f; }

    /* 4. Multiply each element by BF16 reciprocal, round to BF16, encode E2M1 */
    for (i = 0; i < 16; i++) {
      float v0 = bf16_in[2*i]     * rcp_f32;
      float v1 = bf16_in[2*i + 1] * rcp_f32;
      libxsmm_bfloat16 bf16_v0, bf16_v1;
      union { float f; unsigned int u; } u0, u1;
      unsigned char s0, s1, lo, hi;
      libxsmm_rne_convert_fp32_bf16(&v0, &bf16_v0, 1);
      libxsmm_rne_convert_fp32_bf16(&v1, &bf16_v1, 1);
      { libxsmm_bfloat16_f32 t0, t1;
        t0.i[0] = 0; t0.i[1] = bf16_v0; v0 = t0.f;
        t1.i[0] = 0; t1.i[1] = bf16_v1; v1 = t1.f;
      }
      u0.f = bf16_in[2*i];     s0 = (u0.u >> 31) ? 0x8u : 0u;
      u1.f = bf16_in[2*i + 1]; s1 = (u1.u >> 31) ? 0x8u : 0u;
      lo = (unsigned char)(s0 | libxsmm_gemm_ref_encode_e2m1_abs(LIBXSMM_FABSF(v0)));
      hi = (unsigned char)(s1 | libxsmm_gemm_ref_encode_e2m1_abs(LIBXSMM_FABSF(v1)));
      out_data[i] = (unsigned char)((hi << 4) | lo);
    }
  }
}

/* Convert a block of 32 FP32 values to mxbf8 (E5M2) with shared E8M0 scale.
   out_data[32] = BF8 (E5M2) bytes, *out_scale = E8M0 biased exponent byte. */
LIBXSMM_API_INTERN
void libxsmm_gemm_ref_fp32_to_mxbf8_block(const float* in, unsigned char* out_data, unsigned char* out_scale);
LIBXSMM_API_INTERN
void libxsmm_gemm_ref_fp32_to_mxbf8_block(const float* in, unsigned char* out_data, unsigned char* out_scale) {
  union { float f; unsigned int u; } mx;
  float amax = 0.0f;
  int shared_exp, scale_mant, i;
  float scale;
  float bf16_in[32];

  /* 0. Convert F32 -> BF16 -> F32 to match JIT precision */
  for (i = 0; i < 32; i++) {
    libxsmm_bfloat16 tmp_bf16;
    libxsmm_bfloat16_f32 tmp_u;
    float tmp_f32 = in[i];
    libxsmm_rne_convert_fp32_bf16(&tmp_f32, &tmp_bf16, 1);
    tmp_u.i[0] = 0; tmp_u.i[1] = tmp_bf16;
    bf16_in[i] = tmp_u.f;
  }

  /* 1. Max absolute value */
  for (i = 0; i < 32; i++) {
    float a = LIBXSMM_FABSF(bf16_in[i]);
    if (a > amax || a != a) amax = a;
  }

  /* 2. Shared biased exponent, offset by elem_emax = 15 (for E5M2) */
  mx.f = amax;
  shared_exp = (amax == 0.0f) ? 0 : (int)((mx.u >> 23) & 0xFFu);
  shared_exp -= 15;
  if (shared_exp < 0)   shared_exp = 0;
  if (shared_exp > 254) shared_exp = 254;
  *out_scale = (unsigned char)shared_exp;

  /* 3. Construct scale, round to BF16, compute BF16 reciprocal */
  scale_mant = (shared_exp == 0) ? (1 << 22) : 0;
  mx.u = ((unsigned int)shared_exp << 23) | (unsigned int)scale_mant;
  scale = mx.f;

  {
    libxsmm_bfloat16 bf16_scale;
    float scale_f32, rcp_f32;
    libxsmm_bfloat16 bf16_rcp;
    libxsmm_rne_convert_fp32_bf16(&scale, &bf16_scale, 1);
    { libxsmm_bfloat16_f32 ts; ts.i[0] = 0; ts.i[1] = bf16_scale; scale_f32 = ts.f; }
    rcp_f32 = 1.0f / scale_f32;
    libxsmm_rne_convert_fp32_bf16(&rcp_f32, &bf16_rcp, 1);
    { libxsmm_bfloat16_f32 tr; tr.i[0] = 0; tr.i[1] = bf16_rcp; rcp_f32 = tr.f; }

    /* 4. Multiply each element by BF16 reciprocal, round to BF16, convert to BF8 (E5M2) */
    for (i = 0; i < 32; i++) {
      float v = bf16_in[i] * rcp_f32;
      libxsmm_bfloat16 bf16_v;
      libxsmm_bfloat8 bf8_v;
      libxsmm_rne_convert_fp32_bf16(&v, &bf16_v, 1);
      { libxsmm_bfloat16_f32 tv; tv.i[0] = 0; tv.i[1] = bf16_v; v = tv.f; }
      libxsmm_rne_convert_fp32_bf8(&v, &bf8_v, 1);
      /* Saturate: if the BF8 is inf or NaN, clamp to max normal */
      if (((unsigned char)bf8_v & 0x7C) == 0x7C) {
        bf8_v = (libxsmm_bfloat8)(((unsigned char)bf8_v & 0x80) | 0x7B);
      }
      out_data[i] = (unsigned char)bf8_v;
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_ref_matmul( const libxsmm_gemm_def* i_gemm_def, void* a, void* b, void* c ) {
  libxsmm_blasint l_r, l_j, l_i, l_s, l_k2;
  libxsmm_blasint lda = i_gemm_def->lda;
  libxsmm_blasint ldb = i_gemm_def->ldb;

  libxsmm_blasint ldc = i_gemm_def->ldc;
  libxsmm_blasint m = i_gemm_def->m;
  libxsmm_blasint n = i_gemm_def->n;
  libxsmm_blasint k = i_gemm_def->k;
  long long offs_a = 0;
  long long offs_b = 0;
  unsigned char packed_char = 0;
  char even_use = 0;
  char odd_use = 0;

  /* MX x MX reference path does not support the address (addrbr) or offset (offsbr) batch-reduce modes */
  if ( ((i_gemm_def->br_type == 1) || (i_gemm_def->br_type == 2)) &&
       ((i_gemm_def->a_type == LIBXSMM_DATATYPE_MXBF8) || (i_gemm_def->a_type == LIBXSMM_DATATYPE_MXHF8) ||
        (i_gemm_def->a_type == LIBXSMM_DATATYPE_MXBF6) || (i_gemm_def->a_type == LIBXSMM_DATATYPE_MXHF6) ||
        (i_gemm_def->a_type == LIBXSMM_DATATYPE_MXFP4X2)) &&
       (i_gemm_def->b_type == i_gemm_def->a_type) && ((i_gemm_def->c_type == LIBXSMM_DATATYPE_F32) || (i_gemm_def->c_type == i_gemm_def->a_type)) ) {
    fprintf(stderr, "LIBXSMM reference: MX x MX GEMM does not support the address (addrbr) or offset (offsbr) batch-reduce modes!\n");
    return;
  }

  /* MX x MX reference path does not support elementwise (unary or binary) post-fusion */
  if ( ((i_gemm_def->a_type == LIBXSMM_DATATYPE_MXBF8) || (i_gemm_def->a_type == LIBXSMM_DATATYPE_MXHF8) ||
        (i_gemm_def->a_type == LIBXSMM_DATATYPE_MXBF6) || (i_gemm_def->a_type == LIBXSMM_DATATYPE_MXHF6) ||
        (i_gemm_def->a_type == LIBXSMM_DATATYPE_MXFP4X2)) &&
       (i_gemm_def->b_type == i_gemm_def->a_type) && ((i_gemm_def->c_type == LIBXSMM_DATATYPE_F32) || (i_gemm_def->c_type == i_gemm_def->a_type)) &&
       ((i_gemm_def->fuse_colbias_add > 0) || (i_gemm_def->fuse_relu > 0) ||
        (i_gemm_def->fuse_relu_bitmask > 0) || (i_gemm_def->fuse_sigmoid > 0)) ) {
    fprintf(stderr, "LIBXSMM reference: MX x MX GEMM does not support elementwise (unary or binary) post-fusion!\n");
    return;
  }

  if (i_gemm_def->is_spmm > 0) {
    float *a_f32 = (float*)a;
    float *b_f32 = (float*)b;
    float *c_f32 = (float*)c;
    libxsmm_bfloat16 *a_bf16 = (libxsmm_bfloat16*)a;
    libxsmm_float16 *a_f16 = (libxsmm_float16*)a;
    libxsmm_bfloat8 *a_bf8 = (libxsmm_bfloat8*)a;
    libxsmm_hfloat8 *a_hf8 = (libxsmm_hfloat8*)a;
    libxsmm_bfloat16 *b_bf16 = (libxsmm_bfloat16*)b;
    libxsmm_float16 *b_f16 = (libxsmm_float16*)b;
    libxsmm_bfloat16 *c_bf16 = (libxsmm_bfloat16*)c;
    libxsmm_float16 *c_f16 = (libxsmm_float16*)c;
    float *c_scratch = (float*)i_gemm_def->c_scratch;
    unsigned char *bitmap = (unsigned char*)i_gemm_def->decompress_bitmap;
    libxsmm_blasint l_k_block = 1;
    unsigned long long compressed_index = 0;

    if (i_gemm_def->a_type != LIBXSMM_DATATYPE_F32) {
      l_k_block = libxsmm_cpuid_dot_pack_factor(i_gemm_def->b_type);
    }
    for (l_s = 0; l_s < (k / l_k_block); l_s++) {
      for (l_i = 0; l_i < m; l_i++) {
        if (i_gemm_def->beta == 0 && l_s == 0) {
          for (l_j = 0; l_j < n; l_j++) {
            if (i_gemm_def->c_type != LIBXSMM_DATATYPE_F32) {
              c_scratch[(l_j * m) + l_i] = 0.0;
            } else {
              c_f32[(l_j * ldc) + l_i] = 0.0;
            }
          }
        }
        if (i_gemm_def->beta > 0 && l_s == 0 && (i_gemm_def->c_type != LIBXSMM_DATATYPE_F32)) {
          for (l_j = 0; l_j < n; l_j++) {
            if (i_gemm_def->c_type == LIBXSMM_DATATYPE_BF16) {
              c_scratch[(l_j * m) + l_i] = libxsmm_convert_bf16_to_f32(c_bf16[(l_j * ldc) + l_i]);
            }
            if (i_gemm_def->c_type == LIBXSMM_DATATYPE_F16) {
              c_scratch[(l_j * m) + l_i] = libxsmm_convert_f16_to_f32(c_f16[(l_j * ldc) + l_i]);
            }
          }
        }
        for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
          unsigned char bit_val = libxsmm_extract_bit((const char*)bitmap, l_i*l_k_block+l_k2, l_s, m*l_k_block);
          if (bit_val > 0) {
            float a_use = 0.0;
            if (i_gemm_def->a_type == LIBXSMM_DATATYPE_F32) {
              a_use = a_f32[compressed_index];
            } else if (i_gemm_def->a_type == LIBXSMM_DATATYPE_BF16) {
              a_use = libxsmm_convert_bf16_to_f32(a_bf16[compressed_index]);
            } else if (i_gemm_def->a_type == LIBXSMM_DATATYPE_F16) {
              a_use = libxsmm_convert_f16_to_f32(a_f16[compressed_index]);
            } else if (i_gemm_def->a_type == LIBXSMM_DATATYPE_BF8) {
              a_use = libxsmm_convert_bf8_to_f32(a_bf8[compressed_index]);
            } else if (i_gemm_def->a_type == LIBXSMM_DATATYPE_HF8) {
              a_use = libxsmm_convert_hf8_to_f32(a_hf8[compressed_index]);
            } else {

            }
            for (l_j = 0; l_j < n; l_j++) {
              float b_use = 0.0;
              if (i_gemm_def->b_type == LIBXSMM_DATATYPE_F32) {
                b_use = b_f32[(l_j * ldb) + (l_s*l_k_block) + l_k2];
              } else if (i_gemm_def->b_type == LIBXSMM_DATATYPE_BF16) {
                b_use = libxsmm_convert_bf16_to_f32(b_bf16[(l_j * ldb) + (l_s*l_k_block) + l_k2]);
              } else if (i_gemm_def->b_type == LIBXSMM_DATATYPE_F16) {
                b_use = libxsmm_convert_f16_to_f32(b_f16[(l_j * ldb) + (l_s*l_k_block) + l_k2]);
              } else {

              }
              if (i_gemm_def->c_type == LIBXSMM_DATATYPE_F32) {
                c_f32[(l_j * ldc) + l_i] += a_use * b_use;
              } else {
                c_scratch[(l_j * m) + l_i] += a_use * b_use;
              }
            }
            compressed_index++;
          }
        }
      }
    }
    if (i_gemm_def->c_type != LIBXSMM_DATATYPE_F32) {
      for (l_i = 0; l_i < m; l_i++) {
        for (l_j = 0; l_j < n; l_j++) {
          if (i_gemm_def->c_type == LIBXSMM_DATATYPE_BF16) {
            c_bf16[(l_j * ldc) + l_i] = libxsmm_convert_f32_to_bf16_rne(c_scratch[(l_j * m) + l_i]);
          }
          if (i_gemm_def->c_type == LIBXSMM_DATATYPE_F16) {
            c_f16[(l_j * ldc) + l_i] = libxsmm_convert_f32_to_f16(c_scratch[(l_j * m) + l_i]);
          }
        }
      }
    }
  } else if (i_gemm_def->is_Amxfp4Bbf16_gemm > 0 || i_gemm_def->is_Amxfp4Bfp32_gemm > 0) {
    unsigned char* c_a = (unsigned char*)a;
    float* f32_b = (float*)b;
    libxsmm_bfloat16* bf16_b = (libxsmm_bfloat16*)b;
    float*  i_c_f = (float*)c;
    libxsmm_bfloat16* i_c_bf16 = (libxsmm_bfloat16*)c;
    libxsmm_blasint l_k_block = i_gemm_def->mxfp4_block_size;
    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        float f32_accum = 0.0;
        if ( i_gemm_def->beta == 0 ) {
          if (i_gemm_def->c_type == LIBXSMM_DATATYPE_F32) {
            i_c_f[(l_j * ldc) + l_i] = 0.0f;
          }
          if (i_gemm_def->c_type == LIBXSMM_DATATYPE_BF16) {
            i_c_bf16[(l_j * ldc) + l_i] = (libxsmm_bfloat16)0;
          }
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          if (i_gemm_def->b_type == LIBXSMM_DATATYPE_BF16) {
            libxsmm_calculate_brgemm_offsets((void**)&c_a, (void**)&bf16_b, &offs_a, &offs_b, l_r, i_gemm_def);
          }
          if (i_gemm_def->b_type == LIBXSMM_DATATYPE_F32) {
            libxsmm_calculate_brgemm_offsets((void**)&c_a, (void**)&f32_b, &offs_a, &offs_b, l_r, i_gemm_def);
          }
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            /* Load MXFP4 scaling factor and upconvert to f32 */
            float scf_a = libxsmm_calculate_mxfp4_scf(l_r, l_s, l_i, i_gemm_def);
            for (l_k2 = 0; l_k2 < l_k_block; l_k2 += 2) {
              unsigned char even = 0, odd = 0;
              float evenf = 0, oddf = 0, evenf_b = 0, oddf_b = 0;
              unsigned char packed_a = c_a[offs_a + (l_s*l_k_block+l_k2)*lda/2+l_i];
              even = packed_a & 0x0f;
              evenf = libxsmm_convert_mxfp4_to_float(even) * scf_a;
              odd = (packed_a >> 4) & 0x0f;
              oddf = libxsmm_convert_mxfp4_to_float(odd) * scf_a;
              if (i_gemm_def->b_type == LIBXSMM_DATATYPE_BF16) {
                evenf_b = libxsmm_convert_bf16_to_f32(bf16_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2 + 0]);
                oddf_b  = libxsmm_convert_bf16_to_f32(bf16_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2 + 1]);
              }
              if (i_gemm_def->b_type == LIBXSMM_DATATYPE_F32) {
                evenf_b = f32_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2 + 0];
                oddf_b  = f32_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2 + 1];
              }
              f32_accum += evenf * evenf_b;
              f32_accum += oddf * oddf_b;
            }
          }
        }
        if (i_gemm_def->c_type == LIBXSMM_DATATYPE_F32) {
          i_c_f[(l_j * ldc) + l_i] += f32_accum;
        }
        if (i_gemm_def->c_type == LIBXSMM_DATATYPE_BF16) {
          float up_tmp;
          up_tmp = libxsmm_convert_bf16_to_f32(i_c_bf16[(l_j * ldc) + l_i]);
          up_tmp += f32_accum;
          i_c_bf16[(l_j * ldc) + l_i] = libxsmm_convert_f32_to_bf16_rne(up_tmp);
        }
      }
    }
  } else if (i_gemm_def->is_Amxfp4Bi8_gemm > 0) {
    unsigned char* c_a = (unsigned char*)a;
    char* c_b = (char*)b;
    float*  i_c_f = (float*)c;
    libxsmm_bfloat16* i_c_bf16 = (libxsmm_bfloat16*)c;
    libxsmm_blasint l_k_block = i_gemm_def->mxfp4_block_size;
    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        float f32_accum = 0.0;
        int tmp_val = 0;
        if ( i_gemm_def->beta == 0 ) {
          if (i_gemm_def->c_type == LIBXSMM_DATATYPE_F32) {
            i_c_f[(l_j * ldc) + l_i] = 0.0f;
          }
          if (i_gemm_def->c_type == LIBXSMM_DATATYPE_BF16) {
            i_c_bf16[(l_j * ldc) + l_i] = (libxsmm_bfloat16)0;
          }
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          libxsmm_calculate_brgemm_offsets((void**)&c_a, (void**)&c_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            /* Load MXFP4 scaling factor and upconvert to f32 */
            float scf_a = libxsmm_calculate_mxfp4_scf(l_r, l_s, l_i, i_gemm_def);
            float scf_b = libxsmm_calculate_mxfp4_scf_b(l_r, l_s, l_j, i_gemm_def);
            tmp_val = 0;
            for (l_k2 = 0; l_k2 < l_k_block; l_k2 += 8) {
              unsigned char even = 0, odd = 0;
              unsigned char packed_a = 0;
              char even_a = 0, odd_a = 0, even_b = 0, odd_b = 0;

              packed_a = c_a[offs_a + (((long long)l_s*l_k_block+l_k2)/8)*lda*4+l_i*4+0];
              even = packed_a & 0x0f;
              even_a = libxsmm_convert_mxfp4_to_char(even);
              odd = (packed_a >> 4) & 0x0f;
              odd_a = libxsmm_convert_mxfp4_to_char(odd);
              even_b = c_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2 + 0];
              odd_b = c_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2 + 4];
              tmp_val += (int)even_a * even_b + (int)odd_a * odd_b;

              packed_a = c_a[offs_a + (((long long)l_s*l_k_block+l_k2)/8)*lda*4+l_i*4+1];
              even = packed_a & 0x0f;
              even_a = libxsmm_convert_mxfp4_to_char(even);
              odd = (packed_a >> 4) & 0x0f;
              odd_a = libxsmm_convert_mxfp4_to_char(odd);
              even_b = c_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2 + 1];
              odd_b = c_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2 + 5];
              tmp_val += (int)even_a * even_b + (int)odd_a * odd_b;

              packed_a = c_a[offs_a + (((long long)l_s*l_k_block+l_k2)/8)*lda*4+l_i*4+2];
              even = packed_a & 0x0f;
              even_a = libxsmm_convert_mxfp4_to_char(even);
              odd = (packed_a >> 4) & 0x0f;
              odd_a = libxsmm_convert_mxfp4_to_char(odd);
              even_b = c_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2 + 2];
              odd_b = c_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2 + 6];
              tmp_val += (int)even_a * even_b + (int)odd_a * odd_b;

              packed_a = c_a[offs_a + (((long long)l_s*l_k_block+l_k2)/8)*lda*4+l_i*4+3];
              even = packed_a & 0x0f;
              even_a = libxsmm_convert_mxfp4_to_char(even);
              odd = (packed_a >> 4) & 0x0f;
              odd_a = libxsmm_convert_mxfp4_to_char(odd);
              even_b = c_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2 + 3];
              odd_b = c_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2 + 7];
              tmp_val += (int)even_a * even_b + (int)odd_a * odd_b;
            }
            f32_accum += ((float)tmp_val) * scf_a * scf_b;
          }
        }
        if (i_gemm_def->c_type == LIBXSMM_DATATYPE_F32) {
          i_c_f[(l_j * ldc) + l_i] += f32_accum;
        }
        if (i_gemm_def->c_type == LIBXSMM_DATATYPE_BF16) {
          float up_tmp;
          up_tmp = libxsmm_convert_bf16_to_f32(i_c_bf16[(l_j * ldc) + l_i]);
          up_tmp += f32_accum;
          i_c_bf16[(l_j * ldc) + l_i] = libxsmm_convert_f32_to_bf16_rne(up_tmp);
        }
      }
    }
  } else if ((i_gemm_def->is_Ai2Bi8_gemm > 0) && (i_gemm_def->unsigned_b == 0)) {
    char* c_a = (char*)a;
    char* c_b = (char*)b;
    int*  i_c = (int*)c;
    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m/4; l_i++) {
        if ( i_gemm_def->beta == 0 ) {
          long long l_i_m0 = (long long)l_i + 0*(m/4);
          long long l_i_m1 = (long long)l_i + 1*(m/4);
          long long l_i_m2 = (long long)l_i + 2*(m/4);
          long long l_i_m3 = (long long)l_i + 3*(m/4);
          i_c[(l_j * ldc) + l_i_m0] = 0;
          i_c[(l_j * ldc) + l_i_m1] = 0;
          i_c[(l_j * ldc) + l_i_m2] = 0;
          i_c[(l_j * ldc) + l_i_m3] = 0;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          libxsmm_calculate_brgemm_offsets((void**)&c_a, (void**)&c_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / 4); l_s++) {
            long long l_i_m0 = (long long)l_i + 0*(m/4);
            long long l_i_m1 = (long long)l_i + 1*(m/4);
            long long l_i_m2 = (long long)l_i + 2*(m/4);
            long long l_i_m3 = (long long)l_i + 3*(m/4);
            char m0k0 = 0, m1k0 = 0, m2k0 = 0, m3k0 = 0;
            char spacked_char = c_a[offs_a + (l_s * lda)+ l_i * 4 + 0];
            unpack2bit(&m0k0, &m1k0, &m2k0, &m3k0, spacked_char);
            i_c[(l_j * ldc) + l_i_m0] += m0k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 0];
            i_c[(l_j * ldc) + l_i_m1] += m1k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 0];
            i_c[(l_j * ldc) + l_i_m2] += m2k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 0];
            i_c[(l_j * ldc) + l_i_m3] += m3k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 0];

            spacked_char = c_a[offs_a + (l_s * lda)+ l_i * 4 + 1];
            unpack2bit(&m0k0, &m1k0, &m2k0, &m3k0, spacked_char);
            i_c[(l_j * ldc) + l_i_m0] += m0k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 1];
            i_c[(l_j * ldc) + l_i_m1] += m1k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 1];
            i_c[(l_j * ldc) + l_i_m2] += m2k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 1];
            i_c[(l_j * ldc) + l_i_m3] += m3k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 1];

            spacked_char = c_a[offs_a + (l_s * lda)+ l_i * 4 + 2];
            unpack2bit(&m0k0, &m1k0, &m2k0, &m3k0, spacked_char);
            i_c[(l_j * ldc) + l_i_m0] += m0k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 2];
            i_c[(l_j * ldc) + l_i_m1] += m1k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 2];
            i_c[(l_j * ldc) + l_i_m2] += m2k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 2];
            i_c[(l_j * ldc) + l_i_m3] += m3k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 2];

            spacked_char = c_a[offs_a + (l_s * lda)+ l_i * 4 + 3];
            unpack2bit(&m0k0, &m1k0, &m2k0, &m3k0, spacked_char);
            i_c[(l_j * ldc) + l_i_m0] += m0k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 3];
            i_c[(l_j * ldc) + l_i_m1] += m1k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 3];
            i_c[(l_j * ldc) + l_i_m2] += m2k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 3];
            i_c[(l_j * ldc) + l_i_m3] += m3k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 3];
          }
        }
      }
    }
  } else if ((i_gemm_def->is_Ai2Bi8_gemm > 0) && (i_gemm_def->unsigned_b == 1)) {
    char* c_a = (char*)a;
    unsigned char* c_b = (unsigned char*)b;
    int*  i_c = (int*)c;
    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m/4; l_i++) {
        if ( i_gemm_def->beta == 0 ) {
          long long l_i_m0 = (long long)l_i + 0*(m/4);
          long long l_i_m1 = (long long)l_i + 1*(m/4);
          long long l_i_m2 = (long long)l_i + 2*(m/4);
          long long l_i_m3 = (long long)l_i + 3*(m/4);
          i_c[(l_j * ldc) + l_i_m0] = 0;
          i_c[(l_j * ldc) + l_i_m1] = 0;
          i_c[(l_j * ldc) + l_i_m2] = 0;
          i_c[(l_j * ldc) + l_i_m3] = 0;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          libxsmm_calculate_brgemm_offsets((void**)&c_a, (void**)&c_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / 4); l_s++) {
            long long l_i_m0 = (long long)l_i + 0*(m/4);
            long long l_i_m1 = (long long)l_i + 1*(m/4);
            long long l_i_m2 = (long long)l_i + 2*(m/4);
            long long l_i_m3 = (long long)l_i + 3*(m/4);
            char m0k0 = 0, m1k0 = 0, m2k0 = 0, m3k0 = 0;
            char spacked_char = c_a[offs_a + (l_s * lda)+ l_i * 4 + 0];
            unpack2bit(&m0k0, &m1k0, &m2k0, &m3k0, spacked_char);
            i_c[(l_j * ldc) + l_i_m0] += m0k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 0];
            i_c[(l_j * ldc) + l_i_m1] += m1k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 0];
            i_c[(l_j * ldc) + l_i_m2] += m2k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 0];
            i_c[(l_j * ldc) + l_i_m3] += m3k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 0];

            spacked_char = c_a[offs_a + (l_s * lda)+ l_i * 4 + 1];
            unpack2bit(&m0k0, &m1k0, &m2k0, &m3k0, spacked_char);
            i_c[(l_j * ldc) + l_i_m0] += m0k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 1];
            i_c[(l_j * ldc) + l_i_m1] += m1k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 1];
            i_c[(l_j * ldc) + l_i_m2] += m2k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 1];
            i_c[(l_j * ldc) + l_i_m3] += m3k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 1];

            spacked_char = c_a[offs_a + (l_s * lda)+ l_i * 4 + 2];
            unpack2bit(&m0k0, &m1k0, &m2k0, &m3k0, spacked_char);
            i_c[(l_j * ldc) + l_i_m0] += m0k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 2];
            i_c[(l_j * ldc) + l_i_m1] += m1k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 2];
            i_c[(l_j * ldc) + l_i_m2] += m2k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 2];
            i_c[(l_j * ldc) + l_i_m3] += m3k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 2];

            spacked_char = c_a[offs_a + (l_s * lda)+ l_i * 4 + 3];
            unpack2bit(&m0k0, &m1k0, &m2k0, &m3k0, spacked_char);
            i_c[(l_j * ldc) + l_i_m0] += m0k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 3];
            i_c[(l_j * ldc) + l_i_m1] += m1k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 3];
            i_c[(l_j * ldc) + l_i_m2] += m2k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 3];
            i_c[(l_j * ldc) + l_i_m3] += m3k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 3];
          }
        }
      }
    }
  } else if ((i_gemm_def->is_Ai1Bi8_gemm > 0) && (i_gemm_def->unsigned_b == 0)) {
    char* c_a = (char*)a;
    char* c_b = (char*)b;
    int*  i_c = (int*)c;
    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i+=2) {
        if ( i_gemm_def->beta == 0 ) {
          i_c[(l_j * ldc) + l_i + 0] = 0;
          i_c[(l_j * ldc) + l_i + 1] = 0;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          libxsmm_calculate_brgemm_offsets((void**)&c_a, (void**)&c_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / 4); l_s++) {
            char m0k0 = 0, m0k1 = 0, m0k2 = 0, m0k3 = 0, m1k0 = 0, m1k1 = 0, m1k2 = 0, m1k3 = 0;
            char spacked_char = c_a[offs_a + (l_s * lda)/2 + l_i/2];
            m0k0 = ((spacked_char & 0x1) == 0) ? 1 : -1;
            m0k1 = ((spacked_char & 0x2) == 0) ? 1 : -1;
            m0k2 = ((spacked_char & 0x4) == 0) ? 1 : -1;
            m0k3 = ((spacked_char & 0x8) == 0) ? 1 : -1;
            m1k0 = ((spacked_char & 0x10) == 0) ? 1 : -1;
            m1k1 = ((spacked_char & 0x20) == 0) ? 1 : -1;
            m1k2 = ((spacked_char & 0x40) == 0) ? 1 : -1;
            m1k3 = ((spacked_char & 0x80) == 0) ? 1 : -1;

            i_c[(l_j * ldc) + l_i + 0] += m0k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 0];
            i_c[(l_j * ldc) + l_i + 0] += m0k1 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 1];
            i_c[(l_j * ldc) + l_i + 0] += m0k2 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 2];
            i_c[(l_j * ldc) + l_i + 0] += m0k3 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 3];

            i_c[(l_j * ldc) + l_i + 1] += m1k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 0];
            i_c[(l_j * ldc) + l_i + 1] += m1k1 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 1];
            i_c[(l_j * ldc) + l_i + 1] += m1k2 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 2];
            i_c[(l_j * ldc) + l_i + 1] += m1k3 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 3];
          }
        }
      }
    }
  } else if ((i_gemm_def->is_Ai1Bi8_gemm > 0) && (i_gemm_def->unsigned_b == 1)) {
    char* c_a = (char*)a;
    unsigned char* c_b = (unsigned char*)b;
    int*  i_c = (int*)c;
    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i+=2) {
        if ( i_gemm_def->beta == 0 ) {
          i_c[(l_j * ldc) + l_i + 0] = 0;
          i_c[(l_j * ldc) + l_i + 1] = 0;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          libxsmm_calculate_brgemm_offsets((void**)&c_a, (void**)&c_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / 4); l_s++) {
            char m0k0 = 0, m0k1 = 0, m0k2 = 0, m0k3 = 0, m1k0 = 0, m1k1 = 0, m1k2 = 0, m1k3 = 0;
            char spacked_char = c_a[offs_a + (l_s * lda)/2 + l_i/2];
            m0k0 = ((spacked_char & 0x1) == 0) ? 1 : -1;
            m0k1 = ((spacked_char & 0x2) == 0) ? 1 : -1;
            m0k2 = ((spacked_char & 0x4) == 0) ? 1 : -1;
            m0k3 = ((spacked_char & 0x8) == 0) ? 1 : -1;
            m1k0 = ((spacked_char & 0x10) == 0) ? 1 : -1;
            m1k1 = ((spacked_char & 0x20) == 0) ? 1 : -1;
            m1k2 = ((spacked_char & 0x40) == 0) ? 1 : -1;
            m1k3 = ((spacked_char & 0x80) == 0) ? 1 : -1;

            i_c[(l_j * ldc) + l_i + 0] += m0k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 0];
            i_c[(l_j * ldc) + l_i + 0] += m0k1 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 1];
            i_c[(l_j * ldc) + l_i + 0] += m0k2 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 2];
            i_c[(l_j * ldc) + l_i + 0] += m0k3 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 3];

            i_c[(l_j * ldc) + l_i + 1] += m1k0 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 0];
            i_c[(l_j * ldc) + l_i + 1] += m1k1 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 1];
            i_c[(l_j * ldc) + l_i + 1] += m1k2 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 2];
            i_c[(l_j * ldc) + l_i + 1] += m1k3 * c_b[offs_b + (l_j * ldb) + (l_s*4) + 3];
          }
        }
      }
    }
  } else if (i_gemm_def->is_Ai4Bi8_gemm > 0) {
    unsigned char* c_a = (unsigned char*)a;
    unsigned char* c_b = (unsigned char*)b;
    int*           i_c = (int*)c;
    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        if ( i_gemm_def->beta == 0 ) {
          i_c[(l_j * ldc) + l_i] = 0;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          unsigned char zpt = libxsmm_calculate_zpt(l_r, l_i, i_gemm_def);
          libxsmm_calculate_brgemm_offsets((void**)&c_a, (void**)&c_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / 8); l_s++) {
            unsigned int odd = 0, even = 0;
            packed_char = c_a[offs_a + (l_s * lda * 4) + 4 * l_i + 0];
            even = packed_char & 0x0f;
            odd = (packed_char >> 4) & 0x0f;
            even_use = even - zpt;
            odd_use = odd - zpt;
            i_c[(l_j * ldc) + l_i] += even_use * c_b[offs_b + (l_j * ldb) + (l_s*8) + 0];
            i_c[(l_j * ldc) + l_i] += odd_use  * c_b[offs_b + (l_j * ldb) + (l_s*8) + 4];

            packed_char = c_a[offs_a + (l_s * lda * 4) + 4 * l_i + 1];
            even = packed_char & 0x0f;
            odd = (packed_char >> 4) & 0x0f;
            even_use = even - zpt;
            odd_use = odd - zpt;
            i_c[(l_j * ldc) + l_i] += even_use * c_b[offs_b + (l_j * ldb) + (l_s*8) + 1];
            i_c[(l_j * ldc) + l_i] += odd_use  * c_b[offs_b + (l_j * ldb) + (l_s*8) + 5];

            packed_char = c_a[offs_a + (l_s * lda * 4) + 4 * l_i + 2];
            even = packed_char & 0x0f;
            odd = (packed_char >> 4) & 0x0f;
            even_use = even - zpt;
            odd_use = odd - zpt;
            i_c[(l_j * ldc) + l_i] += even_use * c_b[offs_b + (l_j * ldb) + (l_s*8) + 2];
            i_c[(l_j * ldc) + l_i] += odd_use  * c_b[offs_b + (l_j * ldb) + (l_s*8) + 6];

            packed_char = c_a[offs_a + (l_s * lda * 4) + 4 * l_i + 3];
            even = packed_char & 0x0f;
            odd = (packed_char >> 4) & 0x0f;
            even_use = even - zpt;
            odd_use = odd - zpt;
            i_c[(l_j * ldc) + l_i] += even_use * c_b[offs_b + (l_j * ldb) + (l_s*8) + 3];
            i_c[(l_j * ldc) + l_i] += odd_use  * c_b[offs_b + (l_j * ldb) + (l_s*8) + 7];
          }
        }
      }
    }
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_F64) &&
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
          libxsmm_calculate_brgemm_offsets((void**)&d_a, (void**)&d_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < k; l_s++) {
            if (i_gemm_def->trans_b == 0) {
              if (i_gemm_def->trans_a == 0) {
                d_c[(l_j * ldc) + l_i] += d_a[offs_a + (l_s * lda) + l_i] *
                                                   d_b[offs_b + (l_j * ldb) + l_s];
              } else {
                d_c[(l_j * ldc) + l_i] += d_a[offs_a + (l_i * lda) + l_s] *
                                                   d_b[offs_b + (l_j * ldb) + l_s];
              } /* if-else l_trans_a */
            } else {
              if (i_gemm_def->trans_a == 0) {
                d_c[(l_j * ldc) + l_i] += d_a[offs_a + (l_s * lda) + l_i] *
                                                   d_b[offs_b + (l_s * ldb) + l_j];
              } else {
                d_c[(l_j * ldc) + l_i] += d_a[offs_a + (l_i * lda) + l_s] *
                                                   d_b[offs_b + (l_s * ldb) + l_j];
              } /* if-else l_trans_a */
            } /* if-else l_trans_b */
          }
        }
      }
    }
  } else if ( ((i_gemm_def->a_type    == LIBXSMM_DATATYPE_F32) || (i_gemm_def->a_type == LIBXSMM_DATATYPE_BF32)) &&
              ((i_gemm_def->b_type    == LIBXSMM_DATATYPE_F32) || (i_gemm_def->b_type == LIBXSMM_DATATYPE_BF32)) &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_F32) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F32)    ) {
    float* f_a = (float*)a;
    float* f_b = (float*)b;
    float* f_c = (float*)c;
    unsigned int l_cvt_ab_to_bf16 = (i_gemm_def->a_type == LIBXSMM_DATATYPE_BF32) ? 1 : 0;
    float a_val = 0.0f, b_val = 0.0f;
    libxsmm_bfloat16 tmp_bf16;

      for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        if ( i_gemm_def->beta == 0 ) {
          f_c[(l_j * ldc) + l_i] = 0.0f;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          libxsmm_calculate_brgemm_offsets((void**)&f_a, (void**)&f_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < k; l_s++) {
            if (i_gemm_def->trans_b == 0) {
              if (i_gemm_def->trans_a == 0) {
                a_val = f_a[offs_a + (l_s * lda) + l_i];
                b_val = f_b[offs_b + (l_j * ldb) + l_s];
                if (l_cvt_ab_to_bf16 == 1) {
                  libxsmm_rne_convert_fp32_bf16(&a_val, &tmp_bf16, 1);
                  libxsmm_convert_bf16_f32(&tmp_bf16, &a_val, 1);
                  libxsmm_rne_convert_fp32_bf16(&b_val, &tmp_bf16, 1);
                  libxsmm_convert_bf16_f32(&tmp_bf16, &b_val, 1);
                }
                f_c[(l_j * ldc) + l_i] += a_val * b_val;
              } else {
                a_val = f_a[offs_a + (l_i * lda) + l_s];
                b_val = f_b[offs_b + (l_j * ldb) + l_s];
                if (l_cvt_ab_to_bf16 == 1) {
                  libxsmm_rne_convert_fp32_bf16(&a_val, &tmp_bf16, 1);
                  libxsmm_convert_bf16_f32(&tmp_bf16, &a_val, 1);
                  libxsmm_rne_convert_fp32_bf16(&b_val, &tmp_bf16, 1);
                  libxsmm_convert_bf16_f32(&tmp_bf16, &b_val, 1);
                }
                f_c[(l_j * ldc) + l_i] += a_val * b_val;
              } /* if-else l_trans_a */
            } else {
              if (i_gemm_def->trans_a == 0) {
                a_val = f_a[offs_a + (l_s * lda) + l_i];
                b_val = f_b[offs_b + (l_s * ldb) + l_j];
                if (l_cvt_ab_to_bf16 == 1) {
                  libxsmm_rne_convert_fp32_bf16(&a_val, &tmp_bf16, 1);
                  libxsmm_convert_bf16_f32(&tmp_bf16, &a_val, 1);
                  libxsmm_rne_convert_fp32_bf16(&b_val, &tmp_bf16, 1);
                  libxsmm_convert_bf16_f32(&tmp_bf16, &b_val, 1);
                }
                f_c[(l_j * ldc) + l_i] += a_val * b_val;
              } else {
                a_val = f_a[offs_a + (l_i * lda) + l_s];
                b_val = f_b[offs_b + (l_s * ldb) + l_j];
                if (l_cvt_ab_to_bf16 == 1) {
                  libxsmm_rne_convert_fp32_bf16(&a_val, &tmp_bf16, 1);
                  libxsmm_convert_bf16_f32(&tmp_bf16, &a_val, 1);
                  libxsmm_rne_convert_fp32_bf16(&b_val, &tmp_bf16, 1);
                  libxsmm_convert_bf16_f32(&tmp_bf16, &b_val, 1);
                }
                f_c[(l_j * ldc) + l_i] += a_val * b_val;
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
    libxsmm_blasint l_k_block = (i_gemm_def->vnni_a != 0) ? libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type) : 1;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        if ( i_gemm_def->beta == 0 ) {
          i_c[(l_j * ldc) + l_i] = 0;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          libxsmm_calculate_brgemm_offsets((void**)&s_a, (void**)&s_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              i_c[(l_j * ldc) + l_i] += s_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2] *
                                        s_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2];
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
    libxsmm_blasint l_k_block = (i_gemm_def->vnni_a != 0) ? libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type) : 1;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        if ( i_gemm_def->beta == 0 ) {
          i_c[(l_j * ldc) + l_i] = 0;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          libxsmm_calculate_brgemm_offsets((void**)&c_a, (void**)&c_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              i_c[(l_j * ldc) + l_i] += c_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2] *
                                        c_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2];
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
    libxsmm_blasint l_k_block = (i_gemm_def->vnni_a != 0) ? libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type) : 1;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        if ( i_gemm_def->beta == 0 ) {
          i_c[(l_j * ldc) + l_i] = 0;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          libxsmm_calculate_brgemm_offsets((void**)&c_a, (void**)&c_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              i_c[(l_j * ldc) + l_i] += c_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2] *
                                        c_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2];
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
    libxsmm_blasint l_k_block = (i_gemm_def->vnni_a != 0) ? libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type) : 1;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        if ( i_gemm_def->beta == 0 ) {
          i_c[(l_j * ldc) + l_i] = 0;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          libxsmm_calculate_brgemm_offsets((void**)&c_a, (void**)&c_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              i_c[(l_j * ldc) + l_i] += c_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2] *
                                        c_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2];
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
    libxsmm_blasint l_k_block = (i_gemm_def->vnni_a != 0) ? libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type) : 1;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        if ( i_gemm_def->beta == 0 ) {
          i_c[(l_j * ldc) + l_i] = 0;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          libxsmm_calculate_brgemm_offsets((void**)&c_a, (void**)&c_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              i_c[(l_j * ldc) + l_i] += c_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2] *
                                        c_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2];
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
    libxsmm_blasint l_k_block = libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type);

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        int tmp = 0;
        float ftmp;
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          libxsmm_calculate_brgemm_offsets((void**)&c_a, (void**)&c_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              tmp += c_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2] *
                     c_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2];
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
    libxsmm_blasint l_k_block = libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type);

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        int tmp = 0;
        float ftmp;
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          libxsmm_calculate_brgemm_offsets((void**)&c_a, (void**)&c_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              tmp += c_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2] *
                     c_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2];
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
    libxsmm_blasint l_k_block = libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type);

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        int tmp = 0;
        float ftmp;
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          libxsmm_calculate_brgemm_offsets((void**)&c_a, (void**)&c_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              tmp += c_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2] *
                     c_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2];
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
    libxsmm_blasint l_k_block = libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type);

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        int tmp = 0;
        float ftmp;
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          libxsmm_calculate_brgemm_offsets((void**)&c_a, (void**)&c_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              tmp += c_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2] *
                     c_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2];
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
    libxsmm_blasint l_k_block = 1;
    libxsmm_bfloat16 tmp_bf16;
    float up_c;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        float ftmp = 0.0;
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          libxsmm_calculate_brgemm_offsets((void**)&c_a, (void**)&bf16_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              float a_use, b_use;
              char char_a = c_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              int int_a = (int) char_a;
              a_use = (float) int_a;
              a_use = a_use * i_gemm_def->scf_f32[l_i];
              tmp_bf16 = libxsmm_convert_f32_to_bf16_rne(a_use);
              a_use = libxsmm_convert_bf16_to_f32(tmp_bf16);
              tmp_bf16 = bf16_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2];
              b_use = libxsmm_convert_bf16_to_f32(tmp_bf16);
              ftmp += a_use * b_use;
            }
          }
        }
        if ( i_gemm_def->c_type    == LIBXSMM_DATATYPE_BF16 ) {
          if ( i_gemm_def->beta == 1 ) {
            tmp_bf16 = bf16_c[(l_j * ldc) + l_i];
            up_c = libxsmm_convert_bf16_to_f32(tmp_bf16);
            ftmp += up_c;
          }
          tmp_bf16 = libxsmm_convert_f32_to_bf16_rne(ftmp);
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
    libxsmm_blasint l_k_block = (i_gemm_def->vnni_a != 0) ? libxsmm_cpuid_dot_pack_factor(i_gemm_def->b_type) : 1;
    const char* env_arch = getenv("LIBXSMM_TARGET");
    const int is_env_SPR = (env_arch == NULL) ? 0 : (
      env_arch == strstr(env_arch, "spr") ||
      env_arch == strstr(env_arch, "amx"));
    int arch_cpuid = libxsmm_cpuid(NULL);
    int l_is_gt_spr = (is_env_SPR > 0) ? 1 : ((env_arch == NULL && arch_cpuid >= LIBXSMM_X86_AVX512_SPR) ? 1 : 0);
    int l_use_replacement_fma = (((i_gemm_def->comp_type == LIBXSMM_DATATYPE_F16) || (i_gemm_def->comp_type == LIBXSMM_DATATYPE_IMPLICIT && l_is_gt_spr > 0))) ? 1 : 0;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        float ftmp = 0.0, b_use;
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          libxsmm_calculate_brgemm_offsets((void**)&bf8_a, (void**)&f16_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              union libxsmm_bfloat8_f16 tmp_a_hf;
              float tmp_a_f;
              tmp_a_hf.i[0] = 0;
              tmp_a_hf.i[1] = bf8_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              tmp_a_f = libxsmm_convert_f16_to_f32( tmp_a_hf.hf );
              if (i_gemm_def->trans_b == 0) {
                cur_b = f16_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2];
              } else {
                cur_b = f16_b[offs_b + (l_s*l_k_block+l_k2) * (long long)ldb + l_j];
              }
              b_use = libxsmm_convert_f16_to_f32(cur_b);
              ftmp += tmp_a_f * b_use;
              if (l_use_replacement_fma > 0) {
                c_tmp = libxsmm_convert_f32_to_f16(ftmp);
                ftmp = libxsmm_convert_f16_to_f32(c_tmp);
              }
            }
          }
        }
        if ( i_gemm_def->c_type    == LIBXSMM_DATATYPE_F16 ) {
          if ( i_gemm_def->beta == 1 ) {
            c_tmp = f16_c[(l_j * ldc) + l_i];
            up_c = libxsmm_convert_f16_to_f32(c_tmp);
            ftmp += up_c;
          }
          c_tmp = libxsmm_convert_f32_to_f16(ftmp);
          f16_c[(l_j * ldc) + l_i] = c_tmp;
        } else {
          if ( i_gemm_def->beta == 1 ) {
            ftmp += f32_c[(l_j * ldc) + l_i];
          }
          f32_c[(l_j * ldc) + l_i] = ftmp;
        }
      }
    }
  } else if (i_gemm_def->is_Ai4Bf16_gemm > 0) {
    char* c_a = (char*)a;
    libxsmm_float16* f16_c = (libxsmm_float16*)c;
    libxsmm_float16* f16_b = (libxsmm_float16*)b;
    libxsmm_float16 c_tmp;
    libxsmm_float16 cur_a, cur_b;
    float* f32_c = (float*)c;
    char char_a;
    short short_a;
    int int_a;
    float up_c;
    libxsmm_blasint l_k_block = 2;
    const char* env_arch = getenv("LIBXSMM_TARGET");
    const int is_env_SPR = (env_arch == NULL) ? 0 : (
      env_arch == strstr(env_arch, "spr") ||
      env_arch == strstr(env_arch, "amx"));
    int arch_cpuid = libxsmm_cpuid(NULL);
    int l_is_gt_spr = (is_env_SPR > 0) ? 1 : ((env_arch == NULL && arch_cpuid >= LIBXSMM_X86_AVX512_SPR) ? 1 : 0);
    int l_use_replacement_fma = (((i_gemm_def->comp_type == LIBXSMM_DATATYPE_F16) || (i_gemm_def->comp_type == LIBXSMM_DATATYPE_IMPLICIT && l_is_gt_spr > 0))) ? 1 : 0;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        float ftmp = 0.0, a_use, b_use;
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          libxsmm_calculate_brgemm_offsets((void**)&c_a, (void**)&f16_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              char _char_a = c_a[offs_a + (l_s * lda) + l_i];
              if ((i_gemm_def->a_type == LIBXSMM_DATATYPE_I8 || i_gemm_def->a_type == LIBXSMM_DATATYPE_I4X2) && i_gemm_def->unsigned_a == 0) {
                char_a = (l_k2 == 0) ? ((char)((_char_a & 0x0f) << 4)) >> 4 : ((char)((_char_a & 0xf0)) >> 4);
              } else {
                char_a = (l_k2 == 0) ? _char_a & 0x0f : ((_char_a & 0xf0) >> 4) & 0x0f;
              }
              short_a = (short) char_a;
              int_a = (int) char_a;
              if (l_use_replacement_fma > 0) {
                a_use = (float) short_a;
                cur_a = libxsmm_convert_f32_to_f16(a_use);
                a_use = libxsmm_convert_f16_to_f32(cur_a);
              } else {
                a_use = (float) int_a;
              }
              if (i_gemm_def->fuse_zpt_sub > 0) {
                float zptf32 = libxsmm_convert_f16_to_f32(i_gemm_def->zpt_f16[l_i]);
                a_use = a_use - zptf32;
                if (l_use_replacement_fma > 0) {
                  cur_a = libxsmm_convert_f32_to_f16(a_use);
                  a_use = libxsmm_convert_f16_to_f32(cur_a);
                }
              }
              a_use = a_use * libxsmm_convert_f16_to_f32(i_gemm_def->scf_f16[l_i]);
              if (l_use_replacement_fma > 0) {
                c_tmp = libxsmm_convert_f32_to_f16(a_use);
                a_use = libxsmm_convert_f16_to_f32(c_tmp);
              }
              if (i_gemm_def->trans_b == 0) {
                cur_b = f16_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2];
              } else {
                cur_b = f16_b[offs_b + ((long long)(l_s*l_k_block+l_k2)) * ldb + l_j];
              }
              b_use = libxsmm_convert_f16_to_f32(cur_b);
              ftmp += a_use * b_use;
              if (l_use_replacement_fma > 0) {
                c_tmp = libxsmm_convert_f32_to_f16(ftmp);
                ftmp = libxsmm_convert_f16_to_f32(c_tmp);
              }
            }
          }
        }
        if (i_gemm_def->c_type    == LIBXSMM_DATATYPE_F16) {
          if ( i_gemm_def->beta == 1 ) {
            c_tmp = f16_c[(l_j * ldc) + l_i];
            up_c = libxsmm_convert_f16_to_f32(c_tmp);
            ftmp += up_c;
          }
          c_tmp = libxsmm_convert_f32_to_f16(ftmp);
          f16_c[(l_j * ldc) + l_i] = c_tmp;
        } else {
          if ( i_gemm_def->beta == 1 ) {
            up_c = f32_c[(l_j * ldc) + l_i];
            c_tmp = libxsmm_convert_f32_to_f16(up_c);
            up_c = libxsmm_convert_f16_to_f32(c_tmp);
            ftmp += up_c;
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
    libxsmm_blasint l_k_block = 1;
    const char* env_arch = getenv("LIBXSMM_TARGET");
    const int is_env_SPR = (env_arch == NULL) ? 0 : (
      env_arch == strstr(env_arch, "spr") ||
      env_arch == strstr(env_arch, "amx"));
    int arch_cpuid = libxsmm_cpuid(NULL);
    int l_is_gt_spr = (is_env_SPR > 0) ? 1 : ((env_arch == NULL && arch_cpuid >= LIBXSMM_X86_AVX512_SPR) ? 1 : 0);
    int l_use_replacement_fma = (((i_gemm_def->comp_type == LIBXSMM_DATATYPE_F16) || (i_gemm_def->comp_type == LIBXSMM_DATATYPE_IMPLICIT && l_is_gt_spr > 0))) ? 1 : 0;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        float ftmp = 0.0, a_use, b_use;
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          libxsmm_calculate_brgemm_offsets((void**)&c_a, (void**)&f16_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              char char_a = c_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              short short_a = (short) char_a;
              int int_a = (int) char_a;
              if (l_use_replacement_fma > 0) {
                a_use = (float) short_a;
                cur_a = libxsmm_convert_f32_to_f16(a_use);
                a_use = libxsmm_convert_f16_to_f32(cur_a);
              } else {
                a_use = (float) int_a;
              }
              if (i_gemm_def->fuse_zpt_sub > 0) {
                float zptf32 = libxsmm_convert_f16_to_f32(i_gemm_def->zpt_f16[l_i]);
                a_use = a_use - zptf32;
                if (l_use_replacement_fma > 0) {
                  cur_a = libxsmm_convert_f32_to_f16(a_use);
                  a_use = libxsmm_convert_f16_to_f32(cur_a);
                }
              }
              a_use = a_use * libxsmm_convert_f16_to_f32(i_gemm_def->scf_f16[l_i]);
              if (l_use_replacement_fma > 0) {
                c_tmp = libxsmm_convert_f32_to_f16(a_use);
                a_use = libxsmm_convert_f16_to_f32(c_tmp);
              }
              if (i_gemm_def->trans_b == 0) {
                cur_b = f16_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2];
              } else {
                cur_b = f16_b[offs_b + (long long)(l_s*l_k_block+l_k2) * ldb + l_j];
              }
              b_use = libxsmm_convert_f16_to_f32(cur_b);
              ftmp += a_use * b_use;
              if (l_use_replacement_fma > 0) {
                c_tmp = libxsmm_convert_f32_to_f16(ftmp);
                ftmp = libxsmm_convert_f16_to_f32(c_tmp);
              }
            }
          }
        }
        if ( i_gemm_def->beta == 1 ) {
          c_tmp = f16_c[(l_j * ldc) + l_i];
          up_c = libxsmm_convert_f16_to_f32(c_tmp);
          ftmp += up_c;
        }
        c_tmp = libxsmm_convert_f32_to_f16(ftmp);
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
    libxsmm_blasint l_k_block = 1;
    const char* env_arch = getenv("LIBXSMM_TARGET");
    const int is_env_SPR = (env_arch == NULL) ? 0 : (
      env_arch == strstr(env_arch, "spr") ||
      env_arch == strstr(env_arch, "amx"));
    int arch_cpuid = libxsmm_cpuid(NULL);
    int l_is_gt_spr = (is_env_SPR > 0) ? 1 : ((env_arch == NULL && arch_cpuid >= LIBXSMM_X86_AVX512_SPR) ? 1 : 0);
    int l_use_replacement_fma = (((i_gemm_def->comp_type == LIBXSMM_DATATYPE_F16) || (i_gemm_def->comp_type == LIBXSMM_DATATYPE_IMPLICIT && l_is_gt_spr > 0))) ? 1 : 0;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        float ftmp = 0.0, a_use, b_use;
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          libxsmm_calculate_brgemm_offsets((void**)&c_a, (void**)&f16_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              char char_a = c_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              short short_a = (short) char_a;
              int int_a = (int) char_a;
              if (l_use_replacement_fma > 0) {
                a_use = (float) short_a;
                cur_a = libxsmm_convert_f32_to_f16(a_use);
                a_use = libxsmm_convert_f16_to_f32(cur_a);
              } else {
                a_use = (float) int_a;
              }
              if (i_gemm_def->fuse_zpt_sub > 0) {
                float zptf32 = libxsmm_convert_f16_to_f32(i_gemm_def->zpt_f16[l_i]);
                a_use = a_use - zptf32;
                if (l_use_replacement_fma > 0) {
                  cur_a = libxsmm_convert_f32_to_f16(a_use);
                  a_use = libxsmm_convert_f16_to_f32(cur_a);
                }
              }
              a_use = a_use * libxsmm_convert_f16_to_f32(i_gemm_def->scf_f16[l_i]);
              if (l_use_replacement_fma > 0) {
                c_tmp = libxsmm_convert_f32_to_f16(a_use);
                a_use = libxsmm_convert_f16_to_f32(c_tmp);
              }
              if (i_gemm_def->trans_b == 0) {
                cur_b = f16_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2];
              } else {
                cur_b = f16_b[offs_b + ((long long)l_s*l_k_block+l_k2) * ldb + l_j];
              }
              b_use = libxsmm_convert_f16_to_f32(cur_b);
              ftmp += a_use * b_use;
              if (l_use_replacement_fma > 0) {
                c_tmp = libxsmm_convert_f32_to_f16(ftmp);
                ftmp = libxsmm_convert_f16_to_f32(c_tmp);
              }
            }
          }
        }
        if ( i_gemm_def->beta == 1 ) {
          c_tmp_f32 = f32_c[(l_j * ldc) + l_i];
          c_tmp = libxsmm_convert_f32_to_f16(c_tmp_f32);
          c_tmp_f32 = libxsmm_convert_f16_to_f32(c_tmp);
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
    libxsmm_blasint l_k_block = ( i_gemm_def->vnni_a != 0) ? libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type) : 1;
    const char* env_arch = getenv("LIBXSMM_TARGET");
    const int is_env_SPR = (env_arch == NULL) ? 0 : (
      env_arch == strstr(env_arch, "spr") ||
      env_arch == strstr(env_arch, "amx"));
    int arch_cpuid = libxsmm_cpuid(NULL);
    int l_is_gt_spr = (is_env_SPR > 0) ? 1 : ((env_arch == NULL && arch_cpuid >= LIBXSMM_X86_AVX512_SPR) ? 1 : 0);
    int l_use_replacement_fma = (((i_gemm_def->comp_type == LIBXSMM_DATATYPE_F16) || (i_gemm_def->comp_type == LIBXSMM_DATATYPE_IMPLICIT && l_is_gt_spr > 0))) ? 1 : 0;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        float ftmp = 0.0, a_use, b_use;
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          libxsmm_calculate_brgemm_offsets((void**)&f16_a, (void**)&f16_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              cur_a = f16_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              a_use = libxsmm_convert_f16_to_f32(cur_a);
              if (i_gemm_def->trans_b == 0) {
                cur_b = f16_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2];
              } else {
                cur_b = f16_b[offs_b + ((long long)l_s*l_k_block+l_k2) * (long long)ldb + l_j];
              }
              b_use = libxsmm_convert_f16_to_f32(cur_b);
              ftmp += a_use * b_use;
              if (l_use_replacement_fma > 0) {
                c_tmp = libxsmm_convert_f32_to_f16(ftmp);
                ftmp = libxsmm_convert_f16_to_f32(c_tmp);
              }
            }
          }
        }
        if ( i_gemm_def->beta == 1 ) {
          c_tmp = f16_c[(l_j * ldc) + l_i];
          up_c = libxsmm_convert_f16_to_f32(c_tmp);
          ftmp += up_c;
        }
        c_tmp = libxsmm_convert_f32_to_f16(ftmp);
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
    libxsmm_blasint l_k_block = ( i_gemm_def->vnni_a != 0) ? libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type) : 1;
    const char* env_arch = getenv("LIBXSMM_TARGET");
    const int is_env_SPR = (env_arch == NULL) ? 0 : (
      env_arch == strstr(env_arch, "spr") ||
      env_arch == strstr(env_arch, "amx"));
    int arch_cpuid = libxsmm_cpuid(NULL);
    int l_is_gt_spr = (is_env_SPR > 0) ? 1 : ((env_arch == NULL && arch_cpuid >= LIBXSMM_X86_AVX512_SPR) ? 1 : 0);
    int l_use_replacement_fma = (((i_gemm_def->comp_type == LIBXSMM_DATATYPE_F16) || (i_gemm_def->comp_type == LIBXSMM_DATATYPE_IMPLICIT && l_is_gt_spr > 0))) ? 1 : 0;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        float ftmp = 0.0, a_use, b_use;
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          libxsmm_calculate_brgemm_offsets((void**)&f16_a, (void**)&f16_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              cur_a = f16_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              a_use = libxsmm_convert_f16_to_f32(cur_a);
              if (i_gemm_def->trans_b == 0) {
                cur_b = f16_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2];
              } else {
                cur_b = f16_b[offs_b + ((long long)l_s*l_k_block+(long long)l_k2) * ldb + l_j];
              }
              b_use = libxsmm_convert_f16_to_f32(cur_b);
              ftmp += a_use * b_use;
              if (l_use_replacement_fma > 0) {
                c_tmp = libxsmm_convert_f32_to_f16(ftmp);
                ftmp = libxsmm_convert_f16_to_f32(c_tmp);
              }
            }
          }
        }
        if ( i_gemm_def->beta == 1 ) {
          c_tmp_f32 = f32_c[(l_j * ldc) + l_i];
          c_tmp = libxsmm_convert_f32_to_f16(c_tmp_f32);
          c_tmp_f32 = libxsmm_convert_f16_to_f32(c_tmp);
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
    libxsmm_blasint l_k_block = ( i_gemm_def->vnni_a != 0) ? libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type) : 1;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        if ( i_gemm_def->beta == 0 ) {
          f_c[(l_j * ldc) + l_i] = 0.0f;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          libxsmm_calculate_brgemm_offsets((void**)&h_a, (void**)&h_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = l_k_block - 1; l_k2 >= 0; l_k2--) {
              union libxsmm_bfloat16_f32 tmp_a_f;
              union libxsmm_bfloat16_f32 tmp_b_f;
              tmp_a_f.i[0] = 0;
              if ( (i_gemm_def->trans_a == 0) ) {
                tmp_a_f.i[1] = h_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              } else if ( (i_gemm_def->trans_a > 0) && ( i_gemm_def->vnni_a == 0) ) {
                tmp_a_f.i[1] = h_a[offs_a + (l_i * lda) + (l_s*l_k_block) + l_k2];
              } else {
                /* should happen */
              }
              tmp_b_f.i[0] = 0;
              if ((i_gemm_def->trans_b > 0) && (i_gemm_def->vnni_b > 0)) {
                tmp_b_f.i[1] = h_b[offs_b + (l_j * l_k_block) + (l_s * (ldb*l_k_block)) + l_k2];
              } else if ( (i_gemm_def->trans_b > 0) && (i_gemm_def->vnni_b == 0) ) {
                tmp_b_f.i[1] = h_b[offs_b + (((l_s*l_k_block) + l_k2) * ldb) + l_j];
              } else if ( (i_gemm_def->trans_b == 0) && (i_gemm_def->vnni_b == 0 ) ) {
                tmp_b_f.i[1] = h_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2];
              } else {
                /* should not happen */
              }
              f_c[(l_j * ldc) + l_i] += tmp_a_f.f * tmp_b_f.f;
            }
          }
        }
      }
    }
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_BF8) &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_BF16) &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_F32)  &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F32)     ) {
    libxsmm_bfloat8* h_a = (libxsmm_bfloat8*)a;
    libxsmm_bfloat16* h_b = (libxsmm_bfloat16*)b;
    float*            f_c = (float*)c;
    libxsmm_blasint l_k_block = ( i_gemm_def->vnni_a != 0) ? libxsmm_cpuid_dot_pack_factor(i_gemm_def->b_type) : 1;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        if ( i_gemm_def->beta == 0 ) {
          f_c[(l_j * ldc) + l_i] = 0.0f;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          libxsmm_calculate_brgemm_offsets((void**)&h_a, (void**)&h_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = l_k_block - 1; l_k2 >= 0; l_k2--) {
              union libxsmm_bfloat8_f16 tmp_a_hf;
              float tmp_a_f;
              union libxsmm_bfloat16_f32 tmp_b_f;
              tmp_a_hf.i[0] = 0;
              if ( (i_gemm_def->trans_a == 0) ) {
                tmp_a_hf.i[1] = h_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              } else if ( (i_gemm_def->trans_a > 0) && ( i_gemm_def->vnni_a == 0) ) {
                tmp_a_hf.i[1] = h_a[offs_a + (l_i * lda) + (l_s*l_k_block) + l_k2];
              } else {
                /* should happen */
              }
              tmp_a_f = libxsmm_convert_f16_to_f32( tmp_a_hf.hf );
              tmp_b_f.i[0] = 0;
              if ((i_gemm_def->trans_b > 0) && (i_gemm_def->vnni_b > 0)) {
                tmp_b_f.i[1] = h_b[offs_b + (l_j * l_k_block) + (l_s * (ldb*l_k_block)) + l_k2];
              } else if ( (i_gemm_def->trans_b > 0) && (i_gemm_def->vnni_b == 0) ) {
                tmp_b_f.i[1] = h_b[offs_b + (((l_s*l_k_block) + l_k2) * ldb) + l_j];
              } else if ( (i_gemm_def->trans_b == 0) && (i_gemm_def->vnni_b == 0 ) ) {
                tmp_b_f.i[1] = h_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2];
              } else {
                /* should not happen */
              }
              f_c[(l_j * ldc) + l_i] += tmp_a_f * tmp_b_f.f;
            }
          }
        }
      }
    }
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_HF8) &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_BF16) &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_F32)  &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F32)     ) {
    libxsmm_hfloat8* h_a = (libxsmm_hfloat8*)a;
    libxsmm_bfloat16* h_b = (libxsmm_bfloat16*)b;
    float*            f_c = (float*)c;
    libxsmm_blasint l_k_block = ( i_gemm_def->vnni_a != 0) ? libxsmm_cpuid_dot_pack_factor(i_gemm_def->b_type) : 1;

    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        if ( i_gemm_def->beta == 0 ) {
          f_c[(l_j * ldc) + l_i] = 0.0f;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          libxsmm_calculate_brgemm_offsets((void**)&h_a, (void**)&h_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = l_k_block - 1; l_k2 >= 0; l_k2--) {
              float tmp_a_f = 0.0;
              union libxsmm_bfloat16_f32 tmp_b_f;
              if ( (i_gemm_def->trans_a == 0) ) {
                tmp_a_f = libxsmm_convert_hf8_to_f32(h_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2]);
              } else if ( (i_gemm_def->trans_a > 0) && ( i_gemm_def->vnni_a == 0) ) {
                tmp_a_f = libxsmm_convert_hf8_to_f32(h_a[offs_a + (l_i * lda) + (l_s*l_k_block) + l_k2]);
              } else {
                /* should happen */
              }
              tmp_b_f.i[0] = 0;
              if ((i_gemm_def->trans_b > 0) && (i_gemm_def->vnni_b > 0)) {
                tmp_b_f.i[1] = h_b[offs_b + (l_j * l_k_block) + (l_s * (ldb*l_k_block)) + l_k2];
              } else if ( (i_gemm_def->trans_b > 0) && (i_gemm_def->vnni_b == 0) ) {
                tmp_b_f.i[1] = h_b[offs_b + (((l_s*l_k_block) + l_k2) * ldb) + l_j];
              } else if ( (i_gemm_def->trans_b == 0) && (i_gemm_def->vnni_b == 0 ) ) {
                tmp_b_f.i[1] = h_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2];
              } else {
                /* should not happen */
              }
              f_c[(l_j * ldc) + l_i] += tmp_a_f * tmp_b_f.f;
            }
          }
        }
      }
    }
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_HF8) &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_BF16) &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_BF16) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F32)     ) {
    libxsmm_hfloat8* h_a = (libxsmm_hfloat8*)a;
    libxsmm_bfloat16* h_b = (libxsmm_bfloat16*)b;
    libxsmm_bfloat16* h_c = (libxsmm_bfloat16*)c;
    libxsmm_blasint l_k_block = ( i_gemm_def->vnni_a != 0) ? libxsmm_cpuid_dot_pack_factor(i_gemm_def->b_type) : 1;
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
          libxsmm_calculate_brgemm_offsets((void**)&h_a, (void**)&h_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = l_k_block - 1; l_k2 >= 0; l_k2--) {
              float tmp_a_f = 0.0;
              union libxsmm_bfloat16_f32 tmp_b_f;
              if ( (i_gemm_def->trans_a == 0) ) {
                tmp_a_f = libxsmm_convert_hf8_to_f32(h_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2]);
              } else if ( (i_gemm_def->trans_a > 0) && ( i_gemm_def->vnni_a == 0) ) {
                tmp_a_f = libxsmm_convert_hf8_to_f32(h_a[offs_a + (l_i * lda) + (l_s*l_k_block) + l_k2]);
              } else {
                /* should happen */
              }
              tmp_b_f.i[0] = 0;
              if ((i_gemm_def->trans_b > 0) && (i_gemm_def->vnni_b > 0)) {
                tmp_b_f.i[1] = h_b[offs_b + (l_j * l_k_block) + (l_s * (ldb*l_k_block)) + l_k2];
              } else if ( (i_gemm_def->trans_b > 0) && (i_gemm_def->vnni_b == 0) ) {
                tmp_b_f.i[1] = h_b[offs_b + (((l_s*l_k_block) + l_k2) * ldb) + l_j];
              } else if ( (i_gemm_def->trans_b == 0) && (i_gemm_def->vnni_b == 0 ) ) {
                tmp_b_f.i[1] = h_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2];
              } else {
                /* should not happen */
              }
              acc += tmp_a_f * tmp_b_f.f;
            }
          }
        }
        h_acc = libxsmm_convert_f32_to_bf16_rne(acc);
        h_c[(l_j * ldc) + l_i] = h_acc;
      }
    }
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_BF8) &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_BF16) &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_BF16) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F32)     ) {
    libxsmm_bfloat8* h_a = (libxsmm_bfloat8*)a;
    libxsmm_bfloat16* h_b = (libxsmm_bfloat16*)b;
    libxsmm_bfloat16* h_c = (libxsmm_bfloat16*)c;
    libxsmm_blasint l_k_block = ( i_gemm_def->vnni_a != 0) ? libxsmm_cpuid_dot_pack_factor(i_gemm_def->b_type) : 1;
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
          libxsmm_calculate_brgemm_offsets((void**)&h_a, (void**)&h_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = l_k_block - 1; l_k2 >= 0; l_k2--) {
              union libxsmm_bfloat8_f16 tmp_a_hf;
              float tmp_a_f;
              union libxsmm_bfloat16_f32 tmp_b_f;
              tmp_a_hf.i[0] = 0;
              if ( (i_gemm_def->trans_a == 0) ) {
                tmp_a_hf.i[1] = h_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              } else if ( (i_gemm_def->trans_a > 0) && ( i_gemm_def->vnni_a == 0) ) {
                tmp_a_hf.i[1] = h_a[offs_a + (l_i * lda) + (l_s*l_k_block) + l_k2];
              } else {
                /* should happen */
              }
              tmp_a_f = libxsmm_convert_f16_to_f32( tmp_a_hf.hf );
              tmp_b_f.i[0] = 0;
              if ((i_gemm_def->trans_b > 0) && (i_gemm_def->vnni_b > 0)) {
                tmp_b_f.i[1] = h_b[offs_b + (l_j * l_k_block) + (l_s * (ldb*l_k_block)) + l_k2];
              } else if ( (i_gemm_def->trans_b > 0) && (i_gemm_def->vnni_b == 0) ) {
                tmp_b_f.i[1] = h_b[offs_b + (((l_s*l_k_block) + l_k2) * ldb) + l_j];
              } else if ( (i_gemm_def->trans_b == 0) && (i_gemm_def->vnni_b == 0 ) ) {
                tmp_b_f.i[1] = h_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2];
              } else {
                /* should not happen */
              }
              acc += tmp_a_f * tmp_b_f.f;
            }
          }
        }
        h_acc = libxsmm_convert_f32_to_bf16_rne(acc);
        h_c[(l_j * ldc) + l_i] = h_acc;
      }
    }
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_BF16) &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_BF16) &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_BF16) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F32)     ) {
    libxsmm_bfloat16* h_a = (libxsmm_bfloat16*)a;
    libxsmm_bfloat16* h_b = (libxsmm_bfloat16*)b;
    libxsmm_bfloat16* h_c = (libxsmm_bfloat16*)c;
    libxsmm_blasint l_k_block = ( i_gemm_def->vnni_a != 0) ? libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type) : 1;
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
          libxsmm_calculate_brgemm_offsets((void**)&h_a, (void**)&h_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = l_k_block - 1; l_k2 >= 0; l_k2--) {
              union libxsmm_bfloat16_f32 tmp_a_f;
              union libxsmm_bfloat16_f32 tmp_b_f;
              tmp_a_f.i[0] = 0;
              if ( (i_gemm_def->trans_a == 0) ) {
                tmp_a_f.i[1] = h_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              } else if ( (i_gemm_def->trans_a > 0) && ( i_gemm_def->vnni_a == 0) ) {
                tmp_a_f.i[1] = h_a[offs_a + (l_i * lda) + (l_s*l_k_block) + l_k2];
              } else {
                /* should happen */
              }
              tmp_b_f.i[0] = 0;
              if ((i_gemm_def->trans_b > 0) && (i_gemm_def->vnni_b > 0)) {
                tmp_b_f.i[1] = h_b[offs_b + (l_j * l_k_block) + (l_s * (ldb*l_k_block)) + l_k2];
              } else if ( (i_gemm_def->trans_b > 0) && (i_gemm_def->vnni_b == 0) ) {
                tmp_b_f.i[1] = h_b[offs_b + (((l_s*l_k_block) + l_k2) * ldb) + l_j];
              } else if ( (i_gemm_def->trans_b == 0) && (i_gemm_def->vnni_b == 0 ) ) {
                tmp_b_f.i[1] = h_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2];
              } else {
                /* should not happen */
              }
              acc += tmp_a_f.f * tmp_b_f.f;
            }
          }
        }
        h_acc = libxsmm_convert_f32_to_bf16_rne(acc);
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
    libxsmm_blasint l_k_block = ( i_gemm_def->vnni_a != 0) ? libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type) : 1;
    for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
      libxsmm_calculate_brgemm_offsets((void**)&h_a, (void**)&h_b, &offs_a, &offs_b, l_r, i_gemm_def);
      for (l_j = 0; l_j < n; l_j++) {
        for (l_i = 0; l_i < m; l_i++) {
          if ( (i_gemm_def->beta == 0) && (l_r == 0) ) {
            f_c[(l_j * ldc) + l_i] = 0.0f;
          }
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              union libxsmm_bfloat8_f16 tmp_a_hf;
              union libxsmm_bfloat8_f16 tmp_b_hf;
              float tmp_a_f = 0.0f;
              float tmp_b_f = 0.0f;
              tmp_a_hf.i[0] = 0;
              tmp_b_hf.i[0] = 0;
              if ( (i_gemm_def->trans_a == 0) && (i_gemm_def->vnni_a == 0) ) {
                tmp_a_hf.i[1] = h_a[offs_a + (((l_s*l_k_block) + l_k2) * lda) + l_i];
              } else if ( (i_gemm_def->trans_a == 0) && (i_gemm_def->vnni_a != 0) ) {
                tmp_a_hf.i[1] = h_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              } else if ( (i_gemm_def->trans_a != 0) && (i_gemm_def->vnni_a == 0) ) {
                tmp_a_hf.i[1] = h_a[offs_a + (l_i * lda) + (l_s*l_k_block) + l_k2];
              } else {
                /* should not happen */
              }
              if ( (i_gemm_def->trans_b == 0) && (i_gemm_def->vnni_b == 0) ) {
                tmp_b_hf.i[1] = h_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2];
              } else if ( (i_gemm_def->trans_b != 0) && (i_gemm_def->vnni_b == 0) ) {
                tmp_b_hf.i[1] = h_b[offs_b + (((l_s*l_k_block) + l_k2) * ldb) + l_j];
              } else if ( (i_gemm_def->trans_b != 0) && (i_gemm_def->vnni_b != 0) ) {
                tmp_b_hf.i[1] = h_b[offs_b + (l_j * l_k_block) + (l_s * (ldb*l_k_block)) + l_k2];
              } else {
                /* should not happen */
              }
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
    libxsmm_blasint l_k_block = ( i_gemm_def->vnni_a != 0) ? libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type) : 1;
    for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
      libxsmm_calculate_brgemm_offsets((void**)&h_a, (void**)&h_b, &offs_a, &offs_b, l_r, i_gemm_def);
      for (l_j = 0; l_j < n; l_j++) {
        for (l_i = 0; l_i < m; l_i++) {
          if ( (i_gemm_def->beta == 0) && (l_r == 0) ) {
            f_c[(l_j * ldc) + l_i] = 0.0f;
          }
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              float tmp_a_f = 0.0f, tmp_b_f = 0.0f;
              if ( (i_gemm_def->trans_a == 0) && (i_gemm_def->vnni_a == 0) ) {
                tmp_a_f = libxsmm_convert_hf8_to_f32(h_a[offs_a + (((l_s*l_k_block) + l_k2) * lda) + l_i]);
              } else if ( (i_gemm_def->trans_a == 0) && (i_gemm_def->vnni_a != 0) ) {
                tmp_a_f = libxsmm_convert_hf8_to_f32(h_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2]);
              } else if ( (i_gemm_def->trans_a != 0) && (i_gemm_def->vnni_a == 0) ) {
                tmp_a_f = libxsmm_convert_hf8_to_f32(h_a[offs_a + (l_i * lda) + (l_s*l_k_block) + l_k2]);
              } else {
                /* should not happen */
              }
              if ( (i_gemm_def->trans_b == 0) && (i_gemm_def->vnni_b == 0) ) {
                tmp_b_f = libxsmm_convert_hf8_to_f32(h_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2]);
              } else if ( (i_gemm_def->trans_b != 0) && (i_gemm_def->vnni_b == 0) ) {
                tmp_b_f = libxsmm_convert_hf8_to_f32(h_b[offs_b + (((l_s*l_k_block) + l_k2) * ldb) + l_j]);
              } else if ( (i_gemm_def->trans_b != 0) && (i_gemm_def->vnni_b != 0) ) {
                tmp_b_f = libxsmm_convert_hf8_to_f32(h_b[offs_b + (l_j * l_k_block) + (l_s * (ldb*l_k_block)) + l_k2]);
              } else {
                /* should not happen */
              }
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
    libxsmm_blasint l_k_block = ( i_gemm_def->vnni_a != 0) ? libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type) : 1;
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
          libxsmm_calculate_brgemm_offsets((void**)&h_a, (void**)&h_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              union libxsmm_bfloat8_f16 tmp_a_hf;
              union libxsmm_bfloat8_f16 tmp_b_hf;
              float tmp_a_f = 0.0f;
              float tmp_b_f = 0.0f;
              tmp_a_hf.i[0] = 0;
              tmp_b_hf.i[0] = 0;
              if ( (i_gemm_def->trans_a == 0) && (i_gemm_def->vnni_a == 0) ) {
                tmp_a_hf.i[1] = h_a[offs_a + (((l_s*l_k_block) + l_k2) * lda) + l_i];
              } else if ( (i_gemm_def->trans_a == 0) && (i_gemm_def->vnni_a != 0) ) {
                tmp_a_hf.i[1] = h_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              } else if ( (i_gemm_def->trans_a != 0) && (i_gemm_def->vnni_a == 0) ) {
                tmp_a_hf.i[1] = h_a[offs_a + (l_i * lda) + (l_s*l_k_block) + l_k2];
              } else {
                /* should not happen */
              }
              if ( (i_gemm_def->trans_b == 0) && (i_gemm_def->vnni_b == 0) ) {
                tmp_b_hf.i[1] = h_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2];
              } else if ( (i_gemm_def->trans_b != 0) && (i_gemm_def->vnni_b == 0) ) {
                tmp_b_hf.i[1] = h_b[offs_b + (((l_s*l_k_block) + l_k2) * ldb) + l_j];
              } else if ( (i_gemm_def->trans_b != 0) && (i_gemm_def->vnni_b != 0) ) {
                tmp_b_hf.i[1] = h_b[offs_b + (l_j * l_k_block) + (l_s * (ldb*l_k_block)) + l_k2];
              } else {
                /* should not happen */
              }
              tmp_a_f = libxsmm_convert_f16_to_f32( tmp_a_hf.hf );
              tmp_b_f = libxsmm_convert_f16_to_f32( tmp_b_hf.hf );

              acc += tmp_a_f * tmp_b_f;
            }
          }
        }
        bf8_acc =  libxsmm_convert_f32_to_bf8_rne(acc);
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
    libxsmm_blasint l_k_block = ( i_gemm_def->vnni_a != 0) ? libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type) : 1;
    float acc = 0.0f;
    libxsmm_hfloat8 hf8_acc;
    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        if ( i_gemm_def->beta == 0 ) {
          acc = 0.0f;
        } else {
          float tmp_c_f;
          tmp_c_f = libxsmm_convert_hf8_to_f32(h_c[(l_j * ldc) + l_i]);
          acc = tmp_c_f;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          libxsmm_calculate_brgemm_offsets((void**)&h_a, (void**)&h_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
              float tmp_a_f = 0.0f, tmp_b_f = 0.0f;
              if ( (i_gemm_def->trans_a == 0) && (i_gemm_def->vnni_a == 0) ) {
                tmp_a_f = libxsmm_convert_hf8_to_f32(h_a[offs_a + (((l_s*l_k_block) + l_k2) * lda) + l_i]);
              } else if ( (i_gemm_def->trans_a == 0) && (i_gemm_def->vnni_a != 0) ) {
                tmp_a_f = libxsmm_convert_hf8_to_f32(h_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2]);
              } else if ( (i_gemm_def->trans_a != 0) && (i_gemm_def->vnni_a == 0) ) {
                tmp_a_f = libxsmm_convert_hf8_to_f32(h_a[offs_a + (l_i * lda) + (l_s*l_k_block) + l_k2]);
              } else {
                /* should not happen */
              }
              if ( (i_gemm_def->trans_b == 0) && (i_gemm_def->vnni_b == 0) ) {
                tmp_b_f = libxsmm_convert_hf8_to_f32(h_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2]);
              } else if ( (i_gemm_def->trans_b != 0) && (i_gemm_def->vnni_b == 0) ) {
                tmp_b_f = libxsmm_convert_hf8_to_f32(h_b[offs_b + (((l_s*l_k_block) + l_k2) * ldb) + l_j]);
              } else if ( (i_gemm_def->trans_b != 0) && (i_gemm_def->vnni_b != 0) ) {
                tmp_b_f = libxsmm_convert_hf8_to_f32(h_b[offs_b + (l_j * l_k_block) + (l_s * (ldb*l_k_block)) + l_k2]);
              } else {
                /* should not happen */
              }
              acc += tmp_a_f * tmp_b_f;
            }
          }
        }
        hf8_acc =  libxsmm_convert_f32_to_hf8_rne(acc);
        h_c[(l_j * ldc) + l_i] = hf8_acc;
      }
    }
  } else if ( ((i_gemm_def->a_type == LIBXSMM_DATATYPE_MXBF8) || (i_gemm_def->a_type == LIBXSMM_DATATYPE_MXHF8)) &&
              (i_gemm_def->b_type    == i_gemm_def->a_type)     &&
              ((i_gemm_def->c_type   == LIBXSMM_DATATYPE_F32) || (i_gemm_def->c_type == LIBXSMM_DATATYPE_MXBF8)) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F32)    ) {
    /* MX-scaled FP8 (E8M0 shared scales): A in VNNI, B in VNNI and transposed, C in F32 or MXBF8 */
    unsigned char* fp8_a = (unsigned char*)a;
    unsigned char* fp8_b = (unsigned char*)b;
    int l_is_hf8 = (i_gemm_def->a_type == LIBXSMM_DATATYPE_MXHF8) ? 1 : 0;
    int l_c_is_mxbf8 = ((i_gemm_def->c_type == LIBXSMM_DATATYPE_MXBF8) && (l_is_hf8 == 0)) ? 1 : 0;
    float* l_c_tmp = NULL;
    float* f_c = (float*)c;
    int l_k_block = 4;
    if ( !((i_gemm_def->vnni_a != 0) && (i_gemm_def->vnni_b != 0) && (i_gemm_def->trans_b != 0)) ) {
      fprintf(stderr, "LIBXSMM reference: MXFP8 GEMM requires A in VNNI and B in VNNI-and-transposed format!\n");
      return;
    }
    if ( l_c_is_mxbf8 != 0 ) {
      l_c_tmp = (float*)malloc((size_t)ldc * (size_t)n * sizeof(float));
      f_c = l_c_tmp;
    }
    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        float f32_accum = 0.0f;
        if ( i_gemm_def->beta == 0 ) {
          f_c[(l_j * ldc) + l_i] = 0.0f;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            float tmp_val = 0.0f;
            float scale_a, scale_b;
            unsigned int scale_u32; float *scalef_ptr = (float*)&scale_u32;
            unsigned char sa = i_gemm_def->scf_u8[(l_r * lda * (k/(l_k_block*8))) + (l_s/8) * lda + l_i];
            unsigned char sb = i_gemm_def->scf_b_u8[(l_r * ldb * (k/(l_k_block*8))) + (l_s/8) * ldb + l_j];
            for (l_k2 = l_k_block - 1; l_k2 >= 0; l_k2--) {
              unsigned char a_byte = fp8_a[(l_r * lda * k) + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              unsigned char b_byte = fp8_b[(l_r * ldb * k) + (l_j * l_k_block) + (l_s * (ldb*l_k_block)) + l_k2];
              float a_f = (l_is_hf8 != 0) ? libxsmm_convert_hf8_to_f32((libxsmm_hfloat8)a_byte) : libxsmm_convert_bf8_to_f32((libxsmm_bfloat8)a_byte);
              float b_f = (l_is_hf8 != 0) ? libxsmm_convert_hf8_to_f32((libxsmm_hfloat8)b_byte) : libxsmm_convert_bf8_to_f32((libxsmm_bfloat8)b_byte);
              tmp_val += a_f * b_f;
            }
            scale_u32 = ((unsigned int)sa) << 23; scale_a = *scalef_ptr;
            scale_u32 = ((unsigned int)sb) << 23; scale_b = *scalef_ptr;
            f32_accum += tmp_val * scale_a * scale_b;
          }
        }
        f_c[(l_j * ldc) + l_i] += f32_accum;
      }
    }
    if ( l_c_is_mxbf8 != 0 ) {
      unsigned char* l_c_data = (unsigned char*)c;
      unsigned char* l_c_scf  = i_gemm_def->scf_c_u8;
      for (l_j = 0; l_j < n; l_j++) {
        for (l_i = 0; l_i < m; l_i += 32) {
          libxsmm_gemm_ref_fp32_to_mxbf8_block(&l_c_tmp[(l_j * ldc) + l_i],
                                               &l_c_data[(l_j * ldc) + l_i],
                                               &l_c_scf[(l_j * (ldc/32)) + (l_i/32)]);
        }
      }
      free(l_c_tmp);
    }
  } else if ( ((i_gemm_def->a_type == LIBXSMM_DATATYPE_MXHF6) || (i_gemm_def->a_type == LIBXSMM_DATATYPE_MXBF6)) &&
              (i_gemm_def->b_type    == i_gemm_def->a_type)     &&
              (i_gemm_def->c_type    == LIBXSMM_DATATYPE_F32)    &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F32)    ) {
    /* MX-scaled FP6 (E8M0 shared scales): 6-bit packed, A in VNNI, B in VNNI and transposed, C in F32 */
    unsigned char* fp6_a = (unsigned char*)a;
    unsigned char* fp6_b = (unsigned char*)b;
    float* f_c = (float*)c;
    int l_is_bf6 = (i_gemm_def->a_type == LIBXSMM_DATATYPE_MXBF6) ? 1 : 0;
    int l_k_block = 4;
    long long l_a_slab = (((long long)lda * 6) / 8) * (long long)k;
    long long l_b_slab = (((long long)ldb * 6) / 8) * (long long)k;
    if ( !((i_gemm_def->vnni_a != 0) && (i_gemm_def->vnni_b != 0) && (i_gemm_def->trans_b != 0)) ) {
      fprintf(stderr, "LIBXSMM reference: MXFP6 GEMM requires A in VNNI and B in VNNI-and-transposed format!\n");
      return;
    }
    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        float f32_accum = 0.0f;
        if ( i_gemm_def->beta == 0 ) {
          f_c[(l_j * ldc) + l_i] = 0.0f;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            float tmp_val = 0.0f;
            float scale_a, scale_b;
            unsigned int scale_u32; float *scalef_ptr = (float*)&scale_u32;
            unsigned char sa = i_gemm_def->scf_u8[(l_r * lda * (k/(l_k_block*8))) + (l_s/8) * lda + l_i];
            unsigned char sb = i_gemm_def->scf_b_u8[(l_r * ldb * (k/(l_k_block*8))) + (l_s/8) * ldb + l_j];
            unsigned char* pa = &fp6_a[(l_r * l_a_slab) + ((long long)l_s * lda * 3) + ((long long)l_i * 3)];
            unsigned char* pb = &fp6_b[(l_r * l_b_slab) + ((long long)l_s * ldb * 3) + ((long long)l_j * 3)];
            unsigned int va = (unsigned int)pa[0] | ((unsigned int)pa[1] << 8) | ((unsigned int)pa[2] << 16);
            unsigned int vb = (unsigned int)pb[0] | ((unsigned int)pb[1] << 8) | ((unsigned int)pb[2] << 16);
            for (l_k2 = l_k_block - 1; l_k2 >= 0; l_k2--) {
              unsigned char v6a = (unsigned char)((va >> (6*l_k2)) & 0x3f);
              unsigned char v6b = (unsigned char)((vb >> (6*l_k2)) & 0x3f);
              unsigned char ha = (l_is_bf6 != 0) ? libxsmm_convert_fp6_e3m2_to_hf8(v6a) : libxsmm_convert_fp6_e2m3_to_hf8(v6a);
              unsigned char hb = (l_is_bf6 != 0) ? libxsmm_convert_fp6_e3m2_to_hf8(v6b) : libxsmm_convert_fp6_e2m3_to_hf8(v6b);
              float a_f = libxsmm_convert_hf8_to_f32((libxsmm_hfloat8)ha);
              float b_f = libxsmm_convert_hf8_to_f32((libxsmm_hfloat8)hb);
              tmp_val += a_f * b_f;
            }
            scale_u32 = ((unsigned int)sa) << 23; scale_a = *scalef_ptr;
            scale_u32 = ((unsigned int)sb) << 23; scale_b = *scalef_ptr;
            f32_accum += tmp_val * scale_a * scale_b;
          }
        }
        f_c[(l_j * ldc) + l_i] += f32_accum;
      }
    }
  } else if ( (i_gemm_def->a_type    == LIBXSMM_DATATYPE_MXFP4X2) &&
              (i_gemm_def->b_type    == LIBXSMM_DATATYPE_MXFP4X2) &&
              ((i_gemm_def->c_type   == LIBXSMM_DATATYPE_F32) || (i_gemm_def->c_type == LIBXSMM_DATATYPE_MXFP4X2)) &&
              (i_gemm_def->comp_type == LIBXSMM_DATATYPE_F32)   ) {
    /* MX-scaled FP4 (E8M0 shared scales): 4-bit packed, A in VNNI, B in VNNI and transposed, C in F32 or MXFP4 */
    unsigned char* fp4_a = (unsigned char*)a;
    unsigned char* fp4_b = (unsigned char*)b;
    int l_c_is_mxfp4 = (i_gemm_def->c_type == LIBXSMM_DATATYPE_MXFP4X2) ? 1 : 0;
    float* l_c_tmp = NULL;
    float* f_c = (float*)c;
    int l_k_block = 32;
    unsigned int k_scale_freq = 8;
    if ( !((i_gemm_def->vnni_a != 0) && (i_gemm_def->vnni_b != 0) && (i_gemm_def->trans_b != 0)) ) {
      fprintf(stderr, "LIBXSMM reference: MXFP4 GEMM requires A in VNNI and B in VNNI-and-transposed format!\n");
      return;
    }
    if ( l_c_is_mxfp4 != 0 ) {
      l_c_tmp = (float*)malloc((size_t)ldc * (size_t)n * sizeof(float));
      f_c = l_c_tmp;
    }
    for (l_j = 0; l_j < n; l_j++) {
      for (l_i = 0; l_i < m; l_i++) {
        float f32_accum = 0.0f;
        if ( i_gemm_def->beta == 0 ) {
          f_c[(l_j * ldc) + l_i] = 0.0f;
        }
        for (l_r = 0; l_r < i_gemm_def->br_count; l_r++) {
          for (l_s = 0; l_s < (k / l_k_block); l_s++) {
            unsigned int l_k3;
            float scale_a, scale_b;
            unsigned int scale_u32; float *scalef_ptr = (float*)&scale_u32;
            unsigned char sa = i_gemm_def->scf_u8[(l_r * lda * (k/l_k_block)) + l_s * lda + l_i];
            unsigned char sb = i_gemm_def->scf_b_u8[(l_r * ldb * (k/l_k_block)) + l_s * ldb + l_j];
            scale_u32 = ((unsigned int)sa) << 23; scale_a = *scalef_ptr;
            scale_u32 = ((unsigned int)sb) << 23; scale_b = *scalef_ptr;
            for (l_k3 = 0; l_k3 < (unsigned int)l_k_block/k_scale_freq; l_k3++) {
              float tmp_val = 0.0f;
              for (l_k2 = 0; l_k2 < (int)k_scale_freq; l_k2++) {
                int kpos = l_s*l_k_block + (int)(k_scale_freq*l_k3) + l_k2;
                int kk = kpos % 8;
                int kblk = kpos / 8;
                unsigned char a4, b4, byte_a, byte_b;
                int byte_idx;
                byte_idx = kk / 2;
                byte_a = fp4_a[(l_r * lda * (k/2)) + (kblk*lda*4) + l_i*4 + byte_idx];
                byte_b = fp4_b[(l_r * ldb * (k/2)) + (kblk*ldb*4) + l_j*4 + byte_idx];
                a4 = ((kk % 2) == 0) ? (byte_a & 0x0f) : ((byte_a >> 4) & 0x0f);
                b4 = ((kk % 2) == 0) ? (byte_b & 0x0f) : ((byte_b >> 4) & 0x0f);
                tmp_val += libxsmm_convert_mxfp4_to_float(a4) * libxsmm_convert_mxfp4_to_float(b4);
              }
              f32_accum += tmp_val * scale_a * scale_b;
            }
          }
        }
        f_c[(l_j * ldc) + l_i] += f32_accum;
      }
    }
    if ( l_c_is_mxfp4 != 0 ) {
      unsigned char* l_c_data = (unsigned char*)c;
      unsigned char* l_c_scf  = i_gemm_def->scf_c_u8;
      for (l_j = 0; l_j < n; l_j++) {
        for (l_i = 0; l_i < m; l_i += 32) {
          libxsmm_gemm_ref_fp32_to_mxfp4_block(&l_c_tmp[(l_j * ldc) + l_i],
                                               &l_c_data[(l_j * (ldc/2)) + (l_i/2)],
                                               &l_c_scf[(l_j * (ldc/32)) + (l_i/32)]);
        }
      }
      free(l_c_tmp);
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_ref_apply_c_vnni(libxsmm_gemm_def* i_gemm_def, void *c_ptr, const libxsmm_gemm_descriptor *i_xgemm_desc) {
  if (i_gemm_def->fuse_vnni_c > 0) {
    unsigned short vnni_cvt_type = (unsigned short)(( LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) == LIBXSMM_DATATYPE_BF16 || LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) == LIBXSMM_DATATYPE_F16 ) ? (( libxsmm_cpuid_dot_pack_factor((libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) == 4 && (i_xgemm_desc->n % 4 == 0)) ? LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4 : LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2) : LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4);
    libxsmm_meltw_unary_param unary_param;
    libxsmm_descriptor_blob blob;
    const libxsmm_meltw_descriptor *const desc = libxsmm_meltw_descriptor_init2(&blob, (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ), LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_F32, (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ), i_xgemm_desc->m, i_xgemm_desc->n,
        i_xgemm_desc->ldc, i_xgemm_desc->ldc, 0, 0, (unsigned short)LIBXSMM_MELTW_FLAG_UNARY_NONE, (unsigned short)vnni_cvt_type, LIBXSMM_MELTW_OPERATION_UNARY);
    memcpy(i_gemm_def->c_vnni_scratch, c_ptr, (size_t)i_xgemm_desc->ldc * i_xgemm_desc->n * LIBXSMM_TYPESIZE((libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )));
    unary_param.in.primary  = (void*)i_gemm_def->c_vnni_scratch;
    unary_param.out.primary = (void*)c_ptr;
    libxsmm_reference_unary_elementwise(&unary_param, desc);
  }
}

LIBXSMM_API_INTERN
void libxsmm_reference_gemm(void *param, const libxsmm_gemm_descriptor *i_xgemm_desc) {
  libxsmm_gemm_def l_gemm_def;
  /* Return if kernel is tileconfig/tilerelease  */
  if (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG) > 0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG) == 0)) {
    return;
  }
  if (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG) == 0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG) > 0)) {
    return;
  }
  if (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI) > 0)) {
    libxsmm_gemm_descriptor l_adjusted_desc = *i_xgemm_desc;
    libxsmm_gemm_ext_param *gemm_param = (libxsmm_gemm_ext_param*)param;
    libxsmm_setup_gemm_def(&l_gemm_def, param, &l_adjusted_desc);
    l_gemm_def.c_scratch = (void*)gemm_param->c.primary;
    libxsmm_ref_allocate_c_scratch(&l_gemm_def);
    libxsmm_ref_allocate_c_vnni_scratch(&l_gemm_def);
    libxsmm_ref_apply_preop(&l_gemm_def, gemm_param, &l_adjusted_desc);
    libxsmm_ref_matmul(&l_gemm_def, (void*)gemm_param->a.primary, (void*)gemm_param->b.primary, (void*)l_gemm_def.c_scratch);
    libxsmm_ref_apply_postop(&l_gemm_def, gemm_param, &l_adjusted_desc);
    libxsmm_ref_apply_c_vnni(&l_gemm_def, (void*)gemm_param->c.primary, i_xgemm_desc);
    libxsmm_ref_deallocate_c_scratch(&l_gemm_def);
    libxsmm_ref_deallocate_c_vnni_scratch(&l_gemm_def);
  } else {
    libxsmm_gemm_param *gemm_param = (libxsmm_gemm_param*)param;
    libxsmm_setup_gemm_def(&l_gemm_def, param, i_xgemm_desc);
    libxsmm_ref_allocate_c_vnni_scratch(&l_gemm_def);
    libxsmm_ref_matmul( &l_gemm_def, (void*)gemm_param->a.primary, (void*)gemm_param->b.primary, (void*)gemm_param->c.primary);
    libxsmm_ref_apply_c_vnni(&l_gemm_def, (void*)gemm_param->c.primary, i_xgemm_desc);
    libxsmm_ref_deallocate_c_vnni_scratch(&l_gemm_def);
  }
  if (l_gemm_def.is_spmm > 0) {
    if (l_gemm_def.c_scratch != NULL) {
      free(l_gemm_def.c_scratch);
    }
  }
}

