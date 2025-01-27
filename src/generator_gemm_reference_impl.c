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
  libxsmm_float16 *scf_f16;
  int binary_postop;
  int unary_postop;
  unsigned int mxfp4_block_size;
  unsigned int is_Ai4Bf16_gemm;
  unsigned int is_Amxfp4Bbf16_gemm;
  unsigned int is_Amxfp4Bfp32_gemm;
  unsigned int is_Amxfp4Bi8_gemm;
  unsigned int is_Ai4Bi8_gemm;
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

  /* dynamic LD */
  unsigned int is_dynld;
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

  if (i_xgemm_desc->meltw_operation == LIBXSMM_MELTW_OPERATION_BINARY) {
    if (i_xgemm_desc->meltw_param == LIBXSMM_MELTW_TYPE_BINARY_ADD) {
      if (((i_xgemm_desc->meltw_flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0) > 0 ) ||
          ((i_xgemm_desc->meltw_flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1) > 0 )) {
        l_gemm_def.fuse_colbias_add = 1;
      }
    }
  }

  l_gemm_def.fuse_zpt_sub = 0;
  l_gemm_def.is_Ai4Bf16_gemm = 0;
  l_gemm_def.is_Ai4Bi8_gemm = 0;
  l_gemm_def.is_Amxfp4Bbf16_gemm = 0;
  l_gemm_def.is_Amxfp4Bfp32_gemm = 0;
  l_gemm_def.is_Abf8Bbf16_gemm = 0;
  l_gemm_def.is_Abf8Bf16_gemm = 0;
  l_gemm_def.is_Ahf8Bbf16_gemm = 0;
  l_gemm_def.is_Ai4Bi8_gemm = 0;
  l_gemm_def.is_Amxfp4Bbf16_gemm = 0;
  l_gemm_def.is_Amxfp4Bfp32_gemm = 0;
  l_gemm_def.is_Amxfp4Bi8_gemm = 0;
  l_gemm_def.unsigned_a = 0;
  l_gemm_def.unsigned_b = 0;
  l_gemm_def.is_spmm = 0;

  if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) {
    l_gemm_def.unsigned_a = 1;
  }

  if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) {
    l_gemm_def.unsigned_b = 1;
  }

  if ((l_dtype_a == LIBXSMM_DATATYPE_I8) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_MXFP4_VNNI2) > 0)) {
    if (l_dtype_b == LIBXSMM_DATATYPE_BF16) {
      l_gemm_def.is_Amxfp4Bbf16_gemm = 1;
    }
    if (l_dtype_b == LIBXSMM_DATATYPE_F32) {
      l_gemm_def.is_Amxfp4Bfp32_gemm = 1;
    }
    l_gemm_def.mxfp4_block_size = 32;
  }

  if ((l_dtype_a == LIBXSMM_DATATYPE_I8) && (l_dtype_b == LIBXSMM_DATATYPE_I8) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_MXFP4_VNNI8_INTLV) > 0)) {
    l_gemm_def.is_Amxfp4Bi8_gemm = 1;
    l_gemm_def.mxfp4_block_size = 32;
  }

  if ((l_dtype_a == LIBXSMM_DATATYPE_I8 || l_dtype_a == LIBXSMM_DATATYPE_U8) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_INT4_VNNI2) > 0) && (l_dtype_b == LIBXSMM_DATATYPE_F16)) {
    l_gemm_def.is_Ai4Bf16_gemm = 1;
    l_gemm_def.fuse_zpt_sub = 1;
  }

  if ((l_dtype_a == LIBXSMM_DATATYPE_I8 || l_dtype_a == LIBXSMM_DATATYPE_U8) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_INT4_VNNI8_INTLV) > 0) && (l_dtype_b == LIBXSMM_DATATYPE_I8)) {
    l_gemm_def.is_Ai4Bi8_gemm = 1;
    l_gemm_def.fuse_zpt_sub = 1;
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
    if ( libxsmm_is_runtime_set_ld_gemm( i_xgemm_desc ) != 0 ) {
      long long l_dyn_stride_a = 0;
      long long l_dyn_stride_b = 0;
      if (gemm_param_ext == NULL) {
        l_dyn_stride_a = *((long long*)gemm_param->a.secondary);
        l_dyn_stride_b = *((long long*)gemm_param->b.secondary);
      } else {
        l_dyn_stride_a = *((long long*)gemm_param_ext->a.secondary);
        l_dyn_stride_b = *((long long*)gemm_param_ext->b.secondary);
      }
      l_gemm_def.stride_a = l_dyn_stride_a/LIBXSMM_TYPESIZE(l_dtype_a);
      l_gemm_def.stride_b = l_dyn_stride_b/LIBXSMM_TYPESIZE(l_dtype_b);
    } else {
      l_gemm_def.stride_a = i_xgemm_desc->c1/LIBXSMM_TYPESIZE(l_dtype_a);
      l_gemm_def.stride_b = i_xgemm_desc->c2/LIBXSMM_TYPESIZE(l_dtype_b);
    }
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
      if ( (l_dtype_a    == LIBXSMM_DATATYPE_I8)  && (l_dtype_b == LIBXSMM_DATATYPE_F16) &&
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
      if ( (l_dtype_a    == LIBXSMM_DATATYPE_I8)  && (l_dtype_b == LIBXSMM_DATATYPE_BF16) &&
             (l_dtype_c == LIBXSMM_DATATYPE_BF16 || l_dtype_c == LIBXSMM_DATATYPE_F32) ) {
          l_gemm_def.scf_f32 =  (float*)gemm_param->a.tertiary;
      }
      if ( l_dtype_a == LIBXSMM_DATATYPE_I8 && l_dtype_b == LIBXSMM_DATATYPE_I8 && l_dtype_c == LIBXSMM_DATATYPE_F32 ) {
        l_gemm_def.scf =  *((float*)gemm_param->c.tertiary);
      }
    }
  } else {
    if ( (l_dtype_a    == LIBXSMM_DATATYPE_I8)  && (l_dtype_b == LIBXSMM_DATATYPE_F16) &&
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
  if ( libxsmm_is_runtime_set_ld_gemm( i_xgemm_desc ) != 0 ) {
    long long l_dyn_lda = 0;
    long long l_dyn_ldb = 0;
    long long l_dyn_ldc = 0;
    if (gemm_param_ext == NULL) {
      l_dyn_lda = *((long long*)gemm_param->a.quinary);
      l_dyn_ldb = *((long long*)gemm_param->b.quinary);
      l_dyn_ldc = *((long long*)gemm_param->c.quinary);
    } else {
      l_dyn_lda = *((long long*)gemm_param_ext->a.quinary);
      l_dyn_ldb = *((long long*)gemm_param_ext->b.quinary);
      l_dyn_ldc = *((long long*)gemm_param_ext->c.quinary);
    }
    l_gemm_def.lda = (libxsmm_blasint)l_dyn_lda;
    l_gemm_def.ldb = (libxsmm_blasint)l_dyn_ldb;
    l_gemm_def.ldc = (libxsmm_blasint)l_dyn_ldc;
  } else {
    l_gemm_def.lda = l_lda;
    l_gemm_def.ldb = l_ldb;
    l_gemm_def.ldc = l_ldc;
  }
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
          libxsmm_calculate_brgemm_offsets((void**)&f_a, (void**)&f_b, &offs_a, &offs_b, l_r, i_gemm_def);
          for (l_s = 0; l_s < k; l_s++) {
            if (i_gemm_def->trans_b == 0) {
              if (i_gemm_def->trans_a == 0) {
                f_c[(l_j * ldc) + l_i] += f_a[offs_a + (l_s * lda) + l_i] *
                                                   f_b[offs_b + (l_j * ldb) + l_s];
              } else {
                f_c[(l_j * ldc) + l_i] += f_a[offs_a + (l_i * lda) + l_s] *
                                                   f_b[offs_b + (l_j * ldb) + l_s];
              } /* if-else l_trans_a */
            } else {
              if (i_gemm_def->trans_a == 0) {
                f_c[(l_j * ldc) + l_i] += f_a[offs_a + (l_s * lda) + l_i] *
                                                   f_b[offs_b + (l_s * ldb) + l_j];
              } else {
                f_c[(l_j * ldc) + l_i] += f_a[offs_a + (l_i * lda) + l_s] *
                                                   f_b[offs_b + (l_s * ldb) + l_j];
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
    libxsmm_blasint l_k_block = libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type);

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
    libxsmm_blasint l_k_block = libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type);

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
    libxsmm_blasint l_k_block = libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type);

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
    libxsmm_blasint l_k_block = libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type);

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
    libxsmm_blasint l_k_block = libxsmm_cpuid_dot_pack_factor(i_gemm_def->a_type);

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
              if (i_gemm_def->a_type == LIBXSMM_DATATYPE_I8 && i_gemm_def->unsigned_a == 0) {
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
              float tmp_a_f;
              float tmp_b_f;
              tmp_a_hf.i[0] = 0;
              tmp_a_hf.i[1] = h_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              tmp_b_hf.i[0] = 0;
              tmp_b_hf.i[1] = h_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2];
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
              float tmp_a_f, tmp_b_f;
              tmp_a_f = libxsmm_convert_hf8_to_f32(h_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2]);
              tmp_b_f = libxsmm_convert_hf8_to_f32(h_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2]);
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
              float tmp_a_f;
              float tmp_b_f;
              tmp_a_hf.i[0] = 0;
              tmp_a_hf.i[1] = h_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2];
              tmp_b_hf.i[0] = 0;
              tmp_b_hf.i[1] = h_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2];
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
              float tmp_a_f;
              float tmp_b_f;
              tmp_a_f = libxsmm_convert_hf8_to_f32(h_a[offs_a + (l_s * (lda*l_k_block)) + (l_i*l_k_block) + l_k2]);
              tmp_b_f = libxsmm_convert_hf8_to_f32(h_b[offs_b + (l_j * ldb) + (l_s*l_k_block) + l_k2]);
              acc += tmp_a_f * tmp_b_f;
            }
          }
        }
        hf8_acc =  libxsmm_convert_f32_to_hf8_rne(acc);
        h_c[(l_j * ldc) + l_i] = hf8_acc;
      }
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

