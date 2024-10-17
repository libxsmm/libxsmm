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
#include "generator_matequation_avx_avx512.h"
#include "generator_mateltwise_sse_avx_avx512.h"
#include "generator_mateltwise_unary_binary_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "generator_common_x86.h"
#include "generator_matequation_scratch_avx_avx512.h"
#include "generator_mateltwise_reduce_avx_avx512.h"
#include "generator_mateltwise_gather_scatter_avx_avx512.h"
#include "generator_mateltwise_transform_common.h"
#include "generator_gemm_sse_avx_avx2_avx512.h"
#include "generator_gemm_amx.h"

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_create_unary_descriptor(libxsmm_descriptor_blob *blob, libxsmm_meqn_elem *cur_op, libxsmm_meltw_descriptor **desc, libxsmm_datatype in_precision, libxsmm_datatype out_precision) {
  if (libxsmm_meqn_is_unary_opcode_transform_kernel(cur_op->info.u_op.type) > 0) {
    *desc = libxsmm_meltw_descriptor_init2(blob, in_precision, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED,
      cur_op->info.u_op.dtype, out_precision, cur_op->le->tmp.m, cur_op->le->tmp.n, cur_op->le->tmp.ld, cur_op->tmp.ld, 0, 0,
      LIBXSMM_CAST_USHORT(cur_op->info.u_op.flags), LIBXSMM_CAST_USHORT(cur_op->info.u_op.type), LIBXSMM_MELTW_OPERATION_UNARY);
  } else if ((libxsmm_meqn_is_unary_opcode_reduce_kernel(cur_op->info.u_op.type) > 0) || (libxsmm_meqn_is_unary_opcode_reduce_cols_idx_kernel(cur_op->info.u_op.type) > 0)) {
    *desc = libxsmm_meltw_descriptor_init2(blob, in_precision, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED,
      cur_op->info.u_op.dtype, out_precision, cur_op->le->tmp.m, cur_op->le->tmp.n, cur_op->le->tmp.ld, cur_op->tmp.ld, 0, 0,
      LIBXSMM_CAST_USHORT(cur_op->info.u_op.flags), LIBXSMM_CAST_USHORT(cur_op->info.u_op.type), LIBXSMM_MELTW_OPERATION_UNARY);
  } else {
    if ((cur_op->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_IDENTITY) && (in_precision != out_precision)) {
      *desc = libxsmm_meltw_descriptor_init2(blob, in_precision, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED,
        LIBXSMM_DATATYPE_F32, out_precision, cur_op->tmp.m, cur_op->tmp.n, cur_op->le->tmp.ld, cur_op->tmp.ld, 0, 0,
        LIBXSMM_CAST_USHORT(cur_op->info.u_op.flags), LIBXSMM_CAST_USHORT(cur_op->info.u_op.type), LIBXSMM_MELTW_OPERATION_UNARY);
    } else {
      *desc = libxsmm_meltw_descriptor_init2(blob, in_precision, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED,
         cur_op->info.u_op.dtype, out_precision, cur_op->tmp.m, cur_op->tmp.n, cur_op->le->tmp.ld, cur_op->tmp.ld, 0, 0,
         LIBXSMM_CAST_USHORT(cur_op->info.u_op.flags), LIBXSMM_CAST_USHORT(cur_op->info.u_op.type), LIBXSMM_MELTW_OPERATION_UNARY);
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_create_binary_descriptor(libxsmm_descriptor_blob *blob, libxsmm_meqn_elem *cur_op, libxsmm_meltw_descriptor **desc, libxsmm_datatype in_precision, libxsmm_datatype out_precision) {
  *desc = libxsmm_meltw_descriptor_init2(blob, in_precision, cur_op->ri->tmp.dtype, LIBXSMM_DATATYPE_UNSUPPORTED,
    cur_op->info.b_op.dtype, out_precision, cur_op->tmp.m, cur_op->tmp.n, cur_op->le->tmp.ld, cur_op->tmp.ld, cur_op->ri->tmp.ld, 0,
    LIBXSMM_CAST_USHORT(cur_op->info.b_op.flags), LIBXSMM_CAST_USHORT(cur_op->info.b_op.type), LIBXSMM_MELTW_OPERATION_BINARY);
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_create_ternary_descriptor(libxsmm_descriptor_blob *blob, libxsmm_meqn_elem *cur_op, libxsmm_meltw_descriptor **desc, libxsmm_datatype in_precision, libxsmm_datatype out_precision) {
  libxsmm_datatype in2_dtype = (cur_op->info.t_op.type == LIBXSMM_MELTW_TYPE_TERNARY_SELECT) ? LIBXSMM_DATATYPE_IMPLICIT : cur_op->r2->tmp.dtype;
  *desc = libxsmm_meltw_descriptor_init2(blob, in_precision, cur_op->ri->tmp.dtype, in2_dtype,
    cur_op->info.t_op.dtype, out_precision, cur_op->tmp.m, cur_op->tmp.n, cur_op->le->tmp.ld, cur_op->tmp.ld, cur_op->ri->tmp.ld, cur_op->r2->tmp.ld,
    LIBXSMM_CAST_USHORT(cur_op->info.t_op.flags), LIBXSMM_CAST_USHORT(cur_op->info.t_op.type), LIBXSMM_MELTW_OPERATION_TERNARY);
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_gemm_set_descriptor(libxsmm_generated_code* io_generated_code, const libxsmm_meqn_elem *cur_op,
  libxsmm_descriptor_blob* blob, libxsmm_gemm_descriptor **out_desc)
{
  libxsmm_gemm_descriptor *desc = NULL;
  int  gemm_flags = LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI;
  libxsmm_gemm_batch_reduce_config br_config;
  libxsmm_blasint m = 0, n = 0, k = 0, lda = 0, ldb = 0, ldc = 0;
  libxsmm_datatype a_in_type;
  libxsmm_datatype b_in_type;

  m = cur_op->tmp.m;
  n = cur_op->tmp.n;
  if (cur_op->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
    k = cur_op->le->info.arg.n;
    lda = cur_op->le->info.arg.ld;
    a_in_type = cur_op->le->info.arg.dtype;
  } else {
    k = cur_op->le->tmp.n;
    lda = cur_op->le->tmp.ld;
    a_in_type = cur_op->le->tmp.dtype;
  }
  if (cur_op->ri->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
    ldb = cur_op->ri->info.arg.ld;
    b_in_type = cur_op->ri->info.arg.dtype;
  } else {
    ldb = cur_op->ri->tmp.ld;
    b_in_type = cur_op->ri->tmp.dtype;
  }
  ldc = cur_op->tmp.ld;
  memset(&br_config, 0, sizeof(libxsmm_gemm_batch_reduce_config));
  if (cur_op->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) {
    gemm_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
  }
  if (((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) && (cur_op->info.b_op.is_brgemm == 1)) ||
      ((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) && (cur_op->info.t_op.is_brgemm == 1))) {
    if ((cur_op->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG) &&
        (cur_op->le->info.arg.arg_attr.type == LIBXSMM_MATRIX_ARG_TYPE_SET)) {
      br_config.br_unroll_hint = LIBXSMM_CAST_UCHAR(( cur_op->le->info.arg.arg_attr.set_cardinality_hint <= 255 ) ? cur_op->le->info.arg.arg_attr.set_cardinality_hint : 0);
      if (cur_op->le->info.arg.arg_attr.set_type == LIBXSMM_MATRIX_ARG_SET_TYPE_ABS_ADDRESS) {
        br_config.br_type = LIBXSMM_GEMM_BATCH_REDUCE_ADDRESS;
      } else if (cur_op->le->info.arg.arg_attr.set_type == LIBXSMM_MATRIX_ARG_SET_TYPE_OFFSET_BASE) {
        br_config.br_type = LIBXSMM_GEMM_BATCH_REDUCE_OFFSET;
      } else if (cur_op->le->info.arg.arg_attr.set_type == LIBXSMM_MATRIX_ARG_SET_TYPE_STRIDE_BASE) {
        br_config.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
        br_config.br_stride_a_hint = cur_op->le->info.arg.arg_attr.set_stride_hint;
      } else {
        /* This should not happen */
      }
    }
    if ((cur_op->ri->type == LIBXSMM_MATRIX_EQN_NODE_ARG) &&
        (cur_op->ri->info.arg.arg_attr.type == LIBXSMM_MATRIX_ARG_TYPE_SET)) {
      if (cur_op->ri->info.arg.arg_attr.set_type == LIBXSMM_MATRIX_ARG_SET_TYPE_STRIDE_BASE) {
        br_config.br_stride_b_hint = cur_op->ri->info.arg.arg_attr.set_stride_hint;
      }
    }
    /* set BRGEMM option */
    if ( br_config.br_type == LIBXSMM_GEMM_BATCH_REDUCE_ADDRESS ) {
      gemm_flags |= LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS;
    } else if ( br_config.br_type == LIBXSMM_GEMM_BATCH_REDUCE_OFFSET ) {
      gemm_flags |= LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET;
    } else if ( br_config.br_type == LIBXSMM_GEMM_BATCH_REDUCE_STRIDE ) {
      gemm_flags |= LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE;
    } else {
      /* not a BRGEMM */
    }
  }

  /* Check flags for A in vnni */
  if (((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) &&
       ((cur_op->info.b_op.type == LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_A_VNNI) ||
        (cur_op->info.b_op.type == LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_A_VNNI_B_TRANS) ||
        (cur_op->info.b_op.type == LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_A_VNNI_TRANS) ||
        (cur_op->info.b_op.type == LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_A_VNNI_TRANS_B_TRANS)))  ||
      ((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) &&
       ((cur_op->info.t_op.type == LIBXSMM_MELTW_TYPE_TERNARY_BRGEMM_A_VNNI) ||
        (cur_op->info.t_op.type == LIBXSMM_MELTW_TYPE_TERNARY_BRGEMM_A_VNNI_B_TRANS) ||
        (cur_op->info.t_op.type == LIBXSMM_MELTW_TYPE_TERNARY_BRGEMM_A_VNNI_TRANS) ||
        (cur_op->info.t_op.type == LIBXSMM_MELTW_TYPE_TERNARY_BRGEMM_A_VNNI_TRANS_B_TRANS)))) {
    gemm_flags |= LIBXSMM_GEMM_FLAG_VNNI_A;
  }

  if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) &&
              ( io_generated_code->arch >= LIBXSMM_X86_AVX2 ) && LIBXSMM_DATATYPE_BF16 == a_in_type ) {
    /* some checks as we cannot mask everything */
    if ( (k % 2 != 0) && ((gemm_flags & LIBXSMM_GEMM_FLAG_VNNI_A) != 0) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
      return;
    }
    if ( (gemm_flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) {
    } else {
      k = k/2;
      ldb = ldb/2;
    }
  } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) &&
              ( io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX ) && ((LIBXSMM_DATATYPE_BF8 == a_in_type) || (LIBXSMM_DATATYPE_HF8 == a_in_type)) ) {
    /* some checks as we cannot mask everything */
    if ( (k % 4 != 0) && ((gemm_flags & LIBXSMM_GEMM_FLAG_VNNI_A) != 0) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
      return;
    }
    if ( (gemm_flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) {
    } else {
      k = k/4;
      ldb = ldb/4;
    }
  }

#if 0
  printf("Dispatching GEMM %d %d %d %d %d %d %d\n", m, n, k, lda, ldb, ldc, gemm_flags);
#endif

  /* build descriptor */
  /* Add fusion-related setup of descriptor */
  if ((cur_op->fusion_info.xgemm.fused_colbias_add_op == 1) ||
      (cur_op->fusion_info.xgemm.fused_relu_op == 1) ||
      (cur_op->fusion_info.xgemm.fused_sigmoid_op == 1)) {
    int remove_flag = 0xffffffff ^ LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI;
    gemm_flags = (gemm_flags & remove_flag) | LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI;
  }

  desc = libxsmm_gemm_descriptor_init(blob, a_in_type, b_in_type,
    cur_op->tmp.dtype, LIBXSMM_DATATYPE_F32,
    m, n, k, lda, ldb, ldc, gemm_flags, 0);

  /* Enforce overwrite C flag */
  desc->internal_flags_2 = desc->internal_flags_2 & 0xfb;

  /* add more BRGEMM related fields */
  if ( (br_config.br_type != LIBXSMM_GEMM_BATCH_REDUCE_NONE) && (br_config.br_unroll_hint != 0) ) {
    desc->c3 = LIBXSMM_CAST_UCHAR(((br_config.br_unroll_hint < 255) && (br_config.br_unroll_hint > 0)) ? br_config.br_unroll_hint : 0);
  }
  if ( br_config.br_type == LIBXSMM_GEMM_BATCH_REDUCE_STRIDE ) {
    desc->c1 = br_config.br_stride_a_hint;
    desc->c2 = br_config.br_stride_b_hint;
  }

  if (cur_op->fusion_info.xgemm.fused_colbias_add_op == 1) {
    desc->meltw_operation     = LIBXSMM_MELTW_OPERATION_BINARY;
    desc->meltw_param         = LIBXSMM_MELTW_TYPE_BINARY_ADD;
    desc->meltw_flags         = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1;
    desc->meltw_datatype_aux  = LIBXSMM_CAST_UCHAR(cur_op->fusion_info.xgemm.colbias_dtype);
  }
  if (cur_op->fusion_info.xgemm.fused_relu_op == 1) {
    desc->eltw_cp_op    = LIBXSMM_MELTW_OPERATION_UNARY;
    desc->eltw_cp_param = LIBXSMM_MELTW_TYPE_UNARY_RELU;
    desc->eltw_cp_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  }
  if (cur_op->fusion_info.xgemm.fused_sigmoid_op == 1) {
    desc->eltw_cp_op    = LIBXSMM_MELTW_OPERATION_UNARY;
    desc->eltw_cp_param = LIBXSMM_MELTW_TYPE_UNARY_SIGMOID;
    desc->eltw_cp_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  }

  *out_desc = desc;
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_gemm_set_reg_mapping( libxsmm_gemm_descriptor* i_xgemm_desc, libxsmm_gp_reg_mapping*  i_gp_reg_mapping ) {
  libxsmm_gp_reg_mapping l_gp_reg_mapping;
  /* define gp register mapping */
  memset( &l_gp_reg_mapping, 0, sizeof(libxsmm_gp_reg_mapping) );
#if defined(_WIN32) || defined(__CYGWIN__)
  LIBXSMM_UNUSED(i_xgemm_desc);
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_R8;
  /* TODO: full support for Windows calling convention */
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_UNDEF;
#else /* match calling convention on Linux */
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R8;
  if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) ) {
    l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_RCX;
    l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_R8;
    l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R9;
  } else {
    l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_UNDEF;
    l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_RCX;
    l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R8;
  }
  /* If we are generating the batchreduce kernel, then we rename the registers */
  if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
    l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
    l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
    l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
    l_gp_reg_mapping.gp_reg_reduce_count = LIBXSMM_X86_GP_REG_RCX;
    if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) ) {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_R8;
      l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_R9;
      l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_UNDEF;
    } else {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_UNDEF;
      l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_R8;
      l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R9;
    }
    l_gp_reg_mapping.gp_reg_reduce_loop = LIBXSMM_X86_GP_REG_R13;
    l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_X86_GP_REG_R14;
  } else if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) {
    l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
    l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
    l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
    l_gp_reg_mapping.gp_reg_reduce_count = LIBXSMM_X86_GP_REG_RCX;
    l_gp_reg_mapping.gp_reg_a_offset = LIBXSMM_X86_GP_REG_R8;
    l_gp_reg_mapping.gp_reg_b_offset = LIBXSMM_X86_GP_REG_R9;
    if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) ) {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_RAX;
    } else {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_UNDEF;
    }
    l_gp_reg_mapping.gp_reg_reduce_loop = LIBXSMM_X86_GP_REG_R13;
    l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_X86_GP_REG_R14;
  }
#endif
  l_gp_reg_mapping.gp_reg_mloop = LIBXSMM_X86_GP_REG_R10;
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_X86_GP_REG_R11;
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_X86_GP_REG_R12;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_X86_GP_REG_R15;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_X86_GP_REG_RBX;
  *i_gp_reg_mapping = l_gp_reg_mapping;
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_gemm_set_reg_mapping_amx( libxsmm_gemm_descriptor* i_xgemm_desc, libxsmm_gp_reg_mapping*  i_gp_reg_mapping ) {
  libxsmm_gp_reg_mapping l_gp_reg_mapping;
  /* define gp register mapping */
  memset( &l_gp_reg_mapping, 0, sizeof(libxsmm_gp_reg_mapping) );
#if defined(_WIN32) || defined(__CYGWIN__)
  LIBXSMM_UNUSED(i_xgemm_desc);
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RSI;
#else
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R8;
  /* If we are generating the batchreduce kernel, then we rename the registers */
  if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
    l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RAX;
    l_gp_reg_mapping.gp_reg_a_ptrs = LIBXSMM_X86_GP_REG_RDI;
    l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RBX;
    l_gp_reg_mapping.gp_reg_b_ptrs = LIBXSMM_X86_GP_REG_RSI;
    l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
    l_gp_reg_mapping.gp_reg_reduce_count = LIBXSMM_X86_GP_REG_RCX;
    l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_R8;
    l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R9;
    l_gp_reg_mapping.gp_reg_reduce_loop = LIBXSMM_X86_GP_REG_R10;
  } else if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) {
    l_gp_reg_mapping.gp_reg_a_base = LIBXSMM_X86_GP_REG_RDI;
    l_gp_reg_mapping.gp_reg_a_offset = LIBXSMM_X86_GP_REG_R8;
    l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RAX;
    l_gp_reg_mapping.gp_reg_b_base = LIBXSMM_X86_GP_REG_RSI;
    l_gp_reg_mapping.gp_reg_b_offset = LIBXSMM_X86_GP_REG_R9;
    l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RBX;
    l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
    l_gp_reg_mapping.gp_reg_reduce_count = LIBXSMM_X86_GP_REG_RCX;
    l_gp_reg_mapping.gp_reg_reduce_loop = LIBXSMM_X86_GP_REG_R10;
  } else if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) {
    l_gp_reg_mapping.gp_reg_a_base = LIBXSMM_X86_GP_REG_RDI;
    l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_R8;
    l_gp_reg_mapping.gp_reg_b_base = LIBXSMM_X86_GP_REG_RSI;
    l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RBX;
    l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
    l_gp_reg_mapping.gp_reg_reduce_count = LIBXSMM_X86_GP_REG_RCX;
    l_gp_reg_mapping.gp_reg_reduce_loop = LIBXSMM_X86_GP_REG_R10;
  }
#endif
  l_gp_reg_mapping.gp_reg_decompressed_a = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_bitmap_a = LIBXSMM_X86_GP_REG_R10;
  l_gp_reg_mapping.gp_reg_mloop = LIBXSMM_X86_GP_REG_R12;
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_X86_GP_REG_R13;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_X86_GP_REG_R12;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_X86_GP_REG_R13;
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_X86_GP_REG_R14;
  l_gp_reg_mapping.gp_reg_lda = LIBXSMM_X86_GP_REG_R11;
  l_gp_reg_mapping.gp_reg_ldb = LIBXSMM_X86_GP_REG_R15;
  l_gp_reg_mapping.gp_reg_ldc = LIBXSMM_X86_GP_REG_R14;
  *i_gp_reg_mapping = l_gp_reg_mapping;
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_set_input_in_stack_param_struct( libxsmm_generated_code*   io_generated_code,
    libxsmm_matequation_kernel_config*                  i_micro_kernel_config,
    libxsmm_matequation_gp_reg_mapping*                 i_gp_reg_mapping,
    libxsmm_meqn_elem*                            cur_node,
    unsigned int                                        temp_reg,
    unsigned int                                        ptr_id ) {

  if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
    if (cur_node->info.arg.in_pos >= 0) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          8,
          temp_reg,
          0 );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          temp_reg,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          cur_node->info.arg.in_pos*32,
          temp_reg,
          0 );
    } else {
      libxsmm_generator_meqn_getaddr_stack_tmp_i( io_generated_code, (-1-cur_node->info.arg.in_pos) * i_micro_kernel_config->tmp_size, temp_reg);
    }
  } else {
    if (cur_node->tmp.id >= 0) {
      libxsmm_generator_meqn_getaddr_stack_tmp_i( io_generated_code, cur_node->tmp.id * i_micro_kernel_config->tmp_size, temp_reg);
    } else {
      libxsmm_blasint arg_tmp_id = -1-cur_node->tmp.id;
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          8,
          temp_reg,
          0 );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          temp_reg,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          arg_tmp_id*32,
          temp_reg,
          0 );
    }
  }

  if (ptr_id == 0) {
    libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR4, temp_reg );
  } else if (ptr_id == 1) {
    libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR8, temp_reg );
  } else {
    libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR12, temp_reg );
  }

  /* Setup secondaries if need be */
  if ((cur_node->up->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) &&
      ((cur_node->up->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_GATHER) ||
       (libxsmm_meqn_is_unary_opcode_reduce_cols_idx_kernel(cur_node->up->info.u_op.type) > 0))) {
    if ((cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_node->info.arg.in_pos >= 0)) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          8,
          temp_reg,
          0 );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          temp_reg,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          cur_node->info.arg.in_pos*32+8,
          temp_reg,
          0 );
      libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR5, temp_reg );
    } else {
      fprintf( stderr, "The requested GATHER operation accepts arguments given by the user only...\n" );
      return;
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_set_output_in_stack_param_struct(libxsmm_generated_code*   io_generated_code,
    libxsmm_matequation_kernel_config*                  i_micro_kernel_config,
    libxsmm_matequation_gp_reg_mapping*                 i_gp_reg_mapping,
    libxsmm_meqn_elem*                            cur_node,
    unsigned int                                        temp_reg,
    unsigned int                                        is_last_op ) {
  if (is_last_op > 0) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        16,
        temp_reg,
        0 );
  } else {
    if (cur_node->tmp.id >= 0) {
      libxsmm_generator_meqn_getaddr_stack_tmp_i( io_generated_code, cur_node->tmp.id * i_micro_kernel_config->tmp_size, temp_reg);
    } else {
      libxsmm_blasint arg_tmp_id = -1-cur_node->tmp.id;
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          8,
          temp_reg,
          0 );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          temp_reg,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          arg_tmp_id*32,
          temp_reg,
          0 );
    }
  }

  if (((cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) && (cur_node->info.b_op.is_matmul == 1)) ||
      ((cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) && (cur_node->info.b_op.is_brgemm == 1)) ||
      ((cur_node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) && (cur_node->info.t_op.is_matmul == 1)) ||
      ((cur_node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) && (cur_node->info.t_op.is_brgemm == 1))) {
    libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR12, temp_reg );
  } else {
    if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) {
      libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR8, temp_reg );
    } else if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ||
               (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY && (cur_node->info.t_op.type == LIBXSMM_MELTW_TYPE_TERNARY_MULADD ||  cur_node->info.t_op.type == LIBXSMM_MELTW_TYPE_TERNARY_NMULADD))) {
      libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR12, temp_reg );
    } else if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) {
      libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR16, temp_reg );
    } else {
      /* Should not happen */
    }
  }

  /* Setup secondaries if need be */
  if ((cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) &&
      ((cur_node->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_UNZIP) ||
       (cur_node->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_SCATTER) ||
       ((cur_node->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_RELU) && ((cur_node->info.u_op.flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) ))) {
    if (is_last_op > 0) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          24,
          temp_reg,
          0 );
      libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR9, temp_reg );
    } else {
      if (cur_node->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_SCATTER) {
        fprintf( stderr, "The requested SCATTER operation can only be the head of the equation...\n" );
      } else if (cur_node->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_UNZIP) {
        fprintf( stderr, "The requested UNPACK_TO_BLOCKS operation can only be the head of the equation...\n" );
      } else if (cur_node->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_RELU) {
        fprintf( stderr, "The requested RELU operation with bitmask can only be the head of the equation...\n" );
      }
      return;
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_tmp_stack_scratch_avx_avx512_kernel( libxsmm_generated_code* io_generated_code,
    const libxsmm_meqn_descriptor*          i_mateqn_desc,
    libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
    libxsmm_matequation_kernel_config*      i_micro_kernel_config,
    libxsmm_loop_label_tracker*             io_loop_label_tracker,
    libxsmm_matrix_eqn*                     eqn ) {

  libxsmm_descriptor_blob   blob;
  libxsmm_meltw_descriptor  *meltw_desc = NULL;
#if 0
  libxsmm_gemm_descriptor   *xgemm_desc = NULL;
#endif
  unsigned int timestamp = 0;
  unsigned int last_timestamp;
  unsigned int temp_reg = LIBXSMM_X86_GP_REG_R8;

  if ( eqn == NULL ) {
    fprintf( stderr, "The requested equation does not exist... nothing to JIT,,,\n" );
    return;
  } else {
    last_timestamp = eqn->eqn_root->visit_timestamp;
  }

  i_gp_reg_mapping->gp_reg_mapping_eltwise.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RSI;
  libxsmm_generator_meqn_getaddr_stack_var(  io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR0, LIBXSMM_X86_GP_REG_RSI );

  /* Iterate over the equation tree based on the optimal traversal order and call the proper JITer */
  for (timestamp = 0; timestamp <= last_timestamp; timestamp++) {
    libxsmm_meqn_elem *cur_op = libxsmm_generator_matequation_find_op_at_timestamp(eqn->eqn_root, timestamp);
#if 0
    libxsmm_datatype out_precision = (timestamp == last_timestamp) ? (libxsmm_datatype) LIBXSMM_GETENUM_OUT(i_mateqn_desc->datatype) : cur_op->tmp.dtype;
    libxsmm_datatype in_precision = cur_op->le->tmp.dtype;
#else
    /* FIXME: This approach that avoids intermediate converts needs extra tmps, because when input is BF16 and output is FP32 we cannot reuse/overwrite the same tmp scratch... */
    libxsmm_datatype out_precision = LIBXSMM_DATATYPE_F32;
    libxsmm_datatype in_precision = LIBXSMM_DATATYPE_F32;

    /* Find input precision of op */
    in_precision = cur_op->le->tmp.dtype;

    /* Adjust precisions to F64 if the input is also F64 */
    if (in_precision == LIBXSMM_DATATYPE_F64) {
      in_precision = LIBXSMM_DATATYPE_F64;
      out_precision = LIBXSMM_DATATYPE_F64;
    }

    /* Find sibling if applicable. If it is an Arg, set output precision to  that precision... */
    if (timestamp == last_timestamp) {
      out_precision = (libxsmm_datatype) LIBXSMM_GETENUM_OUT(i_mateqn_desc->datatype);
      cur_op->tmp.ld = i_mateqn_desc->ldo;
    } else {
      out_precision = cur_op->tmp.dtype;
    }
    /* Adjust the tmp precision in the tree */
    /* printf("Node at timestamp %d has input precision %d and  output precision %d\n", timestamp, libxsmm_typesize(in_precision), libxsmm_typesize(out_precision)); */
#endif

    if (((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) && (cur_op->info.b_op.is_matmul == 1)) ||
        ((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) && (cur_op->info.b_op.is_brgemm == 1)) ||
        ((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) && (cur_op->info.t_op.is_matmul == 1)) ||
        ((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) && (cur_op->info.t_op.is_brgemm == 1))) {
      libxsmm_gemm_descriptor *desc = NULL;
      libxsmm_gp_reg_mapping l_gp_reg_mapping;

      /* Setup GEMM descriptor and register mapping */
      libxsmm_generator_matequation_gemm_set_descriptor( io_generated_code, cur_op, &blob, &desc);

      libxsmm_generator_matequation_set_input_in_stack_param_struct( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->le, temp_reg, 0);
      libxsmm_generator_matequation_set_input_in_stack_param_struct( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->ri, temp_reg, 1);
      libxsmm_generator_matequation_set_output_in_stack_param_struct( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op, temp_reg, (timestamp == last_timestamp) );

      /* If BRGEMM, set br count address in struct */
      if (((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) && (cur_op->info.b_op.is_brgemm == 1)) ||
          ((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) && (cur_op->info.t_op.is_brgemm == 1))) {
        libxsmm_x86_instruction_alu_mem( io_generated_code,
            i_micro_kernel_config->alu_mov_instruction,
            i_gp_reg_mapping->gp_reg_param_struct,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            temp_reg,
            0 );
        if (cur_op->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) {
          libxsmm_x86_instruction_alu_mem( io_generated_code,
              i_micro_kernel_config->alu_mov_instruction,
              temp_reg,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              cur_op->info.b_op.op_arg_pos*32+16,
              temp_reg,
              0 );
        } else {
          libxsmm_x86_instruction_alu_mem( io_generated_code,
              i_micro_kernel_config->alu_mov_instruction,
              temp_reg,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              cur_op->info.t_op.op_arg_pos*32+16,
              temp_reg,
              0 );
        }
        libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR2, temp_reg);
      }

      /* Setup stack for fusion related params */
      if (cur_op->fusion_info.xgemm.fused_colbias_add_op == 1) {
        if (cur_op->fusion_info.xgemm.colbias_pos_in_arg >= 0) {
          libxsmm_x86_instruction_alu_mem( io_generated_code,
              i_micro_kernel_config->alu_mov_instruction,
              i_gp_reg_mapping->gp_reg_param_struct,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              8,
              temp_reg,
              0 );
          libxsmm_x86_instruction_alu_mem( io_generated_code,
              i_micro_kernel_config->alu_mov_instruction,
              temp_reg,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              cur_op->fusion_info.xgemm.colbias_pos_in_arg*32,
              temp_reg,
              0 );
        } else {
          libxsmm_generator_meqn_getaddr_stack_tmp_i( io_generated_code, (-1-cur_op->fusion_info.xgemm.colbias_pos_in_arg) * i_micro_kernel_config->tmp_size, temp_reg);
        }
        libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR16, temp_reg);
      }

      if ( (((io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) && (io_generated_code->arch < LIBXSMM_X86_ALLFEAT))) &&
           ( ( LIBXSMM_DATATYPE_BF16 == cur_op->le->tmp.dtype ) ||
             ( LIBXSMM_DATATYPE_I8 == cur_op->le->tmp.dtype ) ) &&
           ((desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) != 0) ) {
        libxsmm_generator_matequation_gemm_set_reg_mapping_amx( desc, &l_gp_reg_mapping );
        /* Since this is a GEMM, store calle-save regs */
        libxsmm_generator_x86_save_gpr_regs( io_generated_code, 0xf008  );
        libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDI );
        /* Call GEMM JITer */
        libxsmm_generator_gemm_amx_kernel( io_generated_code, io_loop_label_tracker, &l_gp_reg_mapping, desc );
        /* Since this is a GEMM, restore calle-save regs */
        libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDI );
        libxsmm_generator_x86_restore_gpr_regs( io_generated_code, 0xf008 );
      } else {
        libxsmm_generator_matequation_gemm_set_reg_mapping( desc, &l_gp_reg_mapping );
        /* Since this is a GEMM, store calle-save regs */
        libxsmm_generator_x86_save_gpr_regs( io_generated_code, 0xf008  );
        libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDI );
        /* Call GEMM JITer */
        libxsmm_generator_gemm_sse_avx_avx2_avx512_kernel( io_generated_code, io_loop_label_tracker, &l_gp_reg_mapping, desc );
        /* Since this is a GEMM, restore calle-save regs */
        libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDI );
        libxsmm_generator_x86_restore_gpr_regs( io_generated_code, 0xf008 );
      }

      /* Restore RSI as struct param ptr */
      libxsmm_generator_meqn_getaddr_stack_var(  io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR0, LIBXSMM_X86_GP_REG_RSI );
    } else {
      if (cur_op->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) {
        /* Prepare struct param */
        libxsmm_generator_matequation_set_input_in_stack_param_struct( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->le,
            temp_reg, 0);
        libxsmm_generator_matequation_set_output_in_stack_param_struct( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op,
            temp_reg, (timestamp == last_timestamp) );

        /* If need be, set n for reduce cols_idx which is a fusion result */
        if (libxsmm_meqn_is_unary_opcode_reduce_cols_idx_kernel(cur_op->info.u_op.type) > 0) {
          libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, temp_reg, cur_op->le->tmp.n);
          libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR7, temp_reg );
          libxsmm_generator_meqn_getaddr_stack_var(  io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR7, temp_reg );
          libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR6, temp_reg );
        }

        /* If need, be set properly the offset param from scratch...  */
        if ((eqn->eqn_root->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (eqn->eqn_root->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_UNZIP) && (timestamp == last_timestamp)) {
          libxsmm_generator_meqn_getval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_CONST_9, temp_reg );
          libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR9, temp_reg );
        }

        /* If DUMP operator set secondary output in stack param struct */
        if ((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (cur_op->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_DUMP)) {
          libxsmm_x86_instruction_alu_mem( io_generated_code,
              i_micro_kernel_config->alu_mov_instruction,
              i_gp_reg_mapping->gp_reg_param_struct,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              0,
              temp_reg,
              0 );
           libxsmm_x86_instruction_alu_mem( io_generated_code,
              i_micro_kernel_config->alu_mov_instruction,
              temp_reg,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              cur_op->info.u_op.op_arg_pos*32,
              temp_reg,
              0 );
           libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR9, temp_reg);
        }

        /* Prepare descriptor */
        libxsmm_generator_matequation_create_unary_descriptor( &blob, cur_op, &meltw_desc, in_precision, out_precision );
      } else if (cur_op->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ||
                 (cur_op->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY && (cur_op->info.t_op.type == LIBXSMM_MELTW_TYPE_TERNARY_MULADD ||  cur_op->info.t_op.type == LIBXSMM_MELTW_TYPE_TERNARY_NMULADD))) {
        libxsmm_generator_matequation_set_input_in_stack_param_struct( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->le,
            temp_reg, 0);
        libxsmm_generator_matequation_set_input_in_stack_param_struct( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->ri,
            temp_reg, 1);
        libxsmm_generator_matequation_set_output_in_stack_param_struct( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op,
            temp_reg, (timestamp == last_timestamp) );
        libxsmm_generator_matequation_create_binary_descriptor( &blob, cur_op, &meltw_desc, in_precision, out_precision );
      } else if (cur_op->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) {
        libxsmm_generator_matequation_set_input_in_stack_param_struct( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->le,
            temp_reg, 0);
        libxsmm_generator_matequation_set_input_in_stack_param_struct( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->ri,
            temp_reg, 1);
        libxsmm_generator_matequation_set_input_in_stack_param_struct( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->r2,
            temp_reg, 2);
        libxsmm_generator_matequation_set_output_in_stack_param_struct( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op,
            temp_reg, (timestamp == last_timestamp) );
        libxsmm_generator_matequation_create_ternary_descriptor( &blob, cur_op, &meltw_desc, in_precision, out_precision );
      } else {
        /* This should not happen */
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
        return;
      }
      /* Configure the unary-binary microkernel */
      libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
      /* If GELU or GELU inv and architecture < AVX512, set the priper rbp offsets of stack... */
      if ((io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && (cur_op->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && ((cur_op->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_GELU) || (cur_op->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV))  ) {
        i_micro_kernel_config->meltw_kernel_config.rbp_offs_thres = libxsmm_generator_mateqn_get_rbp_relative_offset(LIBXSMM_MEQN_STACK_VAR_CONST_0);
        i_micro_kernel_config->meltw_kernel_config.rbp_offs_signmask = libxsmm_generator_mateqn_get_rbp_relative_offset(LIBXSMM_MEQN_STACK_VAR_CONST_1);
        i_micro_kernel_config->meltw_kernel_config.rbp_offs_absmask = libxsmm_generator_mateqn_get_rbp_relative_offset(LIBXSMM_MEQN_STACK_VAR_CONST_2);
        i_micro_kernel_config->meltw_kernel_config.rbp_offs_scale = libxsmm_generator_mateqn_get_rbp_relative_offset(LIBXSMM_MEQN_STACK_VAR_CONST_3);
        i_micro_kernel_config->meltw_kernel_config.rbp_offs_shifter = libxsmm_generator_mateqn_get_rbp_relative_offset(LIBXSMM_MEQN_STACK_VAR_CONST_4);
        i_micro_kernel_config->meltw_kernel_config.rbp_offs_half = libxsmm_generator_mateqn_get_rbp_relative_offset(LIBXSMM_MEQN_STACK_VAR_CONST_5);
      }
      /* Call proper JITer */
      if ((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (libxsmm_meqn_is_unary_opcode_reduce_kernel(meltw_desc->param) > 0)) {
        if ((meltw_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS) > 0) {
          libxsmm_generator_meltw_setup_stack_frame( io_generated_code, meltw_desc, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config);
          libxsmm_generator_reduce_rows_avx512_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
          libxsmm_generator_meltw_destroy_stack_frame(  io_generated_code, meltw_desc, &i_micro_kernel_config->meltw_kernel_config );
        } else if ((meltw_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS) > 0) {
          libxsmm_generator_reduce_cols_avx512_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
        } else {
          /* This should not happen */
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
          return;
        }
      } else if ((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (libxsmm_meqn_is_unary_opcode_reduce_cols_idx_kernel(meltw_desc->param) > 0)) {
        libxsmm_generator_reduce_cols_index_avx512_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
      } else if ((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (libxsmm_meqn_is_unary_opcode_transform_kernel(meltw_desc->param) > 0)) {
        libxsmm_generator_transform_x86_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
      } else if ((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && ((cur_op->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_GATHER) || (cur_op->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_SCATTER) )) {
        libxsmm_generator_gather_scatter_avx_avx512_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
      } else {
        libxsmm_generator_unary_binary_avx512_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
      }
    }
  }
}

