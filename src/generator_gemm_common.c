/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include "generator_gemm_common.h"
#include "generator_common.h"
#include "generator_x86_instructions.h"
#include "generator_common_x86.h"
#include "generator_mateltwise_unary_binary_avx_avx512.h"
#include "generator_mateltwise_sse_avx_avx512.h"
#include "generator_mateltwise_transform_common.h"
#include "generator_mateltwise_transform_avx512.h"
#include "generator_mateltwise_transform_avx.h"

LIBXSMM_API_INTERN unsigned int libxsmm_x86_is_Amxfp4_Bi8_gemm ( const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  unsigned int result =  (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_MXFP4_VNNI8_INTLV) > 0) &&
                         ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
                          (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) &&
                          (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )))) ? 1 : 0;
                         return result;
}

LIBXSMM_API_INTERN unsigned int libxsmm_x86_is_Amxfp4_Bfp32_gemm ( const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  unsigned int result =  (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_MXFP4_VNNI2) > 0) &&
                         ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
                          (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) &&
                          (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )))) ? 1 : 0;
  return result;
}

LIBXSMM_API_INTERN unsigned int libxsmm_x86_is_Amxfp4_Bbf16_gemm ( const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  unsigned int result = (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_MXFP4_VNNI2) > 0) &&
                         ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
                          (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) &&
                          (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )))) ? 1 : 0;
  return result;
}

LIBXSMM_API_INTERN unsigned int libxsmm_x86_is_Ai4_Bi8_gemm ( const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  unsigned int result = (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_INT4_VNNI8_INTLV) > 0) &&
                         ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
                          (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) &&
                          (LIBXSMM_DATATYPE_I32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )))) ? 1 : 0;
  return result;
}

LIBXSMM_API_INTERN unsigned int libxsmm_x86_is_Abf8_Bbf16_gemm ( const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  unsigned int result = ( (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
                          (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) &&
                          (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
  return result;
}

LIBXSMM_API_INTERN unsigned int libxsmm_x86_is_Ahf8_Bbf16_gemm ( const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  unsigned int result = ( (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
                          (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) &&
                          (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
  return result;
}

LIBXSMM_API_INTERN void libxsmm_generator_gemm_apply_ops_input_tensor_and_store_to_stack( libxsmm_generated_code*    io_generated_code,
                                                                                      libxsmm_loop_label_tracker*    io_loop_label_tracker,
                                                                                      libxsmm_micro_kernel_config*   i_micro_kernel_config,
                                                                                      libxsmm_gemm_descriptor*       i_xgemm_desc,
                                                                                      unsigned int                   i_gp_reg_in,
                                                                                      unsigned int                   i_struct_gp_reg,
                                                                                      unsigned int                   i_tmp_reg,
                                                                                      unsigned int                   i_loop_reg,
                                                                                      unsigned int                   i_bound_reg,
                                                                                      unsigned                       i_tmp_reg2,
                                                                                      libxsmm_meltw_unary_type       i_op_type,
                                                                                      libxsmm_blasint                i_m,
                                                                                      libxsmm_blasint                i_n,
                                                                                      libxsmm_blasint                i_ldi,
                                                                                      libxsmm_blasint                i_ldo,
                                                                                      libxsmm_blasint                i_tensor_stride,
                                                                                      libxsmm_datatype               i_in_dtype,
                                                                                      libxsmm_datatype               i_comp_dtype,
                                                                                      libxsmm_datatype               i_out_dtype,
                                                                                      libxsmm_gemm_stack_var         i_stack_var_offs_ptr,
                                                                                      libxsmm_gemm_stack_var         i_stack_var_scratch_ptr,
                                                                                      libxsmm_gemm_stack_var         i_stack_var_dst_ptr,
                                                                                      libxsmm_meltw_unary_type       i_op2_type,
                                                                                      libxsmm_blasint                i_m2,
                                                                                      libxsmm_blasint                i_n2,
                                                                                      libxsmm_blasint                i_ldi2,
                                                                                      libxsmm_blasint                i_ldo2,
                                                                                      libxsmm_datatype               i_in2_dtype,
                                                                                      libxsmm_datatype               i_comp2_dtype,
                                                                                      libxsmm_datatype               i_out2_dtype ) {
  libxsmm_descriptor_blob           l_meltw_blob;
  libxsmm_mateltwise_kernel_config  l_mateltwise_kernel_config;
  libxsmm_mateltwise_gp_reg_mapping l_mateltwise_gp_reg_mapping;
  const libxsmm_meltw_descriptor *  l_mateltwise_desc;
  int is_stride_brgemm        = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0) ? 1 : 0;
  int is_offset_brgemm        = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) > 0) ? 1 : 0;
  int is_address_brgemm       = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0) ? 1 : 0;
  int is_brgemm               = ((is_stride_brgemm == 1) || (is_offset_brgemm == 1) || (is_address_brgemm == 1)) ? 1 : 0;

  l_mateltwise_gp_reg_mapping.gp_reg_param_struct = i_struct_gp_reg;
  libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MELTW_STRUCT_PTR, i_struct_gp_reg );

  /* Loop over all batch-reduce iterations to cover all tensor blocks */
  if (is_brgemm > 0) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, i_bound_reg );
    libxsmm_generator_generic_loop_header_no_idx_inc( io_generated_code, io_loop_label_tracker, i_loop_reg, 0);
  }
  /* Setup input pointer of input in eltwise struct */
  if (is_offset_brgemm > 0) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_in, i_tmp_reg2);
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, i_stack_var_offs_ptr, i_tmp_reg );
    libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_tmp_reg, i_loop_reg, 8, 0, i_tmp_reg, 0 );
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_tmp_reg, i_gp_reg_in);
  }
  if (is_address_brgemm > 0) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_in, i_tmp_reg2);
    libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_in, i_loop_reg, 8, 0, i_gp_reg_in, 0 );
  }
  libxsmm_x86_instruction_alu_mem( io_generated_code,
          LIBXSMM_X86_INSTR_MOVQ,
          i_struct_gp_reg,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          32,
          i_gp_reg_in,
          1 );
  if ((is_offset_brgemm > 0) || (is_address_brgemm > 0)) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_tmp_reg2, i_gp_reg_in);
  }

  if (i_op2_type != LIBXSMM_MELTW_TYPE_UNARY_NONE) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, i_stack_var_scratch_ptr, i_tmp_reg );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
            LIBXSMM_X86_INSTR_MOVQ,
            i_struct_gp_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            64,
            i_tmp_reg,
            1 );
  } else {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, i_stack_var_dst_ptr, i_tmp_reg );
    if (is_brgemm > 0) {
      libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_loop_reg, i_tmp_reg2);
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_tmp_reg2, (long long)LIBXSMM_TYPESIZE(i_out_dtype) * i_m * i_n);
      libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_tmp_reg2, i_tmp_reg);
    }
    libxsmm_x86_instruction_alu_mem( io_generated_code,
            LIBXSMM_X86_INSTR_MOVQ,
            i_struct_gp_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            64,
            i_tmp_reg,
            1 );
  }
  l_mateltwise_desc = libxsmm_meltw_descriptor_init2(&l_meltw_blob, i_in_dtype, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED,
    i_comp_dtype, i_out_dtype, i_m, i_n, i_ldi, i_ldo, 0, 0,
    0, LIBXSMM_CAST_USHORT(i_op_type), LIBXSMM_MELTW_OPERATION_UNARY);
  libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_mateltwise_kernel_config, l_mateltwise_desc );
  if (libxsmm_meqn_is_unary_opcode_transform_kernel(LIBXSMM_CAST_USHORT(i_op_type)) > 0) {
    libxsmm_generator_transform_x86_microkernel( io_generated_code, io_loop_label_tracker, &l_mateltwise_gp_reg_mapping, &l_mateltwise_kernel_config, l_mateltwise_desc );
  } else {
    libxsmm_generator_unary_binary_avx512_microkernel( io_generated_code, io_loop_label_tracker, &l_mateltwise_gp_reg_mapping, &l_mateltwise_kernel_config, l_mateltwise_desc );
  }

  if (i_op2_type != LIBXSMM_MELTW_TYPE_UNARY_NONE) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, i_stack_var_scratch_ptr, i_tmp_reg );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
            LIBXSMM_X86_INSTR_MOVQ,
            i_struct_gp_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            32,
            i_tmp_reg,
            1 );
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, i_stack_var_dst_ptr, i_tmp_reg );
    if (is_brgemm > 0) {
      libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_loop_reg, i_tmp_reg2);
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_tmp_reg2, (long long)LIBXSMM_TYPESIZE(i_out2_dtype) * i_m2 * i_n2);
      libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_tmp_reg2, i_tmp_reg);
    }
    libxsmm_x86_instruction_alu_mem( io_generated_code,
            LIBXSMM_X86_INSTR_MOVQ,
            i_struct_gp_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            64,
            i_tmp_reg,
            1 );
    l_mateltwise_desc = libxsmm_meltw_descriptor_init2(&l_meltw_blob, i_in2_dtype, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED,
      i_comp2_dtype, i_out2_dtype, i_m2, i_n2, i_ldi2, i_ldo2, 0, 0,
      0, LIBXSMM_CAST_USHORT(i_op2_type), LIBXSMM_MELTW_OPERATION_UNARY);
    libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_mateltwise_kernel_config, l_mateltwise_desc );
    if (libxsmm_meqn_is_unary_opcode_transform_kernel(LIBXSMM_CAST_USHORT(i_op2_type)) > 0) {
      libxsmm_generator_transform_x86_microkernel( io_generated_code, io_loop_label_tracker, &l_mateltwise_gp_reg_mapping, &l_mateltwise_kernel_config, l_mateltwise_desc );
    } else {
      libxsmm_generator_unary_binary_avx512_microkernel( io_generated_code, io_loop_label_tracker, &l_mateltwise_gp_reg_mapping, &l_mateltwise_kernel_config, l_mateltwise_desc );
    }
  }

  if (is_brgemm > 0) {
    if (is_stride_brgemm > 0) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_in, i_tensor_stride);
    }
    libxsmm_generator_generic_loop_footer_with_idx_inc_reg_bound( io_generated_code, io_loop_label_tracker, i_loop_reg, 1, i_bound_reg);
  }
}

/* Setup A transpose tensor in stack */
LIBXSMM_API_INTERN void libxsmm_generator_gemm_setup_A_vnni_or_trans_B_vnni_or_trans_tensor_to_stack( libxsmm_generated_code*       io_generated_code,
                                                                                                      libxsmm_loop_label_tracker*    io_loop_label_tracker,
                                                                                                      const libxsmm_gp_reg_mapping*  i_gp_reg_mapping,
                                                                                                      libxsmm_micro_kernel_config*   i_micro_kernel_config,
                                                                                                      libxsmm_gemm_descriptor*       i_xgemm_desc,
                                                                                                      const libxsmm_gemm_descriptor* i_xgemm_desc_orig,
                                                                                                      libxsmm_datatype               i_in_dtype,
                                                                                                      unsigned int                   i_is_amx ) {
  int is_stride_brgemm        = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0) ? 1 : 0;
  int is_offset_brgemm        = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) > 0) ? 1 : 0;
  int is_address_brgemm       = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0) ? 1 : 0;
  int is_brgemm               = ((is_stride_brgemm == 1) || (is_offset_brgemm == 1) || (is_address_brgemm == 1)) ? 1 : 0;
  unsigned int struct_gp_reg  = LIBXSMM_X86_GP_REG_R15;
  unsigned int tmp_reg        = LIBXSMM_X86_GP_REG_R14;
  unsigned int loop_reg       = LIBXSMM_X86_GP_REG_R13;
  unsigned int bound_reg      = LIBXSMM_X86_GP_REG_R12;
  unsigned int tmp_reg2       = LIBXSMM_X86_GP_REG_RDX;
  unsigned int l_reg_a        = (i_is_amx != 0) ? LIBXSMM_X86_GP_REG_RDI : i_gp_reg_mapping->gp_reg_a;
  unsigned int l_reg_b        = (i_is_amx != 0) ? LIBXSMM_X86_GP_REG_RSI : i_gp_reg_mapping->gp_reg_b;
  unsigned short gp_save_bitmask = 0x2 | 0x4 | 0x100 | 0x200 | 0x400 | 0x800 | 0x1000 | 0x2000 | 0x4000 | 0x8000;
  libxsmm_meltw_unary_type l_trans_unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT;
  libxsmm_datatype l_trans_dt = i_in_dtype;
  unsigned int l_trans_m = i_xgemm_desc_orig->k;
  unsigned int l_trans_n = i_xgemm_desc_orig->m;
  unsigned int l_trans_ldi = i_xgemm_desc_orig->lda;
  unsigned int l_trans_ldo = i_xgemm_desc_orig->m;

  /* In this case we have to call Unary TPP (copy) for B and it is not supported for arch < avx */
  if ((i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX) && (is_offset_brgemm > 0 || is_address_brgemm > 0)) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
    return;
  }

  libxsmm_generator_x86_save_gpr_regs( io_generated_code, gp_save_bitmask);

  if ( (i_micro_kernel_config->atvnni_gemm_stack_alloc_tensors !=0) || (i_micro_kernel_config->atvnni_btrans_gemm_stack_alloc_tensors !=0) ) {
    l_trans_unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT;
    l_trans_dt = LIBXSMM_DATATYPE_F32;
    l_trans_m = i_xgemm_desc_orig->k/2;
    l_trans_n = i_xgemm_desc_orig->m;
    l_trans_ldi = i_xgemm_desc_orig->lda/2;
    l_trans_ldo = i_xgemm_desc_orig->m;
  } else if ( (i_micro_kernel_config->avnni_gemm_stack_alloc_tensors !=0) || (i_micro_kernel_config->avnni_btrans_gemm_stack_alloc_tensors !=0) ) {
    l_trans_unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2;
    l_trans_dt = i_in_dtype;
    l_trans_m = i_xgemm_desc_orig->m;
    l_trans_n = i_xgemm_desc_orig->k;
    l_trans_ldi = i_xgemm_desc_orig->lda;
    l_trans_ldo = i_xgemm_desc_orig->m;
  }

  /* Setup A in stack (if A in vnni perform vnni4->norm and then fp8->fp32, else perform fp8->fp32 only) */
  libxsmm_generator_gemm_apply_ops_input_tensor_and_store_to_stack( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
      l_reg_a, struct_gp_reg, tmp_reg, loop_reg, bound_reg, tmp_reg2,
      l_trans_unary_type, l_trans_m, l_trans_n, l_trans_ldi, l_trans_ldo, LIBXSMM_CAST_BLASINT(i_xgemm_desc_orig->c1),
      l_trans_dt, l_trans_dt, l_trans_dt,
      LIBXSMM_GEMM_STACK_VAR_A_OFFS_BRGEMM_PTR, LIBXSMM_GEMM_STACK_VAR_A_SCRATCH_PTR, LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR,
      LIBXSMM_MELTW_TYPE_UNARY_NONE, 0, 0, 0, 0, (libxsmm_datatype)0, (libxsmm_datatype)0, (libxsmm_datatype)0);
  /* Adjust A/B gp_regs to point to the fp32 tensors in stack */
  libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR, tmp_reg );
  libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, tmp_reg, l_reg_a);

  /* In this case we have to copy over also B in strided BRGEMM format */
  if ( (i_micro_kernel_config->avnni_btrans_gemm_stack_alloc_tensors != 0) || (i_micro_kernel_config->atvnni_btrans_gemm_stack_alloc_tensors != 0) ) {
    libxsmm_generator_gemm_apply_ops_input_tensor_and_store_to_stack( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
        l_reg_b, struct_gp_reg, tmp_reg, loop_reg, bound_reg, tmp_reg2,
        LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, i_xgemm_desc->n, i_xgemm_desc->k, i_xgemm_desc_orig->ldb, i_xgemm_desc->k, LIBXSMM_CAST_BLASINT(i_xgemm_desc_orig->c2),
        i_in_dtype, i_in_dtype, i_in_dtype,
        LIBXSMM_GEMM_STACK_VAR_B_OFFS_BRGEMM_PTR, LIBXSMM_GEMM_STACK_VAR_A_SCRATCH_PTR, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR,
        LIBXSMM_MELTW_TYPE_UNARY_NONE, 0, 0, 0, 0, (libxsmm_datatype)0, (libxsmm_datatype)0, (libxsmm_datatype)0);
    /* Adjust B gp_regs to point to the transpose tensors in stack */
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, tmp_reg );
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, tmp_reg, l_reg_b);
    i_xgemm_desc->ldb = i_xgemm_desc->k;
    i_xgemm_desc->c2 = (long long)LIBXSMM_TYPESIZE(i_in_dtype) * i_xgemm_desc->n * i_xgemm_desc->k;
  } else {
    if ( (is_offset_brgemm > 0) || (is_address_brgemm > 0) ) {
      unsigned int l_is_trans_b = (i_xgemm_desc_orig->flags & LIBXSMM_GEMM_FLAG_TRANS_B);
      libxsmm_generator_gemm_apply_ops_input_tensor_and_store_to_stack( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
          l_reg_b, struct_gp_reg, tmp_reg, loop_reg, bound_reg, tmp_reg2,
          LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, (l_is_trans_b) ? i_xgemm_desc->n : i_xgemm_desc->k, (l_is_trans_b) ? i_xgemm_desc->k : i_xgemm_desc->n, i_xgemm_desc_orig->ldb, (l_is_trans_b) ? i_xgemm_desc->n : i_xgemm_desc->k, LIBXSMM_CAST_BLASINT(i_xgemm_desc_orig->c2),
          i_in_dtype, i_in_dtype, i_in_dtype,
          LIBXSMM_GEMM_STACK_VAR_B_OFFS_BRGEMM_PTR, LIBXSMM_GEMM_STACK_VAR_A_SCRATCH_PTR, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR,
          LIBXSMM_MELTW_TYPE_UNARY_NONE, 0, 0, 0, 0, (libxsmm_datatype)0, (libxsmm_datatype)0, (libxsmm_datatype)0);
      i_xgemm_desc->ldb = (l_is_trans_b) ? i_xgemm_desc->n : i_xgemm_desc->k;
      /* Adjust B gp_regs to point to the transpose tensors in stack */
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, tmp_reg );
      libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, tmp_reg, l_reg_b);
    }
  }

  /* Adjust descriptor for internal strided BRGEMM */
  if (is_brgemm > 0) {
    if (is_offset_brgemm > 0) {
      i_xgemm_desc->flags = i_xgemm_desc->flags ^ LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET;
      i_xgemm_desc->flags = i_xgemm_desc->flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE;
      i_xgemm_desc->c2 = (long long)LIBXSMM_TYPESIZE(i_in_dtype) * i_xgemm_desc->n * i_xgemm_desc->k;
    }
    if (is_address_brgemm > 0) {
      i_xgemm_desc->flags = i_xgemm_desc->flags ^ LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS;
      i_xgemm_desc->flags = i_xgemm_desc->flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE;
      i_xgemm_desc->c2 = (long long)LIBXSMM_TYPESIZE(i_in_dtype) * i_xgemm_desc->n * i_xgemm_desc->k;
    }
    i_xgemm_desc->c1 = (long long)LIBXSMM_TYPESIZE(i_in_dtype) * i_xgemm_desc->m * i_xgemm_desc->k;
  }

  libxsmm_generator_x86_restore_gpr_regs( io_generated_code, gp_save_bitmask);
}

/* Setup B vnnit tensor in stack and A if need be*/
LIBXSMM_API_INTERN void libxsmm_generator_gemm_setup_B_vnni2t_to_norm_into_stack(   libxsmm_generated_code*        io_generated_code,
                                                                                    libxsmm_loop_label_tracker*    io_loop_label_tracker,
                                                                                    const libxsmm_gp_reg_mapping*  i_gp_reg_mapping,
                                                                                    libxsmm_micro_kernel_config*   i_micro_kernel_config,
                                                                                    libxsmm_gemm_descriptor*       i_xgemm_desc,
                                                                                    const libxsmm_gemm_descriptor* i_xgemm_desc_orig,
                                                                                    libxsmm_datatype               i_in_dtype ) {
  int is_stride_brgemm        = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0) ? 1 : 0;
  int is_offset_brgemm        = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) > 0) ? 1 : 0;
  int is_address_brgemm       = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0) ? 1 : 0;
  int is_brgemm               = ((is_stride_brgemm == 1) || (is_offset_brgemm == 1) || (is_address_brgemm == 1)) ? 1 : 0;
  unsigned int struct_gp_reg  = LIBXSMM_X86_GP_REG_R15;
  unsigned int tmp_reg        = LIBXSMM_X86_GP_REG_R14;
  unsigned int loop_reg       = LIBXSMM_X86_GP_REG_R13;
  unsigned int bound_reg      = LIBXSMM_X86_GP_REG_R12;
  unsigned int tmp_reg2       = LIBXSMM_X86_GP_REG_RDX;
  unsigned short gp_save_bitmask = 0x2 | 0x4 | 0x100 | 0x200 | 0x400 | 0x800 | 0x1000 | 0x2000 | 0x4000 | 0x8000;

  /* In this case we have to call Unary TPP (copy) for B and it is not supported for arch < avx */
  if ((i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX) && (is_offset_brgemm > 0 || is_address_brgemm > 0)) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
    return;
  }

  libxsmm_generator_x86_save_gpr_regs( io_generated_code, gp_save_bitmask);

  /* Setup B in stack */
  libxsmm_generator_gemm_apply_ops_input_tensor_and_store_to_stack( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
      i_gp_reg_mapping->gp_reg_b, struct_gp_reg, tmp_reg, loop_reg, bound_reg, tmp_reg2,
      LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2T_TO_NORM, i_xgemm_desc_orig->n, i_xgemm_desc_orig->k, i_xgemm_desc_orig->ldb, i_xgemm_desc_orig->k, LIBXSMM_CAST_BLASINT(i_xgemm_desc_orig->c2),
      i_in_dtype, i_in_dtype, i_in_dtype,
      LIBXSMM_GEMM_STACK_VAR_B_OFFS_BRGEMM_PTR, LIBXSMM_GEMM_STACK_VAR_A_SCRATCH_PTR, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR,
      LIBXSMM_MELTW_TYPE_UNARY_NONE, 0, 0, 0, 0, (libxsmm_datatype)0, (libxsmm_datatype)0, (libxsmm_datatype)0);

  /* Adjust A/B gp_regs to point to the fp32 tensors in stack */
  libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, tmp_reg );
  libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, tmp_reg, i_gp_reg_mapping->gp_reg_b);

  /* In this case we have to copy over also A in strided BRGEMM format */
  if ( (is_offset_brgemm > 0) || (is_address_brgemm > 0) ) {
    libxsmm_generator_gemm_apply_ops_input_tensor_and_store_to_stack( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
        i_gp_reg_mapping->gp_reg_a, struct_gp_reg, tmp_reg, loop_reg, bound_reg, tmp_reg2,
        LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, i_xgemm_desc->m*2, i_xgemm_desc->k/2, i_xgemm_desc_orig->lda*2, i_xgemm_desc->m*2, LIBXSMM_CAST_BLASINT(i_xgemm_desc_orig->c1),
        i_in_dtype, i_in_dtype, i_in_dtype,
        LIBXSMM_GEMM_STACK_VAR_A_OFFS_BRGEMM_PTR, LIBXSMM_GEMM_STACK_VAR_A_SCRATCH_PTR, LIBXSMM_GEMM_STACK_VAR_A_EMU_PTR,
        LIBXSMM_MELTW_TYPE_UNARY_NONE, 0, 0, 0, 0, (libxsmm_datatype)0, (libxsmm_datatype)0, (libxsmm_datatype)0);
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_A_EMU_PTR, tmp_reg );
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, tmp_reg, i_gp_reg_mapping->gp_reg_a);
  }

  /* Adjust descriptor for internal strided BRGEMM */
  if (is_brgemm > 0) {
    if (is_offset_brgemm > 0) {
      i_xgemm_desc->flags = i_xgemm_desc->flags ^ LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET;
      i_xgemm_desc->flags = i_xgemm_desc->flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE;
      i_xgemm_desc->lda = i_xgemm_desc->m;
      i_xgemm_desc->c1 = (long long)LIBXSMM_TYPESIZE(i_in_dtype) * i_xgemm_desc->m * i_xgemm_desc->k;
    }
    if (is_address_brgemm > 0) {
      i_xgemm_desc->flags = i_xgemm_desc->flags ^ LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS;
      i_xgemm_desc->flags = i_xgemm_desc->flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE;
      i_xgemm_desc->lda = i_xgemm_desc->m;
      i_xgemm_desc->c1 = (long long)LIBXSMM_TYPESIZE(i_in_dtype) * i_xgemm_desc->m * i_xgemm_desc->k;
    }
    i_xgemm_desc->c2 = (long long)LIBXSMM_TYPESIZE(i_in_dtype) * i_xgemm_desc->n * i_xgemm_desc->k;
  }

  libxsmm_generator_x86_restore_gpr_regs( io_generated_code, gp_save_bitmask);
}

/* Setup A (in vnni4 or flat) and B bf8 tensors as fp32 tensors in stack */
LIBXSMM_API_INTERN void libxsmm_generator_gemm_setup_f8_AB_tensors_to_stack(  libxsmm_generated_code*       io_generated_code,
                                                                              libxsmm_loop_label_tracker*    io_loop_label_tracker,
                                                                              const libxsmm_gp_reg_mapping*  i_gp_reg_mapping,
                                                                              libxsmm_micro_kernel_config*   i_micro_kernel_config,
                                                                              libxsmm_gemm_descriptor*       i_xgemm_desc,
                                                                              const libxsmm_gemm_descriptor* i_xgemm_desc_orig,
                                                                              libxsmm_datatype               i_in_dtype,
                                                                              libxsmm_datatype               i_target_dtype ) {
  int is_stride_brgemm        = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0) ? 1 : 0;
  int is_offset_brgemm        = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) > 0) ? 1 : 0;
  int is_address_brgemm       = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0) ? 1 : 0;
  int is_brgemm               = ((is_stride_brgemm == 1) || (is_offset_brgemm == 1) || (is_address_brgemm == 1)) ? 1 : 0;
  unsigned int a_in_vnni      = ((i_xgemm_desc_orig->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) ? 1 : 0;
  unsigned int struct_gp_reg  = LIBXSMM_X86_GP_REG_R15;
  unsigned int tmp_reg        = LIBXSMM_X86_GP_REG_R14;
  unsigned int loop_reg       = LIBXSMM_X86_GP_REG_R13;
  unsigned int bound_reg      = LIBXSMM_X86_GP_REG_R12;
  unsigned int tmp_reg2       = LIBXSMM_X86_GP_REG_RDX;
  unsigned short gp_save_bitmask = 0x2 | 0x4 | 0x100 | 0x200 | 0x400 | 0x800 | 0x1000 | 0x2000 | 0x4000 | 0x8000;
  libxsmm_generator_x86_save_gpr_regs( io_generated_code, gp_save_bitmask);

  /* When target dtype is FP32 */
  /* Setup A in stack (if A in vnni perform vnni4->norm and then fp8->fp32, else perform fp8->fp32 only) */

  /* When target dtype is BF16 */
  /* If A is originally in VNNI4 format: First convert VNNI4 to VNNI2 (8bit)*/
  /* If A is originally in VNNI4 format: Second convert BF8 to BF16 */
  /* If A is originally in flat format: First convert BF8 to BF16 */
  /* If A is originally in flat format: Second transform NORM to VNNI2 */
  if (a_in_vnni > 0) {
    libxsmm_generator_gemm_apply_ops_input_tensor_and_store_to_stack( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
        i_gp_reg_mapping->gp_reg_a, struct_gp_reg, tmp_reg, loop_reg, bound_reg, tmp_reg2,
        (i_target_dtype == LIBXSMM_DATATYPE_F32) ? LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_NORM : LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI2, i_xgemm_desc->m, i_xgemm_desc->k, i_xgemm_desc_orig->lda, i_xgemm_desc->lda, LIBXSMM_CAST_BLASINT(i_xgemm_desc_orig->c1),
        i_in_dtype, i_in_dtype, i_in_dtype,
        LIBXSMM_GEMM_STACK_VAR_A_OFFS_BRGEMM_PTR, LIBXSMM_GEMM_STACK_VAR_A_SCRATCH_PTR, LIBXSMM_GEMM_STACK_VAR_A_EMU_PTR,
        LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, i_xgemm_desc->m, i_xgemm_desc->k, i_xgemm_desc->lda, i_xgemm_desc->lda,
        i_in_dtype, LIBXSMM_DATATYPE_F32, i_target_dtype);
  } else {
    libxsmm_generator_gemm_apply_ops_input_tensor_and_store_to_stack( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
        i_gp_reg_mapping->gp_reg_a, struct_gp_reg, tmp_reg, loop_reg, bound_reg, tmp_reg2,
        LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, i_xgemm_desc->m, i_xgemm_desc->k, i_xgemm_desc_orig->lda, i_xgemm_desc->lda, LIBXSMM_CAST_BLASINT(i_xgemm_desc_orig->c1),
        i_in_dtype, LIBXSMM_DATATYPE_F32, i_target_dtype,
        LIBXSMM_GEMM_STACK_VAR_A_OFFS_BRGEMM_PTR, LIBXSMM_GEMM_STACK_VAR_A_SCRATCH_PTR, LIBXSMM_GEMM_STACK_VAR_A_EMU_PTR,
        (i_target_dtype == LIBXSMM_DATATYPE_F32) ? LIBXSMM_MELTW_TYPE_UNARY_NONE : LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2, i_xgemm_desc->m, i_xgemm_desc->k, i_xgemm_desc->lda, i_xgemm_desc->lda,
        i_target_dtype, i_target_dtype, i_target_dtype);
  }

  /* Setup B in stack */
  libxsmm_generator_gemm_apply_ops_input_tensor_and_store_to_stack( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
      i_gp_reg_mapping->gp_reg_b, struct_gp_reg, tmp_reg, loop_reg, bound_reg, tmp_reg2,
      LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, i_xgemm_desc->k, i_xgemm_desc->n, i_xgemm_desc_orig->ldb, i_xgemm_desc->k, LIBXSMM_CAST_BLASINT(i_xgemm_desc_orig->c2),
      i_in_dtype, LIBXSMM_DATATYPE_F32, i_target_dtype,
      LIBXSMM_GEMM_STACK_VAR_B_OFFS_BRGEMM_PTR, LIBXSMM_GEMM_STACK_VAR_A_SCRATCH_PTR, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR,
      LIBXSMM_MELTW_TYPE_UNARY_NONE, 0, 0, 0, 0, (libxsmm_datatype)0, (libxsmm_datatype)0, (libxsmm_datatype)0);

  /* Adjust A/B gp_regs to point to the fp32 tensors in stack */
  libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_A_EMU_PTR, tmp_reg );
  libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, tmp_reg, i_gp_reg_mapping->gp_reg_a);
  libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, tmp_reg );
  libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, tmp_reg, i_gp_reg_mapping->gp_reg_b);

  /* Adjust descriptor for internal strided BRGEMM */
  if (is_brgemm > 0) {
    if (is_offset_brgemm > 0) {
      i_xgemm_desc->flags = i_xgemm_desc->flags ^ LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET;
      i_xgemm_desc->flags = i_xgemm_desc->flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE;
    }
    if (is_address_brgemm > 0) {
      i_xgemm_desc->flags = i_xgemm_desc->flags ^ LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS;
      i_xgemm_desc->flags = i_xgemm_desc->flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE;
    }
    i_xgemm_desc->c1 = (long long)LIBXSMM_TYPESIZE(i_target_dtype) * i_xgemm_desc->m * i_xgemm_desc->k;
    i_xgemm_desc->c2 = (long long)LIBXSMM_TYPESIZE(i_target_dtype) * i_xgemm_desc->n * i_xgemm_desc->k;
  }

  libxsmm_generator_x86_restore_gpr_regs( io_generated_code, gp_save_bitmask);
}

LIBXSMM_API_INTERN void libxsmm_generator_gemm_apply_opA_opB( libxsmm_generated_code*        io_generated_code,
                                                              libxsmm_loop_label_tracker*    io_loop_label_tracker,
                                                              const libxsmm_gp_reg_mapping*  i_gp_reg_mapping,
                                                              libxsmm_micro_kernel_config*   i_micro_kernel_config,
                                                              libxsmm_gemm_descriptor*       i_xgemm_desc,
                                                              const libxsmm_gemm_descriptor* i_xgemm_desc_orig ) {
  if ( ( i_xgemm_desc_orig->flags & LIBXSMM_GEMM_FLAG_TRANS_A) && (i_xgemm_desc_orig->m != 0) && (i_xgemm_desc_orig->k != 0) ) {
    /* if A needs to be transposed, use scratch in stack */
    libxsmm_generator_gemm_setup_A_vnni_or_trans_B_vnni_or_trans_tensor_to_stack( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, i_xgemm_desc_orig, (libxsmm_datatype) LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc_orig->datatype ), 0);
  } else if ( (i_micro_kernel_config->avnni_gemm_stack_alloc_tensors !=0) || (i_micro_kernel_config->avnni_btrans_gemm_stack_alloc_tensors !=0) ) {
    /* if B is in vnni2T, use scratch in stack */
    libxsmm_generator_gemm_setup_A_vnni_or_trans_B_vnni_or_trans_tensor_to_stack( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, i_xgemm_desc_orig, (libxsmm_datatype) LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc_orig->datatype ), 0);
  } else if ( ((i_xgemm_desc_orig->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0) && ((i_xgemm_desc_orig->flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0) ) {
    /* if B is in vnni2T, use scratch in stack */
    libxsmm_generator_gemm_setup_B_vnni2t_to_norm_into_stack( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, i_xgemm_desc_orig, (libxsmm_datatype) LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc_orig->datatype ));
  } else if (( i_micro_kernel_config->bf8_gemm_via_stack_alloc_tensors > 0) || (i_micro_kernel_config->hf8_gemm_via_stack_alloc_tensors > 0)) {
    /* Now setup A and B tensors in stack as FP32 flat tensors */
    if ((io_generated_code->arch >= LIBXSMM_X86_AVX512_CPX) || (io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CPX)) {
      libxsmm_generator_gemm_setup_f8_AB_tensors_to_stack( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, i_xgemm_desc_orig, (libxsmm_datatype) LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc_orig->datatype ), LIBXSMM_DATATYPE_BF16);
    } else {
      libxsmm_generator_gemm_setup_f8_AB_tensors_to_stack( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, i_xgemm_desc_orig, (libxsmm_datatype) LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc_orig->datatype ), LIBXSMM_DATATYPE_F32);
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_vnni_store_C_from_scratch( libxsmm_generated_code*            io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc) {
  if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
    libxsmm_descriptor_blob blob;
    const libxsmm_meltw_descriptor *const trans_desc = libxsmm_meltw_descriptor_init2(&blob,
      LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, i_xgemm_desc->m, i_xgemm_desc->n,
      i_xgemm_desc->m, i_xgemm_desc->ldc, 0, 0,
      (unsigned short)LIBXSMM_MELTW_FLAG_UNARY_NONE, (unsigned short)LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2, LIBXSMM_MELTW_OPERATION_UNARY);
    libxsmm_mateltwise_kernel_config l_trans_config;
    unsigned int l_gp_reg_in = i_gp_reg_mapping->gp_reg_help_2;

    libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_trans_config, trans_desc);
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANS_EXT_BUF_C, i_gp_reg_mapping->gp_reg_c );
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, l_gp_reg_in );
    libxsmm_x86_instruction_alu_imm_i64( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mapping->gp_reg_help_1, (long long)32*64 );
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_1, l_gp_reg_in);
    if ((io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) && (io_generated_code->arch < LIBXSMM_X86_ALLFEAT)) {
      libxsmm_generator_transform_norm_to_vnni2_16bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
          l_gp_reg_in, i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_mloop, i_gp_reg_mapping->gp_reg_nloop,
          i_gp_reg_mapping->gp_reg_help_1, 1, 2,
          &l_trans_config, trans_desc, 0 );
    } else {
      libxsmm_generator_transform_norm_to_vnni2_16bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
          l_gp_reg_in, i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_mloop, i_gp_reg_mapping->gp_reg_nloop,
          &l_trans_config, trans_desc, 0 );
    }
  } else if ( (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))  ) {
    libxsmm_descriptor_blob blob;
    const libxsmm_meltw_descriptor *const trans_desc = libxsmm_meltw_descriptor_init2(&blob,
      LIBXSMM_DATATYPE_I8, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_I8, LIBXSMM_DATATYPE_I8, i_xgemm_desc->m, i_xgemm_desc->n,
      i_xgemm_desc->m, i_xgemm_desc->ldc, 0, 0,
      (unsigned short)LIBXSMM_MELTW_FLAG_UNARY_NONE, (unsigned short)LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4, LIBXSMM_MELTW_OPERATION_UNARY);
    libxsmm_mateltwise_kernel_config l_trans_config;
    unsigned int l_gp_reg_in = i_gp_reg_mapping->gp_reg_help_2;

    libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_trans_config, trans_desc);
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANS_EXT_BUF_C, i_gp_reg_mapping->gp_reg_c );
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, l_gp_reg_in );
    libxsmm_x86_instruction_alu_imm_i64( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mapping->gp_reg_help_1, (long long)32*64 );
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_1, l_gp_reg_in);
    if ((io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) && (io_generated_code->arch < LIBXSMM_X86_ALLFEAT)) {
      libxsmm_generator_transform_norm_to_vnni4_08bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
          l_gp_reg_in, i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_mloop, i_gp_reg_mapping->gp_reg_nloop,
          i_gp_reg_mapping->gp_reg_help_1, 1, 2,
          &l_trans_config, trans_desc, 0 );
    } else {
      libxsmm_generator_transform_norm_to_vnni4_08bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
          l_gp_reg_in, i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_mloop, i_gp_reg_mapping->gp_reg_nloop,
          &l_trans_config, trans_desc, 0 );
    }
  } else {
    /* should not happen */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_apply_relu_to_vreg( libxsmm_generated_code*             io_generated_code,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const unsigned int                 zero_vreg,
    const unsigned int                 inout_vreg,
    const unsigned int                 store_bitmask,
    const unsigned int                 gpr_bitmask,
    const unsigned int                 store_bitmask_offset,
    const unsigned int                 is_32_bit_relu,
    const unsigned int                 sse_scratch_gpr,
    const unsigned int                 aux_gpr,
    const unsigned int                 aux_vreg,
    const unsigned int                 use_masked_cmp,
    const unsigned int                 sse_mask_pos ) {
  if (io_generated_code->arch  < LIBXSMM_X86_AVX512_VL256_SKX) {
    if (is_32_bit_relu == 1) {
      if ( io_generated_code->arch < LIBXSMM_X86_AVX ) {
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_MAXPS, i_micro_kernel_config->vector_name, zero_vreg, inout_vreg );
        if (store_bitmask == 1) {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_MOVUPS, i_micro_kernel_config->vector_name, inout_vreg, aux_vreg );
          libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_CMPPS, i_micro_kernel_config->vector_name, zero_vreg, aux_vreg, 4 );
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_MOVMSKPS, i_micro_kernel_config->vector_name, aux_vreg, aux_gpr );
          if ( sse_mask_pos == 0 ) {
            libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVB, gpr_bitmask, LIBXSMM_X86_GP_REG_UNDEF, 0, store_bitmask_offset, aux_gpr, 1);
          } else {
            libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVB, gpr_bitmask, LIBXSMM_X86_GP_REG_UNDEF, 0, store_bitmask_offset, sse_scratch_gpr, 0);
            libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SHLQ, aux_gpr, 4 );
            libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_ORQ_RM_R, aux_gpr, sse_scratch_gpr);
            libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVB, gpr_bitmask, LIBXSMM_X86_GP_REG_UNDEF, 0, store_bitmask_offset, sse_scratch_gpr, 1);
          }
        }
      } else {
        if (store_bitmask == 1) {
          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCMPPS, i_micro_kernel_config->vector_name, zero_vreg, inout_vreg, aux_vreg, 6 );
          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VMOVMSKPS, i_micro_kernel_config->vector_name, aux_vreg, LIBXSMM_X86_VEC_REG_UNDEF, aux_gpr, 0 );
          libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVB, gpr_bitmask, LIBXSMM_X86_GP_REG_UNDEF, 0, store_bitmask_offset, aux_gpr, 1);
        }
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMAXPS, i_micro_kernel_config->vector_name, inout_vreg, zero_vreg, inout_vreg );
      }
    } else {
      /* should not happen */
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  } else {
    if (store_bitmask == 0) {
      libxsmm_x86_instruction_vec_compute_3reg(  io_generated_code,
          (is_32_bit_relu == 1) ? LIBXSMM_X86_INSTR_VMAXPS : LIBXSMM_X86_INSTR_VPMAXSW,
          i_micro_kernel_config->vector_name,
          inout_vreg,
          zero_vreg,
          inout_vreg);
    } else {
      unsigned int current_mask_reg = 7;
      libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
          (is_32_bit_relu == 1) ? LIBXSMM_X86_INSTR_VCMPPS : LIBXSMM_X86_INSTR_VPCMPW,
          i_micro_kernel_config->vector_name,
          zero_vreg,
          inout_vreg,
          current_mask_reg, use_masked_cmp, 0, 0, 6);
      /* Blend output result with zero reg based on relu mask */
      libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
          (is_32_bit_relu == 1) ? LIBXSMM_X86_INSTR_VPBLENDMD : LIBXSMM_X86_INSTR_VPBLENDMW,
          i_micro_kernel_config->vector_name,
          inout_vreg,
          zero_vreg,
          inout_vreg,
          current_mask_reg,
          0 );
      /* Store bitmask */
      libxsmm_x86_instruction_mask_move_mem( io_generated_code,
          (is_32_bit_relu == 1) ? ((i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX) ? LIBXSMM_X86_INSTR_KMOVB_ST: LIBXSMM_X86_INSTR_KMOVW_ST) : ((i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX) ? LIBXSMM_X86_INSTR_KMOVW_ST:  LIBXSMM_X86_INSTR_KMOVD_ST),
          gpr_bitmask,
          LIBXSMM_X86_GP_REG_UNDEF,
          0,
          store_bitmask_offset,
          current_mask_reg );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_apply_sigmoid_to_vreg_from_scratch( libxsmm_generated_code*             io_generated_code,
    libxsmm_micro_kernel_config*       i_micro_kernel_config_mod,
    const unsigned int                 scratch_gpr,
    const unsigned int                 in_vreg,
    const unsigned int                 out_vreg ) {

  /* Load accumulator from scratch */
  libxsmm_x86_instruction_vec_move( io_generated_code,
      i_micro_kernel_config_mod->instruction_set,
      ( io_generated_code->arch >= LIBXSMM_X86_AVX ) ? LIBXSMM_X86_INSTR_VMOVUPS : LIBXSMM_X86_INSTR_MOVUPS,
      scratch_gpr,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      in_vreg * 64,
      i_micro_kernel_config_mod->vector_name,
      out_vreg, 0, 1, 0 );

  /* Apply sigmoid */
  if (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) {
    const char i_vname = (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 'y' : 'z';
    libxsmm_generator_sigmoid_ps_rational_78_avx512( io_generated_code, out_vreg, i_micro_kernel_config_mod->vec_x2,
        i_micro_kernel_config_mod->vec_nom, i_micro_kernel_config_mod->vec_denom,
        i_micro_kernel_config_mod->mask_hi, i_micro_kernel_config_mod->mask_lo,
        i_micro_kernel_config_mod->vec_c0, i_micro_kernel_config_mod->vec_c1, i_micro_kernel_config_mod->vec_c2, i_micro_kernel_config_mod->vec_c3,
        i_micro_kernel_config_mod->vec_c1_d, i_micro_kernel_config_mod->vec_c2_d, i_micro_kernel_config_mod->vec_c3_d,
        i_micro_kernel_config_mod->vec_hi_bound, i_micro_kernel_config_mod->vec_lo_bound, i_micro_kernel_config_mod->vec_ones,
        i_micro_kernel_config_mod->vec_neg_ones, i_micro_kernel_config_mod->vec_halves, i_vname  );
  } else {
    if (io_generated_code->arch >= LIBXSMM_X86_AVX) {
      libxsmm_generator_sigmoid_ps_rational_78_avx( io_generated_code, out_vreg, i_micro_kernel_config_mod->vec_x2,
          i_micro_kernel_config_mod->vec_nom, i_micro_kernel_config_mod->vec_denom,
          i_micro_kernel_config_mod->vec_c0, i_micro_kernel_config_mod->vec_c1, i_micro_kernel_config_mod->vec_c2, i_micro_kernel_config_mod->vec_c3,
          i_micro_kernel_config_mod->vec_c1_d, i_micro_kernel_config_mod->vec_c2_d, i_micro_kernel_config_mod->vec_c3_d,
          i_micro_kernel_config_mod->vec_hi_bound, i_micro_kernel_config_mod->vec_lo_bound, i_micro_kernel_config_mod->vec_ones,
          i_micro_kernel_config_mod->vec_neg_ones );
    } else {
      libxsmm_generator_sigmoid_ps_rational_78_sse( io_generated_code, out_vreg, i_micro_kernel_config_mod->vec_x2,
          i_micro_kernel_config_mod->vec_nom, i_micro_kernel_config_mod->vec_denom,
          i_micro_kernel_config_mod->vec_c0, i_micro_kernel_config_mod->vec_c1, i_micro_kernel_config_mod->vec_c2, i_micro_kernel_config_mod->vec_c3,
          i_micro_kernel_config_mod->vec_c1_d, i_micro_kernel_config_mod->vec_c2_d, i_micro_kernel_config_mod->vec_c3_d,
          i_micro_kernel_config_mod->vec_hi_bound, i_micro_kernel_config_mod->vec_lo_bound, i_micro_kernel_config_mod->vec_ones,
          i_micro_kernel_config_mod->vec_neg_ones );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_restore_2D_regblock_from_scratch( libxsmm_generated_code*             io_generated_code,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const unsigned int                 scratch_gpr,
    const unsigned int                 l_vec_reg_acc_start,
    const unsigned int                 l_m_blocking,
    const unsigned int                 i_n_blocking) {
  unsigned int l_n, l_m;
  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          ( io_generated_code->arch >= LIBXSMM_X86_AVX ) ? LIBXSMM_X86_INSTR_VMOVUPS : LIBXSMM_X86_INSTR_MOVUPS,
          scratch_gpr,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (l_vec_reg_acc_start + l_m + (l_m_blocking * l_n)) * 64,
          i_micro_kernel_config->vector_name,
          l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), 0, 0, 0 );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_store_2D_regblock_to_scratch( libxsmm_generated_code*             io_generated_code,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const unsigned int                 scratch_gpr,
    const unsigned int                 l_vec_reg_acc_start,
    const unsigned int                 l_m_blocking,
    const unsigned int                 i_n_blocking) {
  unsigned int l_n, l_m;
  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          ( io_generated_code->arch >= LIBXSMM_X86_AVX ) ? LIBXSMM_X86_INSTR_VMOVUPS : LIBXSMM_X86_INSTR_MOVUPS,
          scratch_gpr,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (l_vec_reg_acc_start + l_m + (l_m_blocking * l_n)) * 64,
          i_micro_kernel_config->vector_name,
          l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), 0, 0, 1 );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_dump_2D_block_and_prepare_sigmoid_fusion( libxsmm_generated_code*             io_generated_code,
    libxsmm_micro_kernel_config*       i_micro_kernel_config,
    const unsigned int                 l_vec_reg_acc_start,
    const unsigned int                 l_m_blocking,
    const unsigned int                 i_n_blocking,
    const unsigned int                 scratch_gpr,
    const unsigned int                 aux_gpr) {
  unsigned int n_avail_vregs = (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) ? 32 : 16;
  unsigned int n_avail_masks = (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) ? 8 : 16;
  /* First dump the accumulators to scratch and then setup sigmoid coefficients to be reused */
  libxsmm_x86_instruction_push_reg( io_generated_code, scratch_gpr);
  libxsmm_x86_instruction_push_reg( io_generated_code, aux_gpr );
  libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, scratch_gpr);
  libxsmm_generator_gemm_store_2D_regblock_to_scratch( io_generated_code, i_micro_kernel_config,
      scratch_gpr, l_vec_reg_acc_start, l_m_blocking, i_n_blocking);
  libxsmm_generator_gemm_prepare_coeffs_sigmoid_ps_rational_78_sse_avx_avx512( io_generated_code, i_micro_kernel_config, n_avail_vregs, n_avail_masks, aux_gpr );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_prepare_relu_fusion( libxsmm_generated_code*             io_generated_code,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const unsigned int                 zero_vreg,
    const unsigned int                 store_bitmask,
    const unsigned int                 bitmask_gpr,
    const unsigned int                 sse_scratch_gpr,
    const unsigned int                 aux_gpr) {
  /* Zero out register 0 to perform relu */
  if ( io_generated_code->arch < LIBXSMM_X86_AVX ) {
    libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
        i_micro_kernel_config->vxor_instruction,
        i_micro_kernel_config->vector_name,
        zero_vreg, zero_vreg);
  } else {
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
        i_micro_kernel_config->vxor_instruction,
        i_micro_kernel_config->vector_name,
        zero_vreg, zero_vreg, zero_vreg);
  }
  if (store_bitmask == 1) {
    libxsmm_x86_instruction_push_reg( io_generated_code, bitmask_gpr );
    if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
      libxsmm_x86_instruction_push_reg( io_generated_code, aux_gpr );
    }
    if (io_generated_code->arch < LIBXSMM_X86_AVX) {
      libxsmm_x86_instruction_push_reg( io_generated_code, sse_scratch_gpr );
    }
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, bitmask_gpr );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_cleanup_relu_fusion( libxsmm_generated_code*             io_generated_code,
    const unsigned int                 store_bitmask,
    const unsigned int                 bitmask_gpr,
    const unsigned int                 sse_scratch_gpr,
    const unsigned int                 aux_gpr) {
  if (store_bitmask == 1) {
    if (io_generated_code->arch < LIBXSMM_X86_AVX) {
      libxsmm_x86_instruction_pop_reg( io_generated_code, sse_scratch_gpr );
    }
    if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
      libxsmm_x86_instruction_pop_reg( io_generated_code, aux_gpr );
    }
    libxsmm_x86_instruction_pop_reg( io_generated_code, bitmask_gpr);
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_cleanup_sigmoid_fusion( libxsmm_generated_code*             io_generated_code,
    const unsigned int                 scratch_gpr,
    const unsigned int                 aux_gpr ) {
  libxsmm_x86_instruction_pop_reg( io_generated_code, aux_gpr );
  libxsmm_x86_instruction_pop_reg( io_generated_code, scratch_gpr );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_load_colbias_to_2D_block( libxsmm_generated_code*             io_generated_code,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    libxsmm_datatype                   colbias_precision,
    libxsmm_datatype                   accumul_precision,
    const unsigned int                 l_vec_reg_acc_start,
    const unsigned int                 l_m_blocking,
    const unsigned int                 i_n_blocking,
    const unsigned int                 i_m_remain ) {
  unsigned int l_n = 0, l_m = 0;

  libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
  if (colbias_precision == LIBXSMM_DATATYPE_HF8) {
    libxsmm_generator_vcvt_hf8_tofrom_f32_avx512_prep_stack( io_generated_code,  i_gp_reg_mapping->gp_reg_help_2);
  }
  libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_2 );
  for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      if (colbias_precision == LIBXSMM_DATATYPE_BF16 || colbias_precision == LIBXSMM_DATATYPE_F16) {
        if (l_n == 0) {
          /* Load bias vector */
          /* load 16 bit values into xmm portion of the register */
          if ( (i_micro_kernel_config->use_masking_a_c != 0) && ( l_m == (l_m_blocking - 1) ) ) {
            if ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX ) {
              libxsmm_generator_maskedload_16bit_sse( io_generated_code, i_gp_reg_mapping->gp_reg_help_0, 0,
                                                      i_gp_reg_mapping->gp_reg_help_2, LIBXSMM_X86_GP_REG_UNDEF, 0, (l_m * (i_micro_kernel_config->vector_length)) * 2,
                                                      l_vec_reg_acc_start + l_m, i_m_remain );
            } else if ( (i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX2 ) && (i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_VL256_SKX) ) {
              libxsmm_generator_maskedload_16bit_avx2( io_generated_code, i_gp_reg_mapping->gp_reg_help_0,
                                                       i_gp_reg_mapping->gp_reg_help_2, LIBXSMM_X86_GP_REG_UNDEF, 0, (l_m * (i_micro_kernel_config->vector_length)) * 2,
                                                       l_vec_reg_acc_start + l_m, i_m_remain );
            } else {
              char l_vname = ( (i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_VL256_SKX) && ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX ) ) ? 'x' : 'y';
              if (colbias_precision == LIBXSMM_DATATYPE_F16 && accumul_precision == LIBXSMM_DATATYPE_F16 && i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_SPR) {
                l_vname = 'z';
              }
              libxsmm_x86_instruction_vec_move( io_generated_code,
                  i_micro_kernel_config->instruction_set,
                  LIBXSMM_X86_INSTR_VMOVDQU16,
                  i_gp_reg_mapping->gp_reg_help_2,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  (l_m * (i_micro_kernel_config->vector_length)) * 2,
                  l_vname,
                  l_vec_reg_acc_start + l_m, (colbias_precision == LIBXSMM_DATATYPE_BF16) ? 2 : 1, 1, 0 );
            }
          } else {
            char l_vname = ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX ) ? 'x' :
                                 ( ( i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX2 ) && ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_VL256_SKX ) ) ? 'x' :
                                 ( ( i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_VL256_SKX) && ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX ) ) ? 'x' : 'y';
            if (colbias_precision == LIBXSMM_DATATYPE_F16 && accumul_precision == LIBXSMM_DATATYPE_F16 && i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_SPR) {
              l_vname = 'z';
            }
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX ) ? LIBXSMM_X86_INSTR_MOVSD : i_micro_kernel_config->c_vmove_instruction,
                i_gp_reg_mapping->gp_reg_help_2,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                (l_m * (i_micro_kernel_config->vector_length)) * 2,
                l_vname,
                l_vec_reg_acc_start + l_m, 0, 1, 0 );
          }
          /* up-convert bf16 to fp32 */
          if (colbias_precision == LIBXSMM_DATATYPE_BF16) {
            libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                         l_vec_reg_acc_start + l_m, l_vec_reg_acc_start + l_m );
          }
          if (colbias_precision == LIBXSMM_DATATYPE_F16 && accumul_precision == LIBXSMM_DATATYPE_F32) {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, l_vec_reg_acc_start + l_m, l_vec_reg_acc_start + l_m);
          }
        } else {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS,
              ( ( i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX2) && ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX ) ) ? 'y' : 'z',
              l_vec_reg_acc_start + l_m, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
        }
      } else if (colbias_precision == LIBXSMM_DATATYPE_BF8) {
        if (l_n == 0) {
          /* load 16 bit values into xmm portion of the register */
          if ( (i_micro_kernel_config->use_masking_a_c != 0) && ( l_m == (l_m_blocking - 1) ) ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVDQU8,
                i_gp_reg_mapping->gp_reg_help_2,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                l_m * (i_micro_kernel_config->vector_length),
                ( ( i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_VL256_SKX) && ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX ) ) ? 'y' : 'z',
                l_vec_reg_acc_start + l_m, 2, 1, 0 );
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                ( io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? LIBXSMM_X86_INSTR_VMOVSD : i_micro_kernel_config->c_vmove_instruction,
                i_gp_reg_mapping->gp_reg_help_2,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                l_m * (i_micro_kernel_config->vector_length),
                'x',
                l_vec_reg_acc_start + l_m, 0, 1, 0 );
          }
          /* up-convert bf8 to fp32 */
          libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                             l_vec_reg_acc_start + l_m, l_vec_reg_acc_start + l_m );
        } else {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS,
              ( ( i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_VL256_SKX) && ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX ) ) ? 'y' : 'z',
              l_vec_reg_acc_start + l_m, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
        }
      } else if (colbias_precision == LIBXSMM_DATATYPE_HF8) {
        if (l_n == 0) {
          /* load 16 bit values into xmm portion of the register */
          if ( (i_micro_kernel_config->use_masking_a_c != 0) && ( l_m == (l_m_blocking - 1) ) ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVDQU8,
                i_gp_reg_mapping->gp_reg_help_2,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                l_m * (i_micro_kernel_config->vector_length),
                ( ( i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_VL256_SKX) && ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX ) ) ? 'y' : 'z',
                l_vec_reg_acc_start + l_m, 2, 1, 0 );
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                ( io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? LIBXSMM_X86_INSTR_VMOVSD : i_micro_kernel_config->c_vmove_instruction,
                i_gp_reg_mapping->gp_reg_help_2,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                l_m * (i_micro_kernel_config->vector_length),
                'x',
                l_vec_reg_acc_start + l_m, 0, 1, 0 );
          }
          /* up-convert bf8 to fp32 */
          libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                                             l_vec_reg_acc_start + l_m, l_vec_reg_acc_start + l_m, 0, 1, 6, 7 );
        } else {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS,
              ( ( i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_VL256_SKX) && ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX ) ) ? 'y' : 'z',
              l_vec_reg_acc_start + l_m, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
        }
      } else if (colbias_precision == LIBXSMM_DATATYPE_F32) {
        if (l_n == 0) {
          /* Load bias vector */
          const unsigned int aux_vreg = i_micro_kernel_config->use_masking_a_c;
          const unsigned int mask_gpr = i_gp_reg_mapping->gp_reg_help_0;
          const unsigned int l_mask_reg_or_val = ( i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX ) ? 1 : i_m_remain;

          /* in case of AVX/AVX2 we need to load the mask into an ymm */
          if ( (i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX) && (i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_VL256_SKX) &&
               ((i_micro_kernel_config->use_masking_a_c != 0) && (l_m == (l_m_blocking - 1))) ) {
            libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_AVX2_MASK_PTR, mask_gpr );
            libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_VMOVUPS,
                                              mask_gpr, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 'y', aux_vreg, 0, 0, 0 );
          }
          libxsmm_x86_instruction_unified_vec_move( io_generated_code, i_micro_kernel_config->c_vmove_instruction,
              i_gp_reg_mapping->gp_reg_help_2, LIBXSMM_X86_GP_REG_UNDEF, 0,
              ((l_m * (i_micro_kernel_config->vector_length))) * 4,
              i_micro_kernel_config->vector_name,
              l_vec_reg_acc_start + l_m, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, l_mask_reg_or_val, 0 );
        } else {
          unsigned int l_mov_instr = ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX ) ? LIBXSMM_X86_INSTR_MOVUPS: LIBXSMM_X86_INSTR_VMOVUPS;
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, l_mov_instr, i_micro_kernel_config->vector_name, l_vec_reg_acc_start + l_m, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
        }
      } else {
        /* should not happen */
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }
    }
  }
  if (colbias_precision == LIBXSMM_DATATYPE_HF8) {
    libxsmm_generator_vcvt_hf8_tofrom_f32_avx512_clean_stack( io_generated_code,  i_gp_reg_mapping->gp_reg_help_2);
  }
  libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_add_colbias_to_2D_block( libxsmm_generated_code*             io_generated_code,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    libxsmm_datatype                   colbias_precision,
    libxsmm_datatype                   accumul_precision,
    const unsigned int                 l_vec_reg_acc_start,
    const unsigned int                 l_m_blocking,
    const unsigned int                 i_n_blocking,
    const unsigned int                 i_m_remain ) {
  unsigned int l_n = 0, l_m = 0;

  libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
  if (colbias_precision == LIBXSMM_DATATYPE_HF8) {
    libxsmm_generator_vcvt_hf8_tofrom_f32_avx512_prep_stack( io_generated_code,  i_gp_reg_mapping->gp_reg_help_2);
  }
  libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_2 );

  for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
    /* Load bias vector */
    if (colbias_precision == LIBXSMM_DATATYPE_BF16 || colbias_precision == LIBXSMM_DATATYPE_F16) {
      /* load 16 bit values into xmm portion of the register */
      if ( (i_micro_kernel_config->use_masking_a_c != 0) && ( l_m == (l_m_blocking - 1) ) ) {
        if ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX ) {
          libxsmm_generator_maskedload_16bit_sse( io_generated_code, i_gp_reg_mapping->gp_reg_help_0, 0,
                                                   i_gp_reg_mapping->gp_reg_help_2, LIBXSMM_X86_GP_REG_UNDEF, 0, (l_m * (i_micro_kernel_config->vector_length)) * 2,
                                                   0, i_m_remain );
        } else if ( (i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX2 ) && (i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_VL256_SKX) ) {
          libxsmm_generator_maskedload_16bit_avx2( io_generated_code, i_gp_reg_mapping->gp_reg_help_0,
                                                   i_gp_reg_mapping->gp_reg_help_2, LIBXSMM_X86_GP_REG_UNDEF, 0, (l_m * (i_micro_kernel_config->vector_length)) * 2,
                                                   0, i_m_remain );
        } else {
          char l_vname = ( (i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_VL256_SKX) && ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX ) ) ? 'x' : 'y';
          if (colbias_precision == LIBXSMM_DATATYPE_F16 && accumul_precision == LIBXSMM_DATATYPE_F16 && i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_SPR) {
            l_vname = 'z';
          }
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVDQU16,
              i_gp_reg_mapping->gp_reg_help_2,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (l_m * (i_micro_kernel_config->vector_length)) * 2,
              l_vname,
              0, (colbias_precision == LIBXSMM_DATATYPE_BF16) ? 2 : 1, 1, 0 );
        }
      } else {
        char l_vname = ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX ) ? 'x' :
                             ( ( i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX2 ) && ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_VL256_SKX ) ) ? 'x' :
                             ( ( i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_VL256_SKX) && ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX ) ) ? 'x' : 'y';
        if (colbias_precision == LIBXSMM_DATATYPE_F16 && accumul_precision == LIBXSMM_DATATYPE_F16 && i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_SPR) {
          l_vname = 'z';
        }
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX ) ? LIBXSMM_X86_INSTR_MOVSD : i_micro_kernel_config->c_vmove_instruction,
            i_gp_reg_mapping->gp_reg_help_2,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (l_m * (i_micro_kernel_config->vector_length)) * 2,
            l_vname,
            0, 0, 1, 0 );
      }
      /* up-convert bf16 to fp32 */
      if (colbias_precision == LIBXSMM_DATATYPE_BF16) {
        libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 0 );
      }
      if (colbias_precision == LIBXSMM_DATATYPE_F16) {
        unsigned int l_use_f32_comp = ((accumul_precision == LIBXSMM_DATATYPE_F32) || (accumul_precision == LIBXSMM_DATATYPE_F16 && i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SPR)) ? 1 : 0;
        if (l_use_f32_comp) libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, 0, 0 );
      }
    } else if (colbias_precision == LIBXSMM_DATATYPE_BF8) {
      /* load 16 bit values into xmm portion of the register */
      if ( (i_micro_kernel_config->use_masking_a_c != 0) && ( l_m == (l_m_blocking - 1) ) ) {
        if ( (i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX2 ) && (i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_VL256_SKX) ) {
#if 0
          /* should not happen */
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
          return;
#else
          libxsmm_generator_maskedload_16bit_avx2( io_generated_code, i_gp_reg_mapping->gp_reg_help_0,
                                                   i_gp_reg_mapping->gp_reg_help_2, LIBXSMM_X86_GP_REG_UNDEF, 0, (l_m * (i_micro_kernel_config->vector_length)) * 2,
                                                   0, i_m_remain );
#endif
        } else {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVDQU8,
              i_gp_reg_mapping->gp_reg_help_2,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              l_m * (i_micro_kernel_config->vector_length),
              ( ( i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_VL256_SKX) && ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX ) ) ? 'y' : 'z',
              0, 2, 1, 0 );
        }
      } else {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            ( io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? LIBXSMM_X86_INSTR_VMOVSD : i_micro_kernel_config->c_vmove_instruction,
            i_gp_reg_mapping->gp_reg_help_2,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            l_m * (i_micro_kernel_config->vector_length),
            'x',
            0, 0, 1, 0 );
      }
      /* up-convert bf8 to fp32 */
      libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                         0, 0 );
    } else if (colbias_precision == LIBXSMM_DATATYPE_HF8) {
      /* load 16 bit values into xmm portion of the register */
      if ( (i_micro_kernel_config->use_masking_a_c != 0) && ( l_m == (l_m_blocking - 1) ) ) {
        if ( (i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX2 ) && (i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_VL256_SKX) ) {
#if 0
          /* should not happen */
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
          return;
#else
          libxsmm_generator_maskedload_16bit_avx2( io_generated_code, i_gp_reg_mapping->gp_reg_help_0,
                                                   i_gp_reg_mapping->gp_reg_help_2, LIBXSMM_X86_GP_REG_UNDEF, 0, (l_m * (i_micro_kernel_config->vector_length)) * 2,
                                                   0, i_m_remain );
#endif
        } else {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVDQU8,
              i_gp_reg_mapping->gp_reg_help_2,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              l_m * (i_micro_kernel_config->vector_length),
              ( ( i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_VL256_SKX) && ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX ) ) ? 'y' : 'z',
              0, 2, 1, 0 );
        }
      } else {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            ( io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? LIBXSMM_X86_INSTR_VMOVSD : i_micro_kernel_config->c_vmove_instruction,
            i_gp_reg_mapping->gp_reg_help_2,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            l_m * (i_micro_kernel_config->vector_length),
            'x',
            0, 0, 1, 0 );
      }
      /* up-convert hf8 to fp32 */
      libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                                                          0, 0, 1, 2, 6, 7);
    } else if (colbias_precision == LIBXSMM_DATATYPE_F32) {
      const unsigned int aux_vreg = i_micro_kernel_config->use_masking_a_c;
      const unsigned int mask_gpr = i_gp_reg_mapping->gp_reg_help_0;
      const unsigned int l_mask_reg_or_val = ( i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX ) ? 1 : i_m_remain;

      /* in case of AVX/AVX2 we need to load the mask into an ymm */
      if ( (i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX) && (i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_VL256_SKX) &&
           ((i_micro_kernel_config->use_masking_a_c != 0) && (l_m == (l_m_blocking - 1))) ) {
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_AVX2_MASK_PTR, mask_gpr );
        libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_VMOVUPS,
                                          mask_gpr, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 'y', aux_vreg, 0, 0, 0 );
      }

      libxsmm_x86_instruction_unified_vec_move( io_generated_code, i_micro_kernel_config->c_vmove_instruction,
          i_gp_reg_mapping->gp_reg_help_2, LIBXSMM_X86_GP_REG_UNDEF, 0,
          ((l_m * (i_micro_kernel_config->vector_length))) * 4,
          i_micro_kernel_config->vector_name,
          0, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, l_mask_reg_or_val, 0 );
    } else {
      /* should not happen */
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }

    /* Add colbias */
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      if ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX ) {
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_ADDPS, i_micro_kernel_config->vector_name,
            0, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
      } else {
        unsigned int l_vadd_instr = LIBXSMM_X86_INSTR_VADDPS;
        unsigned int l_emu_vadd_ph = 0;
        if (colbias_precision == LIBXSMM_DATATYPE_F16 && accumul_precision == LIBXSMM_DATATYPE_F16 && i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_SPR) {
          l_vadd_instr = LIBXSMM_X86_INSTR_VADDPH;
        }
        if (colbias_precision == LIBXSMM_DATATYPE_F16 && accumul_precision == LIBXSMM_DATATYPE_F16 && i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SPR) {
          l_emu_vadd_ph = 1;
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), l_vec_reg_acc_start + l_m + (l_m_blocking * l_n));
        }
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, l_vadd_instr, i_micro_kernel_config->vector_name,
            l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), 0, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
        if (l_emu_vadd_ph) {
          libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), 0,
              (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
        }
      }
    }
  }

  if (colbias_precision == LIBXSMM_DATATYPE_HF8) {
    libxsmm_generator_vcvt_hf8_tofrom_f32_avx512_clean_stack( io_generated_code,  i_gp_reg_mapping->gp_reg_help_2);
  }
  libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_prepare_coeffs_sigmoid_ps_rational_78_sse_avx_avx512( libxsmm_generated_code*                        io_generated_code,
    libxsmm_micro_kernel_config*        i_micro_kernel_config,
    unsigned int                        reserved_zmms,
    unsigned int                        reserved_mask_regs,
    unsigned int                        temp_reg ) {
  if ( io_generated_code->arch < LIBXSMM_X86_AVX ) {
    libxsmm_generator_gemm_prepare_coeffs_sigmoid_ps_rational_78_sse( io_generated_code, i_micro_kernel_config, reserved_zmms, reserved_mask_regs, temp_reg );
  } else {
    float pade78_sigm_array[16] = { 2027025.0f, 270270.0f, 6930.0f, 36.0f, 945945.0f, 51975.0f,  630.0f, 4.97f, -4.97f,  1.0f, -1.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f };

    i_micro_kernel_config->vec_x2        = reserved_zmms - 1;
    i_micro_kernel_config->vec_nom       = reserved_zmms - 2;
    i_micro_kernel_config->vec_denom     = reserved_zmms - 3;
    i_micro_kernel_config->vec_c0        = reserved_zmms - 4;
    i_micro_kernel_config->vec_c1        = reserved_zmms - 5;
    i_micro_kernel_config->vec_c2        = reserved_zmms - 6;
    i_micro_kernel_config->vec_c3        = reserved_zmms - 7;
    i_micro_kernel_config->vec_c1_d      = reserved_zmms - 8;
    i_micro_kernel_config->vec_c2_d      = reserved_zmms - 9;
    i_micro_kernel_config->vec_c3_d      = reserved_zmms - 10;
    i_micro_kernel_config->vec_hi_bound  = reserved_zmms - 11;
    i_micro_kernel_config->vec_lo_bound  = reserved_zmms - 12;
    i_micro_kernel_config->vec_ones      = reserved_zmms - 13;
    i_micro_kernel_config->vec_neg_ones  = reserved_zmms - 14;
    i_micro_kernel_config->vec_halves    = reserved_zmms - 15;

    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) pade78_sigm_array, "pade78_sigm_array_", i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_c0);
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, temp_reg );
    libxsmm_x86_instruction_vec_move( io_generated_code,
        i_micro_kernel_config->instruction_set,
        LIBXSMM_X86_INSTR_VMOVUPS,
        temp_reg,
        LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
        i_micro_kernel_config->vector_name,
        i_micro_kernel_config->vec_c0, 0, 1, 1 );
    if (io_generated_code->arch  < LIBXSMM_X86_AVX512_SKX) {
      libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) &pade78_sigm_array[8], "pade78_sigm_array2_", i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_c0);
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VMOVUPS,
          temp_reg,
          LIBXSMM_X86_GP_REG_UNDEF, 0, 32,
          i_micro_kernel_config->vector_name,
          i_micro_kernel_config->vec_c0, 0, 1, 1 );
    }

    libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_VBROADCASTSS, temp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
        0, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_c0, 0, 1, 0 );
    libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_VBROADCASTSS, temp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
        4, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_c1, 0, 1, 0 );
    libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_VBROADCASTSS, temp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
        8, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_c2, 0, 1, 0 );
    libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_VBROADCASTSS, temp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
        12, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_c3, 0, 1, 0 );
    libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_VBROADCASTSS, temp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
        16, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_c1_d, 0, 1, 0 );
    libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_VBROADCASTSS, temp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
        20, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_c2_d, 0, 1, 0 );
    libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_VBROADCASTSS, temp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
        24, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_c3_d, 0, 1, 0 );
    libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_VBROADCASTSS, temp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
        28, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_hi_bound, 0, 1, 0 );
    libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_VBROADCASTSS, temp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
        32, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_lo_bound, 0, 1, 0 );
    libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_VBROADCASTSS, temp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
        36, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_ones, 0, 1, 0 );
    libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_VBROADCASTSS, temp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
        40, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_neg_ones, 0, 1, 0 );
    if (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) {
      libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_VBROADCASTSS, temp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
          44, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_halves, 0, 1, 0 );
    }

    i_micro_kernel_config->mask_hi  = reserved_mask_regs - 1;
    i_micro_kernel_config->mask_lo  = reserved_mask_regs - 2;
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_prepare_coeffs_sigmoid_ps_rational_78_sse( libxsmm_generated_code*                        io_generated_code,
    libxsmm_micro_kernel_config*        i_micro_kernel_config,
    unsigned int                        reserved_zmms,
    unsigned int                        reserved_mask_regs,
    unsigned int                        temp_reg ) {
  float pade78_sigm_array[16] = { 2027025.0f, 270270.0f, 6930.0f, 36.0f, 945945.0f, 51975.0f,  630.0f, 4.97f, -4.97f,  1.0f, -1.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f };
  i_micro_kernel_config->vec_x2        = reserved_zmms - 1;
  i_micro_kernel_config->vec_nom       = reserved_zmms - 2;
  i_micro_kernel_config->vec_denom     = reserved_zmms - 3;
  i_micro_kernel_config->vec_c0        = reserved_zmms - 4;
  i_micro_kernel_config->vec_c1        = reserved_zmms - 5;
  i_micro_kernel_config->vec_c2        = reserved_zmms - 6;
  i_micro_kernel_config->vec_c3        = reserved_zmms - 7;
  i_micro_kernel_config->vec_c1_d      = reserved_zmms - 8;
  i_micro_kernel_config->vec_c2_d      = reserved_zmms - 9;
  i_micro_kernel_config->vec_c3_d      = reserved_zmms - 10;
  i_micro_kernel_config->vec_hi_bound  = reserved_zmms - 11;
  i_micro_kernel_config->vec_lo_bound  = reserved_zmms - 12;
  i_micro_kernel_config->vec_ones      = reserved_zmms - 13;
  i_micro_kernel_config->vec_neg_ones  = reserved_zmms - 14;
  i_micro_kernel_config->vec_halves    = reserved_zmms - 15;

  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) pade78_sigm_array, "pade78_sigm_array_", i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_c0);
  libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, temp_reg );
  libxsmm_x86_instruction_vec_move( io_generated_code,
      i_micro_kernel_config->instruction_set,
      LIBXSMM_X86_INSTR_MOVUPS,
      temp_reg,
      LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
      i_micro_kernel_config->vector_name,
      i_micro_kernel_config->vec_c0, 0, 1, 1 );
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) &pade78_sigm_array[4], "pade78_sigm_array2_", i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_c0);
  libxsmm_x86_instruction_vec_move( io_generated_code,
      i_micro_kernel_config->instruction_set,
      LIBXSMM_X86_INSTR_MOVUPS,
      temp_reg,
      LIBXSMM_X86_GP_REG_UNDEF, 0, 16,
      i_micro_kernel_config->vector_name,
      i_micro_kernel_config->vec_c0, 0, 1, 1 );
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) &pade78_sigm_array[8], "pade78_sigm_array3_", i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_c0);
  libxsmm_x86_instruction_vec_move( io_generated_code,
      i_micro_kernel_config->instruction_set,
      LIBXSMM_X86_INSTR_MOVUPS,
      temp_reg,
      LIBXSMM_X86_GP_REG_UNDEF, 0, 32,
      i_micro_kernel_config->vector_name,
      i_micro_kernel_config->vec_c0, 0, 1, 1 );

  libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_MOVSS, temp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
      0, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_c0, 0, 1, 0 );
  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_SHUFPS,
                                                 i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_c0, i_micro_kernel_config->vec_c0, 0 );

  libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_MOVSS, temp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
      4, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_c1, 0, 1, 0 );
  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_SHUFPS,
                                                 i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_c1, i_micro_kernel_config->vec_c1, 0 );

  libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_MOVSS, temp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
      8, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_c2, 0, 1, 0 );
  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_SHUFPS,
                                                 i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_c2, i_micro_kernel_config->vec_c2, 0 );

  libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_MOVSS, temp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
      12, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_c3, 0, 1, 0 );
  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_SHUFPS,
                                                 i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_c3, i_micro_kernel_config->vec_c3, 0 );

  libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_MOVSS, temp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
      16, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_c1_d, 0, 1, 0 );
  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_SHUFPS,
                                                 i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_c1_d, i_micro_kernel_config->vec_c1_d, 0 );

  libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_MOVSS, temp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
      20, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_c2_d, 0, 1, 0 );
  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_SHUFPS,
                                                 i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_c2_d, i_micro_kernel_config->vec_c2_d, 0 );

  libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_MOVSS, temp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
      24, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_c3_d, 0, 1, 0 );
  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_SHUFPS,
                                                 i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_c3_d, i_micro_kernel_config->vec_c3_d, 0 );

  libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_MOVSS, temp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
      28, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_hi_bound, 0, 1, 0 );
  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_SHUFPS,
                                                 i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_hi_bound, i_micro_kernel_config->vec_hi_bound, 0 );

  libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_MOVSS, temp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
      32, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_lo_bound, 0, 1, 0 );
  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_SHUFPS,
                                                 i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_lo_bound, i_micro_kernel_config->vec_lo_bound, 0 );

  libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_MOVSS, temp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
      36, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_ones, 0, 1, 0 );
  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_SHUFPS,
                                                 i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_ones, i_micro_kernel_config->vec_ones, 0 );

  libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_MOVSS, temp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
      40, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_neg_ones, 0, 1, 0 );
  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_SHUFPS,
                                                 i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_neg_ones, i_micro_kernel_config->vec_neg_ones, 0 );

  i_micro_kernel_config->mask_hi  = reserved_mask_regs - 1;
  i_micro_kernel_config->mask_lo  = reserved_mask_regs - 2;
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_setup_stack_frame_fill_ext_gemm_stack_vars( libxsmm_generated_code*            io_generated_code,
    const libxsmm_gemm_descriptor*      i_xgemm_desc,
    libxsmm_micro_kernel_config*        i_micro_kernel_config,
    const libxsmm_gp_reg_mapping*       i_gp_reg_mapping ) {
  unsigned int l_is_Ai4_Bi8_gemm = libxsmm_x86_is_Ai4_Bi8_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bbf16_gemm = libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bfp32_gemm = libxsmm_x86_is_Amxfp4_Bfp32_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bi8_gemm = libxsmm_x86_is_Amxfp4_Bi8_gemm(i_xgemm_desc);
  int is_stride_brgemm  = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0) ? 1 : 0;
  int is_offset_brgemm  = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) > 0) ? 1 : 0;
  int is_address_brgemm = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0) ? 1 : 0;
  int is_brgemm         = ((is_stride_brgemm == 1) || (is_offset_brgemm == 1) || (is_address_brgemm == 1)) ? 1 : 0;
  int has_scf           = ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) && l_is_Amxfp4_Bi8_gemm == 0) ? 1 : 0;
  int has_a_scf         = ( ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype ) && l_is_Amxfp4_Bbf16_gemm == 0 && l_is_Amxfp4_Bi8_gemm == 0) && (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) && ((LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )))) || ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype ) && l_is_Amxfp4_Bbf16_gemm == 0) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) && ((LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))))) ? 1 : 0;
  int has_A_pf_ptr      = (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2_AHEAD || i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD) ? 1 : 0;
  int has_B_pf_ptr      = (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_BL2_VIA_C || i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD ||
      i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C /*|| i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2CL2BL2_VIA_C*/) ? 1 : 0;
  unsigned int temp_reg               = LIBXSMM_X86_GP_REG_R10;

  if (has_scf == 1) {
    libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 112, temp_reg, 0 );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_INT8_SCF, temp_reg );
  }

  if (has_a_scf == 1) {
    libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 48, temp_reg, 0 );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_INT8_SCF, temp_reg );
  }

  if (has_A_pf_ptr == 1) {
    libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 56, temp_reg, 0 );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_PFA_PTR, temp_reg );
  }

  if (has_B_pf_ptr == 1) {
    libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 88, temp_reg, 0 );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_PFB_PTR, temp_reg );
  }

  if ((is_brgemm == 1) && (( i_micro_kernel_config->decompress_A == 1) || (i_micro_kernel_config->atrans_gemm_stack_alloc_tensors > 0) ||
                           (i_micro_kernel_config->avnni_gemm_stack_alloc_tensors > 0) ||
                           (i_micro_kernel_config->avnni_btrans_gemm_stack_alloc_tensors > 0) ||
                           (i_micro_kernel_config->atvnni_gemm_stack_alloc_tensors > 0) ||
                           (i_micro_kernel_config->atvnni_btrans_gemm_stack_alloc_tensors > 0) ||
                           (i_micro_kernel_config->bvnni_btrans_gemm_stack_alloc_tensors > 0) ||
                           (i_micro_kernel_config->bf8_gemm_via_stack_alloc_tensors > 0) || (i_micro_kernel_config->hf8_gemm_via_stack_alloc_tensors > 0))) {
    libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_count, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, temp_reg, 0 );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg );
  }

  if (is_offset_brgemm == 1) {
    libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 40, temp_reg, 0 );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_A_OFFS_BRGEMM_PTR, temp_reg );

    libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 72, temp_reg, 0 );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_B_OFFS_BRGEMM_PTR, temp_reg );
  }

  if (l_is_Ai4_Bi8_gemm > 0 && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0 || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0)) {
    libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 56, temp_reg, 0 );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, temp_reg );
  }

  if (l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0) {
    libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 48, temp_reg, 0 );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, temp_reg );
  }

  if (l_is_Amxfp4_Bi8_gemm > 0) {
    libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 80, temp_reg, 0 );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BSCALE_PTR, temp_reg );
  }

  if (i_micro_kernel_config->fused_eltwise == 1) {
    if (i_micro_kernel_config->has_colbias_act_fused == 1) {
      libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 128, temp_reg, 0 );
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, temp_reg );
      libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 104, temp_reg, 0 );
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, temp_reg );
    }
    if (i_micro_kernel_config->decompress_A == 1) {
      libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 48, temp_reg, 0 );
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BITMAP_PTR, temp_reg );
      libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 160, temp_reg, 0 );
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_DECOMPRESS_BUF, temp_reg );
    }
    if (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) {
      libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 104, temp_reg, 0 );
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, temp_reg );
    }
    if (i_micro_kernel_config->fused_relu_bwd == 1) {
      libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 104, temp_reg, 0 );
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_RELU_BITMASK_PTR, temp_reg );
    }
    if (i_micro_kernel_config->norm_to_normT_B_ext_buf == 1) {
      libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 192, temp_reg, 0 );
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANS_EXT_BUF_B, temp_reg );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_setup_stack_frame_allocate_scratch( libxsmm_generated_code*            io_generated_code,
    const libxsmm_gemm_descriptor*      i_xgemm_desc,
    libxsmm_micro_kernel_config*        i_micro_kernel_config ) {
  unsigned int gemm_scratch_size      = 0;
  unsigned int scratch_pad_size       = 0;
  unsigned int avx2_mask_size         = 64;
  unsigned int avx2_ones_size         = 64;
  short sixteen_ones[16] = { 1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1 };
  unsigned short avx2_bf16_mask[16] = { 0x0, 0xffff, 0x0, 0xffff,   0x0, 0xffff, 0x0, 0xffff,   0x0, 0xffff, 0x0, 0xffff,   0x0, 0xffff, 0x0, 0xffff };
  unsigned int l_is_Ai4_Bi8_gemm = libxsmm_x86_is_Ai4_Bi8_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bbf16_gemm = libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Abf8_Bbf16_gemm = libxsmm_x86_is_Abf8_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Ahf8_Bbf16_gemm = libxsmm_x86_is_Ahf8_Bbf16_gemm(i_xgemm_desc);

  if ((io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) && (io_generated_code->arch < LIBXSMM_X86_ALLFEAT)) {
    i_micro_kernel_config->gemm_scratch_ld = 16;
    gemm_scratch_size = LIBXSMM_MAX(((i_micro_kernel_config->vnni_format_C > 0) ? (32 * 64 + i_xgemm_desc->n * i_xgemm_desc->m * 4) : (32*64)), 64 * i_micro_kernel_config->gemm_scratch_ld * 4 + 8 * 1024/*i_micro_kernel_config->datatype_size*/);
  } else {
    /* Allocate scratch for stashing 32 zmms */
    if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI & i_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI) ) {
      gemm_scratch_size = 32 * 64;
    }
    if (i_micro_kernel_config->vnni_format_C > 0) {
      gemm_scratch_size = 32 * 64 + i_xgemm_desc->n * i_xgemm_desc->m * 4;
    }
  }

  scratch_pad_size  = (gemm_scratch_size % 64 == 0) ? 0 : ((gemm_scratch_size + 63)/64) * 64 - gemm_scratch_size;
  gemm_scratch_size += scratch_pad_size;

  if ( (io_generated_code->arch <= LIBXSMM_X86_SSE42) && (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, avx2_ones_size );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_SSE_AVX2_LP_HELPER_PTR, LIBXSMM_X86_GP_REG_RSP );

    libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char*)sixteen_ones, "sixteen_short_ones", 'x', 0);
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, LIBXSMM_X86_INSTR_MOVUPS, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 'x', 0, 0, 0, 1 );
  }
  if ( (io_generated_code->arch <= LIBXSMM_X86_SSE42) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0)) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, avx2_ones_size );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_SSE_AVX2_LP_HELPER_PTR, LIBXSMM_X86_GP_REG_RSP );

    libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char*)avx2_bf16_mask, "avx2_bf16_mask", 'x', 0);
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, LIBXSMM_X86_INSTR_MOVUPS, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 'x', 0, 0, 0, 1 );
  }
  if ( (io_generated_code->arch >= LIBXSMM_X86_AVX) && (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, avx2_mask_size );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_AVX2_MASK_PTR, LIBXSMM_X86_GP_REG_RSP );

    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, avx2_ones_size );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_SSE_AVX2_LP_HELPER_PTR, LIBXSMM_X86_GP_REG_RSP );
  }
  if ( (io_generated_code->arch == LIBXSMM_X86_AVX2) && (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) ) {
    libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char*)sixteen_ones, "sixteen_short_ones", 'y', 0);
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, LIBXSMM_X86_INSTR_VMOVUPS, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 'y', 0, 0, 0, 1 );
  }
  if ( (io_generated_code->arch >= LIBXSMM_X86_AVX2) && (((io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0)) || (l_is_Amxfp4_Bbf16_gemm > 0)) ) {
    libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char*)avx2_bf16_mask, "avx2_bf16_mask", 'y', 0);
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, LIBXSMM_X86_INSTR_VMOVUPS, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 'y', 0, 0, 0, 1 );
  }

  if (gemm_scratch_size > 0) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, gemm_scratch_size );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, LIBXSMM_X86_GP_REG_RSP );
  }

  if ((io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) && (l_is_Ai4_Bi8_gemm > 0 || l_is_Abf8_Bbf16_gemm > 0 || l_is_Ahf8_Bbf16_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0)) {
    unsigned int l_decompress_dtype = (l_is_Ai4_Bi8_gemm > 0) ? 1 : 2;
    unsigned int scratch_a_decompress_size = 2 * (i_xgemm_desc->m * i_xgemm_desc->k) * l_decompress_dtype;
    unsigned int scratch_a_decompress_pad  = (scratch_a_decompress_size % 64 == 0) ? 0 : ((scratch_a_decompress_size + 63)/64) * 64 - scratch_a_decompress_size;
    scratch_a_decompress_size += scratch_a_decompress_pad;
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, scratch_a_decompress_size );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BUF1, LIBXSMM_X86_GP_REG_RSP );
  }

  if ((i_micro_kernel_config->bf8_gemm_via_stack_alloc_tensors > 0) || (i_micro_kernel_config->hf8_gemm_via_stack_alloc_tensors > 0) ||
      (i_micro_kernel_config->atrans_gemm_stack_alloc_tensors > 0 ) || (i_micro_kernel_config->avnni_gemm_stack_alloc_tensors > 0) ||
      (i_micro_kernel_config->atvnni_gemm_stack_alloc_tensors > 0) || (i_micro_kernel_config->avnni_btrans_gemm_stack_alloc_tensors > 0) ||
      (i_micro_kernel_config->atvnni_btrans_gemm_stack_alloc_tensors > 0) ||
      (i_micro_kernel_config->bvnni_btrans_gemm_stack_alloc_tensors > 0) ) {
    int is_stride_brgemm  = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0) ? 1 : 0;
    int is_offset_brgemm  = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) > 0) ? 1 : 0;
    int is_address_brgemm = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0) ? 1 : 0;
    int is_brgemm         = ((is_stride_brgemm == 1) || (is_offset_brgemm == 1) || (is_address_brgemm == 1)) ? 1 : 0;
    unsigned int inp_dtype_size = ( (i_micro_kernel_config->atrans_gemm_stack_alloc_tensors > 0) || (i_micro_kernel_config->avnni_gemm_stack_alloc_tensors > 0) || (i_micro_kernel_config->atvnni_gemm_stack_alloc_tensors > 0) || (i_micro_kernel_config->avnni_btrans_gemm_stack_alloc_tensors > 0 ) || (i_micro_kernel_config->atvnni_btrans_gemm_stack_alloc_tensors > 0 ) || (i_micro_kernel_config->bvnni_btrans_gemm_stack_alloc_tensors > 0) ) ?  LIBXSMM_TYPESIZE(LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC(i_xgemm_desc->datatype)) : 4;
    unsigned int a_size  = (i_xgemm_desc->m * i_xgemm_desc->k) * inp_dtype_size;
    unsigned int b_size  = (i_xgemm_desc->k * i_xgemm_desc->n) * inp_dtype_size;
    unsigned int c_size  = (i_xgemm_desc->m * i_xgemm_desc->n) * 4;
    unsigned int bias_size = i_xgemm_desc->m * 4;
    unsigned int a_pad  = (a_size % 64 == 0) ? 0 : ((a_size + 63)/64) * 64 - a_size;
    unsigned int b_pad  = (b_size % 64 == 0) ? 0 : ((b_size + 63)/64) * 64 - b_size;
    unsigned int c_pad  = (c_size % 64 == 0) ? 0 : ((c_size + 63)/64) * 64 - c_size;
    unsigned int bias_pad  = (bias_size % 64 == 0) ? 0 : ((bias_size + 63)/64) * 64 - bias_size;
    a_size += a_pad;
    b_size += b_pad;
    c_size += c_pad;
    /* Extra scratch for relu bitmask  */
    if ((i_micro_kernel_config->fused_relu > 0) && ((i_micro_kernel_config->bf8_gemm_via_stack_alloc_tensors > 0) || (i_micro_kernel_config->hf8_gemm_via_stack_alloc_tensors > 0))) {
      c_size  = (i_xgemm_desc->m * i_xgemm_desc->n) * 4 + ((i_xgemm_desc->m+15)/16) * 16 * i_xgemm_desc->n;
      c_pad  = (c_size % 64 == 0) ? 0 : ((c_size + 63)/64) * 64 - c_size;
      c_size += c_pad;
    }
    bias_size += bias_pad;
    if ((i_micro_kernel_config->bf8_gemm_via_stack_alloc_tensors > 0) || (i_micro_kernel_config->hf8_gemm_via_stack_alloc_tensors > 0)) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, a_size );
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_A_SCRATCH_PTR, LIBXSMM_X86_GP_REG_RSP );
    }
    if (((io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) && (io_generated_code->arch < LIBXSMM_X86_ALLFEAT)) && ((i_micro_kernel_config->bf8_gemm_via_stack_alloc_tensors > 0) || (i_micro_kernel_config->hf8_gemm_via_stack_alloc_tensors > 0))) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, c_size );
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_C_SCRATCH_PTR, LIBXSMM_X86_GP_REG_RSP );
      if ((i_micro_kernel_config->fused_b8colbias > 0) || (i_micro_kernel_config->fused_h8colbias > 0 )) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, bias_size );
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BIAS_SCRATCH_PTR, LIBXSMM_X86_GP_REG_RSP );
      }
    }
    if (is_brgemm == 0) {
      if ( i_micro_kernel_config->atrans_gemm_stack_alloc_tensors > 0 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, a_size );
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR, LIBXSMM_X86_GP_REG_RSP );
      } else if ( i_micro_kernel_config->avnni_gemm_stack_alloc_tensors > 0 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, a_size );
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR, LIBXSMM_X86_GP_REG_RSP );
      } else if ( (i_micro_kernel_config->avnni_btrans_gemm_stack_alloc_tensors > 0) || (i_micro_kernel_config->atvnni_btrans_gemm_stack_alloc_tensors > 0) ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, a_size );
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR, LIBXSMM_X86_GP_REG_RSP );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, b_size );
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, LIBXSMM_X86_GP_REG_RSP );
      } else if ( i_micro_kernel_config->atvnni_gemm_stack_alloc_tensors > 0 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, a_size );
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR, LIBXSMM_X86_GP_REG_RSP );
      } else if ( i_micro_kernel_config->bvnni_btrans_gemm_stack_alloc_tensors > 0 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, b_size );
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, LIBXSMM_X86_GP_REG_RSP );
      } else {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, a_size );
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_A_EMU_PTR, LIBXSMM_X86_GP_REG_RSP );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, b_size );
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, LIBXSMM_X86_GP_REG_RSP );
      }
    } else {
      unsigned int temp_reg = LIBXSMM_X86_GP_REG_R10;
      if ( i_micro_kernel_config->atrans_gemm_stack_alloc_tensors > 0 ) {
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, temp_reg, a_size);
        libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, temp_reg, LIBXSMM_X86_GP_REG_RSP );
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR, LIBXSMM_X86_GP_REG_RSP );
        if (is_offset_brgemm > 0 || is_address_brgemm > 0) {
          libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg );
          libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, temp_reg, b_size);
          libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, temp_reg, LIBXSMM_X86_GP_REG_RSP );
          libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, LIBXSMM_X86_GP_REG_RSP );
        }
      } else if ( i_micro_kernel_config->avnni_gemm_stack_alloc_tensors > 0 ) {
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, temp_reg, a_size);
        libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, temp_reg, LIBXSMM_X86_GP_REG_RSP );
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR, LIBXSMM_X86_GP_REG_RSP );
        if (is_offset_brgemm > 0 || is_address_brgemm > 0) {
          libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg );
          libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, temp_reg, b_size);
          libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, temp_reg, LIBXSMM_X86_GP_REG_RSP );
          libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, LIBXSMM_X86_GP_REG_RSP );
        }
      } else if ( (i_micro_kernel_config->avnni_btrans_gemm_stack_alloc_tensors > 0) || (i_micro_kernel_config->atvnni_btrans_gemm_stack_alloc_tensors > 0) ) {
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, temp_reg, a_size);
        libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, temp_reg, LIBXSMM_X86_GP_REG_RSP );
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR, LIBXSMM_X86_GP_REG_RSP );
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, temp_reg, b_size);
        libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, temp_reg, LIBXSMM_X86_GP_REG_RSP );
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, LIBXSMM_X86_GP_REG_RSP );
      } else if ( i_micro_kernel_config->atvnni_gemm_stack_alloc_tensors > 0 ) {
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, temp_reg, a_size);
        libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, temp_reg, LIBXSMM_X86_GP_REG_RSP );
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR, LIBXSMM_X86_GP_REG_RSP );
        if (is_offset_brgemm > 0 || is_address_brgemm > 0) {
          libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg );
          libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, temp_reg, b_size);
          libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, temp_reg, LIBXSMM_X86_GP_REG_RSP );
          libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, LIBXSMM_X86_GP_REG_RSP );
        }
      } else if ( i_micro_kernel_config->bvnni_btrans_gemm_stack_alloc_tensors > 0 ) {
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, temp_reg, b_size);
        libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, temp_reg, LIBXSMM_X86_GP_REG_RSP );
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, LIBXSMM_X86_GP_REG_RSP );
        if (is_offset_brgemm > 0 || is_address_brgemm > 0) {
          libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg );
          libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, temp_reg, a_size);
          libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, temp_reg, LIBXSMM_X86_GP_REG_RSP );
          libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_A_EMU_PTR, LIBXSMM_X86_GP_REG_RSP );
        }
      } else {
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, temp_reg, a_size);
        libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, temp_reg, LIBXSMM_X86_GP_REG_RSP );
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_A_EMU_PTR, LIBXSMM_X86_GP_REG_RSP );
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, temp_reg, b_size);
        libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, temp_reg, LIBXSMM_X86_GP_REG_RSP );
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, LIBXSMM_X86_GP_REG_RSP );
      }
    }
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, 128 );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MELTW_STRUCT_PTR, LIBXSMM_X86_GP_REG_RSP );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_setup_stack_frame( libxsmm_generated_code*            io_generated_code,
    const libxsmm_gemm_descriptor*      i_xgemm_desc,
    const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
    libxsmm_micro_kernel_config*        i_micro_kernel_config ) {
  unsigned int temp_reg               = LIBXSMM_X86_GP_REG_R10;
  unsigned int l_is_Ai4_Bi8_gemm = libxsmm_x86_is_Ai4_Bi8_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bbf16_gemm = libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bfp32_gemm = libxsmm_x86_is_Amxfp4_Bfp32_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bi8_gemm = libxsmm_x86_is_Amxfp4_Bi8_gemm(i_xgemm_desc);
  unsigned int l_is_Abf8_Bbf16_gemm = libxsmm_x86_is_Abf8_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Ahf8_Bbf16_gemm = libxsmm_x86_is_Ahf8_Bbf16_gemm(i_xgemm_desc);

  libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBP );
  libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_RBP);
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, 168 );

  /* The stack now looks like this:
   *      10th param (if applicable)                <-- RBP+80
   *      9th param (if applicable)                 <-- RBP+72
   *      8th param (if applicable)                 <-- RBP+64
   *      7th param (if applicable)                 <-- RBP+56
   *      Return address                            <-- RBP+48
   *      Calle SAVED-regs                          <-- RBP[+8,+16,+24,+32,+40]
   *      Entry/saved RBP                           <-- RBP
   *      prefetch A ptr                            <-- RBP-8
   *      prefetch B ptr                            <-- RBP-16
   *      Offset A array ptr                        <-- RBP-24
   *      Offset B array ptr                        <-- RBP-32
   *      Int8 scaling factor                       <-- RBP-40
   *      GEMM_scratch ptr in stack (to be filled)  <-- RBP-48
   *      Eltwise bias ptr                          <-- RBP-56
   *      Eltwise output_ptr                        <-- RBP-64
   *      Eltwise buf1_ptr                          <-- RBP-72
   *      Eltwise buf2_ptr                          <-- RBP-80
   *      Batch-reduce count                        <-- RBP-88,
   *      Transpose A ptr                           <-- RBP-96,
   *      AVX2 Mask                                 <-- RBP-104,
   *      SSE/AVX2 low precision helper             <-- RBP-112,
   *      FP32 A EMULATION PTR                      <-- RBP-120,
   *      FP32 B EMULATION PTR                      <-- RBP-128,
   *      MELTW STRUCT PTR                          <-- RBP-136,
   *      A SCRATCH PTR                             <-- RBP-144,
   *      C SCRATCH PTR                             <-- RBP-152,
   *      C OUTPUT PTR                              <-- RBP-160,
   *      BIAS SCRATCH PTR                          <-- RBP-168, RSP
   *
   * */
  if ( (((LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI & i_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI) || (l_is_Ai4_Bi8_gemm > 0) || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0 || (l_is_Abf8_Bbf16_gemm > 0 || l_is_Ahf8_Bbf16_gemm > 0) || (i_micro_kernel_config->atrans_gemm_stack_alloc_tensors > 0) || (i_micro_kernel_config->avnni_gemm_stack_alloc_tensors > 0) || (i_micro_kernel_config->avnni_btrans_gemm_stack_alloc_tensors > 0) || (i_micro_kernel_config->atvnni_gemm_stack_alloc_tensors > 0) || (i_micro_kernel_config->atvnni_btrans_gemm_stack_alloc_tensors > 0) || (i_micro_kernel_config->bvnni_btrans_gemm_stack_alloc_tensors > 0) || (i_micro_kernel_config->bf8_gemm_via_stack_alloc_tensors > 0) || (i_micro_kernel_config->hf8_gemm_via_stack_alloc_tensors > 0) ) ) {
    libxsmm_generator_gemm_setup_stack_frame_fill_ext_gemm_stack_vars( io_generated_code, i_xgemm_desc, i_micro_kernel_config, i_gp_reg_mapping );
  } else {
    int has_scf = ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
    int has_a_scf = ( ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) && ((LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )))) || ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) && ((LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))))) ? 1 : 0;

    if (has_scf == 1 || has_a_scf == 1) {
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_INT8_SCF, i_gp_reg_mapping->gp_reg_scf );
    }
  }

  /* Now align RSP to 64 byte boundary */
  libxsmm_x86_instruction_alu_imm_i64( io_generated_code, i_micro_kernel_config->alu_mov_instruction, temp_reg, 0xFFFFFFFFFFFFFFC0 );
  libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_ANDQ, temp_reg, LIBXSMM_X86_GP_REG_RSP);

  /* Now alllocate in stack required GEMM scratch if necessary*/
  libxsmm_generator_gemm_setup_stack_frame_allocate_scratch( io_generated_code, i_xgemm_desc, i_micro_kernel_config );

  /* The stack at exit of setup looks like this:
   *
   *      10th param (if applicable)            <-- RBP+80
   *      9th param (if applicable)             <-- RBP+72
   *      8th param (if applicable)             <-- RBP+64
   *      7th param (if applicable)             <-- RBP+56
   *      Return address                        <-- RBP+48
   *      Calle SAVED-regs                      <-- RBP[+8,+16,+24,+32,+40]
   *      Entry/saved RBP                       <-- RBP
   *      prefetch A ptr                        <-- RBP-8
   *      prefetch B ptr                        <-- RBP-16
   *      Offset A array ptr                    <-- RBP-24
   *      Offset B array ptr                    <-- RBP-32
   *      Int8 scaling factor                   <-- RBP-40
   *      GEMM_scratch ptr in stack             <-- RBP-48
   *      Eltwise bias ptr                      <-- RBP-56
   *      Eltwise output_ptr                    <-- RBP-64
   *      Eltwise buf1_ptr                      <-- RBP-72
   *      Eltwise buf2_ptr                      <-- RBP-80
   *      Batch-reduce count                    <-- RBP-88
   *      Transpose A ptr                       <-- RBP-96
   *      AVX2 Mask                             <-- RBP-104
   *      SSE/AVX2 low precision helper         <-- RBP-112,
   *      FP32 A EMULATION PTR                  <-- RBP-120,
   *      FP32 B EMULATION PTR                  <-- RBP-128,
   *      MELTW STRUCT PTR                      <-- RBP-136,
   *      A SCRATCH PTR                         <-- RBP-144,
   *      C SCRATCH PTR                         <-- RBP-152,
   *      C OUTPUT PTR                          <-- RBP-160,
   *      BIAS SCRATCH PTR                      <-- RBP-168, RSP
   *
   *      [ Potential pad for 64b align ]
   *      AVX2 mask, 64b aligned                <-- (RBP-104) contains this address
   *      SSE/AVX2 low precision helper, 64b aligned <-- (RBP-112) contains this address
   *      GEMM scratch, 64b aligned             <-- (RBP-48) contains this address
   */
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_destroy_stack_frame( libxsmm_generated_code*            io_generated_code,
    const libxsmm_gemm_descriptor*      i_xgemm_desc,
    const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
    const libxsmm_micro_kernel_config*  i_micro_kernel_config ) {
  LIBXSMM_UNUSED(i_xgemm_desc);
  LIBXSMM_UNUSED(i_gp_reg_mapping);
  libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_RSP);
  libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBP );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_setup_fusion_microkernel_properties(const libxsmm_gemm_descriptor*      i_xgemm_desc,
                                                                libxsmm_micro_kernel_config*        i_micro_kernel_config ) {
  i_micro_kernel_config->fused_bcolbias          = 0;
  i_micro_kernel_config->fused_hcolbias          = 0;
  i_micro_kernel_config->fused_b8colbias         = 0;
  i_micro_kernel_config->fused_h8colbias         = 0;
  i_micro_kernel_config->fused_scolbias          = 0;
  i_micro_kernel_config->fused_relu              = 0;
  i_micro_kernel_config->fused_relu_nobitmask    = 0;
  i_micro_kernel_config->fused_relu_bwd          = 0;
  i_micro_kernel_config->fused_sigmoid           = 0;
  i_micro_kernel_config->overwrite_C             = 1;
  i_micro_kernel_config->vnni_format_C           = 0;
  i_micro_kernel_config->decompress_A            = 0;
  i_micro_kernel_config->sparsity_factor_A       = 1;
  i_micro_kernel_config->vnni_cvt_output_ext_buf = 0;
  i_micro_kernel_config->norm_to_normT_B_ext_buf = 0;
  i_micro_kernel_config->stride_b_trans          = 0;
  i_micro_kernel_config->fused_eltwise           = 0;
  i_micro_kernel_config->has_colbias_act_fused   = 0;

  i_micro_kernel_config->vnni_format_C  = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_C) > 0) ? 1 : 0;

  if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI & i_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI) ) {
    i_micro_kernel_config->overwrite_C = ((i_xgemm_desc->internal_flags_2 & 0x4) > 0) ? 0 : 1;

    if (i_xgemm_desc->eltw_cp_op == LIBXSMM_MELTW_OPERATION_UNARY) {
      if (i_xgemm_desc->eltw_cp_param == LIBXSMM_MELTW_TYPE_UNARY_RELU) {
        i_micro_kernel_config->has_colbias_act_fused = 1;
        if ((i_xgemm_desc->eltw_cp_flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) {
          i_micro_kernel_config->fused_relu = 1;
        } else {
          i_micro_kernel_config->fused_relu_nobitmask = 1;
        }
      }

      if (i_xgemm_desc->eltw_cp_param == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID) {
        i_micro_kernel_config->has_colbias_act_fused = 1;
        i_micro_kernel_config->fused_sigmoid = 1;
      }

      if (i_xgemm_desc->eltw_cp_param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2) {
        i_micro_kernel_config->vnni_format_C = 1;
        if (i_micro_kernel_config->overwrite_C == 0) {
          i_micro_kernel_config->vnni_cvt_output_ext_buf = 1;
        }
      }

      if (i_xgemm_desc->eltw_cp_param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) {
        i_micro_kernel_config->has_colbias_act_fused = 1;
        i_micro_kernel_config->fused_relu_bwd = 1;
      }
    }

    if (i_xgemm_desc->meltw_operation == LIBXSMM_MELTW_OPERATION_BINARY) {
      if (i_xgemm_desc->meltw_param == LIBXSMM_MELTW_TYPE_BINARY_ADD) {
        if (((i_xgemm_desc->meltw_flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0) > 0 ) ||
            ((i_xgemm_desc->meltw_flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1) > 0 )) {
          i_micro_kernel_config->has_colbias_act_fused = 1;
          if (i_xgemm_desc->meltw_datatype_aux == LIBXSMM_DATATYPE_BF16) {
            i_micro_kernel_config->fused_bcolbias = 1;
          }
          if (i_xgemm_desc->meltw_datatype_aux == LIBXSMM_DATATYPE_F16) {
            i_micro_kernel_config->fused_hcolbias = 1;
          }
          if (i_xgemm_desc->meltw_datatype_aux == LIBXSMM_DATATYPE_BF8) {
            i_micro_kernel_config->fused_b8colbias = 1;
          }
          if (i_xgemm_desc->meltw_datatype_aux == LIBXSMM_DATATYPE_HF8) {
            i_micro_kernel_config->fused_h8colbias = 1;
          }
          if (i_xgemm_desc->meltw_datatype_aux == LIBXSMM_DATATYPE_F32) {
            i_micro_kernel_config->fused_scolbias = 1;
          }
        }
      }
    }

    if (i_xgemm_desc->eltw_ap_op == LIBXSMM_MELTW_OPERATION_UNARY) {
      if ((i_xgemm_desc->internal_flags_2 & 0x1) > 0) {
        if (i_xgemm_desc->eltw_ap_param == LIBXSMM_MELTW_TYPE_UNARY_DECOMPRESS_SPARSE_FACTOR_1) {
          i_micro_kernel_config->decompress_A = 1;
          i_micro_kernel_config->sparsity_factor_A = 1;
        }
        if (i_xgemm_desc->eltw_ap_param == LIBXSMM_MELTW_TYPE_UNARY_DECOMPRESS_SPARSE_FACTOR_2) {
          i_micro_kernel_config->decompress_A = 1;
          i_micro_kernel_config->sparsity_factor_A = 2;
        }
        if (i_xgemm_desc->eltw_ap_param == LIBXSMM_MELTW_TYPE_UNARY_DECOMPRESS_SPARSE_FACTOR_4) {
          i_micro_kernel_config->decompress_A = 1;
          i_micro_kernel_config->sparsity_factor_A = 4;
        }
        if (i_xgemm_desc->eltw_ap_param == LIBXSMM_MELTW_TYPE_UNARY_DECOMPRESS_SPARSE_FACTOR_8) {
          i_micro_kernel_config->decompress_A = 1;
          i_micro_kernel_config->sparsity_factor_A = 8;
        }
        if (i_xgemm_desc->eltw_ap_param == LIBXSMM_MELTW_TYPE_UNARY_DECOMPRESS_SPARSE_FACTOR_16) {
          i_micro_kernel_config->decompress_A = 1;
          i_micro_kernel_config->sparsity_factor_A = 16;
        }
        if (i_xgemm_desc->eltw_ap_param == LIBXSMM_MELTW_TYPE_UNARY_DECOMPRESS_SPARSE_FACTOR_32) {
          i_micro_kernel_config->decompress_A = 1;
          i_micro_kernel_config->sparsity_factor_A = 32;
        }
      }
    }

    if (i_xgemm_desc->eltw_bp_op == LIBXSMM_MELTW_OPERATION_UNARY) {
      if (i_xgemm_desc->eltw_bp_param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) {
        if ((i_xgemm_desc->internal_flags_2 & 0x2) > 0) {
          i_micro_kernel_config->norm_to_normT_B_ext_buf = 1;
          i_micro_kernel_config->stride_b_trans = i_xgemm_desc->ldbp;
        }
      }
    }

    i_micro_kernel_config->fused_eltwise = (i_micro_kernel_config->has_colbias_act_fused == 1) ? 1: 0;
    if (i_micro_kernel_config->decompress_A == 1) {
      i_micro_kernel_config->fused_eltwise = 1;
    }

    if (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) {
      i_micro_kernel_config->fused_eltwise = 1;
    }

    if (i_micro_kernel_config->norm_to_normT_B_ext_buf == 1) {
      i_micro_kernel_config->fused_eltwise = 1;
    }

    if (i_micro_kernel_config->fused_relu_bwd == 1) {
      i_micro_kernel_config->fused_eltwise = 1;
    }
  }
}

LIBXSMM_API_INTERN
int libxsmm_generator_gemm_get_rbp_relative_offset( libxsmm_gemm_stack_var stack_var ) {
  /* The stack at exit of setup looks like this:
   *
   *      10th param (if applicable)                <-- RBP+40
   *      9th param (if applicable)                 <-- RBP+32
   *      8th param (if applicable)                 <-- RBP+24
   *      7th param (if applicable)                 <-- RBP+16
   *      Return address                            <-- RBP+8
   *      Entry/saved RBP                           <-- RBP
   *      prefetch A ptr                            <-- RBP-8
   *      prefetch B ptr                            <-- RBP-16
   *      Offset A array ptr                        <-- RBP-24
   *      Offset B array ptr                        <-- RBP-32
   *      Int8 scaling factor                       <-- RBP-40
   *      GEMM_scratch ptr in stack (to be filled)  <-- RBP-48
   *      Eltwise bias ptr                          <-- RBP-56
   *      Eltwise output_ptr                        <-- RBP-64
   *      Eltwise buf1_ptr                          <-- RBP-72
   *      Eltwise buf2_ptr                          <-- RBP-80
   *      Batch-reduce count                        <-- RBP-88
   *      Transpose A ptr                           <-- RBP-96
   *      AVX2 Mask PTR                             <-- RBP-104
   *      SSE/AVX2 Low precision helper PTR         <-- RBP-112
   *      FP32 A EMULATION PTR                      <-- RBP-120
   *      FP32 B EMULATION PTR                      <-- RBP-128
   *      MELTW STRUCT PTR                          <-- RBP-136
   *      A SCRATCH PTR                             <-- RBP-144
   *      C SCRATCH PTR                             <-- RBP-152
   *      C OUTPUT PTR                              <-- RBP-160
   *      BIAS SCRATCH PTR                          <-- RBP-168
   */

  switch ( stack_var ) {
    case LIBXSMM_GEMM_STACK_VAR_NONE:
      return 0;
    case LIBXSMM_GEMM_STACK_VAR_PFA_PTR:
      return -8;
    case LIBXSMM_GEMM_STACK_VAR_PFB_PTR:
      return -16;
    case LIBXSMM_GEMM_STACK_VAR_A_OFFS_BRGEMM_PTR:
      return -24;
    case LIBXSMM_GEMM_STACK_VAR_B_OFFS_BRGEMM_PTR:
      return -32;
    case LIBXSMM_GEMM_STACK_VAR_INT8_SCF:
      return -40;
    case LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR:
      return -48;
    case LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR:
      return -56;
    case LIBXSMM_GEMM_STACK_VAR_ZPT_PTR:
      return -56;
    case LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR:
      return -56;
    case LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR:
      return -64;
    case LIBXSMM_GEMM_STACK_VAR_ELT_RELU_BITMASK_PTR:
      return -72;
    case LIBXSMM_GEMM_STACK_VAR_ELT_BUF1:
      return -72;
    case LIBXSMM_GEMM_STACK_VAR_ELT_BUF2:
      return -80;
    case LIBXSMM_GEMM_STACK_VAR_AUX_VAR:
      return -80;
    case LIBXSMM_GEMM_STACK_VAR_BRCOUNT:
      return -88;
    case LIBXSMM_GEMM_STACK_VAR_TRANS_EXT_BUF_B:
      return -72;
    case LIBXSMM_GEMM_STACK_VAR_TRANS_EXT_BUF_C:
      return -80;
    case LIBXSMM_GEMM_STACK_VAR_ELT_BITMAP_PTR:
      return -72;
    case LIBXSMM_GEMM_STACK_VAR_ELT_DECOMPRESS_BUF:
      return -80;
    case LIBXSMM_GEMM_STACK_VAR_ARG_7:
      return 56;
    case LIBXSMM_GEMM_STACK_VAR_ARG_8:
      return 64;
    case LIBXSMM_GEMM_STACK_VAR_ARG_9:
      return 72;
    case LIBXSMM_GEMM_STACK_VAR_ARG_10:
      return 80;
    case LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR:
      return -96;
    case LIBXSMM_GEMM_STACK_VAR_AVX2_MASK_PTR:
      return -104;
    case LIBXSMM_GEMM_STACK_VAR_SSE_AVX2_LP_HELPER_PTR:
      return -112;
    case LIBXSMM_GEMM_STACK_VAR_SCF_BRGEMM_PTR:
      return -112;
    case LIBXSMM_GEMM_STACK_VAR_A_EMU_PTR:
      return -120;
    case LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR:
      return -128;
    case LIBXSMM_GEMM_STACK_VAR_MELTW_STRUCT_PTR:
      return -136;
    case LIBXSMM_GEMM_STACK_VAR_A_SCRATCH_PTR:
      return -144;
    case LIBXSMM_GEMM_STACK_VAR_C_SCRATCH_PTR:
      return -152;
    case LIBXSMM_GEMM_STACK_VAR_C_OUTPUT_PTR:
      return -160;
    case LIBXSMM_GEMM_STACK_VAR_BIAS_SCRATCH_PTR:
      return -168;
    case LIBXSMM_GEMM_STACK_VAR_ZPT_BRGEMM_PTR:
      return -168;
    case LIBXSMM_GEMM_STACK_VAR_BSCALE_BRGEMM_PTR:
      return -96;
    case LIBXSMM_GEMM_STACK_VAR_BSCALE_PTR:
      return -152;
  }
  return 0;
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_getval_stack_var( libxsmm_generated_code*             io_generated_code,
                                              const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                              libxsmm_gemm_stack_var              stack_var,
                                              unsigned int                        i_gp_reg ) {
  int offset = libxsmm_generator_gemm_get_rbp_relative_offset(stack_var);
  /* make sure we requested a legal stack var */
  if (offset == 0) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
  libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, offset, i_gp_reg, 0 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_setval_stack_var( libxsmm_generated_code*             io_generated_code,
                                              const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                              libxsmm_gemm_stack_var              stack_var,
                                              unsigned int                        i_gp_reg ) {
  int offset = libxsmm_generator_gemm_get_rbp_relative_offset(stack_var);
  /* make sure we requested to set  a legal stack var */
  if (offset >= 0) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
  libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, offset, i_gp_reg, 1 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_init_micro_kernel_config( libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                      const unsigned int             i_arch,
                                                      const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                      const unsigned int             i_use_masking_a_c ) {
  unsigned int l_is_Amxfp4_Bfp32_gemm = libxsmm_x86_is_Amxfp4_Bfp32_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bi8_gemm = libxsmm_x86_is_Amxfp4_Bi8_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bbf16_gemm = libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc);
  memset(io_micro_kernel_config, 0, sizeof(*io_micro_kernel_config)); /* avoid warning "maybe used uninitialized" */
  libxsmm_generator_gemm_setup_fusion_microkernel_properties(i_xgemm_desc, io_micro_kernel_config);
  if ( (i_arch <= LIBXSMM_TARGET_ARCH_GENERIC) || (i_arch > LIBXSMM_X86_ALLFEAT) ) {
    io_micro_kernel_config->instruction_set = LIBXSMM_TARGET_ARCH_GENERIC;
    io_micro_kernel_config->vector_reg_count = 0;
    io_micro_kernel_config->use_masking_a_c = 0;
    io_micro_kernel_config->vector_name = 'a';
    io_micro_kernel_config->vector_length = 0;
    io_micro_kernel_config->datatype_size_in = 0;
    io_micro_kernel_config->datatype_size_in2 = 0;
    io_micro_kernel_config->datatype_size_out = 0;
    io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
    io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
    io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
    io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
    io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_UNDEF;
    io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_UNDEF;
    io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_UNDEF;
  } else if ( i_arch  <= LIBXSMM_X86_SSE42  ) {
    io_micro_kernel_config->instruction_set = i_arch;
    io_micro_kernel_config->vector_reg_count = 16;
    io_micro_kernel_config->use_masking_a_c = i_use_masking_a_c;
    io_micro_kernel_config->vector_name = 'x';
    if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 2;
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->datatype_size_in2 = 8;
      io_micro_kernel_config->datatype_size_out = 8;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->lda % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_MOVAPD;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_MOVUPD;
      }
      if ( i_arch == LIBXSMM_X86_GENERIC ) {
        io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_MOVSD;
        io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_SHUFPD;
      } else {
        io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_MOVDDUP;
        io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      }
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->ldc % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_MOVAPD;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_MOVAPD;
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_MOVUPD;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_MOVUPD;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_XORPD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_MULPD;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_ADDPD;
    } else if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 4;
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->datatype_size_in2 = 4;
      io_micro_kernel_config->datatype_size_out = 4;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->lda % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_MOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_MOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_MOVSS;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_SHUFPS;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->ldc % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_MOVAPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_MOVAPS;
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_MOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_MOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_XORPS;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_MULPS;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_ADDPS;
    } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 4;
      io_micro_kernel_config->datatype_size_in = 1;
      io_micro_kernel_config->datatype_size_in2 = 1;
      io_micro_kernel_config->datatype_size_out = 4;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->lda % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_MOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_MOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_MOVSS;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_SHUFPS;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->ldc % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_MOVAPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_MOVAPS;
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_MOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_MOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_PXOR;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_PMADDUBSW;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_PADDD;
    } else if ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 4;
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->datatype_size_in2 = 2;
      io_micro_kernel_config->datatype_size_out = 4;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->lda % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_MOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_MOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_MOVSS;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_SHUFPS;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->ldc % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_MOVAPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_MOVAPS;
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_MOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_MOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_PXOR;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_PMADDWD;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_PADDD;
    } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 4;
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->datatype_size_in2 = 2;
      if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
        io_micro_kernel_config->datatype_size_out = 2;
      } else {
        io_micro_kernel_config->datatype_size_out = 4;
      }
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->lda % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_MOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_MOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_MOVSS;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_SHUFPS;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->ldc % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_MOVAPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_MOVAPS;
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_MOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_MOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_PXOR;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_MULPS;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_ADDPS;
    } else {
      /* should not happen as we caught this case earlier */
      io_micro_kernel_config->instruction_set = LIBXSMM_TARGET_ARCH_GENERIC;
      io_micro_kernel_config->vector_reg_count = 0;
      io_micro_kernel_config->use_masking_a_c = 0;
      io_micro_kernel_config->vector_name = 'a';
      io_micro_kernel_config->vector_length = 0;
      io_micro_kernel_config->datatype_size_in = 0;
      io_micro_kernel_config->datatype_size_in2 = 0;
      io_micro_kernel_config->datatype_size_out = 0;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_UNDEF;
    }
  } else if ( i_arch < LIBXSMM_X86_AVX512_VL128_SKX ) {
    io_micro_kernel_config->instruction_set = i_arch;
    io_micro_kernel_config->vector_reg_count = 16;
    io_micro_kernel_config->use_masking_a_c = i_use_masking_a_c;
    io_micro_kernel_config->vector_name = 'y';
    if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 4;
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->datatype_size_in2 = 8;
      io_micro_kernel_config->datatype_size_out = 8;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->lda % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPD;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPD;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VBROADCASTSD;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->ldc % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPD;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPD;
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPD;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPD;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VXORPD;
      if ( i_arch == LIBXSMM_X86_AVX ) {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VMULPD;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPD;
      } else {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PD;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPD;
      }
    } else if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 8;
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->datatype_size_in2 = 4;
      io_micro_kernel_config->datatype_size_out = 4;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->lda % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VBROADCASTSS;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->ldc % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VXORPS;
      if ( i_arch == LIBXSMM_X86_AVX ) {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VMULPS;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
      } else {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
      }
    } else if (l_is_Amxfp4_Bfp32_gemm > 0) {
      io_micro_kernel_config->vector_length = 8;
      io_micro_kernel_config->datatype_size_in = 1;
      io_micro_kernel_config->datatype_size_in2 = 4;
      io_micro_kernel_config->datatype_size_out = 4;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVSD;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VBROADCASTSS;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->ldc % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VXORPS;
      if ( i_arch == LIBXSMM_X86_AVX ) {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VMULPS;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
      } else {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
      }
    } else if (l_is_Amxfp4_Bbf16_gemm > 0) {
      io_micro_kernel_config->vector_length = 8;
      io_micro_kernel_config->datatype_size_in = 1;
      io_micro_kernel_config->datatype_size_in2 = 2;
      if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
        io_micro_kernel_config->datatype_size_out = 2;
      } else {
        io_micro_kernel_config->datatype_size_out = 4;
      }
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVSD;
      if (i_arch >= LIBXSMM_X86_AVX2_SRF) {
        io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VBCSTNEBF162PS;
      } else {
        io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VBROADCASTSS;
      }
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->ldc % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VXORPS;
      if ( i_arch == LIBXSMM_X86_AVX ) {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VMULPS;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
      } else {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
      }
    } else if (l_is_Amxfp4_Bi8_gemm > 0) {
      io_micro_kernel_config->vector_length = 8;
      io_micro_kernel_config->datatype_size_in = 1;
      io_micro_kernel_config->datatype_size_in2 = 1;
      if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
        io_micro_kernel_config->datatype_size_out = 2;
      } else {
        io_micro_kernel_config->datatype_size_out = 4;
      }
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VBROADCASTSS;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->ldc % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VXORPS;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VPDPBSSD;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
    } else if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) &&
                ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) ) {
      io_micro_kernel_config->vector_length = 8;
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->datatype_size_in2 = 2;
      if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
        io_micro_kernel_config->datatype_size_out = 2;
      } else {
        io_micro_kernel_config->datatype_size_out = 4;
      }
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->lda % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VBROADCASTSS;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->ldc % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VXORPS;
      if ( i_arch == LIBXSMM_X86_AVX ) {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VMULPS;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
      } else {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
      }
    } else if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) &&
                ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0) ) {
      io_micro_kernel_config->vector_length = 8;
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->datatype_size_in2 = 2;
      if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
        io_micro_kernel_config->datatype_size_out = 2;
      } else {
        io_micro_kernel_config->datatype_size_out = 4;
      }
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->lda % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTW;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->ldc % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VXORPS;
      if ( i_arch == LIBXSMM_X86_AVX ) {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VMULPS;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
      } else {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
      }
    } else if ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 8;
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->datatype_size_in2 = 2;
      io_micro_kernel_config->datatype_size_out = 4;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->lda % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTD;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->ldc % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VPDPWSSD;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VPADDD;
    } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 8;
      io_micro_kernel_config->datatype_size_in = 1;
      io_micro_kernel_config->datatype_size_in2 = 1;
      io_micro_kernel_config->datatype_size_out = 4;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->lda % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTD;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->ldc % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VPDPBUSD;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VPADDD;
    } else {
      /* should not happen as we caught this case earlier */
      io_micro_kernel_config->instruction_set = LIBXSMM_TARGET_ARCH_GENERIC;
      io_micro_kernel_config->vector_reg_count = 0;
      io_micro_kernel_config->use_masking_a_c = 0;
      io_micro_kernel_config->vector_name = 'a';
      io_micro_kernel_config->vector_length = 0;
      io_micro_kernel_config->datatype_size_in = 0;
      io_micro_kernel_config->datatype_size_in2 = 0;
      io_micro_kernel_config->datatype_size_out = 0;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_UNDEF;
    }
  } else if ( i_arch < LIBXSMM_X86_AVX512_SKX) {
    io_micro_kernel_config->instruction_set = i_arch;
    io_micro_kernel_config->vector_reg_count = 32;
    io_micro_kernel_config->use_masking_a_c = i_use_masking_a_c;
    io_micro_kernel_config->vector_name = 'y';
    if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 4;
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->datatype_size_in2 = 8;
      io_micro_kernel_config->datatype_size_out = 8;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->lda % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPD;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPD;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VBROADCASTSD;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->ldc % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPD;
        if ( (i_use_masking_a_c == 0) ) {
          io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPD;
        } else {
          io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVAPD;
        }
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPD;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPD;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PD;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPD;
    } else if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 8;
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->datatype_size_in2 = 4;
      io_micro_kernel_config->datatype_size_out = 4;
      /* We overwrite in order to support F32BF8 kernels (currently used internally for BF8 emulation via stack allocated arrays) */
      if ( (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) ||  (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))  ) {
        io_micro_kernel_config->datatype_size_out = 1;
      }
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->lda % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VBROADCASTSS;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->ldc % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        if ( (i_use_masking_a_c == 0) ) {
          io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
        } else {
          io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        }
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
    } else if ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
      /* C is 32bit, so we treat all 3 matrices as 32bit element arrays */
      io_micro_kernel_config->vector_length = 8;
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->datatype_size_in2 = 2;
      io_micro_kernel_config->datatype_size_out = 4;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->lda % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTD;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->ldc % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        if ( (i_use_masking_a_c == 0) ) {
          io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
        } else {
          io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        }
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VPDPWSSD;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VPADDD;
    } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
      /* C is 32bit, so we treat all 3 matrices as 32bit element arrays */
      io_micro_kernel_config->vector_length = 8;
      io_micro_kernel_config->datatype_size_in = 1;
      io_micro_kernel_config->datatype_size_in2 = 1;
      io_micro_kernel_config->datatype_size_out = 4;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->lda % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTD;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->ldc % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        if ( (i_use_masking_a_c == 0) ) {
          io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
        } else {
          io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        }
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VPDPBUSD;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VPADDD;
    } else if ( (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) ) {
      io_micro_kernel_config->vector_length = 8;
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->datatype_size_in2 = 2;
      if (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
        io_micro_kernel_config->datatype_size_out = 2;
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVDQU16;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVDQU16;
      } else if (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
        io_micro_kernel_config->datatype_size_out = 4;
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      } else {
        /* Should not happen */
      }
      io_micro_kernel_config->vector_name = 'y';
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVDQU16;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTW;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
    } else if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ) {
      io_micro_kernel_config->vector_length = 8;
      io_micro_kernel_config->datatype_size_in = 1;
      io_micro_kernel_config->datatype_size_in2 = 2;
      if (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
        io_micro_kernel_config->datatype_size_out = 2;
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVDQU16;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVDQU16;
      } else {
        io_micro_kernel_config->datatype_size_out = 4;
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vector_name = 'y';
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVDQU8;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTW;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
    } else if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ) {
      io_micro_kernel_config->vector_length = 8;
      io_micro_kernel_config->datatype_size_in = 1;
      io_micro_kernel_config->datatype_size_in2 = 2;
      if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
        io_micro_kernel_config->datatype_size_out = 2;
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVDQU16;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVDQU16;
      } else {
        io_micro_kernel_config->datatype_size_out = 4;
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vector_name = 'y';
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVDQU8;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTD;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      if (i_arch == LIBXSMM_X86_AVX512_VL256_CPX) {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VDPBF16PS;
      } else {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
      }
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
    } else if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) &&
                ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) ) {
      /* C is 32bit, so we treat all 3 matrices as 32bit element arrays */
      io_micro_kernel_config->vector_length = 8;
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->datatype_size_in2 = 2;
      if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
        io_micro_kernel_config->datatype_size_out = 2;
      } else if ( (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) ||  (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))  ) {
        io_micro_kernel_config->datatype_size_out = 1;
      } else {
        io_micro_kernel_config->datatype_size_out = 4;
      }
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->lda % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTD;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->ldc % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        if ( (i_use_masking_a_c == 0) ) {
          io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
        } else {
          io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        }
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VDPBF16PS;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
    } else if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) &&
                ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0) ) {
      /* C is 32bit, so we treat all 3 matrices as 32bit element arrays */
      io_micro_kernel_config->vector_length = 8;
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->datatype_size_in2 = 2;
      if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
        io_micro_kernel_config->datatype_size_out = 2;
      } else if ( (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) ||  (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))  ) {
        io_micro_kernel_config->datatype_size_out = 1;
      } else {
        io_micro_kernel_config->datatype_size_out = 4;
      }
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->lda % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTW;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->ldc % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        if ( (i_use_masking_a_c == 0) ) {
          io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
        } else {
          io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        }
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
    } else if ( ((LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ))) &&
                ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) ) {
      /* C is 32bit, so we treat all 3 matrices as 32bit element arrays */
      io_micro_kernel_config->vector_length = 8;
      io_micro_kernel_config->datatype_size_in = 1;
      io_micro_kernel_config->datatype_size_in2 = 1;
      if ( (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) ||
           (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) ) {
        io_micro_kernel_config->datatype_size_out = 1;
      } else {
        io_micro_kernel_config->datatype_size_out = 4;
      }
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTB;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        if ( (i_use_masking_a_c == 0) ) {
          io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
        } else {
          io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        }
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
    } else {
      /* should not happen as we caught this case earlier */
      io_micro_kernel_config->instruction_set = LIBXSMM_TARGET_ARCH_GENERIC;
      io_micro_kernel_config->vector_reg_count = 0;
      io_micro_kernel_config->use_masking_a_c = 0;
      io_micro_kernel_config->vector_name = 'a';
      io_micro_kernel_config->vector_length = 0;
      io_micro_kernel_config->datatype_size_in = 0;
      io_micro_kernel_config->datatype_size_in2 = 0;
      io_micro_kernel_config->datatype_size_out = 0;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_UNDEF;
    }
  } else if ( i_arch <= LIBXSMM_X86_ALLFEAT ) {
    io_micro_kernel_config->instruction_set = i_arch;
    io_micro_kernel_config->vector_reg_count = 32;
    io_micro_kernel_config->use_masking_a_c = i_use_masking_a_c;
    io_micro_kernel_config->vector_name = 'z';
    if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 8;
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->datatype_size_in2 = 8;
      io_micro_kernel_config->datatype_size_out = 8;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->lda % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPD;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPD;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VBROADCASTSD;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->ldc % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPD;
        if ( (i_use_masking_a_c == 0) ) {
          io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPD;
        } else {
          io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVAPD;
        }
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPD;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPD;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PD;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPD;
    } else if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 16;
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->datatype_size_in2 = 4;
      io_micro_kernel_config->datatype_size_out = 4;
      /* We overwrite in order to support F32BF8 kernels (currently used internally for BF8 emulation via stack allocated arrays) */
      if ( (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))  ) {
        io_micro_kernel_config->datatype_size_out = 1;
      }
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->lda % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VBROADCASTSS;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->ldc % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        if ( (i_use_masking_a_c == 0) ) {
          io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
        } else {
          io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        }
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
    } else if ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
      /* C is 32bit, so we treat all 3 matrices as 32bit element arrays */
      io_micro_kernel_config->vector_length = 16;
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->datatype_size_in2 = 2;
      io_micro_kernel_config->datatype_size_out = 4;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->lda % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTD;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->ldc % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        if ( (i_use_masking_a_c == 0) ) {
          io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
        } else {
          io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        }
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VPDPWSSD;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VPADDD;
    } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
      /* C is 32bit, so we treat all 3 matrices as 32bit element arrays */
      io_micro_kernel_config->vector_length = 16;
      io_micro_kernel_config->datatype_size_in = 1;
      io_micro_kernel_config->datatype_size_in2 = 1;
      io_micro_kernel_config->datatype_size_out = 4;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->lda % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTD;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->ldc % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        if ( (i_use_masking_a_c == 0) ) {
          io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
        } else {
          io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        }
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VPDPBUSD;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VPADDD;
    } else if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ) {
      io_micro_kernel_config->vector_length = 16;
      io_micro_kernel_config->datatype_size_in = 1;
      io_micro_kernel_config->datatype_size_in2 = 2;
      if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
        io_micro_kernel_config->datatype_size_out = 2;
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVDQU16;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVDQU16;
      } else {
        io_micro_kernel_config->datatype_size_out = 4;
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vector_name = 'z';
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVDQU8;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTD;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      if (i_arch >= LIBXSMM_X86_AVX512_CPX) {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VDPBF16PS;
      } else {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
      }
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
    } else if ( (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ) {
      io_micro_kernel_config->vector_length = 16;
      io_micro_kernel_config->datatype_size_in = 1;
      io_micro_kernel_config->datatype_size_in2 = 2;
      if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
        io_micro_kernel_config->datatype_size_out = 2;
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVDQU16;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVDQU16;
      } else {
        io_micro_kernel_config->datatype_size_out = 4;
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vector_name = 'z';
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVDQU8;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTD;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      if (i_arch >= LIBXSMM_X86_AVX512_CPX) {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VDPBF16PS;
      } else {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
      }
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
    } else if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) &&
                ((LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype ))) ) {
      io_micro_kernel_config->vector_length = 32;
      io_micro_kernel_config->datatype_size_in = 1;
      io_micro_kernel_config->datatype_size_in2 = 2;
      if (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
        io_micro_kernel_config->datatype_size_out = 2;
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVDQU16;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVDQU16;
      } else {
        io_micro_kernel_config->datatype_size_out = 4;
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      if (i_arch >= LIBXSMM_X86_AVX512_SPR) {
        io_micro_kernel_config->vector_length = 32;
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PH;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPH;
      } else {
        io_micro_kernel_config->vector_length = 16;
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
      }
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVDQU8;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTW;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
    } else if ( (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) &&
                ((LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype ))) ) {
      if (i_arch >= LIBXSMM_X86_AVX512_SPR) {
        io_micro_kernel_config->vector_length = 32;
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PH;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPH;
      } else {
        io_micro_kernel_config->vector_length = 16;
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
      }
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->datatype_size_in2 = 2;
      if (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
        io_micro_kernel_config->datatype_size_out = 2;
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVDQU16;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVDQU16;
      } else if (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
        io_micro_kernel_config->datatype_size_out = 4;
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      } else {
        /* Should not happen */
      }
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVDQU16;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTW;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
    } else if ( (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) &&
                ((LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype ))) ) {
      io_micro_kernel_config->vector_length = 16;
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->datatype_size_in2 = 2;
      if (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
        io_micro_kernel_config->datatype_size_out = 2;
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVDQU16;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVDQU16;
      } else if (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
        io_micro_kernel_config->datatype_size_out = 4;
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      } else {
        /* Should not happen */
      }
      io_micro_kernel_config->vector_name = 'z';
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVDQU16;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTW;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
    } else if (  (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) &&
                ((LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype ))) ) {
      io_micro_kernel_config->vector_length = 16;
      io_micro_kernel_config->datatype_size_in = 1;
      io_micro_kernel_config->datatype_size_in2 = 2;
      if (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
        io_micro_kernel_config->datatype_size_out = 2;
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVDQU16;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVDQU16;
      } else {
        io_micro_kernel_config->datatype_size_out = 4;
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vector_name = 'z';
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVDQU8;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTW;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
    } else if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) &&
                ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) ) {
      /* C is 32bit, so we treat all 3 matrices as 32bit element arrays */
      io_micro_kernel_config->vector_length = 16;
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->datatype_size_in2 = 2;
      if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
        io_micro_kernel_config->datatype_size_out = 2;
      } else if ( (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))  ) {
        io_micro_kernel_config->datatype_size_out = 1;
      } else {
        io_micro_kernel_config->datatype_size_out = 4;
      }
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->lda % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTD;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->ldc % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        if ( (i_use_masking_a_c == 0) ) {
          io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
        } else {
          io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        }
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VDPBF16PS;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
    } else if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) &&
                ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0) ) {
      /* C is 32bit, so we treat all 3 matrices as 32bit element arrays */
      io_micro_kernel_config->vector_length = 16;
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->datatype_size_in2 = 2;
      if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
        io_micro_kernel_config->datatype_size_out = 2;
      } else if ( (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))  ) {
        io_micro_kernel_config->datatype_size_out = 1;
      } else {
        io_micro_kernel_config->datatype_size_out = 4;
      }
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->lda % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVDQU16;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVDQU16;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTW;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        assert(0 == (i_xgemm_desc->ldc % io_micro_kernel_config->vector_length));
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        if ( (i_use_masking_a_c == 0) ) {
          io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
        } else {
          io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        }
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
    } else if ( ((LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) )  || (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) )  ) &&
                ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) ) {
      /* C is 32bit, so we treat all 3 matrices as 32bit element arrays */
      io_micro_kernel_config->vector_length = 16;
      io_micro_kernel_config->datatype_size_in = 1;
      io_micro_kernel_config->datatype_size_in2 = 1;
      if ( (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))) {
        io_micro_kernel_config->datatype_size_out = 1;
      } else {
        io_micro_kernel_config->datatype_size_out = 4;
      }
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTB;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        if ( (i_use_masking_a_c == 0) ) {
          io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
        } else {
          io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        }
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
    } else {
      /* should not happen as we caught this case earlier */
      io_micro_kernel_config->instruction_set = LIBXSMM_TARGET_ARCH_GENERIC;
      io_micro_kernel_config->vector_reg_count = 0;
      io_micro_kernel_config->use_masking_a_c = 0;
      io_micro_kernel_config->vector_name = 'a';
      io_micro_kernel_config->vector_length = 0;
      io_micro_kernel_config->datatype_size_in = 0;
      io_micro_kernel_config->datatype_size_in2 = 0;
      io_micro_kernel_config->datatype_size_out = 0;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_UNDEF;
    }
  } else {
    /* that should no happen */
  }

  io_micro_kernel_config->prefetch_instruction = LIBXSMM_X86_INSTR_PREFETCHT1;
  io_micro_kernel_config->alu_add_instruction = LIBXSMM_X86_INSTR_ADDQ;
  io_micro_kernel_config->alu_sub_instruction = LIBXSMM_X86_INSTR_SUBQ;
  io_micro_kernel_config->alu_cmp_instruction = LIBXSMM_X86_INSTR_CMPQ;
  io_micro_kernel_config->alu_jmp_instruction = LIBXSMM_X86_INSTR_JL;
  io_micro_kernel_config->alu_mov_instruction = LIBXSMM_X86_INSTR_MOVQ;
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_add_flop_counter( libxsmm_generated_code*         io_generated_code,
    const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  if ( io_generated_code->code_type == 0 ) {
    char l_new_code[512] = { 0 };
    const unsigned int l_max_code_length = sizeof(l_new_code) - 1;
    int l_code_length = 0;

    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#ifndef NDEBUG\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#ifdef _OPENMP\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#pragma omp atomic\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#endif\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "libxsmm_num_total_flops += %u;\n", 2u * i_xgemm_desc->m * i_xgemm_desc->n * i_xgemm_desc->k);
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#endif\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_kloop( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const unsigned int                 i_m_blocking,
    const unsigned int                 i_k_blocking ) {
  LIBXSMM_UNUSED(i_m_blocking);
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_kloop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_kloop, i_k_blocking);
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_kloop( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    const unsigned int                 i_m_blocking,
    const unsigned int                 i_max_blocked_k,
    const unsigned int                 i_kloop_complete ) {
  LIBXSMM_UNUSED(i_m_blocking);

  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_kloop, i_max_blocked_k );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
  if ( i_kloop_complete != 0 ) {
    int l_b_offset = 0;
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
      l_b_offset = i_xgemm_desc->ldb * i_xgemm_desc->k * i_micro_kernel_config->datatype_size_in2;
    } else {
      l_b_offset = i_xgemm_desc->k * i_micro_kernel_config->datatype_size_in2;
    }

    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction,
        i_gp_reg_mapping->gp_reg_b, l_b_offset );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_reduceloop( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_reduceloop( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc) {
  LIBXSMM_UNUSED(i_xgemm_desc);
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, 1);
  libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_reduce_count, i_gp_reg_mapping->gp_reg_reduce_loop);
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_nloop( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const unsigned int                 i_n_init,
    const unsigned int                 i_n_blocking) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_nloop, i_n_init );
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_nloop, i_n_blocking );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_nloop( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    const unsigned int                 i_n_blocking,
    const unsigned int                 i_n_done ) {
  unsigned int l_is_Ai4_Bi8_gemm = libxsmm_x86_is_Ai4_Bi8_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bfp32_gemm = libxsmm_x86_is_Amxfp4_Bfp32_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bi8_gemm = libxsmm_x86_is_Amxfp4_Bi8_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bbf16_gemm = libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc);
  if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
#if 0
    if (i_micro_kernel_config->vnni_format_C == 0) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c,
        (i_n_blocking*(i_xgemm_desc->ldc)*2 /*(i_micro_kernel_config->datatype_size/2)*/) - ((i_xgemm_desc->m) * 2 /*(i_micro_kernel_config->datatype_size/2)*/) );
    } else {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c,
        (i_n_blocking*(i_xgemm_desc->ldc)*2 /*(i_micro_kernel_config->datatype_size/2)*/) - ((i_xgemm_desc->m) * 2 * 2 /*(i_micro_kernel_config->datatype_size/2)*/) );
    }
#else
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c,
        ((long long)i_n_blocking*(i_xgemm_desc->ldc)*2 /*(i_micro_kernel_config->datatype_size/2)*/) - ((long long)i_xgemm_desc->m * 2 /*(i_micro_kernel_config->datatype_size/2)*/) );
#endif
  } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c,
        ((long long)i_n_blocking*(i_xgemm_desc->ldc)/**(i_micro_kernel_config->datatype_size/4)*/) - ((i_xgemm_desc->m) /** (i_micro_kernel_config->datatype_size/4)*/) );
  } else {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c,
      ((long long)i_n_blocking*(i_xgemm_desc->ldc)*(i_micro_kernel_config->datatype_size_out)) - ((long long)(i_xgemm_desc->m)*(i_micro_kernel_config->datatype_size_out)) );
  }

  if ( (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) &&
       LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ) &&
       LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype ) &&
       ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) > 0 || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0) ) {
    if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) > 0) {
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_INT8_SCF, i_gp_reg_mapping->gp_reg_scf );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_scf, (long long)(i_xgemm_desc->m)*(i_micro_kernel_config->datatype_size_in2) );
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_INT8_SCF, i_gp_reg_mapping->gp_reg_scf );
    }
    if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_zpt, (long long)(i_xgemm_desc->m)*(i_micro_kernel_config->datatype_size_in2) );
    }
  }

  if (l_is_Ai4_Bi8_gemm > 0 && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_zpt, (long long)(i_xgemm_desc->m)*(i_micro_kernel_config->datatype_size_in) );
  }

  if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) &&
       LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ) &&
       LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype ) &&
       (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) > 0 ) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_INT8_SCF, i_gp_reg_mapping->gp_reg_scf );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_scf, (long long)(i_xgemm_desc->m)*4 );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_INT8_SCF, i_gp_reg_mapping->gp_reg_scf );
  }

  /* Also adjust eltwise pointers */
  if ((i_micro_kernel_config->fused_relu == 1) || (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) || (i_micro_kernel_config->fused_relu_bwd == 1) || (i_micro_kernel_config->fused_bcolbias == 1) || (i_micro_kernel_config->fused_hcolbias == 1) || (i_micro_kernel_config->fused_b8colbias == 1) || (i_micro_kernel_config->fused_h8colbias == 1) || (i_micro_kernel_config->fused_scolbias == 1) || (i_micro_kernel_config->overwrite_C == 0)) {
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if ((i_micro_kernel_config->fused_relu == 1) && (i_micro_kernel_config->overwrite_C == 1) ) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    if ( io_generated_code->arch < LIBXSMM_X86_AVX ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_n_blocking*i_xgemm_desc->ldcp)/8 - ( i_micro_kernel_config->m_bitmask_advance ) );
    } else {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_n_blocking*i_xgemm_desc->ldcp)/8 - (((i_xgemm_desc->m+7)/8) ) );
    }
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_n_blocking*(i_xgemm_desc->ldc)*2/*(i_micro_kernel_config->datatype_size/2)*/) - ((long long)(i_xgemm_desc->m) * 2 * 2 /*(i_micro_kernel_config->datatype_size/2)*/)  );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->fused_relu_bwd == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_RELU_BITMASK_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    if ( io_generated_code->arch < LIBXSMM_X86_AVX ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_n_blocking*i_xgemm_desc->ldcp)/8 - ( i_micro_kernel_config->m_bitmask_advance ) );
    } else {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_n_blocking*i_xgemm_desc->ldcp)/8 - ((i_xgemm_desc->m/8) ) );
    }
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_RELU_BITMASK_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  /* In this case also advance the output ptr */
  if (i_micro_kernel_config->overwrite_C == 0) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_n_blocking*(i_xgemm_desc->ldc)*2/*(i_micro_kernel_config->datatype_size/2)*/) - ((long long)(i_xgemm_desc->m) * 2 /*(i_micro_kernel_config->datatype_size/2)*/) );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->fused_bcolbias == 1 || i_micro_kernel_config->fused_hcolbias == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, ( (long long)i_xgemm_desc->m  * 2/*(i_micro_kernel_config->datatype_size/2)*/) );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->fused_b8colbias == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_xgemm_desc->m );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->fused_h8colbias == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_xgemm_desc->m );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->fused_scolbias == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, ( (long long)i_xgemm_desc->m  * 4/*i_micro_kernel_config->datatype_size*/) );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if ((i_micro_kernel_config->fused_relu == 1) || (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) || (i_micro_kernel_config->fused_relu_bwd == 1) || (i_micro_kernel_config->fused_bcolbias == 1) || (i_micro_kernel_config->fused_hcolbias == 1) || (i_micro_kernel_config->fused_b8colbias == 1) || (i_micro_kernel_config->fused_h8colbias == 1) || (i_micro_kernel_config->fused_scolbias == 1) || (i_micro_kernel_config->overwrite_C == 0)) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
  }

  /* B prefetch */
  if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_BL2_VIA_C          ||
       i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C       ||
       i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD    ) {
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0 ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_b_prefetch,
          ((long long)i_n_blocking*(i_xgemm_desc->ldc)*i_micro_kernel_config->datatype_size_in2) - ((long long)(i_xgemm_desc->m)*i_micro_kernel_config->datatype_size_in2) );
    }
  }

#if 0
  if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_CL2 ||
      i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2CL2BL2_VIA_C ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c_prefetch,
        (i_n_blocking*(i_xgemm_desc->ldc)*(i_micro_kernel_config->datatype_size_out)) - ((i_xgemm_desc->m)*(i_micro_kernel_config->datatype_size_out)) );
  }
#endif

  if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
    /* handle trans B */
    int l_b_offset = 0;
     /* k packing factor for VNNI */
    unsigned int l_k_pack_factor = 1;

    /* for VNNI we are stepping through to pack ks */
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A ) {
      l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) );
    }

    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
      l_b_offset = i_n_blocking * i_micro_kernel_config->datatype_size_in2 * l_k_pack_factor;
    } else {
      l_b_offset = i_n_blocking * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2;
    }

    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
    libxsmm_generator_gemm_header_reduceloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_a,
        i_gp_reg_mapping->gp_reg_reduce_loop, 8,
        0,
        i_gp_reg_mapping->gp_reg_help_0,
        0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction,
        i_gp_reg_mapping->gp_reg_help_0, ((long long)i_xgemm_desc->m*i_micro_kernel_config->datatype_size_in*l_k_pack_factor) );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_a,
        i_gp_reg_mapping->gp_reg_reduce_loop, 8,
        0,
        i_gp_reg_mapping->gp_reg_help_0,
        1 );

    if (l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0 ) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_scf );
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_help_0,
          i_gp_reg_mapping->gp_reg_reduce_loop, 8,
          0,
          i_gp_reg_mapping->gp_reg_scf,
          0 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction,
          i_gp_reg_mapping->gp_reg_scf, ((long long)i_xgemm_desc->m) );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_help_0,
          i_gp_reg_mapping->gp_reg_reduce_loop, 8,
          0,
          i_gp_reg_mapping->gp_reg_scf,
          1 );
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_scf );
      if ( l_is_Amxfp4_Bi8_gemm > 0 ) {
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_scf );
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BSCALE_PTR, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_alu_mem( io_generated_code,
            i_micro_kernel_config->alu_mov_instruction,
            i_gp_reg_mapping->gp_reg_help_0,
            i_gp_reg_mapping->gp_reg_reduce_loop, 8,
            0,
            i_gp_reg_mapping->gp_reg_scf,
            0 );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_scf, ((long long)i_n_blocking * (i_xgemm_desc->ldb/32) * 4) );
        libxsmm_x86_instruction_alu_mem( io_generated_code,
            i_micro_kernel_config->alu_mov_instruction,
            i_gp_reg_mapping->gp_reg_help_0,
            i_gp_reg_mapping->gp_reg_reduce_loop, 8,
            0,
            i_gp_reg_mapping->gp_reg_scf,
            1 );
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_scf );
      }
    }

    if (l_is_Ai4_Bi8_gemm > 0 && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0 ) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_zpt );
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_help_0,
          i_gp_reg_mapping->gp_reg_reduce_loop, 8,
          0,
          i_gp_reg_mapping->gp_reg_zpt,
          0 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction,
          i_gp_reg_mapping->gp_reg_zpt, ((long long)i_xgemm_desc->m) );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_help_0,
          i_gp_reg_mapping->gp_reg_reduce_loop, 8,
          0,
          i_gp_reg_mapping->gp_reg_zpt,
          1 );
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_zpt );
    }

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_b,
        i_gp_reg_mapping->gp_reg_reduce_loop, 8,
        0,
        i_gp_reg_mapping->gp_reg_help_0,
        0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_help_0, l_b_offset );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_b,
        i_gp_reg_mapping->gp_reg_reduce_loop, 8,
        0,
        i_gp_reg_mapping->gp_reg_help_0,
        1 );
    if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2          ||
         i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C    ) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_a_prefetch,
          i_gp_reg_mapping->gp_reg_reduce_loop, 8,
          0,
          i_gp_reg_mapping->gp_reg_help_0,
          0 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction,
          i_gp_reg_mapping->gp_reg_help_0, ((long long)i_xgemm_desc->m*i_micro_kernel_config->datatype_size_in*l_k_pack_factor) );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_a_prefetch,
          i_gp_reg_mapping->gp_reg_reduce_loop, 8,
          0,
          i_gp_reg_mapping->gp_reg_help_0,
          1 );
    }
    libxsmm_generator_gemm_footer_reduceloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc);
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
  } else {
    /* handle trans B */
    int l_b_offset = 0;
     /* k packing factor for VNNI */
    unsigned int l_k_pack_factor = 1;

    /* for VNNI we are stepping through to pack ks */
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A ) {
      l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) );
    }

    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
      l_b_offset = i_n_blocking * i_micro_kernel_config->datatype_size_in2 * l_k_pack_factor;
    } else {
      l_b_offset = i_n_blocking * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2;
    }

    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_b, l_b_offset );

    if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction,
          i_gp_reg_mapping->gp_reg_bitmap_a, ((long long)(i_xgemm_desc->m/8)*l_k_pack_factor) );

    } else {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction,
          i_gp_reg_mapping->gp_reg_a, ((long long)i_xgemm_desc->m*i_micro_kernel_config->datatype_size_in*l_k_pack_factor) );
    }

    if (l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 ||  l_is_Amxfp4_Bi8_gemm > 0) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_xgemm_desc->m));
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      if (l_is_Amxfp4_Bi8_gemm > 0) {
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BSCALE_PTR, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0,((long long)i_n_blocking * (i_xgemm_desc->ldb/32) * 4));
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BSCALE_PTR, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      }
    }

    if (l_is_Ai4_Bi8_gemm > 0 && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0 ) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_xgemm_desc->m));
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
    }

    if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2          ||
         i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C    ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction,
          i_gp_reg_mapping->gp_reg_a_prefetch, ((long long)i_xgemm_desc->m*i_micro_kernel_config->datatype_size_in*l_k_pack_factor) );
    }
  }
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_nloop, i_n_done );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_mloop( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const unsigned int                 i_m_init,
    const unsigned int                 i_m_blocking ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_mloop, i_m_init );
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_mloop, i_m_blocking );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_mloop( libxsmm_generated_code*            io_generated_code,
                                          libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                          const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                          const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                          const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                          const unsigned int                 i_m_blocking,
                                          const unsigned int                 i_m_done ) {
  /* k packing factor for VNNI */
  unsigned int l_k_pack_factor = 1;
  unsigned int l_is_Ai4_Bf16_gemm = (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_INT4_VNNI2) > 0) &&
                                     ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
                                      (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) &&
                                      ((LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))))) ? 1 : 0;
  unsigned int l_is_Ai4_Bi8_gemm = libxsmm_x86_is_Ai4_Bi8_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bfp32_gemm = libxsmm_x86_is_Amxfp4_Bfp32_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bi8_gemm = libxsmm_x86_is_Amxfp4_Bi8_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bbf16_gemm = libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_k_scale = (l_is_Ai4_Bf16_gemm > 0 || l_is_Ai4_Bi8_gemm > 0 || l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bi8_gemm) ? 2 : 1;

  /* for VNNI we are stepping through to pack ks */
  if ( ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A) ) {
    l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) );
  }

  /* advance C pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
      i_gp_reg_mapping->gp_reg_c, (long long)i_m_blocking*(i_micro_kernel_config->datatype_size_out) );

  if ( (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) &&
       LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ) &&
       LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype ) &&
       ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) > 0 || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0) ) {
    if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) > 0) {
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_INT8_SCF, i_gp_reg_mapping->gp_reg_scf );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_scf, (long long)(i_m_blocking)*(i_micro_kernel_config->datatype_size_in2) );
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_INT8_SCF, i_gp_reg_mapping->gp_reg_scf );
    }
    if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_zpt, (long long)(i_m_blocking)*(i_micro_kernel_config->datatype_size_in2) );
    }
  }

  if (l_is_Ai4_Bi8_gemm > 0 && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_zpt, (long long)(i_m_blocking)*(i_micro_kernel_config->datatype_size_in) );
  }

  if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) &&
       LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ) &&
       LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype ) &&
       (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) > 0 ) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_INT8_SCF, i_gp_reg_mapping->gp_reg_scf );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_scf, (long long)(i_m_blocking)*4 );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_INT8_SCF, i_gp_reg_mapping->gp_reg_scf );
  }

  /* Also adjust eltwise pointers */
  if ((i_micro_kernel_config->fused_relu == 1) || (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) || (i_micro_kernel_config->fused_relu_bwd == 1) || (i_micro_kernel_config->fused_bcolbias == 1) || (i_micro_kernel_config->fused_hcolbias == 1) || (i_micro_kernel_config->fused_b8colbias == 1) || (i_micro_kernel_config->fused_h8colbias == 1) || (i_micro_kernel_config->fused_scolbias == 1) || (i_micro_kernel_config->overwrite_C == 0)) {
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if ((i_micro_kernel_config->fused_relu == 1) && (i_micro_kernel_config->overwrite_C == 1) ) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    if ( io_generated_code->arch < LIBXSMM_X86_AVX ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (i_m_blocking+3)/8 );
    } else {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (i_m_blocking+7)/8 );
    }
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_m_blocking*2*2/*(i_micro_kernel_config->datatype_size/2)*/ );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->fused_relu_bwd == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_RELU_BITMASK_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    if ( io_generated_code->arch < LIBXSMM_X86_AVX ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (i_m_blocking+3)/8 );
    } else {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (i_m_blocking+7)/8 );
    }
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_RELU_BITMASK_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->overwrite_C == 0) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_m_blocking*2/*(i_micro_kernel_config->datatype_size/2)*/ );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->fused_bcolbias == 1 || i_micro_kernel_config->fused_hcolbias == 1 ) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_m_blocking * 2/*(i_micro_kernel_config->datatype_size/2)*/ );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->fused_b8colbias == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_m_blocking );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->fused_h8colbias == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_m_blocking );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->fused_scolbias == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_m_blocking * 4 /*i_micro_kernel_config->datatype_size*/ );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if ((i_micro_kernel_config->fused_relu == 1) || (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) || (i_micro_kernel_config->fused_relu_bwd == 1) || (i_micro_kernel_config->fused_bcolbias == 1) || (i_micro_kernel_config->fused_hcolbias == 1) || (i_micro_kernel_config->fused_b8colbias == 1) || (i_micro_kernel_config->fused_h8colbias == 1) ||  (i_micro_kernel_config->fused_scolbias == 1) || (i_micro_kernel_config->overwrite_C == 0)) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
  }


  /* C prefetch */
#if 0
  if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_CL2 ||
      i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2CL2BL2_VIA_C ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_c_prefetch, i_m_blocking*(i_micro_kernel_config->datatype_size_out) );

  }
#endif

  /* B prefetch */
  if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_BL2_VIA_C          ||
       i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C       ||
       i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD    ) {
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0 ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_b_prefetch, (long long)i_m_blocking*i_micro_kernel_config->datatype_size_in2 );
    }
  }

  /* A prefetch */
  if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2 ||
      i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C) {
    if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
      if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2 ) {
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
        libxsmm_generator_gemm_header_reduceloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config );
        libxsmm_x86_instruction_alu_mem( io_generated_code,
            i_micro_kernel_config->alu_mov_instruction,
            i_gp_reg_mapping->gp_reg_a_prefetch,
            i_gp_reg_mapping->gp_reg_reduce_loop, 8,
            0,
            i_gp_reg_mapping->gp_reg_help_0,
            0 );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0,
            ((long long)(i_xgemm_desc->k/l_k_scale) * (i_micro_kernel_config->datatype_size_in) * (i_xgemm_desc->lda) ) -
            ((long long)i_m_blocking * (i_micro_kernel_config->datatype_size_in*l_k_pack_factor)) );
        libxsmm_x86_instruction_alu_mem( io_generated_code,
            i_micro_kernel_config->alu_mov_instruction,
            i_gp_reg_mapping->gp_reg_a_prefetch,
            i_gp_reg_mapping->gp_reg_reduce_loop, 8,
            0,
            i_gp_reg_mapping->gp_reg_help_0,
            1 );
        libxsmm_generator_gemm_footer_reduceloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc);
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      }
    } else {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_a_prefetch,
          ((long long)(i_xgemm_desc->k) * (i_micro_kernel_config->datatype_size_in) * (i_xgemm_desc->lda) ) -
          ((long long)i_m_blocking * (i_micro_kernel_config->datatype_size_in*l_k_pack_factor)) );
    }
  }

  /* advance A pointer */
  if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
    libxsmm_generator_gemm_header_reduceloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_a,
        i_gp_reg_mapping->gp_reg_reduce_loop, 8,
        0,
        i_gp_reg_mapping->gp_reg_help_0,
        0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0,
        ((long long)(i_xgemm_desc->k/l_k_scale) * (i_micro_kernel_config->datatype_size_in) * (i_xgemm_desc->lda) ) - ((long long)i_m_blocking * (i_micro_kernel_config->datatype_size_in) * l_k_pack_factor) );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_a,
        i_gp_reg_mapping->gp_reg_reduce_loop, 8,
        0,
        i_gp_reg_mapping->gp_reg_help_0,
        1 );

    if (l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_help_1 );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_help_1,
          i_gp_reg_mapping->gp_reg_reduce_loop, 8,
          0,
          i_gp_reg_mapping->gp_reg_help_0,
          0 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_m_blocking) );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_help_1,
          i_gp_reg_mapping->gp_reg_reduce_loop, 8,
          0,
          i_gp_reg_mapping->gp_reg_help_0,
          1 );
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
    }

    if (l_is_Ai4_Bi8_gemm > 0 && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_gp_reg_mapping->gp_reg_help_1 );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_help_1,
          i_gp_reg_mapping->gp_reg_reduce_loop, 8,
          0,
          i_gp_reg_mapping->gp_reg_help_0,
          0 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_m_blocking) );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_help_1,
          i_gp_reg_mapping->gp_reg_reduce_loop, 8,
          0,
          i_gp_reg_mapping->gp_reg_help_0,
          1 );
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
    }

    libxsmm_generator_gemm_footer_reduceloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc);
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
  } else {
    if (l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0 ) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_m_blocking));
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
    }
    if (l_is_Ai4_Bi8_gemm > 0 && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_m_blocking));
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
    }

    if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_bitmap_a,
          ((long long)((i_xgemm_desc->k/8)/l_k_scale) * (i_xgemm_desc->lda) ) - ((long long)(i_m_blocking/8) * l_k_pack_factor) );
    } else {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_a,
          ((long long)(i_xgemm_desc->k/l_k_scale) * (i_micro_kernel_config->datatype_size_in) * (i_xgemm_desc->lda) ) - ((long long)i_m_blocking * (i_micro_kernel_config->datatype_size_in) * l_k_pack_factor) );
    }
  }

  /* loop handling */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_mloop, i_m_done );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_load_C( libxsmm_generated_code*             io_generated_code,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    const unsigned int                 i_m_blocking,
    const unsigned int                 i_n_blocking ) {
  unsigned int l_m_blocking, l_vec_reg_acc_start;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  unsigned int l_is_Ai4_Bf16_gemm = (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_INT4_VNNI2) > 0) &&
                                     ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
                                      (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) &&
                                      ((LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))))) ? 1 : 0;
  unsigned int l_is_Ai8_Bf16_gemm = ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
                                     (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) &&
                                     ((LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )))) ? 1 : 0;
  unsigned int l_is_Amxfp4_Bbf16_gemm = libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bfp32_gemm = libxsmm_x86_is_Amxfp4_Bfp32_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bi8_gemm = libxsmm_x86_is_Amxfp4_Bi8_gemm(i_xgemm_desc);
  unsigned int l_is_Ai8_Bbf16_gemm = ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype ) && l_is_Amxfp4_Bbf16_gemm == 0) &&
                                     (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) &&
                                     ((LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )))) ? 1 : 0;
  unsigned int l_is_Ai4_Bi8_gemm = libxsmm_x86_is_Ai4_Bi8_gemm(i_xgemm_desc);
  unsigned int l_load_scf_vector = (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) > 0) && (l_is_Ai8_Bf16_gemm > 0 || l_is_Ai8_Bbf16_gemm > 0)) ? 1 : 0;
  unsigned int l_load_zpt_vector = (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0) && (l_is_Ai8_Bf16_gemm > 0 || l_is_Ai4_Bi8_gemm > 0)) ? 1 : 0;

  assert(0 < i_micro_kernel_config->vector_length);
  /* deriving register blocking from kernel config */
  l_m_blocking = ( i_m_blocking % i_micro_kernel_config->vector_length == 0 ) ? i_m_blocking/i_micro_kernel_config->vector_length : (i_m_blocking/i_micro_kernel_config->vector_length)+1;
  /* start register of accumulator */
  l_vec_reg_acc_start = i_micro_kernel_config->vector_reg_count - (i_n_blocking * l_m_blocking);

  if (l_load_scf_vector > 0) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_INT8_SCF, i_gp_reg_mapping->gp_reg_scf );
    for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
      if (l_is_Ai8_Bf16_gemm > 0) {
        /* we only mask the last m-blocked load */
        unsigned int l_use_f16_replacement_fma = ( io_generated_code->arch < LIBXSMM_X86_AVX512_SPR && LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype )) ? 1 : 0;
        char vname_ld = (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype)) ? ((i_micro_kernel_config->vector_name == 'z') ? 'y' : 'x') : ((l_use_f16_replacement_fma > 0) ? ((i_micro_kernel_config->vector_name == 'z') ? 'y' : 'x') : i_micro_kernel_config->vector_name);

        libxsmm_x86_instruction_unified_vec_move( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU16,
            i_gp_reg_mapping->gp_reg_scf, LIBXSMM_X86_GP_REG_UNDEF, 0,
            (l_m * i_micro_kernel_config->vector_length) * (i_micro_kernel_config->datatype_size_in2),
            vname_ld,
            (l_is_Ai4_Bf16_gemm > 0) ? 2 + l_m : l_m, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 1, 0 );
        /* Convert the scaling factor to FP32 */
        if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype) || io_generated_code->arch < LIBXSMM_X86_AVX512_SPR ) {
          char vname_cvt = (vname_ld == 'y') ? 'z' : ((vname_ld == 'x') ? 'y' : 'z');
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, vname_cvt, (l_is_Ai4_Bf16_gemm > 0) ? 2 + l_m : l_m, (l_is_Ai4_Bf16_gemm > 0) ? 2 + l_m : l_m );
        }
      } else {
        libxsmm_x86_instruction_unified_vec_move( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS,
            i_gp_reg_mapping->gp_reg_scf, LIBXSMM_X86_GP_REG_UNDEF, 0,
            (l_m * i_micro_kernel_config->vector_length) * 4,
            i_micro_kernel_config->vector_name,
            l_m+1, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 1, 0 );
      }
    }
  }

  if (l_load_zpt_vector > 0) {
    for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
      if (l_is_Ai8_Bf16_gemm > 0) {
        /* we only mask the last m-blocked load */
        unsigned int l_use_f16_replacement_fma = ( io_generated_code->arch < LIBXSMM_X86_AVX512_SPR && LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype )) ? 1 : 0;
        char vname_ld = (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype)) ? ((i_micro_kernel_config->vector_name == 'z') ? 'y' : 'x') : ((l_use_f16_replacement_fma > 0) ? ((i_micro_kernel_config->vector_name == 'z') ? 'y' : 'x') : i_micro_kernel_config->vector_name);
        unsigned int l_vreg = (l_is_Ai4_Bf16_gemm > 0) ? 2 + l_m + l_m_blocking : l_m + l_m_blocking;
        libxsmm_x86_instruction_unified_vec_move( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU16,
            i_gp_reg_mapping->gp_reg_zpt, LIBXSMM_X86_GP_REG_UNDEF, 0,
            (l_m * i_micro_kernel_config->vector_length) * (i_micro_kernel_config->datatype_size_in2),
            vname_ld,
            l_vreg, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 1, 0 );
        /* Convert the scaling factor to FP32 */
        if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype) || io_generated_code->arch < LIBXSMM_X86_AVX512_SPR ) {
          char vname_cvt = (vname_ld == 'y') ? 'z' : ((vname_ld == 'x') ? 'y' : 'z');
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, vname_cvt, l_vreg, l_vreg);
        }
      }
      if (l_is_Ai4_Bi8_gemm > 0) {
        char vname_ld = 'x';
        unsigned int l_vreg = 3 + l_m;
        unsigned int l_vperm_reg = 2;
        libxsmm_x86_instruction_unified_vec_move( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU8,
            i_gp_reg_mapping->gp_reg_zpt, LIBXSMM_X86_GP_REG_UNDEF, 0,
            (l_m * i_micro_kernel_config->vector_length) * (i_micro_kernel_config->datatype_size_in2),
            vname_ld,
            l_vreg, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 1, 0 );
        if (io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) {
          /* Permute with vperm register 2 to get partially brodcasted vector: [0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5 ...] */
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPERMB, i_micro_kernel_config->vector_name, l_vreg, l_vperm_reg, l_vreg);
        } else {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VPMOVZXBD, i_micro_kernel_config->vector_name, l_vreg, l_vreg);
          libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, i_micro_kernel_config->vector_name, l_vreg, l_vperm_reg, 8 );
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPORD, i_micro_kernel_config->vector_name, l_vperm_reg, l_vreg, l_vreg );
          libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, i_micro_kernel_config->vector_name, l_vreg, l_vperm_reg, 16 );
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPORD, i_micro_kernel_config->vector_name, l_vperm_reg, l_vreg, l_vreg );
        }
      }
    }
  }

  /* load C accumulator */
  if (0 == (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) { /* Beta=1 */
    /* pure BF16 kernel */
    if (  (i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX) &&
          ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ) ) {
      /* we add when scaling during conversion to FP32 */
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
          /* load 16 bit values into xmm portion of the register */
          if ( (i_micro_kernel_config->use_masking_a_c != 0) && ( l_m == (l_m_blocking - 1) ) ) {
            libxsmm_generator_maskedload_16bit_sse( io_generated_code, i_gp_reg_mapping->gp_reg_help_0, 0,
                                                    i_gp_reg_mapping->gp_reg_c, LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                    ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                                                    0, i_m_blocking % i_micro_kernel_config->vector_length );
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_MOVSD,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                'x', 0, 0, 1, 0 );
          }
          /* up-convert bf16 to fp32 */
          libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, 'x', 0, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
        }
      }
      /* Check if we have to add bias */
      if (i_micro_kernel_config->fused_bcolbias == 1) {
        libxsmm_generator_gemm_add_colbias_to_2D_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
            LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, l_vec_reg_acc_start, l_m_blocking, i_n_blocking, i_m_blocking % i_micro_kernel_config->vector_length );
      }
    } else if (( ( ((i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX2) && (i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_VL256_SKX))
          ) && ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ) ) ||
        ((io_generated_code->arch >= LIBXSMM_X86_AVX2 && (l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0)) && LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))) {
      /* we add when scaling during conversion to FP32 */
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
          /* load 16 bit values into xmm portion of the register */
          if ( (i_micro_kernel_config->use_masking_a_c != 0) && ( l_m == (l_m_blocking - 1) ) ) {
            libxsmm_generator_maskedload_16bit_avx2( io_generated_code, i_gp_reg_mapping->gp_reg_help_0,
                                                     i_gp_reg_mapping->gp_reg_c, LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                     ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                                                     l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), i_m_blocking % i_micro_kernel_config->vector_length );
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                i_micro_kernel_config->c_vmove_instruction,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                'x', l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), 0, 1, 0 );
          }
          /* up-convert bf16 to fp32 */
          libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, 'y', l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
        }
      }
      /* Check if we have to add bias */
      if (i_micro_kernel_config->fused_bcolbias == 1) {
        libxsmm_generator_gemm_add_colbias_to_2D_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
            LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, l_vec_reg_acc_start, l_m_blocking, i_n_blocking, i_m_blocking % i_micro_kernel_config->vector_length );
      }
    } else if ( ( ((i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_SKX) && (i_micro_kernel_config->instruction_set <= LIBXSMM_X86_ALLFEAT)) ||
           ((i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_VL256_SKX) && ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX ))
          ) && (( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ) ||
                ( l_is_Ai8_Bbf16_gemm > 0 && LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))) ) {
      /* we add when scaling during conversion to FP32 */
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
          /* load 16 bit values into xmm portion of the register */
          if ( (i_micro_kernel_config->use_masking_a_c != 0) && ( l_m == (l_m_blocking - 1) ) ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVDQU16,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                ( ( i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_VL256_SKX) && ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX ) ) ? 'y' : 'z',
                l_vec_reg_acc_start-1, 2, 1, 0 );
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                i_micro_kernel_config->c_vmove_instruction,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                ( ( i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_VL256_SKX) && ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX ) ) ? 'x' : 'y',
                l_vec_reg_acc_start-1, 0, 1, 0 );
          }
          /* up-convert bf16 to fp32 */
          libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, ( ( i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_VL256_SKX) && ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX ) ) ? 'y' : 'z',
                                                   l_vec_reg_acc_start-1, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
        }
      }
      /* Check if we have to add bias */
      if (i_micro_kernel_config->fused_bcolbias == 1) {
        libxsmm_generator_gemm_add_colbias_to_2D_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
            LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, l_vec_reg_acc_start, l_m_blocking, i_n_blocking, i_m_blocking % i_micro_kernel_config->vector_length );
      }
    /* pure int8 kernel */
    } else if (( ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch <= LIBXSMM_X86_ALLFEAT) ) &&
                ( ( (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) ) || ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) )  ||  ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) ) )) ||
        ( ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch <= LIBXSMM_X86_ALLFEAT) ) &&
                ( ( (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) ) || ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) ) || ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) ) )) ) {

      if (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
        libxsmm_generator_vcvt_hf8_tofrom_f32_avx512_prep_stack( io_generated_code,  i_gp_reg_mapping->gp_reg_help_2);
      }

      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
          /* load 8bit float values into registers */
          if ( (i_micro_kernel_config->use_masking_a_c != 0) && ( l_m == (l_m_blocking - 1) ) ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                io_generated_code->arch,
                LIBXSMM_X86_INSTR_VMOVDQU8,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                ( ( i_micro_kernel_config->instruction_set > LIBXSMM_X86_AVX2) && ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX ) ) ? 'y' : 'z',
                0, 2, 1, 0 );
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                io_generated_code->arch,
                ( io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? LIBXSMM_X86_INSTR_VMOVSD : i_micro_kernel_config->c_vmove_instruction,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                'x',
                0, 0, 1, 0 );
          }

          if (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
            /* upconconvert 8bit float with replacement sequence */
            libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                               0, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n ));
          } else if (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
            libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name, 0, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n ), 1, 2, 6, 7);
          }
        }
      }

      if ( (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) || (i_micro_kernel_config->fused_h8colbias == 1)) {
        libxsmm_generator_vcvt_hf8_tofrom_f32_avx512_clean_stack( io_generated_code,  i_gp_reg_mapping->gp_reg_help_2);
      }

      /* Check if we have to add bias */
      if (i_micro_kernel_config->fused_b8colbias == 1) {
        libxsmm_generator_gemm_add_colbias_to_2D_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
            LIBXSMM_DATATYPE_BF8, LIBXSMM_DATATYPE_F32, l_vec_reg_acc_start, l_m_blocking, i_n_blocking, i_m_blocking % i_micro_kernel_config->vector_length );
      } else if (i_micro_kernel_config->fused_h8colbias == 1) {
        libxsmm_generator_gemm_add_colbias_to_2D_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
            LIBXSMM_DATATYPE_HF8, LIBXSMM_DATATYPE_F32, l_vec_reg_acc_start, l_m_blocking, i_n_blocking, i_m_blocking % i_micro_kernel_config->vector_length );
      }
    } else if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) && l_is_Amxfp4_Bi8_gemm == 0) {
      if (i_micro_kernel_config->fused_scolbias == 1) {
        libxsmm_generator_gemm_load_colbias_to_2D_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
              LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, l_vec_reg_acc_start, l_m_blocking, i_n_blocking, i_m_blocking % i_micro_kernel_config->vector_length );
      } else {
        /* overwriting C, so let's xout accumulator */
        for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
          for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
            if ( io_generated_code->arch >= LIBXSMM_X86_AVX ) {
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                  i_micro_kernel_config->vxor_instruction,
                  i_micro_kernel_config->vector_name,
                  l_vec_reg_acc_start + l_m + (l_m_blocking * l_n),
                  l_vec_reg_acc_start + l_m + (l_m_blocking * l_n),
                  l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
            } else {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                  i_micro_kernel_config->vxor_instruction,
                  i_micro_kernel_config->vector_name,
                  l_vec_reg_acc_start + l_m + (l_m_blocking * l_n),
                  l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
            }
          }
        }
      }
    } else {
      /* in case of AVX/AVX2 we need to load the mask into an ymm */
      if ( (i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX) && (i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_VL256_SKX) &&
           (i_micro_kernel_config->use_masking_a_c != 0) ) {
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_AVX2_MASK_PTR, i_gp_reg_mapping->gp_reg_help_1 );
        libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_VMOVUPS,
                                          i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 'y', (l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0) ? 2 : 1, 0, 0, 0 );
      }

      /* In this case we have to split loading C in 2 chunks of vlen/2 each  */
      if (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype) &&
          LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype) &&
          io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) {
        for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
          for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
            unsigned int l_tmp_vreg = l_vec_reg_acc_start - 1;
            unsigned int l_dst_vreg = l_vec_reg_acc_start + l_m + (l_m_blocking * l_n);
            char vname_cvt = (i_micro_kernel_config->vector_name == 'y') ? 'z' : ((i_micro_kernel_config->vector_name == 'x') ? 'y' : i_micro_kernel_config->vector_name);
            /* we only mask the last m-blocked load */
            libxsmm_x86_instruction_unified_vec_move( io_generated_code, i_micro_kernel_config->c_vmove_instruction,
                                                      i_gp_reg_mapping->gp_reg_c, LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                      ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                                                      i_micro_kernel_config->vector_name,
                                                      l_dst_vreg, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 2, 0 );

            libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, vname_cvt, l_dst_vreg, l_dst_vreg, 0,
                (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );

            libxsmm_x86_instruction_unified_vec_move( io_generated_code, i_micro_kernel_config->c_vmove_instruction,
                                                      i_gp_reg_mapping->gp_reg_c, LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                      ((l_n * i_xgemm_desc->ldc) + ((2*l_m+1) * (i_micro_kernel_config->vector_length/2))) * (i_micro_kernel_config->datatype_size_out),
                                                      i_micro_kernel_config->vector_name,
                                                      l_tmp_vreg, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 3, 0 );

            libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, vname_cvt, l_tmp_vreg, l_tmp_vreg, 0,
                (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );

            libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VINSERTI64X4, vname_cvt, l_tmp_vreg, l_dst_vreg, l_dst_vreg, 0, 0, 0, 1 );
          }
        }
      } else {
        /* adding to C, so let's load C */
        for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
          for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
            char vname_load = i_micro_kernel_config->vector_name;
            unsigned int l_mask_reg_or_val = ( i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX ) ? (l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0) ? 2 : 1 : i_m_blocking%i_micro_kernel_config->vector_length;
            if ((LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype) && LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype)) ||
                (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype) && LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype) && io_generated_code->arch < LIBXSMM_X86_AVX512_SPR)) {
              vname_load = (vname_load == 'z') ? 'y' : 'x';
            }
            /* we only mask the last m-blocked load */
            libxsmm_x86_instruction_unified_vec_move( io_generated_code, i_micro_kernel_config->c_vmove_instruction,
                                                      i_gp_reg_mapping->gp_reg_c, LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                      ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                                                      vname_load,
                                                      l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, l_mask_reg_or_val, 0 );

            if (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype)) {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
            }

            if (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype) && LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype)) {
              libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), 0,
                  (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
            }

          }
#if 0
          if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_CL2 ||
              i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2CL2BL2_VIA_C )  {
            for (l_m = 0; l_m < l_m_blocking; l_m += l_m++ ) {
              libxsmm_x86_instruction_prefetch( io_generated_code,
                  i_micro_kernel_config->prefetch_instruction,
                  i_gp_reg_mapping->gp_reg_c_prefetch,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out));
            }
          }
#endif
        }
      }
      /* Check if we have to add bias */
      if (i_micro_kernel_config->fused_scolbias == 1) {
        libxsmm_generator_gemm_add_colbias_to_2D_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
            LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, l_vec_reg_acc_start, l_m_blocking, i_n_blocking, i_m_blocking % i_micro_kernel_config->vector_length );
      }
      if (i_micro_kernel_config->fused_hcolbias == 1) {
        libxsmm_generator_gemm_add_colbias_to_2D_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
            LIBXSMM_DATATYPE_F16, (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype)) ? LIBXSMM_DATATYPE_F16 : LIBXSMM_DATATYPE_F32, l_vec_reg_acc_start, l_m_blocking, i_n_blocking, i_m_blocking % i_micro_kernel_config->vector_length );
      }
    }
  } else {
    if (i_micro_kernel_config->fused_scolbias == 1) {
      libxsmm_generator_gemm_load_colbias_to_2D_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
            LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, l_vec_reg_acc_start, l_m_blocking, i_n_blocking, i_m_blocking % i_micro_kernel_config->vector_length );
    } else if (i_micro_kernel_config->fused_bcolbias == 1) {
      libxsmm_generator_gemm_load_colbias_to_2D_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
            LIBXSMM_DATATYPE_BF16,LIBXSMM_DATATYPE_F32, l_vec_reg_acc_start, l_m_blocking, i_n_blocking, i_m_blocking % i_micro_kernel_config->vector_length );
    } else if (i_micro_kernel_config->fused_hcolbias == 1) {
      libxsmm_generator_gemm_load_colbias_to_2D_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
            LIBXSMM_DATATYPE_F16, (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype)) ? LIBXSMM_DATATYPE_F16 : LIBXSMM_DATATYPE_F32, l_vec_reg_acc_start, l_m_blocking, i_n_blocking, i_m_blocking % i_micro_kernel_config->vector_length );
    } else if (i_micro_kernel_config->fused_b8colbias == 1) {
      libxsmm_generator_gemm_load_colbias_to_2D_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
            LIBXSMM_DATATYPE_BF8, LIBXSMM_DATATYPE_F32, l_vec_reg_acc_start, l_m_blocking, i_n_blocking, i_m_blocking % i_micro_kernel_config->vector_length );
    } else if (i_micro_kernel_config->fused_h8colbias == 1) {
      libxsmm_generator_gemm_load_colbias_to_2D_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
            LIBXSMM_DATATYPE_HF8, LIBXSMM_DATATYPE_F32, l_vec_reg_acc_start, l_m_blocking, i_n_blocking, i_m_blocking % i_micro_kernel_config->vector_length );
    } else {
      /* overwriting C, so let's xout accumulator */
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
          if ( io_generated_code->arch >= LIBXSMM_X86_AVX ) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                i_micro_kernel_config->vxor_instruction,
                i_micro_kernel_config->vector_name,
                l_vec_reg_acc_start + l_m + (l_m_blocking * l_n),
                l_vec_reg_acc_start + l_m + (l_m_blocking * l_n),
                l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
          } else {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                i_micro_kernel_config->vxor_instruction,
                i_micro_kernel_config->vector_name,
                l_vec_reg_acc_start + l_m + (l_m_blocking * l_n),
                l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
          }
        }
#if 0
        if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_CL2 ||
            i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2CL2BL2_VIA_C )  {
          for (l_m = 0; l_m < l_m_blocking; l_m += l_m++ ) {
            libxsmm_x86_instruction_prefetch( io_generated_code,
                i_micro_kernel_config->prefetch_instruction,
                i_gp_reg_mapping->gp_reg_c_prefetch,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out));
          }
        }
#endif
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_store_C( libxsmm_generated_code*             io_generated_code,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    const unsigned int                 i_m_blocking,
    const unsigned int                 i_n_blocking )
{
  /* deriving register blocking from kernel config */
  unsigned int l_is_Amxfp4_Bbf16_gemm = libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bfp32_gemm = libxsmm_x86_is_Amxfp4_Bfp32_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bi8_gemm = libxsmm_x86_is_Amxfp4_Bi8_gemm(i_xgemm_desc);
  const unsigned int l_m_blocking = ( i_m_blocking % i_micro_kernel_config->vector_length == 0
    ? (i_m_blocking/i_micro_kernel_config->vector_length) : (i_m_blocking/i_micro_kernel_config->vector_length + 1));
  /* start register of accumulator */
  const unsigned int l_vec_reg_acc_start = i_micro_kernel_config->vector_reg_count - (i_n_blocking * l_m_blocking);
  /* select store instruction */
  unsigned int l_vstore = (LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT == (LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT & i_xgemm_desc->flags))
    ? i_micro_kernel_config->c_vmove_nts_instruction : i_micro_kernel_config->c_vmove_instruction;
  /* register blocking counter in n- and m-direction */
  unsigned int l_n = 0, l_m = 0;

  libxsmm_micro_kernel_config l_micro_kernel_config_mod;
  libxsmm_micro_kernel_config *const i_micro_kernel_config_mod = (libxsmm_micro_kernel_config*)&l_micro_kernel_config_mod;
  memcpy(i_micro_kernel_config_mod, i_micro_kernel_config, sizeof(libxsmm_micro_kernel_config));

  if ( ( (  i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX )                                                                               ||
         ( (i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX2) && (i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_VL256_SKX) )     ||
         ( (i_micro_kernel_config->instruction_set == LIBXSMM_X86_AVX512_SKX) || (i_micro_kernel_config->instruction_set == LIBXSMM_X86_AVX512_CLX) ||
           (i_micro_kernel_config->instruction_set == LIBXSMM_X86_AVX512_VL256_CLX) || (i_micro_kernel_config->instruction_set == LIBXSMM_X86_AVX512_VL256_SKX) )   ) &&
       ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ) ) {
    const unsigned int relu_bitmask_gpr = i_gp_reg_mapping->gp_reg_help_2;
    const unsigned int scratch_gpr = i_gp_reg_mapping->gp_reg_help_2;
    const unsigned int aux_gpr = i_gp_reg_mapping->gp_reg_help_1;
    const unsigned int aux_vreg  = 1;
    const unsigned int zero_vreg = 0;
    const unsigned int sse_scratch_gpr = i_gp_reg_mapping->gp_reg_help_0;

    /* Check out if fusion has to be applied */
    if ((i_micro_kernel_config->fused_relu_nobitmask == 1) || (i_micro_kernel_config->fused_relu == 1)) {
      libxsmm_generator_gemm_prepare_relu_fusion( io_generated_code, i_micro_kernel_config,
          zero_vreg, i_micro_kernel_config->fused_relu, relu_bitmask_gpr, LIBXSMM_X86_GP_REG_UNDEF, aux_gpr);
    } else if (i_micro_kernel_config->fused_sigmoid == 1) {
      libxsmm_generator_gemm_dump_2D_block_and_prepare_sigmoid_fusion( io_generated_code, i_micro_kernel_config_mod,
          l_vec_reg_acc_start, l_m_blocking, i_n_blocking, scratch_gpr, aux_gpr);
    }

    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
        if ((i_micro_kernel_config->fused_relu_nobitmask == 1) || (i_micro_kernel_config->fused_relu == 1)) {
          unsigned int use_masked_compare = ( (i_micro_kernel_config->use_masking_a_c != 0) && ( l_m == (l_m_blocking - 1) ) ) ? i_micro_kernel_config->use_masking_a_c : 0;
          unsigned int bitmask_offset = (i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX) ? ((l_n * i_xgemm_desc->ldcp)/8 + (l_m * 8 + 7)/8) : ((l_n * i_xgemm_desc->ldcp)/8 + (l_m * 16+7)/8);
          unsigned int l_sse_mask_pos = ((l_m * 4) + i_micro_kernel_config->current_m)%8;
          bitmask_offset = ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX ) ? ((l_n * i_xgemm_desc->ldcp)/8 + ((l_m/2) * 8 + 7)/8) : bitmask_offset;
          libxsmm_generator_gemm_apply_relu_to_vreg( io_generated_code, i_micro_kernel_config,
              zero_vreg, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), i_micro_kernel_config->fused_relu, relu_bitmask_gpr, bitmask_offset, 1, sse_scratch_gpr, aux_gpr, aux_vreg, use_masked_compare, l_sse_mask_pos );
        } else if (i_micro_kernel_config->fused_sigmoid == 1) {
          unsigned int tmp_vreg = 0;
          libxsmm_generator_gemm_apply_sigmoid_to_vreg_from_scratch( io_generated_code, i_micro_kernel_config_mod,
              scratch_gpr, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), tmp_vreg );
          /* Store vreg back to scratch */
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX ) ? LIBXSMM_X86_INSTR_MOVUPS : LIBXSMM_X86_INSTR_VMOVUPS,
              scratch_gpr,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (l_vec_reg_acc_start + l_m + (l_m_blocking * l_n)) * 64,
              i_micro_kernel_config->vector_name,
              tmp_vreg, 0, 0, 1 );
        }
      }
    }

    if ((i_micro_kernel_config->fused_relu_nobitmask == 1) || (i_micro_kernel_config->fused_relu == 1)) {
      libxsmm_generator_gemm_cleanup_relu_fusion( io_generated_code, i_micro_kernel_config->fused_relu, relu_bitmask_gpr, LIBXSMM_X86_GP_REG_UNDEF, aux_gpr);
    } else if (i_micro_kernel_config->fused_sigmoid == 1) {
      /* Restore accumulators from scratch */
      libxsmm_generator_gemm_restore_2D_regblock_from_scratch( io_generated_code, i_micro_kernel_config,
          scratch_gpr, l_vec_reg_acc_start, l_m_blocking, i_n_blocking);
      libxsmm_generator_gemm_cleanup_sigmoid_fusion( io_generated_code, scratch_gpr, aux_gpr );
    }

    if ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX ) {
      libxsmm_generator_vcvtneps2bf16_sse_prep_stack( io_generated_code, 0 );

      /* storing downconverted and rounded C accumulator */
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
          unsigned int reg_X = l_vec_reg_acc_start + l_m + (l_m_blocking * l_n);

          libxsmm_generator_vcvtneps2bf16_sse_preppedstack( io_generated_code, 'x', reg_X, 1, 0, 2, 0 );

          /* store 16 bit values into xmm portion of the register */
          if ( (i_micro_kernel_config->use_masking_a_c != 0) && ( l_m == (l_m_blocking - 1) ) ) {
            libxsmm_generator_maskedstore_16bit_sse( io_generated_code, i_gp_reg_mapping->gp_reg_help_0, 0, 1,
                                                      i_gp_reg_mapping->gp_reg_c, LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                      ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                                                      i_m_blocking % i_micro_kernel_config->vector_length );
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_MOVSD,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                'x', 1, 0, 0, 1 );
          }
        }
      }
      libxsmm_generator_vcvtneps2bf16_sse_clean_stack( io_generated_code );
    } else if ( (i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX2) && (i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_VL256_SKX) ) {
      if ( i_micro_kernel_config->instruction_set != LIBXSMM_X86_AVX2_SRF ) {
        libxsmm_generator_vcvtneps2bf16_avx2_prep_stack( io_generated_code, 0 );
      }

      /* storing downconverted and rounded C accumulator */
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
          unsigned int reg_X = l_vec_reg_acc_start + l_m + (l_m_blocking * l_n);

          if ( i_micro_kernel_config->instruction_set != LIBXSMM_X86_AVX2_SRF ) {
            libxsmm_generator_vcvtneps2bf16_avx2_preppedstack( io_generated_code, 'y', reg_X, 0, 1, 2, 0 );
          } else {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                LIBXSMM_X86_INSTR_VCVTNEPS2BF16,
                i_micro_kernel_config->vector_name,
                reg_X, 0 );
          }

          /* store 16 bit values into xmm portion of the register */
          if ( (i_micro_kernel_config->use_masking_a_c != 0) && ( l_m == (l_m_blocking - 1) ) ) {
            libxsmm_generator_maskedstore_16bit_avx2( io_generated_code, i_gp_reg_mapping->gp_reg_help_0, 0,
                                                      i_gp_reg_mapping->gp_reg_c, LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                      ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                                                      i_m_blocking % i_micro_kernel_config->vector_length );
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                l_vstore,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                'x', 0, 0, 0, 1 );
          }
        }
      }

      if ( i_micro_kernel_config->instruction_set != LIBXSMM_X86_AVX2_SRF ) {
        libxsmm_generator_vcvtneps2bf16_avx2_clean_stack( io_generated_code );
      }
    } else {
      libxsmm_generator_vcvtneps2bf16_avx512_prep_stack( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );

      /* storing downconverted and rounded C accumulator */
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
          unsigned int reg_X = l_vec_reg_acc_start + l_m + (l_m_blocking * l_n);

          libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code,
                         ( ( i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_VL256_SKX) && (i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX) ) ? 'y' : 'z',
                         reg_X, 0, 1, 2, 6, 7, 0 );

          /* store 16 bit values into xmm portion of the register */
          if ( (i_micro_kernel_config->use_masking_a_c != 0) && ( l_m == (l_m_blocking - 1) ) ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVDQU16,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                ( ( i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_VL256_SKX) && (i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX) ) ? 'y' : 'z',
                0, 2, 0, 1 );
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                l_vstore,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                ( ( i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_VL256_SKX) && (i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX) ) ? 'x' : 'y',
                0, 0, 0, 1 );
          }
        }
      }
      libxsmm_generator_vcvtneps2bf16_avx512_clean_stack( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
    }
  } else if ( ( ((i_micro_kernel_config->instruction_set <= LIBXSMM_X86_ALLFEAT) && (i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_CPX)) ||
                ((i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX)   && (i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_VL256_CPX)) )
              &&
              ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) )
            ) {
    const unsigned int relu_bitmask_gpr = i_gp_reg_mapping->gp_reg_help_2;
    const unsigned int scratch_gpr = i_gp_reg_mapping->gp_reg_help_2;
    const unsigned int aux_gpr = i_gp_reg_mapping->gp_reg_help_1;
    const unsigned int zero_vreg = 1;
    const unsigned int aux_vreg  = 2;

    /* storing downconverted and rounded C accumulator */
    /* Check out if fusion has to be applied */
    if ((i_micro_kernel_config->fused_relu_nobitmask == 1) || (i_micro_kernel_config->fused_relu == 1)) {
      libxsmm_generator_gemm_prepare_relu_fusion( io_generated_code, i_micro_kernel_config,
          zero_vreg, i_micro_kernel_config->fused_relu, relu_bitmask_gpr, LIBXSMM_X86_GP_REG_UNDEF, aux_gpr);
    } else if (i_micro_kernel_config->fused_sigmoid == 1) {
      /* First dump the accumulators to scratch and then setup sigmoid coefficients to be reused */
      libxsmm_generator_gemm_dump_2D_block_and_prepare_sigmoid_fusion( io_generated_code, i_micro_kernel_config_mod,
          l_vec_reg_acc_start, l_m_blocking, i_n_blocking, scratch_gpr, aux_gpr);
    }

    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      unsigned int l_m_2_blocking = (l_m_blocking/2)*2;
      l_m = 0;

      if ( i_micro_kernel_config->use_masking_a_c != 0 ) {
        for ( l_m = 0 ; l_m < l_m_blocking; l_m++ ) {
          unsigned int reg_X = l_vec_reg_acc_start + l_m + (l_m_blocking * l_n);
          if ((i_micro_kernel_config->fused_relu_nobitmask == 1) || (i_micro_kernel_config->fused_relu == 1)) {
            unsigned int use_masked_compare = ( (i_micro_kernel_config->use_masking_a_c != 0) && ( l_m == (l_m_blocking - 1) ) ) ? i_micro_kernel_config->use_masking_a_c : 0;
            unsigned int bitmask_offset = (i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX) ? ((l_n * i_xgemm_desc->ldcp)/8 + (l_m * 8 + 7)/8) : ((l_n * i_xgemm_desc->ldcp)/8 + (l_m * 16+7)/8);
            libxsmm_generator_gemm_apply_relu_to_vreg( io_generated_code, i_micro_kernel_config,
              zero_vreg, reg_X, i_micro_kernel_config->fused_relu, relu_bitmask_gpr, bitmask_offset, 1, LIBXSMM_X86_GP_REG_UNDEF, aux_gpr, aux_vreg, use_masked_compare, 0 );
          } else if  (i_micro_kernel_config->fused_sigmoid == 1) {
            unsigned int tmp_vreg = 0;
            libxsmm_generator_gemm_apply_sigmoid_to_vreg_from_scratch( io_generated_code, i_micro_kernel_config_mod,
                scratch_gpr, reg_X, tmp_vreg );
            reg_X = tmp_vreg;
          }
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
              LIBXSMM_X86_INSTR_VCVTNEPS2BF16,
              i_micro_kernel_config->vector_name,
              reg_X, 0 );

          /* store 16 bit values into ymm portion of the register bfloat mask fix can lead to errors x should not be masked */
          if ( l_m == (l_m_blocking - 1) ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVDQU16,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                ( i_micro_kernel_config->instruction_set == LIBXSMM_X86_AVX512_VL256_CPX ) ? 'y' : 'z',
                0, 2, 0, 1 );
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                l_vstore,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                ( i_micro_kernel_config->instruction_set == LIBXSMM_X86_AVX512_VL256_CPX ) ? 'x' : 'y',
                0, 0, 0, 1 );
          }
        }
      } else {
        for (; l_m < l_m_2_blocking; l_m+=2 ) {
          unsigned int reg_X = l_vec_reg_acc_start + l_m + (l_m_blocking * l_n);
          unsigned int reg_X2 = l_vec_reg_acc_start + l_m+1 + (l_m_blocking * l_n);

          if (i_micro_kernel_config->fused_sigmoid == 1) {
            unsigned int tmp_vreg = 0;
            unsigned int tmp_vreg2 = 1;
            libxsmm_generator_gemm_apply_sigmoid_to_vreg_from_scratch( io_generated_code, i_micro_kernel_config_mod,
                scratch_gpr, reg_X, tmp_vreg );
            libxsmm_generator_gemm_apply_sigmoid_to_vreg_from_scratch( io_generated_code, i_micro_kernel_config_mod,
                scratch_gpr, reg_X2, tmp_vreg2 );
            reg_X  = tmp_vreg;
            reg_X2 = tmp_vreg2;
          }

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
              LIBXSMM_X86_INSTR_VCVTNE2PS2BF16,
              i_micro_kernel_config->vector_name,
              reg_X, reg_X2, 0 );

          if ((i_micro_kernel_config->fused_relu_nobitmask == 1) || (i_micro_kernel_config->fused_relu == 1)) {
            unsigned int use_masked_compare = ( (i_micro_kernel_config->use_masking_a_c != 0) && ( l_m == (l_m_blocking - 1) ) ) ? i_micro_kernel_config->use_masking_a_c : 0;
            unsigned int bitmask_offset = (i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX) ? ((l_n * i_xgemm_desc->ldcp)/8 + (l_m * 8 + 7)/8) : ((l_n * i_xgemm_desc->ldcp)/8 + (l_m * 16+7)/8);
            libxsmm_generator_gemm_apply_relu_to_vreg( io_generated_code, i_micro_kernel_config,
              zero_vreg, 0, i_micro_kernel_config->fused_relu, relu_bitmask_gpr, bitmask_offset, 0, LIBXSMM_X86_GP_REG_UNDEF, aux_gpr, aux_vreg, use_masked_compare, 0 );
          }

          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              l_vstore,
              i_gp_reg_mapping->gp_reg_c,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
              ( i_micro_kernel_config->instruction_set == LIBXSMM_X86_AVX512_VL256_CPX ) ? 'y' : 'z',
              0, 0, 0, 1 );
        }
        for (; l_m < l_m_blocking; l_m++ ) {
          unsigned int reg_X = l_vec_reg_acc_start + l_m + (l_m_blocking * l_n);

          if ((i_micro_kernel_config->fused_relu_nobitmask == 1) || (i_micro_kernel_config->fused_relu == 1)) {
            unsigned int use_masked_compare = ( (i_micro_kernel_config->use_masking_a_c != 0) && ( l_m == (l_m_blocking - 1) ) ) ? i_micro_kernel_config->use_masking_a_c : 0;
            unsigned int bitmask_offset = (i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX) ? ((l_n * i_xgemm_desc->ldcp)/8 + (l_m * 8 + 7)/8) : ((l_n * i_xgemm_desc->ldcp)/8 + (l_m * 16+7)/8);
            libxsmm_generator_gemm_apply_relu_to_vreg( io_generated_code, i_micro_kernel_config,
              zero_vreg, reg_X, i_micro_kernel_config->fused_relu, relu_bitmask_gpr, bitmask_offset, 1, LIBXSMM_X86_GP_REG_UNDEF, aux_gpr, aux_vreg, use_masked_compare, 0);
          } else if (i_micro_kernel_config->fused_sigmoid == 1) {
            unsigned int tmp_vreg = 0;
            libxsmm_generator_gemm_apply_sigmoid_to_vreg_from_scratch( io_generated_code, i_micro_kernel_config_mod,
                scratch_gpr, reg_X, tmp_vreg );
            reg_X = tmp_vreg;
          }

          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
              LIBXSMM_X86_INSTR_VCVTNEPS2BF16,
              i_micro_kernel_config->vector_name,
              reg_X, 0 );

          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              l_vstore,
              i_gp_reg_mapping->gp_reg_c,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
              ( i_micro_kernel_config->instruction_set == LIBXSMM_X86_AVX512_VL256_CPX ) ? 'x' : 'y',
              0, 0, 0, 1 );
        }
      }
    }

    if ((i_micro_kernel_config->fused_relu_nobitmask == 1) || (i_micro_kernel_config->fused_relu == 1)) {
      libxsmm_generator_gemm_cleanup_relu_fusion( io_generated_code, i_micro_kernel_config->fused_relu, relu_bitmask_gpr, LIBXSMM_X86_GP_REG_UNDEF, aux_gpr);
    } else if (i_micro_kernel_config->fused_sigmoid == 1) {
      libxsmm_generator_gemm_cleanup_sigmoid_fusion( io_generated_code, scratch_gpr, aux_gpr );
    }
  } else if (( ( (i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_VL256_SKX) && (i_micro_kernel_config->instruction_set <= LIBXSMM_X86_ALLFEAT) ) &&
              (  ( (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) && (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ) || ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) && (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ) || ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) && (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ) ) ) ||

( ( (i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_VL256_SKX) && (i_micro_kernel_config->instruction_set <= LIBXSMM_X86_ALLFEAT) ) &&
              (  ( (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) && (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ) || ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) && (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ) || ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) && (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ) ) )) {
    const unsigned int relu_bitmask_gpr = i_gp_reg_mapping->gp_reg_help_2;
    const unsigned int scratch_gpr = i_gp_reg_mapping->gp_reg_help_2;
    const unsigned int aux_gpr = i_gp_reg_mapping->gp_reg_help_1;
    const unsigned int aux_vreg  = 1;
    const unsigned int zero_vreg = 0;

    /* Check out if fusion has to be applied */
    if ((i_micro_kernel_config->fused_relu_nobitmask == 1) || (i_micro_kernel_config->fused_relu == 1)) {
      libxsmm_generator_gemm_prepare_relu_fusion( io_generated_code, i_micro_kernel_config,
          zero_vreg, i_micro_kernel_config->fused_relu, relu_bitmask_gpr, LIBXSMM_X86_GP_REG_UNDEF, aux_gpr);
    } else if (i_micro_kernel_config->fused_sigmoid == 1) {
      libxsmm_generator_gemm_dump_2D_block_and_prepare_sigmoid_fusion( io_generated_code, i_micro_kernel_config_mod,
          l_vec_reg_acc_start, l_m_blocking, i_n_blocking, scratch_gpr, aux_gpr);
    }

    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
        if ((i_micro_kernel_config->fused_relu_nobitmask == 1) || (i_micro_kernel_config->fused_relu == 1)) {
          unsigned int use_masked_compare = ( (i_micro_kernel_config->use_masking_a_c != 0) && ( l_m == (l_m_blocking - 1) ) ) ? i_micro_kernel_config->use_masking_a_c : 0;
          unsigned int bitmask_offset = (i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX) ? ((l_n * i_xgemm_desc->ldcp)/8 + (l_m * 8 + 7)/8) : ((l_n * i_xgemm_desc->ldcp)/8 + (l_m * 16+7)/8);
          libxsmm_generator_gemm_apply_relu_to_vreg( io_generated_code, i_micro_kernel_config,
              zero_vreg, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), i_micro_kernel_config->fused_relu, relu_bitmask_gpr, bitmask_offset, 1, LIBXSMM_X86_GP_REG_UNDEF, aux_gpr, aux_vreg, use_masked_compare, 0 );
        } else if (i_micro_kernel_config->fused_sigmoid == 1) {
          unsigned int tmp_vreg = 0;
          libxsmm_generator_gemm_apply_sigmoid_to_vreg_from_scratch( io_generated_code, i_micro_kernel_config_mod,
              scratch_gpr, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), tmp_vreg );
          /* Store vreg back to scratch */
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVUPS,
              scratch_gpr,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (l_vec_reg_acc_start + l_m + (l_m_blocking * l_n)) * 64,
              i_micro_kernel_config->vector_name,
              tmp_vreg, 0, 0, 1 );
        }
      }
    }

    if ((i_micro_kernel_config->fused_relu_nobitmask == 1) || (i_micro_kernel_config->fused_relu == 1)) {
      libxsmm_generator_gemm_cleanup_relu_fusion( io_generated_code, i_micro_kernel_config->fused_relu, relu_bitmask_gpr, LIBXSMM_X86_GP_REG_UNDEF, aux_gpr);
    } else if (i_micro_kernel_config->fused_sigmoid == 1) {
      /* Restore accumulators from scratch */
      libxsmm_generator_gemm_restore_2D_regblock_from_scratch( io_generated_code, i_micro_kernel_config,
          scratch_gpr, l_vec_reg_acc_start, l_m_blocking, i_n_blocking);
      libxsmm_generator_gemm_cleanup_sigmoid_fusion( io_generated_code, scratch_gpr, aux_gpr );
    }

    if (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
      libxsmm_generator_vcvtneps2bf8_avx512_prep_stack( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
    } else if (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
      libxsmm_generator_vcvt_hf8_tofrom_f32_avx512_prep_stack( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
    }

    /* storing downconverted and rounded C accumulator */
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
        unsigned int reg_X = l_vec_reg_acc_start + l_m + (l_m_blocking * l_n);

        if (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
          libxsmm_generator_vcvtneps2bf8_avx512_preppedstack( io_generated_code,
             ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ) ? 'y' : 'z',
             reg_X, 0, 2, 3, 6, 7, 0, 0 );
        } else if (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
          libxsmm_generator_vcvtf32_to_hf8_avx512_preppedstack( io_generated_code,
              ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ) ? 'y' : 'z',
              reg_X, 0, 0, 1, 2, 3, 7, 6, 5);
        }

        /* store 8 bit values into xmm portion of the register */
        if ( (i_micro_kernel_config->use_masking_a_c != 0) && ( l_m == (l_m_blocking - 1) ) ) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              io_generated_code->arch,
              LIBXSMM_X86_INSTR_VMOVDQU8,
              i_gp_reg_mapping->gp_reg_c,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
              ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ) ? 'y' : 'z',
              0, 2, 0, 1 );
        } else {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              io_generated_code->arch,
              ( io_generated_code->arch < LIBXSMM_X86_AVX512_SKX ) ? LIBXSMM_X86_INSTR_VMOVSD : l_vstore,
              i_gp_reg_mapping->gp_reg_c,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
              'x',
              0, 0, 0, 1 );
        }
      }
    }

    if (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
      libxsmm_generator_vcvtneps2bf8_avx512_clean_stack( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
    } else if (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
      libxsmm_generator_vcvt_hf8_tofrom_f32_avx512_clean_stack( io_generated_code,  i_gp_reg_mapping->gp_reg_help_2);
    }
  } else if ( ( (i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX2) && (i_micro_kernel_config->instruction_set <= LIBXSMM_X86_ALLFEAT) ) &&
              ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) && (l_is_Amxfp4_Bi8_gemm == 0)) ) {
    const unsigned int aux_vreg = 1;
    const unsigned int mask_gpr = i_gp_reg_mapping->gp_reg_help_0;

    /* in case of AVX/AVX2 we need to load the mask into an ymm */
    if ( (i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX) && (i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_VL256_SKX) &&
         (i_micro_kernel_config->use_masking_a_c != 0) ) {
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_AVX2_MASK_PTR, mask_gpr );
      libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_VMOVUPS,
                                        mask_gpr, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 'y', aux_vreg, 0, 0, 0 );
    }

    /* loading scf into register 3 */
    libxsmm_x86_instruction_vec_move( io_generated_code,
        i_micro_kernel_config->instruction_set,
        LIBXSMM_X86_INSTR_VBROADCASTSS,
        i_gp_reg_mapping->gp_reg_scf,
        LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
        i_micro_kernel_config->vector_name,
        3, 0, 1, 0 );

    /* storing downconverted and rounded C accumulator */
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
        unsigned int reg_X = l_vec_reg_acc_start + l_m + (l_m_blocking * l_n);
        /* Convert result to F32 */
        libxsmm_x86_instruction_vec_compute_2reg(  io_generated_code,
            LIBXSMM_X86_INSTR_VCVTDQ2PS,
            i_micro_kernel_config->vector_name,
            reg_X,
            reg_X );

        /* Multiply with scaling factor */
        libxsmm_x86_instruction_vec_compute_3reg(  io_generated_code,
            LIBXSMM_X86_INSTR_VMULPS,
            i_micro_kernel_config->vector_name,
            reg_X,
            3,
            reg_X );

        if (0 == (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code, l_vstore,
                                                    i_gp_reg_mapping->gp_reg_c, LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                    ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                                                    i_micro_kernel_config->vector_name, 2, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, aux_vreg, 0);

          libxsmm_x86_instruction_vec_compute_3reg(  io_generated_code,
              LIBXSMM_X86_INSTR_VADDPS,
              i_micro_kernel_config->vector_name,
              reg_X,
              2,
              reg_X );
        }

        libxsmm_x86_instruction_unified_vec_move( io_generated_code, l_vstore,
                                                  i_gp_reg_mapping->gp_reg_c, LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                  ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                                                  i_micro_kernel_config->vector_name, reg_X, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, aux_vreg, 1);
      }
    }
  } else if ( (i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX) &&
              ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ) ) {
    unsigned int l_mask_reg_or_val = i_m_blocking%i_micro_kernel_config->vector_length;

    /* loading scf into register 3 */
    libxsmm_x86_instruction_vec_move( io_generated_code,
        i_micro_kernel_config->instruction_set,
        LIBXSMM_X86_INSTR_MOVSS,
        i_gp_reg_mapping->gp_reg_scf,
        LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
        i_micro_kernel_config->vector_name,
        3, 0, 1, 0 );
    libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_SHUFPS,
                                                   i_micro_kernel_config->vector_name, 3, 3, 0 );

    /* storing downconverted and rounded C accumulator */
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
        unsigned int reg_X = l_vec_reg_acc_start + l_m + (l_m_blocking * l_n);
        /* Convert result to F32 */
        libxsmm_x86_instruction_vec_compute_2reg(  io_generated_code,
            LIBXSMM_X86_INSTR_CVTDQ2PS,
            i_micro_kernel_config->vector_name,
            reg_X,
            reg_X );

        /* Multiply with scaling factor */
        libxsmm_x86_instruction_vec_compute_2reg(  io_generated_code,
            LIBXSMM_X86_INSTR_MULPS,
            i_micro_kernel_config->vector_name,
            3,
            reg_X );

        if (0 == (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code, l_vstore,
                                                    i_gp_reg_mapping->gp_reg_c, LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                    ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                                                    i_micro_kernel_config->vector_name, 2, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, l_mask_reg_or_val, 0);

          libxsmm_x86_instruction_vec_compute_2reg(  io_generated_code,
              LIBXSMM_X86_INSTR_ADDPS,
              i_micro_kernel_config->vector_name,
              2,
              reg_X );
        }

        libxsmm_x86_instruction_unified_vec_move( io_generated_code, l_vstore,
                                                  i_gp_reg_mapping->gp_reg_c, LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                  ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                                                  i_micro_kernel_config->vector_name, reg_X, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, l_mask_reg_or_val, 1);
      }
    }
  } else {
    /* storing C accumulator */
    const unsigned int relu_bitmask_gpr = i_gp_reg_mapping->gp_reg_help_2;
    const unsigned int scratch_gpr = i_gp_reg_mapping->gp_reg_help_2;
    const unsigned int aux_gpr = i_gp_reg_mapping->gp_reg_help_1;
    const unsigned int zero_vreg = 0;
    const unsigned int aux_vreg  = (l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0) ? 2 : 1;
    const unsigned int mask_gpr = i_gp_reg_mapping->gp_reg_help_0;
    const unsigned int sse_scratch_gpr = i_gp_reg_mapping->gp_reg_help_0;

    /* in case of AVX/AVX2 we need to load the mask into an ymm */
    if ( (i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX) && (i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_VL256_SKX) &&
         (i_micro_kernel_config->use_masking_a_c != 0) ) {
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_AVX2_MASK_PTR, mask_gpr );
    }

    /* Check out if fusion has to be applied */
    if ((i_micro_kernel_config->fused_relu_nobitmask == 1) || (i_micro_kernel_config->fused_relu == 1)) {
      libxsmm_generator_gemm_prepare_relu_fusion( io_generated_code, i_micro_kernel_config,
          zero_vreg, i_micro_kernel_config->fused_relu, relu_bitmask_gpr, sse_scratch_gpr, aux_gpr);
    } else if (i_micro_kernel_config->fused_sigmoid == 1) {
      libxsmm_generator_gemm_dump_2D_block_and_prepare_sigmoid_fusion( io_generated_code, i_micro_kernel_config_mod,
          l_vec_reg_acc_start, l_m_blocking, i_n_blocking, scratch_gpr, aux_gpr);
    }

    /* In this case we have to split store C in 2 chunks of vlen/2 each  */
    if (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype) &&
        LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype) &&
        io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) {
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
          unsigned int reg_X = l_vec_reg_acc_start + l_m + (l_m_blocking * l_n);
          unsigned int l_tmp_vreg = l_vec_reg_acc_start - 1;
          char vname_cvt = (i_micro_kernel_config->vector_name == 'y') ? 'z' : ((i_micro_kernel_config->vector_name == 'x') ? 'y' : i_micro_kernel_config->vector_name);

          /* Exctract */
          libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VEXTRACTI64X4, vname_cvt, reg_X, l_tmp_vreg, 0x1 );

          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, vname_cvt, reg_X, reg_X );

          /* Store  */
          libxsmm_x86_instruction_unified_vec_move( io_generated_code, l_vstore,
                                                    i_gp_reg_mapping->gp_reg_c, LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                    ((l_n * i_xgemm_desc->ldc) + (2*l_m * (i_micro_kernel_config->vector_length/2))) * (i_micro_kernel_config->datatype_size_out),
                                                    i_micro_kernel_config->vector_name, reg_X, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 2, 1);

          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, vname_cvt, l_tmp_vreg, l_tmp_vreg );

          /* Store  */
          libxsmm_x86_instruction_unified_vec_move( io_generated_code, l_vstore,
                                                    i_gp_reg_mapping->gp_reg_c, LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                    ((l_n * i_xgemm_desc->ldc) + ((2*l_m+1) * (i_micro_kernel_config->vector_length/2))) * (i_micro_kernel_config->datatype_size_out),
                                                    i_micro_kernel_config->vector_name, l_tmp_vreg, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 3, 1);
        }
      }
    } else {
      unsigned int l_bf16cvt_replacement = 0;
      if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype)) {
        if ((i_micro_kernel_config->vmul_instruction != LIBXSMM_X86_INSTR_VDPBF16PS) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
          l_bf16cvt_replacement = 1;
          libxsmm_generator_vcvtneps2bf16_avx512_prep_stack( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
        }
      }
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
          unsigned int reg_X = l_vec_reg_acc_start + l_m + (l_m_blocking * l_n);
          unsigned int l_mask_reg_or_val = ( i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX ) ? (l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0 ) ? 2 : 1 : i_m_blocking%i_micro_kernel_config->vector_length;
          char vname_store = i_micro_kernel_config->vector_name;

          if ((i_micro_kernel_config->fused_relu_nobitmask == 1) || (i_micro_kernel_config->fused_relu == 1)) {
            unsigned int use_masked_compare = ( (i_micro_kernel_config->use_masking_a_c != 0) && ( l_m == (l_m_blocking - 1) ) ) ? i_micro_kernel_config->use_masking_a_c : 0;
            unsigned int bitmask_offset = (i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_SKX) ? ((l_n * i_xgemm_desc->ldcp)/8 + (l_m * 8 + 7)/8) : ((l_n * i_xgemm_desc->ldcp)/8 + (l_m * 16+7)/8);
            unsigned int l_sse_mask_pos = ((l_m * 4) + i_micro_kernel_config->current_m)%8;
            bitmask_offset = ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX ) ? ((l_n * i_xgemm_desc->ldcp)/8 + ((l_m/2) * 8 + 7)/8) : bitmask_offset;
            libxsmm_generator_gemm_apply_relu_to_vreg( io_generated_code, i_micro_kernel_config,
                zero_vreg, reg_X, i_micro_kernel_config->fused_relu, relu_bitmask_gpr, bitmask_offset, 1, sse_scratch_gpr, aux_gpr, aux_vreg, use_masked_compare, l_sse_mask_pos);
          } else if (i_micro_kernel_config->fused_sigmoid == 1) {
            unsigned int tmp_vreg  = 0;
            libxsmm_generator_gemm_apply_sigmoid_to_vreg_from_scratch( io_generated_code, i_micro_kernel_config_mod,
                scratch_gpr, reg_X, tmp_vreg );
            reg_X = tmp_vreg;
          }

          if ( (i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX) && (i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_VL256_SKX) &&
               (i_micro_kernel_config->use_masking_a_c != 0) && (l_m == (l_m_blocking - 1)) ) {
            libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_VMOVUPS,
                                              mask_gpr, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 'y', aux_vreg, 0, 0, 0 );
          }

          if ((LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype) && LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype)) ||
              (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype) && LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype) && io_generated_code->arch < LIBXSMM_X86_AVX512_SPR)) {
            vname_store = (vname_store == 'z') ? 'y' : 'x';
            if (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype) && LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype)) {
              libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, reg_X, reg_X, 0,
                  (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
            }
          }

          if (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype) && LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype)) {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, reg_X, reg_X );
          }


          if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype) || LIBXSMM_DATATYPE_I32 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype))) {
            vname_store = (vname_store == 'z') ? 'y' : 'x';
            if (l_bf16cvt_replacement == 0) {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, reg_X, reg_X );
            } else {
              libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name, reg_X, reg_X, l_vec_reg_acc_start-2, l_vec_reg_acc_start-1, 6, 7, 0 );
            }
          }

          if ((libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc) > 0 || libxsmm_x86_is_Amxfp4_Bi8_gemm(i_xgemm_desc) > 0) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype)) &&
              (i_micro_kernel_config->use_masking_a_c != 0) && (l_m == (l_m_blocking - 1)) ) {
            libxsmm_generator_maskedstore_16bit_avx2( io_generated_code, i_gp_reg_mapping->gp_reg_help_2, reg_X,
                                                      i_gp_reg_mapping->gp_reg_c, LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                      ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                                                      i_m_blocking % i_micro_kernel_config->vector_length );
          } else {
            libxsmm_x86_instruction_unified_vec_move( io_generated_code, l_vstore,
                                                      i_gp_reg_mapping->gp_reg_c, LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                      ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                                                      vname_store, reg_X, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, l_mask_reg_or_val, 1);
          }
        }

        if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_BL2_VIA_C          ||
             i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C       ||
             i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD    )  {
          if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0 ) {
            /* determining how many prefetches we need in M direction as we just need one prefetch per cache line */
            unsigned int l_m_advance = 64 / ((i_micro_kernel_config->vector_length) * (i_micro_kernel_config->datatype_size_out)); /* 64: hardcoded cache line length */

            for (l_m = 0; l_m < l_m_blocking; l_m += l_m_advance ) {
              libxsmm_x86_instruction_prefetch( io_generated_code,
                  i_micro_kernel_config->prefetch_instruction,
                  i_gp_reg_mapping->gp_reg_b_prefetch,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out));
            }
          }
        }
      }
      if (l_bf16cvt_replacement > 0) {
        libxsmm_generator_vcvtneps2bf16_avx512_clean_stack( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
      }
    }

    if ((i_micro_kernel_config->fused_relu_nobitmask == 1) || (i_micro_kernel_config->fused_relu == 1)) {
      libxsmm_generator_gemm_cleanup_relu_fusion( io_generated_code, i_micro_kernel_config->fused_relu, relu_bitmask_gpr, sse_scratch_gpr, aux_gpr );
    } else if (i_micro_kernel_config->fused_sigmoid == 1) {
      libxsmm_generator_gemm_cleanup_sigmoid_fusion( io_generated_code, scratch_gpr, aux_gpr );
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_generator_gemm_get_blocking_and_mask( unsigned int i_range, unsigned int i_max_block, unsigned int i_nomask_block, unsigned int *io_block, unsigned int *o_use_mask ) {
  /* TODO: check if there is a better blocking strategy */
  if ( *io_block == i_max_block ) {
    *io_block = i_range % i_max_block;
    if ( *io_block % i_nomask_block != 0 ) {
      *o_use_mask = 1;
    }
  } else if ( *io_block == 0 ) {
    if ( i_range >= i_max_block ) {
      *io_block = i_max_block;
    } else {
      *io_block = i_range;
      /* in case we do not have a full vector length, we use masking */
      if ( (*io_block) % i_nomask_block != 0 ) {
        *o_use_mask = 1;
      }
    }
  } else {
#if 0
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
    return 0;
#endif
  }
}
