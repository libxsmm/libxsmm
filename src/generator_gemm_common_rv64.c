/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved                       *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Siddharth Rai, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_gemm_common.h"
#include "generator_rv64_instructions.h"
#include "generator_gemm_common_rv64.h"
#include "generator_common.h"
#include "generator_mateltwise_unary_binary_rv64.h"
#include "generator_common_rv64.h"
#include "generator_mateltwise_rv64.h"

LIBXSMM_API_INTERN void libxsmm_generator_gemm_apply_ops_input_tensor_and_store_to_stack_rv64( libxsmm_generated_code*    io_generated_code,
                                                                                      libxsmm_loop_label_tracker*    io_loop_label_tracker,
                                                                                      libxsmm_micro_kernel_config*   i_micro_kernel_config,
                                                                                      libxsmm_gemm_descriptor*       i_xgemm_desc,
                                                                                      unsigned int                   i_gp_reg_in,
                                                                                      unsigned int                   i_struct_gp_reg,
                                                                                      unsigned int                   i_tmp_reg,
                                                                                      unsigned int                   i_loop_reg,
                                                                                      unsigned int                   i_bound_reg,
                                                                                      unsigned int                   i_tmp_reg2,
                                                                                      unsigned int                   i_tmp_reg3,
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
  int l_offset_ptr_a = (int)sizeof(libxsmm_matrix_op_arg);
  int l_offset_ptr_b = (int)(sizeof(libxsmm_matrix_op_arg) + sizeof(libxsmm_matrix_arg));

  LIBXSMM_UNUSED(i_micro_kernel_config);

  l_mateltwise_gp_reg_mapping.gp_reg_param_struct = i_struct_gp_reg;
  libxsmm_generator_gemm_getval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_MELTW_STRUCT_PTR, i_struct_gp_reg);

  /* Loop over all batch-reduce iterations to cover all tensor blocks */
  if (is_brgemm > 0) {
    libxsmm_generator_gemm_getval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, i_bound_reg);
    libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, i_loop_reg, 0 );
    libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, i_tmp_reg3, 0 );
    libxsmm_rv64_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker);
  }
  /* Setup input pointer of input in eltwise struct */
  if (is_offset_brgemm > 0) {
    libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, i_gp_reg_in, i_tmp_reg2, 0 );
    libxsmm_generator_gemm_getval_stack_var_rv64( io_generated_code, i_stack_var_offs_ptr, i_tmp_reg);
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD, i_tmp_reg, i_loop_reg, 0);
    libxsmm_rv64_instruction_alu_compute( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, i_tmp_reg, i_gp_reg_in, i_gp_reg_in );
  }
  if (is_address_brgemm > 0) {
    libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, i_gp_reg_in, i_tmp_reg2, 0 );
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD, i_gp_reg_in, i_loop_reg, 0 );
  }

  libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_SD, i_struct_gp_reg, LIBXSMM_RV64_GP_REG_UNDEF, l_offset_ptr_a, i_gp_reg_in );

  if ((is_offset_brgemm > 0) || (is_address_brgemm > 0)) {
    libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, i_tmp_reg2, i_gp_reg_in, 0 );
  }

  if (i_op2_type != LIBXSMM_MELTW_TYPE_UNARY_NONE) {
    libxsmm_generator_gemm_getval_stack_var_rv64( io_generated_code, i_stack_var_scratch_ptr, i_tmp_reg);
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_STR_I_OFF, i_struct_gp_reg, LIBXSMM_RV64_GP_REG_UNDEF, l_offset_ptr_b, i_tmp_reg );
  } else {
    libxsmm_generator_gemm_getval_stack_var_rv64( io_generated_code, i_stack_var_dst_ptr, i_tmp_reg);
    if (is_brgemm > 0) {
      libxsmm_rv64_instruction_alu_compute( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, i_tmp_reg3, i_tmp_reg, i_tmp_reg );
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD, i_tmp_reg3, i_tmp_reg2, i_tmp_reg3, (long long)LIBXSMM_TYPESIZE(i_out_dtype) * i_m * i_n );
    }
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_SD, i_struct_gp_reg, i_tmp_reg, l_offset_ptr_b );
  }

  l_mateltwise_desc = libxsmm_meltw_descriptor_init2(&l_meltw_blob, i_in_dtype, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED,
    i_comp_dtype, i_out_dtype, i_m, i_n, i_ldi, i_ldo, 0, 0,
    0, LIBXSMM_CAST_USHORT(i_op_type), LIBXSMM_MELTW_OPERATION_UNARY);
  libxsmm_generator_mateltwise_rv64_init_micro_kernel_config_fullvector( io_generated_code, &l_mateltwise_kernel_config, l_mateltwise_desc );

  if (libxsmm_meqn_is_unary_opcode_transform_kernel(LIBXSMM_CAST_USHORT(i_op_type)) > 0) {
    libxsmm_generator_transform_rv64_microkernel( io_generated_code, io_loop_label_tracker, &l_mateltwise_gp_reg_mapping, &l_mateltwise_kernel_config, l_mateltwise_desc );
  } else {
    libxsmm_generator_unary_binary_rv64_microkernel( io_generated_code, io_loop_label_tracker, &l_mateltwise_gp_reg_mapping, &l_mateltwise_kernel_config, l_mateltwise_desc  );
  }

  if (i_op2_type != LIBXSMM_MELTW_TYPE_UNARY_NONE) {
    libxsmm_generator_gemm_getval_stack_var_rv64( io_generated_code, i_stack_var_scratch_ptr, i_tmp_reg);
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_SD, i_struct_gp_reg, l_offset_ptr_a, i_tmp_reg );

    libxsmm_generator_gemm_getval_stack_var_rv64( io_generated_code, i_stack_var_dst_ptr, i_tmp_reg);
    if (is_brgemm > 0) {
      libxsmm_rv64_instruction_alu_compute( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, i_tmp_reg3, i_tmp_reg, i_tmp_reg );
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD, i_tmp_reg3, i_tmp_reg2, i_tmp_reg3, (long long)LIBXSMM_TYPESIZE(i_out2_dtype) * i_m2 * i_n2);
    }
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_SD, i_struct_gp_reg, i_tmp_reg, l_offset_ptr_b);

    l_mateltwise_desc = libxsmm_meltw_descriptor_init2(&l_meltw_blob, i_in2_dtype, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED,
      i_comp2_dtype, i_out2_dtype, i_m2, i_n2, i_ldi2, i_ldo2, 0, 0,
      0, LIBXSMM_CAST_USHORT(i_op2_type), LIBXSMM_MELTW_OPERATION_UNARY);
    libxsmm_generator_mateltwise_rv64_init_micro_kernel_config_fullvector( io_generated_code, &l_mateltwise_kernel_config, l_mateltwise_desc );

    if (libxsmm_meqn_is_unary_opcode_transform_kernel(LIBXSMM_CAST_USHORT(i_op2_type)) > 0) {
      libxsmm_generator_transform_rv64_microkernel( io_generated_code, io_loop_label_tracker, &l_mateltwise_gp_reg_mapping, &l_mateltwise_kernel_config, l_mateltwise_desc );
    } else {
      libxsmm_generator_unary_binary_rv64_microkernel( io_generated_code, io_loop_label_tracker, &l_mateltwise_gp_reg_mapping, &l_mateltwise_kernel_config, l_mateltwise_desc  );
    }
  }

  if (is_brgemm > 0) {
    if (is_stride_brgemm > 0) {
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD, i_gp_reg_in, i_gp_reg_in, (long long)i_tensor_stride );
    }
    libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, i_loop_reg, i_loop_reg, 8 );
    libxsmm_generator_loop_footer_rv64( io_generated_code, io_loop_label_tracker, i_bound_reg, 1 );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_getval_stack_var_rv64( libxsmm_generated_code* io_generated_code,
                                                   libxsmm_gemm_stack_var  stack_var,
                                                   unsigned int            i_gp_reg ) {
  int offset = libxsmm_generator_gemm_get_rbp_relative_offset(stack_var);
  /* make sure we requested a legal stack var */
  if (offset == 0) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_XSP, i_gp_reg, -1 * offset );
  libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg, i_gp_reg, 0 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_setval_stack_var_rv64( libxsmm_generated_code*  io_generated_code,
                                                   libxsmm_gemm_stack_var   stack_var,
                                                   unsigned int             i_aux_reg,
                                                   unsigned int             i_gp_reg ) {
  int offset = libxsmm_generator_gemm_get_rbp_relative_offset(stack_var);
  /* make sure we requested to set  a legal stack var */
  if (offset >= 0) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_XSP, i_aux_reg, -1 * offset );
  libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_SW, i_aux_reg, i_gp_reg, 0 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_setup_stack_frame_fill_ext_gemm_stack_vars_rv64(
    libxsmm_generated_code*         io_generated_code,
    const libxsmm_gemm_descriptor*  i_xgemm_desc,
    libxsmm_micro_kernel_config*    i_micro_kernel_config,
    const libxsmm_gp_reg_mapping*   i_gp_reg_mapping ) {
  int is_stride_brgemm  = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0) ? 1 : 0;
  int is_offset_brgemm  = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) > 0) ? 1 : 0;
  int is_address_brgemm = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0) ? 1 : 0;
  int is_brgemm         = ((is_stride_brgemm == 1) || (is_offset_brgemm == 1) || (is_address_brgemm == 1)) ? 1 : 0;
  int has_scf           = ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
  int has_A_pf_ptr      = (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2) ? 1 : 0;
  int has_B_pf_ptr      = 0;
  unsigned int struct_reg = i_gp_reg_mapping->gp_reg_help_1;
  unsigned int val_reg    = i_gp_reg_mapping->gp_reg_help_0;
  unsigned int aux_reg    = i_gp_reg_mapping->gp_reg_help_2;
  int l_offset_ptr_a = (int)sizeof(libxsmm_matrix_op_arg);
  int l_offset_ptr_b = (int)(sizeof(libxsmm_matrix_op_arg) + sizeof(libxsmm_matrix_arg));
  int l_offset_ptr_c = (int)(sizeof(libxsmm_matrix_op_arg) + 2*sizeof(libxsmm_matrix_arg));
  int l_offset_ptr_d = (int)(sizeof(libxsmm_matrix_op_arg) + 3*sizeof(libxsmm_matrix_arg));
  int l_offset_ptr_e = (int)(sizeof(libxsmm_matrix_op_arg) + 4*sizeof(libxsmm_matrix_arg));
  int l_offset_ptr_f = (int)(sizeof(libxsmm_matrix_op_arg) + 5*sizeof(libxsmm_matrix_arg));

  if (has_scf == 1) {
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW,
        struct_reg, val_reg, l_offset_ptr_c +16 );
    libxsmm_generator_gemm_setval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_INT8_SCF, aux_reg, val_reg );
  }

  if (has_A_pf_ptr == 1) {
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW,
        struct_reg, val_reg, l_offset_ptr_a + LIBXSMM_MATRIX_ARG_OFFSET_PREFETCH );
    libxsmm_generator_gemm_setval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_PFA_PTR, aux_reg, val_reg );
  }

  if (has_B_pf_ptr == 1) {
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW,
        struct_reg, val_reg, l_offset_ptr_b + LIBXSMM_MATRIX_ARG_OFFSET_PREFETCH );
    libxsmm_generator_gemm_setval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_PFB_PTR, aux_reg, val_reg );
  }

  if ((is_brgemm == 1) && ((libxsmm_cpuid_arm_mmla_gemm_pack_b_to_vnnit_on_stack() > 0) || (i_micro_kernel_config->decompress_A == 1) || ((LIBXSMM_GEMM_FLAG_TRANS_A & i_xgemm_desc->flags) > 0))) {
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW,
        i_gp_reg_mapping->gp_reg_reduce_count, val_reg, 0 );
    libxsmm_generator_gemm_setval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, aux_reg, val_reg );
  }

  if (is_offset_brgemm == 1) {
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW,
        struct_reg, val_reg, l_offset_ptr_a + 8 );
    libxsmm_generator_gemm_setval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_A_OFFS_BRGEMM_PTR, aux_reg, val_reg );

    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW,
        struct_reg, val_reg, l_offset_ptr_b + 8 );
    libxsmm_generator_gemm_setval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_B_OFFS_BRGEMM_PTR, aux_reg, val_reg );
  }

  if (i_micro_kernel_config->fused_eltwise == 1) {
    if (i_micro_kernel_config->has_colbias_act_fused == 1) {
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW,
          struct_reg, val_reg, l_offset_ptr_d );
      libxsmm_generator_gemm_setval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, aux_reg, val_reg );
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW,
          struct_reg, val_reg, l_offset_ptr_c + 8 );
      libxsmm_generator_gemm_setval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, aux_reg, val_reg );
    }
    if (i_micro_kernel_config->decompress_A == 1) {
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW,
          struct_reg, val_reg, l_offset_ptr_a + 16 );
      libxsmm_generator_gemm_setval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_BITMAP_PTR, aux_reg, val_reg );
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW,
          struct_reg, val_reg, l_offset_ptr_e );
      libxsmm_generator_gemm_setval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_DECOMPRESS_BUF, aux_reg, val_reg );
    }
    if (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) {
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW,
          struct_reg, val_reg, l_offset_ptr_c + 8 );
      libxsmm_generator_gemm_setval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, aux_reg, val_reg );
    }
    if (i_micro_kernel_config->fused_relu_bwd == 1) {
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW,
          struct_reg, val_reg, l_offset_ptr_c + 8 );
      libxsmm_generator_gemm_setval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_RELU_BITMASK_PTR, aux_reg, val_reg );
    }
    if (i_micro_kernel_config->norm_to_normT_B_ext_buf == 1) {
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW,
          struct_reg, val_reg, l_offset_ptr_f );
      libxsmm_generator_gemm_setval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_TRANS_EXT_BUF_B, aux_reg, val_reg );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_setup_stack_frame_allocate_scratch_rv64(
    libxsmm_generated_code*             io_generated_code,
    const libxsmm_gemm_descriptor*      i_xgemm_desc,
    const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
    libxsmm_micro_kernel_config*        i_micro_kernel_config ) {
  unsigned int gemm_scratch_size      = 0;
  unsigned int scratch_pad_size       = 0;
  unsigned int temp_reg = i_gp_reg_mapping->gp_reg_help_1;
  unsigned int temp_reg2 = i_gp_reg_mapping->gp_reg_help_0;

  /* Allocate scratch for stashing 32 zmms */
  if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI & i_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI) ) {
    gemm_scratch_size = 32 * 64;
  }
  if (i_micro_kernel_config->vnni_format_C > 0) {
    gemm_scratch_size = 32 * 64 + i_xgemm_desc->n * i_xgemm_desc->m * 4;
  }

  scratch_pad_size  = (gemm_scratch_size % 64 == 0) ? 0 : ((gemm_scratch_size + 63)/64) * 64 - gemm_scratch_size;
  gemm_scratch_size += scratch_pad_size;

  /* Allocate scratch for the A transpose */
  if ( (LIBXSMM_GEMM_FLAG_TRANS_A & i_xgemm_desc->flags) > 0 ) {
    int is_stride_brgemm  = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0) ? 1 : 0;
    int is_offset_brgemm  = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) > 0) ? 1 : 0;
    int is_address_brgemm = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0) ? 1 : 0;
    int is_brgemm         = ((is_stride_brgemm == 1) || (is_offset_brgemm == 1) || (is_address_brgemm == 1)) ? 1 : 0;
    unsigned int inp_dtype_size = ((LIBXSMM_GEMM_FLAG_TRANS_A & i_xgemm_desc->flags) > 0) ?  LIBXSMM_TYPESIZE(LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC(i_xgemm_desc->datatype)) : 4;
    unsigned int a_size  = (i_xgemm_desc->m * i_xgemm_desc->k) * inp_dtype_size;
    unsigned int b_size  = (i_xgemm_desc->k * i_xgemm_desc->n) * inp_dtype_size;
    unsigned int a_pad  = (a_size % 64 == 0) ? 0 : ((a_size + 63)/64) * 64 - a_size;
    unsigned int b_pad  = (b_size % 64 == 0) ? 0 : ((b_size + 63)/64) * 64 - b_size;
    a_size += a_pad;
    b_size += b_pad;

    if (is_brgemm == 0) {
      libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_XSP, temp_reg, 0 );
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD, temp_reg, temp_reg2, temp_reg, -1 * a_size );
      libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, temp_reg, LIBXSMM_RV64_GP_REG_XSP, 0 );
      libxsmm_generator_gemm_setval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR, temp_reg2, temp_reg );
    } else {
      libxsmm_generator_gemm_getval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg);
      libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, temp_reg2, a_size );
      libxsmm_rv64_instruction_alu_compute( io_generated_code, LIBXSMM_RV64_INSTR_GP_MUL, temp_reg, temp_reg2, temp_reg );
      libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_XSP, temp_reg2, 0 );
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_SUB, temp_reg2, temp_reg, temp_reg2, 0 );
      libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, temp_reg2, LIBXSMM_RV64_GP_REG_XSP, 0 );
      libxsmm_generator_gemm_setval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR, temp_reg, temp_reg2 );
      if (is_offset_brgemm > 0 || is_address_brgemm > 0) {
        libxsmm_generator_gemm_getval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, temp_reg);
        libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, temp_reg2, b_size );
        libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_MUL, temp_reg, temp_reg2, temp_reg, 0 );
        libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_XSP, temp_reg2, 0 );
        libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_SUB, temp_reg2, temp_reg, temp_reg2, 0 );
        libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, temp_reg2, LIBXSMM_RV64_GP_REG_XSP, 0 );
        libxsmm_generator_gemm_setval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, temp_reg, temp_reg2 );
      }
    }
    libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_XSP, temp_reg, 0 );
    libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD, temp_reg, temp_reg2, temp_reg, 128 );
    libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, temp_reg, LIBXSMM_RV64_GP_REG_XSP, 0 );
    libxsmm_generator_gemm_setval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_MELTW_STRUCT_PTR, temp_reg2, temp_reg );
  }

  if (gemm_scratch_size > 0) {
    libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_XSP, temp_reg, 0 );
    libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_SUB, temp_reg, temp_reg2, temp_reg, gemm_scratch_size );
    libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, temp_reg, LIBXSMM_RV64_GP_REG_XSP, 0 );
    libxsmm_generator_gemm_setval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, temp_reg2, temp_reg );
  }

}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_setup_stack_frame_rv64(
    libxsmm_generated_code*         io_generated_code,
    const libxsmm_gemm_descriptor*  i_xgemm_desc,
    const libxsmm_gp_reg_mapping*   i_gp_reg_mapping,
    libxsmm_micro_kernel_config*    i_micro_kernel_config ) {
  unsigned int temp_reg = i_gp_reg_mapping->gp_reg_help_1;
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_XSP, LIBXSMM_RV64_GP_REG_X27, 0 );
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_XSP, LIBXSMM_RV64_GP_REG_XSP, -192 );

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
   *      AVX2 low precision helper                 <-- RBP-112, RSP
   */
  if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI & i_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI) || ((LIBXSMM_GEMM_FLAG_TRANS_A & i_xgemm_desc->flags) > 0) ) {
    libxsmm_generator_gemm_setup_stack_frame_fill_ext_gemm_stack_vars_rv64( io_generated_code, i_xgemm_desc, i_micro_kernel_config, i_gp_reg_mapping );
  } else {
    int has_scf = ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
    if (has_scf == 1) {
      libxsmm_generator_gemm_setval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_INT8_SCF, temp_reg, i_gp_reg_mapping->gp_reg_scf );
    }
  }

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
   *      AVX2 low precision helper             <-- RBP-112, RSP
   *      [ Potential pad for 64b align ]
   *      AV2 mask, 64b aligned                 <-- (RBP-104) contains this address
   *      AV2 low precision helper, 64b aligned <-- (RBP-112) contains this address
   *      GEMM scratch, 64b aligned             <-- (RBP-48) contains this address
   */
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_destroy_stack_frame_rv64( libxsmm_generated_code* io_generated_code) {
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_X27, LIBXSMM_RV64_GP_REG_XSP, 0 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_init_micro_kernel_config_rv64( libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                           const unsigned int             i_arch,
                                                           const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  memset(io_micro_kernel_config, 0, sizeof(*io_micro_kernel_config)); /* avoid warning "maybe used uninitialized" */
  libxsmm_generator_gemm_setup_fusion_microkernel_properties(i_xgemm_desc, io_micro_kernel_config);
  if ( i_arch == LIBXSMM_RV64_MVL128 || i_arch == LIBXSMM_RV64_MVL128_LMUL ) {
    io_micro_kernel_config->instruction_set = i_arch;
    io_micro_kernel_config->vector_reg_count = 32;
    io_micro_kernel_config->use_masking_a_c = 0;
    io_micro_kernel_config->vector_name = 'v';
    if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 2;
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->datatype_size_out = 8;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_RV64_INSTR_RVV_VLE64_V;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_RV64_INSTR_RVV_VLE64_V;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_RV64_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_RV64_INSTR_RVV_VSE64_V;
      io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_RV64_INSTR_RVV_VSE64_V;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_RV64_INSTR_UNDEF;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_RV64_INSTR_RVV_VFMACC_VV;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_RV64_INSTR_UNDEF;
    } else if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 4;
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->datatype_size_out = 4;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_RV64_INSTR_RVV_VLE32_V;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_RV64_INSTR_RVV_VLE32_V;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_RV64_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_RV64_INSTR_RVV_VSE32_V;
      io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_RV64_INSTR_RVV_VSE32_V;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_RV64_INSTR_UNDEF;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_RV64_INSTR_RVV_VFMACC_VV;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_RV64_INSTR_UNDEF;
    } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) { /* TODO (MMLA): do a proper integration; right now just assumes A in MMLA format, rest col-major */
      io_micro_kernel_config->vector_length = 8;
      io_micro_kernel_config->datatype_size_in = 2;
      if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
        io_micro_kernel_config->datatype_size_out = 4;
      } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
        io_micro_kernel_config->datatype_size_out = 2;
      } else {
        /* Should not happen */
      }
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_RV64_INSTR_RVV_VLE16_V;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_RV64_INSTR_RVV_VLE16_V;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_RV64_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_RV64_INSTR_RVV_VSE16_V;
      io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_RV64_INSTR_RVV_VSE16_V;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_RV64_INSTR_UNDEF;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_RV64_INSTR_UNDEF;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_RV64_INSTR_UNDEF;
    } else {
      /* should not happend */
    }
  }
  else if ( i_arch == LIBXSMM_RV64_MVL256 || i_arch == LIBXSMM_RV64_MVL256_LMUL ) {
    io_micro_kernel_config->instruction_set = i_arch;
    io_micro_kernel_config->vector_reg_count = 32;
    io_micro_kernel_config->use_masking_a_c = 0;
    io_micro_kernel_config->vector_name = 'v';
    if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 4;
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->datatype_size_out = 8;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_RV64_INSTR_RVV_VLE64_V;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_RV64_INSTR_RVV_VLE64_V;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_RV64_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_RV64_INSTR_RVV_VSE64_V;
      io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_RV64_INSTR_RVV_VSE64_V;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_RV64_INSTR_UNDEF;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_RV64_INSTR_RVV_VFMACC_VV;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_RV64_INSTR_UNDEF;
    } else if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 8;
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->datatype_size_out = 4;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_RV64_INSTR_RVV_VLE32_V;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_RV64_INSTR_RVV_VLE32_V;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_RV64_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_RV64_INSTR_RVV_VSE32_V;
      io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_RV64_INSTR_RVV_VSE32_V;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_RV64_INSTR_UNDEF;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_RV64_INSTR_RVV_VFMACC_VV;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_RV64_INSTR_UNDEF;
    } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) { /* TODO (MMLA): do a proper integration; right now just assumes A in MMLA format, rest col-major */
      io_micro_kernel_config->vector_length = 16;
      io_micro_kernel_config->datatype_size_in = 2;
      if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
        io_micro_kernel_config->datatype_size_out = 4;
      } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
        io_micro_kernel_config->datatype_size_out = 2;
      } else {
        /* Should not happen */
      }
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_RV64_INSTR_RVV_VLE16_V;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_RV64_INSTR_RVV_VLE16_V;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_RV64_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_RV64_INSTR_RVV_VSE16_V;
      io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_RV64_INSTR_RVV_VSE16_V;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_RV64_INSTR_UNDEF;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_RV64_INSTR_UNDEF;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_RV64_INSTR_UNDEF;
    } else {
      /* should not happend */
    }
  }
  else {
      /* should not happend */
  }
}

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_rv64_get_max_n_blocking( const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                                              const libxsmm_gemm_descriptor*      i_xgemm_desc,
                                                              const unsigned int                  i_arch ) {
  LIBXSMM_UNUSED( i_micro_kernel_config );
  LIBXSMM_UNUSED( i_xgemm_desc );

  if ( i_arch >= LIBXSMM_RV64_MVL128 ) {
    /* return libxsmm_cpuid_mvl_rv64(); */
    return 10;
  } else {
    return 0;
  }
}

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_rv64_get_initial_m_blocking( libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                                 const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                                 const unsigned int              i_arch ) {
  unsigned int l_m_blocking = 1;

  if ( ( i_arch >= LIBXSMM_RV64_MVL128 ) && ( ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ||
                                                                                           ( LIBXSMM_DATATYPE_I32  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ||
                                                                                           ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) )    ) ) {
    /* Remark switching ti OUT datatype check here to cover BF16 in, Fp32/Int32 out kernel with the same logic */
    /* TODO: check if there is a better blocking strategy */
    if ( i_xgemm_desc->m >= 32 ) {
      l_m_blocking = 32;
    } else if ( i_xgemm_desc->m >= 16 ) {
      l_m_blocking = 16;
    } else if ( i_xgemm_desc->m >= 8 ) {
      l_m_blocking = 8;
    } else if ( i_xgemm_desc->m >= 4 ) {
      l_m_blocking = 4;
    } else{
      l_m_blocking = i_xgemm_desc->m % 4;
    }
  } else if ( ( i_arch >= LIBXSMM_RV64_MVL128 ) && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ) {
    /* TODO: check if there is a better blocking strategy */
    if ( i_xgemm_desc->m >= 16 ) {
      l_m_blocking = 16;
    } else
    if ( i_xgemm_desc->m >= 8 ) {
      l_m_blocking = 8;
    } else if ( i_xgemm_desc->m >= 4 ) {
      l_m_blocking = 4;
    } else if ( i_xgemm_desc->m >= 2 ) {
      l_m_blocking = 2;
    } else {
      l_m_blocking = i_xgemm_desc->m;
    }
  } else {
    /* we should never end up here, if we do let the user know */
    /*LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
    return 0;*/
  }

  libxsmm_generator_gemm_init_micro_kernel_config_rv64( io_micro_kernel_config, i_arch, i_xgemm_desc );

  return l_m_blocking;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_rv64_update_m_blocking( libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                            const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                            const unsigned int             i_arch,
                                                            const unsigned int             i_current_m_blocking ) {
  unsigned int l_m_blocking = 0;

  if ( ( i_arch >= LIBXSMM_RV64_MVL128 ) && (( LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) || ( LIBXSMM_DATATYPE_I32  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) || ( LIBXSMM_DATATYPE_BF16  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) )) ) {
    /* Remark switching ti OUT datatype check here to cover BF16 in, Fp32 out kernel with the same logic */
    if (i_current_m_blocking == 32 ) {
      if ((i_xgemm_desc->m % 32) >= 16){
        l_m_blocking = 16;
      }
      else if ((i_xgemm_desc->m % 16) >= 8){
        l_m_blocking = 8;
      }
      else if ((i_xgemm_desc->m % 8) >= 4){
        l_m_blocking = 4;
      }
      else {
        l_m_blocking = i_xgemm_desc->m % 4;
      }
    } else if (i_current_m_blocking == 16 ) {
      if ((i_xgemm_desc->m % 16) >= 8){
        l_m_blocking = 8;
      }
      else if ((i_xgemm_desc->m % 8) >= 4){
        l_m_blocking = 4;
      }
      else {
        l_m_blocking = i_xgemm_desc->m % 4;
      }
    } else if (i_current_m_blocking == 8 ) {
      if ((i_xgemm_desc->m % 8) >= 4){
        l_m_blocking = 4;
      }
      else {
        l_m_blocking = i_xgemm_desc->m % 8;
      }
    } else if (i_current_m_blocking == 4 ) {
      l_m_blocking = i_xgemm_desc->m % 4;
      /* we are done with m_blocking */
    }
  } else if ( ( i_arch >= LIBXSMM_RV64_MVL128 ) && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ) {
    if (i_current_m_blocking == 16) {
      if ((i_xgemm_desc->m % 16) >= 8){
        l_m_blocking = 8;
      } else if ((i_xgemm_desc->m % 8) >= 4){
        l_m_blocking = 4;
      } else {
        l_m_blocking = i_xgemm_desc->m % 4;
      }
    } else if (i_current_m_blocking == 8) {
      if ((i_xgemm_desc->m % 8) >= 4){
        l_m_blocking = 4;
      } else {
        l_m_blocking = i_xgemm_desc->m % 4;
      }
    } else if (i_current_m_blocking == 4) {
      if ((i_xgemm_desc->m % 4) >= 2){
        l_m_blocking = 2;
      } else {
        l_m_blocking = i_xgemm_desc->m % 4;
      }
    } else if (i_current_m_blocking == 2) {
      l_m_blocking = i_xgemm_desc->m % 2;
    }
    /* we are done with m_blocking */
  } else {
    /* we should never end up here, if we do let the user know */
    /*LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
    return 0;*/
  }

  libxsmm_generator_gemm_init_micro_kernel_config_rv64( io_micro_kernel_config, i_arch, i_xgemm_desc );

  return l_m_blocking;
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_rv64_setup_n_blocking( libxsmm_generated_code*        io_generated_code,
                                                   libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                   const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                   const unsigned int             i_arch,
                                                   unsigned int*                  o_n_N,
                                                   unsigned int*                  o_n_n) {
  unsigned int max_n_blocking = libxsmm_generator_gemm_rv64_get_max_n_blocking( io_micro_kernel_config, i_xgemm_desc, i_arch );
  const unsigned int init_m_blocking = libxsmm_generator_gemm_rv64_get_initial_m_blocking( io_micro_kernel_config, i_xgemm_desc, i_arch );
  unsigned int init_m_blocks = 0;

  /* check for valid values */
  if ( max_n_blocking == 0 || io_micro_kernel_config->vector_length == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  init_m_blocks = LIBXSMM_UPDIV(init_m_blocking, io_micro_kernel_config->vector_length);

  /* increment m register blocking in case of 2 remainder registers */
  if (init_m_blocking > 0) {
    if ( (init_m_blocking % io_micro_kernel_config->vector_length == 3) || ((i_xgemm_desc->m % init_m_blocking) % io_micro_kernel_config->vector_length == 3) ) {
      init_m_blocks++;
    }
  }

  while ((init_m_blocks * max_n_blocking + init_m_blocks + 1) > io_micro_kernel_config->vector_reg_count) {
    max_n_blocking--;
  }

  if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0 ) {
    max_n_blocking = 16;
    if (i_xgemm_desc->n < 16) {
      o_n_N[0] = i_xgemm_desc->n;
      o_n_n[0] = i_xgemm_desc->n;
      o_n_N[1] = 0;
      o_n_n[1] = 0;
    } else {
      o_n_N[0] = i_xgemm_desc->n - i_xgemm_desc->n % max_n_blocking;
      o_n_n[0] = max_n_blocking;
      o_n_N[1] = i_xgemm_desc->n % max_n_blocking;
      o_n_n[1] = i_xgemm_desc->n % max_n_blocking;
    }
  } else {
    libxsmm_compute_equalized_blocking( i_xgemm_desc->n, max_n_blocking, &(o_n_N[0]), &(o_n_n[0]), &(o_n_N[1]), &(o_n_n[1]) );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_rv64_setup_k_strides( libxsmm_generated_code*            io_generated_code,
                                                  const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                  const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                  const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                  const unsigned int                 i_m_blocking,
                                                  const unsigned int                 i_n_blocking ) {
  /* temp variable for b-offset to handle no-trans/trans B */
  int l_b_offset = 0;
  /* register blocking counter in n */
  libxsmm_blasint l_rv64_bfdot = (libxsmm_blasint)libxsmm_cpuid_arm_use_bfdot();
  libxsmm_blasint l_rv64_i8dot = (libxsmm_blasint)libxsmm_cpuid_arm_use_i8dot();

  /* preload offset of B */
  if ( (io_generated_code->arch >= LIBXSMM_RV64_MVL128) ) {
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
      l_b_offset = (i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in) - (i_micro_kernel_config->datatype_size_in*i_n_blocking);
    } else {
      l_b_offset = i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in;
      libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_help_2, l_b_offset );

      l_b_offset = i_xgemm_desc->ldb - i_n_blocking;
      l_b_offset *= i_micro_kernel_config->datatype_size_in;;
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }

  libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_help_1, l_b_offset );

  /* preload offset of A */
  if ( (l_rv64_bfdot != 0) && ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ) {
    libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_help_0,
                                             ((long long)i_xgemm_desc->lda - i_m_blocking) * 2*i_micro_kernel_config->datatype_size_in );
  } else if ( (l_rv64_i8dot != 0) && ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ) {
    libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_help_0,
                                             ((long long)i_xgemm_desc->lda - i_m_blocking) * 4*i_micro_kernel_config->datatype_size_in );
  } else {
    libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_help_0,
                                             ((long long)i_xgemm_desc->lda - i_m_blocking) * i_micro_kernel_config->datatype_size_in );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_apply_relu_fusion_2dregblock_rv64(  libxsmm_generated_code*         io_generated_code,
                                                                const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                                libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                                const unsigned int              i_gp_reg_scratch0,
                                                                const unsigned int              i_gp_reg_scratch1,
                                                                const unsigned int              i_vec_length,
                                                                const unsigned int              i_vec_reg_count,
                                                                const unsigned int              i_m_blocking,
                                                                const unsigned int              i_n_blocking ) {
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  unsigned int l_m_blocks[2] = { 0 }; /* 0: #full vector stores, 1: #predicate stores (0 or 1) */
  unsigned int l_m_total_blocks = 0;
  unsigned int l_vec_reg_acc_start = 0;
  unsigned int l_remainder_size = 0;

  l_m_blocks[0] = i_m_blocking / i_vec_length;
  l_remainder_size = i_m_blocking % i_vec_length;
  l_m_blocks[1] = (l_remainder_size > 0);
  l_m_total_blocks = l_m_blocks[0] + l_m_blocks[1];

  /* start register of accumulator */
  l_vec_reg_acc_start = i_vec_reg_count - (i_n_blocking * l_m_total_blocks);

  libxsmm_rv64_instruction_alu_compute( io_generated_code, LIBXSMM_RV64_INSTR_GP_FMV_W_X,
                                        LIBXSMM_RV64_GP_REG_X0, LIBXSMM_RV64_GP_REG_X0,
                                        LIBXSMM_RV64_GP_REG_F31);

  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    /* this is the jump size to be performed after a m-block is complete */
    for ( l_m = 0; l_m < l_m_total_blocks; l_m++ ) {
      unsigned int cur_vreg = l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m;

      libxsmm_rv64_instruction_rvv_compute( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VFMAX_VF,
                                               LIBXSMM_RV64_GP_REG_F31, cur_vreg, cur_vreg, 1);
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_apply_fusion_2dregblock_rv64(  libxsmm_generated_code*         io_generated_code,
                                                           const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                           libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                           const unsigned int              i_gp_reg_scratch0,
                                                           const unsigned int              i_gp_reg_scratch1,
                                                           const unsigned int              i_vec_length,
                                                           const unsigned int              i_vec_reg_count,
                                                           const unsigned int              i_m_blocking,
                                                           const unsigned int              i_n_blocking ) {
  if ( (io_micro_kernel_config->fused_relu_nobitmask > 0) ) {
    libxsmm_generator_gemm_apply_relu_fusion_2dregblock_rv64( io_generated_code, i_xgemm_desc, io_micro_kernel_config, i_gp_reg_scratch0, i_gp_reg_scratch1, i_vec_length, i_vec_reg_count, i_m_blocking, i_n_blocking );
  } else if ( (io_micro_kernel_config->fused_relu > 0) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    return;
  } else if ( (io_micro_kernel_config->fused_sigmoid > 0) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    return;
  } else {
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_load_add_colbias_2dregblock_rv64(  libxsmm_generated_code*     io_generated_code,
                                                               const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                               const unsigned int              i_gp_reg_addr,
                                                               const unsigned int              i_gp_reg_scratch0,
                                                               const unsigned int              i_vec_length,
                                                               const unsigned int              i_vec_reg_count,
                                                               libxsmm_datatype                colbias_precision,
                                                               const unsigned int              i_m_blocking,
                                                               const unsigned int              i_n_blocking,
                                                               const unsigned int              i_ld ) {
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;

  unsigned int l_m_blocks[2] = { 0 }; /* 0: #full vector loads, 1: #predicate loads (0 or 1) */
  unsigned int l_m_total_blocks = 0;
  unsigned int l_m_bytes_full = 0;
  unsigned int l_vec_reg_acc_start = 0;
  unsigned int l_remainder_size = 0;
  unsigned int l_gp_reg_bias = i_gp_reg_scratch0;
  unsigned int l_bias_tsize = LIBXSMM_TYPESIZE( colbias_precision );
  unsigned int l_matrix_tsize = LIBXSMM_TYPESIZE( (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) );
  unsigned int l_matrix_load_instr = (LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_BF16) == l_matrix_tsize) ? LIBXSMM_RV64_INSTR_RVV_VLE16_V : ((LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_F32) == l_bias_tsize) ? LIBXSMM_RV64_INSTR_RVV_VLE32_V : LIBXSMM_RV64_INSTR_RVV_VLE64_V);
  unsigned int l_bias_load_instr = (LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_BF16) == l_bias_tsize) ? LIBXSMM_RV64_INSTR_RVV_VLE16_V : ((LIBXSMM_TYPESIZE(LIBXSMM_DATATYPE_F32) == l_bias_tsize) ? LIBXSMM_RV64_INSTR_RVV_VLE32_V : LIBXSMM_RV64_INSTR_RVV_VLE64_V);

  l_m_blocks[0] = i_m_blocking / i_vec_length;
  l_remainder_size = i_m_blocking % i_vec_length;
  l_m_blocks[1] = (l_remainder_size > 0);
  l_m_total_blocks = l_m_blocks[0] + l_m_blocks[1];
  l_m_bytes_full = l_m_blocks[0] * i_vec_length * l_matrix_tsize;

  /* start register of accumulator */
  l_vec_reg_acc_start = i_vec_reg_count - (i_n_blocking * l_m_total_blocks);

  /* load C accumulator */
  libxsmm_generator_gemm_getval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, l_gp_reg_bias);

  if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BETA_0) == 0) {
    /* this is the jump size to be performed after a n-block is complete */
    unsigned long long l_jump_block_n_last = 0;

    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      /* this is the jump size to be performed after a m-block is complete */
      unsigned long long l_jump_block_m_last = 0;

      for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
        if (l_n == 0) {
          libxsmm_rv64_instruction_rvv_move( io_generated_code,
                                             l_bias_load_instr,
                                             l_gp_reg_bias,
                                             LIBXSMM_RV64_GP_REG_UNDEF,
                                             l_m, 1 );

          if ( l_m_blocks[1] != 0 || l_m != l_m_blocks[0] - 1 ) {
            libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code,
                                                        LIBXSMM_RV64_INSTR_GP_ADDI,
                                                        l_gp_reg_bias,
                                                        l_gp_reg_bias,
                                                        i_vec_length * l_bias_tsize );
          }
        }

        libxsmm_rv64_instruction_rvv_move( io_generated_code,
                                           l_matrix_load_instr,
                                           i_gp_reg_addr,
                                           LIBXSMM_RV64_GP_REG_UNDEF,
                                           l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m, 1 );

        if ( l_m_blocks[1] != 0 || l_m != l_m_blocks[0] - 1 ) {
          libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code,
                                                      LIBXSMM_RV64_INSTR_GP_ADDI,
                                                      i_gp_reg_addr,
                                                      i_gp_reg_addr,
                                                      i_vec_length * l_matrix_tsize );
        }
        /* combine the m-jump with the n one*/
        else {
          l_jump_block_m_last = (long long)i_vec_length * l_matrix_tsize;
        }
        libxsmm_rv64_instruction_rvv_compute( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VADD_VV,
            l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m, l_m, l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m, 1 );
      }

      if ( l_m_blocks[1] != 0 ) {
        if (l_n == 0) {
          libxsmm_rv64_instruction_rvv_move( io_generated_code,
                                             l_bias_load_instr,
                                             l_gp_reg_bias,
                                             LIBXSMM_RV64_GP_REG_UNDEF,
                                             l_m_blocks[0],
                                             1 );
        }

        libxsmm_rv64_instruction_rvv_move( io_generated_code,
                                           l_matrix_load_instr,
                                           i_gp_reg_addr,
                                           LIBXSMM_RV64_GP_REG_UNDEF,
                                           l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m_blocks[0],
                                           1 );

        libxsmm_rv64_instruction_rvv_compute( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VFADD_VV,
            l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m_blocks[0], l_m_blocks[0], l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m_blocks[0], 1 );

      }
      l_jump_block_m_last += (unsigned long long)i_ld - l_m_bytes_full;

      if ( l_n != i_n_blocking - 1 ) {
        libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_RV64_INSTR_GP_ADD,
                                                    i_gp_reg_addr,
                                                    i_gp_reg_scratch0,
                                                    i_gp_reg_addr,
                                                    l_jump_block_m_last );
      }
      else {
        l_jump_block_n_last = l_jump_block_m_last;
      }
    }

    /* reset C-ptr to original address */
    l_jump_block_n_last = (long long)i_ld * i_n_blocking - l_jump_block_n_last;
    libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code,
                                                LIBXSMM_RV64_INSTR_GP_SUB,
                                                i_gp_reg_addr,
                                                i_gp_reg_scratch0,
                                                i_gp_reg_addr,
                                                l_jump_block_n_last );
  } else {
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
        if (l_n == 0) {
          libxsmm_rv64_instruction_rvv_move( io_generated_code,
                                                l_bias_load_instr,
                                                l_gp_reg_bias,
                                                LIBXSMM_RV64_GP_REG_UNDEF,
                                                l_m, 1 );

          if ( l_m_blocks[1] != 0 || l_m != l_m_blocks[0] - 1 ) {
            libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code,
                                                        LIBXSMM_RV64_INSTR_GP_ADDI,
                                                        l_gp_reg_bias,
                                                        l_gp_reg_bias,
                                                        i_vec_length * l_bias_tsize );
          }
        }
        libxsmm_rv64_instruction_rvv_compute( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VOR_VV,
            l_m, l_m, l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m, 1 );
      }

      if ( l_m_blocks[1] != 0 ) {
        if (l_n == 0) {
          libxsmm_rv64_instruction_rvv_move( io_generated_code,
                                             l_bias_load_instr,
                                             l_gp_reg_bias,
                                             LIBXSMM_RV64_GP_REG_UNDEF,
                                             l_m_blocks[0],
                                             1 );
        }
        libxsmm_rv64_instruction_rvv_compute( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VOR_VV,
            l_m_blocks[0], l_m_blocks[0], l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m_blocks[0], 1 );
      }
    }
  }
}

/* Setup A transpose tensor in stack */
LIBXSMM_API_INTERN void libxsmm_generator_gemm_setup_A_vnni_or_trans_B_vnni_or_trans_tensor_to_stack_rv64( libxsmm_generated_code*        io_generated_code,
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
  unsigned int struct_gp_reg  = LIBXSMM_RV64_GP_REG_X18;
  unsigned int tmp_reg        = LIBXSMM_RV64_GP_REG_X19;
  unsigned int loop_reg       = LIBXSMM_RV64_GP_REG_X20;
  unsigned int bound_reg      = LIBXSMM_RV64_GP_REG_X21;
  unsigned int tmp_reg2       = LIBXSMM_RV64_GP_REG_X22;
  unsigned int tmp_reg3       = LIBXSMM_RV64_GP_REG_X23;
  libxsmm_meltw_unary_type l_trans_unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT;
  libxsmm_datatype l_trans_dt = i_in_dtype;
  unsigned int l_trans_m = i_xgemm_desc_orig->k;
  unsigned int l_trans_n = i_xgemm_desc_orig->m;
  unsigned int l_trans_ldi = i_xgemm_desc_orig->lda;
  unsigned int l_trans_ldo = i_xgemm_desc_orig->m;

  libxsmm_rv64_instruction_open_stream( io_generated_code, 0xe0f );

  if ( (i_micro_kernel_config->atvnni_gemm_stack_alloc_tensors !=0) || (i_micro_kernel_config->atvnni_btrans_gemm_stack_alloc_tensors !=0) ) {
    unsigned int l_use_mmla = ((libxsmm_cpuid_arm_use_bfdot() == 0) && (i_xgemm_desc_orig->k%4 == 0)) ? 1 : 0;
    l_trans_unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT;
    l_trans_dt = (l_use_mmla == 1) ? LIBXSMM_DATATYPE_F64 : LIBXSMM_DATATYPE_F32;
    l_trans_m = (l_use_mmla == 1) ? i_xgemm_desc_orig->k/4 : i_xgemm_desc_orig->k/2;
    l_trans_n = i_xgemm_desc_orig->m;
    l_trans_ldi = (l_use_mmla == 1) ? i_xgemm_desc_orig->lda/4 : i_xgemm_desc_orig->lda/2;
    l_trans_ldo = i_xgemm_desc_orig->m;
  } else if ( (i_micro_kernel_config->avnni_gemm_stack_alloc_tensors !=0) || (i_micro_kernel_config->avnni_btrans_gemm_stack_alloc_tensors !=0) ) {
    unsigned int l_use_mmla = ((libxsmm_cpuid_arm_use_bfdot() == 0) && (i_xgemm_desc_orig->k%4 == 0)) ? 1 : 0;
    l_trans_unary_type = (l_use_mmla == 1) ? LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4 : LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2;
    l_trans_dt = i_in_dtype;
    l_trans_m = i_xgemm_desc_orig->m;
    l_trans_n = i_xgemm_desc_orig->k;
    l_trans_ldi = i_xgemm_desc_orig->lda;
    l_trans_ldo = i_xgemm_desc_orig->m;
  }

  /* Setup A in stack (if A in vnni perform vnni4->norm and then fp8->fp32, else perform fp8->fp32 only) */
  libxsmm_generator_gemm_apply_ops_input_tensor_and_store_to_stack_rv64( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
      i_gp_reg_mapping->gp_reg_a, struct_gp_reg, tmp_reg, loop_reg, bound_reg, tmp_reg2, tmp_reg3,
      l_trans_unary_type, l_trans_m, l_trans_n, l_trans_ldi, l_trans_ldo, LIBXSMM_CAST_BLASINT(i_xgemm_desc_orig->c1),
      l_trans_dt, l_trans_dt, l_trans_dt,
      LIBXSMM_GEMM_STACK_VAR_A_OFFS_BRGEMM_PTR, LIBXSMM_GEMM_STACK_VAR_A_SCRATCH_PTR, LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR,
      LIBXSMM_MELTW_TYPE_UNARY_NONE, 0, 0, 0, 0, (libxsmm_datatype)0, (libxsmm_datatype)0, (libxsmm_datatype)0);

  /* Adjust A/B gp_regs to point to the fp32 tensors in stack */
  libxsmm_generator_gemm_getval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR, tmp_reg);
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, tmp_reg, i_gp_reg_mapping->gp_reg_a, 0 );

  /* In this case we have to copy over also B in strided BRGEMM format */
  if ( (i_micro_kernel_config->avnni_btrans_gemm_stack_alloc_tensors != 0) || (i_micro_kernel_config->atvnni_btrans_gemm_stack_alloc_tensors != 0) ) {
    libxsmm_generator_gemm_apply_ops_input_tensor_and_store_to_stack_rv64( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
        i_gp_reg_mapping->gp_reg_b, struct_gp_reg, tmp_reg, loop_reg, bound_reg, tmp_reg2, tmp_reg3,
        LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, i_xgemm_desc->n, i_xgemm_desc->k, i_xgemm_desc_orig->ldb, i_xgemm_desc->k, LIBXSMM_CAST_BLASINT(i_xgemm_desc_orig->c2),
        i_in_dtype, i_in_dtype, i_in_dtype,
        LIBXSMM_GEMM_STACK_VAR_B_OFFS_BRGEMM_PTR, LIBXSMM_GEMM_STACK_VAR_A_SCRATCH_PTR, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR,
        LIBXSMM_MELTW_TYPE_UNARY_NONE, 0, 0, 0, 0, (libxsmm_datatype)0, (libxsmm_datatype)0, (libxsmm_datatype)0);
    libxsmm_generator_gemm_getval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, tmp_reg);
    libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, tmp_reg, i_gp_reg_mapping->gp_reg_b, 0 );
    i_xgemm_desc->ldb = i_xgemm_desc->k;
    i_xgemm_desc->c2 = (long long)LIBXSMM_TYPESIZE(i_in_dtype) * i_xgemm_desc->n * i_xgemm_desc->k;
  } else {
    if ( (is_offset_brgemm > 0) || (is_address_brgemm > 0) ) {
      unsigned int l_is_trans_b = (i_xgemm_desc_orig->flags & LIBXSMM_GEMM_FLAG_TRANS_B);
      libxsmm_generator_gemm_apply_ops_input_tensor_and_store_to_stack_rv64( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
          i_gp_reg_mapping->gp_reg_b, struct_gp_reg, tmp_reg, loop_reg, bound_reg, tmp_reg2, tmp_reg3,
          LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, (l_is_trans_b) ? i_xgemm_desc->n : i_xgemm_desc->k, (l_is_trans_b) ? i_xgemm_desc->k : i_xgemm_desc->n, i_xgemm_desc_orig->ldb, (l_is_trans_b) ? i_xgemm_desc->n : i_xgemm_desc->k, LIBXSMM_CAST_BLASINT(i_xgemm_desc_orig->c2),
          i_in_dtype, i_in_dtype, i_in_dtype,
          LIBXSMM_GEMM_STACK_VAR_B_OFFS_BRGEMM_PTR, LIBXSMM_GEMM_STACK_VAR_A_SCRATCH_PTR, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR,
          LIBXSMM_MELTW_TYPE_UNARY_NONE, 0, 0, 0, 0, (libxsmm_datatype)0, (libxsmm_datatype)0, (libxsmm_datatype)0);
      i_xgemm_desc->ldb = (l_is_trans_b) ? i_xgemm_desc->n : i_xgemm_desc->k;
      libxsmm_generator_gemm_getval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, tmp_reg);
      libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, tmp_reg, i_gp_reg_mapping->gp_reg_b, 0, 0 );
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

  libxsmm_rv64_instruction_restore_regs( io_generated_code, 0xe0f );
}

/* Setup A transpose tensor in stack */
LIBXSMM_API_INTERN void libxsmm_generator_gemm_setup_B_in_vnniT_to_stack_rv64( libxsmm_generated_code*        io_generated_code,
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
  unsigned int struct_gp_reg  = LIBXSMM_RV64_GP_REG_X26;
  unsigned int tmp_reg        = LIBXSMM_RV64_GP_REG_X27;
  unsigned int loop_reg       = LIBXSMM_RV64_GP_REG_X28;
  unsigned int bound_reg      = LIBXSMM_RV64_GP_REG_X29;
  unsigned int tmp_reg2       = LIBXSMM_RV64_GP_REG_X30;
  unsigned int tmp_reg3       = LIBXSMM_RV64_GP_REG_X31;

  libxsmm_rv64_instruction_open_stream( io_generated_code, 0xe0f );

  libxsmm_generator_gemm_apply_ops_input_tensor_and_store_to_stack_rv64( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
      i_gp_reg_mapping->gp_reg_b, struct_gp_reg, tmp_reg, loop_reg, bound_reg, tmp_reg2, tmp_reg3,
      LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4T, i_xgemm_desc_orig->k, i_xgemm_desc_orig->n, i_xgemm_desc_orig->ldb, i_xgemm_desc_orig->n, LIBXSMM_CAST_BLASINT(i_xgemm_desc_orig->c2),
      LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16,
      LIBXSMM_GEMM_STACK_VAR_B_OFFS_BRGEMM_PTR, LIBXSMM_GEMM_STACK_VAR_A_SCRATCH_PTR, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR,
      LIBXSMM_MELTW_TYPE_UNARY_NONE, 0, 0, 0, 0, (libxsmm_datatype)0, (libxsmm_datatype)0, (libxsmm_datatype)0);

  libxsmm_generator_gemm_getval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, tmp_reg);
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, tmp_reg, i_gp_reg_mapping->gp_reg_b, 0 );

  /* In this case we have to copy over also A in strided BRGEMM format */
  if ( (is_offset_brgemm > 0) || (is_address_brgemm > 0) ) {
    unsigned int a_vnni_factor  = 1;
    int l_use_bfdot = libxsmm_cpuid_arm_use_bfdot();
    char l_use_mmla = 0;

    /* enable MMLA settings for supported datatypes */
    if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) ||
         (LIBXSMM_DATATYPE_I8   == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) ) {
      if ( l_use_bfdot == 0 && (i_xgemm_desc_orig->k % 4 == 0) ) {
        l_use_mmla = 1;
      } else {
        l_use_mmla = 0;
      }

      if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
        a_vnni_factor = ( l_use_mmla == 0 ) ? 2 : 4;
      }
      if (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
        a_vnni_factor = ( l_use_mmla == 0 ) ? 4 : 8;
      }
    }

    libxsmm_generator_gemm_apply_ops_input_tensor_and_store_to_stack_rv64( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
        i_gp_reg_mapping->gp_reg_a, struct_gp_reg, tmp_reg, loop_reg, bound_reg, tmp_reg2, tmp_reg3,
        LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, i_xgemm_desc_orig->m*a_vnni_factor, i_xgemm_desc_orig->k/a_vnni_factor, i_xgemm_desc_orig->lda*a_vnni_factor, i_xgemm_desc_orig->m*a_vnni_factor, LIBXSMM_CAST_BLASINT(i_xgemm_desc_orig->c1),
        i_in_dtype, i_in_dtype, i_in_dtype,
        LIBXSMM_GEMM_STACK_VAR_A_OFFS_BRGEMM_PTR, LIBXSMM_GEMM_STACK_VAR_A_SCRATCH_PTR, LIBXSMM_GEMM_STACK_VAR_A_EMU_PTR,
        LIBXSMM_MELTW_TYPE_UNARY_NONE, 0, 0, 0, 0, (libxsmm_datatype)0, (libxsmm_datatype)0, (libxsmm_datatype)0);
    libxsmm_generator_gemm_getval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_A_EMU_PTR, tmp_reg);
    libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, tmp_reg, i_gp_reg_mapping->gp_reg_a, 0 );
  }

  /* Adjust descriptor for internal strided BRGEMM */
  if (is_brgemm > 0) {
    if (is_offset_brgemm > 0) {
      i_xgemm_desc->flags = i_xgemm_desc->flags ^ LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET;
      i_xgemm_desc->flags = i_xgemm_desc->flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE;
      i_xgemm_desc->c1 = (long long)LIBXSMM_TYPESIZE(i_in_dtype) * i_xgemm_desc->m * i_xgemm_desc->k;
      i_xgemm_desc->lda = i_xgemm_desc->m;
    }
    if (is_address_brgemm > 0) {
      i_xgemm_desc->flags = i_xgemm_desc->flags ^ LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS;
      i_xgemm_desc->flags = i_xgemm_desc->flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE;
      i_xgemm_desc->c1 = (long long)LIBXSMM_TYPESIZE(i_in_dtype) * i_xgemm_desc->m * i_xgemm_desc->k;
      i_xgemm_desc->lda = i_xgemm_desc->m;
    }
    i_xgemm_desc->c2 = (long long)LIBXSMM_TYPESIZE(i_in_dtype) * i_xgemm_desc->n * i_xgemm_desc->k;
  }

  i_xgemm_desc->flags = i_xgemm_desc->flags | LIBXSMM_GEMM_FLAG_VNNI_B | LIBXSMM_GEMM_FLAG_TRANS_B;
  i_xgemm_desc->ldb = i_xgemm_desc->n;

  libxsmm_rv64_instruction_restore_regs( io_generated_code, 0xe0f );
}

LIBXSMM_API_INTERN void libxsmm_generator_gemm_apply_opA_opB_rv64( libxsmm_generated_code*        io_generated_code,
                                                                   libxsmm_loop_label_tracker*    io_loop_label_tracker,
                                                                   const libxsmm_gp_reg_mapping*  i_gp_reg_mapping,
                                                                   libxsmm_micro_kernel_config*   i_micro_kernel_config,
                                                                   libxsmm_gemm_descriptor*       i_xgemm_desc,
                                                                   const libxsmm_gemm_descriptor* i_xgemm_desc_orig ) {
  if ( ( i_xgemm_desc_orig->flags & LIBXSMM_GEMM_FLAG_TRANS_A) && (i_xgemm_desc_orig->m != 0) && (i_xgemm_desc_orig->k != 0) ) {
    /* if A needs to be transposed, use scratch in stack */
    libxsmm_generator_gemm_setup_A_vnni_or_trans_B_vnni_or_trans_tensor_to_stack_rv64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, i_xgemm_desc_orig, (libxsmm_datatype) LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc_orig->datatype ));
  } else if ( (i_micro_kernel_config->avnni_gemm_stack_alloc_tensors != 0) || (i_micro_kernel_config->avnni_btrans_gemm_stack_alloc_tensors !=0) ) {
    libxsmm_generator_gemm_setup_A_vnni_or_trans_B_vnni_or_trans_tensor_to_stack_rv64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, i_xgemm_desc_orig, (libxsmm_datatype) LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc_orig->datatype ));
  }
}
