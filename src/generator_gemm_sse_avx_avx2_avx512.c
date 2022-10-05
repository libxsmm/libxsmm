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
#include "generator_common.h"
#include "generator_common_x86.h"
#include "generator_x86_instructions.h"
#include "generator_gemm_common.h"
#include "generator_gemm_sse_avx_avx2_avx512.h"
#include "generator_gemm_sse_microkernel.h"
#include "generator_gemm_avx_microkernel.h"
#include "generator_gemm_avx2_microkernel.h"
#include "generator_gemm_avx512_microkernel.h"
#include "generator_mateltwise_transform_avx512.h"
#include "generator_mateltwise_transform_avx.h"
#include "generator_mateltwise_transform_sse.h"
#include "generator_mateltwise_sse_avx_avx512.h"
#include "generator_mateltwise_unary_binary_avx_avx512.h"
#include "generator_mateltwise_transform_common.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN void libxsmm_generator_gemm_sse_avx_avx2_avx512_kernel_wrapper( libxsmm_generated_code*        io_generated_code,
                                                                                   const libxsmm_gemm_descriptor* i_xgemm_desc       ) {
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;

  /* define gp register mapping */
  libxsmm_reset_x86_gp_reg_mapping( &l_gp_reg_mapping );
#if defined(_WIN32) || defined(__CYGWIN__)
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_R8;
  /* TODO: full support for Windows calling convention */
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_UNDEF;
#else /* match calling convention on Linux */
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R8;
  if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype )) ) {
    l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_RCX;
    l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_R8;
    l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R9;
  } else {
    l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_UNDEF;
    l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_RCX;
    l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R8;
  }
  /* If we are generating the batchreduce kernel, then we rename the registers  */
  if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
    l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
    l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
    l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
    l_gp_reg_mapping.gp_reg_reduce_count = LIBXSMM_X86_GP_REG_RCX;
    if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype )) ) {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_R8;
      l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_R9;
      l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_UNDEF;
    } else {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_UNDEF;
      l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_R8;
      l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R9;
    }
    l_gp_reg_mapping.gp_reg_reduce_loop = LIBXSMM_X86_GP_REG_R13;
  } else if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) {
    l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
    l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
    l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
    l_gp_reg_mapping.gp_reg_reduce_count = LIBXSMM_X86_GP_REG_RCX;
    l_gp_reg_mapping.gp_reg_a_offset = LIBXSMM_X86_GP_REG_R8;
    l_gp_reg_mapping.gp_reg_b_offset = LIBXSMM_X86_GP_REG_R9;
    if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype )) ) {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_RAX;
    } else {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_UNDEF;
    }
    l_gp_reg_mapping.gp_reg_reduce_loop = LIBXSMM_X86_GP_REG_R13;
  }
#endif
  l_gp_reg_mapping.gp_reg_mloop = LIBXSMM_X86_GP_REG_R10;
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_X86_GP_REG_R11;
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_X86_GP_REG_R12;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_X86_GP_REG_R14;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_X86_GP_REG_R15;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_X86_GP_REG_RBX;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* open asm */
  libxsmm_x86_instruction_open_stream_gemm( io_generated_code, &l_gp_reg_mapping, 0, i_xgemm_desc->prefetch );

  /* call Intel SIMD kernel */
  libxsmm_generator_gemm_sse_avx_avx2_avx512_kernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, i_xgemm_desc );

  /* close asm */
  libxsmm_x86_instruction_close_stream_gemm( io_generated_code, &l_gp_reg_mapping, 0, i_xgemm_desc->prefetch );
}

/* Setup A (in vnni4 or flat) and B bf8 tensors as fp32 tensors in stack */
LIBXSMM_API_INTERN void libxsmm_generator_gemm_setup_f8_AB_tensors_to_stack_as_fp32( libxsmm_generated_code*      io_generated_code,
                                                                                    libxsmm_loop_label_tracker*    io_loop_label_tracker,
                                                                                    const libxsmm_gp_reg_mapping*  i_gp_reg_mapping,
                                                                                    libxsmm_micro_kernel_config*   i_micro_kernel_config,
                                                                                    libxsmm_gemm_descriptor*       i_xgemm_desc,
                                                                                    const libxsmm_gemm_descriptor* i_xgemm_desc_orig,
                                                                                    libxsmm_datatype               i_in_dtype ) {
  libxsmm_descriptor_blob           l_meltw_blob;
  libxsmm_mateltwise_kernel_config  l_mateltwise_kernel_config;
  libxsmm_mateltwise_gp_reg_mapping l_mateltwise_gp_reg_mapping;
  const libxsmm_meltw_descriptor *  l_mateltwise_desc;

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

  libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R8 );
  libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R9 );
  libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R10 );
  libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R11 );
  libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R12 );
  libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R13 );
  libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R14 );
  libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R15 );
  libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RCX );
  libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );

  l_mateltwise_gp_reg_mapping.gp_reg_param_struct = struct_gp_reg;
  libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MELTW_STRUCT_PTR, struct_gp_reg );

  /* Loop over all batch-reduce iterations to cover all Ai / Bi*/
  if (is_brgemm > 0) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, bound_reg );
    libxsmm_generator_generic_loop_header_no_idx_inc( io_generated_code, io_loop_label_tracker, loop_reg, 0);
  }

  /* Setup input pointer of A in eltwise struct */
  if (is_offset_brgemm > 0) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_a, tmp_reg2);
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_A_OFFS_BRGEMM_PTR, tmp_reg );
    libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction, tmp_reg, loop_reg, 8, 0, tmp_reg, 0 );
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, tmp_reg, i_gp_reg_mapping->gp_reg_a);
  }
  if (is_address_brgemm > 0) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_a, tmp_reg2);
    libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_a, loop_reg, 8, 0, i_gp_reg_mapping->gp_reg_a, 0 );
  }
  libxsmm_x86_instruction_alu_mem( io_generated_code,
          LIBXSMM_X86_INSTR_MOVQ,
          struct_gp_reg,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          32,
          i_gp_reg_mapping->gp_reg_a,
          1 );
  if ((is_offset_brgemm > 0) || (is_address_brgemm > 0)) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, tmp_reg2, i_gp_reg_mapping->gp_reg_a);
  }

  /* If A in VNNI4 we need first to transform it to norm */
  if (a_in_vnni > 0) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_A_SCRATCH_PTR, tmp_reg );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
            LIBXSMM_X86_INSTR_MOVQ,
            struct_gp_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            64,
            tmp_reg,
            1 );
    l_mateltwise_desc = libxsmm_meltw_descriptor_init2(&l_meltw_blob, i_in_dtype, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED,
      i_in_dtype, i_in_dtype, i_xgemm_desc->m, i_xgemm_desc->k, i_xgemm_desc_orig->lda, i_xgemm_desc->lda, 0, 0,
      0, LIBXSMM_CAST_USHORT(LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_NORM), LIBXSMM_MELTW_OPERATION_UNARY);
    libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_mateltwise_kernel_config, l_mateltwise_desc );
    libxsmm_generator_transform_x86_microkernel( io_generated_code, io_loop_label_tracker, &l_mateltwise_gp_reg_mapping, &l_mateltwise_kernel_config, l_mateltwise_desc );
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_A_SCRATCH_PTR, tmp_reg );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
            LIBXSMM_X86_INSTR_MOVQ,
            struct_gp_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            32,
            tmp_reg,
            1 );
  }

  /* Setup output pointer of A in eltwise struct */
  libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_A_EMU_PTR, tmp_reg );
  if (is_brgemm > 0) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, loop_reg, tmp_reg2);
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, tmp_reg2, i_xgemm_desc->m * i_xgemm_desc->k * 4);
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, tmp_reg2, tmp_reg);
  }
  libxsmm_x86_instruction_alu_mem( io_generated_code,
          LIBXSMM_X86_INSTR_MOVQ,
          struct_gp_reg,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          64,
          tmp_reg,
          1 );

  /* Setup A chunk in stack as FP32 (flat) */
  l_mateltwise_desc = libxsmm_meltw_descriptor_init2(&l_meltw_blob, i_in_dtype, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED,
    LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, i_xgemm_desc->m, i_xgemm_desc->k, (a_in_vnni > 0) ? i_xgemm_desc->lda : i_xgemm_desc_orig->lda, i_xgemm_desc->lda, 0, 0,
    0, LIBXSMM_CAST_USHORT(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY), LIBXSMM_MELTW_OPERATION_UNARY);
  libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_mateltwise_kernel_config, l_mateltwise_desc );
  libxsmm_generator_unary_binary_avx512_microkernel( io_generated_code, io_loop_label_tracker, &l_mateltwise_gp_reg_mapping, &l_mateltwise_kernel_config, l_mateltwise_desc );

  /* Setup input pointer of B in eltwise struct */
  if (is_offset_brgemm > 0) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_b, tmp_reg2);
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_B_OFFS_BRGEMM_PTR, tmp_reg );
    libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction, tmp_reg, loop_reg, 8, 0, tmp_reg, 0 );
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, tmp_reg, i_gp_reg_mapping->gp_reg_b);
  }
  if (is_address_brgemm > 0) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_b, tmp_reg2);
    libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_b, loop_reg, 8, 0, i_gp_reg_mapping->gp_reg_b, 0 );
  }
  libxsmm_x86_instruction_alu_mem( io_generated_code,
          LIBXSMM_X86_INSTR_MOVQ,
          struct_gp_reg,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          32,
          i_gp_reg_mapping->gp_reg_b,
          1 );
  if ((is_offset_brgemm > 0) || (is_address_brgemm > 0)) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, tmp_reg2, i_gp_reg_mapping->gp_reg_b);
  }

  /* Setup output pointer of B in eltwise struct */
  libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, tmp_reg );
  if (is_brgemm > 0) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, loop_reg, tmp_reg2);
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, tmp_reg2, i_xgemm_desc->n * i_xgemm_desc->k * 4);
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, tmp_reg2, tmp_reg);
  }
  libxsmm_x86_instruction_alu_mem( io_generated_code,
         LIBXSMM_X86_INSTR_MOVQ,
          struct_gp_reg,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          64,
          tmp_reg,
          1 );

  /* Setup B chunk in stack as FP32 (flat) */
  l_mateltwise_desc = libxsmm_meltw_descriptor_init2(&l_meltw_blob, i_in_dtype, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED,
    LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, i_xgemm_desc->k, i_xgemm_desc->n, i_xgemm_desc_orig->ldb * 4, i_xgemm_desc->k, 0, 0,
    0, LIBXSMM_CAST_USHORT(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY), LIBXSMM_MELTW_OPERATION_UNARY);
  libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_mateltwise_kernel_config, l_mateltwise_desc );
  libxsmm_generator_unary_binary_avx512_microkernel( io_generated_code, io_loop_label_tracker, &l_mateltwise_gp_reg_mapping, &l_mateltwise_kernel_config, l_mateltwise_desc );

  /* Adjust A/B ptrs in case of strided brgemm */
  if (is_brgemm > 0) {
    if (is_stride_brgemm > 0) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_mapping->gp_reg_a, i_xgemm_desc_orig->c1);
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_mapping->gp_reg_b, i_xgemm_desc_orig->c2);
    }
    libxsmm_generator_generic_loop_footer_with_idx_inc_reg_bound( io_generated_code, io_loop_label_tracker, loop_reg, 1, bound_reg);
  }

  /* Adjust a/b gp_regs to point to the fp32 tensors in stack */
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
    i_xgemm_desc->c1 = i_xgemm_desc->m * i_xgemm_desc->k * 4;
    i_xgemm_desc->c2 = i_xgemm_desc->n * i_xgemm_desc->k * 4;
  }

  libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
  libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RCX );
  libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R15 );
  libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R14 );
  libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R13 );
  libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R12 );
  libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R11 );
  libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R10 );
  libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R9 );
  libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R8 );
}

LIBXSMM_API_INTERN void libxsmm_generator_gemm_sse_avx_avx2_avx512_kernel( libxsmm_generated_code*        io_generated_code,
                                                                           libxsmm_loop_label_tracker*    io_loop_label_tracker,
                                                                           const libxsmm_gp_reg_mapping*  i_gp_reg_mapping,
                                                                           const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_gemm_descriptor l_xgemm_desc_mod = *i_xgemm_desc;
  libxsmm_gemm_descriptor *l_xgemm_desc = (libxsmm_gemm_descriptor*) &l_xgemm_desc_mod;
  const char *const env_bf8_gemm_via_stack_alloc_tensors = getenv("LIBXSMM_BF8_GEMM_VIA_STACK");
  int bf8_gemm_via_stack_alloc_tensors = 0;
  int hf8_gemm_via_stack_alloc_tensors = 0;

  /* initialize n-blocking */
  unsigned int l_n_count = 0;          /* array counter for blocking arrays */
  unsigned int l_n_done = 0;           /* progress tracker */
  unsigned int l_n_n[2] = {0,0};       /* blocking sizes for blocks */
  unsigned int l_n_N[2] = {0,0};       /* size of blocks */

  unsigned int adjust_A_pf_ptrs = 0;
  unsigned int adjust_B_pf_ptrs = 0;

  /* Local variables used for A transpose case */
  const libxsmm_gemm_descriptor *   l_xgemm_desc_opa;
  libxsmm_gemm_descriptor           l_new_xgemm_desc_opa;
  libxsmm_descriptor_blob           l_meltw_blob;
  libxsmm_mateltwise_kernel_config  l_mateltwise_kernel_config;
  libxsmm_mateltwise_gp_reg_mapping l_mateltwise_gp_reg_mapping;
  unsigned int                      lda_transpose;
  /* Local variables used only for older gemm setup (not LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI) */
  const libxsmm_meltw_descriptor *  l_mateltwise_desc;
  unsigned int                      l_max_n_blocking = 0;
  unsigned int a_in_vnni = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) ? 1 : 0;

  if ( 0 == env_bf8_gemm_via_stack_alloc_tensors ) {
    if (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_INP( l_xgemm_desc->datatype ) ) {
      if (a_in_vnni == 0) {
        bf8_gemm_via_stack_alloc_tensors = 1;
      }
    }
  } else {
    if (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_INP( l_xgemm_desc->datatype ) ) {
      bf8_gemm_via_stack_alloc_tensors = atoi(env_bf8_gemm_via_stack_alloc_tensors);
      if (a_in_vnni == 0) {
        bf8_gemm_via_stack_alloc_tensors = 1;
      }
    }
  }

  if (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GETENUM_INP( l_xgemm_desc->datatype ) ) {
    hf8_gemm_via_stack_alloc_tensors = 1;
  }

  /* @TODO we need to implement a consolidate solution for callee save stuff
   * here we need to handle AMX stuff to allow AMX optimized TPPs to run lower platforms */
  if ( !( (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & l_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & l_xgemm_desc->flags) == 0)) ||
          (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & l_xgemm_desc->flags) != 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & l_xgemm_desc->flags) != 0))    ) ) {
    return;
  }

  /* Make sure we properly adjust A,B prefetch pointers in case of batch-reduce gemm kernel  */
  if (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
    if ( l_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2          ||
         l_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C    ) {
      adjust_A_pf_ptrs = 1;
    }
  }

  /* in case when A needs to be transposed, we need to change temporarily the desciptor dimensions for gemm */
  lda_transpose = l_xgemm_desc->m;
  if (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) {
    if ((LIBXSMM_DATATYPE_F32 == (libxsmm_datatype)(l_xgemm_desc->datatype)) || (LIBXSMM_DATATYPE_F64 == (libxsmm_datatype)(l_xgemm_desc->datatype))) {
      l_new_xgemm_desc_opa = *l_xgemm_desc;
      l_new_xgemm_desc_opa.lda = lda_transpose;
      l_new_xgemm_desc_opa.flags = (unsigned int)((unsigned int)(l_xgemm_desc->flags) & (~LIBXSMM_GEMM_FLAG_TRANS_A));
      l_xgemm_desc_opa = (const libxsmm_gemm_descriptor *) &l_new_xgemm_desc_opa;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  } else {
    l_xgemm_desc_opa = l_xgemm_desc;
  }

  if ((bf8_gemm_via_stack_alloc_tensors > 0) || (hf8_gemm_via_stack_alloc_tensors > 0)) {
    /* Adjust descriptor to perform GEMM with f32 inputs */
    l_xgemm_desc->datatype = LIBXSMM_GETENUM(LIBXSMM_DATATYPE_F32, LIBXSMM_GETENUM_OUT(i_xgemm_desc->datatype));
    l_xgemm_desc->lda = l_xgemm_desc->m;
    l_xgemm_desc->k = i_xgemm_desc->k*4;
    l_xgemm_desc->ldb = l_xgemm_desc->k;
  }

  /* define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config_fullvector( &l_micro_kernel_config, io_generated_code->arch, l_xgemm_desc_opa, 0 );

  /* block according to the number of available registers or given limits */
  l_max_n_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_max_n_blocking( &l_micro_kernel_config, l_xgemm_desc_opa, io_generated_code->arch );
#if 1
  if (3 < l_max_n_blocking)
#endif
  {
    const unsigned int init_m_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_initial_m_blocking( &l_micro_kernel_config, l_xgemm_desc_opa, io_generated_code->arch );
    const unsigned int init_m_blocks = LIBXSMM_UPDIV(init_m_blocking, l_micro_kernel_config.vector_length);
    if ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256) && (io_generated_code->arch < LIBXSMM_X86_AVX512) ) {
      while ((init_m_blocks * l_max_n_blocking + l_max_n_blocking + 1) > l_micro_kernel_config.vector_reg_count) {
        l_max_n_blocking--;
      }
    } else {
      while ((init_m_blocks * l_max_n_blocking + init_m_blocks + 1) > l_micro_kernel_config.vector_reg_count) {
        l_max_n_blocking--;
      }
    }
  }
  if ( l_max_n_blocking == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }
  libxsmm_compute_equalized_blocking( l_xgemm_desc_opa->n, l_max_n_blocking, &(l_n_N[0]), &(l_n_n[0]), &(l_n_N[1]), &(l_n_n[1]) );

  /* check that l_n_N1 is non-zero */
  if ( l_n_N[0] == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* implementing load from struct */
  if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI & l_xgemm_desc_opa->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI) ||
       ((LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI & l_xgemm_desc_opa->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI) ) {
    /* RDI holds the pointer to the strcut, so lets first move this one into R15 */
    libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_help_1 );
    /* A pointer */
    libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                     i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 32, i_gp_reg_mapping->gp_reg_a, 0 );
    /* B pointer */
    libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                     i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 64, i_gp_reg_mapping->gp_reg_b, 0 );
    /* C pointer */
    libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                     i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 96, i_gp_reg_mapping->gp_reg_c, 0 );
    if ( l_xgemm_desc_opa->prefetch != LIBXSMM_GEMM_PREFETCH_NONE ) {
      /* A prefetch pointer */
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 56, i_gp_reg_mapping->gp_reg_a_prefetch, 0 );
      /* B preftech pointer */
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 88, i_gp_reg_mapping->gp_reg_b_prefetch, 0 );
    }
    /* batch reduce count & offsett arrays*/
    if ((l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) || (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET)) {
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 16, i_gp_reg_mapping->gp_reg_reduce_count, 0 );

      if ( l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET ) {
        libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                         i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 40, i_gp_reg_mapping->gp_reg_a_offset, 0 );
        libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                         i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 72, i_gp_reg_mapping->gp_reg_b_offset, 0 );
      }
    }
    /* loading scaling factor for tertenary C */
    if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( l_xgemm_desc_opa->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( l_xgemm_desc_opa->datatype )) ) {
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 112, i_gp_reg_mapping->gp_reg_scf, 0 );
    }
  }

  if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI & l_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI) ||
       (l_micro_kernel_config.vnni_format_C > 0) ) {
    /* For now disable fusion for < AVX archs  */
    if ( (io_generated_code->arch < LIBXSMM_X86_AVX) &&
         ((l_xgemm_desc->meltw_operation != LIBXSMM_MELTW_OPERATION_NONE) ||
          (l_xgemm_desc->eltw_ap_op != LIBXSMM_MELTW_OPERATION_NONE) ||
          (l_xgemm_desc->eltw_bp_op != LIBXSMM_MELTW_OPERATION_NONE) ||
          (l_xgemm_desc->eltw_cp_op != LIBXSMM_MELTW_OPERATION_NONE) ) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
      return;
    }
    /* Illegal ext_abi when precision is not fp32 or bf16 */
    if (!(LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( l_xgemm_desc->datatype ) ||
          LIBXSMM_DATATYPE_BF8  == LIBXSMM_GETENUM_INP( l_xgemm_desc->datatype ) ||
          LIBXSMM_DATATYPE_HF8  == LIBXSMM_GETENUM_INP( l_xgemm_desc->datatype ) ||
          LIBXSMM_DATATYPE_F32  == LIBXSMM_GETENUM_INP( l_xgemm_desc->datatype )) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ILLEGAL_ABI );
      return;
    }
  }

  /* Setting up the stack frame */
  l_micro_kernel_config.bf8_gemm_via_stack_alloc_tensors = bf8_gemm_via_stack_alloc_tensors;
  l_micro_kernel_config.hf8_gemm_via_stack_alloc_tensors = hf8_gemm_via_stack_alloc_tensors;
  libxsmm_generator_gemm_setup_stack_frame( io_generated_code, l_xgemm_desc, i_gp_reg_mapping, &l_micro_kernel_config);

  /* In this case we store C to scratch  */
  if (l_micro_kernel_config.vnni_format_C > 0) {
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANS_EXT_BUF_C, i_gp_reg_mapping->gp_reg_c );
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, i_gp_reg_mapping->gp_reg_c );
    libxsmm_x86_instruction_alu_imm_i64( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mapping->gp_reg_help_1, 32LL * 64LL );
    libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_c);
    l_xgemm_desc->ldc = l_xgemm_desc->m;
    l_new_xgemm_desc_opa.ldc = l_xgemm_desc->m;
  }

  if ( ( l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) && (l_xgemm_desc->m != 0) && (l_xgemm_desc->k != 0) ) {
    /* initializing required register variables for meltwise transform (transpose) */
    unsigned int l_gp_reg_in  = LIBXSMM_X86_GP_REG_R8;
    unsigned int l_gp_reg_out = LIBXSMM_X86_GP_REG_R9;
    unsigned int l_gp_reg_mloop = LIBXSMM_X86_GP_REG_RAX;
    unsigned int l_gp_reg_nloop = LIBXSMM_X86_GP_REG_RDX;
    unsigned int l_gp_reg_mask = LIBXSMM_X86_GP_REG_R10;
    unsigned int l_gp_reg_mask_2 = LIBXSMM_X86_GP_REG_R11;
    unsigned int l_mask_reg_0 = 1;
    unsigned int l_mask_reg_1 = 2;
    unsigned int l_mask_reg_2 = 3;
    unsigned int l_mask_reg_3 = 4;
    unsigned int l_mask_reg_4 = 5;
    unsigned int l_mask_reg_5 = 6;
    unsigned int l_mask_reg_6 = 7;

    /* pushing RDX, RCX, RDI, R8 and R9 to restore them later after transpose */
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RCX );
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDI );
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R8 );
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R9 );

    /* the transpose microkernels called below use r8 for input and r9 for output so they are set here */
    libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RDI, LIBXSMM_X86_GP_REG_R8);
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR, LIBXSMM_X86_GP_REG_R9 );

    /* creating a descriptor for the meltwise transform (transpose) */
    l_mateltwise_desc = libxsmm_meltw_descriptor_init(&l_meltw_blob,
      (libxsmm_datatype)(l_xgemm_desc->datatype), (libxsmm_datatype)(l_xgemm_desc->datatype), /* FIXME: should go away after rebasing, cast would not be needed */
      l_xgemm_desc->k /*m*/, l_xgemm_desc->m /*n*/,
      l_xgemm_desc->lda, l_xgemm_desc->m,
      /*LIBXSMM_CAST_USHORT*/(unsigned short)(l_xgemm_desc->flags), LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, LIBXSMM_MELTW_OPERATION_UNARY);

    /* define mateltwise kernel config */
    libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_mateltwise_kernel_config, l_mateltwise_desc);

    /* define gp register mapping */
    memset(&l_mateltwise_gp_reg_mapping, 0, sizeof(l_mateltwise_gp_reg_mapping));
#if defined(_WIN32) || defined(__CYGWIN__)
    l_mateltwise_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RCX;
#else /* match calling convention on Linux */
    l_mateltwise_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RDI;
#endif

    /* stack management at the start for meltw kernel */
    libxsmm_generator_meltw_setup_stack_frame( io_generated_code, l_mateltwise_desc, &l_mateltwise_gp_reg_mapping, &l_mateltwise_kernel_config);

    /* main transform (transpose) kernel generator call dispatched over supported microkernel ISA */
    if ( LIBXSMM_DATATYPE_F32 == (libxsmm_datatype)(l_xgemm_desc->datatype) ) {
      if ( io_generated_code->arch >= LIBXSMM_X86_AVX512_CORE ) {
          libxsmm_generator_transform_norm_to_normt_32bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                              l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                              l_gp_reg_mask, l_gp_reg_mask_2, l_mask_reg_0, l_mask_reg_1, l_mask_reg_2, l_mask_reg_3,
                                                                              l_mask_reg_4, l_mask_reg_5, l_mask_reg_6,
                                                                              &l_mateltwise_kernel_config, l_mateltwise_desc );
      } else if ( io_generated_code->arch >= LIBXSMM_X86_AVX ) {
        unsigned int l_save_arch = 0;
        if ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256 ) {
          l_save_arch = io_generated_code->arch;
          io_generated_code->arch = LIBXSMM_X86_AVX2;
          libxsmm_generator_mateltwise_update_micro_kernel_config_dtype_aluinstr( io_generated_code, (libxsmm_mateltwise_kernel_config*)&l_mateltwise_kernel_config, (libxsmm_meltw_descriptor*)l_mateltwise_desc);
        }

        libxsmm_generator_transform_norm_to_normt_32bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                         &l_mateltwise_kernel_config, l_mateltwise_desc );

        if ( l_save_arch != 0 ) {
          io_generated_code->arch = l_save_arch;
          libxsmm_generator_mateltwise_update_micro_kernel_config_dtype_aluinstr( io_generated_code, (libxsmm_mateltwise_kernel_config*)&l_mateltwise_kernel_config, (libxsmm_meltw_descriptor*)l_mateltwise_desc);
        }
      } else if ( (io_generated_code->arch >= LIBXSMM_X86_GENERIC) && (io_generated_code->arch < LIBXSMM_X86_AVX) ) {
          libxsmm_generator_transform_norm_to_normt_32bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                           l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                          &l_mateltwise_kernel_config, l_mateltwise_desc );
      }
    } else if ( LIBXSMM_DATATYPE_F64 == (libxsmm_datatype)(l_xgemm_desc->datatype) ) {
      if ( io_generated_code->arch >= LIBXSMM_X86_AVX512_CORE ) {
          libxsmm_generator_transform_norm_to_normt_64bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                              l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                              l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1, l_mask_reg_2, l_mask_reg_3,
                                                                              l_mask_reg_4, l_mask_reg_5,
                                                                              &l_mateltwise_kernel_config, l_mateltwise_desc );
      } else if ( io_generated_code->arch >= LIBXSMM_X86_AVX ) {
        unsigned int l_save_arch = 0;
        if ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256 ) {
          l_save_arch = io_generated_code->arch;
          io_generated_code->arch = LIBXSMM_X86_AVX2;
          libxsmm_generator_mateltwise_update_micro_kernel_config_dtype_aluinstr( io_generated_code, (libxsmm_mateltwise_kernel_config*)&l_mateltwise_kernel_config, (libxsmm_meltw_descriptor*)l_mateltwise_desc);
        }

        libxsmm_generator_transform_norm_to_normt_64bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                        &l_mateltwise_kernel_config, l_mateltwise_desc );

        if ( l_save_arch != 0 ) {
          io_generated_code->arch = l_save_arch;
          libxsmm_generator_mateltwise_update_micro_kernel_config_dtype_aluinstr( io_generated_code, (libxsmm_mateltwise_kernel_config*)&l_mateltwise_kernel_config, (libxsmm_meltw_descriptor*)l_mateltwise_desc);
        }
      } else if ( (io_generated_code->arch >= LIBXSMM_X86_GENERIC) && (io_generated_code->arch < LIBXSMM_X86_AVX) ) {
          libxsmm_generator_transform_norm_to_normt_64bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                           l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                          &l_mateltwise_kernel_config, l_mateltwise_desc );
      }
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    } /* dispatch over datatypes */

    /* stack management at the end for meltw kernel */
    libxsmm_generator_meltw_destroy_stack_frame( io_generated_code, l_mateltwise_desc, &l_mateltwise_kernel_config );

    /* popping back R9, R8, RDI and RDX after transpose */
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R9 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R8 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDI );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RCX );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );

    /* setting RDI (pointer to A) for the gemm code to the transpose on the stack */
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR, LIBXSMM_X86_GP_REG_RDI );
  } /* if A needs to be transposed */

  /* calling gemm kernel with the modified pointer to the first matrix (now trans_a on the stack) should go here */
  /* at this point RDI must point to the first matrix (A or its transpose) in both trans_a = 0 and trans_a = 1 cases */

  /* Now setup A and B tensors in stack as FP32 flat tensors */
  if ((bf8_gemm_via_stack_alloc_tensors > 0) || (hf8_gemm_via_stack_alloc_tensors > 0)) {
    libxsmm_generator_gemm_setup_f8_AB_tensors_to_stack_as_fp32( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc, i_xgemm_desc, (libxsmm_datatype) LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ));
  }

  libxsmm_reset_loop_label_tracker( io_loop_label_tracker );

  /* generate hoisted BF16 emulation mask for AVX512 */
  if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( l_xgemm_desc_opa->datatype )) &&
         ((l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) &&
         (io_generated_code->arch != LIBXSMM_X86_AVX512_CPX) &&
         (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256) &&
         (io_generated_code->arch <= LIBXSMM_X86_ALLFEAT) &&
         (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX)) {
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
    libxsmm_x86_instruction_alu_imm_i64( io_generated_code,  LIBXSMM_X86_INSTR_MOVQ,
                                         i_gp_reg_mapping->gp_reg_help_2, 0xaaaaaaaa );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mapping->gp_reg_help_2, 3 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
  }

  /* generated hoisted helpers for BF8 emulation */
  if ( LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_INP( l_xgemm_desc_opa->datatype ) ) {
    unsigned short bf8_perm_512[32] = { 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31 };
    unsigned short bf8_perm_256[16] = { 0, 2, 4, 6, 8, 10, 12, 14,                                 1, 3, 5, 7, 9, 11, 13, 15 };

    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
                                                         (const unsigned char *) ( (io_generated_code->arch >= LIBXSMM_X86_AVX512) ? bf8_perm_512 : bf8_perm_256 ),
                                                         "my_bf8_perm",
                                                         l_micro_kernel_config.vector_name,
                                                         1 );
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
    libxsmm_x86_instruction_alu_imm_i64( io_generated_code,  LIBXSMM_X86_INSTR_MOVQ,
                                         i_gp_reg_mapping->gp_reg_help_2, 0x2222222222222222 );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVQ_GPR_LD, i_gp_reg_mapping->gp_reg_help_2, 3 );
    libxsmm_x86_instruction_alu_imm_i64( io_generated_code,  LIBXSMM_X86_INSTR_MOVQ,
                                         i_gp_reg_mapping->gp_reg_help_2, 0x4444444444444444 );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVQ_GPR_LD, i_gp_reg_mapping->gp_reg_help_2, 4 );
    libxsmm_x86_instruction_alu_imm_i64( io_generated_code,  LIBXSMM_X86_INSTR_MOVQ,
                                         i_gp_reg_mapping->gp_reg_help_2, 0x8888888888888888 );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVQ_GPR_LD, i_gp_reg_mapping->gp_reg_help_2, 5 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
  }

  /* Load the actual batch-reduce trip count */
  if ((l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        l_micro_kernel_config.alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_reduce_count,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        0,
        i_gp_reg_mapping->gp_reg_reduce_count,
        0 );
  }

  /* apply n_blocking */
  while (l_n_done != (unsigned int)l_xgemm_desc_opa->n) {
    unsigned int l_n_blocking = l_n_n[l_n_count];
    unsigned int l_m_done = 0;
    unsigned int l_m_done_old = 0;
    unsigned int l_m_blocking = 0;

    /* open N loop */
    libxsmm_generator_gemm_header_nloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_n_done, l_n_blocking );

    /* advance N */
    l_n_done += l_n_N[l_n_count];
    l_n_count++;

    /* define the micro kernel code gen properties, especially m-blocking affects the vector instruction length */
    l_m_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_initial_m_blocking( &l_micro_kernel_config, l_xgemm_desc_opa, io_generated_code->arch );

    /* apply m_blocking */
    while (l_m_done != (unsigned int)l_xgemm_desc_opa->m) {
      if ( l_m_blocking == 0 ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
        return;
      }

      l_m_done_old = l_m_done;
      LIBXSMM_ASSERT(0 != l_m_blocking);
      /* coverity[divide_by_zero] */
      l_m_done = l_m_done + (((l_xgemm_desc_opa->m - l_m_done_old) / l_m_blocking) * l_m_blocking);

      if ( (l_m_done != l_m_done_old) && (l_m_done > 0) ) {
        /* when on AVX512, load mask, if needed */
        if ( ( l_micro_kernel_config.use_masking_a_c != 0 ) && ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256 ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
          /* compute the mask count, depends on vlen as block in M */
          unsigned int l_corrected_vlen = l_micro_kernel_config.vector_length;
          unsigned int l_mask_count = l_corrected_vlen - ( l_m_blocking % l_corrected_vlen );

          if ( ( ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( l_xgemm_desc->datatype ) ) ) ||
               ( ( LIBXSMM_DATATYPE_I8   == LIBXSMM_GETENUM_INP( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_I8   == LIBXSMM_GETENUM_OUT( l_xgemm_desc->datatype ) ) ) ||
               ( ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GETENUM_INP( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GETENUM_OUT( l_xgemm_desc->datatype ) ) ) ||
               ( ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GETENUM_INP( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GETENUM_OUT( l_xgemm_desc->datatype ) ) ) ||
               ( ( LIBXSMM_DATATYPE_HF8  == LIBXSMM_GETENUM_INP( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_HF8  == LIBXSMM_GETENUM_OUT( l_xgemm_desc->datatype ) ) ) ||
               ( ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GETENUM_INP( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_HF8  == LIBXSMM_GETENUM_OUT( l_xgemm_desc->datatype ) ) ) ) {
            libxsmm_generator_initialize_avx512_mask( io_generated_code, i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_AVX512_MASK, l_mask_count, (libxsmm_datatype)LIBXSMM_DATATYPE_I32 );
            /* we have to adjust mask count as for now we are using ymm for 16bit and xmm for 8bit */
            if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256 ) && ( io_generated_code->arch < LIBXSMM_X86_AVX512 ) ) {
              l_mask_count = ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( l_xgemm_desc->datatype ) ) ? l_mask_count + 8 : l_mask_count + 24;
            } else {
              l_mask_count = ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( l_xgemm_desc->datatype ) ) ? l_mask_count + 16 : l_mask_count + 48;
            }
            libxsmm_generator_initialize_avx512_mask( io_generated_code, i_gp_reg_mapping->gp_reg_help_1, 2, l_mask_count, (libxsmm_datatype)LIBXSMM_GETENUM_OUT( l_xgemm_desc->datatype) );
          } else {
            libxsmm_generator_initialize_avx512_mask( io_generated_code, i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_AVX512_MASK, l_mask_count, (libxsmm_datatype)LIBXSMM_GETENUM_OUT( l_xgemm_desc->datatype) );
          }
        } else if ( ( l_micro_kernel_config.use_masking_a_c != 0 ) && ( io_generated_code->arch >= LIBXSMM_X86_AVX ) && ( io_generated_code->arch < LIBXSMM_X86_AVX512_VL256 )  ) {
          unsigned int l_corrected_vlen = l_micro_kernel_config.vector_length;
          unsigned int l_mask_count = l_m_blocking % l_corrected_vlen;

          if ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( l_xgemm_desc->datatype )) ||
               (LIBXSMM_DATATYPE_I32 == LIBXSMM_GETENUM_OUT( l_xgemm_desc->datatype )) ||
               (LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_OUT( l_xgemm_desc->datatype ))    ) {
            libxsmm_generator_initialize_avx_mask( io_generated_code, 0, l_mask_count, (libxsmm_datatype)LIBXSMM_GETENUM_OUT( l_xgemm_desc->datatype ) );
          } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( l_xgemm_desc->datatype ) ) {
            libxsmm_generator_initialize_avx_mask( io_generated_code, 0, l_mask_count, LIBXSMM_DATATYPE_I32 );
          } else {
            /* should not happen */
          }
          /* store mask into stack frame */
          libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_AVX2_MASK_PTR, i_gp_reg_mapping->gp_reg_help_1 );
          libxsmm_x86_instruction_vec_move( io_generated_code, l_micro_kernel_config.instruction_set, LIBXSMM_X86_INSTR_VMOVUPS,
                                            i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 'y', 0, 0, 0, 1 );
        }

        libxsmm_generator_gemm_header_mloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_m_done_old, l_m_blocking );
        libxsmm_generator_gemm_load_C( io_generated_code, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc_opa, l_m_blocking, l_n_blocking );

        if ((l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
          if ((l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
            libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_b);
            libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_a);
          }
          /* This is the reduce loop  */
          libxsmm_generator_gemm_header_reduceloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config );
          if (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
            libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_a);
            libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_b);

            if (adjust_A_pf_ptrs) {
              /* coverity[dead_error_line] */
              libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_a_prefetch );
            }
            if (adjust_B_pf_ptrs) {
              libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_b_prefetch );
            }
            /* load to reg_a the proper array based on the reduce loop index  */
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                l_micro_kernel_config.alu_mov_instruction,
                i_gp_reg_mapping->gp_reg_a,
                i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                0,
                i_gp_reg_mapping->gp_reg_a,
                0 );
            /* load to reg_b the proper array based on the reduce loop index  */
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                l_micro_kernel_config.alu_mov_instruction,
                i_gp_reg_mapping->gp_reg_b,
                i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                0,
                i_gp_reg_mapping->gp_reg_b,
                0 );
            if (adjust_A_pf_ptrs) {
              libxsmm_x86_instruction_alu_mem( io_generated_code,
                  l_micro_kernel_config.alu_mov_instruction,
                  i_gp_reg_mapping->gp_reg_a_prefetch,
                  i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                  0,
                  i_gp_reg_mapping->gp_reg_a_prefetch,
                  0 );
            }
            if (adjust_B_pf_ptrs) {
              /* coverity[dead_error_line] */
              libxsmm_x86_instruction_alu_mem( io_generated_code,
                  l_micro_kernel_config.alu_mov_instruction,
                  i_gp_reg_mapping->gp_reg_b_prefetch,
                  i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                  0,
                  i_gp_reg_mapping->gp_reg_b_prefetch,
                  0 );
            }
          } else if (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) {
            libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_a);
            libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_b);
            libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_b);
            libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_a);
            /* Calculate to reg_a the proper address based on the reduce loop index  */
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                l_micro_kernel_config.alu_mov_instruction,
                i_gp_reg_mapping->gp_reg_a_offset,
                i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                0,
                i_gp_reg_mapping->gp_reg_help_0,
                0 );
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a);
            /* Calculate to reg_b the proper address based on the reduce loop index  */
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                l_micro_kernel_config.alu_mov_instruction,
                i_gp_reg_mapping->gp_reg_b_offset,
                i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                0,
                i_gp_reg_mapping->gp_reg_help_0,
                0 );
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_b);
          } else if (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) {
            libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_a);
            libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_b);
            libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_b);
            libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_a);
            /* Calculate to reg_a the proper address based on the reduce loop index  */
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, i_gp_reg_mapping->gp_reg_help_0);
            libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_help_0, l_xgemm_desc_opa->c1);
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a);
            /* Calculate to reg_b the proper address based on the reduce loop index  */
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, i_gp_reg_mapping->gp_reg_help_0);
            libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_help_0, l_xgemm_desc_opa->c2);
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_b);
          }
        }

        libxsmm_generator_gemm_sse_avx_avx2_avx512_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config,
                                                           l_xgemm_desc_opa, l_m_blocking, l_n_blocking );

        if ((l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
          if (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
            if (adjust_B_pf_ptrs) {
              /* coverity[dead_error_begin] */
              libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0);
              libxsmm_x86_instruction_alu_mem( io_generated_code,
                  l_micro_kernel_config.alu_mov_instruction,
                  i_gp_reg_mapping->gp_reg_help_0,
                  i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                  0,
                  i_gp_reg_mapping->gp_reg_b_prefetch,
                  1 );
              libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_b_prefetch);
            }
            if (adjust_A_pf_ptrs) {
              libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0);
              libxsmm_x86_instruction_alu_mem( io_generated_code,
                  l_micro_kernel_config.alu_mov_instruction,
                  i_gp_reg_mapping->gp_reg_help_0,
                  i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                  0,
                  i_gp_reg_mapping->gp_reg_a_prefetch,
                  1 );
              libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a_prefetch);
            }
            /* Pop address of B_array to help_0 and store proper address of B   */
            libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0);
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                l_micro_kernel_config.alu_mov_instruction,
                i_gp_reg_mapping->gp_reg_help_0,
                i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                0,
                i_gp_reg_mapping->gp_reg_b,
                1 );
            /* Move to reg_b the address of B_array  */
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_b);
            /* Pop address of A_array to help_0 and store proper address of A   */
            libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0);
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                l_micro_kernel_config.alu_mov_instruction,
                i_gp_reg_mapping->gp_reg_help_0,
                i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                0,
                i_gp_reg_mapping->gp_reg_a,
                1 );
            /* Move to reg_a the address of A_array  */
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a);
          }
          libxsmm_generator_gemm_footer_reduceloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc_opa);
          if (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) {
            /* Calculate to reg_a the proper A advance form the microkernel */
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                l_micro_kernel_config.alu_mov_instruction,
                i_gp_reg_mapping->gp_reg_a_offset,
                i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                -8,
                i_gp_reg_mapping->gp_reg_help_0,
                0 );
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a);
            /* Calculate to reg_b the proper B advance form the microkernel */
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                l_micro_kernel_config.alu_mov_instruction,
                i_gp_reg_mapping->gp_reg_b_offset,
                i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                -8,
                i_gp_reg_mapping->gp_reg_help_0,
                0 );
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_b);
            /* Consume the last two pushes form the stack */
            libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0);
            libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0);
          }
          if (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) {
            /* Calculate to reg_a the proper A advance form the microkernel */
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_count, i_gp_reg_mapping->gp_reg_help_0);
            libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_help_0, l_xgemm_desc_opa->c1);
            libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, l_xgemm_desc_opa->c1);
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a);
            /* Calculate to reg_b the proper B advance form the microkernel */
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_count, i_gp_reg_mapping->gp_reg_help_0);
            libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_help_0, l_xgemm_desc_opa->c2);
            libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, l_xgemm_desc_opa->c2);
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_b);
            /* Consume the last two pushes form the stack */
            libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0);
            libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0);
          }
        }

        libxsmm_generator_gemm_store_C( io_generated_code, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc_opa, l_m_blocking, l_n_blocking );
        libxsmm_generator_gemm_footer_mloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc_opa, l_m_blocking, l_m_done );
      }

      /* switch to next smaller m_blocking */
      l_m_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_update_m_blocking( &l_micro_kernel_config, l_xgemm_desc_opa, io_generated_code->arch, l_m_blocking );
    }
    libxsmm_generator_gemm_footer_nloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc_opa, l_n_blocking, l_n_done );
  } /* while l_n_done */

  /* In this case we vnni-format C from scratch  */
  if (l_micro_kernel_config.vnni_format_C > 0) {
    l_xgemm_desc->ldc = i_xgemm_desc->ldc;
    l_new_xgemm_desc_opa.ldc = i_xgemm_desc->ldc;
    libxsmm_generator_gemm_vnni_store_C_from_scratch( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc);
  }

  /* destry stack frame */
  libxsmm_generator_gemm_destroy_stack_frame( io_generated_code, l_xgemm_desc, i_gp_reg_mapping, &l_micro_kernel_config );
}

LIBXSMM_API_INTERN void libxsmm_generator_gemm_sse_avx_avx2_avx512_kloop( libxsmm_generated_code*            io_generated_code,
                                                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                           const unsigned int                 i_m_blocking,
                                                                           const unsigned int                 i_n_blocking ) {
  void (*l_generator_microkernel)(libxsmm_generated_code*, const libxsmm_gp_reg_mapping*, const libxsmm_micro_kernel_config*,
                                  const libxsmm_gemm_descriptor*, const unsigned int, const unsigned int, const int);

  /* some hard coded parameters for k-blocking */
  unsigned int l_k_blocking = 0;
  unsigned int l_k_threshold = 0;

  /* calculate m_blocking such that we choose the right AVX512 kernel */
  unsigned int l_m_vector = ( i_m_blocking % i_micro_kernel_config->vector_length  == 0 ) ? i_m_blocking/i_micro_kernel_config->vector_length : (i_m_blocking/i_micro_kernel_config->vector_length)+1;

  /* in case of 1d blocking and KNL/KNM we unroll aggressively */
  /*
   if ( (( io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CLX) || ( io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CPX) ||
         ( io_generated_code->arch == LIBXSMM_X86_AVX512_VL256)
        ) && ( LIBXSMM_DATATYPE_F64 != LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype))
      ) {
    l_k_blocking = 4;
    l_k_threshold = 12;
  } else */
  if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256 ) && ( io_generated_code->arch <= LIBXSMM_X86_AVX512_KNM ) && ( l_m_vector == 1 ) ) {
    l_k_blocking = 16;
    l_k_threshold = 47;
  } else {
    l_k_blocking = 4;
    l_k_threshold = 23;
  }

  /* for BF8 we need to limit the unrolling */
  if (  LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype) || LIBXSMM_DATATYPE_HF8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype) ) {
    l_k_blocking = 2;
    l_k_threshold = 7;
  }

  /* set up architecture dependent compute micro kernel generator */
  if ( io_generated_code->arch < LIBXSMM_TARGET_ARCH_GENERIC ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  } else if ( io_generated_code->arch <= LIBXSMM_X86_SSE42 ) {
    l_generator_microkernel = libxsmm_generator_gemm_sse_microkernel;
  } else if ( io_generated_code->arch == LIBXSMM_X86_AVX ) {
    l_generator_microkernel = libxsmm_generator_gemm_avx_microkernel;
  } else if ( io_generated_code->arch == LIBXSMM_X86_AVX2 || io_generated_code->arch == LIBXSMM_X86_AVX2_ADL ) {
    l_generator_microkernel = libxsmm_generator_gemm_avx2_microkernel;
  } else if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256 ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
    l_generator_microkernel = libxsmm_generator_gemm_avx512_microkernel_nofsdbcst;
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }

  /* apply multiple k_blocking strategies */
  /* 1. we are larger the k_threshold and a multiple of a predefined blocking parameter */
  if ((i_xgemm_desc->k % l_k_blocking) == 0 && (l_k_threshold < (unsigned int)i_xgemm_desc->k)) {
    unsigned int l_k;
    libxsmm_generator_gemm_header_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_m_blocking, l_k_blocking);

    if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256 ) && ( l_m_vector == 1 ) ) {
      if ( io_generated_code->arch != LIBXSMM_X86_AVX512_KNM ) {
        libxsmm_generator_gemm_avx512_microkernel_fsdbcst( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
          i_xgemm_desc, i_n_blocking, l_k_blocking );
      } else {
        libxsmm_generator_gemm_avx512_microkernel_fsdbcst_qfma( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
          i_xgemm_desc, i_n_blocking, l_k_blocking );
      }
    } else {
      for ( l_k = 0; l_k < l_k_blocking; l_k++) {
        l_generator_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
          i_xgemm_desc, i_m_blocking, i_n_blocking, -1);
      }
    }

    libxsmm_generator_gemm_footer_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config,
      i_xgemm_desc, i_m_blocking, i_xgemm_desc->k, 1 );
  } else {
    /* 2. we want to fully unroll below the threshold */
    if ((unsigned int)i_xgemm_desc->k <= l_k_threshold) {
      unsigned int l_k;

      if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256 ) && ( l_m_vector == 1 ) ) {
        if ( io_generated_code->arch != LIBXSMM_X86_AVX512_KNM ) {
          libxsmm_generator_gemm_avx512_microkernel_fsdbcst( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
            i_xgemm_desc, i_n_blocking, (unsigned int)i_xgemm_desc->k );
        } else {
          libxsmm_generator_gemm_avx512_microkernel_fsdbcst_qfma( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
            i_xgemm_desc, i_n_blocking, (unsigned int)i_xgemm_desc->k );
        }
      } else {
        for ( l_k = 0; l_k < (unsigned int)i_xgemm_desc->k; l_k++) {
          l_generator_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
            i_xgemm_desc, i_m_blocking, i_n_blocking, l_k);
        }
      }
      /* 3. we are larger than the threshold but not a multiple of the blocking factor -> largest possible blocking + remainder handling */
    } else {
      unsigned int l_max_blocked_k = ((i_xgemm_desc->k)/l_k_blocking)*l_k_blocking;
      unsigned int l_k;
      int l_b_offset = 0;

      /* we can block as k is large enough */
      if ( l_max_blocked_k > 0 ) {
        libxsmm_generator_gemm_header_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_m_blocking, l_k_blocking);

        if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256 ) && ( l_m_vector == 1 ) ) {
          if ( io_generated_code->arch != LIBXSMM_X86_AVX512_KNM ) {
            libxsmm_generator_gemm_avx512_microkernel_fsdbcst( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
              i_xgemm_desc, i_n_blocking, l_k_blocking );
          } else {
            libxsmm_generator_gemm_avx512_microkernel_fsdbcst_qfma( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
              i_xgemm_desc, i_n_blocking, l_k_blocking );
          }
        } else {
          for ( l_k = 0; l_k < l_k_blocking; l_k++) {
            l_generator_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
              i_xgemm_desc, i_m_blocking, i_n_blocking, -1);
          }
        }

        libxsmm_generator_gemm_footer_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config,
          i_xgemm_desc, i_m_blocking, l_max_blocked_k, 0 );
      }

      /* now we handle the remainder handling */
      if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256 ) && ( l_m_vector == 1 ) ) {
        if ( io_generated_code->arch != LIBXSMM_X86_AVX512_KNM ) {
          libxsmm_generator_gemm_avx512_microkernel_fsdbcst( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
            i_xgemm_desc, i_n_blocking, ((unsigned int)i_xgemm_desc->k) - l_max_blocked_k );
        } else {
          libxsmm_generator_gemm_avx512_microkernel_fsdbcst_qfma( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
            i_xgemm_desc, i_n_blocking, ((unsigned int)i_xgemm_desc->k) - l_max_blocked_k );
        }
      } else {
        for ( l_k = l_max_blocked_k; l_k < (unsigned int)i_xgemm_desc->k; l_k++) {
          l_generator_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
            i_xgemm_desc, i_m_blocking, i_n_blocking, -1);
        }
      }

      /* reset B pointer */
      if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
        l_b_offset = i_xgemm_desc->ldb * i_xgemm_desc->k * i_micro_kernel_config->datatype_size_in;
      } else {
        l_b_offset = i_xgemm_desc->k * i_micro_kernel_config->datatype_size_in;
      }

      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction,
        i_gp_reg_mapping->gp_reg_b, l_b_offset );
    }
  }
}

LIBXSMM_API_INTERN unsigned int libxsmm_generator_gemm_sse_avx_avx2_avx512_get_initial_m_blocking( libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                                                                    const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                                                                    const unsigned int              i_arch ) {
  unsigned int l_use_masking_a_c = 0;
  unsigned int l_m_blocking = 0;

  if ( ( i_arch <= LIBXSMM_X86_SSE42 )           && ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    l_m_blocking = 16;
  } else if ( ( i_arch <= LIBXSMM_X86_SSE42 )    && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    l_m_blocking = 8;
  } else if ( ( i_arch == LIBXSMM_X86_AVX )      && ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 24, 8, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( i_arch == LIBXSMM_X86_AVX )      && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 12, 4, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( i_arch == LIBXSMM_X86_AVX2 || i_arch == LIBXSMM_X86_AVX2_ADL ) && ( ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) ||
                                                                                    ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) ||
                                                                                    ( LIBXSMM_DATATYPE_I32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) )    ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 32, 8, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( i_arch == LIBXSMM_X86_AVX2 || i_arch == LIBXSMM_X86_AVX2_ADL ) && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 16, 4, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( (i_arch == LIBXSMM_X86_AVX512_MIC) || (i_arch == LIBXSMM_X86_AVX512_KNM) ) && ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 16, 16, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( (i_arch == LIBXSMM_X86_AVX512_MIC) || (i_arch == LIBXSMM_X86_AVX512_KNM) ) && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 8, 8, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ((i_arch >= LIBXSMM_X86_AVX512_VL256) && (i_arch < LIBXSMM_X86_AVX512)) && ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 )  ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 8, 8, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ((i_arch >= LIBXSMM_X86_AVX512) && (i_arch <= LIBXSMM_X86_ALLFEAT)) && ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 )  ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 16, 16, &l_m_blocking, &l_use_masking_a_c );
  } else if ( (i_arch == LIBXSMM_X86_AVX512_VL256) && ( ( LIBXSMM_DATATYPE_I8  == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) )  ||
                                                        ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) )      ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 8, 8, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( i_arch >= LIBXSMM_X86_AVX512) && ( i_arch <= LIBXSMM_X86_AVX512_CORE ) &&
                             ( ( LIBXSMM_DATATYPE_I8  == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) )  ||
                               ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) )      ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 16, 16, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( (i_arch >= LIBXSMM_X86_AVX512_VL256 ) && (i_arch < LIBXSMM_X86_AVX512) ) &&
              ( ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) )  ||
                ( LIBXSMM_DATATYPE_I32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) )  ||
                ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0 ) ||
                ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0 ) ||
                ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) ||
                ( LIBXSMM_DATATYPE_HF8  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0 ) ||
                ( LIBXSMM_DATATYPE_HF8  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) ) ) {
    /* Remark switching ti OUT datatype check here to cover BF16 in, Fp32/Int32 out kernel with the same logic */
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 64, 8, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( (i_arch >= LIBXSMM_X86_AVX512_VL256 ) && (i_arch < LIBXSMM_X86_AVX512 ) )
              && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 32, 4, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( i_arch <= LIBXSMM_X86_ALLFEAT ) &&
              ( ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) )  ||
                ( LIBXSMM_DATATYPE_I32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) )  ||
                ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0 ) ||
                ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0 ) ||
                ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 )||
                ( LIBXSMM_DATATYPE_HF8  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0 ) ||
                ( LIBXSMM_DATATYPE_HF8  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) ) ) {
    /* Remark switching ti OUT datatype check here to cover BF16 in, Fp32/Int32 out kernel with the same logic */
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 64, 16, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( i_arch <= LIBXSMM_X86_ALLFEAT ) && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 32, 8, &l_m_blocking, &l_use_masking_a_c );
  } else {
    /* we should never end up here, if we do let the user know */
    /*LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
    return 0;*/
  }

  libxsmm_generator_gemm_init_micro_kernel_config_fullvector( io_micro_kernel_config, i_arch, i_xgemm_desc, l_use_masking_a_c );

  return l_m_blocking;
}

LIBXSMM_API_INTERN unsigned int libxsmm_generator_gemm_sse_avx_avx2_avx512_update_m_blocking( libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                                                               const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                                                               const unsigned int             i_arch,
                                                                                               const unsigned int             i_current_m_blocking ) {
  unsigned int l_use_masking_a_c = 0;
  unsigned int l_m_blocking = i_current_m_blocking;

  if ( ( i_arch <= LIBXSMM_X86_SSE42 ) && ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    if (i_current_m_blocking == 4) {
      l_m_blocking = 1;
      libxsmm_generator_gemm_init_micro_kernel_config_scalar( io_micro_kernel_config, i_arch, i_xgemm_desc, 0 );
    } else if (i_current_m_blocking == 8) {
      l_m_blocking = 4;
    } else if (i_current_m_blocking == 12) {
      l_m_blocking = 8;
    } else if (i_current_m_blocking == 16) {
      l_m_blocking = 12;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( ( i_arch <= LIBXSMM_X86_SSE42 ) && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    if (i_current_m_blocking == 2) {
      l_m_blocking = 1;
      libxsmm_generator_gemm_init_micro_kernel_config_scalar( io_micro_kernel_config, i_arch, i_xgemm_desc, 0 );
    } else if (i_current_m_blocking == 4) {
      l_m_blocking = 2;
    } else if (i_current_m_blocking == 6) {
      l_m_blocking = 4;
    } else if (i_current_m_blocking == 8) {
      l_m_blocking = 6;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( ( i_arch == LIBXSMM_X86_AVX ) && ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 24, 8, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( i_arch == LIBXSMM_X86_AVX ) && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 12, 4, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( i_arch == LIBXSMM_X86_AVX2 || i_arch == LIBXSMM_X86_AVX2_ADL ) &&  ( ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) ||
                                                                                     ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) ||
                                                                                     ( LIBXSMM_DATATYPE_I32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) )    ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 32, 8, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( i_arch == LIBXSMM_X86_AVX2 || i_arch == LIBXSMM_X86_AVX2_ADL ) && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 16, 4, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( (i_arch == LIBXSMM_X86_AVX512_MIC) || (i_arch == LIBXSMM_X86_AVX512_KNM) ) && ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 16, 16, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( (i_arch == LIBXSMM_X86_AVX512_MIC) || (i_arch == LIBXSMM_X86_AVX512_KNM) ) && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 8, 8, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ((i_arch >= LIBXSMM_X86_AVX512_VL256) && (i_arch < LIBXSMM_X86_AVX512)) && ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 )  ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 8, 8, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ((i_arch >= LIBXSMM_X86_AVX512) && (i_arch <= LIBXSMM_X86_ALLFEAT)) && ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 )  ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 16, 16, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( i_arch == LIBXSMM_X86_AVX512_VL256) && ( ( LIBXSMM_DATATYPE_I8  == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) )  ||
                                                         ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) )      ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 8, 8, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( i_arch >= LIBXSMM_X86_AVX512) && ( i_arch <= LIBXSMM_X86_AVX512_CORE ) &&
                                   ( ( LIBXSMM_DATATYPE_I8  == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) )  ||
                                     ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) )      ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 16, 16, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( (i_arch >= LIBXSMM_X86_AVX512_VL256 ) && (i_arch < LIBXSMM_X86_AVX512 ) ) &&
              ( ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) )  ||
                ( LIBXSMM_DATATYPE_I32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) )  ||
                ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0 ) ||
                ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0 ) ||
                ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) ||
                ( LIBXSMM_DATATYPE_HF8  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0 ) ||
                ( LIBXSMM_DATATYPE_HF8  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 64, 8, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( (i_arch >= LIBXSMM_X86_AVX512_VL256 ) && (i_arch < LIBXSMM_X86_AVX512) )
              && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 32, 4, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( i_arch <= LIBXSMM_X86_ALLFEAT ) &&
              ( ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) )  ||
                ( LIBXSMM_DATATYPE_I32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) )  ||
                ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0 ) ||
                ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0 ) ||
                ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) ||
                ( LIBXSMM_DATATYPE_HF8  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0 ) ||
                ( LIBXSMM_DATATYPE_HF8  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 64, 16, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( i_arch <= LIBXSMM_X86_ALLFEAT ) && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 32, 8, &l_m_blocking, &l_use_masking_a_c );
  } else {
    /* we should never end up here, if we do let the user know */
    /*LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
    return 0;*/
  }

  if ( (i_arch >= LIBXSMM_X86_AVX) && (i_arch <= LIBXSMM_X86_ALLFEAT) ) {
    libxsmm_generator_gemm_init_micro_kernel_config_fullvector( io_micro_kernel_config, i_arch, i_xgemm_desc, l_use_masking_a_c );
  }

  return l_m_blocking;
}


LIBXSMM_API_INTERN unsigned int libxsmm_generator_gemm_sse_avx_avx2_avx512_get_max_n_blocking( const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                                                                               const libxsmm_gemm_descriptor*      i_xgemm_desc,
                                                                                               const unsigned int                  i_arch ) {
  if ( i_arch >= LIBXSMM_X86_GENERIC && i_arch < LIBXSMM_X86_AVX512_VL256 ) {
    LIBXSMM_UNUSED(i_micro_kernel_config);
    return 3;
  } else if ( i_arch >= LIBXSMM_X86_AVX512_VL256 && i_arch < LIBXSMM_X86_AVX512 ) {
    if ( ( i_arch == LIBXSMM_X86_AVX512_VL256_CPX ) && ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
      return 28;
    }
    if ( ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ||
         ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
      return 28;
    }
    if ( ( (i_xgemm_desc->flags &  LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) && ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
      return 28;
    }
    return 30;
  } else if ( i_arch >= LIBXSMM_X86_AVX512 && i_arch <= LIBXSMM_X86_ALLFEAT) {
    /* handle KNM qmadd */
    if ( ( i_arch == LIBXSMM_X86_AVX512_KNM ) && ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
      return 28;
    }
    /* handle KNM qvnni */
    if ( ( i_arch == LIBXSMM_X86_AVX512_KNM ) && ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
      return 28;
    }
    /* handle int16 on SKX */
    if ( ( i_arch == LIBXSMM_X86_AVX512_CORE ) && ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
      return 28;
    }
    /* handle int8 on all AVX512 */
    if ( ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
      return 28;
    }
    /* handle bfoat8 on all AVX512 */
    if ( ( LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) || ( LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) ) {
      return 28;
    }
    /* handle bfoat8 on all AVX512 */
    if ( ( LIBXSMM_DATATYPE_HF8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) || ( LIBXSMM_DATATYPE_HF8 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) ) {
      return 28;
    }
     /* handle bf16 */
    if ( ( i_arch < LIBXSMM_X86_AVX512_CPX ) && ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) && ( i_arch != LIBXSMM_X86_AVX512_VL256_CPX ) ) ) {
      return 28;
    }
    if ( ( (i_xgemm_desc->flags &  LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) && ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
      return 28;
    }
    return 30;
  } else {
    /* shouldn’t happen */
  }
  return 0;
}


