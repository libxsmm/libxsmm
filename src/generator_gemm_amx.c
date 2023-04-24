/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
***************************************************`***************************/
#include "generator_gemm_common.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "generator_gemm_amx.h"
#include "generator_common_x86.h"
#include "generator_gemm_amx_microkernel.h"
#include "generator_mateltwise_sse_avx_avx512.h"
#include "generator_mateltwise_unary_binary_avx_avx512.h"
#include "generator_mateltwise_transform_common.h"

#if !defined(LIBXSMM_GENERATOR_GEMM_AMX_JUMP_LABEL_TRACKER_MALLOC)
# define LIBXSMM_GENERATOR_GEMM_AMX_JUMP_LABEL_TRACKER_MALLOC
#endif


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_reduceloop_amx( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_nloop_amx( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const unsigned int                 i_n_blocking) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_nloop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_nloop, i_n_blocking );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_mloop_amx( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const unsigned int                 i_m_blocking ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_mloop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_mloop, i_m_blocking );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_reduceloop_amx( libxsmm_generated_code*             io_generated_code,
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
void libxsmm_generator_gemm_footer_nloop_amx( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    const unsigned int                 i_n_blocking,
    const unsigned int                 i_n_done,
    const unsigned int                 i_m_loop_exists) {

  if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) {
    if (i_micro_kernel_config->vnni_format_C == 0) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c,
        ((long long)i_n_blocking*i_xgemm_desc->ldc*2 /*(i_micro_kernel_config->datatype_size/2)*/) - ((long long)i_xgemm_desc->m * i_m_loop_exists * 2 /*(i_micro_kernel_config->datatype_size/2)*/) );
    } else {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c,
        ((long long)i_n_blocking*i_xgemm_desc->ldc*2 /*(i_micro_kernel_config->datatype_size/2)*/) - ((long long)i_xgemm_desc->m * i_m_loop_exists * 2 * 2 /*(i_micro_kernel_config->datatype_size/2)*/) );
    }
  } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c,
        ((long long)i_n_blocking*i_xgemm_desc->ldc/**(i_micro_kernel_config->datatype_size/4)*/) - ((long long)i_xgemm_desc->m * i_m_loop_exists /** (i_micro_kernel_config->datatype_size/4)*/) );
  } else {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c,
        ((long long)i_n_blocking*i_xgemm_desc->ldc*4/*(i_micro_kernel_config->datatype_size)*/) - ((long long)i_xgemm_desc->m * i_m_loop_exists * 4 /*(i_micro_kernel_config->datatype_size)*/) );
  }

  /* Also adjust eltwise pointers */
  if ((i_micro_kernel_config->fused_relu == 1) || (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) || (i_micro_kernel_config->fused_relu_bwd == 1) || (i_micro_kernel_config->fused_bcolbias == 1) || (i_micro_kernel_config->fused_scolbias == 1) || (i_micro_kernel_config->overwrite_C == 0)) {
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if ((i_micro_kernel_config->fused_relu == 1) && (i_micro_kernel_config->overwrite_C == 1) ) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_n_blocking*i_xgemm_desc->ldcp)/8 - ((long long)i_m_loop_exists * LIBXSMM_UPDIV(i_xgemm_desc->m,8)) );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_n_blocking*i_xgemm_desc->ldc*2/*(i_micro_kernel_config->datatype_size/2)*/) - ((long long)i_xgemm_desc->m * i_m_loop_exists * 2 * 2 /*(i_micro_kernel_config->datatype_size/2)*/)  );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->fused_relu_bwd == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_RELU_BITMASK_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_n_blocking*i_xgemm_desc->ldcp)/8 - ((long long)i_m_loop_exists * LIBXSMM_UPDIV(i_xgemm_desc->m, 8)));
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_RELU_BITMASK_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  /* In this case also advance the output ptr */
  if (i_micro_kernel_config->overwrite_C == 0) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_n_blocking*i_xgemm_desc->ldc*2/*(i_micro_kernel_config->datatype_size/2)*/) - ((long long)i_xgemm_desc->m * i_m_loop_exists * 2 /*(i_micro_kernel_config->datatype_size/2)*/) );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->fused_bcolbias == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_xgemm_desc->m * i_m_loop_exists * 2/*(i_micro_kernel_config->datatype_size/2)*/) );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->fused_scolbias == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_xgemm_desc->m * i_m_loop_exists * 4/*i_micro_kernel_config->datatype_size*/) );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if ((i_micro_kernel_config->fused_relu == 1) || (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) || (i_micro_kernel_config->fused_relu_bwd == 1) || (i_micro_kernel_config->fused_bcolbias == 1) || (i_micro_kernel_config->fused_scolbias == 1) || (i_micro_kernel_config->overwrite_C == 0)) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
  }

#if 0
  if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_CL2 ||
      i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2CL2BL2_VIA_C ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c_prefetch,
        ((long long)i_n_blocking*i_xgemm_desc->ldc*(i_micro_kernel_config->datatype_size)) - ((long long)i_xgemm_desc->m*(i_micro_kernel_config->datatype_size)) );
  }
#endif
  if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
    /* handle trans B */
    int l_b_offset = 0;
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
      l_b_offset = i_n_blocking * i_micro_kernel_config->datatype_size_in /*i_micro_kernel_config->datatype_size*/;
    } else {
      l_b_offset = i_n_blocking * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in/*i_micro_kernel_config->datatype_size*/;
    }
    libxsmm_generator_gemm_header_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config );
    if (i_m_loop_exists) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_a_ptrs,
          i_gp_reg_mapping->gp_reg_reduce_loop, 8,
          0,
          i_gp_reg_mapping->gp_reg_a,
          0 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction,
          i_gp_reg_mapping->gp_reg_a, ((long long)i_xgemm_desc->m*4/*(i_micro_kernel_config->datatype_size)*/) );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_a_ptrs,
          i_gp_reg_mapping->gp_reg_reduce_loop, 8,
          0,
          i_gp_reg_mapping->gp_reg_a,
          1 );
    }

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_b_ptrs,
        i_gp_reg_mapping->gp_reg_reduce_loop, 8,
        0,
        i_gp_reg_mapping->gp_reg_b,
        0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_b, l_b_offset );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_b_ptrs,
        i_gp_reg_mapping->gp_reg_reduce_loop, 8,
        0,
        i_gp_reg_mapping->gp_reg_b,
        1 );
    libxsmm_generator_gemm_footer_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc);
  } else if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
    /* handle trans B */
    int l_b_offset = 0;
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
      l_b_offset = i_n_blocking * i_micro_kernel_config->datatype_size_in /*i_micro_kernel_config->datatype_size*/;
    } else {
      l_b_offset = i_n_blocking * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in /*i_micro_kernel_config->datatype_size*/;
    }

    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_b_base, l_b_offset );

    if (i_micro_kernel_config->norm_to_normT_B_ext_buf == 1) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANS_EXT_BUF_B, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_n_blocking * 2 /*(i_micro_kernel_config->datatype_size/2)*/ );
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANS_EXT_BUF_B, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
    }

    if (i_m_loop_exists) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction,
          i_gp_reg_mapping->gp_reg_a_base, ((long long)i_xgemm_desc->m*4/*(i_micro_kernel_config->datatype_size)*/)/i_micro_kernel_config->sparsity_factor_A );
      if (i_micro_kernel_config->decompress_A == 1) {
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BITMAP_PTR, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_xgemm_desc->m*4/*(i_micro_kernel_config->datatype_size)*/)/16 );
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BITMAP_PTR, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_DECOMPRESS_BUF, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_xgemm_desc->m*4/*(i_micro_kernel_config->datatype_size)*/) );
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_DECOMPRESS_BUF, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      }
    }
  } else {
    /* handle trans B */
    int l_b_offset = 0;
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
      l_b_offset = i_n_blocking * i_micro_kernel_config->datatype_size_in /*i_micro_kernel_config->datatype_size*/;
    } else {
      l_b_offset = i_n_blocking * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in /*i_micro_kernel_config->datatype_size*/;
    }

    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_b, l_b_offset );

    if (i_micro_kernel_config->norm_to_normT_B_ext_buf == 1) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANS_EXT_BUF_B, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_n_blocking * 2 /*(i_micro_kernel_config->datatype_size/2)*/ );
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANS_EXT_BUF_B, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
    }

    if (i_m_loop_exists) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction,
          i_gp_reg_mapping->gp_reg_a, ((long long)i_xgemm_desc->m*4/*(i_micro_kernel_config->datatype_size)*/)/i_micro_kernel_config->sparsity_factor_A );

      if (i_micro_kernel_config->decompress_A == 1) {
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BITMAP_PTR, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_xgemm_desc->m*4/*(i_micro_kernel_config->datatype_size)*/)/16 );
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BITMAP_PTR, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_DECOMPRESS_BUF, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_xgemm_desc->m*4/*(i_micro_kernel_config->datatype_size)*/) );
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_DECOMPRESS_BUF, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      }
       if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2          ||
          i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C    ) {
            libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction,
            i_gp_reg_mapping->gp_reg_a_prefetch, ((long long)i_xgemm_desc->m*4/*(i_micro_kernel_config->datatype_size)*/) );
      }
    }
  }
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_nloop, i_n_done );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_mloop_amx( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    const unsigned int                 i_m_blocking,
    const unsigned int                 i_m_done,
    const unsigned int                 i_k_unrolled ) {
  /* advance C pointer */
  if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) {
    if (i_micro_kernel_config->vnni_format_C == 0) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_c, (long long)i_m_blocking*2/*(i_micro_kernel_config->datatype_size/2)*/ );
    } else {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_c, (long long)i_m_blocking*2*2/*(i_micro_kernel_config->datatype_size/2)*/ );
    }
  } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_c, (long long)i_m_blocking/**(i_micro_kernel_config->datatype_size/4)*/ );
  } else {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_c, (long long)i_m_blocking*4/*(i_micro_kernel_config->datatype_size)*/ );
  }

  /* Also adjust eltwise pointers */
  if ((i_micro_kernel_config->fused_relu == 1) || (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) || (i_micro_kernel_config->fused_relu_bwd == 1) || (i_micro_kernel_config->fused_bcolbias == 1) || (i_micro_kernel_config->fused_scolbias == 1) || (i_micro_kernel_config->overwrite_C == 0)) {
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if ((i_micro_kernel_config->fused_relu == 1) && (i_micro_kernel_config->overwrite_C == 1) ) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, LIBXSMM_UPDIV((long long)i_m_blocking,8) );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_m_blocking*2*2/*(i_micro_kernel_config->datatype_size/2)*/ );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->fused_relu_bwd == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_RELU_BITMASK_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, LIBXSMM_UPDIV((long long)i_m_blocking,8) );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_RELU_BITMASK_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->overwrite_C == 0) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_m_blocking*2/*(i_micro_kernel_config->datatype_size/2)*/ );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->fused_bcolbias == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_m_blocking * 2/*(i_micro_kernel_config->datatype_size/2)*/ );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->fused_scolbias == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_m_blocking * 4 /*i_micro_kernel_config->datatype_size*/ );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if ((i_micro_kernel_config->fused_relu == 1) || (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) || (i_micro_kernel_config->fused_relu_bwd == 1) || (i_micro_kernel_config->fused_bcolbias == 1) || (i_micro_kernel_config->fused_scolbias == 1) || (i_micro_kernel_config->overwrite_C == 0)) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
  }

  /* C prefetch */
#if 0
  if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_CL2 ||
      i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2CL2BL2_VIA_C ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_c_prefetch, (long long)i_m_blocking*4/*(i_micro_kernel_config->datatype_size)*/ );

  }
#endif

  /* B prefetch */
  if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_BL2_VIA_C ||
      i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C ||
      i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD) {
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0 ) {
      unsigned int l_type_scaling;
      if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) ||
          (LIBXSMM_DATATYPE_I16  == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ))    ) {
        l_type_scaling = 2;
      } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
        l_type_scaling = 4;
      } else {
        l_type_scaling = 1;
      }
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_b_prefetch, (long long)i_m_blocking*(4/*i_micro_kernel_config->datatype_size*//l_type_scaling) );
    }
  }

  if (i_k_unrolled == 0) {
    /* A prefetch */
    if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2 ||
        i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C) {
      if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
        if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2 ) {
          libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
          libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
          libxsmm_generator_gemm_header_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config );
          libxsmm_x86_instruction_alu_mem( io_generated_code,
              i_micro_kernel_config->alu_mov_instruction,
              i_gp_reg_mapping->gp_reg_a_prefetch,
              i_gp_reg_mapping->gp_reg_reduce_loop, 8,
              0,
              i_gp_reg_mapping->gp_reg_help_0,
              0 );
          libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0,
              ((long long)i_xgemm_desc->k * i_micro_kernel_config->datatype_size_in /*(i_micro_kernel_config->datatype_size)*/ * i_xgemm_desc->lda ) -
              ((long long)i_m_blocking * 4 /*(i_micro_kernel_config->datatype_size)*/) );
          libxsmm_x86_instruction_alu_mem( io_generated_code,
              i_micro_kernel_config->alu_mov_instruction,
              i_gp_reg_mapping->gp_reg_a_prefetch,
              i_gp_reg_mapping->gp_reg_reduce_loop, 8,
              0,
              i_gp_reg_mapping->gp_reg_help_0,
              1 );
          libxsmm_generator_gemm_footer_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc);
          libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
          libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
        }
      } else if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
        /* TODO: Add prefetching handling */
      } else {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_a_prefetch,
            ((long long)i_xgemm_desc->k * i_micro_kernel_config->datatype_size_in/*(i_micro_kernel_config->datatype_size)*/ * i_xgemm_desc->lda ) -
            ((long long)i_m_blocking * 4/*(i_micro_kernel_config->datatype_size)*/) );
      }
    }
    /* advance A pointer */
    if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
      libxsmm_generator_gemm_header_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_a,
          i_gp_reg_mapping->gp_reg_reduce_loop, 8,
          0,
          i_gp_reg_mapping->gp_reg_help_0,
          0 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0,
          ((long long)i_xgemm_desc->k * i_micro_kernel_config->datatype_size_in/*(i_micro_kernel_config->datatype_size)*/ * i_xgemm_desc->lda ) - ((long long)i_m_blocking * 4/*(i_micro_kernel_config->datatype_size)*/) );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_a,
          i_gp_reg_mapping->gp_reg_reduce_loop, 8,
          0,
          i_gp_reg_mapping->gp_reg_help_0,
          1 );
      libxsmm_generator_gemm_footer_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc);
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
    } else if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_a_base,
          0LL - ((long long)i_m_blocking * 4 /*(i_micro_kernel_config->datatype_size)*/) );
    } else {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_a,
          ((long long)i_xgemm_desc->k * i_micro_kernel_config->datatype_size_in /*(i_micro_kernel_config->datatype_size)*/ * i_xgemm_desc->lda ) - ((long long)i_m_blocking * 4 /*(i_micro_kernel_config->datatype_size)*/) );
    }
  } else {
    /* A prefetch */
    if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2 ||
        i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C) {
      if ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS ) {
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
        libxsmm_generator_gemm_header_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config );
        libxsmm_x86_instruction_alu_mem( io_generated_code,
            i_micro_kernel_config->alu_mov_instruction,
            i_gp_reg_mapping->gp_reg_a_prefetch,
            i_gp_reg_mapping->gp_reg_reduce_loop, 8,
            0,
            i_gp_reg_mapping->gp_reg_a,
            0 );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a,
            ((long long)i_m_blocking * 4/*(i_micro_kernel_config->datatype_size)*/) );
        libxsmm_x86_instruction_alu_mem( io_generated_code,
            i_micro_kernel_config->alu_mov_instruction,
            i_gp_reg_mapping->gp_reg_a_prefetch,
            i_gp_reg_mapping->gp_reg_reduce_loop, 8,
            0,
            i_gp_reg_mapping->gp_reg_a,
            1 );
        libxsmm_generator_gemm_footer_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc);
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      } else if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
        /* TODO: Add prefetching handling */
      } else {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a_prefetch,
            ((long long)i_m_blocking * 4/*(i_micro_kernel_config->datatype_size)*/) );
      }
    }

    if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
      libxsmm_generator_gemm_header_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_a_ptrs,
          i_gp_reg_mapping->gp_reg_reduce_loop, 8,
          0,
          i_gp_reg_mapping->gp_reg_a,
          0 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a,
          ((long long)i_m_blocking * 4/*(i_micro_kernel_config->datatype_size)*/) );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_a_ptrs,
          i_gp_reg_mapping->gp_reg_reduce_loop, 8,
          0,
          i_gp_reg_mapping->gp_reg_a,
          1 );
      libxsmm_generator_gemm_footer_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc);
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
    } else if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
      /* advance A pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a_base,
          ((long long)i_m_blocking * 4/*(i_micro_kernel_config->datatype_size)*/)/i_micro_kernel_config->sparsity_factor_A );
      if (i_micro_kernel_config->decompress_A == 1) {
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BITMAP_PTR, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_m_blocking * 4/*(i_micro_kernel_config->datatype_size)*/)/16 );
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BITMAP_PTR, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_DECOMPRESS_BUF, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_m_blocking * 4/*(i_micro_kernel_config->datatype_size)*/) );
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_DECOMPRESS_BUF, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      }
    } else {
      /* advance A pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a,
          ((long long)i_m_blocking * 4/*(i_micro_kernel_config->datatype_size)*/)/i_micro_kernel_config->sparsity_factor_A );
      if (i_micro_kernel_config->decompress_A == 1) {
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BITMAP_PTR, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_m_blocking * 4/*(i_micro_kernel_config->datatype_size)*/)/16 );
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BITMAP_PTR, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_DECOMPRESS_BUF, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_m_blocking * 4/*(i_micro_kernel_config->datatype_size)*/) );
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_DECOMPRESS_BUF, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      }
    }
  }

  /* loop handling */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_mloop, i_m_done );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_load_C_amx( libxsmm_generated_code*            io_generated_code,
    libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    libxsmm_blocking_info_t*           n_blocking_info,
    libxsmm_blocking_info_t*           m_blocking_info ) {

  int im, in, acc_id = 0, i_n_offset, i_m_offset, i_m_offset_bias = 0, zmm_reg = 0;
  int vbias_reg = 31;
  int m_tiles = m_blocking_info->tiles;
  int n_tiles = n_blocking_info->tiles;
  unsigned int l_enforce_Mx1_amx_tile_blocking = libxsmm_cpuid_x86_amx_gemm_enforce_mx1_tile_blocking();
  unsigned int col = 0;
  unsigned int gp_reg_bias = (i_micro_kernel_config->m_loop_exists == 0) ? i_gp_reg_mapping->gp_reg_help_0 : i_gp_reg_mapping->gp_reg_help_1;

  if ((0 == (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) &&
      !((LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype )) )) { /* Beta=1 */
    /* Check if we have to fuse colbias bcast */
    if ((i_micro_kernel_config->fused_bcolbias == 1) || (i_micro_kernel_config->fused_scolbias == 1)) {
      gp_reg_bias = i_gp_reg_mapping->gp_reg_lda;
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, gp_reg_bias );
    }
    if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype )) {
      unsigned int gp_reg_gemm_scratch = (i_micro_kernel_config->n_loop_exists == 0) ? i_gp_reg_mapping->gp_reg_help_1 : i_gp_reg_mapping->gp_reg_help_0;
      /* Check if we have to save the tmp registers */
      if ( (gp_reg_gemm_scratch == i_gp_reg_mapping->gp_reg_help_0) && (i_micro_kernel_config->m_loop_exists == 1)  ) {
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      }
      libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_ldc, ((long long)i_micro_kernel_config->gemm_scratch_ld * 4/*l_micro_kernel_config.datatype_size*/)/4);
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, gp_reg_gemm_scratch );
      i_m_offset = 0;
      i_m_offset_bias = 0;
      for (im = 0; im < m_tiles; im++) {
        i_n_offset = 0;
        if (i_micro_kernel_config->fused_bcolbias == 1) {
          /* load 16 bit values into ymm portion of the register */
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVDQU16,
              gp_reg_bias,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              i_m_offset_bias * 2/*(i_micro_kernel_config->datatype_size/2)*/,
              'y',
              vbias_reg, (im == m_tiles-1) ? i_micro_kernel_config->mask_m_fp32 : 0, 1, 0 );
          /* convert 16 bit values into 32 bit (integer convert) */
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
              LIBXSMM_X86_INSTR_VPMOVSXWD,
              i_micro_kernel_config->vector_name,
              vbias_reg, vbias_reg );
          /* shift 16 bits to the left to generate valid FP32 numbers */
          libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code,
              LIBXSMM_X86_INSTR_VPSLLD_I,
              i_micro_kernel_config->vector_name,
              vbias_reg,
              vbias_reg,
              16);
          i_m_offset_bias += m_blocking_info->sizes[im];
        }
        if (i_micro_kernel_config->fused_scolbias == 1) {
          /* load FP32 bias */
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVUPS,
              gp_reg_bias,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              i_m_offset_bias * 4,
              'z',
              vbias_reg, (im == m_tiles-1) ? i_micro_kernel_config->mask_m_fp32 : 0, 1, 0 );
          i_m_offset_bias += m_blocking_info->sizes[im];
        }

        for (in = 0; in < n_tiles; in++) {
          /* Now for all the columns in the tile, upconvert them to F32 from BF16 */
          for (col = 0; col < n_blocking_info->sizes[in]; col++) {
            zmm_reg = (col % 4) + i_micro_kernel_config->reserved_zmms;  /* we do mod 4 as are otherwise running out ymms */
            /* load 16 bit values into ymm portion of the register */
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVDQU16,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ( ((i_n_offset+col) * i_xgemm_desc->ldc) + i_m_offset) * 2/*(i_micro_kernel_config->datatype_size/2)*/,
                'y',
                zmm_reg, (im == m_tiles-1) ? i_micro_kernel_config->mask_m_fp32 : 0, 1, 0 );
            /* convert 16 bit values into 32 bit (integer convert) */
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                LIBXSMM_X86_INSTR_VPMOVSXWD,
                i_micro_kernel_config->vector_name,
                zmm_reg, zmm_reg );
            /* shift 16 bits to the left to generate valid FP32 numbers */
            libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code,
                LIBXSMM_X86_INSTR_VPSLLD_I,
                i_micro_kernel_config->vector_name,
                zmm_reg,
                zmm_reg,
                16);
            if ((i_micro_kernel_config->fused_bcolbias == 1) || (i_micro_kernel_config->fused_scolbias == 1)) {
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VADDPS, 'z', zmm_reg, vbias_reg, zmm_reg );
            }
            /* Store upconverted column to GEMM scratch */
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVUPS,
                gp_reg_gemm_scratch,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((i_n_offset+col) * i_micro_kernel_config->gemm_scratch_ld + i_m_offset) * 4/*i_micro_kernel_config->datatype_size*/,
                i_micro_kernel_config->vector_name,
                zmm_reg, (im == m_tiles-1) ? i_micro_kernel_config->mask_m_fp32 : 0, 0, 1 );
          }
          /* Move zmm registers stored in GEMM scratch to the proper tile */

          libxsmm_x86_instruction_tile_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_TILELOADD,
              gp_reg_gemm_scratch,
              i_gp_reg_mapping->gp_reg_ldc,
              4,
              (i_n_offset * i_micro_kernel_config->gemm_scratch_ld + i_m_offset) * 4/*i_micro_kernel_config->datatype_size*/,
              acc_id);
          acc_id++;
          if ((n_tiles == 1) && (l_enforce_Mx1_amx_tile_blocking == 0)) {
            acc_id++;
          }
          i_n_offset += n_blocking_info->sizes[in];
        }
        i_m_offset += m_blocking_info->sizes[im];
      }
      /* Check if we have to restore the tmp registers */
      libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_ldc, ((long long)i_xgemm_desc->ldc * 4/*l_micro_kernel_config.datatype_size*/)/4);
      if ( (gp_reg_gemm_scratch == i_gp_reg_mapping->gp_reg_help_0) && (i_micro_kernel_config->m_loop_exists == 1)  ) {
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      }
    } else {
      i_m_offset = 0;
      i_m_offset_bias = 0;
      for (im = 0; im < m_tiles; im++) {
        i_n_offset = 0;
        if (i_micro_kernel_config->fused_bcolbias == 1) {
          /* load 16 bit values into ymm portion of the register */
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVDQU16,
              gp_reg_bias,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              i_m_offset_bias * 2/*(i_micro_kernel_config->datatype_size/2)*/,
              'y',
              vbias_reg, (im == m_tiles-1) ? i_micro_kernel_config->mask_m_fp32 : 0, 1, 0 );
          /* convert 16 bit values into 32 bit (integer convert) */
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
              LIBXSMM_X86_INSTR_VPMOVSXWD,
              i_micro_kernel_config->vector_name,
              vbias_reg, vbias_reg );
          /* shift 16 bits to the left to generate valid FP32 numbers */
          libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code,
              LIBXSMM_X86_INSTR_VPSLLD_I,
              i_micro_kernel_config->vector_name,
              vbias_reg,
              vbias_reg,
              16);
          i_m_offset_bias += m_blocking_info->sizes[im];
        }
        if (i_micro_kernel_config->fused_scolbias == 1) {
          /* load FP32 bias */
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVUPS,
              gp_reg_bias,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              i_m_offset_bias * 4,
              'z',
              vbias_reg, (im == m_tiles-1) ? i_micro_kernel_config->mask_m_fp32 : 0, 1, 0 );
          i_m_offset_bias += m_blocking_info->sizes[im];
        }
        for (in = 0; in < n_tiles; in++) {
          if ((i_micro_kernel_config->fused_bcolbias == 1) || (i_micro_kernel_config->fused_scolbias == 1)) {
            for (col = 0; col < n_blocking_info->sizes[in]; col++) {
              zmm_reg = (col % 16) + i_micro_kernel_config->reserved_zmms;
              libxsmm_x86_instruction_vec_move( io_generated_code,
                  i_micro_kernel_config->instruction_set,
                  LIBXSMM_X86_INSTR_VMOVUPS,
                  i_gp_reg_mapping->gp_reg_c,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  ((i_n_offset+col) * i_xgemm_desc->ldc + i_m_offset) * 4/*i_micro_kernel_config->datatype_size*/,
                  'z',
                  zmm_reg, (im == m_tiles-1) ? i_micro_kernel_config->mask_m_fp32 : 0, 1, 0 );
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VADDPS, 'z', zmm_reg, vbias_reg, zmm_reg );
              libxsmm_x86_instruction_vec_move( io_generated_code,
                  i_micro_kernel_config->instruction_set,
                  LIBXSMM_X86_INSTR_VMOVUPS,
                  i_gp_reg_mapping->gp_reg_c,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  ((i_n_offset+col) * i_xgemm_desc->ldc + i_m_offset) * 4/*i_micro_kernel_config->datatype_size*/,
                  'z',
                  zmm_reg, (im == m_tiles-1) ? i_micro_kernel_config->mask_m_fp32 : 0, 0, 1 );
            }
          }
          libxsmm_x86_instruction_tile_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_TILELOADD,
              i_gp_reg_mapping->gp_reg_c,
              i_gp_reg_mapping->gp_reg_ldc,
              4,
              (i_n_offset * i_xgemm_desc->ldc + i_m_offset) * 4/*i_micro_kernel_config->datatype_size*/,
              acc_id);

          acc_id++;
          if ((n_tiles == 1) && (l_enforce_Mx1_amx_tile_blocking == 0)) {
            acc_id++;
          }
          i_n_offset += n_blocking_info->sizes[in];
        }
        i_m_offset += m_blocking_info->sizes[im];
      }
    }
    if ((i_micro_kernel_config->fused_bcolbias == 1) || (i_micro_kernel_config->fused_scolbias == 1)) {
      libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_lda, ((long long)i_xgemm_desc->lda * 4/*l_micro_kernel_config.datatype_size*/)/4);
    }
  } else { /* Beta=0 */
    /* Check if we have to fuse colbias bcast */
    if (((i_micro_kernel_config->fused_bcolbias == 1) || (i_micro_kernel_config->fused_scolbias == 1)) &&
        !((LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype )) )) {

      if (i_micro_kernel_config->fused_scolbias == 1) {
        if ( (gp_reg_bias == i_gp_reg_mapping->gp_reg_help_1) && (i_micro_kernel_config->n_loop_exists == 1)  ) {
          libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
        }
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, gp_reg_bias );
        /* Set gp_reg_ldc to 0 in order to broadcast the bias */
        libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_ldc, 0);
        i_m_offset = 0;
        for (im = 0; im < m_tiles; im++) {
          i_n_offset = 0;
          for (in = 0; in < n_tiles; in++) {
            libxsmm_x86_instruction_tile_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_TILELOADD,
                gp_reg_bias,
                i_gp_reg_mapping->gp_reg_ldc,
                4,
                i_m_offset * 4/*i_micro_kernel_config->datatype_size*/,
                acc_id);

            acc_id++;
            if ((n_tiles == 1) && (l_enforce_Mx1_amx_tile_blocking == 0)) {
              acc_id++;
            }
            i_n_offset += n_blocking_info->sizes[in];
          }
          i_m_offset += m_blocking_info->sizes[im];
        }

        if ( (gp_reg_bias == i_gp_reg_mapping->gp_reg_help_1) && (i_micro_kernel_config->n_loop_exists == 1)  ) {
          libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
        }
        /* Restore gp_reg_ldc to proper value */
        libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_ldc, ((long long)i_xgemm_desc->ldc * 4/*i_micro_kernel_config->datatype_size*/)/4);

      } else if (i_micro_kernel_config->fused_bcolbias == 1) {

        unsigned int gp_reg_gemm_scratch = (i_micro_kernel_config->m_loop_exists == 0) ? i_gp_reg_mapping->gp_reg_help_1 : i_gp_reg_mapping->gp_reg_help_0;

        if ( (gp_reg_bias == i_gp_reg_mapping->gp_reg_help_1) && (i_micro_kernel_config->n_loop_exists == 1)  ) {
          libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
        }
        if ( (gp_reg_bias == i_gp_reg_mapping->gp_reg_help_0) && (i_micro_kernel_config->m_loop_exists == 1)  ) {
          libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
        }
        if ( (gp_reg_gemm_scratch == i_gp_reg_mapping->gp_reg_help_1) && (i_micro_kernel_config->n_loop_exists == 1)  ) {
          libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
        }
        if ( (gp_reg_gemm_scratch == i_gp_reg_mapping->gp_reg_help_0) && (i_micro_kernel_config->m_loop_exists == 1)  ) {
          libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
        }

        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, gp_reg_bias );
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, gp_reg_gemm_scratch );

        /* Upconvert bf16 bias to GEMM scratch */
        i_m_offset = 0;
        for (im = 0; im < m_tiles; im++) {
          zmm_reg = (im % (16-i_micro_kernel_config->reserved_zmms)) + i_micro_kernel_config->reserved_zmms;
          /* load 16 bit values into ymm portion of the register */
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVDQU16,
              gp_reg_bias,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              i_m_offset * 2/*(i_micro_kernel_config->datatype_size/2)*/,
              'y',
              zmm_reg, (im == m_tiles-1) ? i_micro_kernel_config->mask_m_fp32 : 0, 1, 0 );
          /* convert 16 bit values into 32 bit (integer convert) */
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
              LIBXSMM_X86_INSTR_VPMOVSXWD,
              i_micro_kernel_config->vector_name,
              zmm_reg, zmm_reg );
          /* shift 16 bits to the left to generate valid FP32 numbers */
          libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code,
              LIBXSMM_X86_INSTR_VPSLLD_I,
              i_micro_kernel_config->vector_name,
              zmm_reg,
              zmm_reg,
              16);
          /* Store upconverted column to GEMM scratch */
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVUPS,
              gp_reg_gemm_scratch,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              i_m_offset * 4/*i_micro_kernel_config->datatype_size*/,
              i_micro_kernel_config->vector_name,
              zmm_reg, (im == m_tiles-1) ? i_micro_kernel_config->mask_m_fp32 : 0, 0, 1 );

          i_m_offset += m_blocking_info->sizes[im];
        }

        /* Set gp_reg_ldc to 0 in order to broadcast the bias */
        libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_ldc, 0);
        i_m_offset = 0;
        for (im = 0; im < m_tiles; im++) {
          i_n_offset = 0;
          for (in = 0; in < n_tiles; in++) {
            libxsmm_x86_instruction_tile_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_TILELOADD,
                gp_reg_gemm_scratch,
                i_gp_reg_mapping->gp_reg_ldc,
                4,
                i_m_offset * 4/*i_micro_kernel_config->datatype_size*/,
                acc_id);

            acc_id++;
            if ((n_tiles == 1) && (l_enforce_Mx1_amx_tile_blocking == 0)) {
              acc_id++;
            }
            i_n_offset += n_blocking_info->sizes[in];
          }
          i_m_offset += m_blocking_info->sizes[im];
        }

        if ( (gp_reg_gemm_scratch == i_gp_reg_mapping->gp_reg_help_1) && (i_micro_kernel_config->n_loop_exists == 1)  ) {
          libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
        }
        if ( (gp_reg_gemm_scratch == i_gp_reg_mapping->gp_reg_help_0) && (i_micro_kernel_config->m_loop_exists == 1)  ) {
          libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
        }
        if ( (gp_reg_bias == i_gp_reg_mapping->gp_reg_help_1) && (i_micro_kernel_config->n_loop_exists == 1)  ) {
          libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
        }
        if ( (gp_reg_bias == i_gp_reg_mapping->gp_reg_help_0) && (i_micro_kernel_config->m_loop_exists == 1)  ) {
          libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
        }

        /* Restore gp_reg_ldc to proper value */
        libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_ldc, ((long long)i_xgemm_desc->ldc * 4/*i_micro_kernel_config->datatype_size*/)/4);
      }

    } else {
      for (im = 0; im < m_tiles; im++) {
        for (in = 0; in < n_tiles; in++) {
          libxsmm_x86_instruction_tile_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_TILEZERO,
              LIBXSMM_X86_GP_REG_UNDEF,
              LIBXSMM_X86_GP_REG_UNDEF,
              0,
              0,
              acc_id);
          acc_id++;
          if ((n_tiles == 1) && (l_enforce_Mx1_amx_tile_blocking == 0)) {
            acc_id++;
          }
        }
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_store_C_amx( libxsmm_generated_code*            io_generated_code,
    libxsmm_gp_reg_mapping*            i_gp_reg_mapping,
    libxsmm_micro_kernel_config*       i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    libxsmm_blocking_info_t*           n_blocking_info,
    libxsmm_blocking_info_t*           m_blocking_info ) {

  int m_tiles = m_blocking_info->tiles;
  int n_tiles = n_blocking_info->tiles;
  int _C_tile_done[4] = { 0 };
  int i, im, in;

  for (i = 0; i < m_tiles*n_tiles; i++) {
    im = i_micro_kernel_config->_im[i];
    in = i_micro_kernel_config->_in[i];
    _C_tile_done[i_micro_kernel_config->_C_tile_id[i]] = 1;
    if (i_micro_kernel_config->use_paired_tilestores == 1) {
      /* If mate C tile is also ready, then two paired tilestore */
      if (_C_tile_done[i_micro_kernel_config->_C_tile_mate_id[i_micro_kernel_config->_C_tile_id[i]]] == 1) {
        int min_mate_C_id = (i_micro_kernel_config->_C_tile_id[i] < i_micro_kernel_config->_C_tile_mate_id[i_micro_kernel_config->_C_tile_id[i]]) ? i_micro_kernel_config->_C_tile_id[i] : i_micro_kernel_config->_C_tile_mate_id[i_micro_kernel_config->_C_tile_id[i]];
        int im_store = min_mate_C_id / n_tiles;
        int in_store = min_mate_C_id % n_tiles;
        libxsmm_generator_gemm_amx_paired_tilestore( io_generated_code,
            i_gp_reg_mapping,
            i_micro_kernel_config,
            i_xgemm_desc,
            min_mate_C_id,
            i_micro_kernel_config->_C_tile_mate_id[min_mate_C_id],
            i_micro_kernel_config->_im_offset_prefix_sums[im_store],
            i_micro_kernel_config->_in_offset_prefix_sums[in_store],
            n_blocking_info->sizes[in_store]);
      }
    } else {
      libxsmm_generator_gemm_amx_single_tilestore( io_generated_code,
          i_gp_reg_mapping,
          i_micro_kernel_config,
          i_xgemm_desc,
          i_micro_kernel_config->_C_tile_id[i],
          i_micro_kernel_config->_im_offset_prefix_sums[im],
          i_micro_kernel_config->_in_offset_prefix_sums[in],
          n_blocking_info->sizes[in]);
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_setup_tile( unsigned int tile_id, unsigned int n_rows, unsigned int n_cols, libxsmm_tile_config *tc) {
  switch (tile_id) {
    case 0:
      tc->tile0rowsb = (unsigned short)(n_rows * 4);
      tc->tile0cols  = (unsigned char)n_cols;
      break;
    case 1:
      tc->tile1rowsb = (unsigned short)(n_rows * 4);
      tc->tile1cols  = (unsigned char)n_cols;
      break;
    case 2:
      tc->tile2rowsb = (unsigned short)(n_rows * 4);
      tc->tile2cols  = (unsigned char)n_cols;
      break;
    case 3:
      tc->tile3rowsb = (unsigned short)(n_rows * 4);
      tc->tile3cols  = (unsigned char)n_cols;
      break;
    case 4:
      tc->tile4rowsb = (unsigned short)(n_rows * 4);
      tc->tile4cols  = (unsigned char)n_cols;
      break;
    case 5:
      tc->tile5rowsb = (unsigned short)(n_rows * 4);
      tc->tile5cols  = (unsigned char)n_cols;
      break;
    case 6:
      tc->tile6rowsb = (unsigned short)(n_rows * 4);
      tc->tile6cols  = (unsigned char)n_cols;
      break;
    case 7:
      tc->tile7rowsb = (unsigned short)(n_rows * 4);
      tc->tile7cols  = (unsigned char)n_cols;
      break;
    default:
      LIBXSMM_ASSERT_MSG(0, "valid tile id");
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_setup_masking_infra( libxsmm_generated_code* io_generated_code, libxsmm_micro_kernel_config*  i_micro_kernel_config ) {

  unsigned int reserved_mask_regs = i_micro_kernel_config->reserved_mask_regs;
  unsigned int temp_reg = LIBXSMM_X86_GP_REG_R10;
  i_micro_kernel_config->mask_m_fp32        = 0;
  i_micro_kernel_config->mask_m_bf16        = 0;

  if (i_micro_kernel_config->m_remainder > 0) {
    reserved_mask_regs  += 2;
    i_micro_kernel_config->mask_m_fp32  = reserved_mask_regs - 1;
    i_micro_kernel_config->mask_m_bf16  = reserved_mask_regs - 2;
    libxsmm_generator_initialize_avx512_mask( io_generated_code, temp_reg, i_micro_kernel_config->mask_m_fp32, 16 - i_micro_kernel_config->m_remainder, LIBXSMM_DATATYPE_F32);
    libxsmm_generator_initialize_avx512_mask( io_generated_code, temp_reg, i_micro_kernel_config->mask_m_bf16, 16 - i_micro_kernel_config->m_remainder, LIBXSMM_DATATYPE_BF16);
  }
  i_micro_kernel_config->reserved_mask_regs = reserved_mask_regs;
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_setup_fusion_infra( libxsmm_generated_code*            io_generated_code,
                                                    const libxsmm_gemm_descriptor*      i_xgemm_desc,
                                                    const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
                                                    libxsmm_micro_kernel_config*  i_micro_kernel_config ) {

  unsigned int temp_reg = LIBXSMM_X86_GP_REG_R10;
  unsigned int reserved_zmms      = 0;
  unsigned int reserved_mask_regs = 1;
  int has_scf                     = ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) &&
                                     (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ))) ? 1 : 0;
  LIBXSMM_UNUSED(i_gp_reg_mapping);
  LIBXSMM_UNUSED(i_xgemm_desc);

  if (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) {
    if (i_micro_kernel_config->vnni_format_C == 1) {
      /* For now we support C norm->vnni external only when C is norm */
      fprintf(stderr, "For now we support C norm->vnni to external buffer only when C output is in normal format...\n");
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  }

  if (has_scf > 0) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_INT8_SCF, temp_reg );
    i_micro_kernel_config->scf_vreg = reserved_zmms;
    if ( (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & i_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & i_xgemm_desc->flags) == 0)) ||
        (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & i_xgemm_desc->flags) != 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & i_xgemm_desc->flags) != 0))     ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VBROADCASTSS,
          temp_reg,
          LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
          i_micro_kernel_config->vector_name,
          i_micro_kernel_config->scf_vreg, 0, 1, 0 );
    }
    reserved_zmms++;
    if (0 == (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) {
      i_micro_kernel_config->aux_vreg = reserved_zmms;
      reserved_zmms++;
    }
  }

  /* Setup zmms to be reused throughout the kernel */
  if ((i_micro_kernel_config->fused_relu == 1) || (i_micro_kernel_config->fused_relu_nobitmask == 1) || (i_micro_kernel_config->fused_relu_bwd == 1) ) {
    i_micro_kernel_config->zero_reg = reserved_zmms;
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             LIBXSMM_X86_INSTR_VPXORD,
                                             i_micro_kernel_config->vector_name,
                                             i_micro_kernel_config->zero_reg, i_micro_kernel_config->zero_reg, i_micro_kernel_config->zero_reg );
    reserved_zmms++;
  }

  if (i_micro_kernel_config->vnni_format_C == 1) {
    short vnni_perm_array[32] = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};
    i_micro_kernel_config->vnni_perm_reg = reserved_zmms;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) vnni_perm_array, "vnni_perm_array_", i_micro_kernel_config->vector_name, i_micro_kernel_config->vnni_perm_reg);
    reserved_zmms++;
  }

  if (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) {
    short perm_table_vnni_lo[32] = { 0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47};
    short perm_table_vnni_hi[32] = {16,48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55, 24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63};

    i_micro_kernel_config->perm_table_vnni_lo = reserved_zmms;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) perm_table_vnni_lo, "perm_table_vnni_lo_", i_micro_kernel_config->vector_name, i_micro_kernel_config->perm_table_vnni_lo);
    reserved_zmms++;
    i_micro_kernel_config->perm_table_vnni_hi = reserved_zmms;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) perm_table_vnni_hi, "perm_table_vnni_hi_", i_micro_kernel_config->vector_name, i_micro_kernel_config->perm_table_vnni_hi);
    reserved_zmms++;
  }

  if (i_micro_kernel_config->norm_to_normT_B_ext_buf == 1) {
    reserved_mask_regs  += 2;
    i_micro_kernel_config->norm_to_normT_mask_reg_0  = reserved_mask_regs - 1;
    i_micro_kernel_config->norm_to_normT_mask_reg_1  = reserved_mask_regs - 2;
  }

  if (i_micro_kernel_config->fused_sigmoid == 1) {
    reserved_zmms       += 15;
    reserved_mask_regs  += 2;
    libxsmm_generator_gemm_prepare_coeffs_sigmoid_ps_rational_78_avx_avx512( io_generated_code, i_micro_kernel_config, reserved_zmms, reserved_mask_regs, temp_reg );
  }

  i_micro_kernel_config->reserved_zmms      = reserved_zmms;
  i_micro_kernel_config->reserved_mask_regs = reserved_mask_regs;
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_init_micro_kernel_config_tileblocking(libxsmm_gemm_descriptor*      i_xgemm_desc,
    libxsmm_micro_kernel_config*  i_micro_kernel_config,
    libxsmm_blocking_info_t*      m_blocking_info,
    libxsmm_blocking_info_t*      n_blocking_info,
    libxsmm_tile_config*          tile_config ) {
  unsigned int im = 0, in = 0, m_blocking = 0, n_blocking = 0, k_blocking = 0, ii = 0, m_tiles = 0, n_tiles = 0, l_k_pack_factor = 2;
  unsigned int has_fused_relu_bitmask = ((i_xgemm_desc->eltw_cp_flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) ? 1 : 0;
  unsigned int l_enforce_Mx1_amx_tile_blocking = libxsmm_cpuid_x86_amx_gemm_enforce_mx1_tile_blocking();

  if (l_enforce_Mx1_amx_tile_blocking > 0) {
    i_micro_kernel_config->m_remainder  = 0;
    m_blocking = 64;
    while (i_xgemm_desc->m % m_blocking != 0) {
      m_blocking--;
    }
    if ((i_xgemm_desc->m > 64) && (has_fused_relu_bitmask > 0) && (m_blocking % 16 != 0)) {
      m_blocking = 64;
      while ((i_xgemm_desc->m % m_blocking != 0) || (m_blocking % 16 != 0)) {
        m_blocking--;
      }
    }
    if (m_blocking <= 16) {
      m_blocking_info[0].blocking = m_blocking;
      m_blocking_info[0].block_size = i_xgemm_desc->m;
      m_blocking_info[0].tiles = 1;
      m_blocking_info[0].sizes[0] = m_blocking;
      i_micro_kernel_config->m_remainder  = m_blocking_info[0].sizes[0] % 16;
    } else if (m_blocking <= 32) {
      m_blocking_info[0].blocking = m_blocking;
      m_blocking_info[0].block_size = i_xgemm_desc->m;
      m_blocking_info[0].tiles = 2;
      m_blocking_info[0].sizes[0] = 16;
      m_blocking_info[0].sizes[1] = m_blocking - m_blocking_info[0].sizes[0];
      i_micro_kernel_config->m_remainder  = m_blocking_info[0].sizes[1] % 16;
    } else if (m_blocking <= 48) {
      m_blocking_info[0].blocking = m_blocking;
      m_blocking_info[0].block_size = i_xgemm_desc->m;
      m_blocking_info[0].tiles = 3;
      m_blocking_info[0].sizes[0] = 16;
      m_blocking_info[0].sizes[1] = 16;
      m_blocking_info[0].sizes[2] = m_blocking - 2 * 16;
      i_micro_kernel_config->m_remainder  = m_blocking_info[0].sizes[2] % 16;
    } else {
      m_blocking_info[0].blocking = m_blocking;
      m_blocking_info[0].block_size = i_xgemm_desc->m;
      m_blocking_info[0].tiles = 4;
      m_blocking_info[0].sizes[0] = 16;
      m_blocking_info[0].sizes[1] = 16;
      m_blocking_info[0].sizes[2] = 16;
      m_blocking_info[0].sizes[3] = m_blocking - 3 * 16;
      i_micro_kernel_config->m_remainder  = m_blocking_info[0].sizes[3] % 16;
    }
    n_blocking = 16;
    while (i_xgemm_desc->n % n_blocking != 0) {
      n_blocking--;
    }
    n_blocking_info[0].blocking = n_blocking;
    n_blocking_info[0].block_size = i_xgemm_desc->n;
    n_blocking_info[0].tiles = 1;
    n_blocking_info[0].sizes[0] = n_blocking;
  } else {
    i_micro_kernel_config->m_remainder  = 0;
    m_blocking = 32;
    while (i_xgemm_desc->m % m_blocking != 0) {
      m_blocking--;
    }
    if ((i_xgemm_desc->m > 32) && (has_fused_relu_bitmask > 0) && (m_blocking % 16 != 0)) {
      m_blocking = 32;
      while ((i_xgemm_desc->m % m_blocking != 0) || (m_blocking % 16 != 0)) {
        m_blocking--;
      }
    }

    if (m_blocking <= 16) {
      m_blocking_info[0].blocking = m_blocking;
      m_blocking_info[0].block_size = i_xgemm_desc->m;
      m_blocking_info[0].tiles = 1;
      m_blocking_info[0].sizes[0] = m_blocking;
      i_micro_kernel_config->m_remainder  = m_blocking_info[0].sizes[0] % 16;
    } else {
      m_blocking_info[0].blocking = m_blocking;
      m_blocking_info[0].block_size = i_xgemm_desc->m;
      m_blocking_info[0].tiles = 2;
      m_blocking_info[0].sizes[0] = 16 /*(m_blocking+1)/2*/;
      m_blocking_info[0].sizes[1] = m_blocking - m_blocking_info[0].sizes[0];
      i_micro_kernel_config->m_remainder  = m_blocking_info[0].sizes[1] % 16;
    }

    n_blocking = 32;
    while (i_xgemm_desc->n % n_blocking != 0) {
      n_blocking--;
    }
    if (n_blocking <= 16) {
      n_blocking_info[0].blocking = n_blocking;
      n_blocking_info[0].block_size = i_xgemm_desc->n;
      n_blocking_info[0].tiles = 1;
      n_blocking_info[0].sizes[0] = n_blocking;
    } else {
      n_blocking_info[0].blocking = n_blocking;
      n_blocking_info[0].block_size = i_xgemm_desc->n;
      n_blocking_info[0].tiles = 2;
      n_blocking_info[0].sizes[0] = (n_blocking+1)/2;
      n_blocking_info[0].sizes[1] = n_blocking - n_blocking_info[0].sizes[0];
    }

    /* Special case when N = 49 or N = 61 -- we do 1x4 blocking */
    if (i_xgemm_desc->n == 49 || i_xgemm_desc->n == 61) {
      m_blocking = 16;
      while (i_xgemm_desc->m % m_blocking != 0) {
        m_blocking--;
      }
      m_blocking_info[0].blocking = m_blocking;
      m_blocking_info[0].block_size = i_xgemm_desc->m;
      m_blocking_info[0].tiles = 1;
      m_blocking_info[0].sizes[0] = m_blocking;
      i_micro_kernel_config->m_remainder  = m_blocking_info[0].sizes[0] % 16;
      if (i_xgemm_desc->n == 49) {
        n_blocking_info[0].blocking = 49;
        n_blocking_info[0].block_size = 49;
        n_blocking_info[0].tiles = 4;
        /* I.e. N = 49 = 3 * 13 + 10 */
        n_blocking_info[0].sizes[0] = 13;
        n_blocking_info[0].sizes[1] = 13;
        n_blocking_info[0].sizes[2] = 13;
        n_blocking_info[0].sizes[3] = 10;
      }
      if (i_xgemm_desc->n == 61) {
        n_blocking_info[0].blocking = 61;
        n_blocking_info[0].block_size = 61;
        n_blocking_info[0].tiles = 4;
        /* I.e. N = 61 = 3 * 16 + 13 */
        n_blocking_info[0].sizes[0] = 16;
        n_blocking_info[0].sizes[1] = 16;
        n_blocking_info[0].sizes[2] = 16;
        n_blocking_info[0].sizes[3] = 13;
      }
    }
  }

  /* Find K blocking */
  l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype) );
  if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) {
    k_blocking = 32;
  } else if (LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) {
    k_blocking = 64;
  } else {
    /* should not happen */
    k_blocking = 1; /* surpress div by zero in following while loop */
  }
  while (i_xgemm_desc->k % k_blocking != 0) {
    k_blocking -= l_k_pack_factor;
  }

  /* First init all tiles with default value 16 */
  for (im = 0; im < 8; im++) {
    libxsmm_setup_tile(im, 16, 16, tile_config);
  }

  /* For now introduce here tileconfig redundantly -- want to move it externally somewhere... */
  /* Create array with tileconfig */
  /* TODO: revisit */
  tile_config->palette_id = 1;
  /* First configure the accumulator tiles 0-4 */
  m_tiles = m_blocking_info[0].tiles;
  n_tiles = n_blocking_info[0].tiles;
  ii = 0;
  for (im = 0; im < m_tiles; im++) {
    for (in = 0; in < n_tiles; in++) {
      libxsmm_setup_tile(ii, m_blocking_info[0].sizes[im], n_blocking_info[0].sizes[in], tile_config);
      ii++;
      if ((n_tiles == 1) && (l_enforce_Mx1_amx_tile_blocking == 0)) {
        ii++;
      }
    }
  }
  /* Configure tiles for A */
  libxsmm_setup_tile(4, m_blocking_info[0].sizes[0], k_blocking/l_k_pack_factor, tile_config);
  if (m_tiles == 2 || m_tiles == 3 || m_tiles == 4) {
    libxsmm_setup_tile(5, m_blocking_info[0].sizes[m_tiles-1], k_blocking/l_k_pack_factor, tile_config);
  }
  /* Configure tiles for B */
  libxsmm_setup_tile(6, k_blocking/l_k_pack_factor, n_blocking_info[0].sizes[0], tile_config);
  if (n_tiles == 2) {
    libxsmm_setup_tile(7, k_blocking/l_k_pack_factor, n_blocking_info[0].sizes[1], tile_config);
  }
  if (n_tiles == 4) {
    libxsmm_setup_tile(7, k_blocking/l_k_pack_factor, n_blocking_info[0].sizes[3], tile_config);
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_adjust_m_advancement( libxsmm_generated_code* io_generated_code,
    libxsmm_loop_label_tracker*         io_loop_label_tracker,
    const libxsmm_gemm_descriptor*      i_xgemm_desc,
    const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
    const libxsmm_micro_kernel_config*  i_micro_kernel_config,
    libxsmm_blasint                     i_m_adjustment ) {
  /* Adjust C pointer */
  if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) {
    if (i_micro_kernel_config->vnni_format_C == 0) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_c, (long long)i_m_adjustment*2);
    } else {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_c, (long long)i_m_adjustment*2*2 );
    }
  } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_c, (long long)i_m_adjustment);
  } else {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_c, (long long)i_m_adjustment*4);
  }

  /* Also adjust eltwise pointers */
  if ((i_micro_kernel_config->fused_relu == 1) || (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) || (i_micro_kernel_config->fused_relu_bwd == 1) || (i_micro_kernel_config->fused_bcolbias == 1) || (i_micro_kernel_config->fused_scolbias == 1) || (i_micro_kernel_config->overwrite_C == 0)) {
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if ((i_micro_kernel_config->fused_relu == 1) && (i_micro_kernel_config->overwrite_C == 1) ) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, LIBXSMM_UPDIV((long long)i_m_adjustment,8) );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_m_adjustment*2*2 );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->fused_relu_bwd == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_RELU_BITMASK_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, LIBXSMM_UPDIV((long long)i_m_adjustment,8) );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_RELU_BITMASK_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->overwrite_C == 0) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_m_adjustment*2 );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->fused_bcolbias == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_m_adjustment * 2 );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->fused_scolbias == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_m_adjustment * 4 );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if ((i_micro_kernel_config->fused_relu == 1) || (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) || (i_micro_kernel_config->fused_relu_bwd == 1) || (i_micro_kernel_config->fused_bcolbias == 1) || (i_micro_kernel_config->fused_scolbias == 1) || (i_micro_kernel_config->overwrite_C == 0)) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
  }

  /* Adjust A pointers */
  if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
    libxsmm_generator_gemm_header_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_a_ptrs,
        i_gp_reg_mapping->gp_reg_reduce_loop, 8,
        0,
        i_gp_reg_mapping->gp_reg_a,
        0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a,
        ((long long)i_m_adjustment * 4) );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_a_ptrs,
        i_gp_reg_mapping->gp_reg_reduce_loop, 8,
        0,
        i_gp_reg_mapping->gp_reg_a,
        1 );
    libxsmm_generator_gemm_footer_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc);
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
  } else if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
    /* advance A pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a_base,
        ((long long)i_m_adjustment * 4)/i_micro_kernel_config->sparsity_factor_A );
    if (i_micro_kernel_config->decompress_A == 1) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BITMAP_PTR, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_m_adjustment * 4)/16 );
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BITMAP_PTR, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_DECOMPRESS_BUF, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_m_adjustment * 4) );
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_DECOMPRESS_BUF, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
    }
  } else {
    /* advance A pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a,
        ((long long)i_m_adjustment * 4)/i_micro_kernel_config->sparsity_factor_A );
    if (i_micro_kernel_config->decompress_A == 1) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BITMAP_PTR, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_m_adjustment * 4)/16 );
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BITMAP_PTR, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_DECOMPRESS_BUF, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_m_adjustment * 4) );
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_DECOMPRESS_BUF, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_adjust_n_advancement( libxsmm_generated_code* io_generated_code,
    libxsmm_loop_label_tracker*         io_loop_label_tracker,
    const libxsmm_gemm_descriptor*      i_xgemm_desc,
    const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
    const libxsmm_micro_kernel_config*  i_micro_kernel_config,
    libxsmm_blasint                     i_n_adjustment ) {
  /* Adjust C pointer */
  if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c,
        ((long long)i_n_adjustment*i_xgemm_desc->ldc*2) );
  } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c,
        ((long long)i_n_adjustment*i_xgemm_desc->ldc) );
  } else {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c,
        ((long long)i_n_adjustment*i_xgemm_desc->ldc*4) );
  }

  /* Also adjust eltwise pointers */
  if ((i_micro_kernel_config->fused_relu == 1) || (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) || (i_micro_kernel_config->fused_relu_bwd == 1) || (i_micro_kernel_config->overwrite_C == 0)) {
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if ((i_micro_kernel_config->fused_relu == 1) && (i_micro_kernel_config->overwrite_C == 1) ) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_n_adjustment*i_xgemm_desc->ldcp)/8 );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_n_adjustment*i_xgemm_desc->ldc*2) );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->fused_relu_bwd == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_RELU_BITMASK_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_n_adjustment*i_xgemm_desc->ldcp)/8 );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_RELU_BITMASK_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  /* In this case also advance the output ptr */
  if (i_micro_kernel_config->overwrite_C == 0) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_n_adjustment*i_xgemm_desc->ldc*2) );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if ((i_micro_kernel_config->fused_relu == 1) || (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) || (i_micro_kernel_config->fused_relu_bwd == 1) || (i_micro_kernel_config->overwrite_C == 0)) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
  }

  /* Adjust B pointers */
  if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
    /* handle trans B */
    int l_b_offset = 0;
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
      l_b_offset = i_n_adjustment * i_micro_kernel_config->datatype_size_in ;
    } else {
      l_b_offset = i_n_adjustment * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in;
    }
    libxsmm_generator_gemm_header_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config );

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_b_ptrs,
        i_gp_reg_mapping->gp_reg_reduce_loop, 8,
        0,
        i_gp_reg_mapping->gp_reg_b,
        0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_b, l_b_offset );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_b_ptrs,
        i_gp_reg_mapping->gp_reg_reduce_loop, 8,
        0,
        i_gp_reg_mapping->gp_reg_b,
        1 );
    libxsmm_generator_gemm_footer_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc);
  } else if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
    /* handle trans B */
    int l_b_offset = 0;
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
      l_b_offset = i_n_adjustment * i_micro_kernel_config->datatype_size_in ;
    } else {
      l_b_offset = i_n_adjustment * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in;
    }

    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_b_base, l_b_offset );

    if (i_micro_kernel_config->norm_to_normT_B_ext_buf == 1) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANS_EXT_BUF_B, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_n_adjustment * 2 );
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANS_EXT_BUF_B, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
    }
  } else {
    /* handle trans B */
    int l_b_offset = 0;
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
      l_b_offset = i_n_adjustment * i_micro_kernel_config->datatype_size_in;
    } else {
      l_b_offset = i_n_adjustment * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in;
    }

    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_b, l_b_offset );

    if (i_micro_kernel_config->norm_to_normT_B_ext_buf == 1) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANS_EXT_BUF_B, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_n_adjustment * 2);
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANS_EXT_BUF_B, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_kernel_wrapper( libxsmm_generated_code* io_generated_code, const libxsmm_gemm_descriptor* i_xgemm_desc_const ) {
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;

  /* define gp register mapping */
  libxsmm_reset_x86_gp_reg_mapping( &l_gp_reg_mapping );
#if defined(_WIN32) || defined(__CYGWIN__)
#else
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R8;
  if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc_const->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_xgemm_desc_const->datatype )) ) {
    l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_RCX;
    l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_R8;
    l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R9;
  } else {
    l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_UNDEF;
  }
  /* If we are generating the batchreduce kernel, then we rename the registers */
  if (i_xgemm_desc_const->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
    l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RAX;
    l_gp_reg_mapping.gp_reg_a_ptrs = LIBXSMM_X86_GP_REG_RDI;
    l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RBX;
    l_gp_reg_mapping.gp_reg_b_ptrs = LIBXSMM_X86_GP_REG_RSI;
    l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
    l_gp_reg_mapping.gp_reg_reduce_count = LIBXSMM_X86_GP_REG_RCX;
    if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc_const->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_xgemm_desc_const->datatype )) ) {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_R8;
      l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_R9;
      l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_UNDEF;
    } else {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_UNDEF;
      l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_R8;
      l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R9;
    }
    l_gp_reg_mapping.gp_reg_reduce_loop = LIBXSMM_X86_GP_REG_R10;
  } else if (i_xgemm_desc_const->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) {
    l_gp_reg_mapping.gp_reg_a_base = LIBXSMM_X86_GP_REG_RDI;
    l_gp_reg_mapping.gp_reg_a_offset = LIBXSMM_X86_GP_REG_R8;
    l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RAX;
    l_gp_reg_mapping.gp_reg_b_base = LIBXSMM_X86_GP_REG_RSI;
    l_gp_reg_mapping.gp_reg_b_offset = LIBXSMM_X86_GP_REG_R9;
    if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc_const->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_xgemm_desc_const->datatype )) ) {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_RAX;
    } else {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_UNDEF;
    }
    l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RBX;
    l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
    l_gp_reg_mapping.gp_reg_reduce_count = LIBXSMM_X86_GP_REG_RCX;
    l_gp_reg_mapping.gp_reg_reduce_loop = LIBXSMM_X86_GP_REG_R10;
  } else if (i_xgemm_desc_const->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) {
    l_gp_reg_mapping.gp_reg_a_base = LIBXSMM_X86_GP_REG_RDI;
    l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_R8;
    l_gp_reg_mapping.gp_reg_b_base = LIBXSMM_X86_GP_REG_RSI;
    l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RBX;
    l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
    l_gp_reg_mapping.gp_reg_reduce_count = LIBXSMM_X86_GP_REG_RCX;
    if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc_const->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_xgemm_desc_const->datatype )) ) {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_R8;
    } else {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_UNDEF;
    }
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

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* open asm */
  libxsmm_x86_instruction_open_stream_v2( io_generated_code, 0, 0 );

  /* call Intel AMX kernel */
  libxsmm_generator_gemm_amx_kernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, i_xgemm_desc_const );

  /* close asm */
  libxsmm_x86_instruction_close_stream_v2( io_generated_code, 0 );
}

/* Setup A (in vnni4 or flat) and B bf8 tensors as bf16 tensors in stack for execution on amx (A in vnni2)*/
LIBXSMM_API_INTERN void libxsmm_generator_gemm_setup_f8_ABC_tensors_to_stack_for_amx(  libxsmm_generated_code*        io_generated_code,
                                                                                        libxsmm_loop_label_tracker*    io_loop_label_tracker,
                                                                                        libxsmm_gp_reg_mapping*        i_gp_reg_mapping,
                                                                                        libxsmm_micro_kernel_config*   i_micro_kernel_config,
                                                                                        libxsmm_gemm_descriptor*       i_xgemm_desc,
                                                                                        const libxsmm_gemm_descriptor* i_xgemm_desc_orig,
                                                                                        libxsmm_datatype               i_in_dtype ) {
  libxsmm_descriptor_blob           l_meltw_blob;
  libxsmm_mateltwise_kernel_config  l_mateltwise_kernel_config;
  libxsmm_mateltwise_gp_reg_mapping l_mateltwise_gp_reg_mapping;
  const libxsmm_meltw_descriptor *  l_mateltwise_desc;
  int is_stride_brgemm  = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0) ? 1 : 0;
  int is_offset_brgemm  = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) > 0) ? 1 : 0;
  int is_address_brgemm = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0) ? 1 : 0;
  int is_brgemm         = ((is_stride_brgemm == 1) || (is_offset_brgemm == 1) || (is_address_brgemm == 1)) ? 1 : 0;
  unsigned int a_in_vnni        = ((i_xgemm_desc_orig->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) ? 1 : 0;
  unsigned int bf8_output_gemm  = (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_OUT( i_xgemm_desc_orig->datatype ) ) ? 1 : 0;
  unsigned int hf8_output_gemm  = (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GETENUM_OUT( i_xgemm_desc_orig->datatype ) ) ? 1 : 0;
  unsigned int struct_gp_reg    = LIBXSMM_X86_GP_REG_R15;
  unsigned int tmp_reg          = LIBXSMM_X86_GP_REG_R14;
  unsigned int loop_reg         = LIBXSMM_X86_GP_REG_R13;
  unsigned int bound_reg        = LIBXSMM_X86_GP_REG_R12;
  unsigned int tmp_reg2         = LIBXSMM_X86_GP_REG_RDX;
  unsigned int gp_reg_a         = LIBXSMM_X86_GP_REG_RDI;
  unsigned int gp_reg_b         = LIBXSMM_X86_GP_REG_RSI;
  unsigned short gp_save_bitmask = 0x2 | 0x4 | 0x100 | 0x200 | 0x400 | 0x800 | 0x1000 | 0x2000 | 0x4000 | 0x8000;

  libxsmm_generator_x86_save_gpr_regs( io_generated_code, gp_save_bitmask);

  l_mateltwise_gp_reg_mapping.gp_reg_param_struct = struct_gp_reg;
  libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MELTW_STRUCT_PTR, struct_gp_reg );
  libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_C_OUTPUT_PTR, i_gp_reg_mapping->gp_reg_c );

  /* In case of pure BF8 kernel, we use scratch for C (the amx gemm that is performed is BF16F32) */
  if ((bf8_output_gemm > 0) || (hf8_output_gemm > 0)) {
    if (0 == (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) {
      /* First convert BF8 to F32 output if beta is 1 */
      libxsmm_x86_instruction_alu_mem( io_generated_code,
              LIBXSMM_X86_INSTR_MOVQ,
              struct_gp_reg,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              32,
              i_gp_reg_mapping->gp_reg_c,
              1 );
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_C_SCRATCH_PTR, tmp_reg );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
              LIBXSMM_X86_INSTR_MOVQ,
              struct_gp_reg,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              64,
              tmp_reg,
              1 );
      l_mateltwise_desc = libxsmm_meltw_descriptor_init2(&l_meltw_blob, (libxsmm_datatype) LIBXSMM_GETENUM_OUT( i_xgemm_desc_orig->datatype ), LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED,
        LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, i_xgemm_desc_orig->m, i_xgemm_desc_orig->n, i_xgemm_desc_orig->ldc, i_xgemm_desc->ldc, 0, 0,
        0, LIBXSMM_CAST_USHORT(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY), LIBXSMM_MELTW_OPERATION_UNARY);
      libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_mateltwise_kernel_config, l_mateltwise_desc );
      libxsmm_generator_unary_binary_avx512_microkernel( io_generated_code, io_loop_label_tracker, &l_mateltwise_gp_reg_mapping, &l_mateltwise_kernel_config, l_mateltwise_desc );
    }
  }

  if ((i_micro_kernel_config->fused_b8colbias > 0) || (i_micro_kernel_config->fused_h8colbias > 0) ) {
    /* Convert BF8 colbias to F32 for later use */
    i_micro_kernel_config->fused_b8colbias = 0;
    i_micro_kernel_config->fused_h8colbias = 0;
    i_micro_kernel_config->fused_scolbias = 1;
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, tmp_reg );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
            LIBXSMM_X86_INSTR_MOVQ,
            struct_gp_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            32,
            tmp_reg,
            1 );
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BIAS_SCRATCH_PTR, tmp_reg );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
            LIBXSMM_X86_INSTR_MOVQ,
            struct_gp_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            64,
            tmp_reg,
            1 );
    l_mateltwise_desc = libxsmm_meltw_descriptor_init2(&l_meltw_blob, i_in_dtype, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED,
      LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, i_xgemm_desc_orig->m, 1, i_xgemm_desc_orig->m, i_xgemm_desc_orig->m, 0, 0,
      0, LIBXSMM_CAST_USHORT(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY), LIBXSMM_MELTW_OPERATION_UNARY);
    libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_mateltwise_kernel_config, l_mateltwise_desc );
    libxsmm_generator_unary_binary_avx512_microkernel( io_generated_code, io_loop_label_tracker, &l_mateltwise_gp_reg_mapping, &l_mateltwise_kernel_config, l_mateltwise_desc );
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BIAS_SCRATCH_PTR, tmp_reg );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, tmp_reg );
  }

  /* Setup A in stack */
  if (a_in_vnni > 0) {
    /* If A is originally in VNNI4 format: First convert VNNI4 to VNNI2 (8bit)*/
    /* If A is originally in VNNI4 format: Second convert BF8 to BF16 */
    libxsmm_generator_gemm_apply_ops_input_tensor_and_store_to_stack( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
        gp_reg_a, struct_gp_reg, tmp_reg, loop_reg, bound_reg, tmp_reg2,
        LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI2, i_xgemm_desc->m, i_xgemm_desc->k, i_xgemm_desc_orig->lda, i_xgemm_desc->lda, LIBXSMM_CAST_BLASINT(i_xgemm_desc_orig->c1),
        i_in_dtype, i_in_dtype, i_in_dtype,
        LIBXSMM_GEMM_STACK_VAR_A_OFFS_BRGEMM_PTR, LIBXSMM_GEMM_STACK_VAR_A_SCRATCH_PTR, LIBXSMM_GEMM_STACK_VAR_A_EMU_PTR,
        LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, i_xgemm_desc->m, i_xgemm_desc->k, i_xgemm_desc->lda, i_xgemm_desc->lda,
        i_in_dtype, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16);
  } else {
    /* If A is originally in flat format: First convert BF8 to BF16 */
    /* If A is originally in flat format: Second transform NORM to VNNI2 */
    libxsmm_generator_gemm_apply_ops_input_tensor_and_store_to_stack( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
        gp_reg_a, struct_gp_reg, tmp_reg, loop_reg, bound_reg, tmp_reg2,
        LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, i_xgemm_desc->m, i_xgemm_desc->k, i_xgemm_desc_orig->lda, i_xgemm_desc->lda, LIBXSMM_CAST_BLASINT(i_xgemm_desc_orig->c1),
        i_in_dtype, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16,
        LIBXSMM_GEMM_STACK_VAR_A_OFFS_BRGEMM_PTR, LIBXSMM_GEMM_STACK_VAR_A_SCRATCH_PTR, LIBXSMM_GEMM_STACK_VAR_A_EMU_PTR,
        LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2, i_xgemm_desc->m, i_xgemm_desc->k, i_xgemm_desc->lda, i_xgemm_desc->lda,
        LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16);
  }

  /* Setup B in stack */
  libxsmm_generator_gemm_apply_ops_input_tensor_and_store_to_stack( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
      gp_reg_b, struct_gp_reg, tmp_reg, loop_reg, bound_reg, tmp_reg2,
      LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, i_xgemm_desc->k, i_xgemm_desc->n, i_xgemm_desc_orig->ldb, i_xgemm_desc->k, LIBXSMM_CAST_BLASINT(i_xgemm_desc_orig->c2),
      i_in_dtype, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16,
      LIBXSMM_GEMM_STACK_VAR_B_OFFS_BRGEMM_PTR, LIBXSMM_GEMM_STACK_VAR_A_SCRATCH_PTR, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR,
      LIBXSMM_MELTW_TYPE_UNARY_NONE, 0, 0, 0, 0, (libxsmm_datatype)0, (libxsmm_datatype)0, (libxsmm_datatype)0);

  /* Adjust a/b gp_regs to point to the bf16 tensors in stack */
  libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_A_EMU_PTR, tmp_reg );
  libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, tmp_reg, gp_reg_a);
  libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_B_EMU_PTR, tmp_reg );
  libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, tmp_reg, gp_reg_b);

  /* Adjust descriptor for internal strided BRGEMM */
  if (is_brgemm > 0) {
    if (is_offset_brgemm > 0) {
      i_xgemm_desc->flags = i_xgemm_desc->flags ^ LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET;
      i_xgemm_desc->flags = i_xgemm_desc->flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE;
    }
    if (is_address_brgemm > 0) {
      i_xgemm_desc->flags = i_xgemm_desc->flags ^ LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS;
      i_xgemm_desc->flags = i_xgemm_desc->flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE;
      i_gp_reg_mapping->gp_reg_a_base = gp_reg_a;
      i_gp_reg_mapping->gp_reg_b_base = gp_reg_b;
    }
    i_xgemm_desc->c1 = 2LL * i_xgemm_desc->m * i_xgemm_desc->k;
    i_xgemm_desc->c2 = 2LL * i_xgemm_desc->n * i_xgemm_desc->k;
  }

  /* Readjusting descriptor for upcoming bf16 gemm */
  if ((i_xgemm_desc_orig->m % 16 == 0) || (i_xgemm_desc_orig->m <= 32)) {
  } else {
    i_xgemm_desc->m = i_xgemm_desc_orig->m - (i_xgemm_desc_orig->m % 32);
  }
  i_xgemm_desc->k = i_xgemm_desc_orig->k;
  i_xgemm_desc->ldb = i_xgemm_desc->k;

  libxsmm_generator_x86_restore_gpr_regs( io_generated_code, gp_save_bitmask);

  /* In case of pure BF8 kernel, we use scratch for C (the amx gemm that is performed is BF16F32) */
  if ((bf8_output_gemm > 0) || (hf8_output_gemm > 0)) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_C_SCRATCH_PTR, i_gp_reg_mapping->gp_reg_c );
  }
}

/* Applies elementwise fusion on C-scratch and stores the result to the output tensor */
LIBXSMM_API_INTERN void libxsmm_generator_gemm_emit_f8_eltwise_fusion(   libxsmm_generated_code*        io_generated_code,
                                                                          libxsmm_loop_label_tracker*    io_loop_label_tracker,
                                                                          libxsmm_micro_kernel_config*   i_micro_kernel_config,
                                                                          libxsmm_gemm_descriptor*       i_xgemm_desc,
                                                                          const libxsmm_gemm_descriptor* i_xgemm_desc_orig,
                                                                          unsigned int                   i_defer_c_vnni_format,
                                                                          unsigned int                   i_defer_relu_bitmask_compute,
                                                                          libxsmm_datatype               i_dtype ) {
  libxsmm_descriptor_blob           l_meltw_blob;
  libxsmm_mateltwise_kernel_config  l_mateltwise_kernel_config;
  libxsmm_mateltwise_gp_reg_mapping l_mateltwise_gp_reg_mapping;
  const libxsmm_meltw_descriptor *  l_mateltwise_desc;
  unsigned int bf8_output_gemm  = (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_OUT( i_xgemm_desc_orig->datatype ) ) ? 1 : 0;
  unsigned int hf8_output_gemm  = (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GETENUM_OUT( i_xgemm_desc_orig->datatype ) ) ? 1 : 0;
  unsigned int struct_gp_reg    = LIBXSMM_X86_GP_REG_R15;
  unsigned int tmp_reg          = LIBXSMM_X86_GP_REG_R14;
  l_mateltwise_gp_reg_mapping.gp_reg_param_struct = struct_gp_reg;

  /* Apply sigmoid fusion if requested */
  if (i_micro_kernel_config->fused_sigmoid > 0) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R14 );
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R15 );
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MELTW_STRUCT_PTR, struct_gp_reg );
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_C_SCRATCH_PTR, tmp_reg );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
            LIBXSMM_X86_INSTR_MOVQ,
            struct_gp_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            32,
            tmp_reg,
            1 );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
            LIBXSMM_X86_INSTR_MOVQ,
            struct_gp_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            64,
            tmp_reg,
            1 );
    l_mateltwise_desc = libxsmm_meltw_descriptor_init2(&l_meltw_blob, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED,
      LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, i_xgemm_desc_orig->m, i_xgemm_desc_orig->n, i_xgemm_desc->ldc, i_xgemm_desc->ldc, 0, 0,
      0, LIBXSMM_CAST_USHORT(LIBXSMM_MELTW_TYPE_UNARY_SIGMOID), LIBXSMM_MELTW_OPERATION_UNARY);
    libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_mateltwise_kernel_config, l_mateltwise_desc );
    libxsmm_generator_unary_binary_avx512_microkernel( io_generated_code, io_loop_label_tracker, &l_mateltwise_gp_reg_mapping, &l_mateltwise_kernel_config, l_mateltwise_desc );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R15 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R14 );
  }

  /* Convert output to BF8 */
  if ((bf8_output_gemm > 0) || (hf8_output_gemm > 0)) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R14 );
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R15 );
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MELTW_STRUCT_PTR, struct_gp_reg );
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_C_SCRATCH_PTR, tmp_reg );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
            LIBXSMM_X86_INSTR_MOVQ,
            struct_gp_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            32,
            tmp_reg,
            1 );
    if (i_defer_c_vnni_format == 0) {
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_C_OUTPUT_PTR, tmp_reg );
    }
    libxsmm_x86_instruction_alu_mem( io_generated_code,
            LIBXSMM_X86_INSTR_MOVQ,
            struct_gp_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            64,
            tmp_reg,
            1 );
    l_mateltwise_desc = libxsmm_meltw_descriptor_init2(&l_meltw_blob, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED,
      LIBXSMM_DATATYPE_F32, i_dtype, i_xgemm_desc_orig->m, i_xgemm_desc_orig->n, i_xgemm_desc->ldc, (i_defer_c_vnni_format == 0) ? i_xgemm_desc_orig->ldc : i_xgemm_desc->ldc, 0, 0,
      0, LIBXSMM_CAST_USHORT(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY), LIBXSMM_MELTW_OPERATION_UNARY);
    libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_mateltwise_kernel_config, l_mateltwise_desc );
    libxsmm_generator_unary_binary_avx512_microkernel( io_generated_code, io_loop_label_tracker, &l_mateltwise_gp_reg_mapping, &l_mateltwise_kernel_config, l_mateltwise_desc );

    /* Apply C-vnni4 formating if requested */
    if (i_defer_c_vnni_format > 0) {
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_C_OUTPUT_PTR, tmp_reg );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
              LIBXSMM_X86_INSTR_MOVQ,
              struct_gp_reg,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              64,
              tmp_reg,
              1 );
      l_mateltwise_desc = libxsmm_meltw_descriptor_init2(&l_meltw_blob, i_dtype, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED,
        i_dtype, i_dtype, i_xgemm_desc_orig->m, i_xgemm_desc_orig->n, i_xgemm_desc->ldc, i_xgemm_desc_orig->ldc, 0, 0,
        0, LIBXSMM_CAST_USHORT(LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4), LIBXSMM_MELTW_OPERATION_UNARY);
      libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_mateltwise_kernel_config, l_mateltwise_desc );
      libxsmm_generator_transform_x86_microkernel( io_generated_code, io_loop_label_tracker, &l_mateltwise_gp_reg_mapping, &l_mateltwise_kernel_config, l_mateltwise_desc );
    }
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R15 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R14 );
  }

  /* Apply RELU if requested */
  if ( (i_micro_kernel_config->fused_relu_nobitmask > 0) || (i_defer_relu_bitmask_compute > 0)) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R14 );
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R15 );
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MELTW_STRUCT_PTR, struct_gp_reg );
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_C_OUTPUT_PTR, tmp_reg );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
            LIBXSMM_X86_INSTR_MOVQ,
            struct_gp_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            32,
            tmp_reg,
            1 );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
            LIBXSMM_X86_INSTR_MOVQ,
            struct_gp_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            64,
            tmp_reg,
            1 );
    if (i_defer_relu_bitmask_compute > 0) {
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, tmp_reg );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
              LIBXSMM_X86_INSTR_MOVQ,
              struct_gp_reg,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              72,
              tmp_reg,
              1 );
    }
    l_mateltwise_desc = libxsmm_meltw_descriptor_init2(&l_meltw_blob, (libxsmm_datatype)LIBXSMM_GETENUM_OUT( i_xgemm_desc_orig->datatype ), LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED,
      LIBXSMM_DATATYPE_F32, (libxsmm_datatype)LIBXSMM_GETENUM_OUT( i_xgemm_desc_orig->datatype ), i_xgemm_desc_orig->m, i_xgemm_desc_orig->n, i_xgemm_desc_orig->ldc, i_xgemm_desc_orig->ldc, 0, 0,
      (i_defer_relu_bitmask_compute > 0) ? LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT : 0, LIBXSMM_CAST_USHORT(LIBXSMM_MELTW_TYPE_UNARY_RELU), LIBXSMM_MELTW_OPERATION_UNARY);
    libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_mateltwise_kernel_config, l_mateltwise_desc );
    libxsmm_generator_unary_binary_avx512_microkernel( io_generated_code, io_loop_label_tracker, &l_mateltwise_gp_reg_mapping, &l_mateltwise_kernel_config, l_mateltwise_desc );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R15 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R14 );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_kernel( libxsmm_generated_code*            io_generated_code,
                                                                           libxsmm_loop_label_tracker*    io_loop_label_tracker,
                                                                           libxsmm_gp_reg_mapping*  i_gp_reg_mapping,
                                                                           const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  libxsmm_micro_kernel_config l_micro_kernel_config;
  /* Allow descriptor to be modified if need be */
  libxsmm_gemm_descriptor l_xgemm_desc_mod = *i_xgemm_desc;
  libxsmm_gemm_descriptor *l_xgemm_desc = &l_xgemm_desc_mod;
  unsigned int m0 = 0, m1 = 0;
  unsigned int l_enforce_Mx1_amx_tile_blocking = libxsmm_cpuid_x86_amx_gemm_enforce_mx1_tile_blocking();

  /* AMX specific blocking info */
  libxsmm_blocking_info_t m_blocking_info[2], n_blocking_info[2];
  unsigned int n_gemm_code_blocks = 0;

  /* Emulating BF8 gemm on AMX */
  int bf8_gemm_via_stack_alloc_tensors = (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_INP( l_xgemm_desc->datatype ) ) ? 1 : 0;
  int bf8_output_gemm = (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_OUT( l_xgemm_desc->datatype ) ) ? 1 : 0;
  int hf8_gemm_via_stack_alloc_tensors = (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GETENUM_INP( l_xgemm_desc->datatype ) ) ? 1 : 0;
  int hf8_output_gemm = (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GETENUM_OUT( l_xgemm_desc->datatype ) ) ? 1 : 0;
  int l_defer_relu_bitmask_compute = 0;
  int l_defer_c_vnni_format = 0;

  libxsmm_tile_config tile_config;
  LIBXSMM_MEMZERO127(&tile_config);

  /* Adjust descriptor to perform GEMM with BF16 inputs and F32 output */
  if ((bf8_gemm_via_stack_alloc_tensors > 0) || (hf8_gemm_via_stack_alloc_tensors > 0) ) {
    if (!( (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & l_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & l_xgemm_desc->flags) == 0)) ||
        (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & l_xgemm_desc->flags) != 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & l_xgemm_desc->flags) != 0))     )){
      return;
    }

    l_xgemm_desc->datatype = LIBXSMM_GETENUM(LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32);
    l_xgemm_desc->lda = l_xgemm_desc->m;
    l_xgemm_desc->k = i_xgemm_desc->k;
    l_xgemm_desc->ldb = l_xgemm_desc->k;
    if ((bf8_output_gemm > 0) || (hf8_output_gemm > 0)) {
      l_xgemm_desc->ldc = l_xgemm_desc->m;
    }
  }

  /* define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config_fullvector( &l_micro_kernel_config, io_generated_code->arch, l_xgemm_desc, 0 );

  /* First stamp out a nice GEMM and the if need be take care of remainder handling */
  if ((l_xgemm_desc->m % 16 == 0) || (l_xgemm_desc->m <= 32) || (l_enforce_Mx1_amx_tile_blocking > 0)) {
    /* Nothing to do here */
    n_gemm_code_blocks = 1;
  } else {
    /* Need to stamp out a remainder handling gemm */
    m0 = l_xgemm_desc->m - (l_xgemm_desc->m % 32);
    m1 = l_xgemm_desc->m - m0;
    l_xgemm_desc->m = m0;
    n_gemm_code_blocks = 2;
  }

  /* Here compute the 2D blocking info based on the M and N values */
  libxsmm_generator_gemm_init_micro_kernel_config_tileblocking(l_xgemm_desc, &l_micro_kernel_config, m_blocking_info, n_blocking_info, & tile_config );

  /* Setup stack frame... */
  l_micro_kernel_config.m_tiles = m_blocking_info[0].tiles;
  l_micro_kernel_config.n_tiles = n_blocking_info[0].tiles;

  /* implementing load from struct */
  if ( (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & l_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & l_xgemm_desc->flags) == 0)) ||
       (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & l_xgemm_desc->flags) != 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & l_xgemm_desc->flags) != 0))) {
    if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI & l_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI) ||
         ((LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI & l_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI) ) {
      /* RDI holds the pointer to the struct, so lets first move this one into R15 */
      libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_help_1 );
      /* A pointer */
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 32, LIBXSMM_X86_GP_REG_RDI, 0 );
      /* B pointer */
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 64, LIBXSMM_X86_GP_REG_RSI, 0 );
      /* C pointer */
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 96, i_gp_reg_mapping->gp_reg_c, 0 );
      /* batch reduce count & offsett arrays*/
      if ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) || (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET)) {
        libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                         i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 16, i_gp_reg_mapping->gp_reg_reduce_count, 0 );
        if ( l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET ) {
          libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
              i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 40, i_gp_reg_mapping->gp_reg_a_offset, 0 );
          libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
              i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 72, i_gp_reg_mapping->gp_reg_b_offset, 0 );
        }
      }
      if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( l_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( l_xgemm_desc->datatype )) ) {
        libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                         i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 112, i_gp_reg_mapping->gp_reg_scf, 0 );
      }
    }
  }

  /* Adjusting descriptor for bf8 emulation */
  if ((bf8_gemm_via_stack_alloc_tensors > 0) || (hf8_gemm_via_stack_alloc_tensors > 0)) {
    l_xgemm_desc->k = i_xgemm_desc->k;
    l_xgemm_desc->ldb = l_xgemm_desc->k;
    l_xgemm_desc->m =  i_xgemm_desc->m;
  }

  if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI & l_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI) ||
       ((LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI & l_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI) ) {
    libxsmm_generator_gemm_setup_fusion_microkernel_properties_v2(l_xgemm_desc, &l_micro_kernel_config );
  } else {
    /* AMX kernels are supported only under the new abi */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ILLEGAL_ABI );
  }

  l_micro_kernel_config.bf8_gemm_via_stack_alloc_tensors = bf8_gemm_via_stack_alloc_tensors;
  l_micro_kernel_config.hf8_gemm_via_stack_alloc_tensors = hf8_gemm_via_stack_alloc_tensors;
  libxsmm_generator_gemm_setup_stack_frame( io_generated_code, l_xgemm_desc, i_gp_reg_mapping, &l_micro_kernel_config);

  /* Copy A/B/C to stack in case of emulating BF8 on amx stack */
  if ((bf8_gemm_via_stack_alloc_tensors > 0) || (hf8_gemm_via_stack_alloc_tensors > 0) ) {
    if ( l_micro_kernel_config.fused_relu > 0 ) {
      l_defer_relu_bitmask_compute = 1;
      l_micro_kernel_config.fused_relu = 0;
    }
    if (l_micro_kernel_config.vnni_format_C > 0) {
      l_defer_c_vnni_format = 1;
      l_micro_kernel_config.vnni_format_C = 0;
    }
    libxsmm_generator_gemm_setup_f8_ABC_tensors_to_stack_for_amx(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc, i_xgemm_desc, (libxsmm_datatype)LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) );
  }

  libxsmm_generator_gemm_amx_setup_fusion_infra( io_generated_code, l_xgemm_desc, i_gp_reg_mapping, &l_micro_kernel_config );
  libxsmm_generator_gemm_amx_setup_masking_infra( io_generated_code, &l_micro_kernel_config );

  if ((((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & l_xgemm_desc->flags) != 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & l_xgemm_desc->flags) == 0)) ||
      (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & l_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & l_xgemm_desc->flags) == 0)) ||
      (n_gemm_code_blocks > 1) || (bf8_gemm_via_stack_alloc_tensors > 0) || (hf8_gemm_via_stack_alloc_tensors > 0) ) {
    libxsmm_x86_instruction_tile_control( io_generated_code,
        0,
        l_micro_kernel_config.instruction_set,
        LIBXSMM_X86_INSTR_LDTILECFG,
        LIBXSMM_X86_GP_REG_UNDEF,
        0,
        &tile_config );
  }

  if ( (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & l_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & l_xgemm_desc->flags) == 0)) ||
      (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & l_xgemm_desc->flags) != 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & l_xgemm_desc->flags) != 0))     ) {
    /* Set the LD registers for A and B matrices */
    libxsmm_x86_instruction_alu_imm(io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_lda, ((long long)l_xgemm_desc->lda * 4/*l_micro_kernel_config.datatype_size*/)/4);
    libxsmm_x86_instruction_alu_imm(io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_ldb, ((long long)l_xgemm_desc->ldb * l_micro_kernel_config.datatype_size_in/*l_micro_kernel_config.datatype_size*/)/4);
    libxsmm_x86_instruction_alu_imm(io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_ldc, ((long long)l_xgemm_desc->ldc * 4/*l_micro_kernel_config.datatype_size*/)/4);

    /* Load the actual batch-reduce trip count */
    if ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          l_micro_kernel_config.alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_reduce_count,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          0,
          i_gp_reg_mapping->gp_reg_reduce_count,
          0 );
    }

    libxsmm_generator_gemm_amx_kernel_nloop(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc, n_blocking_info, m_blocking_info);
  }

  /* Conditionally perform a tilerelease */
  if ( (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & l_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & l_xgemm_desc->flags) != 0)) ||
      (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & l_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & l_xgemm_desc->flags) == 0)) ||
      (n_gemm_code_blocks > 1) || (bf8_gemm_via_stack_alloc_tensors > 0) || (hf8_gemm_via_stack_alloc_tensors > 0)) {
    libxsmm_x86_instruction_tile_control( io_generated_code,
        0,
        l_micro_kernel_config.instruction_set,
        LIBXSMM_X86_INSTR_TILERELEASE,
        LIBXSMM_X86_GP_REG_UNDEF,
        0,
        &tile_config );
  }

  if (n_gemm_code_blocks > 1) {
    l_xgemm_desc->m = m1;
    l_micro_kernel_config.m_remainder  = 0;

    /* Adjust descriptor in case of emulating BF8 on amx stack */
    if ((bf8_gemm_via_stack_alloc_tensors > 0) || (hf8_gemm_via_stack_alloc_tensors > 0)) {
      l_xgemm_desc->k = i_xgemm_desc->k;
      l_xgemm_desc->ldb = l_xgemm_desc->k;
    }

    /* Here compute the 2D blocking info based on the M and N values */
    libxsmm_generator_gemm_init_micro_kernel_config_tileblocking(l_xgemm_desc, &l_micro_kernel_config, m_blocking_info, n_blocking_info, & tile_config );
    libxsmm_generator_gemm_amx_setup_masking_infra( io_generated_code, &l_micro_kernel_config );

    libxsmm_x86_instruction_tile_control( io_generated_code,
        0,
        l_micro_kernel_config.instruction_set,
        LIBXSMM_X86_INSTR_LDTILECFG,
        LIBXSMM_X86_GP_REG_UNDEF,
        0,
        &tile_config );

    if ( (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & l_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & l_xgemm_desc->flags) == 0)) ||
      (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & l_xgemm_desc->flags) != 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & l_xgemm_desc->flags) != 0))     ) {

      if (l_micro_kernel_config.n_loop_exists > 0) {
        /* We should adjust n advancements in C/B etc. Also advance by M since all M adjustments have been revoked by the n loop footer */
        libxsmm_generator_gemm_amx_adjust_n_advancement(io_generated_code, io_loop_label_tracker, l_xgemm_desc, i_gp_reg_mapping, &l_micro_kernel_config, (libxsmm_blasint)l_xgemm_desc->n * -1 );
        libxsmm_generator_gemm_amx_adjust_m_advancement(io_generated_code, io_loop_label_tracker, l_xgemm_desc, i_gp_reg_mapping, &l_micro_kernel_config, m0 );
      } else if (l_micro_kernel_config.m_loop_exists == 0) {
        /* We should advance by M since no advancements have been made */
        libxsmm_generator_gemm_amx_adjust_m_advancement(io_generated_code, io_loop_label_tracker, l_xgemm_desc, i_gp_reg_mapping, &l_micro_kernel_config, m0 );
      } else {
        /* Nothing should be done since the M loop exists and has made the proper advancements in the M direction */
      }

      libxsmm_x86_instruction_alu_imm(io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_lda, ((long long)l_xgemm_desc->lda * 4/*l_micro_kernel_config.datatype_size*/)/4);
      libxsmm_x86_instruction_alu_imm(io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_ldb, ((long long)l_xgemm_desc->ldb * l_micro_kernel_config.datatype_size_in/*l_micro_kernel_config.datatype_size*/)/4);
      libxsmm_x86_instruction_alu_imm(io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_ldc, ((long long)l_xgemm_desc->ldc * 4/*l_micro_kernel_config.datatype_size*/)/4);

      libxsmm_generator_gemm_amx_kernel_nloop(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc, n_blocking_info, m_blocking_info);
    }

    libxsmm_x86_instruction_tile_control( io_generated_code,
        0,
        l_micro_kernel_config.instruction_set,
        LIBXSMM_X86_INSTR_TILERELEASE,
        LIBXSMM_X86_GP_REG_UNDEF,
        0,
        &tile_config );
  }

  /* Apply eltwise fusion in case of emulating BF8 on amx stack */
  if ((bf8_gemm_via_stack_alloc_tensors > 0) || (hf8_gemm_via_stack_alloc_tensors > 0)) {
    libxsmm_generator_gemm_emit_f8_eltwise_fusion(io_generated_code, io_loop_label_tracker, &l_micro_kernel_config, l_xgemm_desc, i_xgemm_desc, l_defer_c_vnni_format, l_defer_relu_bitmask_compute, (libxsmm_datatype)LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) );
  }

  /* Properly destroy stack frame... */
  libxsmm_generator_gemm_destroy_stack_frame( io_generated_code, l_xgemm_desc, i_gp_reg_mapping, &l_micro_kernel_config );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_kernel_mloop( libxsmm_generated_code*            io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    libxsmm_gp_reg_mapping*            i_gp_reg_mapping,
    libxsmm_micro_kernel_config*       i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    libxsmm_blocking_info_t*           n_blocking_info,
    libxsmm_blocking_info_t*           m_blocking_info ) {

  void (*l_generator_kloop)(libxsmm_generated_code*, libxsmm_loop_label_tracker*, const libxsmm_gp_reg_mapping*, libxsmm_micro_kernel_config*, const libxsmm_gemm_descriptor*,  libxsmm_blocking_info_t*,  libxsmm_blocking_info_t*, long long, long long, unsigned int);
  unsigned int l_m_done = 0;
  unsigned int l_m_count = 0;
  unsigned int l_m_blocking = m_blocking_info[0].blocking;
  unsigned int l_m_block_size = 0;
  unsigned int m_assembly_loop_exists = (l_m_blocking == i_xgemm_desc->m) ? 0 : 1;
  unsigned int fully_unroll_k = 1;
  unsigned int NON_UNROLLED_BR_LOOP_LABEL_START = 0;
  unsigned int NON_UNROLLED_BR_LOOP_LABEL_END = 1;
  unsigned int i;
  long long A_offs = 0, B_offs = 0;
#if defined(LIBXSMM_GENERATOR_GEMM_AMX_JUMP_LABEL_TRACKER_MALLOC)
  libxsmm_jump_label_tracker *const p_jump_label_tracker = (libxsmm_jump_label_tracker*)malloc(sizeof(libxsmm_jump_label_tracker));
#else
  libxsmm_jump_label_tracker l_jump_label_tracker;
  libxsmm_jump_label_tracker *const p_jump_label_tracker = &l_jump_label_tracker;
#endif
  libxsmm_reset_jump_label_tracker(p_jump_label_tracker);
  l_generator_kloop = libxsmm_generator_gemm_amx_kernel_kloop;
  i_micro_kernel_config->B_offs_trans = 0;
  i_micro_kernel_config->loop_label_id = 2;
  i_micro_kernel_config->p_jump_label_tracker = p_jump_label_tracker;

  /* apply m_blocking */
  while (l_m_done != (unsigned int)i_xgemm_desc->m) {
    l_m_blocking = m_blocking_info[l_m_count].blocking;
    l_m_block_size = m_blocking_info[l_m_count].block_size;
    l_m_done += l_m_block_size;

    if (m_assembly_loop_exists) {
      libxsmm_generator_gemm_header_mloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, l_m_blocking );
    }

    libxsmm_generator_gemm_load_C_amx( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, n_blocking_info, &m_blocking_info[l_m_count] );

    if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
      /* Compare actual trip count to the hint value. If equal jump to UNROLL START LABEL*/
      if (i_xgemm_desc->c3 > 0) {
        if (i_micro_kernel_config->decompress_A == 1) {
          libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, i_gp_reg_mapping->gp_reg_reduce_count );
        }
        libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_reduce_count, i_xgemm_desc->c3);
        libxsmm_x86_instruction_jump_to_label(io_generated_code, LIBXSMM_X86_INSTR_JNE, NON_UNROLLED_BR_LOOP_LABEL_START, p_jump_label_tracker);
      }

      /* UNROLLED version code is here */
      if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) && (i_xgemm_desc->c3 > 0) ) {
        libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_a_base, i_gp_reg_mapping->gp_reg_a);
        libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_b_base, i_gp_reg_mapping->gp_reg_b);
        if (i_micro_kernel_config->decompress_A == 1) {
          libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BITMAP_PTR, i_gp_reg_mapping->gp_reg_bitmap_a );
          libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_DECOMPRESS_BUF, i_gp_reg_mapping->gp_reg_decompressed_a );
        }
      }
      /* This is the reduce loop */
      for (i = 0; i < i_xgemm_desc->c3; i++) {
        if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
          /* load to reg_a the proper array based on the reduce loop index */
          libxsmm_x86_instruction_alu_mem( io_generated_code,
              i_micro_kernel_config->alu_mov_instruction,
              i_gp_reg_mapping->gp_reg_a_ptrs,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              i*8,
              i_gp_reg_mapping->gp_reg_a,
              0 );
          /* load to reg_b the proper array based on the reduce loop index */
          libxsmm_x86_instruction_alu_mem( io_generated_code,
              i_micro_kernel_config->alu_mov_instruction,
              i_gp_reg_mapping->gp_reg_b_ptrs,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              i*8,
              i_gp_reg_mapping->gp_reg_b,
              0 );
        } else if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) {
          /* Calculate to reg_b the proper address based on the reduce loop index */
          libxsmm_x86_instruction_alu_mem( io_generated_code,
              i_micro_kernel_config->alu_mov_instruction,
              i_gp_reg_mapping->gp_reg_b_offset,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              i*8,
              i_gp_reg_mapping->gp_reg_b,
              0 );
          libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_b_base, i_gp_reg_mapping->gp_reg_b);

          /* Calculate to reg_a the proper address based on the reduce loop index */
          libxsmm_x86_instruction_alu_mem( io_generated_code,
              i_micro_kernel_config->alu_mov_instruction,
              i_gp_reg_mapping->gp_reg_a_offset,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              i*8,
              i_gp_reg_mapping->gp_reg_a,
              0 );

          if (i_micro_kernel_config->decompress_A == 1) {
            libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
            libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0);
            libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BITMAP_PTR, i_gp_reg_mapping->gp_reg_bitmap_a );
            libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_DECOMPRESS_BUF, i_gp_reg_mapping->gp_reg_decompressed_a);
            libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_help_0, i_micro_kernel_config->sparsity_factor_A);
            libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_decompressed_a);
            libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SARQ, i_gp_reg_mapping->gp_reg_help_0, 4);
            libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_bitmap_a);
            libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a_base, i_gp_reg_mapping->gp_reg_a);
            libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
          } else {
            libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a_base, i_gp_reg_mapping->gp_reg_a);
          }
          i_micro_kernel_config->br_loop_index = i;
        } else if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) {
          A_offs = i * i_xgemm_desc->c1;
          B_offs = i * i_xgemm_desc->c2;
          i_micro_kernel_config->B_offs_trans = i * i_micro_kernel_config->stride_b_trans;
        }

        /* Here is the K loop along with the microkernel */
        l_generator_kloop(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, n_blocking_info, &m_blocking_info[l_m_count], A_offs, B_offs, 1);

        /* TODO: this cide is dead: In case of address based batch redcue push the proper A/B address updates if the k loop is not fully unrolled */
#if 0
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) && (fully_unroll_k == 0)) {
          libxsmm_x86_instruction_alu_mem( io_generated_code,
              i_micro_kernel_config->alu_mov_instruction,
              i_gp_reg_mapping->gp_reg_b_ptrs,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              i*8,
              i_gp_reg_mapping->gp_reg_b,
              1 );
          libxsmm_x86_instruction_alu_mem( io_generated_code,
              i_micro_kernel_config->alu_mov_instruction,
              i_gp_reg_mapping->gp_reg_a_ptrs,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              i*8,
              i_gp_reg_mapping->gp_reg_a,
              1 );
        }
#endif
      }

      if (i_xgemm_desc->c3 > 0) {
        if (!((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0 || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) > 0)) {
          libxsmm_generator_gemm_store_C_amx( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, n_blocking_info, &m_blocking_info[l_m_count] );
        }
      }

      /* End of UNROLLED code is here*/
      /* Jump after non-unrolled code variant */
      if (i_xgemm_desc->c3 > 0) {
        libxsmm_x86_instruction_jump_to_label(io_generated_code, LIBXSMM_X86_INSTR_JMP, NON_UNROLLED_BR_LOOP_LABEL_END, p_jump_label_tracker);
      }

      /* NON_UNROLLED_BR_LOOP_LABEL_START */
      libxsmm_x86_instruction_register_jump_label(io_generated_code, NON_UNROLLED_BR_LOOP_LABEL_START, p_jump_label_tracker);
      /* This is the reduce loop */
      libxsmm_generator_gemm_header_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config );

      if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
        /* load to reg_a the proper array based on the reduce loop index */
        libxsmm_x86_instruction_alu_mem( io_generated_code,
            i_micro_kernel_config->alu_mov_instruction,
            i_gp_reg_mapping->gp_reg_a_ptrs,
            i_gp_reg_mapping->gp_reg_reduce_loop, 8,
            0,
            i_gp_reg_mapping->gp_reg_a,
            0 );
        /* load to reg_b the proper array based on the reduce loop index */
        libxsmm_x86_instruction_alu_mem( io_generated_code,
            i_micro_kernel_config->alu_mov_instruction,
            i_gp_reg_mapping->gp_reg_b_ptrs,
            i_gp_reg_mapping->gp_reg_reduce_loop, 8,
            0,
            i_gp_reg_mapping->gp_reg_b,
            0 );
      } else if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) {
        /* Calculate to reg_b the proper address based on the reduce loop index */
        libxsmm_x86_instruction_alu_mem( io_generated_code,
            i_micro_kernel_config->alu_mov_instruction,
            i_gp_reg_mapping->gp_reg_b_offset,
            i_gp_reg_mapping->gp_reg_reduce_loop, 8,
            0,
            i_gp_reg_mapping->gp_reg_b,
            0 );
        libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_b_base, i_gp_reg_mapping->gp_reg_b);

        /* Calculate to reg_a the proper address based on the reduce loop index */
        libxsmm_x86_instruction_alu_mem( io_generated_code,
            i_micro_kernel_config->alu_mov_instruction,
            i_gp_reg_mapping->gp_reg_a_offset,
            i_gp_reg_mapping->gp_reg_reduce_loop, 8,
            0,
            i_gp_reg_mapping->gp_reg_a,
            0 );
        if (i_micro_kernel_config->decompress_A == 1) {
          libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_count);
          libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop);
          libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
          libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0);
          libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BITMAP_PTR, i_gp_reg_mapping->gp_reg_bitmap_a );
          libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_DECOMPRESS_BUF, i_gp_reg_mapping->gp_reg_decompressed_a);
          libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_help_0, i_micro_kernel_config->sparsity_factor_A);
          libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_decompressed_a);
          libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SARQ, i_gp_reg_mapping->gp_reg_help_0, 4);
          libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_bitmap_a);
          libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a_base, i_gp_reg_mapping->gp_reg_a);
          libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
        } else {
          libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a_base, i_gp_reg_mapping->gp_reg_a);
        }
      } else if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) {
        libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, i_gp_reg_mapping->gp_reg_a);
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_a, i_xgemm_desc->c1);
        libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a_base, i_gp_reg_mapping->gp_reg_a);
        libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, i_gp_reg_mapping->gp_reg_b);
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_b, i_xgemm_desc->c2);
        libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_b_base, i_gp_reg_mapping->gp_reg_b);
        if (i_micro_kernel_config->norm_to_normT_B_ext_buf == 1) {
          libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
          libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
          libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, i_gp_reg_mapping->gp_reg_help_0);
          libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANS_EXT_BUF_B, i_gp_reg_mapping->gp_reg_help_1 );
          libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_help_0, i_micro_kernel_config->stride_b_trans);
          libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_help_0);
          libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANS_EXT_BUF_B, i_gp_reg_mapping->gp_reg_help_0 );
          libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
          libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
        }
        if (i_micro_kernel_config->decompress_A == 1) {
          libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_count);
          libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop);
          libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
          libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
          libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, i_gp_reg_mapping->gp_reg_help_0);
          libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, i_gp_reg_mapping->gp_reg_help_1);
          libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BITMAP_PTR, i_gp_reg_mapping->gp_reg_bitmap_a );
          libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_xgemm_desc->c1*i_micro_kernel_config->sparsity_factor_A)/16);
          libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_bitmap_a);
          libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_DECOMPRESS_BUF, i_gp_reg_mapping->gp_reg_decompressed_a);
          libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_help_1, ((long long)i_xgemm_desc->c1*i_micro_kernel_config->sparsity_factor_A));
          libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_decompressed_a);
          libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
          libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
        }
      }

      /* Here is the K loop along with the microkernel */
      l_generator_kloop(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, n_blocking_info, &m_blocking_info[l_m_count], 0, 0, 0);

      /* TODO: this code is dead: In case of address based batch redcue push the proper A/B address updates if the k loop is not fully unrolled */
#if 0
      if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) && (fully_unroll_k == 0)) {
        libxsmm_x86_instruction_alu_mem( io_generated_code,
            i_micro_kernel_config->alu_mov_instruction,
            i_gp_reg_mapping->gp_reg_b_ptrs,
            i_gp_reg_mapping->gp_reg_reduce_loop, 8,
            0,
            i_gp_reg_mapping->gp_reg_b,
            1 );
        libxsmm_x86_instruction_alu_mem( io_generated_code,
            i_micro_kernel_config->alu_mov_instruction,
            i_gp_reg_mapping->gp_reg_a_ptrs,
            i_gp_reg_mapping->gp_reg_reduce_loop, 8,
            0,
            i_gp_reg_mapping->gp_reg_a,
            1 );
      }
#endif

      if (i_micro_kernel_config->decompress_A == 1) {
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop);
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_count);
      }
      libxsmm_generator_gemm_footer_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc);

      libxsmm_generator_gemm_store_C_amx( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, n_blocking_info, &m_blocking_info[l_m_count] );

      /* NON_UNROLLED_BR_LOOP_LABEL_END */
      libxsmm_x86_instruction_register_jump_label(io_generated_code, NON_UNROLLED_BR_LOOP_LABEL_END, p_jump_label_tracker);
    } else {
      /* Here is the K loop along with the microkernel */
      l_generator_kloop(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, n_blocking_info, &m_blocking_info[l_m_count], 0, 0, 0);
      libxsmm_generator_gemm_store_C_amx( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, n_blocking_info, &m_blocking_info[l_m_count] );
    }

    if (m_assembly_loop_exists) {
      libxsmm_generator_gemm_footer_mloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, l_m_blocking, l_m_done, fully_unroll_k );
    }
    l_m_count++;
  }

#if defined(LIBXSMM_GENERATOR_GEMM_AMX_JUMP_LABEL_TRACKER_MALLOC)
  free(p_jump_label_tracker);
#endif
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_kernel_nloop( libxsmm_generated_code*            io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    libxsmm_gp_reg_mapping*            i_gp_reg_mapping,
    libxsmm_micro_kernel_config*       i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    libxsmm_blocking_info_t*           n_blocking_info,
    libxsmm_blocking_info_t*           m_blocking_info ) {

  /* initialize n-blocking */
  unsigned int l_n_count = 0;          /* array counter for blocking info  arrays */
  unsigned int l_n_done = 0;           /* progress tracker */
  unsigned int l_n_blocking = 0;
  unsigned int l_n_block_size = 0;
  unsigned int m_assembly_loop_exists = (m_blocking_info[0].blocking < (unsigned int)i_xgemm_desc->m) ? 1 : 0;
  unsigned int n_assembly_loop_exists = (n_blocking_info[0].blocking < (unsigned int)i_xgemm_desc->n) ? 1 : 0;

  i_micro_kernel_config->m_loop_exists = m_assembly_loop_exists;
  i_micro_kernel_config->n_loop_exists = n_assembly_loop_exists;

  /* apply n_blocking */
  while (l_n_done != (unsigned int)i_xgemm_desc->n) {
    l_n_blocking = n_blocking_info[l_n_count].blocking;
    l_n_block_size = n_blocking_info[l_n_count].block_size;
    /* advance N */
    l_n_done += l_n_block_size;

    if (l_n_blocking < i_xgemm_desc->n) {
      /* Open N loop */
      libxsmm_generator_gemm_header_nloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, l_n_blocking );
    }

    /* Generate M loop */
    libxsmm_generator_gemm_amx_kernel_mloop(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, &n_blocking_info[l_n_count], m_blocking_info);

    if (l_n_blocking < i_xgemm_desc->n) {
      /* Close N loop */
      libxsmm_generator_gemm_footer_nloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, l_n_blocking, l_n_done, m_assembly_loop_exists );
    }
    l_n_count++;
  }
}

