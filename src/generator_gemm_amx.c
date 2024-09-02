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

LIBXSMM_API_INTERN
void libxsmm_generator_brgemm_amx_set_gp_reg_scf( libxsmm_generated_code*             io_generated_code,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    libxsmm_micro_kernel_config*       i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    unsigned int                       i_tmp_reg,
    unsigned int                       i_unrolled_index ) {
  if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0) {
    /* Move base pointer of scf to tmp  */
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_tmp_reg );
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, i_gp_reg_mapping->gp_reg_scf);
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_scf, (i_xgemm_desc->c1/16));
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_tmp_reg, i_gp_reg_mapping->gp_reg_scf);
    if (i_unrolled_index > 0) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_scf, i_unrolled_index*(i_xgemm_desc->c1/16));
    }
  } else if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_tmp_reg );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_tmp_reg,
        i_gp_reg_mapping->gp_reg_reduce_loop, 8,
        i_unrolled_index*8,
        i_gp_reg_mapping->gp_reg_scf,
        0 );
  } else if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_tmp_reg );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_a_offset,
        i_gp_reg_mapping->gp_reg_reduce_loop, 8,
        i_unrolled_index*8,
        i_gp_reg_mapping->gp_reg_scf,
        0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SARQ, i_gp_reg_mapping->gp_reg_scf, 4);
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_tmp_reg, i_gp_reg_mapping->gp_reg_scf);
  } else {
    /* Should not happen */
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_brgemm_amx_set_gp_reg_zpt( libxsmm_generated_code*             io_generated_code,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    libxsmm_micro_kernel_config*       i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    unsigned int                       i_tmp_reg,
    unsigned int                       i_unrolled_index ) {
  if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, LIBXSMM_X86_GP_REG_RDX);
    if (i_unrolled_index > 0) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, LIBXSMM_X86_GP_REG_RDX, i_unrolled_index);
    }
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, LIBXSMM_X86_GP_REG_RDX, i_xgemm_desc->lda);
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_tmp_reg );
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, LIBXSMM_X86_GP_REG_RDX, i_tmp_reg);
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_BRGEMM_PTR, i_tmp_reg );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
  } else if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_tmp_reg );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_tmp_reg,
        i_gp_reg_mapping->gp_reg_reduce_loop, 8,
        i_unrolled_index*8,
        i_tmp_reg,
        0 );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_BRGEMM_PTR, i_tmp_reg );
  } else if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RDX, 0);
    libxsmm_x86_instruction_alu_mem( io_generated_code,
         i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_a_offset,
        i_gp_reg_mapping->gp_reg_reduce_loop, 8,
        i_unrolled_index*8,
        LIBXSMM_X86_GP_REG_RAX,
        0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_tmp_reg, i_xgemm_desc->k);
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SARQ, i_tmp_reg, 1);
    libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_IDIVQ, LIBXSMM_X86_GP_REG_UNDEF, i_tmp_reg);
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_tmp_reg );
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, LIBXSMM_X86_GP_REG_RAX, i_tmp_reg);
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_BRGEMM_PTR, i_tmp_reg );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
  } else {
    /* Should not happen */
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_brgemm_amx_set_gp_reg_a( libxsmm_generated_code*             io_generated_code,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    libxsmm_micro_kernel_config*       i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    unsigned int                       i_unrolled_index ) {
  if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, i_gp_reg_mapping->gp_reg_a);
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_a, i_xgemm_desc->c1);
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a_base, i_gp_reg_mapping->gp_reg_a);
    if (i_unrolled_index > 0) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, i_unrolled_index*i_xgemm_desc->c1);
    }
    i_micro_kernel_config->br_loop_index = i_unrolled_index;
  } else if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_a_ptrs,
        i_gp_reg_mapping->gp_reg_reduce_loop, 8,
        i_unrolled_index*8,
        i_gp_reg_mapping->gp_reg_a,
        0 );
  } else if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_a_offset,
        i_gp_reg_mapping->gp_reg_reduce_loop, 8,
        i_unrolled_index*8,
        i_gp_reg_mapping->gp_reg_a,
        0 );
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a_base, i_gp_reg_mapping->gp_reg_a);
    i_micro_kernel_config->br_loop_index = i_unrolled_index;
  } else {
    /* Should not happen */
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_dequant_loop_amx( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    unsigned int                       cnt_reg ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, cnt_reg, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_dequant_loop_amx( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    unsigned int                       cnt_reg,
    unsigned int                       n_iters) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, cnt_reg, 1);
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, cnt_reg, n_iters );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_API_INTERN void libxsmm_generator_gemm_decompress_i4_vreg ( libxsmm_generated_code*            io_generated_code,
                                                                    const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                    const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                    unsigned int                       i_zpt_vreg,
                                                                    unsigned int                       io_vreg0,
                                                                    unsigned int                       o_vreg1 ) {
  unsigned int l_subtract_zpt = 0;
  unsigned int tmp_lo_vreg = io_vreg0;
  unsigned int tmp_hi_vreg = o_vreg1;
  unsigned int l_vreg_mask_0f = i_micro_kernel_config->mask_lo_i4;
  unsigned int l_vreg_mask_f0 = i_micro_kernel_config->mask_hi_i4;
  if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0 || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0 ) {
    l_subtract_zpt = 1;
  }
  /* Mask higher 4 bits*/
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,  LIBXSMM_X86_INSTR_VPANDD, i_micro_kernel_config->vector_name, tmp_lo_vreg, l_vreg_mask_f0, tmp_hi_vreg);
  /* Shift and Subtract zero pts */
  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSRLD_I, i_micro_kernel_config->vector_name, tmp_hi_vreg, tmp_hi_vreg, 4);
  if (l_subtract_zpt > 0) {
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPSUBB, i_micro_kernel_config->vector_name, i_zpt_vreg, tmp_hi_vreg, tmp_hi_vreg);
  }
  /* Mask lower 4 bits*/
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,  LIBXSMM_X86_INSTR_VPANDD, i_micro_kernel_config->vector_name, tmp_lo_vreg, l_vreg_mask_0f, tmp_lo_vreg);
  /* Subtract zero pts */
  if (l_subtract_zpt > 0) {
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPSUBB, i_micro_kernel_config->vector_name, i_zpt_vreg, tmp_lo_vreg, tmp_lo_vreg);
  }
}

LIBXSMM_API_INTERN void libxsmm_generator_gemm_convert_KxM_fp8_to_bf16( libxsmm_generated_code*            io_generated_code,
                                                                         libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                         const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                         unsigned int                       i_m_tiles,
                                                                         unsigned int                       i_K,
                                                                         unsigned int                       i_ldi,
                                                                         unsigned int                       i_ldo,
                                                                         unsigned int                       i_gp_reg,
                                                                         unsigned int                       o_gp_reg ) {
  unsigned int im = 0, l_vlen = 32;
  unsigned int l_vreg_start = i_micro_kernel_config->reserved_zmms;
  unsigned int cnt_reg = LIBXSMM_X86_GP_REG_R11;
  libxsmm_x86_instruction_push_reg( io_generated_code, cnt_reg );
  libxsmm_generator_gemm_header_dequant_loop_amx( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, cnt_reg );
  for (im = 0; im < i_m_tiles; im += 2) {
    unsigned int l_vreg_lo = l_vreg_start + (im+0) % (32-l_vreg_start);
    unsigned int l_vreg_hi = l_vreg_start + (im+1) % (32-l_vreg_start);
    unsigned int l_process_even_half = ((im + 2 < i_m_tiles) || i_m_tiles == 2 || ((im + 2 >= i_m_tiles-1) && (i_m_tiles > 3))) ? 1 : 0;

    libxsmm_x86_instruction_unified_vec_move( io_generated_code,
        LIBXSMM_X86_INSTR_VMOVDQU16,
        i_gp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
        (im/2) * l_vlen * 2,
        i_micro_kernel_config->vector_name,
        l_vreg_lo, (im + 2 >= i_m_tiles) ? i_micro_kernel_config->mask_m_lp_cvt : 0, i_micro_kernel_config->mask_m_lp_cvt, 0 );

    if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0) {
      libxsmm_x86_instruction_prefetch(io_generated_code,
          LIBXSMM_X86_INSTR_PREFETCHT0,
          i_gp_reg,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (int)((long long)im * l_vlen * 2 + (long long)i_xgemm_desc->c1) );
    }

    /* Extract high part to i_vreg_hi */
    if (l_process_even_half) {
      libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( io_generated_code, LIBXSMM_X86_INSTR_VEXTRACTI32X8, i_micro_kernel_config->vector_name, l_vreg_lo, LIBXSMM_X86_VEC_REG_UNDEF, l_vreg_hi, 0, 0, 0, 1 );
    }
#if 1
    libxsmm_generator_cvt_8bit_to_16bit_lut_prepped_regs_avx512( io_generated_code, i_micro_kernel_config->vector_name,
        l_vreg_lo, l_vreg_lo,
        i_micro_kernel_config->luth_reg0,
        i_micro_kernel_config->luth_reg1,
        i_micro_kernel_config->lutl_reg0,
        i_micro_kernel_config->lutl_reg1,
        i_micro_kernel_config->sign_reg,
        i_micro_kernel_config->blend_reg,
        i_micro_kernel_config->tmp_reg0,
        i_micro_kernel_config->tmp_reg1 );

    if (l_process_even_half) {
      libxsmm_generator_cvt_8bit_to_16bit_lut_prepped_regs_avx512( io_generated_code, i_micro_kernel_config->vector_name,
          l_vreg_hi, l_vreg_hi,
          i_micro_kernel_config->luth_reg0,
          i_micro_kernel_config->luth_reg1,
          i_micro_kernel_config->lutl_reg0,
          i_micro_kernel_config->lutl_reg1,
          i_micro_kernel_config->sign_reg,
          i_micro_kernel_config->blend_reg,
          i_micro_kernel_config->tmp_reg0,
          i_micro_kernel_config->tmp_reg1 );
    }
#else
    libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VEXTRACTI128, 'y',
                                                   l_vreg_lo, i_micro_kernel_config->tmp_reg0, 0x1 );
    libxsmm_generator_cvtbf8bf16_avx512( io_generated_code, i_micro_kernel_config->vector_name, l_vreg_lo, l_vreg_lo );
    libxsmm_generator_cvtbf8bf16_avx512( io_generated_code, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_reg0, i_micro_kernel_config->tmp_reg0 );
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VINSERTI64X4, 'z', i_micro_kernel_config->tmp_reg0, l_vreg_lo, l_vreg_lo, 0, 0, 0, 1 );

    libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VEXTRACTI128, 'y',
                                                   l_vreg_hi, i_micro_kernel_config->tmp_reg0, 0x1 );
    libxsmm_generator_cvtbf8bf16_avx512( io_generated_code, i_micro_kernel_config->vector_name, l_vreg_hi, l_vreg_hi );
    libxsmm_generator_cvtbf8bf16_avx512( io_generated_code, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_reg0, i_micro_kernel_config->tmp_reg0 );
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VINSERTI64X4, 'z', i_micro_kernel_config->tmp_reg0, l_vreg_hi, l_vreg_hi, 0, 0, 0, 1 );
#endif

    libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set,
        LIBXSMM_X86_INSTR_VMOVUPS,
        o_gp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
        im * l_vlen * 2,
        i_micro_kernel_config->vector_name,
        l_vreg_lo, ((im + 2 >= i_m_tiles) && (l_process_even_half == 0)) ? i_micro_kernel_config->mask_m_fp32 : 0, 0, 1 );

    if (l_process_even_half) {
      libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VMOVUPS,
          o_gp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
          im * l_vlen * 2 + 64,
          i_micro_kernel_config->vector_name,
          l_vreg_hi, (im + 2 >= i_m_tiles) ? i_micro_kernel_config->mask_m_fp32 : 0, 0, 1 );
    }
  }

  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg, (long long)i_ldi * 2 );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, o_gp_reg, (long long)i_ldo * 4 );
  libxsmm_generator_gemm_footer_dequant_loop_amx( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, cnt_reg, i_K/2);
  libxsmm_x86_instruction_pop_reg( io_generated_code, cnt_reg );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg, (long long)i_ldi * i_K );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, o_gp_reg, (long long)i_ldo * 2 * i_K );
}

LIBXSMM_API_INTERN void libxsmm_generator_gemm_decompress_KxM_i4_tensor( libxsmm_generated_code*            io_generated_code,
                                                                         libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                         const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                         unsigned int                       i_m_tiles,
                                                                         unsigned int                       i_K,
                                                                         unsigned int                       i_ldi,
                                                                         unsigned int                       i_ldo,
                                                                         unsigned int                       i_gp_reg,
                                                                         unsigned int                       o_gp_reg ) {
  unsigned int im = 0, l_vlen = 16;
  unsigned int l_vreg_start = i_micro_kernel_config->reserved_zmms;

#if 0
  unsigned int ik = 0;
  /* Int4 matrix is in vnni8 interleaved format */
  for (ik = 0; ik < i_K; ik += 8) {
    for (im = 0; im < i_m_tiles; im++) {
      unsigned int l_vreg_lo = l_vreg_start + (im+0) % (32-l_vreg_start);
      unsigned int l_vreg_hi = l_vreg_start + (im+1) % (32-l_vreg_start);
      unsigned int l_zpt_vreg = 3 + im;

      libxsmm_x86_instruction_unified_vec_move( io_generated_code,
          LIBXSMM_X86_INSTR_VMOVUPS,
          i_gp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
          im * l_vlen * 4 * i_micro_kernel_config->datatype_size_in + (ik/8) * i_ldi * 4 * i_micro_kernel_config->datatype_size_in,
          i_micro_kernel_config->vector_name,
          l_vreg_lo, (im == i_m_tiles-1) ? i_micro_kernel_config->mask_m_fp32 : 0, 1, 0 );

      libxsmm_generator_gemm_decompress_i4_vreg ( io_generated_code, i_micro_kernel_config, i_xgemm_desc, l_zpt_vreg, l_vreg_lo, l_vreg_hi );

      libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VMOVUPS,
          o_gp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
          im * l_vlen * 4 * i_micro_kernel_config->datatype_size_in + (ik/4+0) * i_ldo * 4 * i_micro_kernel_config->datatype_size_in,
          i_micro_kernel_config->vector_name,
          l_vreg_lo, (im == i_m_tiles-1) ? i_micro_kernel_config->mask_m_fp32 : 0, 0, 1 );

      libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VMOVUPS,
          o_gp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
          im * l_vlen * 4 * i_micro_kernel_config->datatype_size_in + (ik/4+1) * i_ldo * 4 * i_micro_kernel_config->datatype_size_in,
          i_micro_kernel_config->vector_name,
          l_vreg_hi, (im == i_m_tiles-1) ? i_micro_kernel_config->mask_m_fp32 : 0, 0, 1 );
    }
  }
#else
  unsigned int cnt_reg = LIBXSMM_X86_GP_REG_R11;
  unsigned int gp_reg_zpt = cnt_reg;
  libxsmm_x86_instruction_push_reg( io_generated_code, cnt_reg );
  /* Load m_tiles ZPT registers */
  if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0 ) {
    libxsmm_blocking_info_t  m_blocking_info[2];
    unsigned int i_m_offset = 0;
    m_blocking_info[0] = i_micro_kernel_config->m_blocking_info[0];
    m_blocking_info[1] = i_micro_kernel_config->m_blocking_info[1];
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_BRGEMM_PTR, gp_reg_zpt );
    for ( im = 0; im < i_m_tiles; im++ ) {
      unsigned int l_vreg = 3 + im;
      libxsmm_x86_instruction_unified_vec_move( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU8,
          gp_reg_zpt, LIBXSMM_X86_GP_REG_UNDEF, 0,
          i_m_offset * i_micro_kernel_config->datatype_size_in,
          'x',
          l_vreg, (im == i_m_tiles-1) ? i_micro_kernel_config->mask_m_fp32 : 0, 1, 0 );
      libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPERMB, i_micro_kernel_config->vector_name, l_vreg, i_micro_kernel_config->perm_table_zpt_bcast, l_vreg);
      i_m_offset += m_blocking_info->sizes[im];
    }
  }
  libxsmm_generator_gemm_header_dequant_loop_amx( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, cnt_reg );
  for (im = 0; im < i_m_tiles; im++) {
    unsigned int l_vreg_lo = l_vreg_start + (im+0) % (32-l_vreg_start);
    unsigned int l_vreg_hi = l_vreg_start + (im+1) % (32-l_vreg_start);
    unsigned int l_zpt_vreg = 3 + im;

    libxsmm_x86_instruction_unified_vec_move( io_generated_code,
        LIBXSMM_X86_INSTR_VMOVUPS,
        i_gp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
        im * l_vlen * 4 * i_micro_kernel_config->datatype_size_in,
        i_micro_kernel_config->vector_name,
        l_vreg_lo, (im == i_m_tiles-1) ? i_micro_kernel_config->mask_m_fp32 : 0, 1, 0 );

    if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0) {
      libxsmm_x86_instruction_prefetch(io_generated_code,
          LIBXSMM_X86_INSTR_PREFETCHT0,
          i_gp_reg,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (int)((long long)im * l_vlen * 4 * i_micro_kernel_config->datatype_size_in + (long long)i_xgemm_desc->c1) );
    }

    libxsmm_generator_gemm_decompress_i4_vreg ( io_generated_code, i_micro_kernel_config, i_xgemm_desc, l_zpt_vreg, l_vreg_lo, l_vreg_hi );

    libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set,
        LIBXSMM_X86_INSTR_VMOVUPS,
        o_gp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
        im * l_vlen * 4 * i_micro_kernel_config->datatype_size_in,
        i_micro_kernel_config->vector_name,
        l_vreg_lo, (im == i_m_tiles-1) ? i_micro_kernel_config->mask_m_fp32 : 0, 0, 1 );

    libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set,
        LIBXSMM_X86_INSTR_VMOVUPS,
        o_gp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
        im * l_vlen * 4 * i_micro_kernel_config->datatype_size_in + i_ldo * 4 * i_micro_kernel_config->datatype_size_in,
        i_micro_kernel_config->vector_name,
        l_vreg_hi, (im == i_m_tiles-1) ? i_micro_kernel_config->mask_m_fp32 : 0, 0, 1 );
  }
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg, (long long)i_ldi * 4 * i_micro_kernel_config->datatype_size_in );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, o_gp_reg, (long long)i_ldo * 8 * i_micro_kernel_config->datatype_size_in );
  libxsmm_generator_gemm_footer_dequant_loop_amx( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, cnt_reg, i_K/8);
  libxsmm_x86_instruction_pop_reg( io_generated_code, cnt_reg );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg, (long long)i_ldi * 4 * (i_K/8) * i_micro_kernel_config->datatype_size_in );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, o_gp_reg, (long long)i_ldo * 8 * (i_K/8) * i_micro_kernel_config->datatype_size_in );
#endif
}

LIBXSMM_API_INTERN void libxsmm_generator_gemm_decompress_KxM_mxfp4_tensor( libxsmm_generated_code*         io_generated_code,
                                                                         libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                         const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                         unsigned int                       i_m_tiles,
                                                                         unsigned int                       i_K,
                                                                         unsigned int                       i_ldi,
                                                                         unsigned int                       i_ldo,
                                                                         unsigned int                       i_gp_reg,
                                                                         unsigned int                       i_gp_scf,
                                                                         unsigned int                       o_gp_reg ) {
  unsigned int im = 0, ik = 0, l_vlen = 16;
  unsigned int l_vreg_start = i_micro_kernel_config->reserved_zmms;
  unsigned int l_mask_expon = i_micro_kernel_config->reserved_mask_regs;
  unsigned int l_mask_start = i_micro_kernel_config->reserved_mask_regs+(i_m_tiles+1)/2;
  unsigned int cnt_reg = LIBXSMM_X86_GP_REG_R11;
  unsigned int u_ik = 0, k_unroll = 2;

  for (ik = 0; ik < (i_K/32); ik++) {
    /* Here we load the scaling factors for the upcoming 32-K values */
    for (im = 0; im < i_m_tiles; im += 2) {
      unsigned int l_scf_vreg = i_micro_kernel_config->reserved_zmms + im/2;
      l_mask_expon = i_micro_kernel_config->reserved_mask_regs + im/2;
      libxsmm_x86_instruction_unified_vec_move( io_generated_code,
          LIBXSMM_X86_INSTR_VMOVDQU8,
          i_gp_scf, LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * l_vlen + ik * i_ldi) * i_micro_kernel_config->datatype_size_in,
          'y',
          l_scf_vreg, (im + 2 >= i_m_tiles) ? i_micro_kernel_config->mask_m_lp_cvt : 0, i_micro_kernel_config->mask_m_lp_cvt, 0 );
      if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0 && (((im * l_vlen + ik * i_ldi) * i_micro_kernel_config->datatype_size_in) % 64 == 0)) {
        libxsmm_x86_instruction_prefetch(io_generated_code,
            LIBXSMM_X86_INSTR_PREFETCHT0,
            i_gp_scf,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (int)(((long long)(im * l_vlen + ik * i_ldi)) * i_micro_kernel_config->datatype_size_in + (long long)(i_xgemm_desc->c1/16)) );
      }
      libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VPMOVZXBW, i_micro_kernel_config->vector_name, l_scf_vreg, l_scf_vreg);
      libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPCMPW, i_micro_kernel_config->vector_name, i_micro_kernel_config->zero_reg, l_scf_vreg, l_mask_expon, 0 );
      libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPSUBW, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_ones, l_scf_vreg, l_scf_vreg);
      libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code, LIBXSMM_X86_INSTR_VPSLLW_I, i_micro_kernel_config->vector_name, l_scf_vreg, l_scf_vreg, 7);
    }
    l_vreg_start = i_micro_kernel_config->reserved_zmms + (i_m_tiles+1)/2;
    libxsmm_x86_instruction_push_reg( io_generated_code, cnt_reg );
    libxsmm_generator_gemm_header_dequant_loop_amx( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, cnt_reg );
    for (u_ik = 0; u_ik < k_unroll; u_ik++) {
      for (im = 0; im < i_m_tiles; im += 2) {
        unsigned int l_process_hi_half = ((im + 2 < i_m_tiles) || i_m_tiles == 2 || ((im + 2 >= i_m_tiles-1) && (i_m_tiles > 3))) ? 1 : 0;
        unsigned int l_vreg_lo = l_vreg_start + (u_ik*i_m_tiles*3+im*3+0) % (32-l_vreg_start);
        unsigned int l_vreg_hi = l_vreg_start + (u_ik*i_m_tiles*3+im*3+1) % (32-l_vreg_start);
        unsigned int l_vreg_cpy_lo = l_vreg_start + (u_ik*i_m_tiles*3+im*3+2) % (32-l_vreg_start);
        unsigned int l_scf_vreg = i_micro_kernel_config->reserved_zmms + im/2;
        unsigned int l_mask_reg_even = l_mask_start + (2*u_ik+0) % (7-l_mask_start+1);
        unsigned int l_mask_reg_odd  = l_mask_start + (2*u_ik+1) % (7-l_mask_start+1);
        l_mask_expon = i_micro_kernel_config->reserved_mask_regs + im/2;
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            LIBXSMM_X86_INSTR_VMOVDQU8,
            i_gp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * l_vlen + ik * 16 * i_ldi + u_ik * i_ldi) * i_micro_kernel_config->datatype_size_in,
            'y',
            l_vreg_lo, (im + 2 >= i_m_tiles) ? i_micro_kernel_config->mask_m_lp_cvt : 0, i_micro_kernel_config->mask_m_lp_cvt, 0 );
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0 && (((im * l_vlen + ik * 16 * i_ldi + u_ik * i_ldi) * i_micro_kernel_config->datatype_size_in) % 64 == 0)) {
          libxsmm_x86_instruction_prefetch(io_generated_code,
              LIBXSMM_X86_INSTR_PREFETCHT0,
              i_gp_reg,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (int)((long long)(im * l_vlen + ik * 16 * i_ldi + u_ik * i_ldi) * i_micro_kernel_config->datatype_size_in + (long long)(i_xgemm_desc->c1)) );
        }
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VPMOVZXBW, i_micro_kernel_config->vector_name, l_vreg_lo, l_vreg_lo);
        libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSRLD_I, i_micro_kernel_config->vector_name, l_vreg_lo, l_vreg_hi, 4);
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPERMW, i_micro_kernel_config->vector_name, i_micro_kernel_config->luth_reg0, l_vreg_lo, l_vreg_lo);
        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPCMPW, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_ones, l_vreg_lo, l_mask_reg_even, 0 );
        libxsmm_x86_instruction_mask_compute_reg( io_generated_code, LIBXSMM_X86_INSTR_KORD, l_mask_reg_even, l_mask_expon, l_mask_reg_even, 0);
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPADDW, i_micro_kernel_config->vector_name, l_scf_vreg, l_vreg_lo, l_vreg_lo);
        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code, LIBXSMM_X86_INSTR_VPBLENDMW, i_micro_kernel_config->vector_name, i_micro_kernel_config->zero_reg, l_vreg_lo, l_vreg_lo, l_mask_reg_even, 0 );
        if (l_process_hi_half > 0) {
          libxsmm_x86_instruction_vec_compute_2reg(io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64, i_micro_kernel_config->vector_name, l_vreg_lo, l_vreg_cpy_lo);
        }
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPERMW, i_micro_kernel_config->vector_name, i_micro_kernel_config->luth_reg0, l_vreg_hi, l_vreg_hi);
        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPCMPW, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_ones, l_vreg_hi, l_mask_reg_odd, 0 );
        libxsmm_x86_instruction_mask_compute_reg( io_generated_code, LIBXSMM_X86_INSTR_KORD, l_mask_reg_odd, l_mask_expon, l_mask_reg_odd, 0);
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPADDW, i_micro_kernel_config->vector_name, l_scf_vreg, l_vreg_hi, l_vreg_hi);
        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code, LIBXSMM_X86_INSTR_VPBLENDMW, i_micro_kernel_config->vector_name, i_micro_kernel_config->zero_reg, l_vreg_hi, l_vreg_hi, l_mask_reg_odd, 0 );
        libxsmm_x86_instruction_vec_compute_3reg(io_generated_code, LIBXSMM_X86_INSTR_VPERMT2W, i_micro_kernel_config->vector_name, l_vreg_hi, i_micro_kernel_config->perm_table_vnni_lo, l_vreg_lo);
        if (l_process_hi_half > 0) {
          libxsmm_x86_instruction_vec_compute_3reg(io_generated_code, LIBXSMM_X86_INSTR_VPERMT2W, i_micro_kernel_config->vector_name, l_vreg_hi, i_micro_kernel_config->perm_table_vnni_hi, l_vreg_cpy_lo);
        }
      }
    }

    for (u_ik = 0; u_ik < k_unroll; u_ik++) {
      for (im = 0; im < i_m_tiles; im += 2) {
        unsigned int l_process_hi_half = ((im + 2 < i_m_tiles) || i_m_tiles == 2 || ((im + 2 >= i_m_tiles-1) && (i_m_tiles > 3))) ? 1 : 0;
        unsigned int l_vreg_lo = l_vreg_start + (u_ik*i_m_tiles*3+im*3+0) % (32-l_vreg_start);
        unsigned int l_vreg_cpy_lo = l_vreg_start + (u_ik*i_m_tiles*3+im*3+2) % (32-l_vreg_start);
        libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMOVUPS,
            o_gp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * l_vlen + ik * 16 * i_ldo + u_ik * i_ldo) * 4 * i_micro_kernel_config->datatype_size_in,
            i_micro_kernel_config->vector_name,
            l_vreg_lo, ((im + 2 >= i_m_tiles) && (l_process_hi_half == 0)) ? i_micro_kernel_config->mask_m_fp32 : 0, 0, 1 );
        if (l_process_hi_half > 0) {
          libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVUPS,
              o_gp_reg, LIBXSMM_X86_GP_REG_UNDEF, 0,
              ((im+1) * l_vlen + ik * 16 * i_ldo + u_ik * i_ldo) * 4 * i_micro_kernel_config->datatype_size_in,
              i_micro_kernel_config->vector_name,
              l_vreg_cpy_lo, (im + 2 >= i_m_tiles) ? i_micro_kernel_config->mask_m_fp32 : 0, 0, 1 );
        }
      }
    }


    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg, (long long)i_ldi * k_unroll *  i_micro_kernel_config->datatype_size_in );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, o_gp_reg, (long long)i_ldo * k_unroll * 4 * i_micro_kernel_config->datatype_size_in );
    libxsmm_generator_gemm_footer_dequant_loop_amx( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, cnt_reg, 16/k_unroll);
    libxsmm_x86_instruction_pop_reg( io_generated_code, cnt_reg );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg, (long long)i_ldi * 16 * i_micro_kernel_config->datatype_size_in );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, o_gp_reg, (long long)i_ldo * 64 * i_micro_kernel_config->datatype_size_in );
  }
}

LIBXSMM_API_INTERN
void libxsmm_get_tileinfo( unsigned int tile_id, unsigned int *n_rows, unsigned int *n_cols, libxsmm_tile_config *tc) {
  *n_rows = 0;
  *n_cols = 0;
  switch (tile_id) {
    case 0:
      (*n_rows) = (int) tc->tile0rowsb/4;
      (*n_cols) = (int) tc->tile0cols;
      break;
    case 1:
      (*n_rows) = (int) tc->tile1rowsb/4;
      (*n_cols) = (int) tc->tile1cols;
      break;
    case 2:
      (*n_rows) = (int) tc->tile2rowsb/4;
      (*n_cols) = (int) tc->tile2cols;
      break;
    case 3:
      (*n_rows) = (int) tc->tile3rowsb/4;
      (*n_cols) = (int) tc->tile3cols;
      break;
    case 4:
      (*n_rows) = (int) tc->tile4rowsb/4;
      (*n_cols) = (int) tc->tile4cols;
      break;
    case 5:
      (*n_rows) = (int) tc->tile5rowsb/4;
      (*n_cols) = (int) tc->tile5cols;
      break;
    case 6:
      (*n_rows) = (int) tc->tile6rowsb/4;
      (*n_cols) = (int) tc->tile6cols;
      break;
    case 7:
      (*n_rows) = (int) tc->tile7rowsb/4;
      (*n_cols) = (int) tc->tile7cols;
      break;
    default:
      LIBXSMM_ASSERT_MSG(0, "valid tile id");
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_reduceloop_amx( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_partially_unrolled_reduceloop_amx( libxsmm_generated_code*             io_generated_code,
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
void libxsmm_generator_gemm_footer_partially_unrolled_reduceloop_amx( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    unsigned int                       unroll_factor,
    unsigned int                       n_iters) {
  LIBXSMM_UNUSED(i_xgemm_desc);
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, unroll_factor);
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, n_iters );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_partially_unrolled_reduceloop_dynamic_amx( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    unsigned int                       unroll_factor,
    unsigned int                       loop_bound) {
  LIBXSMM_UNUSED(i_xgemm_desc);
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, unroll_factor);
  libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, loop_bound, i_gp_reg_mapping->gp_reg_reduce_loop);
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
  unsigned int l_is_Ai4_Bi8_gemm = libxsmm_x86_is_Ai4_Bi8_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bbf16_gemm = libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Abf8_Bbf16_gemm = libxsmm_x86_is_Abf8_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Ahf8_Bbf16_gemm = libxsmm_x86_is_Ahf8_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_a_packed_bytes = (l_is_Abf8_Bbf16_gemm > 0 || l_is_Ahf8_Bbf16_gemm > 0) ? 2 : ((l_is_Amxfp4_Bbf16_gemm > 0) ? 1 : 4);

  if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
    if (i_micro_kernel_config->vnni_format_C == 0) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c,
        ((long long)i_n_blocking*i_xgemm_desc->ldc*2 /*(i_micro_kernel_config->datatype_size/2)*/) - ((long long)i_xgemm_desc->m * i_m_loop_exists * 2 /*(i_micro_kernel_config->datatype_size/2)*/) );
    } else {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c,
        ((long long)i_n_blocking*i_xgemm_desc->ldc*2 /*(i_micro_kernel_config->datatype_size/2)*/) - ((long long)i_xgemm_desc->m * i_m_loop_exists * 2 * 2 /*(i_micro_kernel_config->datatype_size/2)*/) );
    }
  } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c,
        ((long long)i_n_blocking*i_xgemm_desc->ldc/**(i_micro_kernel_config->datatype_size/4)*/) - ((long long)i_xgemm_desc->m * i_m_loop_exists /** (i_micro_kernel_config->datatype_size/4)*/) );
  } else {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c,
        ((long long)i_n_blocking*i_xgemm_desc->ldc*4/*(i_micro_kernel_config->datatype_size)*/) - ((long long)i_xgemm_desc->m * i_m_loop_exists * 4 /*(i_micro_kernel_config->datatype_size)*/) );
  }

  /* Also adjust eltwise pointers */
  if ((i_micro_kernel_config->fused_relu == 1) || (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) || (i_micro_kernel_config->fused_relu_bwd == 1) || (i_micro_kernel_config->fused_bcolbias == 1) || (i_micro_kernel_config->fused_hcolbias == 1) || (i_micro_kernel_config->fused_scolbias == 1) || (i_micro_kernel_config->overwrite_C == 0)) {
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (l_is_Ai4_Bi8_gemm > 0 && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0) && i_m_loop_exists > 0) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)(i_xgemm_desc->m)*(i_micro_kernel_config->datatype_size_in) );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
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

  if (i_micro_kernel_config->fused_bcolbias == 1 || i_micro_kernel_config->fused_hcolbias == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_xgemm_desc->m * i_m_loop_exists * 2/*(i_micro_kernel_config->datatype_size/2)*/) );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->fused_scolbias == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_xgemm_desc->m * i_m_loop_exists * 4) );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if ((i_micro_kernel_config->fused_relu == 1) || (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) || (i_micro_kernel_config->fused_relu_bwd == 1) || (i_micro_kernel_config->fused_bcolbias == 1) || (i_micro_kernel_config->fused_hcolbias == 1) || (i_micro_kernel_config->fused_scolbias == 1) || (i_micro_kernel_config->overwrite_C == 0)) {
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
      l_b_offset = i_n_blocking * i_micro_kernel_config->datatype_size_in2 ;
    } else {
      l_b_offset = i_n_blocking * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2;
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
          i_gp_reg_mapping->gp_reg_a, ((long long)i_xgemm_desc->m*l_a_packed_bytes/*(i_micro_kernel_config->datatype_size)*/) );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_a_ptrs,
          i_gp_reg_mapping->gp_reg_reduce_loop, 8,
          0,
          i_gp_reg_mapping->gp_reg_a,
          1 );
      if (l_is_Amxfp4_Bbf16_gemm > 0) {
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
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
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      }
      if (l_is_Ai4_Bi8_gemm > 0 && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0)) {
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
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
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      }
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
      l_b_offset = i_n_blocking * i_micro_kernel_config->datatype_size_in2 ;
    } else {
      l_b_offset = i_n_blocking * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2 ;
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
          i_gp_reg_mapping->gp_reg_a_base, ((long long)i_xgemm_desc->m*l_a_packed_bytes/*(i_micro_kernel_config->datatype_size)*/)/i_micro_kernel_config->sparsity_factor_A );
      if (l_is_Amxfp4_Bbf16_gemm > 0) {
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_xgemm_desc->m));
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      }
      if (l_is_Ai4_Bi8_gemm > 0 && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0)) {
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_xgemm_desc->m));
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      }
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
      l_b_offset = i_n_blocking * i_micro_kernel_config->datatype_size_in2 ;
    } else {
      l_b_offset = i_n_blocking * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2 ;
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
          i_gp_reg_mapping->gp_reg_a, ((long long)i_xgemm_desc->m*l_a_packed_bytes/*(i_micro_kernel_config->datatype_size)*/)/i_micro_kernel_config->sparsity_factor_A );
      if (l_is_Amxfp4_Bbf16_gemm > 0) {
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_xgemm_desc->m));
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      }

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
            i_gp_reg_mapping->gp_reg_a_prefetch, ((long long)i_xgemm_desc->m*l_a_packed_bytes/*(i_micro_kernel_config->datatype_size)*/) );
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
  unsigned int l_is_Ai4_Bi8_gemm = libxsmm_x86_is_Ai4_Bi8_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bbf16_gemm = libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Abf8_Bbf16_gemm = libxsmm_x86_is_Abf8_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Ahf8_Bbf16_gemm = libxsmm_x86_is_Ahf8_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_a_packed_bytes = (l_is_Abf8_Bbf16_gemm > 0 || l_is_Ahf8_Bbf16_gemm > 0) ? 2 : ((l_is_Amxfp4_Bbf16_gemm > 0) ? 1 : 4);

  /* advance C pointer */
  if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ||  LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )  ) {
    if (i_micro_kernel_config->vnni_format_C == 0) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_c, (long long)i_m_blocking*2/*(i_micro_kernel_config->datatype_size/2)*/ );
    } else {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_c, (long long)i_m_blocking*2*2/*(i_micro_kernel_config->datatype_size/2)*/ );
    }
  } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_c, (long long)i_m_blocking/**(i_micro_kernel_config->datatype_size/4)*/ );
  } else {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_c, (long long)i_m_blocking*4/*(i_micro_kernel_config->datatype_size)*/ );
  }

  /* Also adjust eltwise pointers */
  if (l_is_Ai4_Bi8_gemm > 0 && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0)) {
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_m_blocking*i_micro_kernel_config->datatype_size_in);
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if ((i_micro_kernel_config->fused_relu == 1) || (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) || (i_micro_kernel_config->fused_relu_bwd == 1) || (i_micro_kernel_config->fused_bcolbias == 1) || (i_micro_kernel_config->fused_hcolbias == 1) || (i_micro_kernel_config->fused_scolbias == 1) || (i_micro_kernel_config->overwrite_C == 0)) {
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

  if (i_micro_kernel_config->fused_bcolbias == 1 || i_micro_kernel_config->fused_hcolbias == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_m_blocking * 2/*(i_micro_kernel_config->datatype_size/2)*/ );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->fused_scolbias == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_m_blocking * 4  );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if ((i_micro_kernel_config->fused_relu == 1) || (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) || (i_micro_kernel_config->fused_relu_bwd == 1) || (i_micro_kernel_config->fused_bcolbias == 1) || (i_micro_kernel_config->fused_hcolbias == 1) || (i_micro_kernel_config->fused_scolbias == 1) || (i_micro_kernel_config->overwrite_C == 0)) {
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
      if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) ||
           (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) ||
          (LIBXSMM_DATATYPE_I16  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ))    ) {
        l_type_scaling = 2;
      } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
        l_type_scaling = 4;
      } else {
        l_type_scaling = 1;
      }
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_b_prefetch, (long long)i_m_blocking*(4/l_type_scaling) );
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
              ((long long)i_m_blocking * l_a_packed_bytes /*(i_micro_kernel_config->datatype_size)*/) );
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
            ((long long)i_m_blocking * l_a_packed_bytes/*(i_micro_kernel_config->datatype_size)*/) );
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
          ((long long)i_xgemm_desc->k * i_micro_kernel_config->datatype_size_in/*(i_micro_kernel_config->datatype_size)*/ * i_xgemm_desc->lda ) - ((long long)i_m_blocking * l_a_packed_bytes/*(i_micro_kernel_config->datatype_size)*/) );
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
          0LL - ((long long)i_m_blocking * l_a_packed_bytes /*(i_micro_kernel_config->datatype_size)*/) );
    } else {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_a,
          ((long long)i_xgemm_desc->k * i_micro_kernel_config->datatype_size_in /*(i_micro_kernel_config->datatype_size)*/ * i_xgemm_desc->lda ) - ((long long)i_m_blocking * l_a_packed_bytes /*(i_micro_kernel_config->datatype_size)*/) );
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
            ((long long)i_m_blocking * l_a_packed_bytes/*(i_micro_kernel_config->datatype_size)*/) );
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
          ((long long)i_m_blocking * l_a_packed_bytes/*(i_micro_kernel_config->datatype_size)*/) );
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

      if (l_is_Amxfp4_Bbf16_gemm > 0) {
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_help_1 );
        libxsmm_generator_gemm_header_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config );
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
        libxsmm_generator_gemm_footer_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc);
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
      }
      if (l_is_Ai4_Bi8_gemm > 0 && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0)) {
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_gp_reg_mapping->gp_reg_help_1 );
        libxsmm_generator_gemm_header_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config );
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
        libxsmm_generator_gemm_footer_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc);
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
      }
    } else if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
      /* advance A pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a_base,
          ((long long)i_m_blocking * l_a_packed_bytes/*(i_micro_kernel_config->datatype_size)*/)/i_micro_kernel_config->sparsity_factor_A );
      if (l_is_Amxfp4_Bbf16_gemm > 0) {
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_m_blocking));
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      }
      if (l_is_Ai4_Bi8_gemm > 0 && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0)) {
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_m_blocking));
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      }
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
          ((long long)i_m_blocking * l_a_packed_bytes/*(i_micro_kernel_config->datatype_size)*/)/i_micro_kernel_config->sparsity_factor_A );
      if (l_is_Amxfp4_Bbf16_gemm > 0) {
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_m_blocking));
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      }
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
  unsigned int l_enforce_Mx1_amx_tile_blocking = (libxsmm_cpuid_x86_amx_gemm_enforce_mx1_tile_blocking() > 0) ? 1 : (i_xgemm_desc->n <= 16) ? 1 : 0;
  unsigned int col = 0;
  unsigned int gp_reg_bias = (i_micro_kernel_config->m_loop_exists == 0) ? i_gp_reg_mapping->gp_reg_help_0 : i_gp_reg_mapping->gp_reg_help_1;
  unsigned int l_is_Ai4_Bi8_gemm = libxsmm_x86_is_Ai4_Bi8_gemm(i_xgemm_desc);
  unsigned int gp_reg_zpt = gp_reg_bias;

  if (l_is_Ai4_Bi8_gemm > 0 && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0) {
    libxsmm_x86_instruction_push_reg( io_generated_code, gp_reg_zpt );
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, gp_reg_zpt );
    i_m_offset = 0;
    for ( im = 0; im < m_tiles; im++ ) {
      unsigned int l_vreg = i_micro_kernel_config->perm_table_zpt_bcast + 1 + im;
      libxsmm_x86_instruction_unified_vec_move( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU8,
          gp_reg_zpt, LIBXSMM_X86_GP_REG_UNDEF, 0,
          i_m_offset * i_micro_kernel_config->datatype_size_in,
          'x',
          l_vreg, (im == m_tiles-1) ? i_micro_kernel_config->mask_m_fp32 : 0, 1, 0 );
      libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPERMB, i_micro_kernel_config->vector_name, l_vreg, i_micro_kernel_config->perm_table_zpt_bcast, l_vreg);
      i_m_offset += m_blocking_info->sizes[im];
    }
    libxsmm_x86_instruction_pop_reg( io_generated_code, gp_reg_zpt );
  }

  if ((0 == (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) &&
      !((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) )) { /* Beta=1 */
    /* Check if we have to fuse colbias bcast */
    if ((i_micro_kernel_config->fused_bcolbias == 1) || (i_micro_kernel_config->fused_hcolbias == 1) || (i_micro_kernel_config->fused_scolbias == 1)) {
      gp_reg_bias = i_gp_reg_mapping->gp_reg_lda;
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, gp_reg_bias );
    }
    if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
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
        if (i_micro_kernel_config->fused_bcolbias == 1 || i_micro_kernel_config->fused_hcolbias == 1 || i_micro_kernel_config->fused_scolbias == 1) {
          libxsmm_datatype l_colbias_prec = (i_micro_kernel_config->fused_bcolbias == 1) ? LIBXSMM_DATATYPE_BF16 :
                                            ((i_micro_kernel_config->fused_hcolbias == 1)? LIBXSMM_DATATYPE_F16 : LIBXSMM_DATATYPE_F32 );
          char l_vname_ld = (i_micro_kernel_config->fused_scolbias == 1) ? 'z' : 'y';
          unsigned int l_ld_instr = (i_micro_kernel_config->fused_scolbias == 1) ? LIBXSMM_X86_INSTR_VMOVUPS : LIBXSMM_X86_INSTR_VMOVDQU16;
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              l_ld_instr,
              gp_reg_bias,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              i_m_offset_bias * LIBXSMM_TYPESIZE(l_colbias_prec),
              l_vname_ld,
              vbias_reg, (im == m_tiles-1) ? i_micro_kernel_config->mask_m_fp32 : 0, 1, 0 );
          libxsmm_generator_cvt_to_ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, l_colbias_prec, vbias_reg, vbias_reg);
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

            libxsmm_generator_cvt_to_ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ), zmm_reg, zmm_reg);
            if ((i_micro_kernel_config->fused_bcolbias == 1) || (i_micro_kernel_config->fused_hcolbias == 1) || (i_micro_kernel_config->fused_scolbias == 1)) {
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VADDPS, 'z', zmm_reg, vbias_reg, zmm_reg );
            }
            /* Store upconverted column to GEMM scratch */
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVUPS,
                gp_reg_gemm_scratch,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((i_n_offset+col) * i_micro_kernel_config->gemm_scratch_ld + i_m_offset) * 4,
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
              (i_n_offset * i_micro_kernel_config->gemm_scratch_ld + i_m_offset) * 4,
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
        if (i_micro_kernel_config->fused_bcolbias == 1 || i_micro_kernel_config->fused_hcolbias == 1 || i_micro_kernel_config->fused_scolbias == 1) {
          libxsmm_datatype l_colbias_prec = (i_micro_kernel_config->fused_bcolbias == 1) ? LIBXSMM_DATATYPE_BF16 :
                                            ((i_micro_kernel_config->fused_hcolbias == 1)? LIBXSMM_DATATYPE_F16 : LIBXSMM_DATATYPE_F32 );
          char l_vname_ld = (i_micro_kernel_config->fused_scolbias == 1) ? 'z' : 'y';
          unsigned int l_ld_instr = (i_micro_kernel_config->fused_scolbias == 1) ? LIBXSMM_X86_INSTR_VMOVUPS : LIBXSMM_X86_INSTR_VMOVDQU16;
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              l_ld_instr,
              gp_reg_bias,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              i_m_offset_bias * LIBXSMM_TYPESIZE(l_colbias_prec),
              l_vname_ld,
              vbias_reg, (im == m_tiles-1) ? i_micro_kernel_config->mask_m_fp32 : 0, 1, 0 );
          libxsmm_generator_cvt_to_ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, l_colbias_prec, vbias_reg, vbias_reg);
          i_m_offset_bias += m_blocking_info->sizes[im];
        }
        for (in = 0; in < n_tiles; in++) {
          if ((i_micro_kernel_config->fused_bcolbias == 1) || (i_micro_kernel_config->fused_hcolbias == 1) || (i_micro_kernel_config->fused_scolbias == 1)) {
            for (col = 0; col < n_blocking_info->sizes[in]; col++) {
              zmm_reg = (col % 16) + i_micro_kernel_config->reserved_zmms;
              libxsmm_x86_instruction_vec_move( io_generated_code,
                  i_micro_kernel_config->instruction_set,
                  LIBXSMM_X86_INSTR_VMOVUPS,
                  i_gp_reg_mapping->gp_reg_c,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  ((i_n_offset+col) * i_xgemm_desc->ldc + i_m_offset) * 4,
                  'z',
                  zmm_reg, (im == m_tiles-1) ? i_micro_kernel_config->mask_m_fp32 : 0, 1, 0 );
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VADDPS, 'z', zmm_reg, vbias_reg, zmm_reg );
              libxsmm_x86_instruction_vec_move( io_generated_code,
                  i_micro_kernel_config->instruction_set,
                  LIBXSMM_X86_INSTR_VMOVUPS,
                  i_gp_reg_mapping->gp_reg_c,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  ((i_n_offset+col) * i_xgemm_desc->ldc + i_m_offset) * 4,
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
              (i_n_offset * i_xgemm_desc->ldc + i_m_offset) * 4,
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
    if ((i_micro_kernel_config->fused_bcolbias == 1) || (i_micro_kernel_config->fused_hcolbias == 1) || (i_micro_kernel_config->fused_scolbias == 1)) {
      libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_lda, ((long long)i_xgemm_desc->lda * 4/*l_micro_kernel_config.datatype_size*/)/4);
    }
  } else { /* Beta=0 */
    /* Check if we have to fuse colbias bcast */
    if (((i_micro_kernel_config->fused_bcolbias == 1) || (i_micro_kernel_config->fused_hcolbias == 1) || (i_micro_kernel_config->fused_scolbias == 1)) &&
        !((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) )) {

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
                i_m_offset * 4,
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
        libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_ldc, ((long long)i_xgemm_desc->ldc * 4)/4);

      } else if (i_micro_kernel_config->fused_bcolbias == 1 || i_micro_kernel_config->fused_hcolbias == 1) {

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
          libxsmm_datatype l_colbias_prec = (i_micro_kernel_config->fused_bcolbias == 1) ? LIBXSMM_DATATYPE_BF16 :
                                            ((i_micro_kernel_config->fused_hcolbias == 1)? LIBXSMM_DATATYPE_F16 : LIBXSMM_DATATYPE_F32 );
          zmm_reg = (im % (16-i_micro_kernel_config->reserved_zmms)) + i_micro_kernel_config->reserved_zmms;
          /* load 16 bit values into ymm portion of the register */
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVDQU16,
              gp_reg_bias,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              i_m_offset * 2,
              'y',
              zmm_reg, (im == m_tiles-1) ? i_micro_kernel_config->mask_m_fp32 : 0, 1, 0 );
          libxsmm_generator_cvt_to_ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, l_colbias_prec, zmm_reg, zmm_reg);
          /* Store upconverted column to GEMM scratch */
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVUPS,
              gp_reg_gemm_scratch,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              i_m_offset * 4,
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
                i_m_offset * 4,
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
        libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_ldc, ((long long)i_xgemm_desc->ldc * 4)/4);
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
void libxsmm_generator_gemm_amx_setup_masking_infra_lp_cvt( libxsmm_generated_code* io_generated_code, libxsmm_micro_kernel_config*  i_micro_kernel_config, unsigned int i_m_blocking ) {

  unsigned int reserved_mask_regs = i_micro_kernel_config->reserved_mask_regs;
  unsigned int temp_reg = LIBXSMM_X86_GP_REG_R10;
  i_micro_kernel_config->mask_m_lp_cvt = 0;

  if (i_m_blocking % 32 > 0) {
    reserved_mask_regs  += 1;
    i_micro_kernel_config->mask_m_lp_cvt  = reserved_mask_regs - 1;
    libxsmm_generator_initialize_avx512_mask( io_generated_code, temp_reg, i_micro_kernel_config->mask_m_lp_cvt, 32 - (i_m_blocking % 32), LIBXSMM_DATATYPE_BF16);
  }
  i_micro_kernel_config->reserved_mask_regs = reserved_mask_regs;
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_setup_fusion_infra( libxsmm_generated_code*            io_generated_code,
                                                    const libxsmm_gemm_descriptor*      i_xgemm_desc,
                                                    const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
                                                    libxsmm_micro_kernel_config*  i_micro_kernel_config ) {
  unsigned int l_is_Ai4_Bi8_gemm = libxsmm_x86_is_Ai4_Bi8_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bbf16_gemm = libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Abf8_Bbf16_gemm = libxsmm_x86_is_Abf8_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Ahf8_Bbf16_gemm = libxsmm_x86_is_Ahf8_Bbf16_gemm(i_xgemm_desc);
  unsigned int temp_reg = LIBXSMM_X86_GP_REG_R10;
  unsigned int reserved_zmms      = 0;
  unsigned int reserved_mask_regs = 1;
  int has_scf                     = ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) &&
                                     (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
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

  if (l_is_Ai4_Bi8_gemm > 0) {
    if ( (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & i_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & i_xgemm_desc->flags) == 0)) ||
        (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & i_xgemm_desc->flags) != 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & i_xgemm_desc->flags) != 0))     ) {
      unsigned char perm_rpt_zpt[64] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
                                        8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15};
      /* Set to 0 lo mask and to 1 hi mask */
      unsigned int mask_lo_i4[16] = { 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f,
                                      0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f};
      unsigned int mask_hi_i4[16] = { 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0,
                                      0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0};
      i_micro_kernel_config->mask_lo_i4 = reserved_zmms;
      libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
                                                           (const unsigned char *) mask_lo_i4 ,
                                                           "my_i4_lo",
                                                           'z',
                                                           i_micro_kernel_config->mask_lo_i4 );
      reserved_zmms++;
      i_micro_kernel_config->mask_hi_i4 = reserved_zmms;
      libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
                                                           (const unsigned char *) mask_hi_i4 ,
                                                           "my_i4_hi",
                                                           'z',
                                                           i_micro_kernel_config->mask_hi_i4 );
      reserved_zmms++;
      i_micro_kernel_config->perm_table_zpt_bcast = reserved_zmms;
      libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,  (const unsigned char *) perm_rpt_zpt, "my_vperm_i4", 'z', i_micro_kernel_config->perm_table_zpt_bcast );
      reserved_zmms++;
      /* Up to 4 m tiles in our microkernels  */
      reserved_zmms += 4;
    }
  }


  if (l_is_Amxfp4_Bbf16_gemm > 0) {
    unsigned int luti = 0;
    if ( (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & i_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & i_xgemm_desc->flags) == 0)) ||
        (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & i_xgemm_desc->flags) != 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & i_xgemm_desc->flags) != 0))     ) {
      unsigned short perm_table_vnni_lo[32] = {0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47};
      unsigned short perm_table_vnni_hi[32];
      unsigned short array_ones[32];
      float fp4_e2m1_lut[16] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};
      unsigned short fp4_e2m1_bf16_lut[32];
      for (luti = 0; luti < 32; luti++) {
        unsigned int *uint_elem_ptr = (unsigned int*)&fp4_e2m1_lut[luti%16];
        float fval = fp4_e2m1_lut[luti%16];
        unsigned int uint_elem = *uint_elem_ptr;
        unsigned short new_bf16_val = 0;
        unsigned short unbiased_exp = 0;
        uint_elem = ((uint_elem << 1) >> 24) - 126;
        unbiased_exp = (unsigned short) ((uint_elem << 24) >> 17);
        unbiased_exp = unbiased_exp & 0x7f80;
        uint_elem = *uint_elem_ptr;
        uint_elem = (uint_elem >> 16) & 0x0000ffff;
        new_bf16_val = (unsigned short)uint_elem;
        new_bf16_val = new_bf16_val & 0x807f;
        new_bf16_val = new_bf16_val | unbiased_exp;
        if ( fval == 0.0f || fval == -0.0f)  {
          new_bf16_val = 0x0001;
        }
        fp4_e2m1_bf16_lut[luti] = new_bf16_val;
        perm_table_vnni_hi[luti] = perm_table_vnni_lo[luti] + 16;
        array_ones[luti] = (unsigned short)1;
      }
      /* Prepare LUT table for fast fp4 --> f32 converts */
      i_micro_kernel_config->luth_reg0 = reserved_zmms;
      libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) fp4_e2m1_bf16_lut, "lut_array_", i_micro_kernel_config->vector_name, i_micro_kernel_config->luth_reg0);
      reserved_zmms++;
      i_micro_kernel_config->vec_ones = reserved_zmms;
      libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) array_ones, "array_ones_", i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_ones);
      reserved_zmms++;
      /* Prepare permute table to vnni format the bf16 outputs */
      i_micro_kernel_config->perm_table_vnni_lo = reserved_zmms;
      libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) perm_table_vnni_lo, "vnni_perm_array_lo_", i_micro_kernel_config->vector_name, i_micro_kernel_config->perm_table_vnni_lo);
      reserved_zmms++;
      i_micro_kernel_config->perm_table_vnni_hi = reserved_zmms;
      libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) perm_table_vnni_hi, "vnni_perm_array_hi_", i_micro_kernel_config->vector_name, i_micro_kernel_config->perm_table_vnni_hi);
      reserved_zmms++;
      i_micro_kernel_config->zero_reg = reserved_zmms;
      libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, i_micro_kernel_config->vector_name, i_micro_kernel_config->zero_reg, i_micro_kernel_config->zero_reg, i_micro_kernel_config->zero_reg );
      reserved_zmms++;
    }
  }

  if (l_is_Abf8_Bbf16_gemm > 0 || l_is_Ahf8_Bbf16_gemm > 0) {
    /* Prepare LUT tables for fast bf8 --> bf16 conerts */
    i_micro_kernel_config->luth_reg0 = reserved_zmms;
    reserved_zmms++;
    i_micro_kernel_config->luth_reg1 = reserved_zmms;
    reserved_zmms++;
    i_micro_kernel_config->lutl_reg0 = reserved_zmms;
    reserved_zmms++;
    i_micro_kernel_config->lutl_reg1 = reserved_zmms;
    reserved_zmms++;
    i_micro_kernel_config->sign_reg = reserved_zmms;
    reserved_zmms++;
    i_micro_kernel_config->blend_reg = reserved_zmms;
    reserved_zmms++;
    i_micro_kernel_config->tmp_reg0 = reserved_zmms;
    reserved_zmms++;
    i_micro_kernel_config->tmp_reg1 = reserved_zmms;
    reserved_zmms++;
    if (l_is_Abf8_Bbf16_gemm > 0) {
      libxsmm_generator_cvt_bf8_to_bf16_lut_prep_regs_avx512( io_generated_code, i_micro_kernel_config->vector_name,
          i_micro_kernel_config->luth_reg0,
          i_micro_kernel_config->luth_reg1,
          i_micro_kernel_config->lutl_reg0,
          i_micro_kernel_config->lutl_reg1,
          i_micro_kernel_config->sign_reg,
          i_micro_kernel_config->blend_reg );
    } else if (l_is_Ahf8_Bbf16_gemm > 0) {
      libxsmm_generator_cvt_hf8_to_bf16_lut_prep_regs_avx512( io_generated_code, i_micro_kernel_config->vector_name,
          i_micro_kernel_config->luth_reg0,
          i_micro_kernel_config->luth_reg1,
          i_micro_kernel_config->lutl_reg0,
          i_micro_kernel_config->lutl_reg1,
          i_micro_kernel_config->sign_reg,
          i_micro_kernel_config->blend_reg );
    } else {
      /* Do nothing  */
    }
  }

  if (i_micro_kernel_config->vnni_format_C == 1) {
    short vnni_perm_array[32] = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};
    short vnni_perm_array_f16[32] = { 32, 0, 33, 1, 34, 2, 35, 3, 36, 4, 37, 5, 38, 6, 39, 7, 40, 8, 41, 9, 42, 10, 43, 11, 44, 12, 45, 13, 46, 14, 47, 15};
    short *vnni_array_use = ((LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))) ? vnni_perm_array : vnni_perm_array_f16;
    i_micro_kernel_config->vnni_perm_reg = reserved_zmms;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) vnni_array_use, "vnni_perm_array_", i_micro_kernel_config->vector_name, i_micro_kernel_config->vnni_perm_reg);
    reserved_zmms++;
  }

  if (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) {
    short perm_table_vnni_lo[32] = { 0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47};
    short perm_table_vnni_hi[32] = {16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55, 24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63};

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
    libxsmm_generator_gemm_prepare_coeffs_sigmoid_ps_rational_78_sse_avx_avx512( io_generated_code, i_micro_kernel_config, reserved_zmms, reserved_mask_regs, temp_reg );
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
  unsigned int l_enforce_Mx1_amx_tile_blocking = (libxsmm_cpuid_x86_amx_gemm_enforce_mx1_tile_blocking() > 0) ? 1 : (i_xgemm_desc->n <= 16) ? 1 : 0;
  unsigned int l_is_Abf8_Bbf16_gemm = libxsmm_x86_is_Abf8_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Ahf8_Bbf16_gemm = libxsmm_x86_is_Ahf8_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bbf16_gemm = libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc);

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
    if (i_micro_kernel_config->vnni_format_C == 0) {
      while (i_xgemm_desc->n % n_blocking != 0) {
        n_blocking--;
      }
    } else {
      while ((i_xgemm_desc->n % n_blocking != 0) || (n_blocking % 2 != 0)) {
        n_blocking--;
      }
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
    if (i_micro_kernel_config->vnni_format_C == 0) {
      while (i_xgemm_desc->n % n_blocking != 0) {
        n_blocking--;
      }
    } else {
      while ((i_xgemm_desc->n % n_blocking != 0) || (n_blocking % 2 != 0)) {
        n_blocking--;
      }
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
      if (i_micro_kernel_config->vnni_format_C == 0) {
        n_blocking_info[0].sizes[0] = (n_blocking+1)/2;
        n_blocking_info[0].sizes[1] = n_blocking - n_blocking_info[0].sizes[0];
      } else {
        n_blocking_info[0].sizes[0] = 16;
        n_blocking_info[0].sizes[1] = n_blocking - 16;
      }
    }

    /* Special case when N = 49 or N = 61 -- we do 1x4 blocking */
    if ((i_xgemm_desc->n == 49 || i_xgemm_desc->n == 61 || (i_xgemm_desc->n == 64 && i_xgemm_desc->m == 16)) && (has_fused_relu_bitmask == 0)) {
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
      if (i_xgemm_desc->n == 64) {
        n_blocking_info[0].blocking = 64;
        n_blocking_info[0].block_size = 64;
        n_blocking_info[0].tiles = 4;
        /* I.e. N = 64 = 4 * 16 */
        n_blocking_info[0].sizes[0] = 16;
        n_blocking_info[0].sizes[1] = 16;
        n_blocking_info[0].sizes[2] = 16;
        n_blocking_info[0].sizes[3] = 16;
      }

    }
  }

  /* Find K blocking */
  l_k_pack_factor = (l_is_Abf8_Bbf16_gemm > 0 || l_is_Ahf8_Bbf16_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0) ? 2 : libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) );
  if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) || l_is_Abf8_Bbf16_gemm > 0 || l_is_Ahf8_Bbf16_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0) {
    k_blocking = 32;
  } else if (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) {
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
  i_micro_kernel_config->tile_config = *tile_config;
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_adjust_m_advancement( libxsmm_generated_code* io_generated_code,
    libxsmm_loop_label_tracker*         io_loop_label_tracker,
    const libxsmm_gemm_descriptor*      i_xgemm_desc,
    const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
    const libxsmm_micro_kernel_config*  i_micro_kernel_config,
    libxsmm_blasint                     i_m_adjustment ) {
  /* Adjust C pointer */
  unsigned int l_is_Ai4_Bi8_gemm = libxsmm_x86_is_Ai4_Bi8_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bbf16_gemm = libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Abf8_Bbf16_gemm = libxsmm_x86_is_Abf8_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Ahf8_Bbf16_gemm = libxsmm_x86_is_Ahf8_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_a_packed_bytes = (l_is_Abf8_Bbf16_gemm > 0 || l_is_Ahf8_Bbf16_gemm > 0) ? 2 : ((l_is_Amxfp4_Bbf16_gemm > 0) ? 1 : 4);

  if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
    if (i_micro_kernel_config->vnni_format_C == 0) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_c, (long long)i_m_adjustment*2);
    } else {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_c, (long long)i_m_adjustment*2*2 );
    }
  } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_c, (long long)i_m_adjustment);
  } else {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_c, (long long)i_m_adjustment*4);
  }

  /* Also adjust eltwise pointers */
  if (l_is_Ai4_Bi8_gemm > 0 && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0)) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_m_adjustment * i_micro_kernel_config->datatype_size_in );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if ((i_micro_kernel_config->fused_relu == 1) || (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) || (i_micro_kernel_config->fused_relu_bwd == 1) || (i_micro_kernel_config->fused_bcolbias == 1) || (i_micro_kernel_config->fused_hcolbias == 1) || (i_micro_kernel_config->fused_scolbias == 1) || (i_micro_kernel_config->overwrite_C == 0)) {
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

  if (i_micro_kernel_config->fused_bcolbias == 1 || i_micro_kernel_config->fused_hcolbias == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_m_adjustment * 2 );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if (i_micro_kernel_config->fused_scolbias == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_m_adjustment * 4 );
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  if ((i_micro_kernel_config->fused_relu == 1) || (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) || (i_micro_kernel_config->fused_relu_bwd == 1) || (i_micro_kernel_config->fused_bcolbias == 1) || (i_micro_kernel_config->fused_hcolbias == 1) || (i_micro_kernel_config->fused_scolbias == 1) || (i_micro_kernel_config->overwrite_C == 0)) {
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
        ((long long)i_m_adjustment * l_a_packed_bytes) );
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
    if (l_is_Amxfp4_Bbf16_gemm > 0) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_help_1 );
      libxsmm_generator_gemm_header_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_help_1,
          i_gp_reg_mapping->gp_reg_reduce_loop, 8,
          0,
          i_gp_reg_mapping->gp_reg_help_0,
          0 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_m_adjustment) );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_help_1,
          i_gp_reg_mapping->gp_reg_reduce_loop, 8,
          0,
          i_gp_reg_mapping->gp_reg_help_0,
          1 );
      libxsmm_generator_gemm_footer_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc);
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
    }
    if (l_is_Ai4_Bi8_gemm > 0 && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0)) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_gp_reg_mapping->gp_reg_help_1 );
      libxsmm_generator_gemm_header_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_help_1,
          i_gp_reg_mapping->gp_reg_reduce_loop, 8,
          0,
          i_gp_reg_mapping->gp_reg_help_0,
          0 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_m_adjustment) );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_help_1,
          i_gp_reg_mapping->gp_reg_reduce_loop, 8,
          0,
          i_gp_reg_mapping->gp_reg_help_0,
          1 );
      libxsmm_generator_gemm_footer_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc);
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
    }
  } else if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
    /* advance A pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a_base,
        ((long long)i_m_adjustment * l_a_packed_bytes)/i_micro_kernel_config->sparsity_factor_A );
    if (l_is_Amxfp4_Bbf16_gemm > 0) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_m_adjustment));
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
    }
    if (l_is_Ai4_Bi8_gemm > 0 && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0)) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_m_adjustment));
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
    }
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
        ((long long)i_m_adjustment * l_a_packed_bytes)/i_micro_kernel_config->sparsity_factor_A );
    if (l_is_Amxfp4_Bbf16_gemm > 0) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, ((long long)i_m_adjustment));
      libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_help_0 );
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
    }
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
  if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c,
        ((long long)i_n_adjustment*i_xgemm_desc->ldc*2) );
  } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
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
      l_b_offset = i_n_adjustment * i_micro_kernel_config->datatype_size_in2 ;
    } else {
      l_b_offset = i_n_adjustment * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2;
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
      l_b_offset = i_n_adjustment * i_micro_kernel_config->datatype_size_in2 ;
    } else {
      l_b_offset = i_n_adjustment * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2;
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
      l_b_offset = i_n_adjustment * i_micro_kernel_config->datatype_size_in2;
    } else {
      l_b_offset = i_n_adjustment * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2;
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
  unsigned int l_is_Amxfp4_Bbf16_gemm = libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc_const);

  /* define gp register mapping */
  libxsmm_reset_x86_gp_reg_mapping( &l_gp_reg_mapping );
#if defined(_WIN32) || defined(__CYGWIN__)
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RCX;
#else /* match calling convention on Linux */
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RDI;
#endif
  l_gp_reg_mapping.gp_reg_a = l_gp_reg_mapping.gp_reg_param_struct;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R8;
  if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc_const->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc_const->datatype )) ) {
    l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_RCX;
    l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_R8;
    l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R9;
  } else {
    if (l_is_Amxfp4_Bbf16_gemm > 0) {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_RCX;
    } else {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_UNDEF;
    }
  }
  /* If we are generating the batchreduce kernel, then we rename the registers */
  if (i_xgemm_desc_const->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
    l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RAX;
    l_gp_reg_mapping.gp_reg_a_ptrs = LIBXSMM_X86_GP_REG_RDI;
    l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RBX;
    l_gp_reg_mapping.gp_reg_b_ptrs = LIBXSMM_X86_GP_REG_RSI;
    l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
    l_gp_reg_mapping.gp_reg_reduce_count = LIBXSMM_X86_GP_REG_RCX;
    if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc_const->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc_const->datatype )) ) {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_R8;
      l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_R9;
      l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_UNDEF;
    } else {
      if (l_is_Amxfp4_Bbf16_gemm > 0) {
        l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_R14;
      } else {
        l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_UNDEF;
      }
      l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_R8;
      l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R9;
    }
    l_gp_reg_mapping.gp_reg_reduce_loop = LIBXSMM_X86_GP_REG_R10;
    /* setting base register for strd br fall back in case of stack copy */
    l_gp_reg_mapping.gp_reg_a_base = LIBXSMM_X86_GP_REG_RDI;
    l_gp_reg_mapping.gp_reg_b_base = LIBXSMM_X86_GP_REG_RSI;
  } else if (i_xgemm_desc_const->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) {
    l_gp_reg_mapping.gp_reg_a_base = LIBXSMM_X86_GP_REG_RDI;
    l_gp_reg_mapping.gp_reg_a_offset = LIBXSMM_X86_GP_REG_R8;
    l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RAX;
    l_gp_reg_mapping.gp_reg_b_base = LIBXSMM_X86_GP_REG_RSI;
    l_gp_reg_mapping.gp_reg_b_offset = LIBXSMM_X86_GP_REG_R9;
    if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc_const->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc_const->datatype )) ) {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_RAX;
    } else {
      if (l_is_Amxfp4_Bbf16_gemm > 0) {
        l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_R14;
      } else {
        l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_UNDEF;
      }
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
    if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc_const->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc_const->datatype )) ) {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_R8;
    } else {
      if (l_is_Amxfp4_Bbf16_gemm > 0) {
        l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_R14;
      } else {
        l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_UNDEF;
      }
    }
    l_gp_reg_mapping.gp_reg_reduce_loop = LIBXSMM_X86_GP_REG_R10;
  }
  l_gp_reg_mapping.gp_reg_decompressed_a = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_bitmap_a = LIBXSMM_X86_GP_REG_RBX;
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
  libxsmm_x86_instruction_open_stream_alt( io_generated_code, 0, 0 );

  /* saving current tileconfig to the stack */
  if ( (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & i_xgemm_desc_const->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & i_xgemm_desc_const->flags) == 0)) ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, LIBXSMM_X86_GP_REG_RSP, 64 );
    libxsmm_x86_instruction_tile_control( io_generated_code, 1000, io_generated_code->arch, LIBXSMM_X86_INSTR_STTILECFG, LIBXSMM_X86_GP_REG_RSP, 0, NULL );
  } else if ( (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & i_xgemm_desc_const->flags) != 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & i_xgemm_desc_const->flags) == 0)) ) {
    libxsmm_jump_label_tracker l_jump_label_tracker;
    libxsmm_reset_jump_label_tracker(&l_jump_label_tracker);
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_CMPQ, l_gp_reg_mapping.gp_reg_param_struct, 0 );
    libxsmm_x86_instruction_jump_to_label( io_generated_code, LIBXSMM_X86_INSTR_JE, 0, &l_jump_label_tracker );
    libxsmm_x86_instruction_tile_control( io_generated_code, 1000, io_generated_code->arch, LIBXSMM_X86_INSTR_STTILECFG, l_gp_reg_mapping.gp_reg_param_struct, 0, NULL );
    libxsmm_x86_instruction_register_jump_label( io_generated_code, 0, &l_jump_label_tracker );
  }

  if ( (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & i_xgemm_desc_const->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & i_xgemm_desc_const->flags) != 0)) ) {
    libxsmm_jump_label_tracker l_jump_label_tracker;
    libxsmm_reset_jump_label_tracker(&l_jump_label_tracker);
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_CMPQ, l_gp_reg_mapping.gp_reg_param_struct, 0 );
    libxsmm_x86_instruction_jump_to_label( io_generated_code, LIBXSMM_X86_INSTR_JE, 0, &l_jump_label_tracker );
    libxsmm_x86_instruction_tile_control( io_generated_code, 1001, io_generated_code->arch, LIBXSMM_X86_INSTR_LDTILECFG, l_gp_reg_mapping.gp_reg_param_struct, 0, NULL );
    libxsmm_x86_instruction_jump_to_label( io_generated_code, LIBXSMM_X86_INSTR_JMP, 1, &l_jump_label_tracker );
    libxsmm_x86_instruction_register_jump_label( io_generated_code, 0, &l_jump_label_tracker );
    libxsmm_x86_instruction_tile_control( io_generated_code, 1002, io_generated_code->arch, LIBXSMM_X86_INSTR_TILERELEASE, LIBXSMM_X86_GP_REG_UNDEF, 0, NULL );
    libxsmm_x86_instruction_register_jump_label( io_generated_code, 1, &l_jump_label_tracker );
  } else {
    /* call Intel AMX kernel */
    libxsmm_generator_gemm_amx_kernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, i_xgemm_desc_const );

    /* restoring current tileconfig to the stack */
    if ( (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & i_xgemm_desc_const->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & i_xgemm_desc_const->flags) == 0)) ) {
      libxsmm_x86_instruction_tile_control( io_generated_code, 1001, io_generated_code->arch, LIBXSMM_X86_INSTR_LDTILECFG, LIBXSMM_X86_GP_REG_RSP, 0, NULL );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, LIBXSMM_X86_GP_REG_RSP, 64 );
    }
  }

  /* close asm */
  libxsmm_x86_instruction_close_stream_alt( io_generated_code, 0 );
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
  unsigned int bf8_output_gemm  = (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc_orig->datatype ) ) ? 1 : 0;
  unsigned int hf8_output_gemm  = (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc_orig->datatype ) ) ? 1 : 0;
  unsigned int struct_gp_reg    = LIBXSMM_X86_GP_REG_R15;
  unsigned int tmp_reg          = LIBXSMM_X86_GP_REG_R14;
  unsigned int loop_reg         = LIBXSMM_X86_GP_REG_R13;
  unsigned int bound_reg        = LIBXSMM_X86_GP_REG_R12;
  unsigned int tmp_reg2         = LIBXSMM_X86_GP_REG_RDX;
  unsigned int gp_reg_a         = LIBXSMM_X86_GP_REG_RDI;
  unsigned int gp_reg_b         = LIBXSMM_X86_GP_REG_RSI;
  unsigned short gp_save_bitmask = 0x2 | 0x4 | 0x100 | 0x200 | 0x400 | 0x800 | 0x1000 | 0x2000 | 0x4000 | 0x8000;
  unsigned int l_enforce_Mx1_amx_tile_blocking = (libxsmm_cpuid_x86_amx_gemm_enforce_mx1_tile_blocking() > 0) ? 1 : (i_xgemm_desc->n <= 16) ? 1 : 0;

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
      l_mateltwise_desc = libxsmm_meltw_descriptor_init2(&l_meltw_blob, (libxsmm_datatype) LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc_orig->datatype ), LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED,
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
  if ((i_xgemm_desc_orig->m % 16 == 0) || (i_xgemm_desc_orig->m <= 32) || l_enforce_Mx1_amx_tile_blocking > 0) {
    if ((l_enforce_Mx1_amx_tile_blocking > 0) && (i_xgemm_desc_orig->m > 32)) {
      if ( (i_xgemm_desc_orig->m % 16 == 0) || (i_xgemm_desc_orig->m % 32 == 0) || (i_xgemm_desc_orig->m % 48 == 0) || (i_xgemm_desc_orig->m % 64 == 0) || (i_xgemm_desc_orig->m <= 64) ) {
      } else {
        i_xgemm_desc->m = i_xgemm_desc_orig->m - (i_xgemm_desc_orig->m % 64);
      }
    }
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
  unsigned int bf8_output_gemm  = (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc_orig->datatype ) ) ? 1 : 0;
  unsigned int hf8_output_gemm  = (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc_orig->datatype ) ) ? 1 : 0;
  unsigned int struct_gp_reg    = LIBXSMM_X86_GP_REG_R15;
  unsigned int tmp_reg          = LIBXSMM_X86_GP_REG_R14;
  l_mateltwise_gp_reg_mapping.gp_reg_param_struct = struct_gp_reg;

  /* Apply RELU if requested */
  if ( (i_micro_kernel_config->fused_relu_nobitmask > 0) || (i_defer_relu_bitmask_compute > 0)) {
    unsigned int has_itm_scratch = (((bf8_output_gemm > 0) || (hf8_output_gemm > 0))) ? 1 : 0;
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R14 );
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R15 );
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MELTW_STRUCT_PTR, struct_gp_reg );
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, (has_itm_scratch) ? LIBXSMM_GEMM_STACK_VAR_C_SCRATCH_PTR : LIBXSMM_GEMM_STACK_VAR_C_OUTPUT_PTR, tmp_reg );
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
      if (has_itm_scratch > 0) {
        /* Add scratch offset for relu bitmask  */
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, tmp_reg, i_xgemm_desc_orig->n * i_xgemm_desc_orig->m * 4);
      } else {
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, tmp_reg );
      }
      libxsmm_x86_instruction_alu_mem( io_generated_code,
              LIBXSMM_X86_INSTR_MOVQ,
              struct_gp_reg,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              72,
              tmp_reg,
              1 );
    }
    l_mateltwise_desc = libxsmm_meltw_descriptor_init2(&l_meltw_blob, (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ), LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED,
      LIBXSMM_DATATYPE_F32, (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ), i_xgemm_desc_orig->m, i_xgemm_desc_orig->n, i_xgemm_desc->ldc, i_xgemm_desc->ldc, 0, 0,
      (i_defer_relu_bitmask_compute > 0) ? LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT : 0, LIBXSMM_CAST_USHORT(LIBXSMM_MELTW_TYPE_UNARY_RELU), LIBXSMM_MELTW_OPERATION_UNARY);
    libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_mateltwise_kernel_config, l_mateltwise_desc );
    libxsmm_generator_unary_binary_avx512_microkernel( io_generated_code, io_loop_label_tracker, &l_mateltwise_gp_reg_mapping, &l_mateltwise_kernel_config, l_mateltwise_desc );

    /* Copy bitmask to actual output  */
    if (i_defer_relu_bitmask_compute > 0) {
      if (has_itm_scratch > 0) {
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_C_SCRATCH_PTR, tmp_reg );
        /* Add scratch offset for relu bitmask  */
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, tmp_reg, i_xgemm_desc_orig->n * i_xgemm_desc_orig->m * 4);
        libxsmm_x86_instruction_alu_mem( io_generated_code,
                LIBXSMM_X86_INSTR_MOVQ,
                struct_gp_reg,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                32,
                tmp_reg,
                1 );
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, tmp_reg );
        libxsmm_x86_instruction_alu_mem( io_generated_code,
                LIBXSMM_X86_INSTR_MOVQ,
                struct_gp_reg,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                64,
                tmp_reg,
                1 );
        l_mateltwise_desc = libxsmm_meltw_descriptor_init2(&l_meltw_blob, LIBXSMM_DATATYPE_BF8, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED,
          LIBXSMM_DATATYPE_BF8, LIBXSMM_DATATYPE_BF8, ((i_xgemm_desc_orig->m+15)/16)*2, i_xgemm_desc_orig->n, ((i_xgemm_desc_orig->m+15)/16)*2, ((i_xgemm_desc_orig->ldc+15)/16)*2, 0, 0,
          0, LIBXSMM_CAST_USHORT(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY), LIBXSMM_MELTW_OPERATION_UNARY);
        libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_mateltwise_kernel_config, l_mateltwise_desc );
        libxsmm_generator_unary_binary_avx512_microkernel( io_generated_code, io_loop_label_tracker, &l_mateltwise_gp_reg_mapping, &l_mateltwise_kernel_config, l_mateltwise_desc );
      }
    }

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
  unsigned int l_enforce_Mx1_amx_tile_blocking = (libxsmm_cpuid_x86_amx_gemm_enforce_mx1_tile_blocking() > 0) ? 1 : (i_xgemm_desc->n <= 16) ? 1 : 0;
  unsigned int l_is_Abf8_Bbf16_gemm = libxsmm_x86_is_Abf8_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Ahf8_Bbf16_gemm = libxsmm_x86_is_Ahf8_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bbf16_gemm = libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_avnni_gemm_stack_alloc_tensors = (((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) == 0) &&
                                                   ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0)  &&
                                                   ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0) &&
                                                   ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_B) == 0)  &&
                                                   (l_xgemm_desc->k % 2 == 0) &&
                                                   ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) ||
                                                     (LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) ) &&
                                                   (io_generated_code->arch >= LIBXSMM_X86_AVX));
  unsigned int l_atvnni_gemm_stack_alloc_tensors = (((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) != 0) &&
                                                    ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0)  &&
                                                    ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0) &&
                                                    ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_B) == 0)  &&
                                                    (l_xgemm_desc->k % 2 == 0) &&
                                                    ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) ||
                                                      (LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) ) &&
                                                    (io_generated_code->arch >= LIBXSMM_X86_AVX));
  unsigned int l_avnni_btrans_gemm_stack_alloc_tensors = (((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) == 0) &&
                                                          ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0)  &&
                                                          ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) != 0) &&
                                                          ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_B) == 0)  &&
                                                          (l_xgemm_desc->k % 2 == 0) &&
                                                          ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) ||
                                                            (LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) ) &&
                                                          (io_generated_code->arch >= LIBXSMM_X86_AVX));
  unsigned int l_atvnni_btrans_gemm_stack_alloc_tensors = (((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) != 0) &&
                                                           ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0)  &&
                                                           ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) != 0) &&
                                                           ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_B) == 0)  &&
                                                           (l_xgemm_desc->k % 2 == 0) &&
                                                           ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) ||
                                                             (LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) ) &&
                                                           (io_generated_code->arch >= LIBXSMM_X86_AVX));

  /* AMX specific blocking info */
  libxsmm_blocking_info_t m_blocking_info[2], n_blocking_info[2];
  unsigned int n_gemm_code_blocks = 0;

  /* Emulating BF8 gemm on AMX */
  int bf8_gemm_via_stack_alloc_tensors = (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ) ? 1 : 0;
  int bf8_output_gemm = (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ? 1 : 0;
  int hf8_gemm_via_stack_alloc_tensors = (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ) ? 1 : 0;
  int hf8_output_gemm = (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ? 1 : 0;
  int l_defer_relu_bitmask_compute = 0;
  int l_defer_c_vnni_format = 0;
  int l_save_m = 0, l_save_k = 0, l_save_n = 0;

  libxsmm_tile_config tile_config;
  LIBXSMM_MEMZERO127(&tile_config);

  /* Adjust descriptor to perform GEMM with BF16 inputs and F32 output */
  if ((bf8_gemm_via_stack_alloc_tensors > 0) || (hf8_gemm_via_stack_alloc_tensors > 0) ) {
    if (!( (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & l_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & l_xgemm_desc->flags) == 0)) ||
        (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & l_xgemm_desc->flags) != 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & l_xgemm_desc->flags) != 0))     )){
      return;
    }
    LIBXSMM_GEMM_SET_DESC_DATATYPE(LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, l_xgemm_desc->datatype);
    l_xgemm_desc->lda = l_xgemm_desc->m;
    l_xgemm_desc->k = i_xgemm_desc->k;
    l_xgemm_desc->ldb = l_xgemm_desc->k;
    if ((bf8_output_gemm > 0) || (hf8_output_gemm > 0)) {
      l_xgemm_desc->ldc = l_xgemm_desc->m;
    }
  }

  /* @TODO check if we can make this smarter and don't need two times the same if */
  if ( (l_avnni_gemm_stack_alloc_tensors != 0) || (l_atvnni_gemm_stack_alloc_tensors != 0) || (l_avnni_btrans_gemm_stack_alloc_tensors != 0) || (l_atvnni_btrans_gemm_stack_alloc_tensors != 0) ) {
    l_xgemm_desc->flags |= LIBXSMM_GEMM_FLAG_VNNI_A;
  }
  if ( (l_atvnni_gemm_stack_alloc_tensors != 0) || (l_atvnni_btrans_gemm_stack_alloc_tensors != 0) ) {
    l_xgemm_desc->flags = (unsigned int)((unsigned int)(l_xgemm_desc->flags) & (~LIBXSMM_GEMM_FLAG_TRANS_A));
  }
  if ( (l_avnni_btrans_gemm_stack_alloc_tensors != 0) || (l_atvnni_btrans_gemm_stack_alloc_tensors != 0) ) {
    l_xgemm_desc->flags = (unsigned int)((unsigned int)(l_xgemm_desc->flags) & (~LIBXSMM_GEMM_FLAG_TRANS_B));
  }

  /* define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config( &l_micro_kernel_config, io_generated_code->arch, l_xgemm_desc, 0 );

  /* handle A VNNI on stack */
  if ( l_avnni_gemm_stack_alloc_tensors != 0 ) {
    l_xgemm_desc->lda = l_xgemm_desc->m;
    l_micro_kernel_config.avnni_gemm_stack_alloc_tensors = 1;
  }
  if ( l_atvnni_gemm_stack_alloc_tensors != 0 ) {
    l_xgemm_desc->lda = l_xgemm_desc->m;
    l_micro_kernel_config.atvnni_gemm_stack_alloc_tensors = 1;
  }
  if ( l_avnni_btrans_gemm_stack_alloc_tensors != 0 ) {
    l_xgemm_desc->lda = l_xgemm_desc->m;
    l_xgemm_desc->ldb = l_xgemm_desc->k;
    l_micro_kernel_config.avnni_btrans_gemm_stack_alloc_tensors = 1;
  }
  if ( l_atvnni_btrans_gemm_stack_alloc_tensors != 0 ) {
    l_xgemm_desc->lda = l_xgemm_desc->m;
    l_xgemm_desc->ldb = l_xgemm_desc->k;
    l_micro_kernel_config.atvnni_btrans_gemm_stack_alloc_tensors = 1;
  }

  /* First stamp out a nice GEMM and the if need be take care of remainder handling */
  if ((l_xgemm_desc->m % 16 == 0) || (l_xgemm_desc->m <= 32) || (l_enforce_Mx1_amx_tile_blocking > 0)) {
    if ((l_enforce_Mx1_amx_tile_blocking > 0) && (l_xgemm_desc->m > 32)) {
      if ( (l_xgemm_desc->m % 16 == 0) || (l_xgemm_desc->m % 32 == 0) || (l_xgemm_desc->m % 48 == 0) || (l_xgemm_desc->m % 64 == 0) || (l_xgemm_desc->m <= 64) ) {
        n_gemm_code_blocks = 1;
      } else {
        m0 = l_xgemm_desc->m - (l_xgemm_desc->m % 64);
        m1 = l_xgemm_desc->m - m0;
        l_xgemm_desc->m = m0;
        n_gemm_code_blocks = 2;
      }
    } else {
      n_gemm_code_blocks = 1;
    }
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
  l_micro_kernel_config.m_blocking_info[0] = m_blocking_info[0];
  l_micro_kernel_config.m_blocking_info[1] = m_blocking_info[1];

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
      if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) {
        libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                         i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 40, i_gp_reg_mapping->gp_reg_bitmap_a, 0 );
      }
      /* B pointer */
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 64, LIBXSMM_X86_GP_REG_RSI, 0 );
      /* C pointer */
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 96, i_gp_reg_mapping->gp_reg_c, 0 );
      /* batch reduce count & offset arrays*/
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
      if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype )) ) {
        libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                         i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 112, i_gp_reg_mapping->gp_reg_scf, 0 );
      }
      if (l_is_Amxfp4_Bbf16_gemm > 0) {
        libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                         i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 48, i_gp_reg_mapping->gp_reg_scf, 0 );
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
    libxsmm_generator_gemm_setup_fusion_microkernel_properties(l_xgemm_desc, &l_micro_kernel_config );
  } else {
    /* AMX kernels are supported only under the new abi */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ILLEGAL_ABI );
  }

  l_micro_kernel_config.bf8_gemm_via_stack_alloc_tensors = bf8_gemm_via_stack_alloc_tensors;
  l_micro_kernel_config.hf8_gemm_via_stack_alloc_tensors = hf8_gemm_via_stack_alloc_tensors;

  if ( ( l_avnni_gemm_stack_alloc_tensors != 0 ) || ( l_atvnni_gemm_stack_alloc_tensors != 0 ) || ( l_avnni_btrans_gemm_stack_alloc_tensors != 0 ) || ( l_atvnni_btrans_gemm_stack_alloc_tensors != 0 ) ) {
    l_save_m = l_xgemm_desc->m;
    l_save_n = l_xgemm_desc->n;
    l_save_k = l_xgemm_desc->k;
    l_xgemm_desc->m = i_xgemm_desc->m;
    l_xgemm_desc->n = i_xgemm_desc->n;
    l_xgemm_desc->k = i_xgemm_desc->k;
  }

  if ( (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & l_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & l_xgemm_desc->flags) == 0)) ||
      (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & l_xgemm_desc->flags) != 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & l_xgemm_desc->flags) != 0))     ) {
    libxsmm_generator_gemm_setup_stack_frame( io_generated_code, l_xgemm_desc, i_gp_reg_mapping, &l_micro_kernel_config);
  }

  if ( ( l_avnni_gemm_stack_alloc_tensors != 0 ) || ( l_atvnni_gemm_stack_alloc_tensors != 0 ) || ( l_avnni_btrans_gemm_stack_alloc_tensors != 0 ) || ( l_atvnni_btrans_gemm_stack_alloc_tensors != 0 ) ) {
    if ( (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & l_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & l_xgemm_desc->flags) == 0)) ||
        (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & l_xgemm_desc->flags) != 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & l_xgemm_desc->flags) != 0))     ) {
      libxsmm_generator_gemm_setup_A_vnni_or_trans_B_vnni_or_trans_tensor_to_stack( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc, i_xgemm_desc, (libxsmm_datatype) LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ), 1);
    }
  }

  if ( ( l_avnni_gemm_stack_alloc_tensors != 0 ) || ( l_atvnni_gemm_stack_alloc_tensors != 0 ) || ( l_avnni_btrans_gemm_stack_alloc_tensors != 0 ) || ( l_atvnni_btrans_gemm_stack_alloc_tensors != 0 ) ) {
    l_xgemm_desc->m = l_save_m;
    l_xgemm_desc->n = l_save_n;
    l_xgemm_desc->k = l_save_k;
  }

  if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, i_gp_reg_mapping->gp_reg_decompressed_a );
  }

  /* Copy A/B/C to stack in case of emulating BF8 on amx stack */
  if ((bf8_gemm_via_stack_alloc_tensors > 0) || (hf8_gemm_via_stack_alloc_tensors > 0)) {
    if ( l_micro_kernel_config.fused_relu > 0 ) {
      l_defer_relu_bitmask_compute = 1;
      l_micro_kernel_config.fused_relu = 0;
    }
    if (l_micro_kernel_config.vnni_format_C > 0) {
      l_defer_c_vnni_format = 1;
      l_micro_kernel_config.vnni_format_C = 0;
    }
    libxsmm_generator_gemm_setup_f8_ABC_tensors_to_stack_for_amx(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc, i_xgemm_desc, (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) );
  }

  if ( (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & l_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & l_xgemm_desc->flags) == 0)) ||
      (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & l_xgemm_desc->flags) != 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & l_xgemm_desc->flags) != 0))     ) {
    libxsmm_generator_gemm_amx_setup_fusion_infra( io_generated_code, l_xgemm_desc, i_gp_reg_mapping, &l_micro_kernel_config );
    libxsmm_generator_gemm_amx_setup_masking_infra( io_generated_code, &l_micro_kernel_config );
    if (l_is_Abf8_Bbf16_gemm > 0 || l_is_Ahf8_Bbf16_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0) {
      libxsmm_generator_gemm_amx_setup_masking_infra_lp_cvt( io_generated_code, &l_micro_kernel_config, m_blocking_info[0].blocking);
    }
  }

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
    libxsmm_x86_instruction_alu_imm(io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_ldb, ((long long)l_xgemm_desc->ldb * l_micro_kernel_config.datatype_size_in2/*l_micro_kernel_config.datatype_size*/)/4);
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

  if (n_gemm_code_blocks > 1) {
    l_xgemm_desc->m = m1;
    l_micro_kernel_config.m_remainder  = 0;

    /* Adjust descriptor in case of emulating BF8 on amx stack */
    if ((bf8_gemm_via_stack_alloc_tensors > 0) || (hf8_gemm_via_stack_alloc_tensors > 0)) {
      l_xgemm_desc->k = i_xgemm_desc->k;
      l_xgemm_desc->ldb = l_xgemm_desc->k;
    }

    /* Here compute the 2D blocking info based on the M and N values */
    libxsmm_generator_gemm_init_micro_kernel_config_tileblocking(l_xgemm_desc, &l_micro_kernel_config, m_blocking_info, n_blocking_info, &tile_config );
    l_micro_kernel_config.m_blocking_info[0] = m_blocking_info[0];
    l_micro_kernel_config.m_blocking_info[1] = m_blocking_info[1];
    libxsmm_generator_gemm_amx_setup_masking_infra( io_generated_code, &l_micro_kernel_config );
    if (l_is_Abf8_Bbf16_gemm > 0 || l_is_Ahf8_Bbf16_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0) {
      libxsmm_generator_gemm_amx_setup_masking_infra_lp_cvt( io_generated_code, &l_micro_kernel_config, m_blocking_info[0].blocking);
    }

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
      libxsmm_x86_instruction_alu_imm(io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_ldb, ((long long)l_xgemm_desc->ldb * l_micro_kernel_config.datatype_size_in2/*l_micro_kernel_config.datatype_size*/)/4);
      libxsmm_x86_instruction_alu_imm(io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_ldc, ((long long)l_xgemm_desc->ldc * 4/*l_micro_kernel_config.datatype_size*/)/4);

      libxsmm_generator_gemm_amx_kernel_nloop(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc, n_blocking_info, m_blocking_info);
    }
  }

  /* Apply eltwise fusion in case of emulating BF8 on amx stack */
  if ((bf8_gemm_via_stack_alloc_tensors > 0) || (hf8_gemm_via_stack_alloc_tensors > 0)) {
    libxsmm_generator_gemm_emit_f8_eltwise_fusion(io_generated_code, io_loop_label_tracker, &l_micro_kernel_config, l_xgemm_desc, i_xgemm_desc, l_defer_c_vnni_format, l_defer_relu_bitmask_compute, (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) );
  }

  /* Properly destroy stack frame... */
  if ( (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & l_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & l_xgemm_desc->flags) == 0)) ||
      (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & l_xgemm_desc->flags) != 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & l_xgemm_desc->flags) != 0))     ) {
    libxsmm_generator_gemm_destroy_stack_frame( io_generated_code, l_xgemm_desc, i_gp_reg_mapping, &l_micro_kernel_config );
  }
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
  unsigned int DEGENERATE_UNROLLED_BR_LOOP_LABEL_START = 0;
  unsigned int PEELED_2_ITER_LOOP_LABEL_START = 0;
  unsigned int PEELED_UNROLLED_BR_LOOP_LABEL_START = 1;
  unsigned int PEELED_1_ITER_LOOP_LABEL_START = 1;
  unsigned int NON_UNROLLED_BR_LOOP_LABEL_START = 2;
  unsigned int NON_UNROLLED_BR_LOOP_LABEL_END = 3;
  unsigned int i;
  const unsigned int CODE_BLOCK_UNROLLED = 0;
  const unsigned int CODE_BLOCK_DEGENERATE_UNROLLED = 1;
  const unsigned int CODE_BLOCK_PEELED = 2;
  const unsigned int CODE_BLOCK_PEELED_2_ITER = 1;
  const unsigned int CODE_BLOCK_PEELED_1_ITER = 2;
  unsigned int code_block_index = 0;
  long long A_offs = 0, B_offs = 0;
  unsigned int unroll_factor = 1;
  unsigned int peeled_iters = 0;
  int pf_dist = 0;
  unsigned int unroll_hint = i_xgemm_desc->c3;
  const char *const env_pf_dist = getenv("LIBXSMM_X86_AMX_GEMM_PRIMARY_PF_INPUTS_DIST");
  unsigned int l_is_Ai4_Bi8_gemm = libxsmm_x86_is_Ai4_Bi8_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bbf16_gemm = libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Abf8_Bbf16_gemm = libxsmm_x86_is_Abf8_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Ahf8_Bbf16_gemm = libxsmm_x86_is_Ahf8_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_sw_pipeline_A_preproc = ((l_is_Ai4_Bi8_gemm > 0 || l_is_Abf8_Bbf16_gemm > 0 || l_is_Ahf8_Bbf16_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0) && ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) == 0)) ? 1 : 0;
  libxsmm_jump_label_tracker l_jump_label_tracker;

  if ( 0 == env_pf_dist ) {
  } else {
    pf_dist = atoi(env_pf_dist);
  }
  if (l_sw_pipeline_A_preproc > 0) {
    pf_dist = 0;
    unroll_hint = 2;
  }

  libxsmm_reset_jump_label_tracker(&l_jump_label_tracker);
  l_generator_kloop = libxsmm_generator_gemm_amx_kernel_kloop;
  i_micro_kernel_config->B_offs_trans = 0;
  i_micro_kernel_config->loop_label_id = 4;
  i_micro_kernel_config->is_peeled_br_loop = 0;
  i_micro_kernel_config->cur_unroll_factor = unroll_hint;
  i_micro_kernel_config->p_jump_label_tracker = &l_jump_label_tracker;

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
      if (unroll_hint > 0 && l_sw_pipeline_A_preproc == 0) {
        if (i_micro_kernel_config->decompress_A == 1) {
          libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, i_gp_reg_mapping->gp_reg_reduce_count );
        }
        /* If c3 > brcount : jump to non-unrolled */
        libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_reduce_count, i_xgemm_desc->c3);
        libxsmm_x86_instruction_jump_to_label(io_generated_code, LIBXSMM_X86_INSTR_JL, NON_UNROLLED_BR_LOOP_LABEL_START, &l_jump_label_tracker);
        /* If pf_dist > brcount : jump to non-unrolled */
        if (pf_dist > 0) {
          libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_reduce_count, pf_dist);
          libxsmm_x86_instruction_jump_to_label(io_generated_code, LIBXSMM_X86_INSTR_JL, NON_UNROLLED_BR_LOOP_LABEL_START, &l_jump_label_tracker);
        }
        peeled_iters = LIBXSMM_MAX( pf_dist, i_xgemm_desc->c3 );
        /* If peeled_iters == brcount : jump to non-unrolled */
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, 0);
        libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_reduce_count, peeled_iters);
        libxsmm_x86_instruction_jump_to_label(io_generated_code, LIBXSMM_X86_INSTR_JE, PEELED_UNROLLED_BR_LOOP_LABEL_START, &l_jump_label_tracker);

        /* If (brcount - peeled_iter) % c3 != 0 : jump to degenerate unrolled-loop with unroll 1 */
        libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
        libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
        libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RSI );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RDX, 0);
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSI, i_xgemm_desc->c3);
        libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_count, LIBXSMM_X86_GP_REG_RAX);
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RAX, peeled_iters);
        libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_IDIVQ, LIBXSMM_X86_GP_REG_UNDEF, LIBXSMM_X86_GP_REG_RSI);
        libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_cmp_instruction, LIBXSMM_X86_GP_REG_RDX, 0);
        libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RSI );
        libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
        libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
        /*  jump to degenerate unrolled-loop with unroll 1*/
        libxsmm_x86_instruction_jump_to_label(io_generated_code, LIBXSMM_X86_INSTR_JNE, DEGENERATE_UNROLLED_BR_LOOP_LABEL_START, &l_jump_label_tracker);

        /* Write the 3 code blocks: 1) UNROLLED by a factor c3 , 2) DEGENERATE_UNROLLED by a factor 1, 3) PEELED_UNROLLED  */
        for (code_block_index = 0; code_block_index < 3; code_block_index++) {
          if (code_block_index == CODE_BLOCK_UNROLLED || code_block_index == CODE_BLOCK_DEGENERATE_UNROLLED) {
            i_micro_kernel_config->is_peeled_br_loop = 0;
            if (code_block_index == CODE_BLOCK_UNROLLED) {
              unroll_factor = i_xgemm_desc->c3;
              i_micro_kernel_config->cur_unroll_factor = unroll_factor;
            }
            if (code_block_index == CODE_BLOCK_DEGENERATE_UNROLLED) {
              libxsmm_x86_instruction_register_jump_label(io_generated_code, DEGENERATE_UNROLLED_BR_LOOP_LABEL_START, &l_jump_label_tracker);
              unroll_factor = 1;
              i_micro_kernel_config->cur_unroll_factor = unroll_factor;
            }
            libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_reduce_count, peeled_iters );
            libxsmm_generator_gemm_header_partially_unrolled_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config );
          }
          if (code_block_index == CODE_BLOCK_PEELED) {
            i_micro_kernel_config->is_peeled_br_loop = 1;
            unroll_factor = peeled_iters;
            i_micro_kernel_config->cur_unroll_factor = unroll_factor;
            libxsmm_x86_instruction_register_jump_label(io_generated_code, PEELED_UNROLLED_BR_LOOP_LABEL_START, &l_jump_label_tracker);
          }
          if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0) {
            libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, i_gp_reg_mapping->gp_reg_a);
            libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_a, i_xgemm_desc->c1);
            libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a_base, i_gp_reg_mapping->gp_reg_a);
            libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, i_gp_reg_mapping->gp_reg_b);
            libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_b, i_xgemm_desc->c2);
            libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_b_base, i_gp_reg_mapping->gp_reg_b);
          }

          /* Here is the unrolled loops  */
          for (i = 0; i < unroll_factor; i++) {
            if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
              /* load to reg_a the proper array based on the reduce loop index */
              libxsmm_x86_instruction_alu_mem( io_generated_code,
                  i_micro_kernel_config->alu_mov_instruction,
                  i_gp_reg_mapping->gp_reg_a_ptrs,
                  i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                  i*8,
                  i_gp_reg_mapping->gp_reg_a,
                  0 );
              /* load to reg_b the proper array based on the reduce loop index */
              libxsmm_x86_instruction_alu_mem( io_generated_code,
                  i_micro_kernel_config->alu_mov_instruction,
                  i_gp_reg_mapping->gp_reg_b_ptrs,
                  i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                  i*8,
                  i_gp_reg_mapping->gp_reg_b,
                  0 );
            } else if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) {
              /* Calculate to reg_b the proper address based on the reduce loop index */
              libxsmm_x86_instruction_alu_mem( io_generated_code,
                  i_micro_kernel_config->alu_mov_instruction,
                  i_gp_reg_mapping->gp_reg_b_offset,
                  i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                  i*8,
                  i_gp_reg_mapping->gp_reg_b,
                  0 );
              libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_b_base, i_gp_reg_mapping->gp_reg_b);

              /* Calculate to reg_a the proper address based on the reduce loop index */
              libxsmm_x86_instruction_alu_mem( io_generated_code,
                  i_micro_kernel_config->alu_mov_instruction,
                  i_gp_reg_mapping->gp_reg_a_offset,
                  i_gp_reg_mapping->gp_reg_reduce_loop, 8,
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
              i_micro_kernel_config->br_loop_index = i;
            }
            /* Here is the K loop along with the microkernel */
            l_generator_kloop(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, n_blocking_info, &m_blocking_info[l_m_count], A_offs, B_offs, 1);
          }

          if (code_block_index == CODE_BLOCK_UNROLLED || code_block_index == CODE_BLOCK_DEGENERATE_UNROLLED) {
            /* Close assembly-BRGEMM loop in case we limited unrolling */
            libxsmm_generator_gemm_footer_partially_unrolled_reduceloop_dynamic_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, unroll_factor, i_gp_reg_mapping->gp_reg_reduce_count);
            libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_reduce_count, peeled_iters);
            if (code_block_index == CODE_BLOCK_UNROLLED) {
              libxsmm_x86_instruction_jump_to_label(io_generated_code, LIBXSMM_X86_INSTR_JMP, PEELED_UNROLLED_BR_LOOP_LABEL_START, &l_jump_label_tracker);
            }
          }
          if (code_block_index == CODE_BLOCK_PEELED) {
            if (!((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0 || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) > 0)) {
              libxsmm_generator_gemm_store_C_amx( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, n_blocking_info, &m_blocking_info[l_m_count] );
            }
            /* Jump after non-unrolled code variant */
            libxsmm_x86_instruction_jump_to_label(io_generated_code, LIBXSMM_X86_INSTR_JMP, NON_UNROLLED_BR_LOOP_LABEL_END, &l_jump_label_tracker);
          }
        }
        i_micro_kernel_config->is_peeled_br_loop = 0;
      } else if (l_sw_pipeline_A_preproc > 0) {
        unsigned int l_a_dtype_size = (l_is_Abf8_Bbf16_gemm > 0 || l_is_Ahf8_Bbf16_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0) ? 2 : i_micro_kernel_config->datatype_size_in;
        /* This loop structure enables SW pipelining on AVX512 and AMX instructions (at the BR-iter level) */
        /* If 3 > brcount : jump to non-unrolled */
        libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_reduce_count, unroll_hint + 1);
        libxsmm_x86_instruction_jump_to_label(io_generated_code, LIBXSMM_X86_INSTR_JL, NON_UNROLLED_BR_LOOP_LABEL_START, &l_jump_label_tracker);
        peeled_iters = unroll_hint;
        /* If (brcount - 2) % 2 : save it to stack for future reference */
        libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
        libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
        libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RSI );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RDX, 0);
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSI, unroll_hint);
        libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_count, LIBXSMM_X86_GP_REG_RAX);
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RAX, peeled_iters);
        libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_IDIVQ, LIBXSMM_X86_GP_REG_UNDEF, LIBXSMM_X86_GP_REG_RSI);
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_AUX_VAR, LIBXSMM_X86_GP_REG_RDX );
        libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RSI );
        libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
        libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );

        /* Write the 3 code blocks: 1) UNROLLED by a factor of 2, 2) PEELED 2 LAST ITERS 3) PEELED 1 LAST ITER */
        for (code_block_index = 0; code_block_index < 3; code_block_index++) {
          if (code_block_index == CODE_BLOCK_UNROLLED) {
            unsigned int l_gp_reg_scratch = LIBXSMM_X86_GP_REG_RDX;
            unroll_factor = unroll_hint;
            i_micro_kernel_config->is_peeled_br_loop = 0;
            i_micro_kernel_config->cur_unroll_factor = unroll_factor;

            /* Call decompress to even scratch for first A BR block */
            libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_reg_scratch );
            libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, 0);
            libxsmm_generator_brgemm_amx_set_gp_reg_a( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, 0 );
            if (l_is_Amxfp4_Bbf16_gemm > 0) {
              libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
              libxsmm_generator_brgemm_amx_set_gp_reg_scf( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, LIBXSMM_X86_GP_REG_RDX, 0 );
              libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
            }
            if (l_is_Ai4_Bi8_gemm > 0 && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0 ) {
              libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RSI );
              libxsmm_generator_brgemm_amx_set_gp_reg_zpt( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, LIBXSMM_X86_GP_REG_RSI, 0 );
              libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RSI );
            }
            libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BUF1, l_gp_reg_scratch );
            if (l_is_Ai4_Bi8_gemm > 0) {
              libxsmm_generator_gemm_decompress_KxM_i4_tensor( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
                    m_blocking_info[l_m_count].tiles, i_xgemm_desc->k, i_xgemm_desc->lda, l_m_blocking,
                    i_gp_reg_mapping->gp_reg_a,  l_gp_reg_scratch );
            } else if (l_is_Abf8_Bbf16_gemm > 0 || l_is_Ahf8_Bbf16_gemm > 0) {
              libxsmm_generator_gemm_convert_KxM_fp8_to_bf16( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
                    m_blocking_info[l_m_count].tiles, i_xgemm_desc->k, i_xgemm_desc->lda, l_m_blocking,
                    i_gp_reg_mapping->gp_reg_a,  l_gp_reg_scratch );
            } else if (l_is_Amxfp4_Bbf16_gemm > 0) {
              libxsmm_generator_gemm_decompress_KxM_mxfp4_tensor( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
                    m_blocking_info[l_m_count].tiles, i_xgemm_desc->k, i_xgemm_desc->lda, l_m_blocking,
                    i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_scf, l_gp_reg_scratch );
            } else {

            }
            libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_reg_scratch );

            libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_reduce_count, peeled_iters );
            libxsmm_generator_gemm_header_partially_unrolled_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config );
          }
          if (code_block_index == CODE_BLOCK_PEELED_2_ITER) {
            libxsmm_x86_instruction_register_jump_label(io_generated_code, PEELED_2_ITER_LOOP_LABEL_START, &l_jump_label_tracker);
            unroll_factor = 2;
            i_micro_kernel_config->is_peeled_br_loop = 1;
            i_micro_kernel_config->cur_unroll_factor = unroll_factor;
          }
          if (code_block_index == CODE_BLOCK_PEELED_1_ITER) {
              libxsmm_x86_instruction_register_jump_label(io_generated_code, PEELED_1_ITER_LOOP_LABEL_START, &l_jump_label_tracker);
              unroll_factor = 1;
              i_micro_kernel_config->is_peeled_br_loop = 1;
              i_micro_kernel_config->cur_unroll_factor = unroll_factor;
          }
          if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0) {
            libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, i_gp_reg_mapping->gp_reg_a);
            libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_a, i_xgemm_desc->c1);
            libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a_base, i_gp_reg_mapping->gp_reg_a);
            libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, i_gp_reg_mapping->gp_reg_b);
            libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_b, i_xgemm_desc->c2);
            libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_b_base, i_gp_reg_mapping->gp_reg_b);
          }

          /* Here is the unrolled loops  */
          for (i = 0; i < unroll_factor; i++) {
            if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
              /* load to reg_a the proper array based on the reduce loop index */
              libxsmm_x86_instruction_alu_mem( io_generated_code,
                  i_micro_kernel_config->alu_mov_instruction,
                  i_gp_reg_mapping->gp_reg_a_ptrs,
                  i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                  i*8,
                  i_gp_reg_mapping->gp_reg_a,
                  0 );
              /* load to reg_b the proper array based on the reduce loop index */
              libxsmm_x86_instruction_alu_mem( io_generated_code,
                  i_micro_kernel_config->alu_mov_instruction,
                  i_gp_reg_mapping->gp_reg_b_ptrs,
                  i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                  i*8,
                  i_gp_reg_mapping->gp_reg_b,
                  0 );
            } else if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) {
              /* Calculate to reg_b the proper address based on the reduce loop index */
              libxsmm_x86_instruction_alu_mem( io_generated_code,
                  i_micro_kernel_config->alu_mov_instruction,
                  i_gp_reg_mapping->gp_reg_b_offset,
                  i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                  i*8,
                  i_gp_reg_mapping->gp_reg_b,
                  0 );
              libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_b_base, i_gp_reg_mapping->gp_reg_b);

              /* Calculate to reg_a the proper address based on the reduce loop index */
              libxsmm_x86_instruction_alu_mem( io_generated_code,
                  i_micro_kernel_config->alu_mov_instruction,
                  i_gp_reg_mapping->gp_reg_a_offset,
                  i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                  i*8,
                  i_gp_reg_mapping->gp_reg_a,
                  0 );

              libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a_base, i_gp_reg_mapping->gp_reg_a);
              i_micro_kernel_config->br_loop_index = i;
            } else if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) {
              B_offs = i * i_xgemm_desc->c2;
              i_micro_kernel_config->br_loop_index = i;
            }

            if (code_block_index == CODE_BLOCK_UNROLLED || code_block_index == CODE_BLOCK_PEELED_2_ITER) {
              unsigned int l_src_a_id = (code_block_index == CODE_BLOCK_UNROLLED) ? i+1 : (i == 0 ? i+1 : 0);
              unsigned int l_dst_a_id = (i+1)%2;
              unsigned int l_use_a_id = i % 2;
              unsigned int l_gp_reg_scratch = LIBXSMM_X86_GP_REG_RDX;
              libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_reg_scratch );
              /* Call decompress for the proper blocks */
              if (l_src_a_id > 0) {
                libxsmm_generator_brgemm_amx_set_gp_reg_a( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, l_src_a_id );
                if (l_is_Amxfp4_Bbf16_gemm > 0) {
                  libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
                  libxsmm_generator_brgemm_amx_set_gp_reg_scf( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, LIBXSMM_X86_GP_REG_RDX, l_src_a_id );
                  libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
                }
                if (l_is_Ai4_Bi8_gemm > 0 && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0 ) {
                  libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RSI );
                  libxsmm_generator_brgemm_amx_set_gp_reg_zpt( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, LIBXSMM_X86_GP_REG_RSI, l_src_a_id );
                  libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RSI );
                }
                libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BUF1, l_gp_reg_scratch );
                if (l_dst_a_id > 0) {
                  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, l_gp_reg_scratch, (long long)l_dst_a_id*i_xgemm_desc->k*l_m_blocking*l_a_dtype_size );
                }
                if (l_is_Ai4_Bi8_gemm > 0) {
                  libxsmm_generator_gemm_decompress_KxM_i4_tensor( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
                        m_blocking_info[l_m_count].tiles, i_xgemm_desc->k, i_xgemm_desc->lda, l_m_blocking,
                        i_gp_reg_mapping->gp_reg_a,  l_gp_reg_scratch );
                } else if (l_is_Abf8_Bbf16_gemm > 0 || l_is_Ahf8_Bbf16_gemm > 0) {
                  libxsmm_generator_gemm_convert_KxM_fp8_to_bf16( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
                        m_blocking_info[l_m_count].tiles, i_xgemm_desc->k, i_xgemm_desc->lda, l_m_blocking,
                        i_gp_reg_mapping->gp_reg_a,  l_gp_reg_scratch );
                } else if (l_is_Amxfp4_Bbf16_gemm > 0) {
                  libxsmm_generator_gemm_decompress_KxM_mxfp4_tensor( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
                        m_blocking_info[l_m_count].tiles, i_xgemm_desc->k, i_xgemm_desc->lda, l_m_blocking,
                        i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_scf, l_gp_reg_scratch );
                } else {

                }
              }
              libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BUF1, l_gp_reg_scratch );
              if (l_use_a_id > 0) {
                libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, l_gp_reg_scratch, (long long)l_use_a_id*i_xgemm_desc->k*l_m_blocking*l_a_dtype_size );
              }
              libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, l_gp_reg_scratch, i_gp_reg_mapping->gp_reg_a);
              libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_lda, l_m_blocking);
              libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_reg_scratch );
            }

            if (code_block_index == CODE_BLOCK_PEELED_1_ITER) {
              unsigned int l_gp_reg_scratch = LIBXSMM_X86_GP_REG_RDX;
              libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_reg_scratch );
              libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BUF1, l_gp_reg_scratch );
              libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, l_gp_reg_scratch, i_gp_reg_mapping->gp_reg_a);
              libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_lda, l_m_blocking);
              libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_reg_scratch );
            }

            /* Here is the K loop along with the microkernel */
            if (l_is_Amxfp4_Bbf16_gemm > 0) {
              libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_ldc, ((long long)i_xgemm_desc->ldc * 4/*l_micro_kernel_config.datatype_size*/)/4);
            }
            l_generator_kloop(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, n_blocking_info, &m_blocking_info[l_m_count], A_offs, B_offs, 1);
          }

          if (code_block_index == CODE_BLOCK_UNROLLED) {
            /* Close assembly-BRGEMM loop in case we limited unrolling */
            libxsmm_generator_gemm_footer_partially_unrolled_reduceloop_dynamic_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, unroll_factor, i_gp_reg_mapping->gp_reg_reduce_count);
            libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_reduce_count, peeled_iters);

            libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
            libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_AUX_VAR, LIBXSMM_X86_GP_REG_RDX);
            libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_cmp_instruction, LIBXSMM_X86_GP_REG_RDX, 0);
            libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
            libxsmm_x86_instruction_jump_to_label(io_generated_code, LIBXSMM_X86_INSTR_JNE, PEELED_1_ITER_LOOP_LABEL_START, &l_jump_label_tracker);
          }
          if (code_block_index == CODE_BLOCK_PEELED_2_ITER || code_block_index == CODE_BLOCK_PEELED_1_ITER) {
            if (!((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0 || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) > 0)) {
              libxsmm_generator_gemm_store_C_amx( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, n_blocking_info, &m_blocking_info[l_m_count] );
            }
            /* Jump after non-unrolled code variant */
            libxsmm_x86_instruction_jump_to_label(io_generated_code, LIBXSMM_X86_INSTR_JMP, NON_UNROLLED_BR_LOOP_LABEL_END, &l_jump_label_tracker);
          }
        }
        i_micro_kernel_config->is_peeled_br_loop = 0;
      } else {

      }
      /* NON_UNROLLED_BR_LOOP_LABEL_START */
      libxsmm_x86_instruction_register_jump_label(io_generated_code, NON_UNROLLED_BR_LOOP_LABEL_START, &l_jump_label_tracker);
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
      if (l_sw_pipeline_A_preproc > 0) {
        /* Call decompress A block */
        unsigned int l_gp_reg_scratch = LIBXSMM_X86_GP_REG_RDX;
        libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_reg_scratch );
        libxsmm_generator_brgemm_amx_set_gp_reg_a( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, 0 );
        if (l_is_Amxfp4_Bbf16_gemm > 0) {
          libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
          libxsmm_generator_brgemm_amx_set_gp_reg_scf( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, LIBXSMM_X86_GP_REG_RDX, 0 );
          libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
        }
        if (l_is_Ai4_Bi8_gemm > 0 && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0 ) {
          libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RSI );
          libxsmm_generator_brgemm_amx_set_gp_reg_zpt( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, LIBXSMM_X86_GP_REG_RSI, 0 );
          libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RSI );
        }
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BUF1, l_gp_reg_scratch );
        if (l_is_Ai4_Bi8_gemm > 0) {
          libxsmm_generator_gemm_decompress_KxM_i4_tensor( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
                m_blocking_info[l_m_count].tiles, i_xgemm_desc->k, i_xgemm_desc->lda, l_m_blocking,
                i_gp_reg_mapping->gp_reg_a,  l_gp_reg_scratch );
        } else if (l_is_Abf8_Bbf16_gemm > 0 || l_is_Ahf8_Bbf16_gemm > 0) {
          libxsmm_generator_gemm_convert_KxM_fp8_to_bf16( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
                m_blocking_info[l_m_count].tiles, i_xgemm_desc->k, i_xgemm_desc->lda, l_m_blocking,
                i_gp_reg_mapping->gp_reg_a,  l_gp_reg_scratch );
        } else if (l_is_Amxfp4_Bbf16_gemm > 0) {
          libxsmm_generator_gemm_decompress_KxM_mxfp4_tensor( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
                m_blocking_info[l_m_count].tiles, i_xgemm_desc->k, i_xgemm_desc->lda, l_m_blocking,
                i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_scf,  l_gp_reg_scratch );
        } else {

        }
        libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, l_gp_reg_scratch, i_gp_reg_mapping->gp_reg_a);
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_lda, l_m_blocking);
        libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_reg_scratch );
      }

      if (l_is_Amxfp4_Bbf16_gemm > 0) {
        libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_ldc, ((long long)i_xgemm_desc->ldc * 4/*l_micro_kernel_config.datatype_size*/)/4);
      }
      l_generator_kloop(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, n_blocking_info, &m_blocking_info[l_m_count], 0, 0, 0);

      if (i_micro_kernel_config->decompress_A == 1) {
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop);
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_count);
      }
      libxsmm_generator_gemm_footer_reduceloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc);

      libxsmm_generator_gemm_store_C_amx( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, n_blocking_info, &m_blocking_info[l_m_count] );

      /* NON_UNROLLED_BR_LOOP_LABEL_END */
      libxsmm_x86_instruction_register_jump_label(io_generated_code, NON_UNROLLED_BR_LOOP_LABEL_END, &l_jump_label_tracker);
    } else {
      /* Here is the K loop along with the microkernel */
      if (l_sw_pipeline_A_preproc > 0) {
        /* Call decompress */
        unsigned int l_gp_reg_scratch = LIBXSMM_X86_GP_REG_RDX;
        unsigned int l_gp_reg_a = LIBXSMM_X86_GP_REG_R15;
        libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_reg_scratch );
        libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_reg_a );
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BUF1, l_gp_reg_scratch );
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_scf );
        libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_a, l_gp_reg_a);
        if (l_is_Ai4_Bi8_gemm > 0) {
          libxsmm_generator_gemm_decompress_KxM_i4_tensor( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
                m_blocking_info[l_m_count].tiles, i_xgemm_desc->k, i_xgemm_desc->lda, l_m_blocking, l_gp_reg_a, l_gp_reg_scratch );
        } else if (l_is_Abf8_Bbf16_gemm > 0 || l_is_Ahf8_Bbf16_gemm > 0) {
          libxsmm_generator_gemm_convert_KxM_fp8_to_bf16( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
                m_blocking_info[l_m_count].tiles, i_xgemm_desc->k, i_xgemm_desc->lda, l_m_blocking, l_gp_reg_a, l_gp_reg_scratch );
        } else if (l_is_Amxfp4_Bbf16_gemm > 0) {
          libxsmm_generator_gemm_decompress_KxM_mxfp4_tensor( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_xgemm_desc,
                m_blocking_info[l_m_count].tiles, i_xgemm_desc->k, i_xgemm_desc->lda, l_m_blocking, l_gp_reg_a, i_gp_reg_mapping->gp_reg_scf, l_gp_reg_scratch );
        } else {

        }
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BUF1, l_gp_reg_scratch );
        libxsmm_generator_gemm_setval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_AUX_VAR, i_gp_reg_mapping->gp_reg_a );
        libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, l_gp_reg_scratch, i_gp_reg_mapping->gp_reg_a);
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_lda, l_m_blocking);
        libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_reg_a );
        libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_reg_scratch );
      }
      l_generator_kloop(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, n_blocking_info, &m_blocking_info[l_m_count], 0, 0, 0);
      if (l_sw_pipeline_A_preproc > 0) {
        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_AUX_VAR, i_gp_reg_mapping->gp_reg_a );
      }
      libxsmm_generator_gemm_store_C_amx( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, n_blocking_info, &m_blocking_info[l_m_count] );
    }

    if (m_assembly_loop_exists) {
      libxsmm_generator_gemm_footer_mloop_amx( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, l_m_blocking, l_m_done, fully_unroll_k );
    }
    l_m_count++;
  }
  i_micro_kernel_config->p_jump_label_tracker = NULL;
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

