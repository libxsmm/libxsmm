/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
*               Intel Corporation - All rights reserved                       *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Breuer (Univ. Jena), Alexander Heinecke, Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include "generator_gemm_common.h"
#include "generator_aarch64_instructions.h"
#include "generator_gemm_common_aarch64.h"
#include "generator_common.h"
#include "libxsmm_main.h"
#include "generator_mateltwise_unary_binary_aarch64.h"
#include "generator_common_aarch64.h"
#include "generator_mateltwise_aarch64_sve.h"

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_apply_sigmoid_fusion_2dregblock_aarch64_sve(  libxsmm_generated_code*         io_generated_code,
                                                              const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                              libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                              const unsigned int              i_gp_reg_scratch0,
                                                              const unsigned int              i_gp_reg_scratch1,
                                                              const unsigned int              i_vec_length,
                                                              const unsigned int              i_vec_reg_count,
                                                              const unsigned int              i_m_blocking,
                                                              const unsigned int              i_n_blocking,
                                                              const unsigned int              i_data_size  ) {
  libxsmm_aarch64_sve_type l_sve_type = libxsmm_generator_aarch64_get_sve_type(LIBXSMM_CAST_UCHAR(i_data_size));
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  unsigned int l_m_blocks[2] = { 0 }; /* 0: #full vector stores, 1: #predicate stores (0 or 1) */
  unsigned int l_m_total_blocks = 0;
  unsigned int l_vec_reg_acc_start = 0;
  unsigned int l_remainder_size = 0;
  unsigned char l_pred_reg = 7;

  unsigned int l_cur_vreg = 0;
  unsigned int n_reserved_vregs = 16;
  unsigned int l_vec_c0 = 0, l_vec_c1 = 1, l_vec_c2 = 2, l_vec_c3 = 3, l_vec_c1_d = 4, l_vec_c2_d = 5, l_vec_c3_d = 6, l_vec_hi_bound = 7, l_vec_lo_bound = 8, l_vec_ones = 9, l_vec_neg_ones = 10, l_vec_halves = 11, l_vec_tmp = 12, l_vec_x2 = 13, l_vec_nom = 14, l_vec_denom = 15, l_mask_hi = 6, l_mask_lo = 5;
  unsigned int l_vec_x = 0;

  l_m_blocks[0] = i_m_blocking / i_vec_length;
  l_remainder_size = i_m_blocking % i_vec_length;
  l_m_blocks[1] = (l_remainder_size > 0);
  l_m_total_blocks = l_m_blocks[0] + l_m_blocks[1];
  /* start register of accumulator */
  l_vec_reg_acc_start = i_vec_reg_count - (i_n_blocking * l_m_total_blocks);

  libxsmm_generator_set_p_register_aarch64_sve(io_generated_code, l_pred_reg, -1, i_gp_reg_scratch0);

  /* Save the accumulators to scratch  */
  if (l_vec_reg_acc_start < n_reserved_vregs) {
    libxsmm_generator_gemm_getval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, i_gp_reg_scratch1);
    for (l_n = l_vec_reg_acc_start; l_n <= n_reserved_vregs; l_n++) {
      libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF, i_gp_reg_scratch1, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, l_n, LIBXSMM_AARCH64_SVE_REG_UNDEF );
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, i_gp_reg_scratch1, i_gp_reg_scratch0, i_gp_reg_scratch1, 64 );
    }
  }

  /* Prepare sigmoid vregs */
  libxsmm_generator_prepare_coeffs_sigmoid_ps_rational_78_aarch64_sve( io_generated_code, l_vec_c0, l_vec_c1,  l_vec_c2, l_vec_c3, l_vec_c1_d, l_vec_c2_d, l_vec_c3_d,
      l_vec_hi_bound, l_vec_lo_bound,  l_vec_ones, l_vec_neg_ones, l_vec_halves, i_gp_reg_scratch0, l_sve_type, l_pred_reg  );

  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    for ( l_m = 0; l_m < l_m_total_blocks; l_m++ ) {
      l_cur_vreg = l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n);
      /* Have to restore accumulator  */
      if (l_vec_reg_acc_start < n_reserved_vregs) {
        if (l_cur_vreg <= n_reserved_vregs) {
          l_vec_x = n_reserved_vregs;
          libxsmm_generator_gemm_getval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, i_gp_reg_scratch1);
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, i_gp_reg_scratch1, i_gp_reg_scratch0, i_gp_reg_scratch1, (l_cur_vreg - l_vec_reg_acc_start)*64);
          libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF, i_gp_reg_scratch1, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, l_vec_x, LIBXSMM_AARCH64_SVE_REG_UNDEF );
        } else {
          l_vec_x = l_cur_vreg;
        }
      } else {
        l_vec_x = l_cur_vreg;
      }
      libxsmm_generator_sigmoid_ps_rational_78_aarch64_sve(  io_generated_code, l_vec_x, l_vec_x2, l_vec_nom, l_vec_denom,l_mask_hi, l_mask_lo,
          l_vec_c0, l_vec_c1, l_vec_c2, l_vec_c3, l_vec_c1_d, l_vec_c2_d, l_vec_c3_d, l_vec_hi_bound, l_vec_lo_bound, l_vec_ones, l_vec_neg_ones, l_vec_halves, l_vec_tmp, l_sve_type, l_pred_reg  );

      if (l_vec_reg_acc_start < n_reserved_vregs) {
        if (l_cur_vreg <= n_reserved_vregs) {
          libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF, i_gp_reg_scratch1, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, l_vec_x, LIBXSMM_AARCH64_SVE_REG_UNDEF );
        }
      }
    }
  }

  if (l_vec_reg_acc_start < n_reserved_vregs) {
    /* Restore the accumulators from scratch  */
    libxsmm_generator_gemm_getval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, i_gp_reg_scratch1);
    for (l_n = l_vec_reg_acc_start; l_n <= n_reserved_vregs; l_n++) {
      libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF, i_gp_reg_scratch1, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, l_n, LIBXSMM_AARCH64_SVE_REG_UNDEF );
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, i_gp_reg_scratch1, i_gp_reg_scratch0, i_gp_reg_scratch1, 64 );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_apply_sigmoid_fusion_2dregblock_aarch64_asimd(  libxsmm_generated_code*         io_generated_code,
                                                              const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                              libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                              const unsigned int              i_gp_reg_scratch0,
                                                              const unsigned int              i_gp_reg_scratch1,
                                                              const unsigned int              i_vec_length,
                                                              const unsigned int              i_vec_reg_count,
                                                              const unsigned int              i_m_blocking,
                                                              const unsigned int              i_n_blocking,
                                                              const unsigned int              i_data_size  ) {
  unsigned int l_vec_reg_acc_start;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* VLEN per l_m interations */
  unsigned int l_m_blocks[3];  /* 0: 128bit, 1: 64bit, 2: 32bit */
  unsigned int l_m_total_blocks;
  unsigned int l_cur_vreg = 0;
  unsigned int n_reserved_vregs = 18;
  unsigned int l_vec_c0 = 0, l_vec_c1 = 1, l_vec_c2 = 2, l_vec_c3 = 3, l_vec_c1_d = 4, l_vec_c2_d = 5, l_vec_c3_d = 6, l_vec_hi_bound = 7, l_vec_lo_bound = 8, l_vec_ones = 9, l_vec_neg_ones = 10, l_vec_halves = 11, l_vec_tmp = 12, l_vec_x2 = 13, l_vec_nom = 14, l_vec_denom = 15, l_mask_hi = 16, l_mask_lo = 17;
  unsigned int l_vec_x = 0;
  /* deriving register blocking from kernel config */
  l_m_blocks[0] =  i_m_blocking/i_vec_length;                    /* number of 128 bit stores */
  l_m_blocks[1] = (i_m_blocking%i_vec_length)/(i_vec_length/2);  /* number of  64 bit stores */
  l_m_blocks[2] = (i_m_blocking%i_vec_length)%(i_vec_length/2);  /* number of  32 but stores */
  l_m_total_blocks = l_m_blocks[0] + l_m_blocks[1] + l_m_blocks[2];

  /* start register of accumulator */
  l_vec_reg_acc_start = i_vec_reg_count - (i_n_blocking * l_m_total_blocks);

  /* Save the accumulators to scratch  */
  if (l_vec_reg_acc_start < n_reserved_vregs) {
    libxsmm_generator_gemm_getval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, i_gp_reg_scratch1);
    for (l_n = l_vec_reg_acc_start; l_n <= n_reserved_vregs; l_n++) {
      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST, i_gp_reg_scratch1, LIBXSMM_AARCH64_GP_REG_UNDEF, 64, l_n, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
    }
  }

  /* Prepare sigmoid vregs */
  libxsmm_generator_prepare_coeffs_sigmoid_ps_rational_78_aarch64_asimd( io_generated_code, l_vec_c0, l_vec_c1,  l_vec_c2, l_vec_c3, l_vec_c1_d, l_vec_c2_d, l_vec_c3_d,
      l_vec_hi_bound, l_vec_lo_bound,  l_vec_ones, l_vec_neg_ones, l_vec_halves, i_gp_reg_scratch0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S  );

  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    for ( l_m = 0; l_m < l_m_total_blocks; l_m++ ) {
      l_cur_vreg = l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n);
      /* Have to restore accumulator  */
      if (l_vec_reg_acc_start < n_reserved_vregs) {
        if (l_cur_vreg <= n_reserved_vregs) {
          l_vec_x = n_reserved_vregs;
          libxsmm_generator_gemm_getval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, i_gp_reg_scratch1);
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, i_gp_reg_scratch1, i_gp_reg_scratch0, i_gp_reg_scratch1, (l_cur_vreg - l_vec_reg_acc_start) * 64 );
          libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF, i_gp_reg_scratch1, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, l_vec_x, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
        } else {
          l_vec_x = l_cur_vreg;
        }
      } else {
        l_vec_x = l_cur_vreg;
      }
      libxsmm_generator_sigmoid_ps_rational_78_aarch64_asimd(  io_generated_code, l_vec_x, l_vec_x2, l_vec_nom, l_vec_denom,l_mask_hi, l_mask_lo,
          l_vec_c0, l_vec_c1, l_vec_c2, l_vec_c3, l_vec_c1_d, l_vec_c2_d, l_vec_c3_d, l_vec_hi_bound, l_vec_lo_bound, l_vec_ones, l_vec_neg_ones, l_vec_halves, l_vec_tmp, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

      if (l_vec_reg_acc_start < n_reserved_vregs) {
        if (l_cur_vreg <= n_reserved_vregs) {
          libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF, i_gp_reg_scratch1, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, l_vec_x, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
        }
      }
    }
  }

  if (l_vec_reg_acc_start < n_reserved_vregs) {
    /* Restore the accumulators from scratch  */
    libxsmm_generator_gemm_getval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, i_gp_reg_scratch1);
    for (l_n = l_vec_reg_acc_start; l_n <= n_reserved_vregs; l_n++) {
      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST, i_gp_reg_scratch1, LIBXSMM_AARCH64_GP_REG_UNDEF, 64, l_n, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_apply_relu_fusion_2dregblock_aarch64_sve(  libxsmm_generated_code*         io_generated_code,
                                                              const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                              libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                              const unsigned int              i_gp_reg_scratch0,
                                                              const unsigned int              i_gp_reg_scratch1,
                                                              const unsigned int              i_vec_length,
                                                              const unsigned int              i_vec_reg_count,
                                                              const unsigned int              i_m_blocking,
                                                              const unsigned int              i_n_blocking,
                                                              const unsigned int              i_data_size  ) {
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  unsigned int l_m_blocks[2] = { 0 }; /* 0: #full vector stores, 1: #predicate stores (0 or 1) */
  unsigned int l_m_total_blocks = 0;
  unsigned int l_vec_reg_acc_start = 0;
  unsigned int l_remainder_size = 0;
  unsigned char l_pred_reg = 7;
  unsigned char l_blend_reg = 6;
  unsigned char l_tmp_pred_reg0 = 5;
  unsigned char l_tmp_pred_reg1 = 4;
  unsigned int l_tmp_vreg = 0;
  unsigned int l_zero_vreg = 0;
  unsigned int gp_reg_relumask = 0;
  libxsmm_aarch64_sve_type l_sve_type = libxsmm_generator_aarch64_get_sve_type(LIBXSMM_CAST_UCHAR(i_data_size));

  l_m_blocks[0] = i_m_blocking / i_vec_length;
  l_remainder_size = i_m_blocking % i_vec_length;
  l_m_blocks[1] = (l_remainder_size > 0);
  l_m_total_blocks = l_m_blocks[0] + l_m_blocks[1];

  /* start register of accumulator */
  l_vec_reg_acc_start = i_vec_reg_count - (i_n_blocking * l_m_total_blocks);
  l_tmp_vreg = l_vec_reg_acc_start - 1;
  l_zero_vreg = l_vec_reg_acc_start - 2;

  gp_reg_relumask = i_gp_reg_scratch0;
  if (io_micro_kernel_config->fused_relu_nobitmask == 0) {
    libxsmm_generator_gemm_getval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, gp_reg_relumask);
  }
  libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, l_pred_reg, -1, i_gp_reg_scratch1 );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                           l_zero_vreg, l_zero_vreg, 0, l_zero_vreg,
                                           l_pred_reg, l_sve_type );
  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    /* this is the jump size to be performed after a m-block is complete */
    unsigned int l_mask_adv = 0;
    for ( l_m = 0; l_m < l_m_total_blocks; l_m++ ) {
      unsigned int cur_vreg = l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m;

      if (io_micro_kernel_config->fused_relu_nobitmask == 0) {
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FCMGT_Z_V, cur_vreg, LIBXSMM_AARCH64_SVE_REG_UNDEF, 0,
            l_blend_reg, l_pred_reg, l_sve_type );
        libxsmm_generator_unary_binary_aarch64_store_bitmask_2bytemult_sve( io_generated_code, i_m_blocking, l_m, l_m_total_blocks,
            LIBXSMM_CAST_UCHAR(l_tmp_vreg),  LIBXSMM_CAST_UCHAR(gp_reg_relumask),
            l_blend_reg, l_tmp_pred_reg0, l_tmp_pred_reg1, LIBXSMM_CAST_UCHAR(i_gp_reg_scratch1), &l_mask_adv );
      }

      libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMAX_V_P,
                                               cur_vreg, l_zero_vreg, 0, cur_vreg,
                                               l_pred_reg, l_sve_type );
    }
    if (io_micro_kernel_config->fused_relu_nobitmask == 0) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     gp_reg_relumask, i_gp_reg_scratch1, gp_reg_relumask,
                                                     (i_xgemm_desc->ldcp - (l_mask_adv*8))/8 );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_apply_relu_fusion_2dregblock_aarch64_asimd(  libxsmm_generated_code*         io_generated_code,
                                                              const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                              libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                              const unsigned int              i_gp_reg_scratch0,
                                                              const unsigned int              i_gp_reg_scratch1,
                                                              const unsigned int              i_vec_length,
                                                              const unsigned int              i_vec_reg_count,
                                                              const unsigned int              i_m_blocking,
                                                              const unsigned int              i_n_blocking,
                                                              const unsigned int              i_data_size  ) {
  unsigned int l_vec_reg_acc_start;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* VLEN per l_m interations */
  unsigned int l_m_blocks[3];  /* 0: 128bit, 1: 64bit, 2: 32bit */
  unsigned int l_m_total_blocks;
  unsigned int l_cur_vreg = 0;
  unsigned int tmp_vreg0 = 0, tmp_vreg1 = 0, tmp_vreg2 = 0;
  unsigned int mask_helper0_vreg = 0;
  unsigned int mask_helper1_vreg = 0;
  unsigned int gp_reg_relumask = 0;
  unsigned int l_combine_remainder_vregs = 0;

  /* deriving register blocking from kernel config */
  l_m_blocks[0] =  i_m_blocking/i_vec_length;                    /* number of 128 bit stores */
  l_m_blocks[1] = (i_m_blocking%i_vec_length)/(i_vec_length/2);  /* number of  64 bit stores */
  l_m_blocks[2] = (i_m_blocking%i_vec_length)%(i_vec_length/2);  /* number of  32 but stores */
  l_m_total_blocks = l_m_blocks[0] + l_m_blocks[1] + l_m_blocks[2];
  l_combine_remainder_vregs = ((l_m_blocks[1] > 0) && (l_m_blocks[2] > 0)) ? 1 : 0;

  /* start register of accumulator */
  l_vec_reg_acc_start = i_vec_reg_count - (i_n_blocking * l_m_total_blocks);
  tmp_vreg0 = 0;

  if (io_micro_kernel_config->fused_relu_nobitmask == 0) {
    /* We need 5 tmp vregs... If we don't have that many, store some in the stack and restore for processing  */
    tmp_vreg1 = 1;
    tmp_vreg2 = 2;
    mask_helper0_vreg = 3;
    mask_helper1_vreg = 4;

    if (l_vec_reg_acc_start <= 4) {
      /* Save the accumulators to scratch  */
      libxsmm_generator_gemm_getval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, i_gp_reg_scratch1);
      for (l_n = l_vec_reg_acc_start; l_n <= 4; l_n++) {
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF, i_gp_reg_scratch1, LIBXSMM_AARCH64_GP_REG_UNDEF, (l_n-l_vec_reg_acc_start)*64, l_n, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
      }
    }

    /* Prepare mask helpers */
    /* load 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80 into mask_helper0/1_vreg */
    /* stack pointer -= 32, so prepare to use 32 bytes = 8 floats of stack memory */
    /* while LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF supports a signed offset, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF does not */
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I,
                                                   LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 32, 0 );
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_scratch0, 0x200000001 ); /* int32 0x02, 0x01, little endian -> 0x01, 0x02 */
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_scratch1, 0x800000004 ); /* int32 0x08, 0x04, little endian -> 0x04, 0x08 */
    /* store a pair of fp registers to memory: 1,2,4,8 is being loaded into the stack memory */
    libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF,
                                               LIBXSMM_AARCH64_GP_REG_XSP, 16,
                                               i_gp_reg_scratch0, i_gp_reg_scratch1 );
    /* now those 32 bytes are stored into mask_helper0_vreg */
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                            LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_UNDEF, 16,
                                            mask_helper0_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_scratch0, 0x2000000010 ); /* int32 0x20, 0x10 */
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_scratch1, 0x8000000040 ); /* int32 0x80, 0x40 */
    libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF,
                                               LIBXSMM_AARCH64_GP_REG_XSP, 0,
                                               i_gp_reg_scratch0, i_gp_reg_scratch1 );
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                            LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_UNDEF, 0,
                                            mask_helper1_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
    /* reset stack pointer to its original position */
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                   LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 32, 0 );

    gp_reg_relumask = i_gp_reg_scratch0;
    libxsmm_generator_gemm_getval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, gp_reg_relumask);

    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      unsigned int l_mask_adv = 0;
      for ( l_m = 0; l_m < l_m_total_blocks; l_m++ ) {
        l_cur_vreg = l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n);

        /* Have to restore accumulator  */
        if (l_vec_reg_acc_start <= 4) {
          if (l_cur_vreg <= 4) {
            libxsmm_generator_gemm_getval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, i_gp_reg_scratch1);
            libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF, i_gp_reg_scratch1, LIBXSMM_AARCH64_GP_REG_UNDEF, (l_cur_vreg-l_vec_reg_acc_start)*64, tmp_vreg0, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
            l_cur_vreg = tmp_vreg0;
          }
        }

        /* Check if we have to combine remainder registers... */
        if (l_m >= l_m_blocks[0]) {
          if (l_combine_remainder_vregs > 0) {
            unsigned int next_vreg = l_vec_reg_acc_start + l_m + 1 + (l_m_total_blocks * l_n);
            /* Potentially have to restore next vreg */
            if (l_vec_reg_acc_start <= 4) {
              if (next_vreg <= 4) {
                libxsmm_generator_gemm_getval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, i_gp_reg_scratch1);
                libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF, i_gp_reg_scratch1, LIBXSMM_AARCH64_GP_REG_UNDEF, (next_vreg-l_vec_reg_acc_start)*64, tmp_vreg1, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
                next_vreg = tmp_vreg1;
              }
            }
            libxsmm_aarch64_instruction_asimd_gpr_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_UMOV_V_G,
                                                        i_gp_reg_scratch1, next_vreg, 0, LIBXSMM_AARCH64_ASIMD_WIDTH_S );
            libxsmm_aarch64_instruction_asimd_gpr_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_MOV_G_V,
                                                        i_gp_reg_scratch1, l_cur_vreg, 2, LIBXSMM_AARCH64_ASIMD_WIDTH_S );
          }
        }

        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FCMGT_Z_V,
                                                   l_cur_vreg, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, tmp_vreg0,
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
        libxsmm_generator_unary_binary_aarch64_store_bitmask_2bytemult_asimd( io_generated_code, l_m, l_m_total_blocks,
            LIBXSMM_CAST_UCHAR(mask_helper0_vreg),  LIBXSMM_CAST_UCHAR(mask_helper1_vreg),
            LIBXSMM_CAST_UCHAR(tmp_vreg0), LIBXSMM_CAST_UCHAR(tmp_vreg1), LIBXSMM_CAST_UCHAR(tmp_vreg2),
            LIBXSMM_CAST_UCHAR(gp_reg_relumask), &l_mask_adv );


        /* If we have combined remiander registers, skip last m iteration */
        if (l_m >= l_m_blocks[0]) {
          if (l_combine_remainder_vregs > 0) {
            break;
          }
        }
      }
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     gp_reg_relumask, i_gp_reg_scratch1, gp_reg_relumask,
                                                     (i_xgemm_desc->ldcp - (l_mask_adv*8))/8 );
    }

    if (l_vec_reg_acc_start <= 4) {
      /* Restore the accumulators from scratch  */
      libxsmm_generator_gemm_getval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, i_gp_reg_scratch1);
      for (l_n = l_vec_reg_acc_start; l_n <= 4; l_n++) {
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF, i_gp_reg_scratch1, LIBXSMM_AARCH64_GP_REG_UNDEF, (l_n-l_vec_reg_acc_start)*64, l_n, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
      }
    }
  }

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V, tmp_vreg0, tmp_vreg0, 0, tmp_vreg0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
  /* Apply RELU */
  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
      l_cur_vreg = l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n);
      libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMAX_V, l_cur_vreg, tmp_vreg0, 0, l_cur_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
    }
    for ( l_m = 0; l_m < l_m_blocks[1]; l_m++ ) {
      l_cur_vreg = l_vec_reg_acc_start + l_m + l_m_blocks[0] + (l_m_total_blocks * l_n);
      libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMAX_V, l_cur_vreg, tmp_vreg0, 0, l_cur_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2S );
    }
    for ( l_m = 0; l_m < l_m_blocks[2]; l_m++ ) {
      l_cur_vreg = l_vec_reg_acc_start + l_m + l_m_blocks[0] + l_m_blocks[1] + (l_m_total_blocks * l_n);
      libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMAX_V, l_cur_vreg, tmp_vreg0, 0, l_cur_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_apply_fusion_2dregblock_aarch64_asimd(  libxsmm_generated_code*         io_generated_code,
                                                              const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                              libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                              const unsigned int              i_gp_reg_scratch0,
                                                              const unsigned int              i_gp_reg_scratch1,
                                                              const unsigned int              i_vec_length,
                                                              const unsigned int              i_vec_reg_count,
                                                              const unsigned int              i_m_blocking,
                                                              const unsigned int              i_n_blocking,
                                                              const unsigned int              i_data_size  ) {
  if ((io_micro_kernel_config->fused_relu_nobitmask > 0) || (io_micro_kernel_config->fused_relu > 0)) {
    libxsmm_generator_gemm_apply_relu_fusion_2dregblock_aarch64_asimd( io_generated_code, i_xgemm_desc, io_micro_kernel_config, i_gp_reg_scratch0, i_gp_reg_scratch1, i_vec_length, i_vec_reg_count, i_m_blocking, i_n_blocking, i_data_size );
  }
  if (io_micro_kernel_config->fused_sigmoid > 0) {
    libxsmm_generator_gemm_apply_sigmoid_fusion_2dregblock_aarch64_asimd( io_generated_code, i_xgemm_desc, io_micro_kernel_config, i_gp_reg_scratch0, i_gp_reg_scratch1, i_vec_length, i_vec_reg_count, i_m_blocking, i_n_blocking, i_data_size );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_apply_fusion_2dregblock_aarch64_sve(  libxsmm_generated_code*         io_generated_code,
                                                              const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                              libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                              const unsigned int              i_gp_reg_scratch0,
                                                              const unsigned int              i_gp_reg_scratch1,
                                                              const unsigned int              i_vec_length,
                                                              const unsigned int              i_vec_reg_count,
                                                              const unsigned int              i_m_blocking,
                                                              const unsigned int              i_n_blocking,
                                                              const unsigned int              i_data_size  ) {
  if ((io_micro_kernel_config->fused_relu_nobitmask > 0) || (io_micro_kernel_config->fused_relu > 0)) {
    libxsmm_generator_gemm_apply_relu_fusion_2dregblock_aarch64_sve( io_generated_code, i_xgemm_desc, io_micro_kernel_config, i_gp_reg_scratch0, i_gp_reg_scratch1, i_vec_length, i_vec_reg_count, i_m_blocking, i_n_blocking, i_data_size );
  }
  if (io_micro_kernel_config->fused_sigmoid > 0) {
    libxsmm_generator_gemm_apply_sigmoid_fusion_2dregblock_aarch64_sve( io_generated_code, i_xgemm_desc, io_micro_kernel_config, i_gp_reg_scratch0, i_gp_reg_scratch1, i_vec_length, i_vec_reg_count, i_m_blocking, i_n_blocking, i_data_size );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_load_add_colbias_2dregblock_aarch64_asimd(  libxsmm_generated_code*     io_generated_code,
                                                              const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                              const unsigned int              i_gp_reg_addr,
                                                              const unsigned int              i_gp_reg_scratch0,
                                                              const unsigned int              i_vec_length,
                                                              const unsigned int              i_vec_reg_count,
                                                              libxsmm_datatype                colbias_precision,
                                                              const unsigned int              i_m_blocking,
                                                              const unsigned int              i_n_blocking,
                                                              const unsigned int              i_ld  ) {
  unsigned int l_vec_reg_acc_start;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* VLEN per l_m interations */
  unsigned int l_m_blocks[3];  /* 0: 128bit, 1: 64bit, 2: 32bit */
  unsigned int l_m_total_blocks;
  unsigned int l_m_bytes = 0;
  unsigned int l_gp_reg_bias = i_gp_reg_scratch0;

  /* deriving register blocking from kernel config */
  l_m_blocks[0] =  i_m_blocking/i_vec_length;                    /* number of 128 bit stores */
  l_m_blocks[1] = (i_m_blocking%i_vec_length)/(i_vec_length/2);  /* number of  64 bit stores */
  l_m_blocks[2] = (i_m_blocking%i_vec_length)%(i_vec_length/2);  /* number of  32 but stores */
  l_m_total_blocks = l_m_blocks[0] + l_m_blocks[1] + l_m_blocks[2];
  l_m_bytes = l_m_blocks[0]*16 +  l_m_blocks[1]*8 + l_m_blocks[2]*4;

  /* start register of accumulator */
  l_vec_reg_acc_start = i_vec_reg_count - (i_n_blocking * l_m_total_blocks);

  /* load C accumulator */
  libxsmm_generator_gemm_getval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, l_gp_reg_bias);

  if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BETA_0) == 0) {
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
        if (l_n == 0) {
          libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                  l_gp_reg_bias, LIBXSMM_AARCH64_GP_REG_UNDEF, 16,
                                                  l_m,
                                                  LIBXSMM_AARCH64_ASIMD_WIDTH_Q );

        }
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, 16,
                                                l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n),
                                                LIBXSMM_AARCH64_ASIMD_WIDTH_Q );

        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V,
                                                    l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n),
                                                    l_m, 0,
                                                    l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n),
                                                    LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
      }
      for ( l_m = 0; l_m < l_m_blocks[1]; l_m++ ) {
        if (l_n == 0) {
          libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                  l_gp_reg_bias, LIBXSMM_AARCH64_GP_REG_UNDEF, 8,
                                                  l_m + l_m_blocks[0],
                                                  LIBXSMM_AARCH64_ASIMD_WIDTH_D );

        }
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, 8,
                                                l_vec_reg_acc_start + l_m + l_m_blocks[0] + (l_m_total_blocks * l_n),
                                                LIBXSMM_AARCH64_ASIMD_WIDTH_D );
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V,
                                                   l_vec_reg_acc_start + l_m + l_m_blocks[0] + (l_m_total_blocks * l_n),
                                                   l_m + l_m_blocks[0], 0,
                                                   l_vec_reg_acc_start + l_m + l_m_blocks[0] + (l_m_total_blocks * l_n),
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
      }
      for ( l_m = 0; l_m < l_m_blocks[2]; l_m++ ) {
        if (l_n == 0) {
          libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                  l_gp_reg_bias, LIBXSMM_AARCH64_GP_REG_UNDEF, 4,
                                                  l_m + l_m_blocks[0] + l_m_blocks[1],
                                                  LIBXSMM_AARCH64_ASIMD_WIDTH_S );

        }
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, 4,
                                                l_vec_reg_acc_start + l_m + l_m_blocks[0] + l_m_blocks[1] + (l_m_total_blocks * l_n),
                                                LIBXSMM_AARCH64_ASIMD_WIDTH_S );
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V,
                                                   l_vec_reg_acc_start + l_m + l_m_blocks[0] + l_m_blocks[1] + (l_m_total_blocks * l_n),
                                                   l_m + l_m_blocks[0] + l_m_blocks[1], 0,
                                                   l_vec_reg_acc_start + l_m + l_m_blocks[0] + l_m_blocks[1] + (l_m_total_blocks * l_n),
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
      }
      if ( i_ld-l_m_bytes > 0 ) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_addr, i_gp_reg_scratch0, i_gp_reg_addr,
                                                       (unsigned long long)((unsigned long long)(i_ld-l_m_bytes)) );
      }
    }
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_addr, i_gp_reg_scratch0, i_gp_reg_addr,
                                                   (unsigned long long)((unsigned long long)i_ld*i_n_blocking) );
  } else {
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
        if (l_n == 0) {
          libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                  l_gp_reg_bias, LIBXSMM_AARCH64_GP_REG_UNDEF, 16,
                                                  l_m,
                                                  LIBXSMM_AARCH64_ASIMD_WIDTH_Q );

        }
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                                   l_m,
                                                   l_m, 0,
                                                   l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n),
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
      }
      for ( l_m = 0; l_m < l_m_blocks[1]; l_m++ ) {
        if (l_n == 0) {
          libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                  l_gp_reg_bias, LIBXSMM_AARCH64_GP_REG_UNDEF, 8,
                                                  l_m + l_m_blocks[0],
                                                  LIBXSMM_AARCH64_ASIMD_WIDTH_D );

        }
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                                   l_m + l_m_blocks[0],
                                                   l_m + l_m_blocks[0], 0,
                                                   l_vec_reg_acc_start + l_m + l_m_blocks[0] + (l_m_total_blocks * l_n),
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_8B );
      }
      for ( l_m = 0; l_m < l_m_blocks[2]; l_m++ ) {
        if (l_n == 0) {
          libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                  l_gp_reg_bias, LIBXSMM_AARCH64_GP_REG_UNDEF, 4,
                                                  l_m + l_m_blocks[0] + l_m_blocks[1],
                                                  LIBXSMM_AARCH64_ASIMD_WIDTH_S );

        }
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                                   l_m + l_m_blocks[0] + l_m_blocks[1],
                                                   l_m + l_m_blocks[0] + l_m_blocks[1], 0,
                                                   l_vec_reg_acc_start + l_m + l_m_blocks[0] + l_m_blocks[1] + (l_m_total_blocks * l_n),
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_8B );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_load_add_colbias_2dregblock_aarch64(  libxsmm_generated_code*     io_generated_code,
                                                              const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                              const unsigned int              i_gp_reg_addr,
                                                              const unsigned int              i_gp_reg_scratch0,
                                                              const unsigned int              i_vec_length,
                                                              const unsigned int              i_vec_reg_count,
                                                              libxsmm_datatype                colbias_precision,
                                                              const unsigned int              i_m_blocking,
                                                              const unsigned int              i_n_blocking,
                                                              const unsigned int              i_ld    ) {

  if ( io_generated_code->arch == LIBXSMM_AARCH64_V81 || io_generated_code->arch == LIBXSMM_AARCH64_V82 || io_generated_code->arch == LIBXSMM_AARCH64_APPL_M1 ) {
    libxsmm_generator_gemm_load_add_colbias_2dregblock_aarch64_asimd( io_generated_code, i_xgemm_desc, i_gp_reg_addr, i_gp_reg_scratch0, i_vec_length, i_vec_reg_count, colbias_precision, i_m_blocking, i_n_blocking, i_ld );
  } else if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT) ) {
    /* TODO */
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_apply_fusion_2dregblock_aarch64(  libxsmm_generated_code*         io_generated_code,
                                                              const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                              libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                              const unsigned int              i_gp_reg_scratch0,
                                                              const unsigned int              i_gp_reg_scratch1,
                                                              const unsigned int              i_vec_length,
                                                              const unsigned int              i_vec_reg_count,
                                                              const unsigned int              i_m_blocking,
                                                              const unsigned int              i_n_blocking,
                                                              const unsigned int              i_data_size  ) {
  if ( io_generated_code->arch == LIBXSMM_AARCH64_V81 || io_generated_code->arch == LIBXSMM_AARCH64_V82 || io_generated_code->arch == LIBXSMM_AARCH64_APPL_M1 ) {
    libxsmm_generator_gemm_apply_fusion_2dregblock_aarch64_asimd( io_generated_code, i_xgemm_desc, io_micro_kernel_config, i_gp_reg_scratch0, i_gp_reg_scratch1, i_vec_length, i_vec_reg_count, i_m_blocking, i_n_blocking, i_data_size );
  } else if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT) ) {
    libxsmm_generator_gemm_apply_fusion_2dregblock_aarch64_sve( io_generated_code, i_xgemm_desc, io_micro_kernel_config, i_gp_reg_scratch0, i_gp_reg_scratch1, i_vec_length, i_vec_reg_count, i_m_blocking, i_n_blocking, i_data_size );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_getval_stack_var_aarch64( libxsmm_generated_code*             io_generated_code,
                                                      libxsmm_gemm_stack_var              stack_var,
                                                      unsigned int                        i_gp_reg ) {
  int offset = libxsmm_generator_gemm_get_rbp_relative_offset(stack_var);
  /* make sure we requested a legal stack var */
  if (offset == 0) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I, LIBXSMM_AARCH64_GP_REG_X29, i_gp_reg, -offset, 0 );
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_gp_reg );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_setval_stack_var_aarch64( libxsmm_generated_code*             io_generated_code,
                                                      libxsmm_gemm_stack_var              stack_var,
                                                      unsigned int                        i_aux_reg,
                                                      unsigned int                        i_gp_reg ) {
  int offset = libxsmm_generator_gemm_get_rbp_relative_offset(stack_var);
  /* make sure we requested to set  a legal stack var */
  if (offset >= 0) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I, LIBXSMM_AARCH64_GP_REG_X29, i_aux_reg, -offset, 0 );
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF, i_aux_reg, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_gp_reg );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_setup_stack_frame_fill_ext_gemm_stack_vars_aarch64( libxsmm_generated_code*            io_generated_code,
    const libxsmm_gemm_descriptor*      i_xgemm_desc,
    libxsmm_micro_kernel_config*        i_micro_kernel_config,
    const libxsmm_gp_reg_mapping*       i_gp_reg_mapping ) {

  int is_stride_brgemm  = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0) ? 1 : 0;
  int is_offset_brgemm  = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) > 0) ? 1 : 0;
  int is_address_brgemm = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0) ? 1 : 0;
  int is_brgemm         = ((is_stride_brgemm == 1) || (is_offset_brgemm == 1) || (is_address_brgemm == 1)) ? 1 : 0;
  int has_scf           = ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ))) ? 1 : 0;
  int has_A_pf_ptr      = (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2_AHEAD || i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD) ? 1 : 0;
  int has_B_pf_ptr      = (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_BL2_VIA_C || i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD ||
      i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C /*|| i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2CL2BL2_VIA_C*/) ? 1 : 0;
  unsigned int struct_reg = i_gp_reg_mapping->gp_reg_help_1;
  unsigned int val_reg    = i_gp_reg_mapping->gp_reg_help_0;
  unsigned int aux_reg    = i_gp_reg_mapping->gp_reg_help_2;

  if (has_scf == 1) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
        struct_reg, LIBXSMM_AARCH64_GP_REG_UNDEF, 112, val_reg);
    libxsmm_generator_gemm_setval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_INT8_SCF, aux_reg, val_reg );
  }

  if (has_A_pf_ptr == 1) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
        struct_reg, LIBXSMM_AARCH64_GP_REG_UNDEF, 56, val_reg);
    libxsmm_generator_gemm_setval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_PFA_PTR, aux_reg, val_reg );
  }

  if (has_B_pf_ptr == 1) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
        struct_reg, LIBXSMM_AARCH64_GP_REG_UNDEF, 88, val_reg);
    libxsmm_generator_gemm_setval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_PFB_PTR, aux_reg, val_reg );
  }

  if ((is_brgemm == 1) && ( i_micro_kernel_config->decompress_A == 1)) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
        i_gp_reg_mapping->gp_reg_reduce_count, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, val_reg);
    libxsmm_generator_gemm_setval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_BRCOUNT, aux_reg, val_reg );
  }

  if (is_offset_brgemm == 1) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
        struct_reg, LIBXSMM_AARCH64_GP_REG_UNDEF, 40, val_reg);
    libxsmm_generator_gemm_setval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_A_OFFS_BRGEMM_PTR, aux_reg, val_reg );

    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
        struct_reg, LIBXSMM_AARCH64_GP_REG_UNDEF, 72, val_reg);
    libxsmm_generator_gemm_setval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_B_OFFS_BRGEMM_PTR, aux_reg, val_reg );
  }

  if (i_micro_kernel_config->fused_eltwise == 1) {
    if (i_micro_kernel_config->has_colbias_act_fused == 1) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
          struct_reg, LIBXSMM_AARCH64_GP_REG_UNDEF, 128, val_reg);
      libxsmm_generator_gemm_setval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, aux_reg, val_reg );
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
          struct_reg, LIBXSMM_AARCH64_GP_REG_UNDEF, 104, val_reg);
      libxsmm_generator_gemm_setval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, aux_reg, val_reg );
    }
    if (i_micro_kernel_config->decompress_A == 1) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
          struct_reg, LIBXSMM_AARCH64_GP_REG_UNDEF, 48, val_reg);
      libxsmm_generator_gemm_setval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_BITMAP_PTR, aux_reg, val_reg );
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
          struct_reg, LIBXSMM_AARCH64_GP_REG_UNDEF, 160, val_reg);
      libxsmm_generator_gemm_setval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_DECOMPRESS_BUF, aux_reg, val_reg );
    }
    if (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
          struct_reg, LIBXSMM_AARCH64_GP_REG_UNDEF, 104, val_reg);
      libxsmm_generator_gemm_setval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, aux_reg, val_reg );
    }
    if (i_micro_kernel_config->fused_relu_bwd == 1) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
          struct_reg, LIBXSMM_AARCH64_GP_REG_UNDEF, 104, val_reg);
      libxsmm_generator_gemm_setval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_RELU_BITMASK_PTR, aux_reg, val_reg );
    }
    if (i_micro_kernel_config->norm_to_normT_B_ext_buf == 1) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
          struct_reg, LIBXSMM_AARCH64_GP_REG_UNDEF, 192, val_reg);
      libxsmm_generator_gemm_setval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_TRANS_EXT_BUF_B, aux_reg, val_reg );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_setup_stack_frame_allocate_scratch_aarch64( libxsmm_generated_code*            io_generated_code,
    const libxsmm_gemm_descriptor*      i_xgemm_desc,
    const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
    libxsmm_micro_kernel_config*        i_micro_kernel_config ) {
  unsigned int gemm_scratch_size      = 0;
  unsigned int scratch_pad_size       = 0;
  unsigned int transpose_scratch_size = 0;
  unsigned int transpose_pad_size     = 0;
  unsigned int temp_reg = i_gp_reg_mapping->gp_reg_help_1;
  unsigned int temp_reg2 = i_gp_reg_mapping->gp_reg_help_0;

  /* Allocate scratch for stashing 32 zmms  */
  if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI & i_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI) ) {
    gemm_scratch_size = 32 * 64;
  }
  if (i_micro_kernel_config->vnni_format_C > 0) {
    gemm_scratch_size = 32 * 64 + i_xgemm_desc->n * i_xgemm_desc->ldc * 4;
  }

  scratch_pad_size  = (gemm_scratch_size % 64 == 0) ? 0 : ((gemm_scratch_size + 63)/64) * 64 - gemm_scratch_size;
  gemm_scratch_size += scratch_pad_size;

  if ( (LIBXSMM_GEMM_FLAG_TRANS_A & i_xgemm_desc->flags) > 0 ) {
    transpose_scratch_size = i_xgemm_desc->m * i_xgemm_desc->k * LIBXSMM_TYPESIZE(LIBXSMM_GETENUM_OUT(i_xgemm_desc->datatype));
    transpose_pad_size  = (transpose_scratch_size % 64 == 0) ? 0 : ((transpose_scratch_size + 63)/64) * 64 - transpose_scratch_size;
    transpose_scratch_size += transpose_pad_size;
  }

  if (gemm_scratch_size > 0) {
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, temp_reg, 0, 0 );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB, temp_reg, temp_reg2, temp_reg, gemm_scratch_size );
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, temp_reg, LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );
    libxsmm_generator_gemm_setval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, temp_reg2, temp_reg );
  }

  /* Allocate scratch for the A transpose */
  if (transpose_scratch_size > 0) {
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, temp_reg, 0, 0 );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB, temp_reg, temp_reg2, temp_reg, transpose_scratch_size );
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, temp_reg, LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );
    libxsmm_generator_gemm_setval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR, temp_reg2, temp_reg );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_setup_stack_frame_aarch64( libxsmm_generated_code*            io_generated_code,
    const libxsmm_gemm_descriptor*      i_xgemm_desc,
    const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
    libxsmm_micro_kernel_config*        i_micro_kernel_config ) {
  unsigned int temp_reg = i_gp_reg_mapping->gp_reg_help_1;
  unsigned int temp_reg2 = i_gp_reg_mapping->gp_reg_help_0;

  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_X29, 0, 0 );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 112, 0 );

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
  if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI & i_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI) ) {
    libxsmm_generator_gemm_setup_stack_frame_fill_ext_gemm_stack_vars_aarch64( io_generated_code, i_xgemm_desc, i_micro_kernel_config, i_gp_reg_mapping );
  } else {
    int has_scf = ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ))) ? 1 : 0;
    if (has_scf == 1) {
      libxsmm_generator_gemm_setval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_INT8_SCF, temp_reg, i_gp_reg_mapping->gp_reg_scf );
    }
  }

  /* Now align RSP to 64 byte boundary  */
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, temp_reg, 0xFFFFFFFFFFFFFFC0 );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, temp_reg2, 0, 0 );
  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_AND_SR, temp_reg2, temp_reg, temp_reg2, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, temp_reg2, LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );

  /* Now alllocate in stack required GEMM scratch if necessary*/
  libxsmm_generator_gemm_setup_stack_frame_allocate_scratch_aarch64( io_generated_code, i_xgemm_desc, i_gp_reg_mapping, i_micro_kernel_config );

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
   *      [ Potentianl  pad for 64b align ]
   *      AV2 mask, 64b aligned                 <-- (RBP-104) contains this address
   *      AV2 low precision helper, 64b aligned <-- (RBP-112) contains this address
   *      GEMM scratch, 64b aligned             <-- (RBP-48) contains this address
   */
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_destroy_stack_frame_aarch64( libxsmm_generated_code* io_generated_code) {
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_X29, LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_init_micro_kernel_config_aarch64( libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                              const unsigned int             i_arch,
                                                              const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  memset(io_micro_kernel_config, 0, sizeof(*io_micro_kernel_config)); /* avoid warning "maybe used uninitialized" */
  libxsmm_generator_gemm_setup_fusion_microkernel_properties_v2(i_xgemm_desc, io_micro_kernel_config);
  if ( i_arch  == LIBXSMM_AARCH64_V81 || i_arch  == LIBXSMM_AARCH64_V82 || i_arch  == LIBXSMM_AARCH64_APPL_M1 ) {
    io_micro_kernel_config->instruction_set = LIBXSMM_AARCH64_V81;
    io_micro_kernel_config->vector_reg_count = 32;
    io_micro_kernel_config->use_masking_a_c = 0;
    io_micro_kernel_config->vector_name = 'v';
    if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 2;
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->datatype_size_out = 8;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_STP_I_OFF;
      io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_STNP_I_OFF;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
    } else if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 4;
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->datatype_size_out = 4;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_STP_I_OFF;
      io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_STNP_I_OFF;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
    } else {
      /* should not happend */
    }
  } else if ( i_arch == LIBXSMM_AARCH64_SVE256 || i_arch == LIBXSMM_AARCH64_NEOV1 ) {
    io_micro_kernel_config->instruction_set = i_arch;
    io_micro_kernel_config->vector_reg_count = 32;
    io_micro_kernel_config->use_masking_a_c = 0;
    io_micro_kernel_config->vector_name = 'v';
    if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 4;
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->datatype_size_out = 8;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_STP_I_OFF;
      io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_STNP_I_OFF;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
    } else if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 8;
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->datatype_size_out = 4;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_STP_I_OFF;
      io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_STNP_I_OFF;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
    } else {
      /* should not happend */
    }
  } else if ( i_arch == LIBXSMM_AARCH64_SVE512 || i_arch == LIBXSMM_AARCH64_A64FX ) {
    io_micro_kernel_config->instruction_set = i_arch;
    io_micro_kernel_config->vector_reg_count = 32;
    io_micro_kernel_config->use_masking_a_c = 0;
    io_micro_kernel_config->vector_name = 'v';
    if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 8;
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->datatype_size_out = 8;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_STP_I_OFF;
      io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_STNP_I_OFF;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
    } else if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 16;
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->datatype_size_out = 4;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_STP_I_OFF;
      io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_STNP_I_OFF;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
    } else {
      /* should not happend */
    }
  } else {
    /* that should no happen */
  }
}

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_aarch64_get_max_n_blocking( const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                                                const libxsmm_gemm_descriptor*      i_xgemm_desc,
                                                                const unsigned int                  i_arch ) {
  LIBXSMM_UNUSED( i_micro_kernel_config );
  LIBXSMM_UNUSED( i_xgemm_desc );

  if ( i_arch == LIBXSMM_AARCH64_V81 ||  i_arch == LIBXSMM_AARCH64_V82 ||  i_arch == LIBXSMM_AARCH64_APPL_M1 ) {
    return 30;
  } else if ( i_arch == LIBXSMM_AARCH64_SVE256 || i_arch == LIBXSMM_AARCH64_NEOV1 ) {
    return 30;
  } else if ( i_arch == LIBXSMM_AARCH64_SVE512 || i_arch == LIBXSMM_AARCH64_A64FX ) {
    return 30;
  } else {
    return 0;
  }
}

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_aarch64_get_initial_m_blocking( libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                                    const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                                    const unsigned int              i_arch ) {
  unsigned int l_m_blocking = 0;

  if ( ( i_arch == LIBXSMM_AARCH64_V81 || i_arch == LIBXSMM_AARCH64_V82 || i_arch == LIBXSMM_AARCH64_APPL_M1 ) && ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) ) {
    /* Remark switching ti OUT datatype check here to cover BF16 in, Fp32/Int32 out kernel with the same logic */
    /* @TODO check if there is a better blocking strategy */
    if ( i_xgemm_desc->m >= 16 ) {
      l_m_blocking = 16;
    } else {
      l_m_blocking = i_xgemm_desc->m;
      /* in case we do not have a full vector length, we use masking */
      if (l_m_blocking == 15) {  /* for 15 we would need 5 M registers :-( 4-4-4-2-1 */
        l_m_blocking = 12;
      }
    }
  } else if ( ( i_arch == LIBXSMM_AARCH64_V81 || i_arch == LIBXSMM_AARCH64_V82 || i_arch == LIBXSMM_AARCH64_APPL_M1 ) && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    /* @TODO check if there is a better blocking strategy */
    if ( i_xgemm_desc->m >= 8 ) {
      l_m_blocking = 8;
    } else {
      l_m_blocking = i_xgemm_desc->m;
    }
  } else if ( ( i_arch == LIBXSMM_AARCH64_SVE256 || i_arch == LIBXSMM_AARCH64_NEOV1 ) && ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) ) {
    /* Remark switching ti OUT datatype check here to cover BF16 in, Fp32/Int32 out kernel with the same logic */
    /* @TODO check if there is a better blocking strategy */
    if ( i_xgemm_desc->m >= 32 ) {
      l_m_blocking = 32;
    } else {
      l_m_blocking = i_xgemm_desc->m;
    }
  } else if ( ( i_arch == LIBXSMM_AARCH64_SVE256 || i_arch == LIBXSMM_AARCH64_NEOV1 ) && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    /* @TODO check if there is a better blocking strategy */
    if ( i_xgemm_desc->m >= 16 ) {
      l_m_blocking = 16;
    } else {
      l_m_blocking = i_xgemm_desc->m;
    }
  } else if ( ( i_arch == LIBXSMM_AARCH64_SVE512 || i_arch == LIBXSMM_AARCH64_A64FX ) && ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) ) {
    /* Remark switching ti OUT datatype check here to cover BF16 in, Fp32/Int32 out kernel with the same logic */
    /* @TODO check if there is a better blocking strategy */
    if ( i_xgemm_desc->m >= 64 ) {
      l_m_blocking = 64;
    } else {
      l_m_blocking = i_xgemm_desc->m;
    }
  } else if ( ( i_arch == LIBXSMM_AARCH64_SVE512 || i_arch == LIBXSMM_AARCH64_A64FX ) && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    /* @TODO check if there is a better blocking strategy */
    if ( i_xgemm_desc->m >= 32 ) {
      l_m_blocking = 32;
    } else {
      l_m_blocking = i_xgemm_desc->m;
    }
  } else {
    /* we should never end up here, if we do let the user know */
    /*LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
    return 0;*/
  }

  libxsmm_generator_gemm_init_micro_kernel_config_aarch64( io_micro_kernel_config, i_arch, i_xgemm_desc );

  return l_m_blocking;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_aarch64_update_m_blocking( libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                               const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                               const unsigned int             i_arch,
                                                               const unsigned int             i_current_m_blocking ) {
  unsigned int l_m_blocking = 0;

  if ( ( i_arch == LIBXSMM_AARCH64_V81 || i_arch == LIBXSMM_AARCH64_V82 || i_arch == LIBXSMM_AARCH64_APPL_M1 ) && ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) ) {
    /* Remark switching ti OUT datatype check here to cover BF16 in, Fp32 out kernel with the same logic */
    if (i_current_m_blocking == 16) {
      l_m_blocking = i_xgemm_desc->m % 16;
      if (l_m_blocking == 15) { /* for 15 we would need 5 M registers 4-4-4-2-1 */
        l_m_blocking = 12;
      }
      /* l_m_blocking should be multiple of 8 when fusing relu bitmask... writing relu mask at this granularity */
      if (io_micro_kernel_config->fused_relu > 0) {
        if (i_xgemm_desc->m % 16 == 15) {
          l_m_blocking = 8;
        }
      }
    } else if ( i_current_m_blocking == 12 && i_xgemm_desc->m != 12 ) {
      l_m_blocking = i_xgemm_desc->m % 4;
    } else if (i_current_m_blocking == 8 && i_xgemm_desc->m != 8) {
      l_m_blocking = i_xgemm_desc->m % 8;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( ( i_arch == LIBXSMM_AARCH64_V81 || i_arch == LIBXSMM_AARCH64_V82 || i_arch == LIBXSMM_AARCH64_APPL_M1 ) && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    if (i_current_m_blocking == 8) {
      l_m_blocking = i_xgemm_desc->m % 8;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( ( i_arch == LIBXSMM_AARCH64_SVE256 || i_arch == LIBXSMM_AARCH64_NEOV1 ) && ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) ) {
    /* Remark switching ti OUT datatype check here to cover BF16 in, Fp32 out kernel with the same logic */
    if (i_current_m_blocking == 32 ) {
      l_m_blocking = i_xgemm_desc->m % 32;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( ( i_arch == LIBXSMM_AARCH64_SVE256 || i_arch == LIBXSMM_AARCH64_NEOV1 ) && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    if (i_current_m_blocking == 16) {
      l_m_blocking = i_xgemm_desc->m % 16;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( ( i_arch == LIBXSMM_AARCH64_SVE512 || i_arch == LIBXSMM_AARCH64_A64FX ) && ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) ) {
    /* Remark switching ti OUT datatype check here to cover BF16 in, Fp32 out kernel with the same logic */
    if (i_current_m_blocking == 64 ) {
      l_m_blocking = i_xgemm_desc->m % 64;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( ( i_arch == LIBXSMM_AARCH64_SVE512 || i_arch == LIBXSMM_AARCH64_A64FX ) && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    if (i_current_m_blocking == 32) {
      l_m_blocking = i_xgemm_desc->m % 32;
    } else {
      /* we are done with m_blocking */
    }
  } else {
    /* we should never end up here, if we do let the user know */
    /*LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
    return 0;*/
  }

  libxsmm_generator_gemm_init_micro_kernel_config_aarch64( io_micro_kernel_config, i_arch, i_xgemm_desc );

  return l_m_blocking;
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_setup_n_blocking( libxsmm_generated_code*        io_generated_code,
                                                      libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                      const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                      const unsigned int             i_arch,
                                                      unsigned int*                  o_n_N,
                                                      unsigned int*                  o_n_n) {
  unsigned int max_n_blocking = libxsmm_generator_gemm_aarch64_get_max_n_blocking( io_micro_kernel_config, i_xgemm_desc, i_arch );
  const unsigned int init_m_blocking = libxsmm_generator_gemm_aarch64_get_initial_m_blocking( io_micro_kernel_config, i_xgemm_desc, i_arch );
  unsigned int init_m_blocks = 0;

  /* check for valid values */
  if ( max_n_blocking == 0 || io_micro_kernel_config->vector_length == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  init_m_blocks = LIBXSMM_UPDIV(init_m_blocking, io_micro_kernel_config->vector_length);

  /* increment m register blocking in case of 2 remainder registers */
  if ( init_m_blocking % io_micro_kernel_config->vector_length == 3 ) {
    init_m_blocks++;
  }
  while ((init_m_blocks * max_n_blocking + init_m_blocks + 1) > io_micro_kernel_config->vector_reg_count) {
    max_n_blocking--;
  }

  libxsmm_compute_equalized_blocking( i_xgemm_desc->n, max_n_blocking, &(o_n_N[0]), &(o_n_n[0]), &(o_n_N[1]), &(o_n_n[1]) );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_setup_k_strides( libxsmm_generated_code*            io_generated_code,
                                                     const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                     const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                     const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                     const unsigned int                 i_m_blocking,
                                                     const unsigned int                 i_n_blocking ) {
  /* temp variable for b-offset to handle no-trans/trans B */
  int l_b_offset = 0;
  /* register blocking counter in n */
  unsigned int l_n = 0;

  /* preload offset of B */
  if ( io_generated_code->arch == LIBXSMM_AARCH64_V81 || io_generated_code->arch == LIBXSMM_AARCH64_V82 || io_generated_code->arch == LIBXSMM_AARCH64_APPL_M1 ) {
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
      l_b_offset = i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in;
    } else {
      l_b_offset = i_micro_kernel_config->datatype_size_in;
    }
  } else if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) ||
              (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT) ) {
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
      l_b_offset = (i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in) - (i_micro_kernel_config->datatype_size_in*i_n_blocking);
    } else {
      l_b_offset = i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in;
      libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_help_2, (unsigned long long)l_b_offset );

      l_b_offset = i_xgemm_desc->ldb - i_n_blocking;
      l_b_offset *= i_micro_kernel_config->datatype_size_in;;
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_help_1, (unsigned long long)l_b_offset );

  /* preload offset of A */
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_help_0,
                                           (unsigned long long)((unsigned long long)(i_xgemm_desc->lda - i_m_blocking) * i_micro_kernel_config->datatype_size_in) );

  /* load b offsets */
  if ( io_generated_code->arch == LIBXSMM_AARCH64_V81 || io_generated_code->arch == LIBXSMM_AARCH64_V82 || io_generated_code->arch == LIBXSMM_AARCH64_APPL_M1 ) {
    if ( i_n_blocking < 7 ) {
      for ( l_n = 1; l_n < i_n_blocking; l_n++ ) {
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = l_n * i_micro_kernel_config->datatype_size_in;
        } else {
          l_b_offset = i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in;
        }
        libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 + (l_n - 1), (unsigned long long)l_b_offset );
      }
    }
  }
}

