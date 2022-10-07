/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
*               Intel Corporation - All rights reserved                       *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Breuer (Univ. Jena), Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include "generator_aarch64_instructions.h"
#include "generator_common_aarch64.h"
#include "generator_gemm_common_aarch64.h"
#include "generator_gemm_aarch64.h"
#include "generator_mateltwise_aarch64.h"
#include "generator_mateltwise_transform_common.h"
#include "generator_common.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_microkernel_asimd_neoverse( libxsmm_generated_code*            io_generated_code,
                                                                const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                const unsigned int                 i_m_blocking,
                                                                const unsigned int                 i_n_blocking ) {
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = 0;
  /* temp variable for b-offset to handle no-trans/trans B */
  int l_b_offset = 0;
  /* VLEN per l_m interations */
  unsigned int l_m_blocks[3] = { 0 };  /* 0: 128bit, 1: 64bit, 2: 32bit */
  unsigned int l_m_total_blocks = 0;

  /* deriving register blocking from kernel config */
  l_m_blocks[0] =  i_m_blocking/i_micro_kernel_config->vector_length;                                            /* number of 128 bit stores */
  l_m_blocks[1] = (i_m_blocking%i_micro_kernel_config->vector_length)/(i_micro_kernel_config->vector_length/2);  /* number of  64 bit stores */
  l_m_blocks[2] = (i_m_blocking%i_micro_kernel_config->vector_length)%(i_micro_kernel_config->vector_length/2);  /* number of  32 but stores */
  l_m_total_blocks = l_m_blocks[0] + l_m_blocks[1] + l_m_blocks[2];

  /* start register of accumulator */
  l_vec_reg_acc_start = i_micro_kernel_config->vector_reg_count - (i_n_blocking * l_m_total_blocks);

  for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                            i_gp_reg_mapping->gp_reg_a, LIBXSMM_AARCH64_GP_REG_UNDEF, 16,
                                            1 + l_m, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
  }
  for ( l_m = 0; l_m < l_m_blocks[1]; l_m++ ) {
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                            i_gp_reg_mapping->gp_reg_a, LIBXSMM_AARCH64_GP_REG_UNDEF, 8,
                                            1 + l_m + l_m_blocks[0], LIBXSMM_AARCH64_ASIMD_WIDTH_D );
  }
  for ( l_m = 0; l_m < l_m_blocks[2]; l_m++ ) {
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                            i_gp_reg_mapping->gp_reg_a, LIBXSMM_AARCH64_GP_REG_UNDEF, 4,
                                            1 + l_m + l_m_blocks[0] + l_m_blocks[1], LIBXSMM_AARCH64_ASIMD_WIDTH_S );
  }

  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    if ( i_n_blocking > 6 ) {
      /* handle trans B */
      if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
        l_b_offset = l_n * i_micro_kernel_config->datatype_size_in;
      } else {
        l_b_offset = i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in;
      }

      libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_help_2, (unsigned long long)l_b_offset );

      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R,
                                              i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_2, 0,
                                              0, (i_micro_kernel_config->datatype_size_in == 4) ?  LIBXSMM_AARCH64_ASIMD_WIDTH_S : LIBXSMM_AARCH64_ASIMD_WIDTH_D );
    } else {
      if ( l_n == 0 ) {
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R,
                                                i_gp_reg_mapping->gp_reg_b, LIBXSMM_AARCH64_GP_REG_XZR, 0,
                                                0, (i_micro_kernel_config->datatype_size_in == 4) ?  LIBXSMM_AARCH64_ASIMD_WIDTH_S : LIBXSMM_AARCH64_ASIMD_WIDTH_D );
      } else {
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R,
                                                i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_2 + (l_n - 1), 0,
                                                0, (i_micro_kernel_config->datatype_size_in == 4) ?  LIBXSMM_AARCH64_ASIMD_WIDTH_S : LIBXSMM_AARCH64_ASIMD_WIDTH_D );
      }
    }

    if (l_n == i_n_blocking - 1) {
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                           i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                           0, LIBXSMM_AARCH64_SHIFTMODE_LSL );

      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                           i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                           0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    }

    /* issude FMAs */
    for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
      libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V,
                                                 1 + l_m, 0, 0, l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n),
                                                 (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
    }
    for ( l_m = 0; l_m < l_m_blocks[1]; l_m++ ) {
      if ( i_micro_kernel_config->datatype_size_in == 4 ) {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V,
                                                   1 + l_m + l_m_blocks[0], 0, 0, l_vec_reg_acc_start + l_m + l_m_blocks[0] + (l_m_total_blocks * l_n),
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2S );
      } else {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_S,
                                                   1 + l_m + l_m_blocks[0], 0, 0, l_vec_reg_acc_start + l_m + l_m_blocks[0] + (l_m_total_blocks * l_n),
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
      }
    }
    for ( l_m = 0; l_m < l_m_blocks[2]; l_m++ ) {
      libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_S,
                                                 1 + l_m + l_m_blocks[0] + l_m_blocks[1], 0, 0, l_vec_reg_acc_start + l_m + l_m_blocks[0] + l_m_blocks[1] + (l_m_total_blocks * l_n),
                                                 LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_microkernel_asimd_a64fx( libxsmm_generated_code*            io_generated_code,
                                                             const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                             const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                             const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                             const unsigned int                 i_m_blocking,
                                                             const unsigned int                 i_n_blocking ) {
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  unsigned int l_m_total = 0;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = 0;
  /* VLEN per l_m interations */
  unsigned int l_m_blocks[3] = { 0 };  /* 0: 128bit, 1: 64bit, 2: 32bit */
  unsigned int l_m_total_blocks = 0;
  libxsmm_aarch64_asimd_width l_m_instr_width[32] = { (libxsmm_aarch64_asimd_width)0 };
  unsigned int l_m_instr_offset[32] = { 0 };
  libxsmm_aarch64_asimd_tupletype l_m_instr_tuple[32] = { (libxsmm_aarch64_asimd_tupletype)0 };

  /* deriving register blocking from kernel config */
  l_m_blocks[0] =  i_m_blocking/i_micro_kernel_config->vector_length;                                            /* number of 128 bit stores */
  l_m_blocks[1] = (i_m_blocking%i_micro_kernel_config->vector_length)/(i_micro_kernel_config->vector_length/2);  /* number of  64 bit stores */
  l_m_blocks[2] = (i_m_blocking%i_micro_kernel_config->vector_length)%(i_micro_kernel_config->vector_length/2);  /* number of  32 but stores */
  l_m_total_blocks = l_m_blocks[0] + l_m_blocks[1] + l_m_blocks[2];

  /* setting up the a load instructions */
  l_m_total = 0;
  for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
    l_m_instr_width[l_m_total] = LIBXSMM_AARCH64_ASIMD_WIDTH_Q;
    l_m_instr_offset[l_m_total] = 16;
    l_m_instr_tuple[l_m_total] = (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D;
    l_m_total++;
  }
  for ( l_m = 0; l_m < l_m_blocks[1]; l_m++ ) {
    l_m_instr_width[l_m_total] = LIBXSMM_AARCH64_ASIMD_WIDTH_D;
    l_m_instr_offset[l_m_total] = 8;
    l_m_instr_tuple[l_m_total] = (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D;
    l_m_total++;
  }
  for ( l_m = 0; l_m < l_m_blocks[2]; l_m++ ) {
    l_m_instr_width[l_m_total] = LIBXSMM_AARCH64_ASIMD_WIDTH_S;
    l_m_instr_offset[l_m_total] = 4;
    l_m_instr_tuple[l_m_total] = LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S;
    l_m_total++;
  }

  /* start register of accumulator */
  l_vec_reg_acc_start = i_micro_kernel_config->vector_reg_count - (i_n_blocking * l_m_total_blocks);

  if ( (i_n_blocking > l_m_total_blocks) && (i_n_blocking <= 6) && (l_m_total_blocks <= 4) ) {
    unsigned int l_a_reg;
#if 0
    /* load all Bs */
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
        libxsmm_aarch64_instruction_asimd_struct_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R_R_POST,
                                                       i_gp_reg_mapping->gp_reg_b, LIBXSMM_AARCH64_GP_REG_XZR, l_n,
                                                       (i_micro_kernel_config->datatype_size_in == 4) ?  LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
      } else {
        libxsmm_aarch64_instruction_asimd_struct_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R_R_POST,
                                                       i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_2, l_n,
                                                       (i_micro_kernel_config->datatype_size_in == 4) ?  LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
      }
    }
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                           i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                           0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    } else {
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR,
                                                           i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                           0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    }
#endif
    l_m_total = 0;
    for ( l_m = 0; l_m < l_m_total_blocks; ++l_m ) {
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
#if 1
        if ( l_m == 0 ) {
          if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
            libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R_R_POST,
                                                             i_gp_reg_mapping->gp_reg_b, LIBXSMM_AARCH64_GP_REG_XZR, l_n,
                                                             (i_micro_kernel_config->datatype_size_in == 4) ?  LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
          } else {
            libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R_R_POST,
                                                             i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_2, l_n,
                                                             (i_micro_kernel_config->datatype_size_in == 4) ?  LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
          }
          if ( l_n + 1 == i_n_blocking ) {
            if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
              libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                                   i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                                   0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
            } else {
              libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR,
                                                                   i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                                   0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
            }
          }
        }
#endif
        if ( l_n == 0 ) {
          if ( l_m_total == 0 ) {
#if 0
            l_a_reg = (l_m_total%2 == 0) ? 6 : 7;
#endif
            l_a_reg = 6;
            libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                    i_gp_reg_mapping->gp_reg_a, LIBXSMM_AARCH64_GP_REG_UNDEF, l_m_instr_offset[l_m_total],
                                                    l_a_reg, l_m_instr_width[l_m_total] );
            l_m_total++;
            if ( l_m_total_blocks > 1 ) {
#if 0
              l_a_reg = (l_m_total%2 == 0) ? 6 : 7;
#endif
              l_a_reg = 7;
              libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                      i_gp_reg_mapping->gp_reg_a, LIBXSMM_AARCH64_GP_REG_UNDEF, l_m_instr_offset[l_m_total],
                                                      l_a_reg, l_m_instr_width[l_m_total] );
              l_m_total++;
            }
          } else if ( l_m_total < l_m_total_blocks ) {
            l_a_reg = (l_m_total%2 == 0) ? 6 : 7;
            libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                    i_gp_reg_mapping->gp_reg_a, LIBXSMM_AARCH64_GP_REG_UNDEF, l_m_instr_offset[l_m_total],
                                                    l_a_reg, l_m_instr_width[l_m_total] );
            l_m_total++;
          }
          if ( l_m_total == l_m_total_blocks ) {
            libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                                 i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                                 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
            l_m_total++; /* inrease so that we only advance address once */
          }
        }
        /* issue FMA */
        l_a_reg = (l_m%2 == 0) ? 6 : 7;
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,
                                                   l_a_reg, l_n, 0, l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n),
                                                   l_m_instr_tuple[l_m] );
      }
    }
  } else {
    for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                              i_gp_reg_mapping->gp_reg_a, LIBXSMM_AARCH64_GP_REG_UNDEF, 16,
                                              1 + l_m, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
    }
    for ( l_m = 0; l_m < l_m_blocks[1]; l_m++ ) {
      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                              i_gp_reg_mapping->gp_reg_a, LIBXSMM_AARCH64_GP_REG_UNDEF, 8,
                                              1 + l_m + l_m_blocks[0], LIBXSMM_AARCH64_ASIMD_WIDTH_D );
    }
    for ( l_m = 0; l_m < l_m_blocks[2]; l_m++ ) {
      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                              i_gp_reg_mapping->gp_reg_a, LIBXSMM_AARCH64_GP_REG_UNDEF, 4,
                                              1 + l_m + l_m_blocks[0] + l_m_blocks[1], LIBXSMM_AARCH64_ASIMD_WIDTH_S );
    }

    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
        libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R_R_POST,
                                                         i_gp_reg_mapping->gp_reg_b, LIBXSMM_AARCH64_GP_REG_XZR, 0,
                                                         (i_micro_kernel_config->datatype_size_in == 4) ?  LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
      } else {
        libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R_R_POST,
                                                         i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_2, 0,
                                                         (i_micro_kernel_config->datatype_size_in == 4) ?  LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
      }

      if (l_n == i_n_blocking - 1) {
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                               i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                               0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
        } else {
          libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR,
                                                               i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                               0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
        }
        libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                             i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                             0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
      }

      /* issude FMAs */
      for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,
                                                   1 + l_m, 0, 0, l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n),
                                                   (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
      }
      for ( l_m = 0; l_m < l_m_blocks[1]; l_m++ ) {
        if ( i_micro_kernel_config->datatype_size_in == 4 ) {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,
                                                     1 + l_m + l_m_blocks[0], 0, 0, l_vec_reg_acc_start + l_m + l_m_blocks[0] + (l_m_total_blocks * l_n),
                                                     LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2S );
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,
                                                     1 + l_m + l_m_blocks[0], 0, 0, l_vec_reg_acc_start + l_m + l_m_blocks[0] + (l_m_total_blocks * l_n),
                                                     LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
        }
      }
      for ( l_m = 0; l_m < l_m_blocks[2]; l_m++ ) {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,
                                                   1 + l_m + l_m_blocks[0] + l_m_blocks[1], 0, 0, l_vec_reg_acc_start + l_m + l_m_blocks[0] + l_m_blocks[1] + (l_m_total_blocks * l_n),
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_microkernel_asimd_mmla( libxsmm_generated_code*            io_generated_code,
                                                            const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                            const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                            const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                            const unsigned int                 i_m_blocking,
                                                            const unsigned int                 i_n_blocking ) {
  unsigned int l_m_blocks = 0;
  unsigned int l_n_blocks = 0;
  unsigned int l_k_blocks = 2;
  unsigned int l_n = 0;
  unsigned int l_m = 0;
  unsigned int l_k = 0;

  /* operate on two 2x4 blocks at a time */
  unsigned int l_a_stride = 32;
  unsigned int l_b_stride = i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in;

  /* vector registers holding B's values in MMLA form */
  unsigned int l_vr_b[6] = {0, 1, 2, 3, 4, 5};

  /* vector registers used as scratch for zips */
  unsigned int l_vr_zip[2] = {6, 7};

  /* vector registers holding A's values */
  unsigned int l_vr_a[2] = {6, 7};

  /* vector registers holding C's values */
  unsigned int l_vr_c[24] = {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

  /* flips the src registers in the MMLA instructions; require for signed-unsigned / unsigned-signed i8 switch */
  char l_flip_mmla_src = 0;

  /* select instructions */
  unsigned int l_instr_mmla = 0;

  for (l_n = 0; l_n < 6; l_n++) {
    l_vr_b[l_n] = l_n;
  }

  l_m_blocks = i_m_blocking / 4;
  l_n_blocks = i_n_blocking / 2;

  if( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
    l_instr_mmla = LIBXSMM_AARCH64_INSTR_ASIMD_BFMMLA_V;
  } else if( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
    if ( ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0) ) {
      l_instr_mmla = LIBXSMM_AARCH64_INSTR_ASIMD_UMMLA_V;
    } else if ( ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) == 0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0) ) {
      l_instr_mmla = LIBXSMM_AARCH64_INSTR_ASIMD_USMMLA_V;
    } else if ( ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) == 0) ) {
      l_instr_mmla = LIBXSMM_AARCH64_INSTR_ASIMD_USMMLA_V;
      l_flip_mmla_src = 1;
    } else {
      l_instr_mmla = LIBXSMM_AARCH64_INSTR_ASIMD_SMMLA_V;
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }

  /* load B */
  for( l_n = 0; l_n < l_n_blocks; l_n++ ) {
    libxsmm_aarch64_instruction_asimd_move( io_generated_code,
                                            LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                            i_gp_reg_mapping->gp_reg_b,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF,
                                            0,
                                            l_vr_zip[0],
                                            LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_b,
                                                   i_gp_reg_mapping->gp_reg_help_0,
                                                   i_gp_reg_mapping->gp_reg_b,
                                                   l_b_stride );

    libxsmm_aarch64_instruction_asimd_move( io_generated_code,
                                            LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                            i_gp_reg_mapping->gp_reg_b,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF,
                                            0,
                                            l_vr_zip[1],
                                            LIBXSMM_AARCH64_ASIMD_WIDTH_Q );

    if( l_n+1 != l_n_blocks ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_mapping->gp_reg_b,
                                                     i_gp_reg_mapping->gp_reg_help_0,
                                                     i_gp_reg_mapping->gp_reg_b,
                                                     l_b_stride );
    }
    else {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                     i_gp_reg_mapping->gp_reg_b,
                                                     i_gp_reg_mapping->gp_reg_help_0,
                                                     i_gp_reg_mapping->gp_reg_b,
                                                     (l_n_blocks-1)*l_b_stride*2 + l_b_stride - 16 );
    }
    libxsmm_aarch64_instruction_asimd_compute( io_generated_code,
                                               LIBXSMM_AARCH64_INSTR_ASIMD_ZIP1 ,
                                               l_vr_zip[0],
                                               l_vr_zip[1],
                                               0,
                                               l_vr_b[2*l_n + 0],
                                               LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
    libxsmm_aarch64_instruction_asimd_compute( io_generated_code,
                                               LIBXSMM_AARCH64_INSTR_ASIMD_ZIP2 ,
                                               l_vr_zip[0],
                                               l_vr_zip[1],
                                               0,
                                               l_vr_b[2*l_n + 1],
                                               LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
  }

  for( l_k = 0; l_k < l_k_blocks; l_k++ ) {
    for( l_m = 0; l_m < l_m_blocks; l_m++ ) {
      /* load A */
      libxsmm_aarch64_instruction_asimd_pair_move( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_ASIMD_LDP_I_POST,
                                                   i_gp_reg_mapping->gp_reg_a,
                                                   l_a_stride,
                                                   l_vr_a[0],
                                                   l_vr_a[1],
                                                   LIBXSMM_AARCH64_ASIMD_WIDTH_Q );

      /* MMLA compute */
      for( l_n = 0; l_n < l_n_blocks; l_n++ ) {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code,
                                                   l_instr_mmla,
                                                   (l_flip_mmla_src == 0 ) ? l_vr_b[2*l_n+l_k] : l_vr_a[0],
                                                   (l_flip_mmla_src == 0 ) ? l_vr_a[0]         : l_vr_b[2*l_n+l_k],
                                                   0,
                                                   l_vr_c[8*l_n + 2*l_m],
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_8H );

        libxsmm_aarch64_instruction_asimd_compute( io_generated_code,
                                                   l_instr_mmla,
                                                   (l_flip_mmla_src == 0 ) ? l_vr_b[2*l_n+l_k] : l_vr_a[1],
                                                   (l_flip_mmla_src == 0 ) ? l_vr_a[1]         : l_vr_b[2*l_n+l_k],
                                                   0,
                                                   l_vr_c[8*l_n + 2*l_m + 1],
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_8H );
      }
    }
    if( i_m_blocking != i_xgemm_desc->lda ) {
      /* each per-instruction matrix has 16 bytes (*16) and two rows (/2); thus scale by 8 */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_mapping->gp_reg_a,
                                                     i_gp_reg_mapping->gp_reg_help_0,
                                                     i_gp_reg_mapping->gp_reg_a,
                                                     (i_xgemm_desc->lda - i_m_blocking)*8 );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_microkernel_sve_a64fx( libxsmm_generated_code*            io_generated_code,
                                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                           const unsigned int                 i_m_blocking,
                                                           const unsigned int                 i_n_blocking ) {
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;

  unsigned int l_m_blocks[2] = { 0 }; /* 0: full vector ops, 1: remainder ops */
  unsigned int l_m_total_blocks = 0;
  unsigned int l_vec_reg_acc_start = 0;
  unsigned int l_remainder_size = 0;
  unsigned int l_b_stride = i_xgemm_desc->ldb;
  /* prep of B-ptr for next k-iteration */
  unsigned int l_b_next_k = 0;
  unsigned int l_b_next_k_inst = 0;

  l_m_blocks[0] = i_m_blocking / i_micro_kernel_config->vector_length;
  l_remainder_size = i_m_blocking % i_micro_kernel_config->vector_length;
  l_m_blocks[1] = (l_remainder_size > 0);
  l_m_total_blocks = l_m_blocks[0] + l_m_blocks[1];

  /* stride when accessing B */
  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
    l_b_stride = 1;
  }
  l_b_stride *= i_micro_kernel_config->datatype_size_in;

  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0 ) {
    if ( i_n_blocking == 1 ) {
      l_b_next_k = 1;
      l_b_next_k_inst = LIBXSMM_AARCH64_INSTR_GP_META_ADD;
    }
    else {
      l_b_next_k = ( (i_n_blocking - 1) * i_xgemm_desc->ldb - 1);
      l_b_next_k_inst = LIBXSMM_AARCH64_INSTR_GP_META_SUB;
    }
  }
  else {
    l_b_next_k = i_xgemm_desc->ldb - (i_n_blocking - 1);
    l_b_next_k_inst = LIBXSMM_AARCH64_INSTR_GP_META_ADD;
  }
  l_b_next_k *= i_micro_kernel_config->datatype_size_in;

  /* start register of accumulator */
  l_vec_reg_acc_start = i_micro_kernel_config->vector_reg_count - (i_n_blocking * l_m_total_blocks);

  /* full vector loads on a */
  for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
    libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                          LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF,
                                          i_gp_reg_mapping->gp_reg_a,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF,
                                          0,
                                          1 + l_m_total_blocks * l_n + l_m,
                                          LIBXSMM_AARCH64_SVE_REG_UNDEF );
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                   i_gp_reg_mapping->gp_reg_a,
                                                   i_gp_reg_mapping->gp_reg_a,
                                                   i_micro_kernel_config->vector_length * i_micro_kernel_config->datatype_size_in,
                                                   0 );
  }
  /* remainder load on a */
  if ( l_m_blocks[1] > 0) {
    libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                          (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF :
                                                                                           LIBXSMM_AARCH64_INSTR_SVE_LD1D_I_OFF,
                                          i_gp_reg_mapping->gp_reg_a,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF,
                                          0,
                                          1 + l_m_total_blocks * l_n + l_m_blocks[0],
                                          LIBXSMM_AARCH64_SVE_REG_P1 );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_a,
                                                   i_gp_reg_mapping->gp_reg_help_0,
                                                   i_gp_reg_mapping->gp_reg_a,
                                                   (unsigned long long)l_remainder_size * i_micro_kernel_config->datatype_size_in );
  }

  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    /* bcasts on b */
    libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                          (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF :
                                                                                           LIBXSMM_AARCH64_INSTR_SVE_LD1RD_I_OFF,
                                          i_gp_reg_mapping->gp_reg_b,
                                          0,
                                          0,
                                          0,
                                          0 );
    if ( l_n != i_n_blocking - 1 ) {
      /* move on to next entry of B */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_mapping->gp_reg_b,
                                                     i_gp_reg_mapping->gp_reg_help_0,
                                                     i_gp_reg_mapping->gp_reg_b,
                                                     l_b_stride );
    }
    else {
      /* prepare for next call of kernel */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     l_b_next_k_inst,
                                                     i_gp_reg_mapping->gp_reg_b,
                                                     i_gp_reg_mapping->gp_reg_help_0,
                                                     i_gp_reg_mapping->gp_reg_b,
                                                     l_b_next_k );
    }

    /* issue FMAs */
    for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
      libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                               LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_P,
                                               1 + l_m,
                                               0,
                                               (unsigned char)-1,
                                               l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n),
                                               LIBXSMM_AARCH64_SVE_REG_P0,
                                               (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_SVE_TYPE_S : LIBXSMM_AARCH64_SVE_TYPE_D );
    }
    if ( l_m_blocks[1] > 0 ) {
      libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                               LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_P,
                                               1 + l_m_blocks[0],
                                               0,
                                               (unsigned char)-1,
                                               l_vec_reg_acc_start + (l_m_total_blocks * l_n) + l_m_blocks[0],
                                               LIBXSMM_AARCH64_SVE_REG_P1,
                                               (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_SVE_TYPE_S : LIBXSMM_AARCH64_SVE_TYPE_D );
    }
  }

  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                       i_gp_reg_mapping->gp_reg_a,
                                                       i_gp_reg_mapping->gp_reg_help_0,
                                                       i_gp_reg_mapping->gp_reg_a,
                                                       0,
                                                       LIBXSMM_AARCH64_SHIFTMODE_LSL );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_microkernel_sve_mmla( libxsmm_generated_code*            io_generated_code,
                                                          const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                          const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                          const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                          const unsigned int                 i_m_blocking,
                                                          const unsigned int                 i_n_blocking ) {
  unsigned int l_m_blocks = 0;
  unsigned int l_m_blocks_remainder = 0;
  unsigned int l_m_remainder = 0;
  unsigned int l_n_blocks = 0;
  unsigned int l_k_blocks = 2;
  unsigned int l_n = 0;
  unsigned int l_m = 0;
  unsigned int l_k = 0;

  /* operate on four 2x4 blocks at a time */
  unsigned int l_a_stride = 64;
  unsigned int l_b_stride = i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in;

  /* gpr for k+4 (BFMMLA) and k+8 (int8) addresses when loading B */
  unsigned int l_gpr_b_k = LIBXSMM_AARCH64_GP_REG_X12;

  /* predicate register which is true in all relevant bits */
  unsigned int l_pr_all = 0;
  unsigned int l_pr_masked_a = 3;

  /* vector registers holding B's values in MMLA form */
  unsigned int l_vr_b[6] = {0, 1, 2, 3, 4, 5};

  /* vector registers used as scratch for zips */
  unsigned int l_vr_zip[2] = {6, 7};

  /* vector registers holding A's values */
  unsigned int l_vr_a[2] = {6, 7};

  /* vector registers holding C's values */
  unsigned int l_vr_c[24] = {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

  /* flips the src registers in the MMLA instructions; require for signed-unsigned / unsigned-signed i8 switch */
  char l_flip_mmla_src = 0;

  /* select instructions */
  unsigned int l_instr_mmla = 0;

  unsigned int l_k_blocking = 0;

  for (l_n = 0; l_n < 6; l_n++) {
    l_vr_b[l_n] = l_n;
  }

  l_m_blocks = i_m_blocking / 8;
  l_n_blocks = (i_n_blocking + 1) / 2;
  l_m_blocks_remainder = (i_m_blocking % 8 == 0) ? 0 : 1;
  l_m_remainder = i_m_blocking % 8;

  if( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
    l_instr_mmla = LIBXSMM_AARCH64_INSTR_SVE_BFMMLA_V;
    if (i_xgemm_desc->k % 8 == 0) {
      l_k_blocking = 8;
    } else {
      l_k_blocking = 4;
    }
    l_k_blocks = l_k_blocking / 4;
  } else if( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
    if ( ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0) ) {
      l_instr_mmla = LIBXSMM_AARCH64_INSTR_SVE_UMMLA_V;
    } else if ( ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) == 0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0) ) {
      l_instr_mmla = LIBXSMM_AARCH64_INSTR_SVE_USMMLA_V;
    } else if ( ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) == 0) ) {
      l_instr_mmla = LIBXSMM_AARCH64_INSTR_SVE_USMMLA_V;
      l_flip_mmla_src = 1;
    } else {
      l_instr_mmla = LIBXSMM_AARCH64_INSTR_SVE_SMMLA_V;
    }
    if (i_xgemm_desc->k % 16 == 0) {
      l_k_blocking = 16;
    } else {
      l_k_blocking = 8;
    }
    l_k_blocks = l_k_blocking / 8;
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }

  /* load B */
  if (l_k_blocks > 1) {
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                   i_gp_reg_mapping->gp_reg_b,
                                                   l_gpr_b_k,
                                                   8,
                                                   0 );
  }

  for( l_n = 0; l_n < l_n_blocks; l_n++ ) {
    /* load k */
    libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                          LIBXSMM_AARCH64_INSTR_SVE_LD1RD_I_OFF,
                                          i_gp_reg_mapping->gp_reg_b,
                                          0,
                                          0,
                                          l_vr_zip[0],
                                          l_pr_all );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_b,
                                                   i_gp_reg_mapping->gp_reg_help_0,
                                                   i_gp_reg_mapping->gp_reg_b,
                                                   l_b_stride);

    if ((i_n_blocking % 2 == 1) && (l_n == l_n_blocks-1)) {
      libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
          l_vr_zip[1], l_vr_zip[1], 0, l_vr_zip[1], 0, LIBXSMM_AARCH64_SVE_TYPE_S );
    } else {
      libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                            LIBXSMM_AARCH64_INSTR_SVE_LD1RD_I_OFF,
                                            i_gp_reg_mapping->gp_reg_b,
                                            0,
                                            0,
                                            l_vr_zip[1],
                                            l_pr_all );
    }
    if( l_n+1 != l_n_blocks ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_mapping->gp_reg_b,
                                                     i_gp_reg_mapping->gp_reg_help_0,
                                                     i_gp_reg_mapping->gp_reg_b,
                                                     l_b_stride );
    } else {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                     i_gp_reg_mapping->gp_reg_b,
                                                     i_gp_reg_mapping->gp_reg_help_0,
                                                     i_gp_reg_mapping->gp_reg_b,
                                                     (l_n_blocks-1)*l_b_stride*2 + l_b_stride - l_k_blocks * 8 );
    }
    libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                             LIBXSMM_AARCH64_INSTR_SVE_ZIP1_V ,
                                             l_vr_zip[0],
                                             l_vr_zip[1],
                                             0,
                                             l_vr_b[2*l_n + 0],
                                             LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                             LIBXSMM_AARCH64_SVE_TYPE_D );

    /* load k + 8 bytes */
    if (l_k_blocks > 1) {
      libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                            LIBXSMM_AARCH64_INSTR_SVE_LD1RD_I_OFF,
                                            l_gpr_b_k,
                                            0,
                                            0,
                                            l_vr_zip[0],
                                            l_pr_all );
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     l_gpr_b_k,
                                                     i_gp_reg_mapping->gp_reg_help_0,
                                                     l_gpr_b_k,
                                                     l_b_stride );

      if ((i_n_blocking % 2 == 1) && (l_n == l_n_blocks-1)) {
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
            l_vr_zip[1], l_vr_zip[1], 0, l_vr_zip[1], 0, LIBXSMM_AARCH64_SVE_TYPE_S );
      } else {
        libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                              LIBXSMM_AARCH64_INSTR_SVE_LD1RD_I_OFF,
                                              l_gpr_b_k,
                                              0,
                                              0,
                                              l_vr_zip[1],
                                              l_pr_all );
      }
      if( l_n+1 != l_n_blocks ) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       l_gpr_b_k,
                                                       i_gp_reg_mapping->gp_reg_help_0,
                                                       l_gpr_b_k,
                                                       l_b_stride );
      }
      libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                               LIBXSMM_AARCH64_INSTR_SVE_ZIP1_V ,
                                               l_vr_zip[0],
                                               l_vr_zip[1],
                                               0,
                                               l_vr_b[2*l_n + 1],
                                               LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                               LIBXSMM_AARCH64_SVE_TYPE_D );
    }
  }

  for( l_k = 0; l_k < l_k_blocks; l_k++ ) {
    for( l_m = 0; l_m < l_m_blocks + l_m_blocks_remainder; l_m++ ) {
      /* load A */
      if ((l_m <= l_m_blocks - 1) && (l_m_blocks > 0)) {
        libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                              LIBXSMM_AARCH64_INSTR_SVE_LD1D_I_OFF,
                                              i_gp_reg_mapping->gp_reg_a,
                                              0,
                                              0, /* TODO (MMLA): defaults to mul vl offset, function encoding? */
                                              l_vr_a[0],
                                              l_pr_all );

        libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                              LIBXSMM_AARCH64_INSTR_SVE_LD1D_I_OFF,
                                              i_gp_reg_mapping->gp_reg_a,
                                              0,
                                              1, /* TODO (MMLA): defaults to mul vl offset, function encoding? */
                                              l_vr_a[1],
                                              l_pr_all );

        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_mapping->gp_reg_a,
                                                       i_gp_reg_mapping->gp_reg_help_0,
                                                       i_gp_reg_mapping->gp_reg_a,
                                                       l_a_stride );
      } else {
        if (l_m_remainder >= 4) {
          libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                LIBXSMM_AARCH64_INSTR_SVE_LD1D_I_OFF,
                                                i_gp_reg_mapping->gp_reg_a,
                                                0,
                                                0, /* TODO (MMLA): defaults to mul vl offset, function encoding? */
                                                l_vr_a[0],
                                                l_pr_all );
          if (l_m_remainder > 4) {
            libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                  LIBXSMM_AARCH64_INSTR_SVE_LD1D_I_OFF,
                                                  i_gp_reg_mapping->gp_reg_a,
                                                  0,
                                                  1, /* TODO (MMLA): defaults to mul vl offset, function encoding? */
                                                  l_vr_a[1],
                                                  l_pr_masked_a );
          }
        } else {
          libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                LIBXSMM_AARCH64_INSTR_SVE_LD1D_I_OFF,
                                                i_gp_reg_mapping->gp_reg_a,
                                                0,
                                                0, /* TODO (MMLA): defaults to mul vl offset, function encoding? */
                                                l_vr_a[0],
                                                l_pr_masked_a );
        }
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_mapping->gp_reg_a,
                                                       i_gp_reg_mapping->gp_reg_help_0,
                                                       i_gp_reg_mapping->gp_reg_a,
                                                       l_m_remainder * 8 );
      }

      /* MMLA compute */
      for( l_n = 0; l_n < l_n_blocks; l_n++ ) {
        libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                                 l_instr_mmla ,
                                                 (l_flip_mmla_src == 0 ) ? l_vr_b[2*l_n+l_k] : l_vr_a[0],
                                                 (l_flip_mmla_src == 0 ) ? l_vr_a[0]         : l_vr_b[2*l_n+l_k],
                                                 0,
                                                 l_vr_c[8*l_n + 2*l_m],
                                                 LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                                 (libxsmm_aarch64_sve_type)0 );
        if (((l_m <= l_m_blocks - 1) && (l_m_blocks > 0))  || (l_m_remainder > 4)) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                                   l_instr_mmla ,
                                                   (l_flip_mmla_src == 0 ) ? l_vr_b[2*l_n+l_k] : l_vr_a[1],
                                                   (l_flip_mmla_src == 0 ) ? l_vr_a[1]         : l_vr_b[2*l_n+l_k],
                                                   0,
                                                   l_vr_c[8*l_n + 2*l_m + 1],
                                                   LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                                   (libxsmm_aarch64_sve_type)0 );
        }
      }
    }
    if( i_m_blocking != i_xgemm_desc->lda ) {
      /* each per-instruction matrix has 16 bytes (*16) and two rows (/2); thus scale by 8 */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_mapping->gp_reg_a,
                                                     i_gp_reg_mapping->gp_reg_help_0,
                                                     i_gp_reg_mapping->gp_reg_a,
                                                     (i_xgemm_desc->lda - i_m_blocking)*8 );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_kloop( libxsmm_generated_code*            io_generated_code,
                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                           const unsigned int                 i_m_blocking,
                                           const unsigned int                 i_n_blocking ) {
  /* some hard coded parameters for k-blocking */
  unsigned int l_k_blocking = 4;
  unsigned int l_k_threshold = 23;
  unsigned int l_k_stride = 1;
  char l_use_mmla = 0;
  void (*l_generator_microkernel)( libxsmm_generated_code*, const libxsmm_gp_reg_mapping*, const libxsmm_micro_kernel_config*, const libxsmm_gemm_descriptor*,
                                   const unsigned int, const unsigned int );
  /* TODO (MMLA) */
  /* enable MMLA settings for supported datatypes */
  if( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
    l_use_mmla = 1;
    if ( io_generated_code->arch == LIBXSMM_AARCH64_SVE256 || io_generated_code->arch == LIBXSMM_AARCH64_NEOV1 ) {
      if (i_xgemm_desc->k % 8 == 0) {
        l_k_blocking = 8;
        l_k_stride = 8;
      } else {
        l_k_blocking = 4;
        l_k_stride = 4;
      }
    } else {
      l_k_blocking = 8;
      l_k_stride = 8;
    }
  }
  else if( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
    l_use_mmla = 1;
    if ( io_generated_code->arch == LIBXSMM_AARCH64_SVE256 || io_generated_code->arch == LIBXSMM_AARCH64_NEOV1 ) {
      if (i_xgemm_desc->k % 16 == 0) {
        l_k_blocking = 16;
        l_k_stride = 16;
      } else {
        l_k_blocking = 8;
        l_k_stride = 8;
      }
    } else {
      l_k_blocking = 16;
      l_k_stride = 16;
    }
  }

  /* select micro kernel based on aarch64 variant */
  if ( io_generated_code->arch == LIBXSMM_AARCH64_V81 || io_generated_code->arch == LIBXSMM_AARCH64_V82 || io_generated_code->arch == LIBXSMM_AARCH64_APPL_M1 ) {
    /* TODO (MMLA) */
    if( l_use_mmla ) {
      l_generator_microkernel = libxsmm_generator_gemm_aarch64_microkernel_asimd_mmla;
    } else {
      l_generator_microkernel = libxsmm_generator_gemm_aarch64_microkernel_asimd_neoverse;
    }
  } else if ( io_generated_code->arch == LIBXSMM_AARCH64_SVE256 || io_generated_code->arch == LIBXSMM_AARCH64_NEOV1 ) {
    /* TODO (MMLA) */
    if( l_use_mmla ) {
      l_generator_microkernel = libxsmm_generator_gemm_aarch64_microkernel_sve_mmla;
    } else {
      l_generator_microkernel = libxsmm_generator_gemm_aarch64_microkernel_sve_a64fx;
    }
  } else if ( io_generated_code->arch == LIBXSMM_AARCH64_SVE512 || io_generated_code->arch == LIBXSMM_AARCH64_A64FX ) {
    l_generator_microkernel = libxsmm_generator_gemm_aarch64_microkernel_sve_a64fx;
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }

  /* apply multiple k_blocking strategies */
  /* 1. we are larger the k_threshold and a multiple of a predefined blocking parameter */
  if ((i_xgemm_desc->k % l_k_blocking) == 0 && (l_k_threshold < (unsigned int)i_xgemm_desc->k)) {
    unsigned int l_k;

    libxsmm_generator_gemm_aarch64_setup_k_strides(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
                                                   i_xgemm_desc, i_m_blocking, i_n_blocking);

    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_kloop, (unsigned int)i_xgemm_desc->k );

    /* TODO (MMLA): strided k loop breaks with original idea */
    for ( l_k = 0; l_k < l_k_blocking; l_k+=l_k_stride ) {
      l_generator_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, i_m_blocking, i_n_blocking);
    }

    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_kloop, l_k_blocking );
  } else {
    /* 2. we want to fully unroll below the threshold */
    if ((unsigned int)i_xgemm_desc->k <= l_k_threshold) {
      unsigned int l_k;

      libxsmm_generator_gemm_aarch64_setup_k_strides(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
                                                   i_xgemm_desc, i_m_blocking, i_n_blocking);

      /* TODO (MMLA): strided k loop breaks with original idea */
      for ( l_k = 0; l_k < (unsigned int)i_xgemm_desc->k; l_k+=l_k_stride ) {
        l_generator_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, i_m_blocking, i_n_blocking);
      }
    /* 3. we are larger than the threshold but not a multiple of the blocking factor -> largest possible blocking + remainder handling */
    } else {
      unsigned int l_max_blocked_k = ((i_xgemm_desc->k)/l_k_blocking)*l_k_blocking;
      unsigned int l_k;

      libxsmm_generator_gemm_aarch64_setup_k_strides(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
                                                   i_xgemm_desc, i_m_blocking, i_n_blocking);

      /* we can block as k is large enough */
      if ( l_max_blocked_k > 0 ) {
        libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_kloop, l_max_blocked_k );

        /* TODO (MMLA): strided k loop breaks with original idea */
        for ( l_k = 0; l_k < l_k_blocking; l_k+=l_k_stride ) {
          l_generator_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, i_m_blocking, i_n_blocking);
        }

        libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_kloop, l_k_blocking );
      }

      /* now we handle the remainder handling */
      for ( l_k = l_max_blocked_k; l_k < (unsigned int)i_xgemm_desc->k; l_k++) {
        l_generator_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, i_m_blocking, i_n_blocking);
      }
    }
  }

  /* reset A pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                 (unsigned long long)i_xgemm_desc->k * i_xgemm_desc->lda * i_micro_kernel_config->datatype_size_in );

  /* reset B pointer */
  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                   (unsigned long long)i_xgemm_desc->ldb * i_xgemm_desc->k * i_micro_kernel_config->datatype_size_in );
  } else {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                   (unsigned long long)i_xgemm_desc->k * i_micro_kernel_config->datatype_size_in );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_kernel( libxsmm_generated_code*        io_generated_code,
                                            const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;
  unsigned int l_n_n[2]     = {0,0};       /* blocking sizes for blocks */
  unsigned int l_n_N[2]     = {0,0};       /* size of blocks */
  unsigned int l_n_count    = 0;          /* array counter for blocking arrays */
  unsigned int l_n_done     = 0;           /* progress tracker */
  unsigned int l_n_done_old = 0;
  unsigned int a_vnni_factor  = 1;
  unsigned int l_ldc_saved = 0;

  /* Local variables used for A transpose case */
  libxsmm_descriptor_blob           l_blob_opa;
  libxsmm_gemm_descriptor*          l_xgemm_desc_opa;
  libxsmm_descriptor_blob           l_meltw_blob;
  libxsmm_mateltwise_kernel_config  l_mateltwise_kernel_config;
  libxsmm_mateltwise_gp_reg_mapping l_mateltwise_gp_reg_mapping;
  unsigned int                      lda_transpose;
  /* Local variables used only for older gemm setup (without stack frame) */
  unsigned int                      l_trans_a_stack_size       = 0;
  unsigned int                      l_trans_extra_stack_size   = 80;
  unsigned int                      l_transpose_stack_register = LIBXSMM_AARCH64_GP_REG_UNDEF;
  const libxsmm_meltw_descriptor *  l_mateltwise_desc;
  int gemm_stack_frame_is_set = 0;

  /* TODO (MMLA): clean up integration */
  char l_use_mmla = 0;
  char l_mmla_zip_row_major = 0;

  /* enable MMLA settings for supported datatypes */
  if(    LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )
      || LIBXSMM_DATATYPE_I8   == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
    l_use_mmla = 1;
    if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      a_vnni_factor = 4;
    }
    if (LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      a_vnni_factor = 8;
    }
  }

  /* define gp register mapping */
  libxsmm_reset_aarch64_gp_reg_mapping( &l_gp_reg_mapping );

  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_AARCH64_GP_REG_X0;
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_AARCH64_GP_REG_X0;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_AARCH64_GP_REG_X1;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_AARCH64_GP_REG_X2;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_AARCH64_GP_REG_X3;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_AARCH64_GP_REG_X4;
  /*l_gp_reg_mapping.gp_reg_c_prefetch = LIBXSMM_AARCH64_GP_REG_X5;*/
  l_gp_reg_mapping.gp_reg_mloop = LIBXSMM_AARCH64_GP_REG_X6;
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_AARCH64_GP_REG_X7;
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_AARCH64_GP_REG_X8;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_AARCH64_GP_REG_X9;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_AARCH64_GP_REG_X10;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_AARCH64_GP_REG_X11;
  l_gp_reg_mapping.gp_reg_reduce_count = LIBXSMM_AARCH64_GP_REG_X3;
  l_gp_reg_mapping.gp_reg_a_offset = LIBXSMM_AARCH64_GP_REG_X4;
  l_gp_reg_mapping.gp_reg_b_offset = LIBXSMM_AARCH64_GP_REG_X5;
  l_gp_reg_mapping.gp_reg_help_3 = LIBXSMM_AARCH64_GP_REG_X27;      /* storing forward counting BRGEMM interations */
  l_gp_reg_mapping.gp_reg_help_4 = LIBXSMM_AARCH64_GP_REG_X28;      /* for a ptr updates in BRGEMM */
  l_gp_reg_mapping.gp_reg_help_5 = LIBXSMM_AARCH64_GP_REG_X26;      /* for b ptr updates in BRGEMM */
  l_gp_reg_mapping.gp_reg_reduce_loop = LIBXSMM_AARCH64_GP_REG_X30; /* BRGEMM loop */

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* compute n blocking, based on m blocking */
  libxsmm_generator_gemm_aarch64_setup_n_blocking( io_generated_code, &l_micro_kernel_config, i_xgemm_desc, io_generated_code->arch, l_n_N, l_n_n );

  /* check that l_n_N1 is non-zero */
  if ( l_n_N[0] == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* open asm */
  libxsmm_aarch64_instruction_open_stream( io_generated_code, 0xe0f );

  /* ensuring compatibility with X86 AMX */
  if ( !( (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & i_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & i_xgemm_desc->flags) == 0)) ||
          (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & i_xgemm_desc->flags) != 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & i_xgemm_desc->flags) != 0))    ) ) {
    /* close asm */
    libxsmm_aarch64_instruction_close_stream( io_generated_code, 0xe0f );
    return;
  }

  /* in case when A needs to be transposed, we need to change temporarily the desciptor dimensions for gemm, hence the local descriptor */
  lda_transpose = i_xgemm_desc->m;
  if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) {
    if (LIBXSMM_DATATYPE_F32 == (libxsmm_datatype)(i_xgemm_desc->datatype))
      l_xgemm_desc_opa = libxsmm_sgemm_descriptor_init(&l_blob_opa, i_xgemm_desc->m, i_xgemm_desc->n, i_xgemm_desc->k,
        lda_transpose,
        i_xgemm_desc->ldb,
        i_xgemm_desc->ldc,
        1.f /*alpha* is unused but used in the BYPASS check? */, (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BETA_0 ? 0.f : 1.f) /*beta is already in the flags?*/,
        (unsigned int)((unsigned int)(i_xgemm_desc->flags) & (~LIBXSMM_GEMM_FLAG_TRANS_A)), i_xgemm_desc->prefetch);
    else if (LIBXSMM_DATATYPE_F64 == (libxsmm_datatype)(i_xgemm_desc->datatype))
      l_xgemm_desc_opa = libxsmm_dgemm_descriptor_init(&l_blob_opa, i_xgemm_desc->m, i_xgemm_desc->n, i_xgemm_desc->k,
        lda_transpose,
        i_xgemm_desc->ldb,
        i_xgemm_desc->ldc,
        1.0 /*alpha* is unused but used in the BYPASS check? */, (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BETA_0 ? 0.0 : 1.0) /*beta is already in the flags?*/,
        (unsigned int)((unsigned int)(i_xgemm_desc->flags) & (~LIBXSMM_GEMM_FLAG_TRANS_A)), i_xgemm_desc->prefetch);
    else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  } else {
    l_xgemm_desc_opa = (libxsmm_gemm_descriptor*) i_xgemm_desc;
  }

  /* implementing load from struct */
  if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI & l_xgemm_desc_opa->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI) ||
       ((LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI & l_xgemm_desc_opa->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI) ) {
    /* RDI holds the pointer to the struct, so lets first move this one into R15 */
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_AND_SR,
                                                         l_gp_reg_mapping.gp_reg_param_struct, l_gp_reg_mapping.gp_reg_param_struct, l_gp_reg_mapping.gp_reg_help_1,
                                                         0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    /* A pointer */
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 32, l_gp_reg_mapping.gp_reg_a );
    /* B pointer */
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 64, l_gp_reg_mapping.gp_reg_b );
    /* C pointer */
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 96, l_gp_reg_mapping.gp_reg_c );
    if ( l_xgemm_desc_opa->prefetch != LIBXSMM_GEMM_PREFETCH_NONE ) {
      /* A prefetch pointer */
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 56, l_gp_reg_mapping.gp_reg_a_prefetch );
      /* B preftech pointer */
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 88, l_gp_reg_mapping.gp_reg_b_prefetch );
    }
    /* batch reduce count & offsett arrays*/
    if ((l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) || (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET)) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 16, l_gp_reg_mapping.gp_reg_reduce_count );

      if ( l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET ) {
        libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                         l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 40, l_gp_reg_mapping.gp_reg_a_offset );
        libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                         l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 72, l_gp_reg_mapping.gp_reg_b_offset );
      }
    }
    /* check values for gemm_ext */
#if 0
    if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI & l_xgemm_desc_opa->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI) ) {
      if ( (l_xgemm_desc_opa->meltw_operation != LIBXSMM_MELTW_OPERATION_NONE) || (l_xgemm_desc_opa->eltw_ap_op != LIBXSMM_MELTW_OPERATION_NONE) ||
           (l_xgemm_desc_opa->eltw_bp_op != LIBXSMM_MELTW_OPERATION_NONE) || (l_xgemm_desc_opa->eltw_cp_op != LIBXSMM_MELTW_OPERATION_NONE) ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_INVALID_GEMM_CONFIG );
        return;
      }
    }
#endif
  }

  /* setting up the stack frame */
  libxsmm_generator_gemm_setup_stack_frame_aarch64( io_generated_code, i_xgemm_desc, &l_gp_reg_mapping, &l_micro_kernel_config);
  gemm_stack_frame_is_set = 1;

  /* In this case we store C to scratch  */
  if (l_micro_kernel_config.vnni_format_C > 0) {
    libxsmm_generator_gemm_setval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_TRANS_EXT_BUF_C, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_c);
    libxsmm_generator_gemm_getval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, l_gp_reg_mapping.gp_reg_help_1);
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_c, 32LL * 64LL );
    l_ldc_saved = i_xgemm_desc->ldc;
    l_xgemm_desc_opa->ldc = i_xgemm_desc->m;
  }

  if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) &&
       (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT) ) {
    int l_nnz_bits = l_xgemm_desc_opa->m%l_micro_kernel_config.vector_length;
    l_nnz_bits *= l_micro_kernel_config.datatype_size_out;
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                  LIBXSMM_AARCH64_SVE_REG_P0,
                                                  -1,
                                                  l_gp_reg_mapping.gp_reg_help_0 );

    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                  LIBXSMM_AARCH64_SVE_REG_P1,
                                                  l_nnz_bits,
                                                  l_gp_reg_mapping.gp_reg_help_0 );

    if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) {
      l_nnz_bits = l_micro_kernel_config.vector_length * l_micro_kernel_config.datatype_size_out;
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                    LIBXSMM_AARCH64_SVE_REG_P2,
                                                    l_nnz_bits,
                                                    l_gp_reg_mapping.gp_reg_help_0 );
    }

    if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      /* For A we load in chunks of 8 bytes since A in VNNI4 */
      l_nnz_bits = (l_xgemm_desc_opa->m%4) * 8;
      if (l_nnz_bits > 0) {
        libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                      LIBXSMM_AARCH64_SVE_REG_P3,
                                                      l_nnz_bits,
                                                      l_gp_reg_mapping.gp_reg_help_0 );
      }
    }
  }

  if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) {
    if ( !gemm_stack_frame_is_set ) {
      unsigned int temp_reg  = l_gp_reg_mapping.gp_reg_help_1;
      unsigned int temp_reg2 = l_gp_reg_mapping.gp_reg_help_2;
      /* Saving current SP and aligning the stack at 64-byte boundary */
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_X29, 0, 0 );
      libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, temp_reg, 0xFFFFFFFFFFFFFFC0 );
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, temp_reg2, 0, 0 );
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_AND_SR, temp_reg2, temp_reg, temp_reg2, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, temp_reg2, LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );

      /* allocate space on the stack (keeping it 16-byte aligned) */
      l_trans_a_stack_size = LIBXSMM_UP(i_xgemm_desc->m * i_xgemm_desc->k * LIBXSMM_TYPESIZE(LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype )), 16);
      libxsmm_aarch64_instruction_alu_compute_imm24( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I,
                                                     LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP,
                                                     l_trans_a_stack_size );
      /* saving start of the allocated stack space in a register which is not used in the transpose */
      l_transpose_stack_register = LIBXSMM_AARCH64_GP_REG_X12;
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I,
                                                     LIBXSMM_AARCH64_GP_REG_XSP, l_transpose_stack_register,
                                                     0, 0 );
    } else {
      /* If gemm_stack frame has been set (and A transpose scratch space enabled there), some other code is needed (see x86 gemm) */
      l_transpose_stack_register = LIBXSMM_AARCH64_GP_REG_X12;
      libxsmm_generator_gemm_getval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_TRANSPOSE_PTR, l_transpose_stack_register);
    }

    /* allocate space for 80 more bytes on the stack and pushing adress for the input (A) and of the start of the transpose there */
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I,
                                                   LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP,
                                                   l_trans_extra_stack_size, 0 );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XZR, 64,
                                          l_transpose_stack_register);
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XZR, 32,
                                          l_gp_reg_mapping.gp_reg_a);

    /* pushing X0 on the stack to restore it later after the transpose */
    /* FIXME: Unnecessary? As true X0 was saved as X10 in gemm setup and X0 is already re-used as a pointer to A (which we anyway change after transpose) */
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XZR, 8,
                                          LIBXSMM_AARCH64_GP_REG_X0 );

    /* the transpose microkernels called below use x0 + 32 for input and x0 + 64 for output so they are set here */
    /* modifying X0 so that X0+32, X0+64 point to the input/output for mateltwise (put on the stack just above) */
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                   LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_X0,
                                                   0, 0 );

    /* creating a descriptor for the meltwise transform (transpose) */
    l_mateltwise_desc = libxsmm_meltw_descriptor_init(&l_meltw_blob,
      (libxsmm_datatype)i_xgemm_desc->datatype, (libxsmm_datatype)i_xgemm_desc->datatype,
      i_xgemm_desc->k /*m*/, i_xgemm_desc->m /*n*/,
      i_xgemm_desc->lda, i_xgemm_desc->m,
      LIBXSMM_CAST_USHORT(i_xgemm_desc->flags), LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, LIBXSMM_MELTW_OPERATION_UNARY);

    memset(&l_mateltwise_gp_reg_mapping, 0, sizeof(l_mateltwise_gp_reg_mapping));

    l_mateltwise_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_AARCH64_GP_REG_X0;

    /* define mateltwise kernel config */
    libxsmm_generator_mateltwise_aarch64_init_micro_kernel_config_fullvector( io_generated_code, &l_mateltwise_kernel_config, l_mateltwise_desc );

    /* being BLAS aligned, for empty kernels, do nothing */
    /* FIXME: Did not understand the check, copied from libxsmm_generator_mateltwise_aarch64_kernel() */
    if ( (l_mateltwise_desc->m > 0) && ((l_mateltwise_desc->n > 0) || (l_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) || libxsmm_matrix_eqn_is_unary_opcode_reduce_cols_idx_kernel(l_mateltwise_desc->param)) ) {
      /* Stack management for melt kernel */
      libxsmm_generator_meltw_setup_stack_frame_aarch64( io_generated_code, l_mateltwise_desc, &l_mateltwise_gp_reg_mapping, &l_mateltwise_kernel_config);

      if (l_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY && (l_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT)) {
          libxsmm_generator_transform_aarch64_microkernel( io_generated_code, &l_loop_label_tracker, &l_mateltwise_gp_reg_mapping, &l_mateltwise_kernel_config, l_mateltwise_desc );
      } else {
        /* This should not happen  */
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
        return;
      }

      /* Stack management for melt kernel */
      libxsmm_generator_meltw_destroy_stack_frame_aarch64(  io_generated_code, l_mateltwise_desc, &l_mateltwise_kernel_config );
    }

    /* popping back X0 after the transpose */
    /* FIXME: Unnecessary (see the comment at pushing above) */
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XZR, 8,
                                          LIBXSMM_AARCH64_GP_REG_X0 );

    /* deallocate space for 80 bytes on the stack */
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                   LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP,
                                                   l_trans_extra_stack_size, 0 );

    /* setting pointer to A for the gemm code to the transpose on the stack */
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                   l_transpose_stack_register, l_gp_reg_mapping.gp_reg_a,
                                                   0, 0 );
  } /* if A needs to be transposed */

  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* calling gemm kernel with the modified pointer to the first matrix (now trans_a on the stack) should go here */

  /* apply n_blocking */
  while (l_n_done != (unsigned int)l_xgemm_desc_opa->n) {
    unsigned int l_n_blocking = l_n_n[l_n_count];
    unsigned int l_m_done = 0;
    unsigned int l_m_done_old = 0;
    unsigned int l_m_blocking = 0;

    /* advance N */
    l_n_done_old = l_n_done;
    l_n_done += l_n_N[l_n_count];
    l_n_count++;

    /* open N loop */
    libxsmm_generator_loop_header_aarch64( io_generated_code, &l_loop_label_tracker,
                                                l_gp_reg_mapping.gp_reg_nloop, l_n_done - l_n_done_old );

    /* define the micro kernel code gen properties, especially m-blocking affects the vector instruction length */
    l_m_blocking = libxsmm_generator_gemm_aarch64_get_initial_m_blocking( &l_micro_kernel_config, l_xgemm_desc_opa, io_generated_code->arch );

    /* TODO (MMLA): remove, hardcoded */
    if(    LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )
        || LIBXSMM_DATATYPE_I8   == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      if ( io_generated_code->arch < LIBXSMM_AARCH64_SVE128 ) {
        l_m_blocking = 16;
      }
      else {
        l_m_blocking = 32; /* TODO (MMLA): only 256bit */
      }
      if ( ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) == 0) ) {
        l_mmla_zip_row_major = 1;
      }
    }

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
        /* open M loop */
        libxsmm_generator_loop_header_aarch64( io_generated_code, &l_loop_label_tracker,
                                               l_gp_reg_mapping.gp_reg_mloop, l_m_done - l_m_done_old );
        /* load block of C */
        if ( io_generated_code->arch == LIBXSMM_AARCH64_V81 || io_generated_code->arch == LIBXSMM_AARCH64_V82 || io_generated_code->arch == LIBXSMM_AARCH64_APPL_M1 ) {
          /* TODO: MMLA */
          if( l_use_mmla ) {
            libxsmm_generator_load_2dregblock_mmla_aarch64_asimd( io_generated_code, &l_micro_kernel_config, l_xgemm_desc_opa, l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_0,
                                                                  l_micro_kernel_config.vector_length, l_micro_kernel_config.vector_reg_count, l_m_blocking, l_n_blocking,
                                                                  l_xgemm_desc_opa->ldc * l_micro_kernel_config.datatype_size_out,
                                                                  (LIBXSMM_GEMM_FLAG_BETA_0 & l_xgemm_desc_opa->flags),
                                                                  l_mmla_zip_row_major );
          } else {
            if ((l_micro_kernel_config.fused_scolbias == 1) || (l_micro_kernel_config.fused_bcolbias == 1)) {
              libxsmm_generator_gemm_load_add_colbias_2dregblock_aarch64( io_generated_code, l_xgemm_desc_opa, l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_0,
                  l_micro_kernel_config.vector_length, l_micro_kernel_config.vector_reg_count, LIBXSMM_DATATYPE_F32, l_m_blocking, l_n_blocking, l_xgemm_desc_opa->ldc * l_micro_kernel_config.datatype_size_out, l_micro_kernel_config.datatype_size_out );
            } else {
              libxsmm_generator_load_2dregblock_aarch64_asimd( io_generated_code, l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_0,
                                                               l_micro_kernel_config.vector_length, l_micro_kernel_config.vector_reg_count, l_m_blocking, l_n_blocking,
                                                               l_xgemm_desc_opa->ldc * l_micro_kernel_config.datatype_size_out,
                                                               (LIBXSMM_GEMM_FLAG_BETA_0 & l_xgemm_desc_opa->flags) );
            }
          }
        } else if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) &&
                    (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT) ) {
          /* TODO: MMLA */
          if( l_use_mmla ) {
            libxsmm_generator_load_2dregblock_mmla_aarch64_sve( io_generated_code, &l_micro_kernel_config, l_xgemm_desc_opa, l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_help_1,
                                                                l_micro_kernel_config.vector_length, l_micro_kernel_config.vector_reg_count, l_m_blocking, l_n_blocking,
                                                                l_xgemm_desc_opa->ldc * l_micro_kernel_config.datatype_size_out,
                                                                l_micro_kernel_config.datatype_size_out,
                                                                (LIBXSMM_GEMM_FLAG_BETA_0 & l_xgemm_desc_opa->flags),
                                                                l_mmla_zip_row_major );
          } else {
            if ((l_micro_kernel_config.fused_scolbias == 1) || (l_micro_kernel_config.fused_bcolbias == 1)) {
              libxsmm_generator_gemm_load_add_colbias_2dregblock_aarch64( io_generated_code, l_xgemm_desc_opa, l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_0,
                  l_micro_kernel_config.vector_length, l_micro_kernel_config.vector_reg_count, LIBXSMM_DATATYPE_F32, l_m_blocking, l_n_blocking, l_xgemm_desc_opa->ldc * l_micro_kernel_config.datatype_size_out, l_micro_kernel_config.datatype_size_out );
            } else {
              libxsmm_generator_load_2dregblock_aarch64_sve( io_generated_code, l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_0,
                                                             l_micro_kernel_config.vector_length, l_micro_kernel_config.vector_reg_count, l_m_blocking, l_n_blocking,
                                                             l_xgemm_desc_opa->ldc * l_micro_kernel_config.datatype_size_out,
                                                             l_micro_kernel_config.datatype_size_out,
                                                             (LIBXSMM_GEMM_FLAG_BETA_0 & l_xgemm_desc_opa->flags) );
            }
          }
        } else {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
          return;
        }

        /* handle BRGEMM */
        if ( ((l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0) ||
             ((l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) >  0) ||
             ((l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) >  0)    ) {
          /* we need to load the real address */
          if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0 ) {
            libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_gp_reg_mapping.gp_reg_help_4, 0 );
            libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_gp_reg_mapping.gp_reg_help_5, 0 );
          } else {
            libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_gp_reg_mapping.gp_reg_help_3, 0 );
            libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                           l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_help_4, 0, 0 );
            libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                           l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_5, 0, 0 );
          }
          /* open BR loop */
          libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, l_gp_reg_mapping.gp_reg_reduce_count, LIBXSMM_AARCH64_GP_REG_XZR, 0,
                                                l_gp_reg_mapping.gp_reg_reduce_loop );
          libxsmm_aarch64_instruction_register_jump_back_label( io_generated_code, &l_loop_label_tracker );

          /* we need to load the real address of A and B for this reduce operation */
          if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0 ) {
            libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_R, l_gp_reg_mapping.gp_reg_help_4, l_gp_reg_mapping.gp_reg_help_3, 0,
                                                  l_gp_reg_mapping.gp_reg_a );
            libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_R, l_gp_reg_mapping.gp_reg_help_5, l_gp_reg_mapping.gp_reg_help_3, 0,
                                                  l_gp_reg_mapping.gp_reg_b );
          }
          if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) > 0 ) {
            libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_R, l_gp_reg_mapping.gp_reg_a_offset, l_gp_reg_mapping.gp_reg_help_3, 0,
                                                  l_gp_reg_mapping.gp_reg_help_0 );
            libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, l_gp_reg_mapping.gp_reg_help_4,
                                                                 l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
            libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_R, l_gp_reg_mapping.gp_reg_b_offset, l_gp_reg_mapping.gp_reg_help_3, 0,
                                                  l_gp_reg_mapping.gp_reg_help_1 );
            libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, l_gp_reg_mapping.gp_reg_help_5,
                                                                 l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_b, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
          }
        }

        /* compute outer product */
        libxsmm_generator_gemm_aarch64_kloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config,
                                              l_xgemm_desc_opa, l_m_blocking, l_n_blocking );

        if ( ((l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0) ||
             ((l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) >  0) ||
             ((l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) >  0)    ) {
          /* increment forward counting BRGEMM count */
          if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0 ) {
            /* nothing to do */
          } else {
            libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                           l_gp_reg_mapping.gp_reg_help_3, l_gp_reg_mapping.gp_reg_help_3, 8, 0 );
          }
          if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0 ) {
            libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_gp_reg_mapping.gp_reg_help_0, (unsigned long long)l_xgemm_desc_opa->c1 );
            libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_gp_reg_mapping.gp_reg_help_1, (unsigned long long)l_xgemm_desc_opa->c2 );
            libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, l_gp_reg_mapping.gp_reg_help_0,
                                                                 l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_a, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
            libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, l_gp_reg_mapping.gp_reg_help_1,
                                                                 l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_b, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
            libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, l_gp_reg_mapping.gp_reg_help_0,
                                                                 l_gp_reg_mapping.gp_reg_help_4, l_gp_reg_mapping.gp_reg_help_4, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
            libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, l_gp_reg_mapping.gp_reg_help_1,
                                                                 l_gp_reg_mapping.gp_reg_help_5, l_gp_reg_mapping.gp_reg_help_5, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
          }

          /* close BRGEMM loop */
          libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_reduce_loop, 1 );

          /* restore A and B register */
          if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0 ) {
            libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR, l_gp_reg_mapping.gp_reg_a,
                                                                 l_gp_reg_mapping.gp_reg_help_4, l_gp_reg_mapping.gp_reg_a, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
            libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR, l_gp_reg_mapping.gp_reg_b,
                                                                 l_gp_reg_mapping.gp_reg_help_5, l_gp_reg_mapping.gp_reg_b, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
          } else {
            libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                           l_gp_reg_mapping.gp_reg_help_4, l_gp_reg_mapping.gp_reg_a, 0, 0 );
            libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                           l_gp_reg_mapping.gp_reg_help_5, l_gp_reg_mapping.gp_reg_b, 0, 0 );
          }
        }

        /* store block of C */
        if ( io_generated_code->arch == LIBXSMM_AARCH64_V81 || io_generated_code->arch == LIBXSMM_AARCH64_V82 || io_generated_code->arch == LIBXSMM_AARCH64_APPL_M1 ) {
          /* TODO: MMLA */
          if( l_use_mmla ) {
            libxsmm_generator_store_2dregblock_mmla_aarch64_asimd( io_generated_code, &l_micro_kernel_config, l_xgemm_desc_opa, l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_0,
                                                                   l_micro_kernel_config.vector_length, l_micro_kernel_config.vector_reg_count, l_m_blocking, l_n_blocking,
                                                                   l_xgemm_desc_opa->ldc * l_micro_kernel_config.datatype_size_out,
                                                                   l_mmla_zip_row_major );
          } else {
            /* Apply potential fusion to 2dregblock before storing it out */
            libxsmm_generator_gemm_apply_fusion_2dregblock_aarch64( io_generated_code, l_xgemm_desc_opa, &l_micro_kernel_config, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_help_1, l_micro_kernel_config.vector_length,
                l_micro_kernel_config.vector_reg_count, l_m_blocking, l_n_blocking, l_micro_kernel_config.datatype_size_out  );
            libxsmm_generator_store_2dregblock_aarch64_asimd( io_generated_code, l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_0,
                                                              l_micro_kernel_config.vector_length, l_micro_kernel_config.vector_reg_count, l_m_blocking, l_n_blocking,
                                                              l_xgemm_desc_opa->ldc * l_micro_kernel_config.datatype_size_out );
          }
        } else if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) &&
                    (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT) ) {
          if( l_use_mmla ) {
            /* Apply sigmoid fusion at FP32 registers */
            if (l_micro_kernel_config.fused_sigmoid > 0) {
              libxsmm_generator_gemm_apply_sigmoid_fusion_2dregblock_aarch64_sve( io_generated_code, l_xgemm_desc_opa, &l_micro_kernel_config,
                  l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_help_0, l_micro_kernel_config.vector_length, l_micro_kernel_config.vector_reg_count, l_m_blocking, l_n_blocking, sizeof(float), l_use_mmla );
            }
            libxsmm_generator_store_2dregblock_mmla_aarch64_sve( io_generated_code, &l_micro_kernel_config, l_xgemm_desc_opa, l_gp_reg_mapping.gp_reg_c,
                                                                 l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_help_3,
                                                                 l_micro_kernel_config.vector_length, l_micro_kernel_config.vector_reg_count, l_m_blocking, l_n_blocking,
                                                                 l_xgemm_desc_opa->ldc * l_micro_kernel_config.datatype_size_out,
                                                                 l_micro_kernel_config.datatype_size_out,
                                                                 (LIBXSMM_GEMM_FLAG_BETA_0 & l_xgemm_desc_opa->flags),
                                                                 l_mmla_zip_row_major );

            /* reset A mask in case of fused relu bitmask since it is drstroyed by the store algo */
            if ((l_micro_kernel_config.fused_relu == 1) && (l_micro_kernel_config.overwrite_C == 1) && (l_xgemm_desc_opa->m % 4 > 0)) {
              if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
                /* For A we load in chunks of 8 bytes since A in VNNI4 */
                int l_nnz_bits = (l_xgemm_desc_opa->m%4) * 8;
                if (l_nnz_bits > 0) {
                  libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                                LIBXSMM_AARCH64_SVE_REG_P3,
                                                                l_nnz_bits,
                                                                l_gp_reg_mapping.gp_reg_help_0 );
                }
              }
            }
          } else {
            /* Apply potential fusion to 2dregblock before storing it out */
            libxsmm_generator_gemm_apply_fusion_2dregblock_aarch64( io_generated_code, l_xgemm_desc_opa, &l_micro_kernel_config, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_help_1, l_micro_kernel_config.vector_length,
                l_micro_kernel_config.vector_reg_count, l_m_blocking, l_n_blocking, l_micro_kernel_config.datatype_size_out  );
            libxsmm_generator_store_2dregblock_aarch64_sve( io_generated_code, l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_0,
                                                            l_micro_kernel_config.vector_length, l_micro_kernel_config.vector_reg_count, l_m_blocking, l_n_blocking,
                                                            l_xgemm_desc_opa->ldc * l_micro_kernel_config.datatype_size_out,
                                                            l_micro_kernel_config.datatype_size_out );
          }
        } else {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
          return;
        }

        /* advance C pointer */
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_c,
                                                       (unsigned long long)l_m_blocking*l_micro_kernel_config.datatype_size_out );
        /* Adjust relu bitmask pointer  */
        if ((l_micro_kernel_config.fused_relu == 1) && (l_micro_kernel_config.overwrite_C == 1) ) {
          libxsmm_generator_gemm_getval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, l_gp_reg_mapping.gp_reg_help_1);
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                         l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_help_1,
                                                         (unsigned long long)(l_m_blocking+7)/8);
          libxsmm_generator_gemm_setval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_help_1);
        }

        /* Adjust colbias ptr */
        if ((l_micro_kernel_config.fused_scolbias == 1) || (l_micro_kernel_config.fused_bcolbias == 1)) {
          libxsmm_generator_gemm_getval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, l_gp_reg_mapping.gp_reg_help_1);
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                         l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_help_1,
                                                         (l_micro_kernel_config.fused_scolbias == 1) ? (unsigned long long)l_m_blocking*4 : (unsigned long long)l_m_blocking*2 ) ;
          libxsmm_generator_gemm_setval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_help_1);
        }

        /* advance A pointer */
        if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0 ) {
          libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_gp_reg_mapping.gp_reg_help_3, 0 );
          libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                         l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_help_4, 0, 0 );

          /* open BR loop */
          libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, l_gp_reg_mapping.gp_reg_reduce_count, LIBXSMM_AARCH64_GP_REG_XZR, 0,
                                                l_gp_reg_mapping.gp_reg_reduce_loop );
          libxsmm_aarch64_instruction_register_jump_back_label( io_generated_code, &l_loop_label_tracker );

          /* update A pointer */
          libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_R, l_gp_reg_mapping.gp_reg_help_4, l_gp_reg_mapping.gp_reg_help_3, 0,
                                                l_gp_reg_mapping.gp_reg_a );

          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                         l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a,
                                                         (unsigned long long)l_m_blocking*l_micro_kernel_config.datatype_size_in*a_vnni_factor );

          libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STR_R, l_gp_reg_mapping.gp_reg_help_4, l_gp_reg_mapping.gp_reg_help_3, 0,
                                                l_gp_reg_mapping.gp_reg_a );

          libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                         l_gp_reg_mapping.gp_reg_help_3, l_gp_reg_mapping.gp_reg_help_3, 8, 0 );
          /* close BRGEMM loop */
          libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_reduce_loop, 1 );

          /* reset A */
          libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                         l_gp_reg_mapping.gp_reg_help_4, l_gp_reg_mapping.gp_reg_a, 0, 0 );
        } else {
          /* TODO (MMLA): do this properly; right now jumps according to A matrix in MMLA format */
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                         l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a,
                                                         (unsigned long long)l_m_blocking*l_micro_kernel_config.datatype_size_in*a_vnni_factor );
        }

        /* close M loop */
        libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker,
                                               l_gp_reg_mapping.gp_reg_mloop, l_m_blocking );
      }

      /* switch to next smaller m_blocking */
      l_m_blocking = libxsmm_generator_gemm_aarch64_update_m_blocking( &l_micro_kernel_config, l_xgemm_desc_opa, io_generated_code->arch, l_m_blocking );
    }
    /* reset C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_c,
                                                   ((unsigned long long)l_n_blocking * l_xgemm_desc_opa->ldc * l_micro_kernel_config.datatype_size_out) -
                                                   ((unsigned long long)l_xgemm_desc_opa->m * l_micro_kernel_config.datatype_size_out) );

    /* Adjust relu bitmask pointer  */
    if ((l_micro_kernel_config.fused_relu == 1) && (l_micro_kernel_config.overwrite_C == 1) ) {
      libxsmm_generator_gemm_getval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, l_gp_reg_mapping.gp_reg_help_1);
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_help_1,
                                                     (((unsigned long long)l_n_blocking*i_xgemm_desc->ldcp)/8) - (((unsigned long long)l_xgemm_desc_opa->m+7)/8));
      libxsmm_generator_gemm_setval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_help_1);
    }

    /* Adjust colbias ptr */
    if ((l_micro_kernel_config.fused_scolbias == 1) || (l_micro_kernel_config.fused_bcolbias == 1)) {
      libxsmm_generator_gemm_getval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, l_gp_reg_mapping.gp_reg_help_1);
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                     l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_help_1,
                                                     (l_micro_kernel_config.fused_scolbias == 1) ? (unsigned long long)l_xgemm_desc_opa->m*4 : (unsigned long long)l_xgemm_desc_opa->m *2 ) ;
      libxsmm_generator_gemm_setval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_help_1);
    }

    if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0 ) {
      libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_gp_reg_mapping.gp_reg_help_3, 0 );
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                    l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_help_4, 0, 0 );
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                    l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_5, 0, 0 );

      /* open BR loop */
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, l_gp_reg_mapping.gp_reg_reduce_count, LIBXSMM_AARCH64_GP_REG_XZR, 0,
                                            l_gp_reg_mapping.gp_reg_reduce_loop );
      libxsmm_aarch64_instruction_register_jump_back_label( io_generated_code, &l_loop_label_tracker );

      /* update A pointer */
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_R, l_gp_reg_mapping.gp_reg_help_4, l_gp_reg_mapping.gp_reg_help_3, 0,
                                            l_gp_reg_mapping.gp_reg_a );

      /* update B pointer */
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_R, l_gp_reg_mapping.gp_reg_help_5, l_gp_reg_mapping.gp_reg_help_3, 0,
                                            l_gp_reg_mapping.gp_reg_b );
    }

    /* reset A pointer */
    /* TODO (MMLA): hardcoded MMLA fix */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a,
                                                   (unsigned long long)l_xgemm_desc_opa->m*l_micro_kernel_config.datatype_size_in*a_vnni_factor );

    /* advance B pointer */
    if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_b,
                                                     (unsigned long long)l_n_blocking * l_micro_kernel_config.datatype_size_in );
    } else {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_b,
                                                     (unsigned long long)l_n_blocking * l_xgemm_desc_opa->ldb * l_micro_kernel_config.datatype_size_in );
    }

    if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0 ) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STR_R, l_gp_reg_mapping.gp_reg_help_4, l_gp_reg_mapping.gp_reg_help_3, 0,
                                            l_gp_reg_mapping.gp_reg_a );

      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STR_R, l_gp_reg_mapping.gp_reg_help_5, l_gp_reg_mapping.gp_reg_help_3, 0,
                                            l_gp_reg_mapping.gp_reg_b );

      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                     l_gp_reg_mapping.gp_reg_help_3, l_gp_reg_mapping.gp_reg_help_3, 8, 0 );
      /* close BRGEMM loop */
      libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_reduce_loop, 1 );

      /* reset A and B */
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                     l_gp_reg_mapping.gp_reg_help_4, l_gp_reg_mapping.gp_reg_a, 0, 0 );
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                     l_gp_reg_mapping.gp_reg_help_5, l_gp_reg_mapping.gp_reg_b, 0, 0 );
    }

    /* close N loop */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker,
                                                l_gp_reg_mapping.gp_reg_nloop, l_n_blocking );
  }


  /* cleaning up the stack memory for the transpose */
  if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) {
    if ( !gemm_stack_frame_is_set ) {
      /* removing the extra offset applied to RSP to 64-byte boundary */
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_X29, LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );
    }
  }

  /* In this case we vnni-format C from scratch  */
  if (l_micro_kernel_config.vnni_format_C > 0) {
    l_xgemm_desc_opa->ldc = l_ldc_saved;
    libxsmm_generator_gemm_vnni_store_C_from_scratch_aarch64( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc_opa);
  }
  libxsmm_generator_gemm_destroy_stack_frame_aarch64( io_generated_code );

  /* close asm */
  libxsmm_aarch64_instruction_close_stream( io_generated_code, 0xe0f );
}

