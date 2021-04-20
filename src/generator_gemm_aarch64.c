/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
*               Intel Corporation - All rights reserved                       *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Breuer (Univ. Jena), Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include "generator_aarch64_instructions.h"
#include "generator_common_aarch64.h"
#include "generator_gemm_common_aarch64.h"
#include "generator_gemm_aarch64.h"
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
                                                       (i_micro_kernel_config->datatype_size_in == 4) ?  LIBXSMM_AARCH64_ASIMD_STRUCTTYPE_4S : LIBXSMM_AARCH64_ASIMD_STRUCTTYPE_2D );
      } else {
        libxsmm_aarch64_instruction_asimd_struct_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R_R_POST,
                                                       i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_2, l_n,
                                                       (i_micro_kernel_config->datatype_size_in == 4) ?  LIBXSMM_AARCH64_ASIMD_STRUCTTYPE_4S : LIBXSMM_AARCH64_ASIMD_STRUCTTYPE_2D );
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
            libxsmm_aarch64_instruction_asimd_struct_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R_R_POST,
                                                           i_gp_reg_mapping->gp_reg_b, LIBXSMM_AARCH64_GP_REG_XZR, l_n,
                                                           (i_micro_kernel_config->datatype_size_in == 4) ?  LIBXSMM_AARCH64_ASIMD_STRUCTTYPE_4S : LIBXSMM_AARCH64_ASIMD_STRUCTTYPE_2D );
          } else {
            libxsmm_aarch64_instruction_asimd_struct_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R_R_POST,
                                                           i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_2, l_n,
                                                           (i_micro_kernel_config->datatype_size_in == 4) ?  LIBXSMM_AARCH64_ASIMD_STRUCTTYPE_4S : LIBXSMM_AARCH64_ASIMD_STRUCTTYPE_2D );
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
            l_a_reg = (l_m_total%2 == 0) ? 6 : 7;
            libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                    i_gp_reg_mapping->gp_reg_a, LIBXSMM_AARCH64_GP_REG_UNDEF, l_m_instr_offset[l_m_total],
                                                    l_a_reg, l_m_instr_width[l_m_total] );
            l_m_total++;
            if ( l_m_total_blocks > 1 ) {
              l_a_reg = (l_m_total%2 == 0) ? 6 : 7;
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
        libxsmm_aarch64_instruction_asimd_struct_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R_R_POST,
                                                       i_gp_reg_mapping->gp_reg_b, LIBXSMM_AARCH64_GP_REG_XZR, 0,
                                                       (i_micro_kernel_config->datatype_size_in == 4) ?  LIBXSMM_AARCH64_ASIMD_STRUCTTYPE_4S : LIBXSMM_AARCH64_ASIMD_STRUCTTYPE_2D );
      } else {
        libxsmm_aarch64_instruction_asimd_struct_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R_R_POST,
                                                       i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_2, 0,
                                                       (i_micro_kernel_config->datatype_size_in == 4) ?  LIBXSMM_AARCH64_ASIMD_STRUCTTYPE_4S : LIBXSMM_AARCH64_ASIMD_STRUCTTYPE_2D );
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
    if( i_n_blocking == 1 ) {
      l_b_next_k = 1;
      l_b_next_k_inst = LIBXSMM_AARCH64_INSTR_GP_META_ADD;
    }
    else{
      l_b_next_k = ( (i_n_blocking - 1) * i_xgemm_desc->ldb - 1);
      l_b_next_k_inst = LIBXSMM_AARCH64_INSTR_GP_META_SUB;
    }
  }
  else{
    l_b_next_k = i_xgemm_desc->ldb - (i_n_blocking - 1);
    l_b_next_k_inst = LIBXSMM_AARCH64_INSTR_GP_META_ADD;
  }
  l_b_next_k *= i_micro_kernel_config->datatype_size_in;

  /* start register of accumulator */
  l_vec_reg_acc_start = i_micro_kernel_config->vector_reg_count - (i_n_blocking * l_m_total_blocks);

  /* full vector loads on a */
  for( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
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
  if( l_m_blocks[1] > 0) {
    libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                          (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF :
                                                                                           LIBXSMM_AARCH64_INSTR_SVE_LD1D_I_OFF,
                                          i_gp_reg_mapping->gp_reg_a,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF,
                                          0,
                                          1 + l_m_total_blocks * l_n + l_m_blocks[0],
                                          LIBXSMM_AARCH64_SVE_REG_P1 );
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                   i_gp_reg_mapping->gp_reg_a,
                                                   i_gp_reg_mapping->gp_reg_a,
                                                   l_remainder_size * i_micro_kernel_config->datatype_size_in,
                                                   0 );
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
    if( l_n != i_n_blocking - 1 ) {
      /* move on to next entry of B */
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                     i_gp_reg_mapping->gp_reg_b,
                                                     i_gp_reg_mapping->gp_reg_b,
                                                     l_b_stride,
                                                     0 );
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
                                               LIBXSMM_AARCH64_INSTR_SVE_FMLA_V,
                                               1 + l_m,
                                               0,
                                               -1,
                                               l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n),
                                               LIBXSMM_AARCH64_SVE_REG_P0,
                                               (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_SVE_TYPE_S : LIBXSMM_AARCH64_SVE_TYPE_D );
    }
    if( l_m_blocks[1] > 0 ) {
      libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                               LIBXSMM_AARCH64_INSTR_SVE_FMLA_V,
                                               1 + l_m_blocks[0],
                                               0,
                                               -1,
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
  void (*l_generator_microkernel)( libxsmm_generated_code*, const libxsmm_gp_reg_mapping*, const libxsmm_micro_kernel_config*, const libxsmm_gemm_descriptor*,
                                   const unsigned int, const unsigned int );

  /* select micro kernel based on aarch64 variant */
  if ( io_generated_code->arch == LIBXSMM_AARCH64_V81 ) {
    l_generator_microkernel = libxsmm_generator_gemm_aarch64_microkernel_asimd_neoverse;
  } else if ( io_generated_code->arch == LIBXSMM_AARCH64_A64FX ) {
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

    for ( l_k = 0; l_k < l_k_blocking; l_k++) {
      l_generator_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, i_m_blocking, i_n_blocking);
    }

    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_kloop, l_k_blocking );
  } else {
    /* 2. we want to fully unroll below the threshold */
    if ((unsigned int)i_xgemm_desc->k <= l_k_threshold) {
      unsigned int l_k;

      libxsmm_generator_gemm_aarch64_setup_k_strides(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
                                                   i_xgemm_desc, i_m_blocking, i_n_blocking);

      for ( l_k = 0; l_k < (unsigned int)i_xgemm_desc->k; l_k++) {
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

        for ( l_k = 0; l_k < l_k_blocking; l_k++) {
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
                                                 (unsigned long long)((unsigned long long)i_xgemm_desc->k * i_xgemm_desc->lda * i_micro_kernel_config->datatype_size_in) );

  /* reset B pointer */
  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                   (unsigned long long)((unsigned long long)i_xgemm_desc->ldb * i_xgemm_desc->k * i_micro_kernel_config->datatype_size_in) );
  } else {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                   (unsigned long long)((unsigned long long)i_xgemm_desc->k * i_micro_kernel_config->datatype_size_in) );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_kernel( libxsmm_generated_code*        io_generated_code,
                                            const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;
  unsigned int l_n_n[2] = {0,0};       /* blocking sizes for blocks */
  unsigned int l_n_N[2] = {0,0};       /* size of blocks */
  unsigned int l_n_count = 0;          /* array counter for blocking arrays */
  unsigned int l_n_done = 0;           /* progress tracker */
  unsigned int l_n_done_old = 0;

  /* define gp register mapping */
  libxsmm_reset_aarch64_gp_reg_mapping( &l_gp_reg_mapping );

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
  l_gp_reg_mapping.gp_reg_help_5 = LIBXSMM_AARCH64_GP_REG_X29;      /* for b ptr updates in BRGEMM */
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

  if( io_generated_code->arch == LIBXSMM_AARCH64_A64FX ) {
    int l_nnz_bits = i_xgemm_desc->m%l_micro_kernel_config.vector_length;
    l_nnz_bits *= l_micro_kernel_config.datatype_size_out;
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                  LIBXSMM_AARCH64_SVE_REG_P0,
                                                  -1,
                                                  l_gp_reg_mapping.gp_reg_help_0 );

    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                  LIBXSMM_AARCH64_SVE_REG_P1,
                                                  l_nnz_bits,
                                                  l_gp_reg_mapping.gp_reg_help_0 );
  }

  /* apply n_blocking */
  while (l_n_done != (unsigned int)i_xgemm_desc->n) {
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
    l_m_blocking = libxsmm_generator_gemm_aarch64_get_initial_m_blocking( &l_micro_kernel_config, i_xgemm_desc, io_generated_code->arch );

    /* apply m_blocking */
    while (l_m_done != (unsigned int)i_xgemm_desc->m) {
      if ( l_m_blocking == 0 ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
        return;
      }
      l_m_done_old = l_m_done;
      LIBXSMM_ASSERT(0 != l_m_blocking);
      /* coverity[divide_by_zero] */
      l_m_done = l_m_done + (((i_xgemm_desc->m - l_m_done_old) / l_m_blocking) * l_m_blocking);

      if ( (l_m_done != l_m_done_old) && (l_m_done > 0) ) {
        /* open M loop */
        libxsmm_generator_loop_header_aarch64( io_generated_code, &l_loop_label_tracker,
                                               l_gp_reg_mapping.gp_reg_mloop, l_m_done - l_m_done_old );
        /* load block of C */
        if ( io_generated_code->arch == LIBXSMM_AARCH64_V81 ) {
          libxsmm_generator_load_2dregblock_aarch64_asimd( io_generated_code, l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_0,
                                                           l_micro_kernel_config.vector_length, l_micro_kernel_config.vector_reg_count, l_m_blocking, l_n_blocking,
                                                           i_xgemm_desc->ldc * l_micro_kernel_config.datatype_size_out,
                                                           (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags) );
        } else if ( io_generated_code->arch == LIBXSMM_AARCH64_A64FX ) {
          libxsmm_generator_load_2dregblock_aarch64_sve( io_generated_code, l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_0,
                                                         l_micro_kernel_config.vector_length, l_micro_kernel_config.vector_reg_count, l_m_blocking, l_n_blocking,
                                                         i_xgemm_desc->ldc * l_micro_kernel_config.datatype_size_out,
                                                         l_micro_kernel_config.datatype_size_out,
                                                         (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags) );
        } else {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
          return;
        }

        /* handle BRGEMM */
        if ( ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0) ||
             ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) >  0) ||
             ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) >  0)    ) {
          /* we need to load the real address */
          if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0 ) {
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
          if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0 ) {
            libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_R, l_gp_reg_mapping.gp_reg_help_4, l_gp_reg_mapping.gp_reg_help_3, 0,
                                                  l_gp_reg_mapping.gp_reg_a );
            libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_R, l_gp_reg_mapping.gp_reg_help_5, l_gp_reg_mapping.gp_reg_help_3, 0,
                                                  l_gp_reg_mapping.gp_reg_b );
          }
          if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) > 0 ) {
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
                                              i_xgemm_desc, l_m_blocking, l_n_blocking );

        if ( ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0) ||
             ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) >  0) ||
             ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) >  0)    ) {
          /* increment forward counting BRGEMM count */
          if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0 ) {
            /* nothing to do */
          } else {
            libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                           l_gp_reg_mapping.gp_reg_help_3, l_gp_reg_mapping.gp_reg_help_3, 8, 0 );
          }
          if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0 ) {
            libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_gp_reg_mapping.gp_reg_help_0, (unsigned long long)i_xgemm_desc->c1 );
            libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_gp_reg_mapping.gp_reg_help_1, (unsigned long long)i_xgemm_desc->c2 );
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
          if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0 ) {
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
        if ( io_generated_code->arch == LIBXSMM_AARCH64_V81 ) {
          libxsmm_generator_store_2dregblock_aarch64_asimd( io_generated_code, l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_0,
                                                            l_micro_kernel_config.vector_length, l_micro_kernel_config.vector_reg_count, l_m_blocking, l_n_blocking,
                                                            i_xgemm_desc->ldc * l_micro_kernel_config.datatype_size_out );
        } else if ( io_generated_code->arch == LIBXSMM_AARCH64_A64FX ) {
          libxsmm_generator_store_2dregblock_aarch64_sve( io_generated_code, l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_0,
                                                          l_micro_kernel_config.vector_length, l_micro_kernel_config.vector_reg_count, l_m_blocking, l_n_blocking,
                                                          i_xgemm_desc->ldc * l_micro_kernel_config.datatype_size_out,
                                                          l_micro_kernel_config.datatype_size_out );
        } else {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
          return;
        }

        /* advance C pointer */
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_c,
                                                       l_m_blocking*l_micro_kernel_config.datatype_size_out );

        /* advance A pointer */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0 ) {
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
                                                         l_m_blocking*l_micro_kernel_config.datatype_size_in );

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
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                         l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a,
                                                         l_m_blocking*l_micro_kernel_config.datatype_size_in );
        }

        /* close M loop */
        libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker,
                                               l_gp_reg_mapping.gp_reg_mloop, l_m_blocking );
      }

      /* switch to next smaller m_blocking */
      l_m_blocking = libxsmm_generator_gemm_aarch64_update_m_blocking( &l_micro_kernel_config, i_xgemm_desc, io_generated_code->arch, l_m_blocking );
    }
    /* reset C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_c,
                                                   ((unsigned long long)l_n_blocking * i_xgemm_desc->ldc * l_micro_kernel_config.datatype_size_out) -
                                                   ((unsigned long long)i_xgemm_desc->m * l_micro_kernel_config.datatype_size_out) );

    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0 ) {
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

    /* reser A pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a,
                                                   i_xgemm_desc->m*l_micro_kernel_config.datatype_size_in );

    /* advance B pointer */
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_b,
                                                     (unsigned long long)((unsigned long long)l_n_blocking * l_micro_kernel_config.datatype_size_in) );
    } else {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_b,
                                                     (unsigned long long)((unsigned long long)l_n_blocking * i_xgemm_desc->ldb * l_micro_kernel_config.datatype_size_in) );
    }

    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0 ) {
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

  /* close asm */
  libxsmm_aarch64_instruction_close_stream( io_generated_code, 0xe0f );
}

