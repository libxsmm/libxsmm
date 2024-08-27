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

#include "generator_packed_spgemm_bcsc_bsparse_aarch64.h"
#include "generator_aarch64_instructions.h"
#include "generator_common_aarch64.h"
#include "generator_mateltwise_aarch64.h"
#include "generator_gemm_common_aarch64.h"
#include "libxsmm_main.h"


LIBXSMM_API_INTERN
void libxsmm_spgemm_max_mn_blocking_factors_aarch64(libxsmm_generated_code* io_generated_code, unsigned int i_use_mmla, unsigned int i_bn, unsigned int *o_max_m_bf, unsigned int *o_max_n_bf) {
  unsigned int l_available_vregs = 32;
  unsigned int l_n_max_unroll = l_available_vregs - 4;
  unsigned int l_m_max_unroll = 0;
  unsigned int n_blocks = (i_use_mmla == 0) ? i_bn : (i_bn+1)/2;
  LIBXSMM_UNUSED(io_generated_code);
  while (n_blocks % l_n_max_unroll != 0) {
    l_n_max_unroll--;
  }
  if (i_use_mmla > 0) {
    if (l_n_max_unroll >= 8) {
      while (n_blocks % l_n_max_unroll != 0 || l_n_max_unroll >= 8) {
        l_n_max_unroll--;
      }
    }
  }
  if (i_use_mmla == 0) {
    if (l_n_max_unroll > 8) {
      while (n_blocks % l_n_max_unroll != 0 || l_n_max_unroll > 8) {
        l_n_max_unroll--;
      }
    }
  }
  l_m_max_unroll = l_available_vregs;
  while ((l_m_max_unroll * l_n_max_unroll + l_m_max_unroll + 1) > l_available_vregs) {
    l_m_max_unroll--;
  }
  if (i_use_mmla > 0) {
    l_m_max_unroll = LIBXSMM_MIN(4, l_m_max_unroll);
  }
  *o_max_m_bf = l_m_max_unroll;
  *o_max_n_bf = l_n_max_unroll;
}

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_bcsc_bsparse_aarch64( libxsmm_generated_code*         io_generated_code,
                                                           const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                           const unsigned int              i_packed_width,
                                                           const unsigned int              i_bk,
                                                           const unsigned int              i_bn ) {
  unsigned int l_simd_packed_remainder = 0;
  unsigned int l_simd_packed_iters = 0;
  unsigned int l_simd_packed_iters_full = 0;
  unsigned int l_simd_packed_width = 4;
  unsigned int l_packed_done = 0;
  unsigned int l_packed_count = 0;
  unsigned int l_packed_reg_block[2] = {0,0};
  unsigned int l_packed_reg_range[2] = {0,0};
  unsigned int l_is_mmla_kernel = 0;
  unsigned int l_max_m_blocking = 4;
  unsigned int l_max_n_blocking = 0;
  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;
  unsigned int l_column_idx_gpr = LIBXSMM_AARCH64_GP_REG_X3;
  unsigned int l_row_idx_gpr    = LIBXSMM_AARCH64_GP_REG_X4;
  unsigned int l_cur_column_gpr = LIBXSMM_AARCH64_GP_REG_W13;
  unsigned int l_next_column_gpr= LIBXSMM_AARCH64_GP_REG_W14;
  unsigned int l_dynamic_n_gpr  = LIBXSMM_AARCH64_GP_REG_X15;

  libxsmm_jump_label_tracker l_jump_label_tracker;
  libxsmm_reset_jump_label_tracker(&l_jump_label_tracker);

  /* select simd packing width and accumulator blocking */
  if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
    if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) > 0) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_A );
      return;
    }
    if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_B );
      return;
    }
    if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
      return;
    }
    if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
      return;
    }
    if ( io_generated_code->arch >= LIBXSMM_AARCH64_SVE128 ) {
      if ( io_generated_code->arch < LIBXSMM_AARCH64_SVE256 ) {
        l_simd_packed_width = 4;
      } else if ( io_generated_code->arch < LIBXSMM_AARCH64_SVE512 ) {
        l_simd_packed_width = 8;
        l_is_mmla_kernel = 0;
      } else {
        l_simd_packed_width = 16;
      }
    } else { /* asimd */
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
    if (io_generated_code->arch == LIBXSMM_AARCH64_NEOV1) {
      char l_use_bfdot = (char)libxsmm_cpuid_arm_use_bfdot();
      if (l_use_bfdot == 0) {
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) > 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_A );
          return;
        }
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_B );
          return;
        }
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
          return;
        }
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_B) == 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
          return;
        }
        l_simd_packed_width = 4;
        l_is_mmla_kernel = 1;
        if (i_bk % 4 != 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BCSC_BLOCK_SIZE );
          return;
        }
      } else {
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) > 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_A );
          return;
        }
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_B );
          return;
        }
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
          return;
        }
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
          return;
        }
        l_simd_packed_width = 8;
        l_is_mmla_kernel = 0;
        if (i_bk % 2 != 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BCSC_BLOCK_SIZE );
          return;
        }
      }
      /* TODO: Check for A in VNNI and C in VNNI */
      /* TODO: Check provided bk and bn in BCSC format */
    } else if (io_generated_code->arch == LIBXSMM_AARCH64_NEOV2 ) {
          char l_use_bfdot = (char)libxsmm_cpuid_arm_use_bfdot();
          if (l_use_bfdot == 0) {
            if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) > 0) {
              LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_A );
              return;
            }
            if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0) {
              LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_B );
              return;
            }
            if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0) {
              LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
              return;
            }
            if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_B) == 0) {
              LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
              return;
            }
            l_simd_packed_width = 4;
            l_is_mmla_kernel = 1;
            if (i_bk % 4 != 0) {
              LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BCSC_BLOCK_SIZE );
              return;
            }
          } else {
            if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) > 0) {
              LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_A );
              return;
            }
            if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0) {
              LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_B );
              return;
            }
            if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0) {
              LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
              return;
            }
            if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0) {
              LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
              return;
            }
            l_simd_packed_width = 4;
            l_is_mmla_kernel = 0;
            if (i_bk % 2 != 0) {
              LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BCSC_BLOCK_SIZE );
              return;
            }
        }
      } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
      }
  } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
    if (io_generated_code->arch == LIBXSMM_AARCH64_NEOV1) {
      char l_use_bfdot = (char)libxsmm_cpuid_arm_use_bfdot();
      if (l_use_bfdot == 0) {
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) > 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_A );
          return;
        }
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_B );
          return;
        }
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
          return;
        }
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_B) == 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
          return;
        }
        l_simd_packed_width = 4;
        l_is_mmla_kernel = 1;
        if (i_bk % 8 != 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BCSC_BLOCK_SIZE );
          return;
        }
      } else {
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) > 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_A );
          return;
        }
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_B );
          return;
        }
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
          return;
        }
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
          return;
        }
        l_simd_packed_width = 8;
        l_is_mmla_kernel = 0;
        if (i_bk % 4 != 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BCSC_BLOCK_SIZE );
          return;
        }
      }
      /* TODO: Check for A in VNNI and C in VNNI */
      /* TODO: Check provided bk and bn in BCSC format */
    } else if (io_generated_code->arch == LIBXSMM_AARCH64_NEOV2 ) {
      char l_use_bfdot = (char)libxsmm_cpuid_arm_use_bfdot();
      if (l_use_bfdot == 0) {
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) > 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_A );
          return;
        }
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_B );
          return;
        }
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
          return;
        }
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_B) == 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
          return;
        }
        l_simd_packed_width = 4;
        l_is_mmla_kernel = 1;
        if (i_bk % 8 != 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BCSC_BLOCK_SIZE );
          return;
        }
      } else {
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) > 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_A );
          return;
        }
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_B );
          return;
        }
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
          return;
        }
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
          return;
        }
        l_simd_packed_width = 4;
        l_is_mmla_kernel = 0;
        if (i_bk % 4 != 0) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BCSC_BLOCK_SIZE );
          return;
        }
      }
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }

  l_simd_packed_remainder = i_packed_width % l_simd_packed_width;
  l_simd_packed_iters_full = i_packed_width/l_simd_packed_width;
  l_simd_packed_iters = ( l_simd_packed_remainder > 0 ) ? l_simd_packed_iters_full+1 : l_simd_packed_iters_full;

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
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_AARCH64_GP_REG_W8;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_AARCH64_GP_REG_X9;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_AARCH64_GP_REG_X10;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_AARCH64_GP_REG_X11;
  l_gp_reg_mapping.gp_reg_help_3 = LIBXSMM_AARCH64_GP_REG_X12;
  l_gp_reg_mapping.gp_reg_help_4 = l_row_idx_gpr;
  l_gp_reg_mapping.gp_reg_help_5 = l_cur_column_gpr;
  l_gp_reg_mapping.gp_reg_help_6 = l_next_column_gpr;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config_aarch64( &l_micro_kernel_config, io_generated_code->arch, i_xgemm_desc );

  /* open asm */
  libxsmm_aarch64_instruction_open_stream( io_generated_code, 0xf );

  /* ensuring compatibility with X86 AMX */
  if ( !( (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & i_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & i_xgemm_desc->flags) == 0)) ||
          (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & i_xgemm_desc->flags) != 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & i_xgemm_desc->flags) != 0))    ) ) {
    /* close asm */
    libxsmm_aarch64_instruction_close_stream( io_generated_code, 0xf );
    return;
  }

  /* implementing load from struct */
  if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI & i_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI) ) {
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
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 72, l_column_idx_gpr);
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 80, l_row_idx_gpr );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 88, l_dynamic_n_gpr );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                     l_dynamic_n_gpr, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, l_dynamic_n_gpr );

    /* C pointer */
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 96, l_gp_reg_mapping.gp_reg_c );
    if ( i_xgemm_desc->prefetch != LIBXSMM_GEMM_PREFETCH_NONE ) {
      /* A prefetch pointer */
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 56, l_gp_reg_mapping.gp_reg_a_prefetch );
      /* B preftech pointer */
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 88, l_gp_reg_mapping.gp_reg_b_prefetch );
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ILLEGAL_ABI );
    return;
  }

  /* set P0 in case of SVE */
  if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT) ) {
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                  LIBXSMM_AARCH64_SVE_REG_P0,
                                                  -1,
                                                  l_gp_reg_mapping.gp_reg_help_0 );
  }

  /* Set blocking factor decisions...  */
  libxsmm_spgemm_max_mn_blocking_factors_aarch64(io_generated_code, l_is_mmla_kernel, i_bn, &l_max_m_blocking, &l_max_n_blocking);
  if (l_simd_packed_iters <= l_max_m_blocking) {
    l_packed_reg_range[0] = l_simd_packed_iters;
    l_packed_reg_block[0] = l_simd_packed_iters;
    l_packed_reg_range[1] = 0;
    l_packed_reg_block[1] = 0;
  } else {
    l_packed_reg_range[0] = l_simd_packed_iters - l_simd_packed_iters % l_max_m_blocking;
    l_packed_reg_block[0] = l_max_m_blocking;
    l_packed_reg_range[1] = l_simd_packed_iters % l_max_m_blocking;
    l_packed_reg_block[1] = l_simd_packed_iters % l_max_m_blocking;
    if (l_simd_packed_remainder > 0) {
      if ( l_packed_reg_range[1] == 0) {
        if (l_packed_reg_range[0] != l_packed_reg_block[0]) {
          l_packed_reg_range[0] = l_packed_reg_range[0] - l_packed_reg_block[0];
          l_packed_reg_block[0] = l_max_m_blocking;
          l_packed_reg_range[1] = l_max_m_blocking;
          l_packed_reg_block[1] = l_max_m_blocking;
        }
      }
    }
  }

  if (((LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ))) && ((io_generated_code->arch == LIBXSMM_AARCH64_NEOV1) || (io_generated_code->arch == LIBXSMM_AARCH64_NEOV2))) {
    if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_I32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
      int l_nnz_bits2 = 16;
      if (l_is_mmla_kernel > 0) {
        int l_nnz_4m_bits = 4 * l_micro_kernel_config.datatype_size_out;
        libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                      LIBXSMM_AARCH64_SVE_REG_P1,
                                                      l_nnz_4m_bits,
                                                      l_gp_reg_mapping.gp_reg_help_0 );
        if (l_simd_packed_remainder != 0) {
          l_nnz_4m_bits = (l_simd_packed_remainder % 4) * l_micro_kernel_config.datatype_size_out;
          libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                        LIBXSMM_AARCH64_SVE_REG_P2,
                                                        l_nnz_4m_bits,
                                                        l_gp_reg_mapping.gp_reg_help_0 );
        }
      } else {
        libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                      LIBXSMM_AARCH64_SVE_REG_P2,
                                                      l_nnz_bits2,
                                                      l_gp_reg_mapping.gp_reg_help_0 );
      }
      /* mask for M remainder  */
      if ( l_simd_packed_remainder != 0 ) {
        if (l_is_mmla_kernel > 0) {
          int m_nnz_bits = (l_simd_packed_remainder % 2) * 8;
          if (m_nnz_bits > 0) {
            libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                          LIBXSMM_AARCH64_SVE_REG_P4,
                                                          m_nnz_bits,
                                                          l_gp_reg_mapping.gp_reg_help_0 );
          }
          m_nnz_bits = l_simd_packed_remainder * 8;
          libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                        LIBXSMM_AARCH64_SVE_REG_P5,
                                                        m_nnz_bits,
                                                        l_gp_reg_mapping.gp_reg_help_0 );
        } else {
          int m_nnz_bits = l_simd_packed_remainder * l_micro_kernel_config.datatype_size_out;
          if (m_nnz_bits > 0) {
            libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                          LIBXSMM_AARCH64_SVE_REG_P4,
                                                          m_nnz_bits,
                                                          l_gp_reg_mapping.gp_reg_help_0 );
          }
          m_nnz_bits = l_simd_packed_remainder * 4;
          libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                        LIBXSMM_AARCH64_SVE_REG_P5,
                                                        m_nnz_bits,
                                                        l_gp_reg_mapping.gp_reg_help_0 );
        }
      }
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  } else if ((LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && ((io_generated_code->arch == LIBXSMM_AARCH64_NEOV1) || (io_generated_code->arch == LIBXSMM_AARCH64_NEOV2))) {
    /* mask for M remainder  */
    if ( l_simd_packed_remainder != 0 ) {
      int m_nnz_bits = l_simd_packed_remainder * 4;
      if (m_nnz_bits > 0) {
        libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                      LIBXSMM_AARCH64_SVE_REG_P4,
                                                      m_nnz_bits,
                                                      l_gp_reg_mapping.gp_reg_help_0 );
      }
      m_nnz_bits = l_simd_packed_remainder * 4;
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                    LIBXSMM_AARCH64_SVE_REG_P5,
                                                    m_nnz_bits,
                                                    l_gp_reg_mapping.gp_reg_help_0 );

    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }

  /* m loop */
  libxsmm_generator_loop_header_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_mloop, i_xgemm_desc->m );

  /* loop over packed blocks */
  while ( l_packed_done != l_simd_packed_iters ) {
    unsigned int l_packed_blocking;
    unsigned int l_packed_remainder = 0;
    if (l_packed_count > 1) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
      return;
    }
    l_packed_blocking = l_packed_reg_block[l_packed_count];

    if ( (l_simd_packed_remainder != 0) && (l_packed_count == 0) ) {
      if ( l_packed_reg_block[1] > 0 ) {
        l_packed_remainder = 0;
      } else {
        l_packed_remainder = l_simd_packed_remainder;
      }
    } else if (l_simd_packed_remainder != 0) {
      l_packed_remainder = l_simd_packed_remainder;
    }

    /* n loop */
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, l_dynamic_n_gpr, l_gp_reg_mapping.gp_reg_nloop, 0, 0 );
    libxsmm_aarch64_instruction_register_jump_back_label( io_generated_code, &l_loop_label_tracker );
    /* Load column index */
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                          l_column_idx_gpr, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, l_cur_column_gpr );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                          l_column_idx_gpr, LIBXSMM_AARCH64_GP_REG_UNDEF, 4, l_next_column_gpr );
    if (l_is_mmla_kernel > 0) {
      libxsmm_generator_packed_spgemm_bcsc_bsparse_aarch64_kloop_mmla_sve( io_generated_code,
                                                                           &l_loop_label_tracker,
                                                                           &l_jump_label_tracker,
                                                                           &l_gp_reg_mapping,
                                                                           &l_micro_kernel_config,
                                                                           i_xgemm_desc,
                                                                           l_packed_done,
                                                                           l_packed_reg_range[l_packed_count],
                                                                           l_packed_blocking,
                                                                           l_packed_remainder,
                                                                           i_packed_width,
                                                                           l_simd_packed_width,
                                                                           i_bk,
                                                                           i_bn);
    } else if (l_is_mmla_kernel == 0)  {
      libxsmm_generator_packed_spgemm_bcsc_bsparse_aarch64_kloop_bfdot_sve ( io_generated_code,
                                                                             &l_loop_label_tracker,
                                                                             &l_jump_label_tracker,
                                                                             &l_gp_reg_mapping,
                                                                             &l_micro_kernel_config,
                                                                             i_xgemm_desc,
                                                                             l_packed_done,
                                                                             l_packed_reg_range[l_packed_count],
                                                                             l_packed_blocking,
                                                                             l_packed_remainder,
                                                                             i_packed_width,
                                                                             l_simd_packed_width,
                                                                             i_bk,
                                                                             i_bn );

    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }

    /* close n loop */
    /* advance C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_c,
                                                   (long long) i_bn * i_packed_width * l_micro_kernel_config.datatype_size_out );
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                   l_column_idx_gpr, l_column_idx_gpr, 4, 0 );
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I,
                                                   l_gp_reg_mapping.gp_reg_nloop, l_gp_reg_mapping.gp_reg_nloop, 1, 0 );
    libxsmm_aarch64_instruction_cond_jump_back_to_label( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_CBNZ,
                                                         l_gp_reg_mapping.gp_reg_nloop, &l_loop_label_tracker );

    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_gp_reg_mapping.gp_reg_help_2, i_bn * i_packed_width * l_micro_kernel_config.datatype_size_out);
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MUL, l_gp_reg_mapping.gp_reg_help_2, l_dynamic_n_gpr, l_gp_reg_mapping.gp_reg_help_2, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR, l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_c, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );

    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_gp_reg_mapping.gp_reg_help_2, 4);
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MUL, l_gp_reg_mapping.gp_reg_help_2, l_dynamic_n_gpr, l_gp_reg_mapping.gp_reg_help_2, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR, l_column_idx_gpr, l_gp_reg_mapping.gp_reg_help_2, l_column_idx_gpr, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    /* advance M */
    l_packed_done += l_packed_reg_range[l_packed_count];
    l_packed_count++;
  }

  /* advance C pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_c,
                                                 (long long)l_micro_kernel_config.datatype_size_out*i_packed_width*i_xgemm_desc->ldc );

  /* advance A pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a,
                                                 (long long)l_micro_kernel_config.datatype_size_in*i_packed_width*i_xgemm_desc->lda );

  /* close m loop */
  libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_mloop, 1 );

  /* close asm */
  libxsmm_aarch64_instruction_close_stream( io_generated_code, 0xf );
}

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_bcsc_bsparse_aarch64_kloop_mmla_sve( libxsmm_generated_code*            io_generated_code,
                                                                          libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                          libxsmm_jump_label_tracker*        i_jump_label_tracker,
                                                                          const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                          const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                          const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                          const unsigned int                 i_packed_processed,
                                                                          const unsigned int                 i_packed_range,
                                                                          const unsigned int                 i_packed_blocking,
                                                                          const unsigned int                 i_packed_remainder,
                                                                          const unsigned int                 i_packed_width,
                                                                          const unsigned int                 i_simd_packed_width,
                                                                          const unsigned int                 i_bk,
                                                                          const unsigned int                 i_bn ) {
  unsigned int l_n = 0, l_n_blocks = 0, l_n_block_id = 0;
  unsigned int l_p = 0;
  unsigned int l_max_reg_block = 0;
  unsigned int l_n_blocking = 0;
  unsigned int l_vec_reg_tmp[2];
  unsigned int l_n_advancements = 0;
  unsigned int l_i8i32_kernel = (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) && LIBXSMM_DATATYPE_I32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ? 1 : 0;
  unsigned int l_bf16_kernel = (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) && LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ? 1 : 0;
  unsigned int l_vnni_block_size = (l_i8i32_kernel > 0) ? 8 : 4;
  unsigned int l_beta_0 = (0 != (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) ? 1 : 0;
  unsigned int l_is_s8u8s32_kernel = ((l_i8i32_kernel > 0) && ((LIBXSMM_GEMM_FLAG_A_UNSIGNED & i_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_B_UNSIGNED & i_xgemm_desc->flags) > 0) ) ? 1 : 0;

  /* Auxiliary GPRs  */
  unsigned int l_gp_reg_scratch = i_gp_reg_mapping->gp_reg_help_2;
  unsigned int l_gp_reg_scratch_32bit = i_gp_reg_mapping->gp_reg_help_2 - 32;
  unsigned int l_tmp_a_gp_reg = LIBXSMM_AARCH64_GP_REG_X8;
  unsigned int l_k_loop_fma_reg = LIBXSMM_AARCH64_GP_REG_X17;
  unsigned int l_input_bf16_mask = LIBXSMM_AARCH64_SVE_REG_P5;
  unsigned int l_row_idx_gpr = i_gp_reg_mapping->gp_reg_help_4;
  unsigned int l_cur_column_gpr = i_gp_reg_mapping->gp_reg_help_5;
  unsigned int l_cur_column_gpr_64bit = l_cur_column_gpr + 32;
  unsigned int l_next_column_gpr = i_gp_reg_mapping->gp_reg_help_6;
  unsigned int EMPTY_BLOCK_COLUMN_LABEL = (i_packed_processed == 0) ? 0 : 1;
  unsigned int EMPTY_BLOCK_COLUMN_LABEL_BETA0 = 0;

  /* derive zip instructions and auxiliary sve types */
  unsigned int l_fma_iters = i_bk/l_vnni_block_size;
  unsigned int l_fma_i = 0;
  unsigned int l_assm_fma_iters = 1;
  unsigned int l_a_adjustments = 0;
  unsigned int l_b_adjustments = 0;

  if (l_fma_iters > 2) {
    l_assm_fma_iters = l_fma_iters;
    l_fma_iters = 2;
    while (l_assm_fma_iters % l_fma_iters != 0) {
      l_fma_iters--;
    }
    l_assm_fma_iters = l_assm_fma_iters/l_fma_iters;
  }

  LIBXSMM_ASSERT( i_packed_blocking > 0 );

  libxsmm_spgemm_max_mn_blocking_factors_aarch64(io_generated_code, 1, i_bn, &l_n, &l_n_blocking);
  l_max_reg_block = l_n_blocking * i_packed_blocking;
  l_n_blocks = ((i_bn+1)/2) / l_n_blocking;
  EMPTY_BLOCK_COLUMN_LABEL_BETA0 = (i_packed_processed == 0) ? 2 : 2 + l_n_blocks;

  /* temporary vector registers used to load values to before zipping */
  l_vec_reg_tmp[0] = l_max_reg_block+0;
  l_vec_reg_tmp[1] = l_max_reg_block+1;

  /* Adjust A and C pointers for already processed N/M  */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,  LIBXSMM_AARCH64_INSTR_GP_META_ADD,
    i_gp_reg_mapping->gp_reg_c, l_gp_reg_scratch, i_gp_reg_mapping->gp_reg_c, (1ull * i_packed_processed * i_simd_packed_width) * i_micro_kernel_config->datatype_size_out );

  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
      i_gp_reg_mapping->gp_reg_a, l_gp_reg_scratch, i_gp_reg_mapping->gp_reg_a, (long long)i_packed_processed * i_simd_packed_width * l_vnni_block_size * i_micro_kernel_config->datatype_size_in );

  if (l_beta_0 == 0) {
    /* Check if empty B column and beta == 1 and jump at the end of the kernel */
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR, l_next_column_gpr, l_cur_column_gpr, l_gp_reg_scratch_32bit, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    libxsmm_aarch64_instruction_cond_jump_to_label( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_CBZ, l_gp_reg_scratch_32bit, EMPTY_BLOCK_COLUMN_LABEL, i_jump_label_tracker );
  }

  /* packed loop */
  if ( i_packed_range/i_packed_blocking > 1 ) {
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_help_3, i_packed_range/i_packed_blocking );
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, l_cur_column_gpr_64bit, LIBXSMM_AARCH64_GP_REG_X16, 0, 0 );
  }

  for (l_n_block_id = 0; l_n_block_id < l_n_blocks; l_n_block_id++) {
    unsigned int is_last_n_block = ((l_n_block_id + 1) >= l_n_blocks) ? 1 : 0;
    if (l_n_blocks > 1) {
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, l_cur_column_gpr_64bit, LIBXSMM_AARCH64_GP_REG_X16, 0, 0 );
    }
    /* load C accumulator */
    if (l_beta_0 > 0) {
      for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
        for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
          unsigned int l_reg0 = (l_n*i_packed_blocking) + l_p;
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V, l_reg0, l_reg0, 0, l_reg0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
        }
      }
    } else {
      l_n_advancements = 0;
      for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
        l_n_advancements++;
        for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
          unsigned int l_reg0 = (l_n*i_packed_blocking) + l_p;
          unsigned int l_mask_use = 0;
          unsigned int l_c_load_instr = (l_bf16_kernel > 0) ? LIBXSMM_AARCH64_INSTR_SVE_LD1H_I_OFF : LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF;
          l_mask_use = LIBXSMM_AARCH64_SVE_REG_P1;
          if ((i_packed_remainder > 0) && (l_p == i_packed_blocking - 1)) {
            l_mask_use = LIBXSMM_AARCH64_SVE_REG_P2;
          }
          /* Load n0 4m */
          libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                l_c_load_instr,
                                                i_gp_reg_mapping->gp_reg_c,
                                                LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                0,
                                                l_vec_reg_tmp[0],
                                                l_mask_use );
          if (l_bf16_kernel > 0) {
            libxsmm_generator_vcvt_bf16f32_aarch64_sve( io_generated_code, l_vec_reg_tmp[0], 0);
          }
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                         i_gp_reg_mapping->gp_reg_c,
                                                         l_gp_reg_scratch,
                                                         i_gp_reg_mapping->gp_reg_c,
                                                         i_packed_width * i_micro_kernel_config->datatype_size_out );

          /* Load n1 4m */
          if ((l_n == l_n_blocking-1) && ((i_bn % 2 != 0) && (is_last_n_block > 0))) {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V, l_vec_reg_tmp[1], l_vec_reg_tmp[1], 0, l_vec_reg_tmp[1], LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
          } else {
            libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                  l_c_load_instr,
                                                  i_gp_reg_mapping->gp_reg_c,
                                                  LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                  0,
                                                  l_vec_reg_tmp[1],
                                                  l_mask_use );
            if (l_bf16_kernel > 0) {
              libxsmm_generator_vcvt_bf16f32_aarch64_sve( io_generated_code, l_vec_reg_tmp[1], 0);
            }
          }

          /* Zip 64bit elements to get [2M][2n][2m] */
          libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_SVE_ZIP1_V,
                                                   l_vec_reg_tmp[0],
                                                   l_vec_reg_tmp[1],
                                                   0,
                                                   l_reg0,
                                                   LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                                   (l_is_s8u8s32_kernel == 0) ? LIBXSMM_AARCH64_SVE_TYPE_D : LIBXSMM_AARCH64_SVE_TYPE_S );

          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                         i_gp_reg_mapping->gp_reg_c,
                                                         l_gp_reg_scratch,
                                                         i_gp_reg_mapping->gp_reg_c,
                                                         ((long long)i_packed_width - i_simd_packed_width) * i_micro_kernel_config->datatype_size_out );
        }
        if ((i_packed_width * 2 - i_packed_blocking * i_simd_packed_width) > 0) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                         i_gp_reg_mapping->gp_reg_c,
                                                         l_gp_reg_scratch,
                                                         i_gp_reg_mapping->gp_reg_c,
                                                         ((long long)i_packed_width * 2 - i_packed_blocking * i_simd_packed_width) * i_micro_kernel_config->datatype_size_out );
        }
      }
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                     i_gp_reg_mapping->gp_reg_c,
                                                     l_gp_reg_scratch,
                                                     i_gp_reg_mapping->gp_reg_c,
                                                     (i_packed_width * 2 * l_n_advancements) * i_micro_kernel_config->datatype_size_out );
    }

    if (l_beta_0 > 0) {
      /* Check if empty B column and beta == 1 and jump at the C store part of the kernel */
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR, l_next_column_gpr, l_cur_column_gpr, l_gp_reg_scratch_32bit, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
      libxsmm_aarch64_instruction_cond_jump_to_label( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_CBZ, l_gp_reg_scratch_32bit, EMPTY_BLOCK_COLUMN_LABEL_BETA0, i_jump_label_tracker );
    }

    /* k loop header */
    libxsmm_aarch64_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, l_row_idx_gpr, l_cur_column_gpr_64bit, l_gp_reg_scratch, 2, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, l_gp_reg_scratch, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_gp_reg_mapping->gp_reg_kloop );

    /* Prep reg_a with "k" offset */
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_gp_reg_scratch, i_bk * i_packed_width * i_micro_kernel_config->datatype_size_in );
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MUL, i_gp_reg_mapping->gp_reg_kloop, l_gp_reg_scratch_32bit, l_gp_reg_scratch_32bit, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, i_gp_reg_mapping->gp_reg_a, l_gp_reg_scratch, l_tmp_a_gp_reg, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    /* Prep reg_b with "k" offset */
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_gp_reg_scratch, i_bk * i_bn  * i_micro_kernel_config->datatype_size_in);
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MUL, l_cur_column_gpr, l_gp_reg_scratch_32bit, l_gp_reg_scratch_32bit, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, i_gp_reg_mapping->gp_reg_b, l_gp_reg_scratch, i_gp_reg_mapping->gp_reg_help_1, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );

    if (l_n_blocks > 1) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_mapping->gp_reg_help_1,
                                                     l_gp_reg_scratch,
                                                     i_gp_reg_mapping->gp_reg_help_1,
                                                     ((long long)l_n_block_id * l_n_blocking * 2 * l_vnni_block_size) * i_micro_kernel_config->datatype_size_in );
    }

    if ( l_assm_fma_iters > 1 ) {
      libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, l_k_loop_fma_reg, l_assm_fma_iters);
    }

    for (l_fma_i = 0; l_fma_i < l_fma_iters; l_fma_i++) {
      /* Load A registers  */
      l_a_adjustments = 0;
      l_b_adjustments = 0;
      for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
        libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1H_I_OFF,
            l_tmp_a_gp_reg, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, l_max_reg_block+l_p, ((i_packed_remainder > 0) && (l_p == i_packed_blocking - 1)) ? l_input_bf16_mask : LIBXSMM_AARCH64_SVE_REG_P0 );
        if (l_p < i_packed_blocking - 1) {
          l_a_adjustments++;
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                         l_tmp_a_gp_reg,
                                                         l_gp_reg_scratch,
                                                         l_tmp_a_gp_reg,
                                                         (long long) i_simd_packed_width * l_vnni_block_size * i_micro_kernel_config->datatype_size_in );
        }
      }
      if ((l_fma_iters > 1 && l_assm_fma_iters == 1) || (l_assm_fma_iters > 1)) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       l_tmp_a_gp_reg,
                                                       l_gp_reg_scratch,
                                                       l_tmp_a_gp_reg,
                                                       (long long)(i_packed_width * l_vnni_block_size - l_a_adjustments * i_simd_packed_width * l_vnni_block_size ) * i_micro_kernel_config->datatype_size_in );
      }


      /* loop over the columns of B/C */
      for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
        if ((l_n == l_n_blocking - 1) && ((i_bn % 2 != 0) && (is_last_n_block > 0))) {
          libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_LD1RD_I_OFF,
                                                i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, l_max_reg_block+i_packed_blocking, LIBXSMM_AARCH64_SVE_REG_P0 );
        } else {
          libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_LD1RQD_I_OFF,
                                                i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, l_max_reg_block+i_packed_blocking, LIBXSMM_AARCH64_SVE_REG_P0 );
        }
        if (l_n < (l_n_blocking-1)) {
          l_b_adjustments++;
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_mapping->gp_reg_help_1,
                                                       l_gp_reg_scratch,
                                                       i_gp_reg_mapping->gp_reg_help_1,
                                                       ((long long)2 * l_vnni_block_size) * i_micro_kernel_config->datatype_size_in );
        }

        for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
          unsigned int l_mmla_instruction = (l_bf16_kernel > 0) ? LIBXSMM_AARCH64_INSTR_SVE_BFMMLA_V : LIBXSMM_AARCH64_INSTR_SVE_USMMLA_V ;
          libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                                 l_mmla_instruction,
                                                 (l_is_s8u8s32_kernel == 0) ? l_max_reg_block+i_packed_blocking : l_max_reg_block+l_p,
                                                 (l_is_s8u8s32_kernel == 0) ? l_max_reg_block+l_p : l_max_reg_block+i_packed_blocking,
                                                 0,
                                                 (l_n*i_packed_blocking) + l_p,
                                                 LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                                 (libxsmm_aarch64_sve_type)0 );
        }
      }
      if ((l_fma_iters > 1 && l_assm_fma_iters == 1) || (l_assm_fma_iters > 1)) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_mapping->gp_reg_help_1,
                                                       l_gp_reg_scratch,
                                                       i_gp_reg_mapping->gp_reg_help_1,
                                                       (i_bn * l_vnni_block_size - l_b_adjustments * 2 * l_vnni_block_size) * i_micro_kernel_config->datatype_size_in );
      }
    }

    if ( l_assm_fma_iters > 1 ) {
      libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, l_k_loop_fma_reg, 1 );
    }

    /* k loop footer */
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, l_cur_column_gpr, l_cur_column_gpr, 1, 0 );
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR, l_next_column_gpr, l_cur_column_gpr, l_gp_reg_scratch_32bit, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    libxsmm_aarch64_instruction_cond_jump_back_to_label( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_CBNZ, l_gp_reg_scratch_32bit, io_loop_label_tracker );
    if (l_beta_0 > 0) {
      /* LABEL for empty column and beta == 0 */
      libxsmm_aarch64_instruction_register_jump_label(io_generated_code, EMPTY_BLOCK_COLUMN_LABEL_BETA0, i_jump_label_tracker);
    }

    /* store C accumulator */
    l_n_advancements = 0;
    for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
      l_n_advancements++;
      for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
        unsigned int l_reg0 = (l_n*i_packed_blocking) + l_p;
        unsigned int l_mask_use = 0;
        unsigned int l_c_store_instr = (l_bf16_kernel > 0) ? LIBXSMM_AARCH64_INSTR_SVE_ST1H_I_OFF : LIBXSMM_AARCH64_INSTR_SVE_ST1W_I_OFF;
        l_mask_use = LIBXSMM_AARCH64_SVE_REG_P1;
        if ((i_packed_remainder > 0) && (l_p == i_packed_blocking - 1)) {
          l_mask_use = LIBXSMM_AARCH64_SVE_REG_P2;
        }
        /* Unzip 64bit elements from [2M][2n][2m] to get 4m for n0 */
        libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                                 LIBXSMM_AARCH64_INSTR_SVE_UZP1_V,
                                                 l_reg0,
                                                 l_reg0,
                                                 0,
                                                 l_vec_reg_tmp[0],
                                                 LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                                 (l_is_s8u8s32_kernel == 0) ? LIBXSMM_AARCH64_SVE_TYPE_D : LIBXSMM_AARCH64_SVE_TYPE_S );
        if (l_bf16_kernel > 0) {
          libxsmm_generator_vcvt_f32bf16_aarch64_sve( io_generated_code, l_vec_reg_tmp[0], 0);
        }
        libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                              l_c_store_instr,
                                              i_gp_reg_mapping->gp_reg_c,
                                              LIBXSMM_AARCH64_GP_REG_UNDEF,
                                              0,
                                              l_vec_reg_tmp[0],
                                              l_mask_use );
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_mapping->gp_reg_c,
                                                       l_gp_reg_scratch,
                                                       i_gp_reg_mapping->gp_reg_c,
                                                       i_packed_width * i_micro_kernel_config->datatype_size_out );

        if ((l_n == l_n_blocking-1) && ((i_bn % 2 != 0) && (is_last_n_block > 0))) {
          /* Do nothing */
        } else {
          /* Unzip 64bit elements from [2M][2n][2m] to get 4m for n1 */
          libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_SVE_UZP2_V,
                                                   l_reg0,
                                                   l_reg0,
                                                   0,
                                                   l_vec_reg_tmp[1],
                                                   LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                                   (l_is_s8u8s32_kernel == 0) ? LIBXSMM_AARCH64_SVE_TYPE_D : LIBXSMM_AARCH64_SVE_TYPE_S );
          if (l_bf16_kernel > 0) {
            libxsmm_generator_vcvt_f32bf16_aarch64_sve( io_generated_code, l_vec_reg_tmp[1], 0);
          }
          libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                l_c_store_instr,
                                                i_gp_reg_mapping->gp_reg_c,
                                                LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                0,
                                                l_vec_reg_tmp[1],
                                                l_mask_use );
        }

        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                       i_gp_reg_mapping->gp_reg_c,
                                                       l_gp_reg_scratch,
                                                       i_gp_reg_mapping->gp_reg_c,
                                                       ((long long)i_packed_width - i_simd_packed_width) * i_micro_kernel_config->datatype_size_out );
      }
      if ((i_packed_width * 2 - i_packed_blocking * i_simd_packed_width) > 0) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_mapping->gp_reg_c,
                                                       l_gp_reg_scratch,
                                                       i_gp_reg_mapping->gp_reg_c,
                                                       ((long long)i_packed_width * 2 - i_packed_blocking * i_simd_packed_width) * i_micro_kernel_config->datatype_size_out );
      }
    }
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_c,
                                                   l_gp_reg_scratch,
                                                   i_gp_reg_mapping->gp_reg_c,
                                                   (i_packed_width * 2 * l_n_advancements) * i_micro_kernel_config->datatype_size_out );
    if (l_n_blocks > 1) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_mapping->gp_reg_c,
                                                     l_gp_reg_scratch,
                                                     i_gp_reg_mapping->gp_reg_c,
                                                     (long long)i_packed_width * l_n_blocking * 2 * i_micro_kernel_config->datatype_size_out );
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_X16, l_cur_column_gpr_64bit, 0, 0 );
    }
    EMPTY_BLOCK_COLUMN_LABEL_BETA0++;
  }
  if (l_n_blocks > 1) {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_c,
                                                   l_gp_reg_scratch,
                                                   i_gp_reg_mapping->gp_reg_c,
                                                   (long long)i_packed_width * l_n_blocking * 2 * l_n_blocks * i_micro_kernel_config->datatype_size_out );
  }

  /* packed loop */
  if ( i_packed_range/i_packed_blocking > 1 ) {
    /* advance A and C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                   (long long)i_packed_blocking*i_simd_packed_width*l_vnni_block_size*i_micro_kernel_config->datatype_size_in );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                   (long long)i_packed_blocking*i_simd_packed_width*i_micro_kernel_config->datatype_size_out );
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_X16, l_cur_column_gpr_64bit, 0, 0 );
    /* packed loop footer */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_help_3, 1 );

    /* reset A and C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                   ((long long)i_packed_range/i_packed_blocking)*i_packed_blocking*i_simd_packed_width*l_vnni_block_size*i_micro_kernel_config->datatype_size_in );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                   ((long long)i_packed_range/i_packed_blocking)*i_packed_blocking*i_simd_packed_width*i_micro_kernel_config->datatype_size_out );
  }

  if (l_beta_0 == 0) {
    /* LABEL for empty column and beta == 1 */
    libxsmm_aarch64_instruction_register_jump_label(io_generated_code, EMPTY_BLOCK_COLUMN_LABEL, i_jump_label_tracker);
  }

  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
      i_gp_reg_mapping->gp_reg_a, l_gp_reg_scratch, i_gp_reg_mapping->gp_reg_a, (long long)i_packed_processed * i_simd_packed_width * l_vnni_block_size * i_micro_kernel_config->datatype_size_in );

  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,  LIBXSMM_AARCH64_INSTR_GP_META_SUB,
    i_gp_reg_mapping->gp_reg_c, l_gp_reg_scratch, i_gp_reg_mapping->gp_reg_c, (1ull * i_packed_processed * i_simd_packed_width) * i_micro_kernel_config->datatype_size_out );

}

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_bcsc_bsparse_aarch64_kloop_bfdot_sve(libxsmm_generated_code*            io_generated_code,
                                                                          libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                          libxsmm_jump_label_tracker*        i_jump_label_tracker,
                                                                          const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                          const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                          const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                          const unsigned int                 i_packed_processed,
                                                                          const unsigned int                 i_packed_range,
                                                                          const unsigned int                 i_packed_blocking,
                                                                          const unsigned int                 i_packed_remainder,
                                                                          const unsigned int                 i_packed_width,
                                                                          const unsigned int                 i_simd_packed_width,
                                                                          const unsigned int                 i_bk,
                                                                          const unsigned int                 i_bn) {
  unsigned int l_n = 0, l_n_blocks = 0, l_n_block_id = 0;
  unsigned int l_p = 0;
  unsigned int l_max_reg_block = 0;
  unsigned int l_n_blocking = 0;
  unsigned int l_n_advancements = 0;
  unsigned int l_vnni_block_size = ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ? 2 : (( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) ? 4 : 1);
  unsigned int l_c_vnni_block_size = 1;
  unsigned int l_bf16_comp = ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ? 1 : 0;
  unsigned int l_i8_comp = ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ? 1 : 0;
  unsigned int l_fma_instr = ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ? LIBXSMM_AARCH64_INSTR_SVE_BFDOT_V : ( ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ? LIBXSMM_AARCH64_INSTR_SVE_USDOT_V : LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_P);
  unsigned int l_c_bf16 = ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ? 1 : 0;
  unsigned int l_load_c_instr = (l_c_bf16 > 0) ? LIBXSMM_AARCH64_INSTR_SVE_LD1H_I_OFF : LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF;
  unsigned int l_load_a_instr = ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ? LIBXSMM_AARCH64_INSTR_SVE_LD1H_I_OFF : (( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ? LIBXSMM_AARCH64_INSTR_SVE_LD1B_I_OFF : LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF);
  unsigned int l_store_c_instr = (l_c_bf16 > 0) ? LIBXSMM_AARCH64_INSTR_SVE_ST1H_I_OFF : LIBXSMM_AARCH64_INSTR_SVE_ST1W_I_OFF;
  unsigned int l_beta_0 = (0 != (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) ? 1 : 0;
  unsigned int l_is_s8u8s32_kernel = ((l_i8_comp > 0) && ((LIBXSMM_GEMM_FLAG_A_UNSIGNED & i_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_B_UNSIGNED & i_xgemm_desc->flags) > 0) ) ? 1 : 0;

  unsigned int l_fma_iters = i_bk/l_vnni_block_size;
  unsigned int l_assm_fma_iters = 1;
  unsigned int l_fma_i = 0;
  unsigned int l_a_adjustments = 0;
  unsigned int l_b_adjustments = 0;

  /* Auxiliary GPRs  */
  unsigned int l_gp_reg_scratch = i_gp_reg_mapping->gp_reg_help_2;
  unsigned int l_gp_reg_scratch_32bit = i_gp_reg_mapping->gp_reg_help_2 - 32;
  unsigned int l_tmp_a_gp_reg = LIBXSMM_AARCH64_GP_REG_X8;
  unsigned int l_k_loop_fma_reg = LIBXSMM_AARCH64_GP_REG_X17;
  unsigned int l_output_bf16_mask = ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ? LIBXSMM_AARCH64_SVE_REG_P2 : 0;
  unsigned int l_output_remainder_bf16_mask = LIBXSMM_AARCH64_SVE_REG_P4;
  unsigned int l_input_bf16_mask = LIBXSMM_AARCH64_SVE_REG_P5;
  unsigned int l_row_idx_gpr = i_gp_reg_mapping->gp_reg_help_4;
  unsigned int l_cur_column_gpr = i_gp_reg_mapping->gp_reg_help_5;
  unsigned int l_cur_column_gpr_64bit = l_cur_column_gpr + 32;
  unsigned int l_next_column_gpr = i_gp_reg_mapping->gp_reg_help_6;
  unsigned int EMPTY_BLOCK_COLUMN_LABEL = (i_packed_processed == 0) ? 0 : 1;
  unsigned int EMPTY_BLOCK_COLUMN_LABEL_BETA0 = 0;

  if (l_fma_iters > 2) {
    l_assm_fma_iters = l_fma_iters;
    l_fma_iters = 2;
    while (l_assm_fma_iters % l_fma_iters != 0) {
      l_fma_iters--;
    }
    l_assm_fma_iters = l_assm_fma_iters/l_fma_iters;
  }

  LIBXSMM_ASSERT( i_packed_blocking > 0 );

  libxsmm_spgemm_max_mn_blocking_factors_aarch64(io_generated_code, 0, i_bn, &l_n, &l_n_blocking);
  l_max_reg_block = l_n_blocking * i_packed_blocking;
  l_n_blocks = i_bn / l_n_blocking;
  EMPTY_BLOCK_COLUMN_LABEL_BETA0 = (i_packed_processed == 0) ? 2 : 2 + l_n_blocks;

  /* Adjust A and C pointers for already processed N/M  */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,  LIBXSMM_AARCH64_INSTR_GP_META_ADD,
      i_gp_reg_mapping->gp_reg_c, l_gp_reg_scratch, i_gp_reg_mapping->gp_reg_c, (1ull * i_packed_processed * i_simd_packed_width) * i_micro_kernel_config->datatype_size_out );

  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
      i_gp_reg_mapping->gp_reg_a, l_gp_reg_scratch, i_gp_reg_mapping->gp_reg_a, (long long)i_packed_processed * i_simd_packed_width * l_vnni_block_size * i_micro_kernel_config->datatype_size_in );

  if (l_beta_0 == 0) {
    /* Check if empty B column and beta == 1 and jump at the end of the kernel */
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR, l_next_column_gpr, l_cur_column_gpr, l_gp_reg_scratch_32bit, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    libxsmm_aarch64_instruction_cond_jump_to_label( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_CBZ, l_gp_reg_scratch_32bit, EMPTY_BLOCK_COLUMN_LABEL, i_jump_label_tracker );
  }

  /* packed loop */
  if ( i_packed_range/i_packed_blocking > 1 ) {
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_help_3, i_packed_range/i_packed_blocking );
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, l_cur_column_gpr_64bit, LIBXSMM_AARCH64_GP_REG_X16, 0, 0 );
  }

  for (l_n_block_id = 0; l_n_block_id < l_n_blocks; l_n_block_id++) {
    if (l_n_blocks > 1) {
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, l_cur_column_gpr_64bit, LIBXSMM_AARCH64_GP_REG_X16, 0, 0 );
    }
    /* load C accumulator */
    if (l_beta_0 > 0) {
      for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
        for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
          unsigned int l_reg0 = (l_n*i_packed_blocking) + l_p;
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V, l_reg0, l_reg0, 0, l_reg0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
        }
      }
    } else {
      l_n_advancements = 0;
      for ( l_n = 0; l_n < l_n_blocking; l_n++) {
        l_n_advancements++;
        for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
          unsigned int l_reg0 = (l_n*i_packed_blocking) + l_p;
          /* load 1st [8m] */
          libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                l_load_c_instr,
                                                i_gp_reg_mapping->gp_reg_c,
                                                LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                0,
                                                l_reg0,
                                                ((i_packed_remainder > 0) && (l_p == i_packed_blocking - 1)) ? l_output_remainder_bf16_mask  : l_output_bf16_mask );
          if (l_c_bf16 > 0) {
            libxsmm_generator_vcvt_bf16f32_aarch64_sve( io_generated_code, l_reg0, 0);
          }
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                         i_gp_reg_mapping->gp_reg_c,
                                                         l_gp_reg_scratch,
                                                         i_gp_reg_mapping->gp_reg_c,
                                                         i_simd_packed_width * i_micro_kernel_config->datatype_size_out );
        }
        if ((i_packed_width - i_packed_blocking * i_simd_packed_width) > 0) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                         i_gp_reg_mapping->gp_reg_c,
                                                         l_gp_reg_scratch,
                                                         i_gp_reg_mapping->gp_reg_c,
                                                         ((long long)i_packed_width - i_packed_blocking * i_simd_packed_width) * i_micro_kernel_config->datatype_size_out );
        }
      }
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                     i_gp_reg_mapping->gp_reg_c,
                                                     l_gp_reg_scratch,
                                                     i_gp_reg_mapping->gp_reg_c,
                                                     (i_packed_width * l_n_advancements) * i_micro_kernel_config->datatype_size_out );
    }

    if (l_beta_0 > 0) {
      /* Check if empty B column and beta == 1 and jump at the C store part of the kernel */
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR, l_next_column_gpr, l_cur_column_gpr, l_gp_reg_scratch_32bit, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
      libxsmm_aarch64_instruction_cond_jump_to_label( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_CBZ, l_gp_reg_scratch_32bit, EMPTY_BLOCK_COLUMN_LABEL_BETA0, i_jump_label_tracker );
    }

    /* k loop header */
    libxsmm_aarch64_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, l_row_idx_gpr, l_cur_column_gpr_64bit, l_gp_reg_scratch, 2, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, l_gp_reg_scratch, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_gp_reg_mapping->gp_reg_kloop );

    /* Prep reg_a with "k" offset */
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_gp_reg_scratch, i_bk * i_packed_width * i_micro_kernel_config->datatype_size_in );
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MUL, i_gp_reg_mapping->gp_reg_kloop, l_gp_reg_scratch_32bit, l_gp_reg_scratch_32bit, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, i_gp_reg_mapping->gp_reg_a, l_gp_reg_scratch, l_tmp_a_gp_reg, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    /* Prep reg_b with "k" offset */
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_gp_reg_scratch, i_bk * i_bn  * i_micro_kernel_config->datatype_size_in);
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MUL, l_cur_column_gpr, l_gp_reg_scratch_32bit, l_gp_reg_scratch_32bit, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, i_gp_reg_mapping->gp_reg_b, l_gp_reg_scratch, i_gp_reg_mapping->gp_reg_help_1, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );

    if (l_n_blocks > 1) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_mapping->gp_reg_help_1,
                                                     l_gp_reg_scratch,
                                                     i_gp_reg_mapping->gp_reg_help_1,
                                                     ((long long)l_n_block_id * l_n_blocking * i_bk) * i_micro_kernel_config->datatype_size_in );
    }

    if ( l_assm_fma_iters > 1 ) {
      libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, l_k_loop_fma_reg, l_assm_fma_iters);
    }


    for (l_fma_i = 0; l_fma_i < l_fma_iters; l_fma_i++) {
      /* Load A registers  */
      l_a_adjustments = 0;
      l_b_adjustments = 0;
      for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
        libxsmm_aarch64_instruction_sve_move( io_generated_code, l_load_a_instr,
            l_tmp_a_gp_reg, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, l_max_reg_block+l_p, ((i_packed_remainder > 0) && (l_p == i_packed_blocking - 1)) ? l_input_bf16_mask : LIBXSMM_AARCH64_SVE_REG_P0 );
        if (l_p < i_packed_blocking - 1) {
          l_a_adjustments++;
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                         l_tmp_a_gp_reg,
                                                         l_gp_reg_scratch,
                                                         l_tmp_a_gp_reg,
                                                         (long long) i_simd_packed_width * l_vnni_block_size * i_micro_kernel_config->datatype_size_in );
        }
      }
      if ((l_fma_iters > 1 && l_assm_fma_iters == 1) || (l_assm_fma_iters > 1)) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       l_tmp_a_gp_reg,
                                                       l_gp_reg_scratch,
                                                       l_tmp_a_gp_reg,
                                                       ((long long)i_packed_width * l_vnni_block_size - l_a_adjustments * i_simd_packed_width * l_vnni_block_size ) * i_micro_kernel_config->datatype_size_in );
      }

      /* loop over the columns of B/C */
      for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
        libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF,
                                              i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, l_max_reg_block+i_packed_blocking, LIBXSMM_AARCH64_SVE_REG_P0 );
        if (l_n < l_n_blocking - 1) {
            l_b_adjustments++;
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                         i_gp_reg_mapping->gp_reg_help_1,
                                                         l_gp_reg_scratch,
                                                         i_gp_reg_mapping->gp_reg_help_1,
                                                         ((long long)i_bk) * i_micro_kernel_config->datatype_size_in );
        }

        for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                                 l_fma_instr,
                                                 (l_is_s8u8s32_kernel == 0) ? l_max_reg_block+i_packed_blocking : l_max_reg_block+l_p,
                                                 (l_is_s8u8s32_kernel == 0) ? l_max_reg_block+l_p : l_max_reg_block+i_packed_blocking,
                                                 0,
                                                 (l_n*i_packed_blocking) + l_p,
                                                 (l_bf16_comp > 0 || l_i8_comp > 0) ? LIBXSMM_AARCH64_SVE_REG_UNDEF : 0,
                                                 (l_bf16_comp > 0) ? LIBXSMM_AARCH64_SVE_TYPE_H : ((l_i8_comp > 0) ? LIBXSMM_AARCH64_SVE_TYPE_B: LIBXSMM_AARCH64_SVE_TYPE_S));
        }
      }
      if ((l_fma_iters > 1 && l_assm_fma_iters == 1) || (l_assm_fma_iters > 1)) {
        if ( (i_bk * l_b_adjustments - l_vnni_block_size) > 0 ) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                       i_gp_reg_mapping->gp_reg_help_1,
                                                       l_gp_reg_scratch,
                                                       i_gp_reg_mapping->gp_reg_help_1,
                                                       ((long long)i_bk * l_b_adjustments - l_vnni_block_size) * i_micro_kernel_config->datatype_size_in );
        }
      }
    }

    if ( l_assm_fma_iters > 1 ) {
      libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, l_k_loop_fma_reg, 1 );
    }

    /* k loop footer */
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, l_cur_column_gpr, l_cur_column_gpr, 1, 0 );
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR, l_next_column_gpr, l_cur_column_gpr, l_gp_reg_scratch_32bit, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    libxsmm_aarch64_instruction_cond_jump_back_to_label( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_CBNZ, l_gp_reg_scratch_32bit, io_loop_label_tracker );
    if (l_beta_0 > 0) {
      /* LABEL for empty column and beta == 0 */
      libxsmm_aarch64_instruction_register_jump_label(io_generated_code, EMPTY_BLOCK_COLUMN_LABEL_BETA0, i_jump_label_tracker);
    }

    /* store C accumulator */
    l_n_advancements = 0;
    for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
      l_n_advancements++;
      for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
        unsigned int l_reg0 = (l_n*i_packed_blocking) + l_p;
        unsigned int l_mask_use = 0;
        if ((i_packed_remainder > 0) && (l_p == i_packed_blocking - 1)) {
          l_mask_use = l_output_remainder_bf16_mask;
        } else {
          l_mask_use = l_output_bf16_mask;
        }
        /* Cvt + store --> write 1st [8m] */
        if (l_c_bf16 > 0) {
          libxsmm_generator_vcvt_f32bf16_aarch64_sve( io_generated_code, l_reg0, 0);
        }
        libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                              l_store_c_instr,
                                              i_gp_reg_mapping->gp_reg_c,
                                              LIBXSMM_AARCH64_GP_REG_UNDEF,
                                              0,
                                              l_reg0,
                                              l_mask_use );

        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_mapping->gp_reg_c,
                                                       l_gp_reg_scratch,
                                                       i_gp_reg_mapping->gp_reg_c,
                                                       i_simd_packed_width * i_micro_kernel_config->datatype_size_out );
      }
      if ((i_packed_width - i_packed_blocking * i_simd_packed_width) > 0) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_mapping->gp_reg_c,
                                                       l_gp_reg_scratch,
                                                       i_gp_reg_mapping->gp_reg_c,
                                                       ((long long)i_packed_width - i_packed_blocking * i_simd_packed_width) * i_micro_kernel_config->datatype_size_out );
      }
    }
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_c,
                                                   l_gp_reg_scratch,
                                                   i_gp_reg_mapping->gp_reg_c,
                                                   (i_packed_width * l_n_advancements) * i_micro_kernel_config->datatype_size_out );

    if (l_n_blocks > 1) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_mapping->gp_reg_c,
                                                     l_gp_reg_scratch,
                                                     i_gp_reg_mapping->gp_reg_c,
                                                     (long long)i_packed_width * l_n_blocking * i_micro_kernel_config->datatype_size_out );
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_X16, l_cur_column_gpr_64bit, 0, 0 );
    }
    EMPTY_BLOCK_COLUMN_LABEL_BETA0++;
  }
  if (l_n_blocks > 1) {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_c,
                                                   l_gp_reg_scratch,
                                                   i_gp_reg_mapping->gp_reg_c,
                                                   (long long)i_packed_width * l_n_blocking * l_n_blocks * i_micro_kernel_config->datatype_size_out );
  }

  /* packed loop */
  if ( i_packed_range/i_packed_blocking > 1 ) {
    /* advance A and C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                   (long long)i_packed_blocking*i_simd_packed_width*l_vnni_block_size*i_micro_kernel_config->datatype_size_in );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                   (long long)i_packed_blocking*i_simd_packed_width*l_c_vnni_block_size*i_micro_kernel_config->datatype_size_out );
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_X16, l_cur_column_gpr_64bit, 0, 0 );
    /* packed loop footer */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_help_3, 1 );

    /* reset A and C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                   ((long long)i_packed_range/i_packed_blocking)*i_packed_blocking*i_simd_packed_width*l_vnni_block_size*i_micro_kernel_config->datatype_size_in );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                   ((long long)i_packed_range/i_packed_blocking)*i_packed_blocking*i_simd_packed_width*l_c_vnni_block_size*i_micro_kernel_config->datatype_size_out );
  }

  if (l_beta_0 == 0) {
    /* LABEL for empty column and beta == 1 */
    libxsmm_aarch64_instruction_register_jump_label(io_generated_code, EMPTY_BLOCK_COLUMN_LABEL, i_jump_label_tracker);
  }

  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
      i_gp_reg_mapping->gp_reg_a, l_gp_reg_scratch, i_gp_reg_mapping->gp_reg_a, (long long)i_packed_processed * i_simd_packed_width * l_vnni_block_size * i_micro_kernel_config->datatype_size_in );
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,  LIBXSMM_AARCH64_INSTR_GP_META_SUB,
      i_gp_reg_mapping->gp_reg_c, l_gp_reg_scratch, i_gp_reg_mapping->gp_reg_c, (1ull * i_packed_processed * i_simd_packed_width) * i_micro_kernel_config->datatype_size_out );
}
