/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
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
void libxsmm_generator_packed_spgemm_bcsc_bsparse_aarch64( libxsmm_generated_code*         io_generated_code,
                                                           const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                           const unsigned int*             i_row_idx,
                                                           const unsigned int*             i_column_idx,
                                                           const unsigned int              i_packed_width,
                                                           const unsigned int              i_bk,
                                                           const unsigned int              i_bn ) {
  unsigned int l_n = 0;
  unsigned int l_max_cols = 0;
  unsigned int l_simd_packed_remainder = 0;
  unsigned int l_simd_packed_iters = 0;
  unsigned int l_simd_packed_iters_full = 0;
  unsigned int l_simd_packed_width = 4;
  unsigned int l_packed_done = 0;
  unsigned int l_packed_count = 0;
  unsigned int l_packed_reg_block[2] = {0,0};
  unsigned int l_packed_reg_range[2] = {0,0};
  unsigned int l_col_reg_block[2][2] = { {0,0}, {0,0} };
  unsigned int l_col_reg_range[2][2] = { {0,0}, {0,0} };
  unsigned int l_bf16_mmla_kernel = 0;
  unsigned int l_vreg_perm_table_loadstore_c = 31;

  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;

  /* select simd packing width and accumulator blocking */
  if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
    if ( io_generated_code->arch >= LIBXSMM_AARCH64_SVE128 ) {
      if ( io_generated_code->arch < LIBXSMM_AARCH64_SVE256 ) {
        l_simd_packed_width = 2;
      } else if ( io_generated_code->arch < LIBXSMM_AARCH64_SVE512 ) {
        l_simd_packed_width = 4;
      } else {
        l_simd_packed_width = 8;
      }
    } else { /* asimd */
      l_simd_packed_width = 2;
    }
  } else if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
    if ( io_generated_code->arch >= LIBXSMM_AARCH64_SVE128 ) {
      if ( io_generated_code->arch < LIBXSMM_AARCH64_SVE256 ) {
        l_simd_packed_width = 4;
      } else if ( io_generated_code->arch < LIBXSMM_AARCH64_SVE512 ) {
        l_simd_packed_width = 8;
      } else {
        l_simd_packed_width = 16;
      }
    } else { /* asimd */
      l_simd_packed_width = 4;
    }
  } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
    if (io_generated_code->arch == LIBXSMM_AARCH64_NEOV1) {
      char l_use_bfdot = (char)libxsmm_cpuid_arm_use_bfdot();
      if (l_use_bfdot == 0) {
        l_simd_packed_width = 4;
        l_bf16_mmla_kernel = 1;
      } else {
        l_simd_packed_width = 8;
        l_bf16_mmla_kernel = 0;
      }
      /* TODO: Check for A in VNNI and C in VNNI */
      /* TODO: Check provided bk and bn in BCSC format */
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }

  if ( LIBXSMM_DATATYPE_BF16 != LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }

  l_simd_packed_remainder = i_packed_width % l_simd_packed_width;
  l_simd_packed_iters_full = i_packed_width/l_simd_packed_width;
  l_simd_packed_iters = ( l_simd_packed_remainder > 0 ) ? l_simd_packed_iters_full+1 : l_simd_packed_iters_full;

  /* get max column in C */
  l_max_cols = i_xgemm_desc->n/i_bn;
  for ( l_n = 0; l_n < i_xgemm_desc->n/i_bn; l_n++ ) {
    if ( i_column_idx[l_n] == i_column_idx[i_xgemm_desc->n/i_bn] ) {
      l_max_cols = l_n;
      break;
    }
  }

  libxsmm_compute_equalized_blocking( l_simd_packed_iters, LIBXSMM_UPDIV(32, l_max_cols), &(l_packed_reg_range[0]), &(l_packed_reg_block[0]), &(l_packed_reg_range[1]), &(l_packed_reg_block[1]) );

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
  l_gp_reg_mapping.gp_reg_help_3 = LIBXSMM_AARCH64_GP_REG_X12;
  l_gp_reg_mapping.gp_reg_help_4 = LIBXSMM_AARCH64_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_5 = LIBXSMM_AARCH64_GP_REG_UNDEF;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config_aarch64( &l_micro_kernel_config, io_generated_code->arch, i_xgemm_desc );

  /* open asm */
  libxsmm_aarch64_instruction_open_stream( io_generated_code, 0xf );

  /* implementing load from struct */
  if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI & i_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI) ) {
    /* RDI holds the pointer to the strcut, so lets first move this one into R15 */
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

  if ((LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) && (io_generated_code->arch == LIBXSMM_AARCH64_NEOV1)) {
    unsigned int l_loadstore_c_perm_indices[8] = { 0, 2, 1, 3, 4, 6, 5, 7 };
    unsigned int l_max_n_blocking = 6;
    unsigned int l_max_m_blocking = 4;
    /* Set blocking factor decisions...  */
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

    if (l_max_cols <= l_max_n_blocking) {
      l_col_reg_range[0][0] = l_max_cols;
      l_col_reg_block[0][0] = l_max_cols;
      l_col_reg_range[0][1] = 0;
      l_col_reg_block[0][1] = 0;
    } else {
      l_col_reg_range[0][0] = l_max_cols - l_max_cols % l_max_n_blocking;
      l_col_reg_block[0][0] = l_max_n_blocking;
      l_col_reg_range[0][1] = l_max_cols % l_max_n_blocking;
      l_col_reg_block[0][1] = l_max_cols % l_max_n_blocking;
    }

    if (l_packed_reg_range[1] > 0) {
      if (l_max_cols < l_max_n_blocking) {
        l_col_reg_range[1][0] = l_max_cols;
        l_col_reg_block[1][0] = l_max_cols;
        l_col_reg_range[1][1] = 0;
        l_col_reg_block[1][1] = 0;
      } else {
        l_col_reg_range[1][0] = l_max_cols - l_max_cols % l_max_n_blocking;
        l_col_reg_block[1][0] = l_max_n_blocking;
        l_col_reg_range[1][1] = l_max_cols % l_max_n_blocking;
        l_col_reg_block[1][1] = l_max_cols % l_max_n_blocking;
      }
    }


    if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) {
      int l_nnz_bits2 = 16;
      unsigned char l_mask_array[32] = { 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                    LIBXSMM_AARCH64_SVE_REG_P2,
                                                    l_nnz_bits2,
                                                    l_gp_reg_mapping.gp_reg_help_0 );
      /* mask for M remainder  */
      if ( l_simd_packed_remainder != 0 ) {
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
      }

      if ((l_col_reg_block[0][0] % 2 > 0) || (l_col_reg_block[0][1] % 2 > 0) ) {
        libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, l_vreg_perm_table_loadstore_c, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_help_1,
                                                                LIBXSMM_AARCH64_SVE_REG_P0, (unsigned int *)l_mask_array, 32 );
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_CMPGT_Z_V,
                                                 l_vreg_perm_table_loadstore_c, LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, LIBXSMM_AARCH64_SVE_REG_P3,
                                                 LIBXSMM_AARCH64_SVE_REG_P0, libxsmm_generator_aarch64_get_sve_type(LIBXSMM_CAST_UCHAR(1)) );
        if (l_simd_packed_remainder % 2 > 0) {
          libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, LIBXSMM_AARCH64_SVE_REG_P6, 4, l_gp_reg_mapping.gp_reg_help_0 );

        }
      }
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, l_vreg_perm_table_loadstore_c, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_help_1,
                                                            LIBXSMM_AARCH64_SVE_REG_P0, (unsigned int *)l_loadstore_c_perm_indices, 32 );
#if 0
    printf("packed blocking (range0, block0, range1, block1): %u %u %u %u\n", l_packed_reg_range[0], l_packed_reg_block[0], l_packed_reg_range[1], l_packed_reg_block[1]);
    printf("n blocking 0    (range0, block0, range1, block1): %u %u %u %u\n",  l_col_reg_range[0][0],  l_col_reg_block[0][0],  l_col_reg_range[0][1],  l_col_reg_block[0][1]);
#endif
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }

  /* m loop */
  libxsmm_generator_loop_header_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_mloop, i_xgemm_desc->m );

  /* loop over packed blocks */
  while ( l_packed_done != l_simd_packed_iters ) {
    unsigned int l_packed_blocking = l_packed_reg_block[l_packed_count];
    unsigned int l_packed_remainder = 0;
    unsigned int l_n_done = 0;
    unsigned int l_n_count = 0;
    unsigned int l_n_processed = 0;

    /* coverity[dead_error_line] */
    if ( (l_simd_packed_remainder != 0) && (l_packed_count == 0) ) {
      if ( l_packed_reg_block[1] > 0 ) {
        l_packed_remainder = 0;
      } else {
         l_packed_remainder = l_simd_packed_remainder;
      }
    } else if (l_simd_packed_remainder != 0) {
      l_packed_remainder = l_simd_packed_remainder;
    }

    while ( l_n_done < l_max_cols ) {
      unsigned int l_n_blocking = l_col_reg_block[l_packed_count][l_n_count];
      for ( l_n_processed = l_n_done; l_n_processed < l_n_done + l_col_reg_range[l_packed_count][l_n_count]; l_n_processed += l_n_blocking ) {
        if (l_bf16_mmla_kernel > 0) {
          libxsmm_generator_packed_spgemm_bcsc_bsparse_aarch64_kloop_mmla_sve( io_generated_code,
                                                                         &l_loop_label_tracker,
                                                                         &l_gp_reg_mapping,
                                                                         &l_micro_kernel_config,
                                                                         i_xgemm_desc,
                                                                         i_row_idx,
                                                                         i_column_idx,
                                                                         l_n_processed,
                                                                         l_n_processed + l_n_blocking,
                                                                         l_packed_done,
                                                                         l_packed_reg_range[l_packed_count],
                                                                         l_packed_blocking,
                                                                         l_packed_remainder,
                                                                         i_packed_width,
                                                                         l_simd_packed_width,
                                                                         i_bk,
                                                                         i_bn);
        } else {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
          return;
        }
      }
      /* advance N */
      l_n_done += l_col_reg_range[l_packed_count][l_n_count];
      l_n_count++;
    }
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
                                                                          const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                          const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                          const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                          const unsigned int*                i_row_idx,
                                                                          const unsigned int*                i_column_idx,
                                                                          const unsigned int                 i_n_processed,
                                                                          const unsigned int                 i_n_limit,
                                                                          const unsigned int                 i_packed_processed,
                                                                          const unsigned int                 i_packed_range,
                                                                          const unsigned int                 i_packed_blocking,
                                                                          const unsigned int                 i_packed_remainder,
                                                                          const unsigned int                 i_packed_width,
                                                                          const unsigned int                 i_simd_packed_width,
                                                                          const unsigned int                 i_bk,
                                                                          const unsigned int                 i_bn ) {
  unsigned int l_n = 0;
  unsigned int l_p = 0;
  unsigned int l_k = 0;
  unsigned int l_found_mul = 0;
  unsigned int l_max_reg_block = (i_n_limit - i_n_processed) * i_packed_blocking;
  unsigned int l_n_blocking = i_n_limit - i_n_processed;
  unsigned int l_vec_reg_tmp[5];
  unsigned int l_n_advancements = 0;
  unsigned int l_m_advancements = 0;
  unsigned int l_k_advancements = 0;
  unsigned int l_vnni_block_size = 4;
  unsigned int l_beta_0 = (0 != (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) ? 1 : 0;
  int l_used_column[32];

  /* Auxiliary GPRs  */
  unsigned int l_gp_reg_scratch = i_gp_reg_mapping->gp_reg_help_2;
  unsigned int l_output_bf16_mask = LIBXSMM_AARCH64_SVE_REG_P2;
  unsigned int l_output_bf16_mask_skip_odd_n = LIBXSMM_AARCH64_SVE_REG_P3;
  unsigned int l_output_bf16_mask_skip_odd_m = LIBXSMM_AARCH64_SVE_REG_P4;
  unsigned int l_input_bf16_mask = LIBXSMM_AARCH64_SVE_REG_P5;
  unsigned int l_output_bf16_mask_skip_odd_n_skip_odd_m = LIBXSMM_AARCH64_SVE_REG_P6;

  /* derive zip instructions and auxiliary sve types */
  unsigned int l_instr_zip[2] = { LIBXSMM_AARCH64_INSTR_SVE_ZIP1_V, LIBXSMM_AARCH64_INSTR_SVE_ZIP2_V };
  unsigned int l_instr_uzip[2] = { LIBXSMM_AARCH64_INSTR_SVE_UZP1_V, LIBXSMM_AARCH64_INSTR_SVE_UZP2_V };
  libxsmm_aarch64_sve_type l_type_zip = LIBXSMM_AARCH64_SVE_TYPE_D;
  libxsmm_aarch64_sve_type l_sve_type = libxsmm_generator_aarch64_get_sve_type(LIBXSMM_CAST_UCHAR(sizeof(float)));
  unsigned int l_vreg_perm_table_loadstore_c = 31;

  /* temporary vector registers used to load values to before zipping */
  l_vec_reg_tmp[0] = l_max_reg_block+0;
  l_vec_reg_tmp[1] = l_max_reg_block+1;
  l_vec_reg_tmp[2] = l_max_reg_block+2;
  l_vec_reg_tmp[3] = l_max_reg_block+3;
  l_vec_reg_tmp[4] = l_max_reg_block+4;

  LIBXSMM_ASSERT( i_packed_blocking > 0 );

  /* Adjust A and C pointers for already processed N/M  */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,  LIBXSMM_AARCH64_INSTR_GP_META_ADD,
      i_gp_reg_mapping->gp_reg_c, l_gp_reg_scratch, i_gp_reg_mapping->gp_reg_c, (1ull * i_n_processed * i_bn * i_packed_width + 1ull * i_packed_processed * i_simd_packed_width * i_bk) * i_micro_kernel_config->datatype_size_out );

  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
      i_gp_reg_mapping->gp_reg_a, l_gp_reg_scratch, i_gp_reg_mapping->gp_reg_a, (long long)i_packed_processed * i_simd_packed_width * i_bk * i_micro_kernel_config->datatype_size_in );

  /* packed loop */
  if ( i_packed_range/i_packed_blocking > 1 ) {
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_help_3, i_packed_range/i_packed_blocking );
  }

  /* reset helpers */
  for ( l_n = 0; l_n < l_n_blocking + 1; l_n++ ) {
    l_used_column[l_n] = 0;
  }
  /* Iterate over B and mark (paired) columns that are completely empty */
  for ( l_k = 0; l_k < (unsigned int)i_xgemm_desc->k/i_bk; l_k++ ) {
    unsigned int l_col_k = 0;
    for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
      unsigned int l_col_elements = i_column_idx[i_n_processed+l_n+1] - i_column_idx[i_n_processed+l_n];
      unsigned int l_cur_column = i_column_idx[i_n_processed+l_n];
      /* search for entries matching that k */
      for ( l_col_k = 0; l_col_k < l_col_elements; l_col_k++ ) {
        if ( l_k == i_row_idx[l_cur_column + l_col_k] ) {
          l_used_column[l_n] = 1;
          l_col_k = l_col_elements;
        }
      }
    }
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
    for ( l_n = 0; l_n < l_n_blocking; l_n+=2 ) {
      l_n_advancements++;
      l_m_advancements = 0;
      if ((l_used_column[l_n] == 0) && (l_used_column[l_n+1] == 0)) {
        if ((i_packed_width * l_vnni_block_size - l_m_advancements * l_vnni_block_size) > 0) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                         i_gp_reg_mapping->gp_reg_c,
                                                         l_gp_reg_scratch,
                                                         i_gp_reg_mapping->gp_reg_c,
                                                         (i_packed_width * l_vnni_block_size - l_m_advancements * l_vnni_block_size) * i_micro_kernel_config->datatype_size_out );
        }
        continue;
      }
      for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
        unsigned int l_reg0 = (l_n*i_packed_blocking) + l_p;
        unsigned int l_reg1 = ((l_n+1)*i_packed_blocking) + l_p;
        /* load 1st [2m][4n] */
        libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                              LIBXSMM_AARCH64_INSTR_SVE_LD1H_I_OFF,
                                              i_gp_reg_mapping->gp_reg_c,
                                              LIBXSMM_AARCH64_GP_REG_UNDEF,
                                              0,
                                              l_vec_reg_tmp[0],
                                              ((i_packed_remainder == 1) && (l_p == i_packed_blocking - 1)) ? l_output_bf16_mask_skip_odd_m  : l_output_bf16_mask );
        libxsmm_generator_vcvt_bf16f32_aarch64_sve( io_generated_code, l_vec_reg_tmp[0], 0);
        if ((i_packed_remainder == 1 || i_packed_remainder == 2) && (l_p == i_packed_blocking - 1)) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,  l_vec_reg_tmp[1], l_vec_reg_tmp[1], 0, l_vec_reg_tmp[1], 0, LIBXSMM_AARCH64_SVE_TYPE_S );
        } else {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                         i_gp_reg_mapping->gp_reg_c,
                                                         l_gp_reg_scratch,
                                                         i_gp_reg_mapping->gp_reg_c,
                                                         2 * l_vnni_block_size * i_micro_kernel_config->datatype_size_out );
          l_m_advancements += 2;
          /* load 2nd [2m][4n] */
          libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                LIBXSMM_AARCH64_INSTR_SVE_LD1H_I_OFF,
                                                i_gp_reg_mapping->gp_reg_c,
                                                LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                0,
                                                l_vec_reg_tmp[1],
                                                ((i_packed_remainder == 3) && (l_p == i_packed_blocking - 1)) ? l_output_bf16_mask_skip_odd_m  : l_output_bf16_mask );
          libxsmm_generator_vcvt_bf16f32_aarch64_sve( io_generated_code, l_vec_reg_tmp[1], 0);
          if ((i_packed_remainder == 0) || (l_p < i_packed_blocking - 1)) {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                           LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                           i_gp_reg_mapping->gp_reg_c,
                                                           l_gp_reg_scratch,
                                                           i_gp_reg_mapping->gp_reg_c,
                                                           2 * l_vnni_block_size * i_micro_kernel_config->datatype_size_out );
            l_m_advancements += 2;
          }
        }
        /* Perform UZIP1 -- result contains [2M][2m][2n] for first pair of columns */
        libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                                 l_instr_uzip[0],
                                                 l_vec_reg_tmp[0],
                                                 l_vec_reg_tmp[1],
                                                 0,
                                                 l_vec_reg_tmp[2],
                                                 LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                                 l_type_zip );
        /* Permute result to desired format [2M][2n][2m] */
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                                 l_vec_reg_tmp[2], l_vreg_perm_table_loadstore_c, 0, l_reg0, LIBXSMM_AARCH64_SVE_REG_P0, l_sve_type );
        /* Perform UZIP2 -- result contains [2M][2m][2n] for second pair of columns */
        libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                                 l_instr_uzip[1],
                                                 l_vec_reg_tmp[0],
                                                 l_vec_reg_tmp[1],
                                                 0,
                                                 l_vec_reg_tmp[3],
                                                 LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                                 l_type_zip );
        /* Permute result to desired format [2M][2n][2m] */
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                                 l_vec_reg_tmp[3], l_vreg_perm_table_loadstore_c, 0, l_reg1, LIBXSMM_AARCH64_SVE_REG_P0, l_sve_type );
      }
      if ((i_packed_width * l_vnni_block_size - l_m_advancements * l_vnni_block_size) > 0) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_mapping->gp_reg_c,
                                                       l_gp_reg_scratch,
                                                       i_gp_reg_mapping->gp_reg_c,
                                                       (i_packed_width * l_vnni_block_size - l_m_advancements * l_vnni_block_size) * i_micro_kernel_config->datatype_size_out );
      }
    }
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_c,
                                                   l_gp_reg_scratch,
                                                   i_gp_reg_mapping->gp_reg_c,
                                                   (i_packed_width * l_vnni_block_size * l_n_advancements) * i_micro_kernel_config->datatype_size_out );
  }

  /* do dense packed times sparse multiplication */
  l_k_advancements = 0;
  for ( l_k = 0; l_k < (unsigned int)i_xgemm_desc->k/i_bk; l_k++ ) {
    unsigned int l_col_k = 0;
    int l_nnz_idx[28][4] = { {0}, {0} };

    /* reset helpers */
    for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
      l_nnz_idx[l_n][0] = -1; l_nnz_idx[l_n][1] = -1; l_nnz_idx[l_n][2] = -1; l_nnz_idx[l_n][3] = -1;
    }
    l_found_mul = 0;

    /* loop over the columns of B/C */
    for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
      unsigned int l_col_elements = i_column_idx[i_n_processed+l_n+1] - i_column_idx[i_n_processed+l_n];
      unsigned int l_cur_column = i_column_idx[i_n_processed+l_n];
      /* search for entries matching that k */
      for ( l_col_k = 0; l_col_k < l_col_elements; l_col_k++ ) {
        if ( l_k == i_row_idx[l_cur_column + l_col_k] ) {
          l_nnz_idx[l_n][0] = l_cur_column + l_col_k;
          l_col_k = l_col_elements;
        }
      }
      /* let's check if we have an entry in the column that matches the k from A */
      if ( (l_nnz_idx[l_n][0] != -1) ) {
        l_found_mul = 1;
      }
    }

    if ( l_found_mul != 0 ) {
      /* Load A registers  */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_mapping->gp_reg_a,
                                                     l_gp_reg_scratch,
                                                     i_gp_reg_mapping->gp_reg_a,
                                                     (long long)(l_k-l_k_advancements)*i_bk*i_packed_width*i_micro_kernel_config->datatype_size_in );

      for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
        libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1H_I_OFF,
            (l_p == 0) ? i_gp_reg_mapping->gp_reg_a : l_gp_reg_scratch, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, l_max_reg_block+l_p, ((i_packed_remainder > 0) && (l_p == i_packed_blocking - 1)) ? l_input_bf16_mask : LIBXSMM_AARCH64_SVE_REG_P0 );
        if (l_p < i_packed_blocking - 1) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                         i_gp_reg_mapping->gp_reg_a,
                                                         l_gp_reg_scratch,
                                                         l_gp_reg_scratch,
                                                         (long long)(l_p+1) * i_simd_packed_width * i_bk * i_micro_kernel_config->datatype_size_in );
        }
      }
      l_k_advancements = l_k;

      /* loop over the columns of B/C */
      for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
        if ( l_nnz_idx[l_n][0] != -1 ) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                         i_gp_reg_mapping->gp_reg_b,
                                                         l_gp_reg_scratch,
                                                         l_gp_reg_scratch,
                                                         (long long)l_nnz_idx[l_n][0] * i_bk * i_bn * i_micro_kernel_config->datatype_size_in );

          libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_LD1RQD_I_OFF,
                                                l_gp_reg_scratch, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, l_max_reg_block+i_packed_blocking, LIBXSMM_AARCH64_SVE_REG_P0 );

          for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
            libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_SVE_BFMMLA_V ,
                                                   l_max_reg_block+i_packed_blocking,
                                                   l_max_reg_block+l_p,
                                                   0,
                                                   (l_n*i_packed_blocking) + l_p,
                                                   LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                                   (libxsmm_aarch64_sve_type)0 );
          }
        }
      }
    }
  }

  if (l_k_advancements > 0) {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_a,
                                                   l_gp_reg_scratch,
                                                   i_gp_reg_mapping->gp_reg_a,
                                                   (long long)l_k_advancements*i_bk*i_packed_width*i_micro_kernel_config->datatype_size_in );
  }

  /* store C accumulator */
  l_n_advancements = 0;
  for ( l_n = 0; l_n < l_n_blocking; l_n+=2 ) {
    l_n_advancements++;
    l_m_advancements = 0;

    if ((l_used_column[l_n] == 0) && (l_used_column[l_n+1] == 0) && (l_beta_0 == 0)) {
      if ((i_packed_width * l_vnni_block_size - l_m_advancements * l_vnni_block_size) > 0) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_mapping->gp_reg_c,
                                                       l_gp_reg_scratch,
                                                       i_gp_reg_mapping->gp_reg_c,
                                                       (i_packed_width * l_vnni_block_size - l_m_advancements * l_vnni_block_size) * i_micro_kernel_config->datatype_size_out );
      }
      continue;
    }

    for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
      unsigned int l_reg0 = (l_n*i_packed_blocking) + l_p;
      unsigned int l_reg1 = ((l_n+1)*i_packed_blocking) + l_p;
      unsigned int l_mask_use = 0;
      /* Permute l_reg0 */
      libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                               l_reg0, l_vreg_perm_table_loadstore_c, 0, l_vec_reg_tmp[0], LIBXSMM_AARCH64_SVE_REG_P0, l_sve_type );
      /* Permute l_reg1 */
      libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                               l_reg1, l_vreg_perm_table_loadstore_c, 0, l_vec_reg_tmp[1], LIBXSMM_AARCH64_SVE_REG_P0, l_sve_type );
      /* Zip1 */
      libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                               l_instr_zip[0],
                                               l_vec_reg_tmp[0],
                                               l_vec_reg_tmp[1],
                                               0,
                                               l_vec_reg_tmp[2],
                                               LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                               l_type_zip );
      if ((i_packed_remainder == 1) && (l_p == i_packed_blocking - 1)) {
        l_mask_use = ((l_n_blocking % 2 == 1) && (l_n+2 >= l_n_blocking)) ? l_output_bf16_mask_skip_odd_n_skip_odd_m : l_output_bf16_mask_skip_odd_m;
      } else {
        l_mask_use = ((l_n_blocking % 2 == 1) && (l_n+2 >= l_n_blocking)) ? l_output_bf16_mask_skip_odd_n: l_output_bf16_mask;
      }
      /* Cvt + store --> write [2m][4n] */
      libxsmm_generator_vcvt_f32bf16_aarch64_sve( io_generated_code, l_vec_reg_tmp[2], 0);
      libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                            LIBXSMM_AARCH64_INSTR_SVE_ST1H_I_OFF,
                                            i_gp_reg_mapping->gp_reg_c,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF,
                                            0,
                                            l_vec_reg_tmp[2],
                                            l_mask_use );
      if ((i_packed_remainder == 1 || i_packed_remainder == 2) && (l_p == i_packed_blocking - 1)) {
        /* All done for stores in m direction  */
      } else {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_mapping->gp_reg_c,
                                                       l_gp_reg_scratch,
                                                       i_gp_reg_mapping->gp_reg_c,
                                                       2 * l_vnni_block_size* i_micro_kernel_config->datatype_size_out );
        l_m_advancements += 2;
        /* Zip2 */
        libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                                 l_instr_zip[1],
                                                 l_vec_reg_tmp[0],
                                                 l_vec_reg_tmp[1],
                                                 0,
                                                 l_vec_reg_tmp[3],
                                                 LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                                 l_type_zip );

        if ((i_packed_remainder == 3) && (l_p == i_packed_blocking - 1)) {
          l_mask_use = ((l_n_blocking % 2 == 1) && (l_n+2 >= l_n_blocking)) ? l_output_bf16_mask_skip_odd_n_skip_odd_m : l_output_bf16_mask_skip_odd_m;
        } else {
          l_mask_use = ((l_n_blocking % 2 == 1) && (l_n+2 >= l_n_blocking)) ? l_output_bf16_mask_skip_odd_n: l_output_bf16_mask;
        }
        /* Cvt + store --> write [2m][4n] */
        libxsmm_generator_vcvt_f32bf16_aarch64_sve( io_generated_code, l_vec_reg_tmp[3], 0);
        libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                              LIBXSMM_AARCH64_INSTR_SVE_ST1H_I_OFF,
                                              i_gp_reg_mapping->gp_reg_c,
                                              LIBXSMM_AARCH64_GP_REG_UNDEF,
                                              0,
                                              l_vec_reg_tmp[3],
                                              l_mask_use );
        if ((i_packed_remainder == 0) || (l_p < i_packed_blocking - 1)) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                         i_gp_reg_mapping->gp_reg_c,
                                                         l_gp_reg_scratch,
                                                         i_gp_reg_mapping->gp_reg_c,
                                                         2 * l_vnni_block_size* i_micro_kernel_config->datatype_size_out );
          l_m_advancements += 2;
        }
      }
    }
    if ((i_packed_width * l_vnni_block_size - l_m_advancements * l_vnni_block_size) > 0) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_mapping->gp_reg_c,
                                                     l_gp_reg_scratch,
                                                     i_gp_reg_mapping->gp_reg_c,
                                                     (i_packed_width * l_vnni_block_size - l_m_advancements * l_vnni_block_size) * i_micro_kernel_config->datatype_size_out );
    }
  }
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                 LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 i_gp_reg_mapping->gp_reg_c,
                                                 l_gp_reg_scratch,
                                                 i_gp_reg_mapping->gp_reg_c,
                                                 (i_packed_width * l_vnni_block_size * l_n_advancements) * i_micro_kernel_config->datatype_size_out );

  /* packed loop */
  if ( i_packed_range/i_packed_blocking > 1 ) {
    /* advance A and C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                   (long long)i_packed_blocking*i_simd_packed_width*l_vnni_block_size*i_micro_kernel_config->datatype_size_in );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                   (long long)i_packed_blocking*i_simd_packed_width*l_vnni_block_size*i_micro_kernel_config->datatype_size_out );

    /* packed loop footer */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_help_3, 1 );

    /* reset A and C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                   ((long long)i_packed_range/i_packed_blocking)*i_packed_blocking*i_simd_packed_width*l_vnni_block_size*i_micro_kernel_config->datatype_size_in );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                   ((long long)i_packed_range/i_packed_blocking)*i_packed_blocking*i_simd_packed_width*l_vnni_block_size*i_micro_kernel_config->datatype_size_out );
  }

  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
      i_gp_reg_mapping->gp_reg_a, l_gp_reg_scratch, i_gp_reg_mapping->gp_reg_a, (long long)i_packed_processed * i_simd_packed_width * i_bk * i_micro_kernel_config->datatype_size_in );

  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,  LIBXSMM_AARCH64_INSTR_GP_META_SUB,
      i_gp_reg_mapping->gp_reg_c, l_gp_reg_scratch, i_gp_reg_mapping->gp_reg_c, (1ull * i_n_processed * i_bn * i_packed_width + 1ull * i_packed_processed * i_simd_packed_width * i_bk) * i_micro_kernel_config->datatype_size_out );
}


