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

#include "generator_packed_spgemm_bcsc_bsparse_avx_avx2_avx512_amx.h"
#include "generator_gemm_common.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "generator_common_x86.h"
#include "libxsmm_main.h"


LIBXSMM_API_INTERN
void libxsmm_spgemm_setup_tile( unsigned int tile_id, unsigned int n_rows, unsigned int n_cols, libxsmm_tile_config *tc) {
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
void libxsmm_generator_packed_spgemm_bcsc_bsparse_config_tiles_amx( libxsmm_generated_code*         io_generated_code,
                                                                    const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                                    unsigned int                    i_simd_packed_remainder,
                                                                    unsigned int                    i_simd_packed_iters,
                                                                    unsigned int*                   i_packed_reg_block,
                                                                    const unsigned int              i_bk,
                                                                    const unsigned int              i_bn ) {
  libxsmm_tile_config tile_config;
  unsigned int l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype) );
  unsigned int l_k_elements = i_bk/l_k_pack_factor;
  LIBXSMM_MEMZERO127(&tile_config);

  tile_config.palette_id = 1;
  if (i_simd_packed_remainder == 0) {
    if (i_packed_reg_block[0] == 2) {
      /* 0-3 C: 16m  x i_bn
         4-5 A: 16m  x i_bk
         7   B: i_bk x i_bn  */
      libxsmm_spgemm_setup_tile(0, 16, i_bn, &tile_config);
      libxsmm_spgemm_setup_tile(1, 16, i_bn, &tile_config);
      libxsmm_spgemm_setup_tile(2, 16, i_bn, &tile_config);
      libxsmm_spgemm_setup_tile(3, 16, i_bn, &tile_config);
      libxsmm_spgemm_setup_tile(4, 16, l_k_elements, &tile_config);
      libxsmm_spgemm_setup_tile(5, 16, l_k_elements, &tile_config);
      libxsmm_spgemm_setup_tile(6, l_k_elements, i_bn, &tile_config);
      libxsmm_spgemm_setup_tile(7, l_k_elements, i_bn, &tile_config);
    } else {
      /* 0-5 C: 16m  x i_bn
         6   A: 16m  x i_bk
         7   B: i_bk x i_bn  */
      libxsmm_spgemm_setup_tile(0, 16, i_bn, &tile_config);
      libxsmm_spgemm_setup_tile(1, 16, i_bn, &tile_config);
      libxsmm_spgemm_setup_tile(2, 16, i_bn, &tile_config);
      libxsmm_spgemm_setup_tile(3, 16, i_bn, &tile_config);
      libxsmm_spgemm_setup_tile(4, 16, i_bn, &tile_config);
      libxsmm_spgemm_setup_tile(5, 16, i_bn, &tile_config);
      libxsmm_spgemm_setup_tile(6, 16, l_k_elements, &tile_config);
      libxsmm_spgemm_setup_tile(7, l_k_elements, i_bn, &tile_config);

    }
  } else {
    if (i_simd_packed_iters > 1) {
      /* 0,1 C: 16m  x i_bn
         2   A: 16m  x i_bk
         3,4 C:  Rm  x i_bn
         5   A:  Rm  x i_bk
         7   B: i_bk x i_bn  */
      libxsmm_spgemm_setup_tile(0, 16, i_bn, &tile_config);
      libxsmm_spgemm_setup_tile(1, 16, i_bn, &tile_config);
      libxsmm_spgemm_setup_tile(2, 16, l_k_elements, &tile_config);
      libxsmm_spgemm_setup_tile(3, i_simd_packed_remainder, i_bn, &tile_config);
      libxsmm_spgemm_setup_tile(4, i_simd_packed_remainder, i_bn, &tile_config);
      libxsmm_spgemm_setup_tile(5, i_simd_packed_remainder, l_k_elements, &tile_config);
      libxsmm_spgemm_setup_tile(6, l_k_elements, i_bn, &tile_config);
      libxsmm_spgemm_setup_tile(7, l_k_elements, i_bn, &tile_config);
    } else {
      /* 0-5 C: Rm  x i_bn
         6   A: Rm  x i_bk
         7   B: i_bk x i_bn  */
      libxsmm_spgemm_setup_tile(0, i_simd_packed_remainder, i_bn, &tile_config);
      libxsmm_spgemm_setup_tile(1, i_simd_packed_remainder, i_bn, &tile_config);
      libxsmm_spgemm_setup_tile(2, i_simd_packed_remainder, i_bn, &tile_config);
      libxsmm_spgemm_setup_tile(3, i_simd_packed_remainder, i_bn, &tile_config);
      libxsmm_spgemm_setup_tile(4, i_simd_packed_remainder, i_bn, &tile_config);
      libxsmm_spgemm_setup_tile(5, i_simd_packed_remainder, i_bn, &tile_config);
      libxsmm_spgemm_setup_tile(6, i_simd_packed_remainder, l_k_elements, &tile_config);
      libxsmm_spgemm_setup_tile(7, l_k_elements, i_bn, &tile_config);
    }
  }

  libxsmm_x86_instruction_tile_control( io_generated_code, 0, io_generated_code->arch, LIBXSMM_X86_INSTR_LDTILECFG,
      LIBXSMM_X86_GP_REG_UNDEF, 0, &tile_config );
}

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_bcsc_bsparse_avx_avx2_avx512_amx( libxsmm_generated_code*         io_generated_code,
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
  unsigned int l_bf16_amx_kernel = 0;
  unsigned int l_output_mask = 1;
  unsigned int l_input_mask  = 2;
  unsigned int l_vnni_lo_reg_load = 31;
  unsigned int l_vnni_hi_reg_load = 30;
  unsigned int l_vnni_lo_reg_store = 29;
  unsigned int l_vnni_hi_reg_store = 28;
  unsigned int l_split_tiles = 0;

  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;

  /* select simd packing width and accumulator blocking */
  if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
    if (io_generated_code->arch == LIBXSMM_X86_AVX512_CPX) {
      l_simd_packed_width = 16;
      l_bf16_amx_kernel = 0;
      if ((i_bk != 2) || (i_bn != 2)) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BCSC_BLOCK_SIZE );
        return;
      }
    } else if (io_generated_code->arch == LIBXSMM_X86_AVX512_SPR) {
      l_simd_packed_width = 16;
      l_bf16_amx_kernel = 1;
      if ((i_bk % 2 != 0) || (i_bn % 2 != 0)) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BCSC_BLOCK_SIZE );
        return;
      }
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
      return;
    }
  } else {
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

  libxsmm_compute_equalized_blocking( l_simd_packed_iters, LIBXSMM_UPDIV(32, LIBXSMM_MAX(1,l_max_cols)), &(l_packed_reg_range[0]), &(l_packed_reg_block[0]), &(l_packed_reg_range[1]), &(l_packed_reg_block[1]) );

  /* define gp register mapping */
  libxsmm_reset_x86_gp_reg_mapping( &l_gp_reg_mapping );
  /* matching calling convention on Linux */
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
#endif
  l_gp_reg_mapping.gp_reg_mloop = LIBXSMM_X86_GP_REG_R12;
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_X86_GP_REG_R13;
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_X86_GP_REG_R14;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_X86_GP_REG_R15;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_X86_GP_REG_R11;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_3 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_4 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_5 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_lda = LIBXSMM_X86_GP_REG_R11;
  l_gp_reg_mapping.gp_reg_ldb = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_ldc = LIBXSMM_X86_GP_REG_R8;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config_fullvector( &l_micro_kernel_config, io_generated_code->arch, i_xgemm_desc, 0 );

  /* open asm */
  libxsmm_x86_instruction_open_stream_gemm( io_generated_code, &l_gp_reg_mapping, 0, i_xgemm_desc->prefetch );

  /* implementing load from struct */
  if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI & i_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI) ) {
    /* RDI holds the pointer to the strcut, so lets first move this one into R15 */
    libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, l_gp_reg_mapping.gp_reg_param_struct, l_gp_reg_mapping.gp_reg_help_1 );
    /* A pointer */
    libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 32, l_gp_reg_mapping.gp_reg_a, 0 );
    /* B pointer */
    libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 64, l_gp_reg_mapping.gp_reg_b, 0 );
    /* C pointer */
    libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 96, l_gp_reg_mapping.gp_reg_c, 0 );
    if ( i_xgemm_desc->prefetch != LIBXSMM_GEMM_PREFETCH_NONE ) {
      /* A prefetch pointer */
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 56, l_gp_reg_mapping.gp_reg_a_prefetch, 0 );
      /* B preftech pointer */
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 88, l_gp_reg_mapping.gp_reg_b_prefetch, 0 );
    }
  } else {
#if 0
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ILLEGAL_ABI );
    return;
#endif
  }

  if ((LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) &&
      ((io_generated_code->arch == LIBXSMM_X86_AVX512_CPX) || (io_generated_code->arch == LIBXSMM_X86_AVX512_SPR))) {
    unsigned int l_max_n_blocking = 6;
    unsigned int l_max_m_blocking = 2;
    unsigned int l_perm_table_vnni_lo_load[16] = { 0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30 };
    unsigned int l_perm_table_vnni_hi_load[16] = { 17, 1, 19, 3, 21, 5, 23, 7, 25, 9, 27, 11, 29, 13, 31, 15 };
    unsigned int l_perm_table_vnni_lo_store[16] = { 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23 };
    unsigned int l_perm_table_vnni_hi_store[16] = { 24, 8, 25, 9, 26, 10, 27, 11, 28, 12, 29, 13, 30, 14, 31, 15};

    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) l_perm_table_vnni_lo_load, "perm_table_vnni_lo_", l_micro_kernel_config.vector_name, l_vnni_lo_reg_load);
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) l_perm_table_vnni_hi_load, "perm_table_vnni_hi_", l_micro_kernel_config.vector_name, l_vnni_hi_reg_load);
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) l_perm_table_vnni_lo_store, "perm_table_vnni_lo_store_", l_micro_kernel_config.vector_name, l_vnni_lo_reg_store);
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) l_perm_table_vnni_hi_store, "perm_table_vnni_hi_store_", l_micro_kernel_config.vector_name, l_vnni_hi_reg_store);
    if (io_generated_code->arch == LIBXSMM_X86_AVX512_CPX) {
      l_max_n_blocking = 6;
      l_max_m_blocking = 2;
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
    } else if (io_generated_code->arch == LIBXSMM_X86_AVX512_SPR) {
      l_max_n_blocking = 6;
      l_max_m_blocking = 2;
      /* Set blocking factor decisions...  */
      if (l_simd_packed_remainder == 0) {
        if (l_simd_packed_iters == 1 ) {
          l_max_m_blocking = 1;
        }
        l_packed_reg_range[0] = l_simd_packed_iters - l_simd_packed_iters % l_max_m_blocking;
        l_packed_reg_block[0] = l_max_m_blocking;
        l_packed_reg_range[1] = l_simd_packed_iters % l_max_m_blocking;
        l_packed_reg_block[1] = l_simd_packed_iters % l_max_m_blocking;
        if (l_packed_reg_block[0] == 2) {
          l_max_n_blocking = 2;
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
        } else {
          l_max_n_blocking = 6;
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
        }
        if (l_packed_reg_block[1] == 1) {
          l_max_n_blocking = 4;
          if (l_max_cols <= l_max_n_blocking) {
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
        } else {
          l_col_reg_range[1][0] = 0;
          l_col_reg_block[1][0] = 0;
          l_col_reg_range[1][1] = 0;
          l_col_reg_block[1][1] = 0;
        }
      } else {
        if (l_simd_packed_iters > 1) {
          l_split_tiles = 1;
          l_packed_reg_range[0] = l_simd_packed_iters - 1;
          l_packed_reg_block[0] = 1;
          l_packed_reg_range[1] = 1;
          l_packed_reg_block[1] = 1;
          l_max_n_blocking = 2;
          if (l_max_cols <= l_max_n_blocking) {
            l_col_reg_range[0][0] = l_max_cols;
            l_col_reg_block[0][0] = l_max_cols;
            l_col_reg_range[0][1] = 0;
            l_col_reg_block[0][1] = 0;
            l_col_reg_range[1][0] = l_max_cols;
            l_col_reg_block[1][0] = l_max_cols;
            l_col_reg_range[1][1] = 0;
            l_col_reg_block[1][1] = 0;
          } else {
            l_col_reg_range[0][0] = l_max_cols - l_max_cols % l_max_n_blocking;
            l_col_reg_block[0][0] = l_max_n_blocking;
            l_col_reg_range[0][1] = l_max_cols % l_max_n_blocking;
            l_col_reg_block[0][1] = l_max_cols % l_max_n_blocking;
            l_col_reg_range[1][0] = l_max_cols - l_max_cols % l_max_n_blocking;
            l_col_reg_block[1][0] = l_max_n_blocking;
            l_col_reg_range[1][1] = l_max_cols % l_max_n_blocking;
            l_col_reg_block[1][1] = l_max_cols % l_max_n_blocking;
          }
        } else {
          l_packed_reg_range[0] = 1;
          l_packed_reg_block[0] = 1;
          l_packed_reg_range[1] = 0;
          l_packed_reg_block[1] = 0;
          l_max_n_blocking = 6;
          if (l_max_cols <= l_max_n_blocking) {
            l_col_reg_range[0][0] = l_max_cols;
            l_col_reg_block[0][0] = l_max_cols;
            l_col_reg_range[0][1] = 0;
            l_col_reg_block[0][1] = 0;
            l_col_reg_range[1][0] = 0;
            l_col_reg_block[1][0] = 0;
            l_col_reg_range[1][1] = 0;
            l_col_reg_block[1][1] = 0;
          } else {
            l_col_reg_range[0][0] = l_max_cols - l_max_cols % l_max_n_blocking;
            l_col_reg_block[0][0] = l_max_n_blocking;
            l_col_reg_range[0][1] = l_max_cols % l_max_n_blocking;
            l_col_reg_block[0][1] = l_max_cols % l_max_n_blocking;
            l_col_reg_range[1][0] = 0;
            l_col_reg_block[1][0] = 0;
            l_col_reg_range[1][1] = 0;
            l_col_reg_block[1][1] = 0;
          }
        }
      }
    } else {

    }

    if (io_generated_code->arch == LIBXSMM_X86_AVX512_CPX) {
      if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) {
        /* mask for M remainder  */
        if ( l_simd_packed_remainder != 0 ) {
          /* Mask for output loading/storing */
          if (l_simd_packed_remainder % (l_simd_packed_width/2) > 0) {
            libxsmm_generator_initialize_avx512_mask( io_generated_code, l_gp_reg_mapping.gp_reg_help_1, l_output_mask,
                (l_simd_packed_width/2)-l_simd_packed_remainder % (l_simd_packed_width/2), LIBXSMM_DATATYPE_F32);
          }
          /* Mask for input loading */
          if (l_simd_packed_remainder % l_simd_packed_width > 0) {
            libxsmm_generator_initialize_avx512_mask( io_generated_code, l_gp_reg_mapping.gp_reg_help_1, l_input_mask,
                l_simd_packed_width-l_simd_packed_remainder % l_simd_packed_width, LIBXSMM_DATATYPE_F32);
          }
        }
      } else {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }
    }

    if (io_generated_code->arch == LIBXSMM_X86_AVX512_SPR) {
      if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) {
        /* mask for M remainder  */
        if ( l_simd_packed_remainder != 0 ) {
          if ( l_simd_packed_remainder < (l_simd_packed_width/2) ) {
            /* Mask for output loading/storing */
            libxsmm_generator_initialize_avx512_mask( io_generated_code, l_gp_reg_mapping.gp_reg_help_1, l_output_mask,
                l_simd_packed_width - l_simd_packed_remainder, LIBXSMM_DATATYPE_F32);
          } else {
            libxsmm_generator_initialize_avx512_mask( io_generated_code, l_gp_reg_mapping.gp_reg_help_1, l_output_mask,
                l_simd_packed_width - (l_simd_packed_remainder - l_simd_packed_width/2) , LIBXSMM_DATATYPE_F32);
          }
        }
      } else {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }
    }

    if (l_bf16_amx_kernel > 0) {
      /* Configure tiles based on tile blocking decisions */
      libxsmm_generator_packed_spgemm_bcsc_bsparse_config_tiles_amx( io_generated_code,  i_xgemm_desc,
          l_simd_packed_remainder, l_simd_packed_iters, l_packed_reg_block, i_bk, i_bn );
      /* Setup LD gp regs  */
      libxsmm_x86_instruction_alu_imm(io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_lda, ((long long)i_packed_width * 4)/4);
      libxsmm_x86_instruction_alu_imm(io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_ldb, ((long long)i_bk * l_micro_kernel_config.datatype_size_in)/4);
      libxsmm_x86_instruction_alu_imm(io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_ldc, ((long long)16 * 4)/4);
      /* Allocate scratch for accumulator tiles */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, LIBXSMM_X86_GP_REG_RSP, 8*1024 );
    }

  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }

  /* m loop */
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_mloop, 0 );
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, &l_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_mloop, 1 );

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
        if (l_bf16_amx_kernel > 0) {
          libxsmm_generator_packed_spgemm_bcsc_bsparse_kloop_amx(          io_generated_code,
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
                                                                           i_bn,
                                                                           l_vnni_lo_reg_load,
                                                                           l_vnni_hi_reg_load,
                                                                           l_vnni_lo_reg_store,
                                                                           l_vnni_hi_reg_store,
                                                                           l_split_tiles );
        } else if (l_bf16_amx_kernel == 0) {
          libxsmm_generator_packed_spgemm_bcsc_bsparse_kloop_bfdot_avx512( io_generated_code,
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
                                                                           i_bn,
                                                                           l_vnni_lo_reg_load,
                                                                           l_vnni_hi_reg_load,
                                                                           l_vnni_lo_reg_store,
                                                                           l_vnni_hi_reg_store );
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
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_c,
                                   (long long)l_micro_kernel_config.datatype_size_out*i_packed_width*i_xgemm_desc->ldc);

  /* advance A pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_a,
                                   (long long)l_micro_kernel_config.datatype_size_in*i_packed_width*i_xgemm_desc->lda);

  /* close m loop */
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_cmp_instruction, l_gp_reg_mapping.gp_reg_mloop, i_xgemm_desc->m );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, l_micro_kernel_config.alu_jmp_instruction, &l_loop_label_tracker );

  if (l_bf16_amx_kernel > 0) {
    libxsmm_tile_config tile_config;
    libxsmm_x86_instruction_tile_control( io_generated_code, 0, io_generated_code->arch,
        LIBXSMM_X86_INSTR_TILERELEASE, LIBXSMM_X86_GP_REG_UNDEF, 0, &tile_config );
    /* free stack */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, LIBXSMM_X86_GP_REG_RSP, 8*1024 );
  }

  /* close asm */
  libxsmm_x86_instruction_close_stream_gemm( io_generated_code, &l_gp_reg_mapping, 0, i_xgemm_desc->prefetch );
}

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_bcsc_bsparse_kloop_bfdot_avx512(libxsmm_generated_code*            io_generated_code,
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
                                                                     const unsigned int                 i_bn,
                                                                     const unsigned int                 i_vnni_lo_reg_load,
                                                                     const unsigned int                 i_vnni_hi_reg_load,
                                                                     const unsigned int                 i_vnni_lo_reg_store,
                                                                     const unsigned int                 i_vnni_hi_reg_store ) {
  unsigned int l_n = 0;
  unsigned int l_p = 0;
  unsigned int l_k = 0;
  unsigned int l_found_mul = 0;
  unsigned int l_max_reg_block = (i_n_limit - i_n_processed) * 2 * i_packed_blocking;
  unsigned int l_n_blocking = i_n_limit - i_n_processed;
  unsigned int l_vec_reg_tmp[5];
  unsigned int l_vnni_block_size = 2;
  unsigned int l_beta_0 = (0 != (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) ? 1 : 0;
  int l_used_column[32];

  unsigned int l_output_bf16_mask = 1;
  unsigned int l_input_bf16_mask  = 2;

  /* temporary vector registers used to load values to before zipping */
  l_vec_reg_tmp[0] = l_max_reg_block+0;
  l_vec_reg_tmp[1] = l_max_reg_block+1;
  l_vec_reg_tmp[2] = l_max_reg_block+2;
  l_vec_reg_tmp[3] = l_max_reg_block+3;
  l_vec_reg_tmp[4] = l_max_reg_block+4;

  LIBXSMM_ASSERT( i_packed_blocking > 0 );

  /* packed loop */
  if ( i_packed_range/i_packed_blocking > 1 ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_help_0, 0 );
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, 1 );
  }

  /* reset helpers */
  for ( l_n = 0; l_n < 32; l_n++ ) {
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
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  i_micro_kernel_config->vxor_instruction,
                                                  i_micro_kernel_config->vector_name,
                                                  l_reg0, l_reg0, l_reg0 );
      }
    }
  } else {
    for ( l_n = 0; l_n < l_n_blocking; l_n++) {
      if (l_used_column[l_n] == 0) {
        continue;
      }
      for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
        unsigned int l_reg0 = ((l_n*2+0)*i_packed_blocking) + l_p;
        unsigned int l_reg1 = ((l_n*2+1)*i_packed_blocking) + l_p;
        /* load 1st [8m][2n] */
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMOVUPS,
            i_gp_reg_mapping->gp_reg_c,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            ((1ull * i_n_processed * i_bn * i_packed_width + 1ull * i_packed_processed * i_simd_packed_width * i_bk) +
             (1ull * i_packed_width * l_vnni_block_size * l_n) + (l_p * i_packed_blocking * i_simd_packed_width)) * (i_micro_kernel_config->datatype_size_out),
            'y', l_reg0,
            ((i_packed_remainder > 0) && (i_packed_remainder < i_simd_packed_width/2) && (l_p == i_packed_blocking - 1)) ? l_output_bf16_mask : 0, 1, 0 );
        /* up-convert bf16 to fp32 */
        libxsmm_generator_cvtbf16ps_avx2_avx512( io_generated_code, 'z', l_reg0, l_reg0 );
        if ((i_packed_remainder > 0) && (i_packed_remainder <= i_simd_packed_width/2) && (l_p == i_packed_blocking - 1)) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                    i_micro_kernel_config->vxor_instruction,
                                                    i_micro_kernel_config->vector_name,
                                                    l_reg1, l_reg1, l_reg1 );
        } else {
          /* load 2nd [8m][2n] */
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVUPS,
              i_gp_reg_mapping->gp_reg_c,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              ((1ull * i_n_processed * i_bn * i_packed_width + 1ull * i_packed_processed * i_simd_packed_width * i_bk) +
               (1ull * i_packed_width * l_vnni_block_size * l_n) + ((l_p * i_packed_blocking + 1) * i_simd_packed_width)) * (i_micro_kernel_config->datatype_size_out),
              'y', l_reg1,
              ((i_packed_remainder > i_simd_packed_width/2) && (l_p == i_packed_blocking - 1)) ? l_output_bf16_mask : 0, 1, 0 );
          /* up-convert bf16 to fp32 */
          libxsmm_generator_cvtbf16ps_avx2_avx512( io_generated_code, 'z', l_reg1, l_reg1 );
        }
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64_LD, i_micro_kernel_config->vector_name,
                                                                l_reg0, LIBXSMM_X86_VEC_REG_UNDEF, l_vec_reg_tmp[0], 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
        /* Perform vperm -- result contains [16m] for even column */
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2D, i_micro_kernel_config->vector_name,
                                                                l_reg1, i_vnni_lo_reg_load, l_reg0, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
        /* Perform vperm -- result contains [16m] for odd column  */
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2D, i_micro_kernel_config->vector_name,
                                                                l_vec_reg_tmp[0], i_vnni_hi_reg_load, l_reg1, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
      }
    }
  }

  /* do dense packed times sparse multiplication */
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
      for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMOVUPS,
            i_gp_reg_mapping->gp_reg_a,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            ((1ull * i_packed_processed * i_simd_packed_width * i_bk) + (1ull * l_k * i_bk * i_packed_width) + (l_p * i_simd_packed_width * i_bk))* i_micro_kernel_config->datatype_size_in,
            'z', l_max_reg_block+l_p,
            ((i_packed_remainder > 0) && (l_p == i_packed_blocking - 1)) ? l_input_bf16_mask : 0, 1, 0 );
      }

      /* loop over the columns of B/C */
      for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
        if ( l_nnz_idx[l_n][0] != -1 ) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPBROADCASTD,
              i_gp_reg_mapping->gp_reg_b,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (long long)l_nnz_idx[l_n][0] * i_bk * i_bn * i_micro_kernel_config->datatype_size_in,
              i_micro_kernel_config->vector_name,
              l_max_reg_block+i_packed_blocking, 0, 1, 0 );

          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPBROADCASTD,
              i_gp_reg_mapping->gp_reg_b,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              ((long long)l_nnz_idx[l_n][0] * i_bk * i_bn + i_bk) * i_micro_kernel_config->datatype_size_in,
              i_micro_kernel_config->vector_name,
              l_max_reg_block+i_packed_blocking+1, 0, 1, 0 );

          for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VDPBF16PS,
                                              i_micro_kernel_config->vector_name,
                                              l_max_reg_block+l_p,
                                              l_max_reg_block+i_packed_blocking,
                                              ((2*l_n+0)*i_packed_blocking) + l_p );

            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VDPBF16PS,
                                              i_micro_kernel_config->vector_name,
                                              l_max_reg_block+l_p,
                                              l_max_reg_block+i_packed_blocking+1,
                                              ((2*l_n+1)*i_packed_blocking) + l_p );
          }
        }
      }
    }
  }

  /* store C accumulator */
  for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
    if ((l_used_column[l_n] == 0) && (l_beta_0 == 0)) {
      continue;
    }

    for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
      unsigned int l_reg0 = ((2*l_n+0)*i_packed_blocking) + l_p;
      unsigned int l_reg1 = ((2*l_n+1)*i_packed_blocking) + l_p;
      /* Zip1 */
      libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64_LD, i_micro_kernel_config->vector_name,
                                                              l_reg0, LIBXSMM_X86_VEC_REG_UNDEF, l_vec_reg_tmp[0], 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );

      libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2D, i_micro_kernel_config->vector_name,
                                                              l_reg1, i_vnni_lo_reg_store, l_vec_reg_tmp[0], 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
      /* Cvt + store --> write [8m][2n] */
      libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, l_vec_reg_tmp[0], l_vec_reg_tmp[0] );
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VMOVUPS,
          i_gp_reg_mapping->gp_reg_c,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          ((1ull * i_n_processed * i_bn * i_packed_width + 1ull * i_packed_processed * i_simd_packed_width * i_bk) +
           (1ull * i_packed_width * l_vnni_block_size * l_n) + (l_p * i_packed_blocking * i_simd_packed_width)) * (i_micro_kernel_config->datatype_size_out),
          'y', l_vec_reg_tmp[0],
          ((i_packed_remainder > 0) && (i_packed_remainder < i_simd_packed_width/2) && (l_p == i_packed_blocking - 1)) ? l_output_bf16_mask : 0, 0, 1 );

      if ((i_packed_remainder > 0) && (i_packed_remainder <= i_simd_packed_width/2) && (l_p == i_packed_blocking - 1)) {
        /* All done for stores in m direction  */
      } else {
        /* Zip2 */
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2D, i_micro_kernel_config->vector_name,
                                                                l_reg0, i_vnni_hi_reg_store, l_reg1, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
        /* Cvt + store --> write [8m][2n] */
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, l_reg1, l_reg1 );
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMOVUPS,
            i_gp_reg_mapping->gp_reg_c,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            ((1ull * i_n_processed * i_bn * i_packed_width + 1ull * i_packed_processed * i_simd_packed_width * i_bk) +
             (1ull * i_packed_width * l_vnni_block_size * l_n) + ((l_p * i_packed_blocking + 1) * i_simd_packed_width)) * (i_micro_kernel_config->datatype_size_out),
            'y', l_reg1,
            ((i_packed_remainder > i_simd_packed_width/2) && (l_p == i_packed_blocking - 1)) ? l_output_bf16_mask : 0, 0, 1 );
      }
    }
  }

  /* packed loop */
  if ( i_packed_range/i_packed_blocking > 1 ) {
    /* advance A and C pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, (long long)i_packed_blocking*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_in );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c, (long long)i_packed_blocking*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_out );

    /* packed loop footer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_packed_range/i_packed_blocking );
    libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );

    /* reset A and C pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_a, ((long long)i_packed_range/i_packed_blocking)*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_in );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_c, ((long long)i_packed_range/i_packed_blocking)*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_out );
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_bcsc_bsparse_kloop_amx(         libxsmm_generated_code*            io_generated_code,
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
                                                                     const unsigned int                 i_bn,
                                                                     const unsigned int                 i_vnni_lo_reg_load,
                                                                     const unsigned int                 i_vnni_hi_reg_load,
                                                                     const unsigned int                 i_vnni_lo_reg_store,
                                                                     const unsigned int                 i_vnni_hi_reg_store,
                                                                     unsigned int                       i_split_tiles ) {
  unsigned int l_n = 0;
  unsigned int l_p = 0;
  unsigned int l_k = 0;
  unsigned int _l_n = 0;
  unsigned int l_n_tile = 0;
  unsigned int l_found_mul = 0;
  unsigned int l_max_reg_block = (i_n_limit - i_n_processed) * 2 * i_packed_blocking;
  unsigned int l_n_blocking = i_n_limit - i_n_processed;
  unsigned int l_vec_reg_tmp[5];
  unsigned int l_vnni_block_size = 2;
  unsigned int l_beta_0 = (0 != (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) ? 1 : 0;
  int l_used_column[32];
  unsigned int l_c_tile_offset = ((i_split_tiles > 0) && (i_packed_remainder > 0)) ? 3 : 0;
  unsigned int l_a_tile_offset = l_n_blocking * i_packed_blocking;
  unsigned int l_output_bf16_mask = 1;
  unsigned int l_input_bf16_mask  = 2;

  if ((i_split_tiles > 0) && (i_packed_remainder > 0)) {
    l_a_tile_offset = 5;
  }
  if ((i_split_tiles > 0) && (i_packed_remainder == 0)) {
    l_a_tile_offset = 2;
  }

  /* temporary vector registers used to load values to before zipping */
  l_vec_reg_tmp[0] = l_max_reg_block+0;
  l_vec_reg_tmp[1] = l_max_reg_block+1;
  l_vec_reg_tmp[2] = l_max_reg_block+2;
  l_vec_reg_tmp[3] = l_max_reg_block+3;
  l_vec_reg_tmp[4] = l_max_reg_block+4;

  LIBXSMM_ASSERT( i_packed_blocking > 0 );

  /* packed loop */
  if ( i_packed_range/i_packed_blocking > 1 ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_help_0, 0 );
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, 1 );
  }

  /* reset helpers */
  for ( l_n = 0; l_n < 32; l_n++ ) {
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
        unsigned int l_tile0 = (l_n*i_packed_blocking) + l_p + l_c_tile_offset;
        libxsmm_x86_instruction_tile_move( io_generated_code, io_generated_code->arch,
            LIBXSMM_X86_INSTR_TILEZERO, LIBXSMM_X86_GP_REG_UNDEF,  LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
            l_tile0);
      }
    }
  } else {
    for ( l_n_tile = 0; l_n_tile < l_n_blocking; l_n_tile++) {
      if (l_used_column[l_n_tile] == 0) {
        continue;
      }
      for ( _l_n = 0; _l_n < i_bn/2; _l_n++) {
        l_n = l_n_tile * (i_bn/2) + _l_n;
        for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
          unsigned int l_tile = (l_n_tile*i_packed_blocking) + l_p + l_c_tile_offset;
          unsigned int l_reg0 = 0;/* ((l_n*2+0)*i_packed_blocking) + l_p; */
          unsigned int l_reg1 = 1;/* ((l_n*2+1)*i_packed_blocking) + l_p; */
          /* load 1st [8m][2n] */
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVUPS,
              i_gp_reg_mapping->gp_reg_c,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              ((1ull * i_n_processed * i_bn * i_packed_width + 1ull * i_packed_processed * i_simd_packed_width * l_vnni_block_size) +
               (1ull * i_packed_width * l_vnni_block_size * l_n) + (l_p * i_packed_blocking * i_simd_packed_width)) * (i_micro_kernel_config->datatype_size_out),
              'y', l_reg0,
              ((i_packed_remainder > 0) && (i_packed_remainder < i_simd_packed_width/2) && (l_p == i_packed_blocking - 1)) ? l_output_bf16_mask : 0, 1, 0 );
          /* up-convert bf16 to fp32 */
          libxsmm_generator_cvtbf16ps_avx2_avx512( io_generated_code, 'z', l_reg0, l_reg0 );
          if ((i_packed_remainder > 0) && (i_packed_remainder <= i_simd_packed_width/2) && (l_p == i_packed_blocking - 1)) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                      i_micro_kernel_config->vxor_instruction,
                                                      i_micro_kernel_config->vector_name,
                                                      l_reg1, l_reg1, l_reg1 );
          } else {
            /* load 2nd [8m][2n] */
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVUPS,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((1ull * i_n_processed * i_bn * i_packed_width + 1ull * i_packed_processed * i_simd_packed_width * l_vnni_block_size) +
                 (1ull * i_packed_width * l_vnni_block_size * l_n) + ((l_p * i_packed_blocking + 1) * i_simd_packed_width)) * (i_micro_kernel_config->datatype_size_out),
                'y', l_reg1,
                ((i_packed_remainder > i_simd_packed_width/2) && (l_p == i_packed_blocking - 1)) ? l_output_bf16_mask : 0, 1, 0 );
            /* up-convert bf16 to fp32 */
            libxsmm_generator_cvtbf16ps_avx2_avx512( io_generated_code, 'z', l_reg1, l_reg1 );
          }
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64_LD, i_micro_kernel_config->vector_name,
                                                                  l_reg0, LIBXSMM_X86_VEC_REG_UNDEF, l_vec_reg_tmp[0], 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
          /* Perform vperm -- l_reg0 result contains [16m] for even column */
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2D, i_micro_kernel_config->vector_name,
                                                                  l_reg1, i_vnni_lo_reg_load, l_reg0, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
          /* Perform vperm -- l_reg_1 result contains [16m] for odd column  */
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2D, i_micro_kernel_config->vector_name,
                                                                  l_vec_reg_tmp[0], i_vnni_hi_reg_load, l_reg1, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
          /* Store the 2 columns to scratch  */
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVUPS,
              LIBXSMM_X86_GP_REG_RSP,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              l_tile * 1024 + _l_n * 2 * i_simd_packed_width * 4,
              'z', l_reg0,
              0, 0, 1 );
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVUPS,
              LIBXSMM_X86_GP_REG_RSP,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              l_tile * 1024 + (_l_n * 2 + 1) * i_simd_packed_width * 4,
              'z', l_reg1,
              0, 0, 1 );
        }
      }
      /* Load the accumulator tiles */
      for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
        unsigned int l_tile = (l_n_tile*i_packed_blocking) + l_p + l_c_tile_offset;
        libxsmm_x86_instruction_tile_move( io_generated_code,
                io_generated_code->arch,
                LIBXSMM_X86_INSTR_TILELOADD,
                LIBXSMM_X86_GP_REG_RSP,
                i_gp_reg_mapping->gp_reg_ldc,
                4,
                l_tile * 1024,
                l_tile);
      }
    }
  }

  /* do dense packed times sparse multiplication */
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
      /* Load A tile  */
      for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
        unsigned int tile_a = l_a_tile_offset + l_p;
        libxsmm_x86_instruction_tile_move( io_generated_code,
            io_generated_code->arch,
            LIBXSMM_X86_INSTR_TILELOADD,
            i_gp_reg_mapping->gp_reg_a,
            i_gp_reg_mapping->gp_reg_lda,
            4,
            ((1ull * i_packed_processed * i_simd_packed_width * l_vnni_block_size) + (1ull * l_k * i_bk * i_packed_width) + (l_p * i_simd_packed_width * l_vnni_block_size))* i_micro_kernel_config->datatype_size_in,
            tile_a);
      }

      /* loop over the columns of B/C */
      for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
        if ( l_nnz_idx[l_n][0] != -1 ) {
          unsigned int tile_b = 7;
          libxsmm_x86_instruction_tile_move( io_generated_code,
              io_generated_code->arch,
              LIBXSMM_X86_INSTR_TILELOADD,
              i_gp_reg_mapping->gp_reg_b,
              i_gp_reg_mapping->gp_reg_ldb,
              4,
              (long long)l_nnz_idx[l_n][0] * i_bk * i_bn * i_micro_kernel_config->datatype_size_in,
              tile_b);

          for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
            unsigned int tile_a = l_a_tile_offset + l_p;
            unsigned int tile_c = l_n * i_packed_blocking + l_p + l_c_tile_offset;
            libxsmm_x86_instruction_tile_compute( io_generated_code,
                io_generated_code->arch,
                LIBXSMM_X86_INSTR_TDPBF16PS,
                tile_a,
                tile_b,
                tile_c);
          }
        }
      }
    }
  }

  /* store C accumulator */
  for ( l_n_tile = 0; l_n_tile < l_n_blocking; l_n_tile++) {
    if ((l_used_column[l_n_tile] == 0) && (l_beta_0 == 0)) {
      continue;
    }

    /* Store accumulators to scratch */
    for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
      unsigned int l_tile = (l_n_tile*i_packed_blocking) + l_p + l_c_tile_offset;
      libxsmm_x86_instruction_tile_move( io_generated_code,
          io_generated_code->arch,
          LIBXSMM_X86_INSTR_TILESTORED,
          LIBXSMM_X86_GP_REG_RSP,
          i_gp_reg_mapping->gp_reg_ldc,
          4,
          l_tile * 1024,
          l_tile);
    }

    for ( _l_n = 0; _l_n < i_bn/2; _l_n++) {
      l_n = l_n_tile * (i_bn/2) + _l_n;
      for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
        unsigned int l_reg0 = 0; /* ((2*l_n+0)*i_packed_blocking) + l_p; */
        unsigned int l_reg1 = 1; /* ((2*l_n+1)*i_packed_blocking) + l_p; */

        /* Load reg0 and reg1 from scratch */
        unsigned int l_tile = (l_n_tile*i_packed_blocking) + l_p + l_c_tile_offset;

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMOVUPS,
            LIBXSMM_X86_GP_REG_RSP,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            l_tile * 1024 + _l_n * 2 * i_simd_packed_width * 4,
            'z', l_reg0,
            0, 1, 0 );

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMOVUPS,
            LIBXSMM_X86_GP_REG_RSP,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            l_tile * 1024 + (_l_n * 2 + 1) * i_simd_packed_width * 4,
            'z', l_reg1,
            0, 1, 0 );

        /* Zip1 */
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64_LD, i_micro_kernel_config->vector_name,
                                                                l_reg0, LIBXSMM_X86_VEC_REG_UNDEF, l_vec_reg_tmp[0], 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );

        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2D, i_micro_kernel_config->vector_name,
                                                                l_reg1, i_vnni_lo_reg_store, l_vec_reg_tmp[0], 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
        /* Cvt + store --> write [8m][2n] */
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, l_vec_reg_tmp[0], l_vec_reg_tmp[0] );
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMOVUPS,
            i_gp_reg_mapping->gp_reg_c,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            ((1ull * i_n_processed * i_bn * i_packed_width + 1ull * i_packed_processed * i_simd_packed_width * l_vnni_block_size) +
             (1ull * i_packed_width * l_vnni_block_size * l_n) + (l_p * i_packed_blocking * i_simd_packed_width)) * (i_micro_kernel_config->datatype_size_out),
            'y', l_vec_reg_tmp[0],
            ((i_packed_remainder > 0) && (i_packed_remainder < i_simd_packed_width/2) && (l_p == i_packed_blocking - 1)) ? l_output_bf16_mask : 0, 0, 1 );

        if ((i_packed_remainder > 0) && (i_packed_remainder <= i_simd_packed_width/2) && (l_p == i_packed_blocking - 1)) {
          /* All done for stores in m direction  */
        } else {
          /* Zip2 */
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2D, i_micro_kernel_config->vector_name,
                                                                  l_reg0, i_vnni_hi_reg_store, l_reg1, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
          /* Cvt + store --> write [8m][2n] */
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, l_reg1, l_reg1 );
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVUPS,
              i_gp_reg_mapping->gp_reg_c,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              ((1ull * i_n_processed * i_bn * i_packed_width + 1ull * i_packed_processed * i_simd_packed_width * l_vnni_block_size) +
               (1ull * i_packed_width * l_vnni_block_size * l_n) + ((l_p * i_packed_blocking + 1) * i_simd_packed_width)) * (i_micro_kernel_config->datatype_size_out),
              'y', l_reg1,
              ((i_packed_remainder > i_simd_packed_width/2) && (l_p == i_packed_blocking - 1)) ? l_output_bf16_mask : 0, 0, 1 );
        }
      }
    }
  }

  /* packed loop */
  if ( i_packed_range/i_packed_blocking > 1 ) {
    /* advance A and C pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, (long long)i_packed_blocking*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_in );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c, (long long)i_packed_blocking*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_out );

    /* packed loop footer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_packed_range/i_packed_blocking );
    libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );

    /* reset A and C pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_a, ((long long)i_packed_range/i_packed_blocking)*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_in );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_c, ((long long)i_packed_range/i_packed_blocking)*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_out );
  }
}

