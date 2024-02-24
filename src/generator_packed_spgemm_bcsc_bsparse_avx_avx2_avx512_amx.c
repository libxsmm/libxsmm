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

#include "generator_packed_spgemm_bcsc_bsparse_avx_avx2_avx512_amx.h"
#include "generator_gemm_common.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "generator_common_x86.h"
#include "libxsmm_main.h"


LIBXSMM_API_INTERN
unsigned int  libxsmm_generator_x86_packed_spgemm_bcsc_pf_dist_B(void) {
  unsigned int result = 0;
  const char *const l_env_libxsmm_x86_packed_spgemm_bcsc_pf_dist_B = getenv("LIBXSMM_X86_PACKED_SPGEMM_BCSC_PF_DIST_B");

  if ( 0 == l_env_libxsmm_x86_packed_spgemm_bcsc_pf_dist_B ) {
    result = 0;
  } else {
    if ( atoi(l_env_libxsmm_x86_packed_spgemm_bcsc_pf_dist_B) > 0 ) {
      result = atoi(l_env_libxsmm_x86_packed_spgemm_bcsc_pf_dist_B);
    }
  }
  return result;
}

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_bcsc_prefetch_B_block_in_L1(libxsmm_generated_code*     io_generated_code,
    unsigned int i_size_in_bytes,
    unsigned int i_base_reg,
    long long    i_offset_in_bytes) {
  unsigned int l_cache_lines = (i_size_in_bytes + 63)/64;
  unsigned int i;

  for (i=0; i<l_cache_lines; i++) {
    libxsmm_x86_instruction_prefetch(io_generated_code,
        LIBXSMM_X86_INSTR_PREFETCHT0,
        i_base_reg,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        ((int)i_offset_in_bytes + i * 64) );
  }
}

LIBXSMM_API_INTERN
void libxsmm_spgemm_max_mn_blocking_factors_x86(libxsmm_generated_code* io_generated_code, unsigned int i_bn, unsigned int *o_max_m_bf, unsigned int *o_max_n_bf) {
  unsigned int l_available_vregs = 32;
  unsigned int l_n_max_unroll = 0;
  unsigned int l_m_max_unroll = 0;
  if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
    l_available_vregs = 14;
  }
  l_n_max_unroll = l_available_vregs - 2;
  while (i_bn % l_n_max_unroll != 0) {
    l_n_max_unroll--;
  }
  if (l_n_max_unroll > 4) {
    while (i_bn % l_n_max_unroll != 0 || l_n_max_unroll > 4) {
      l_n_max_unroll--;
    }
  }
  l_m_max_unroll = l_available_vregs;
  while ((l_m_max_unroll * l_n_max_unroll + l_m_max_unroll + 1) > l_available_vregs) {
    l_m_max_unroll--;
  }
  *o_max_m_bf = l_m_max_unroll;
  *o_max_n_bf = l_n_max_unroll;
}

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
                                                                    const unsigned int              i_bn,
                                                                    unsigned int*                   io_a_tile_id_starts ) {
  libxsmm_tile_config tile_config;
  unsigned int l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) );
  unsigned int l_k_elements = i_bk/l_k_pack_factor;
  int l_tile_bn, l_tile_bn2;

  LIBXSMM_MEMZERO127(&tile_config);

  while (l_k_elements > 16 || ((i_bk/l_k_pack_factor) % l_k_elements != 0)) {
    l_k_elements--;
  }
  if (i_bn <= 32) {
    l_tile_bn = (i_bn <= 16) ? i_bn : 16;
    l_tile_bn2 = LIBXSMM_MIN(16, LIBXSMM_MAX(i_bn - 16, 1));
  } else {
    unsigned int l_bn_microkernel = i_bn;
    while ((l_bn_microkernel > 32) || (i_bn % l_bn_microkernel != 0)) {
      l_bn_microkernel--;
    }
    l_tile_bn = (l_bn_microkernel <= 16) ? l_bn_microkernel : 16;
    l_tile_bn2 = LIBXSMM_MIN(16, LIBXSMM_MAX(l_bn_microkernel - 16, 1));
  }

  tile_config.palette_id = 1;
  if (i_simd_packed_remainder == 0) {
    if (i_packed_reg_block[0] == 2) {
      /* 0-3 C: 16m  x i_bn
         4-5 A: 16m  x i_bk
         6-7 B: i_bk x i_bn  */
      libxsmm_spgemm_setup_tile(0, 16, l_tile_bn, &tile_config);
      libxsmm_spgemm_setup_tile(1, 16, l_tile_bn, &tile_config);
      libxsmm_spgemm_setup_tile(2, 16, l_tile_bn2, &tile_config);
      libxsmm_spgemm_setup_tile(3, 16, l_tile_bn2, &tile_config);
      libxsmm_spgemm_setup_tile(4, 16, l_k_elements, &tile_config);
      libxsmm_spgemm_setup_tile(5, 16, l_k_elements, &tile_config);
      libxsmm_spgemm_setup_tile(6, l_k_elements, l_tile_bn, &tile_config);
      libxsmm_spgemm_setup_tile(7, l_k_elements, l_tile_bn2, &tile_config);
      io_a_tile_id_starts[0] = 4;
    } else {
      /* 0-4 C: 16m  x i_bn
         5   A: 16m  x i_bk
         6-7 B: i_bk x i_bn  */
      libxsmm_spgemm_setup_tile(0, 16, l_tile_bn, &tile_config);
      libxsmm_spgemm_setup_tile(1, 16, l_tile_bn2, &tile_config);
      libxsmm_spgemm_setup_tile(2, 16, l_tile_bn, &tile_config);
      libxsmm_spgemm_setup_tile(3, 16, l_tile_bn, &tile_config);
      libxsmm_spgemm_setup_tile(4, 16, l_tile_bn, &tile_config);
      libxsmm_spgemm_setup_tile(5, 16, l_k_elements, &tile_config);
      libxsmm_spgemm_setup_tile(6, l_k_elements, l_tile_bn, &tile_config);
      libxsmm_spgemm_setup_tile(7, l_k_elements, l_tile_bn2, &tile_config);
      io_a_tile_id_starts[0] = 5;
    }
  } else {
    if (i_simd_packed_iters == 2) {
      /* 0-3 C: 16m  x i_bn / Rm x i_bn
         4-5 A: 16m  x i_bk
         6-7 B: i_bk x i_bn  */
      libxsmm_spgemm_setup_tile(0, 16, l_tile_bn, &tile_config);
      libxsmm_spgemm_setup_tile(1, i_simd_packed_remainder, l_tile_bn, &tile_config);
      libxsmm_spgemm_setup_tile(2, 16, l_tile_bn2, &tile_config);
      libxsmm_spgemm_setup_tile(3, i_simd_packed_remainder, l_tile_bn2, &tile_config);
      libxsmm_spgemm_setup_tile(4, 16, l_k_elements, &tile_config);
      libxsmm_spgemm_setup_tile(5, i_simd_packed_remainder, l_k_elements, &tile_config);
      libxsmm_spgemm_setup_tile(6, l_k_elements, l_tile_bn, &tile_config);
      libxsmm_spgemm_setup_tile(7, l_k_elements, l_tile_bn2, &tile_config);
      io_a_tile_id_starts[0] = 4;
    } else if (i_simd_packed_iters > 2) {
      /* 0,1 C: 16m  x i_bn
         2   A: 16m  x i_bk
         3,4 C:  Rm  x i_bn
         5   A:  Rm  x i_bk
         7   B: i_bk x i_bn  */
      libxsmm_spgemm_setup_tile(0, 16, l_tile_bn, &tile_config);
      libxsmm_spgemm_setup_tile(1, 16, l_tile_bn2, &tile_config);
      libxsmm_spgemm_setup_tile(2, 16, l_k_elements, &tile_config);
      libxsmm_spgemm_setup_tile(3, i_simd_packed_remainder, l_tile_bn, &tile_config);
      libxsmm_spgemm_setup_tile(4, i_simd_packed_remainder, l_tile_bn2, &tile_config);
      libxsmm_spgemm_setup_tile(5, i_simd_packed_remainder, l_k_elements, &tile_config);
      libxsmm_spgemm_setup_tile(6, l_k_elements, l_tile_bn, &tile_config);
      libxsmm_spgemm_setup_tile(7, l_k_elements, l_tile_bn2, &tile_config);
      io_a_tile_id_starts[0] = 2;
      io_a_tile_id_starts[1] = 5;
    } else {
      /* 0-5 C: Rm  x i_bn
         6   A: Rm  x i_bk
         7   B: i_bk x i_bn  */
      libxsmm_spgemm_setup_tile(0, i_simd_packed_remainder, l_tile_bn, &tile_config);
      libxsmm_spgemm_setup_tile(1, i_simd_packed_remainder, l_tile_bn2, &tile_config);
      libxsmm_spgemm_setup_tile(2, i_simd_packed_remainder, l_tile_bn, &tile_config);
      libxsmm_spgemm_setup_tile(3, i_simd_packed_remainder, l_tile_bn, &tile_config);
      libxsmm_spgemm_setup_tile(4, i_simd_packed_remainder, l_tile_bn, &tile_config);
      libxsmm_spgemm_setup_tile(5, i_simd_packed_remainder, l_k_elements, &tile_config);
      libxsmm_spgemm_setup_tile(6, l_k_elements, l_tile_bn, &tile_config);
      libxsmm_spgemm_setup_tile(7, l_k_elements, l_tile_bn2, &tile_config);
      io_a_tile_id_starts[0] = 5;
    }
  }
  if ((((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & i_xgemm_desc->flags) != 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & i_xgemm_desc->flags) == 0)) ||
      (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & i_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & i_xgemm_desc->flags) == 0)) ) {
    libxsmm_x86_instruction_tile_control( io_generated_code, 0, io_generated_code->arch, LIBXSMM_X86_INSTR_LDTILECFG,
        LIBXSMM_X86_GP_REG_UNDEF, 0, &tile_config );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_bcsc_bsparse_avx_avx2_avx512_amx( libxsmm_generated_code*         io_generated_code,
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
  unsigned int l_a_tile_id_starts[2] = {0,0};
  unsigned int l_packed_reg_block[2] = {0,0};
  unsigned int l_packed_reg_range[2] = {0,0};
  unsigned int l_is_amx_kernel = 0;
  unsigned int l_output_mask = 1;
  unsigned int l_input_mask  = 2;
  unsigned int l_scalar_mask = 3;
  unsigned int l_split_tiles = 0;
  unsigned int l_vnni_block_size = 1;
  unsigned int l_emit_compute =  ((((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & i_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & i_xgemm_desc->flags) == 0)) ||
                                  (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & i_xgemm_desc->flags) != 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & i_xgemm_desc->flags) != 0))   ) ? 1 : 0;
  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;
  unsigned int l_column_idx_gpr = LIBXSMM_X86_GP_REG_R9;
  unsigned int l_row_idx_gpr = LIBXSMM_X86_GP_REG_R10;
  unsigned int l_cur_column_gpr = LIBXSMM_X86_GP_REG_RAX;
  unsigned int l_next_column_gpr = LIBXSMM_X86_GP_REG_RBX;
  unsigned int l_dynamic_n_gpr  = LIBXSMM_X86_GP_REG_R14;
  unsigned int l_is_AT_CT_kernel = 0;
  libxsmm_jump_label_tracker l_jump_label_tracker;
  libxsmm_reset_jump_label_tracker(&l_jump_label_tracker);

  if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) > 0) {
    l_is_AT_CT_kernel = 1;
  }

  if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_B );
    return;
  }
  /* select simd packing width and accumulator blocking */
  if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
    l_vnni_block_size = 4;
    if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) == 0) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
      return;
    }
    if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
      return;
    }
    if (io_generated_code->arch >= LIBXSMM_X86_AVX512_CLX && io_generated_code->arch < LIBXSMM_X86_AVX512_SPR) {
      l_simd_packed_width = 16;
      l_is_amx_kernel = 0;
      if (i_bk % 4 != 0) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BCSC_BLOCK_SIZE );
        return;
      }
    } else if (io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) {
      l_simd_packed_width = 16;
      l_is_amx_kernel = 1;
      if (i_bk % 4 != 0) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BCSC_BLOCK_SIZE );
        return;
      }
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
      return;
    }
  } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
    l_vnni_block_size = 2;
    if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) == 0) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
      return;
    }
    if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
      return;
    }
    if (io_generated_code->arch == LIBXSMM_X86_AVX512_CPX) {
      l_simd_packed_width = 16;
      l_is_amx_kernel = 0;
      if (i_bk % 2 != 0) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BCSC_BLOCK_SIZE );
        return;
      }
    } else if (io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) {
      l_simd_packed_width = 16;
      l_is_amx_kernel = 1;
      if (i_bk % 2 != 0) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BCSC_BLOCK_SIZE );
        return;
      }
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
      return;
    }
  } else if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
    l_vnni_block_size = 1;
    if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
      return;
    }
    if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
      return;
    }
    l_simd_packed_width = 16;
    if (io_generated_code->arch < LIBXSMM_X86_AVX2) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
      return;
    }
    if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
      io_generated_code->arch = LIBXSMM_X86_AVX2;
      l_simd_packed_width = 8;
    }
    l_is_amx_kernel = 0;
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }

  if (l_is_AT_CT_kernel == 0) {
    l_simd_packed_remainder = i_packed_width % l_simd_packed_width;
    l_simd_packed_iters_full = i_packed_width/l_simd_packed_width;
    l_simd_packed_iters = ( l_simd_packed_remainder > 0 ) ? l_simd_packed_iters_full+1 : l_simd_packed_iters_full;
  } else {
    l_simd_packed_remainder = (i_bk/l_vnni_block_size) % l_simd_packed_width;
    l_simd_packed_iters_full = i_packed_width;
    l_simd_packed_iters = i_packed_width;
  }

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
  l_gp_reg_mapping.gp_reg_mloop = LIBXSMM_X86_GP_REG_R12;
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_X86_GP_REG_R13;
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_X86_GP_REG_R14;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_X86_GP_REG_R15;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_X86_GP_REG_R11;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_help_3 = l_row_idx_gpr;
  l_gp_reg_mapping.gp_reg_help_4 = l_cur_column_gpr;
  l_gp_reg_mapping.gp_reg_help_5 = l_next_column_gpr;
  l_gp_reg_mapping.gp_reg_lda = LIBXSMM_X86_GP_REG_R11;
  l_gp_reg_mapping.gp_reg_ldb = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_ldc = LIBXSMM_X86_GP_REG_R8;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config( &l_micro_kernel_config, io_generated_code->arch, i_xgemm_desc, 0 );

  /* open asm */
  libxsmm_x86_instruction_open_stream_gemm( io_generated_code, &l_gp_reg_mapping, 0, i_xgemm_desc->prefetch );

  if (l_is_amx_kernel > 0) {
    /* saving current tileconfig to the stack */
    if ( (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & i_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & i_xgemm_desc->flags) == 0)) ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, LIBXSMM_X86_GP_REG_RSP, 64 );
      libxsmm_x86_instruction_tile_control( io_generated_code, 1000, io_generated_code->arch, LIBXSMM_X86_INSTR_STTILECFG, LIBXSMM_X86_GP_REG_RSP, 0, NULL );
    /* we only set the config in this kernel */
    } else if ( (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & i_xgemm_desc->flags) != 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & i_xgemm_desc->flags) == 0)) ) {
      libxsmm_jump_label_tracker l_jump_label_tracker_tc;
      libxsmm_reset_jump_label_tracker(&l_jump_label_tracker_tc);
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_CMPQ, l_gp_reg_mapping.gp_reg_param_struct, 0 );
      libxsmm_x86_instruction_jump_to_label( io_generated_code, LIBXSMM_X86_INSTR_JE, 0, &l_jump_label_tracker_tc );
      libxsmm_x86_instruction_tile_control( io_generated_code, 1000, io_generated_code->arch, LIBXSMM_X86_INSTR_STTILECFG, l_gp_reg_mapping.gp_reg_param_struct, 0, NULL );
      libxsmm_x86_instruction_register_jump_label( io_generated_code, 0, &l_jump_label_tracker_tc );
    }
  }

  /* implementing load from struct */
  if (l_emit_compute > 0) {
    if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI & i_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI) ) {
      /* RDI holds the pointer to the struct, so lets first move this one into R15 */
      libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, l_gp_reg_mapping.gp_reg_param_struct, l_gp_reg_mapping.gp_reg_help_1 );
      /* A pointer */
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 32, l_gp_reg_mapping.gp_reg_a, 0 );
      /* B pointer */
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 64, l_gp_reg_mapping.gp_reg_b, 0 );
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 72, l_column_idx_gpr, 0 );
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 80, l_row_idx_gpr, 0 );
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 88, l_dynamic_n_gpr, 0 );
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       l_dynamic_n_gpr, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, l_dynamic_n_gpr, 0 );
      /* C pointer */
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 96, l_gp_reg_mapping.gp_reg_c, 0 );
    } else {
#if 0
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ILLEGAL_ABI );
      return;
#endif
    }
  }

  if ((LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) &&
      ((io_generated_code->arch <= LIBXSMM_X86_AVX512_CPX) || (io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR))) {
    unsigned int l_max_m_blocking, l_max_n_blocking;
    if (io_generated_code->arch <= LIBXSMM_X86_AVX512_CPX || l_is_AT_CT_kernel > 0) {
      l_is_amx_kernel = 0;
      /* Set blocking factor decisions...  */
      libxsmm_spgemm_max_mn_blocking_factors_x86(io_generated_code, i_bn, &l_max_m_blocking, &l_max_n_blocking);
      if (l_is_AT_CT_kernel > 0) {
        /* Mask for scalar loading/storing of output entries */
        libxsmm_generator_initialize_avx512_mask( io_generated_code, l_gp_reg_mapping.gp_reg_help_1, l_scalar_mask, l_simd_packed_width-1, LIBXSMM_DATATYPE_F32);
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
        }
      } else {
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
      }
    } else if (io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) {
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
      } else {
        if (l_simd_packed_iters == 2) {
          l_packed_reg_range[0] = l_simd_packed_iters;
          l_packed_reg_block[0] = l_max_m_blocking;
          l_packed_reg_range[1] = 0;
          l_packed_reg_block[1] = 0;
        } else if (l_simd_packed_iters > 2) {
          l_split_tiles = 1;
          l_packed_reg_range[0] = l_simd_packed_iters - 1;
          l_packed_reg_block[0] = 1;
          l_packed_reg_range[1] = 1;
          l_packed_reg_block[1] = 1;
        } else {
          l_packed_reg_range[0] = 1;
          l_packed_reg_block[0] = 1;
          l_packed_reg_range[1] = 0;
          l_packed_reg_block[1] = 0;
        }
      }
    } else {

    }

    if (l_emit_compute > 0) {
      if (io_generated_code->arch <= LIBXSMM_X86_AVX512_CPX || l_is_AT_CT_kernel > 0) {
        if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_I32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
          /* mask for M remainder  */
          if ( l_simd_packed_remainder != 0 ) {
            /* Mask for output loading/storing */
            libxsmm_generator_initialize_avx512_mask( io_generated_code, l_gp_reg_mapping.gp_reg_help_1, l_output_mask,
                l_simd_packed_width-l_simd_packed_remainder, LIBXSMM_DATATYPE_F32);
            /* Mask for input loading */
            libxsmm_generator_initialize_avx512_mask( io_generated_code, l_gp_reg_mapping.gp_reg_help_1, l_input_mask,
                l_simd_packed_width-l_simd_packed_remainder, LIBXSMM_DATATYPE_F32);
          }
        } else {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
          return;
        }
      }

      if (io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) {
        if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_I32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
          /* mask for M remainder  */
          if ( l_simd_packed_remainder != 0 ) {
            libxsmm_generator_initialize_avx512_mask( io_generated_code, l_gp_reg_mapping.gp_reg_help_1, l_output_mask,
                l_simd_packed_width-l_simd_packed_remainder, LIBXSMM_DATATYPE_F32);
          }
        } else {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
          return;
        }
      }
    }

    if (l_is_amx_kernel > 0) {
      /* Configure tiles based on tile blocking decisions */
      libxsmm_generator_packed_spgemm_bcsc_bsparse_config_tiles_amx( io_generated_code,  i_xgemm_desc,
          l_simd_packed_remainder, l_simd_packed_iters, l_packed_reg_block, i_bk, i_bn, l_a_tile_id_starts );
      /* Setup LD gp regs  */
      if (l_emit_compute > 0) {
        libxsmm_x86_instruction_alu_imm(io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_lda, ((long long)i_packed_width * 4)/4);
        libxsmm_x86_instruction_alu_imm(io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_ldb, ((long long)i_bk * l_micro_kernel_config.datatype_size_in)/4);
        if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
          libxsmm_x86_instruction_alu_imm(io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_ldc, ((long long)16 * 4)/4);
          /* Allocate scratch for accumulator tiles */
          libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, LIBXSMM_X86_GP_REG_RSP, 8*1024 + 128 );
        } else if (LIBXSMM_DATATYPE_I32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
          libxsmm_x86_instruction_alu_imm(io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_ldc, ((long long)i_packed_width * 4)/4);
          libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, LIBXSMM_X86_GP_REG_RSP, 128 );
        } else {
        }
      }
    }
  } else if (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) {
    unsigned int l_max_m_blocking, l_max_n_blocking;
    libxsmm_spgemm_max_mn_blocking_factors_x86(io_generated_code, i_bn, &l_max_m_blocking, &l_max_n_blocking);
    /* mask for M remainder  */
    if ( l_simd_packed_remainder != 0 ) {
      /* Mask for output loading/storing */
      if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
        l_output_mask = 15;
        l_input_mask  = 15;
        libxsmm_generator_initialize_avx_mask(io_generated_code, l_input_mask, l_simd_packed_remainder, LIBXSMM_DATATYPE_F32 );
      } else {
        libxsmm_generator_initialize_avx512_mask( io_generated_code, l_gp_reg_mapping.gp_reg_help_1, l_output_mask,
            l_simd_packed_width-l_simd_packed_remainder, LIBXSMM_DATATYPE_F32);
        /* Mask for input loading */
        libxsmm_generator_initialize_avx512_mask( io_generated_code, l_gp_reg_mapping.gp_reg_help_1, l_input_mask,
            l_simd_packed_width-l_simd_packed_remainder, LIBXSMM_DATATYPE_F32);
      }
    }

    if (l_is_AT_CT_kernel > 0) {
      /* Mask for scalar loading/storing of output entries */
      if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
        l_scalar_mask = 14;
        libxsmm_generator_initialize_avx_mask(io_generated_code, l_scalar_mask, 1, LIBXSMM_DATATYPE_F32 );
      } else {
        libxsmm_generator_initialize_avx512_mask( io_generated_code, l_gp_reg_mapping.gp_reg_help_1, l_scalar_mask, l_simd_packed_width-1, LIBXSMM_DATATYPE_F32);
      }
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
      }
    } else {
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
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }

  if (l_emit_compute > 0) {
    /* m loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_mloop, 0 );
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, &l_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_mloop, 1 );

    /* loop over packed blocks */
    while ( l_packed_done != l_simd_packed_iters ) {
      unsigned int l_packed_blocking;
      unsigned int l_packed_remainder = 0;
      if (l_packed_count > 1) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
        return;
      }
      l_packed_blocking = l_packed_reg_block[l_packed_count];

      if (l_is_AT_CT_kernel > 0) {
        l_packed_remainder = 0;
      } else {
        if ( (l_simd_packed_remainder != 0) && (l_packed_count == 0) ) {
          if ( l_packed_reg_block[1] > 0 ) {
            l_packed_remainder = 0;
          } else {
            l_packed_remainder = l_simd_packed_remainder;
          }
        } else if (l_simd_packed_remainder != 0) {
          l_packed_remainder = l_simd_packed_remainder;
        }
      }

      /* n loop */
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_nloop, 0 );
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, &l_loop_label_tracker );
      /* Load column index */
      libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVD,
                                       l_column_idx_gpr, l_gp_reg_mapping.gp_reg_nloop, 4, 0, l_cur_column_gpr, 0 );
      libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVD,
                                       l_column_idx_gpr, l_gp_reg_mapping.gp_reg_nloop, 4, 4, l_next_column_gpr, 0 );
      if (l_is_amx_kernel > 0) {
        libxsmm_generator_packed_spgemm_bcsc_bsparse_kloop_amx(          io_generated_code,
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
                                                                         i_bn,
                                                                         l_split_tiles,
                                                                         l_a_tile_id_starts );
      } else if (l_is_amx_kernel == 0) {
        libxsmm_generator_packed_spgemm_bcsc_bsparse_kloop_bfdot_avx512( io_generated_code,
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
      } else {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }

      if (l_is_AT_CT_kernel > 0) {
        /* close n loop */
        /* advance C pointer */
        libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_c,
                                         (long long) i_bn * l_micro_kernel_config.datatype_size_out );
        libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_nloop, 1 );
        libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_cmp_instruction, l_dynamic_n_gpr, l_gp_reg_mapping.gp_reg_nloop);
        libxsmm_x86_instruction_jump_back_to_label( io_generated_code, l_micro_kernel_config.alu_jmp_instruction, &l_loop_label_tracker );
        /* Adjust C pointer */
        libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_dynamic_n_gpr, l_cur_column_gpr );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, l_cur_column_gpr, (long long)i_bn * l_micro_kernel_config.datatype_size_out );
        libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_sub_instruction, l_cur_column_gpr, l_gp_reg_mapping.gp_reg_c );
      } else {
        /* close n loop */
        /* advance C pointer */
        libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_c,
                                         (long long) i_bn * i_packed_width * l_micro_kernel_config.datatype_size_out );
        libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_nloop, 1 );
        libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_cmp_instruction, l_dynamic_n_gpr, l_gp_reg_mapping.gp_reg_nloop);
        libxsmm_x86_instruction_jump_back_to_label( io_generated_code, l_micro_kernel_config.alu_jmp_instruction, &l_loop_label_tracker );
        /* Adjust C pointer */
        libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_dynamic_n_gpr, l_cur_column_gpr );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, l_cur_column_gpr, (long long)i_bn * i_packed_width * l_micro_kernel_config.datatype_size_out );
        libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_sub_instruction, l_cur_column_gpr, l_gp_reg_mapping.gp_reg_c );
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
  }

  if (l_is_amx_kernel > 0) {
    /* free stack */
    if (l_emit_compute > 0){
      if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, LIBXSMM_X86_GP_REG_RSP, 8*1024 + 128 );
      } else if (LIBXSMM_DATATYPE_I32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, LIBXSMM_X86_GP_REG_RSP, 128 );
      } else {

      }
    }
    /* restoring current tileconfig to the stack */
    if ( (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & i_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & i_xgemm_desc->flags) == 0)) ) {
      libxsmm_x86_instruction_tile_control( io_generated_code, 1001, io_generated_code->arch, LIBXSMM_X86_INSTR_LDTILECFG, LIBXSMM_X86_GP_REG_RSP, 0, NULL );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, LIBXSMM_X86_GP_REG_RSP, 64 );
    } else if ( (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & i_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & i_xgemm_desc->flags) != 0)) ) {
      libxsmm_jump_label_tracker l_jump_label_tracker_tc;
      libxsmm_reset_jump_label_tracker(&l_jump_label_tracker_tc);
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_CMPQ, l_gp_reg_mapping.gp_reg_param_struct, 0 );
      libxsmm_x86_instruction_jump_to_label( io_generated_code, LIBXSMM_X86_INSTR_JE, 0, &l_jump_label_tracker_tc );
      libxsmm_x86_instruction_tile_control( io_generated_code, 1001, io_generated_code->arch, LIBXSMM_X86_INSTR_LDTILECFG, l_gp_reg_mapping.gp_reg_param_struct, 0, NULL );
      libxsmm_x86_instruction_jump_to_label( io_generated_code, LIBXSMM_X86_INSTR_JMP, 1, &l_jump_label_tracker_tc );
      libxsmm_x86_instruction_register_jump_label( io_generated_code, 0, &l_jump_label_tracker_tc );
      libxsmm_x86_instruction_tile_control( io_generated_code, 1002, io_generated_code->arch, LIBXSMM_X86_INSTR_TILERELEASE, LIBXSMM_X86_GP_REG_UNDEF, 0, NULL );
      libxsmm_x86_instruction_register_jump_label( io_generated_code, 1, &l_jump_label_tracker_tc );
    }
  }

  /* close asm */
  libxsmm_x86_instruction_close_stream_gemm( io_generated_code, &l_gp_reg_mapping, 0, i_xgemm_desc->prefetch );
}

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_bcsc_bsparse_kloop_bfdot_avx512(libxsmm_generated_code*            io_generated_code,
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
  unsigned int l_n_in_block = 0;
  unsigned int l_n = 0;
  unsigned int l_p = 0;
  unsigned int l_max_reg_block = 0;
  unsigned int l_n_cols_kernel = 0;
  unsigned int l_beta_0 = (0 != (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) ? 1 : 0;
  unsigned int l_c_bf16 = ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ? 1 : 0;
  char l_c_vname = ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ? 'y' : i_micro_kernel_config->vector_name;
  unsigned int l_c_move_instr = ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ? LIBXSMM_X86_INSTR_VMOVDQU16 : LIBXSMM_X86_INSTR_VMOVUPS;
  unsigned int l_i8i32_kernel = (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) && LIBXSMM_DATATYPE_I32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ? 1 : 0;
  unsigned int l_is_s8u8s32_kernel = ((l_i8i32_kernel > 0) && ((LIBXSMM_GEMM_FLAG_A_UNSIGNED & i_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_B_UNSIGNED & i_xgemm_desc->flags) > 0) ) ? 1 : 0;
  unsigned int l_c_bcast_instr = 0;
  unsigned int l_output_bf16_mask = 1;
  unsigned int l_input_bf16_mask  = 2;
  unsigned int l_scalar_mask  = 3;
  unsigned int l_row_idx_gpr = i_gp_reg_mapping->gp_reg_help_3;
  unsigned int l_cur_column_gpr = i_gp_reg_mapping->gp_reg_help_4;
  unsigned int l_next_column_gpr = i_gp_reg_mapping->gp_reg_help_5;
  unsigned int l_fma_iters = (i_bk+1)/2;
  unsigned int l_fma_i = 0;
  unsigned int l_fma_instr = LIBXSMM_X86_INSTR_VDPBF16PS;
  unsigned int EMPTY_BLOCK_COLUMN_LABEL = (i_packed_processed == 0) ? 0 : 1;
  unsigned int EMPTY_BLOCK_COLUMN_LABEL_BETA0 = 0;
  unsigned int l_vnni_block_size = 2;
  unsigned int l_n_blocks = 0, l_n_block_id = 0;
  unsigned int l_k_loop_fma_reg = i_gp_reg_mapping->gp_reg_mloop;
  unsigned int l_assm_fma_iters = 1;
  unsigned int l_is_AT_CT_kernel = 0;
  unsigned int l_simd_packed_remainder = 0;
  unsigned int l_use_short_vec = 0;
  unsigned int l_AT_CT_vname = i_micro_kernel_config->vector_name;
  unsigned int l_AT_CT_ab_vmove_instr = LIBXSMM_X86_INSTR_VMOVUPS;
  unsigned int l_pf_dist_B = libxsmm_generator_x86_packed_spgemm_bcsc_pf_dist_B();

  if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
    l_scalar_mask = 14;
  }

  if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
    l_vnni_block_size = 1;
  } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
    l_vnni_block_size = 4;
  }

  if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) > 0) {
    l_is_AT_CT_kernel = 1;
    l_simd_packed_remainder = (i_bk/l_vnni_block_size) % i_simd_packed_width;
  }

  if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
    if (l_is_AT_CT_kernel > 0) {
      l_c_bcast_instr = LIBXSMM_X86_INSTR_VBROADCASTSS;
      l_fma_iters = (i_bk + i_simd_packed_width - 1)/i_simd_packed_width;
    } else {
      l_fma_iters = i_bk;
    }
    l_fma_instr = LIBXSMM_X86_INSTR_VFMADD231PS;
  } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
    if (l_is_AT_CT_kernel > 0) {
      l_c_bcast_instr = LIBXSMM_X86_INSTR_VBROADCASTSS;
      l_fma_iters = ((i_bk+3)/4 + i_simd_packed_width - 1)/i_simd_packed_width;
    } else {
      l_fma_iters = (i_bk+3)/4;
    }
    l_fma_instr = LIBXSMM_X86_INSTR_VPDPBUSD;
  } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
    if (l_is_AT_CT_kernel > 0) {
     l_c_bcast_instr = LIBXSMM_X86_INSTR_VPBROADCASTW;
     l_fma_iters = ((i_bk+1)/2 + i_simd_packed_width - 1)/i_simd_packed_width;
    } else {
      l_fma_iters = (i_bk+1)/2;
    }
    l_fma_instr = LIBXSMM_X86_INSTR_VDPBF16PS;
  }

  if (l_is_AT_CT_kernel > 0) {
    if (l_simd_packed_remainder == i_simd_packed_width/2) {
      l_use_short_vec = 1;
      l_AT_CT_vname = (l_AT_CT_vname == 'z') ? 'y' : ((l_AT_CT_vname == 'y' ? 'x' : 'x'));
    } else if (l_simd_packed_remainder == i_simd_packed_width/4) {
      l_use_short_vec = 1;
      l_AT_CT_vname = (l_AT_CT_vname == 'z') ? 'x' : ((l_AT_CT_vname == 'y' ? 'x' : 'x'));
    } else if (l_simd_packed_remainder == i_simd_packed_width/8) {
      l_use_short_vec = 1;
      l_AT_CT_vname = (l_AT_CT_vname == 'z') ? 'x' : ((l_AT_CT_vname == 'y' ? 'x' : 'x'));
      l_AT_CT_ab_vmove_instr = (l_simd_packed_remainder == 2) ? LIBXSMM_X86_INSTR_VMOVSD : LIBXSMM_X86_INSTR_VMOVSS;
    } else if ((l_simd_packed_remainder == i_simd_packed_width/16) && (l_simd_packed_remainder > 0)) {
      l_use_short_vec = 1;
      l_AT_CT_vname = (l_AT_CT_vname == 'z') ? 'x' : ((l_AT_CT_vname == 'y' ? 'x' : 'x'));
      l_AT_CT_ab_vmove_instr = LIBXSMM_X86_INSTR_VMOVSS;
    } else {
      /* Do nothing */
    }
  }

  if (l_is_AT_CT_kernel == 0) {
    if (l_fma_iters > 4) {
      l_assm_fma_iters = l_fma_iters;
      l_fma_iters = 4;
      while (l_assm_fma_iters % l_fma_iters != 0) {
        l_fma_iters--;
      }
      l_assm_fma_iters = l_assm_fma_iters/l_fma_iters;
    }
  }

  libxsmm_spgemm_max_mn_blocking_factors_x86(io_generated_code, i_bn, &l_n, &l_n_cols_kernel);
  l_max_reg_block = i_packed_blocking * l_n_cols_kernel;
  l_n_blocks = i_bn / l_n_cols_kernel;
  EMPTY_BLOCK_COLUMN_LABEL_BETA0 = (i_packed_processed == 0) ? 2 : 2 + l_n_blocks;

  if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
    l_input_bf16_mask = 15;
    l_output_bf16_mask = 15;
  }

  LIBXSMM_ASSERT( i_packed_blocking > 0 );

  libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_kloop );

  if (l_beta_0 == 0) {
    /* Check if empty B column and beta == 1 and jump at the end of the kernel */
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, l_cur_column_gpr, l_next_column_gpr );
    libxsmm_x86_instruction_jump_to_label(io_generated_code, LIBXSMM_X86_INSTR_JE, EMPTY_BLOCK_COLUMN_LABEL, i_jump_label_tracker);
  }

  /* packed loop */
  if ( i_packed_range/i_packed_blocking > 1 ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_help_0, 0 );
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, 1 );
    libxsmm_x86_instruction_push_reg( io_generated_code, l_cur_column_gpr );
  }

  for (l_n_block_id = 0; l_n_block_id < l_n_blocks; l_n_block_id++) {
    if (l_n_blocks > 1) {
      libxsmm_x86_instruction_push_reg( io_generated_code, l_cur_column_gpr );
    }
    /* load C accumulator */
    if (l_beta_0 > 0) {
      for ( l_n = 0; l_n < l_n_cols_kernel; l_n++ ) {
        for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
          unsigned int l_reg0 = (l_n*i_packed_blocking) + l_p;
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                    i_micro_kernel_config->vxor_instruction,
                                                    i_micro_kernel_config->vector_name,
                                                    l_reg0, l_reg0, l_reg0 );
        }
      }
    } else {
      if (l_is_AT_CT_kernel > 0) {
        for ( l_n = 0; l_n < l_n_cols_kernel; l_n++ ) {
          for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
            unsigned int l_reg0 = (l_n*i_packed_blocking) + l_p;
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                      i_micro_kernel_config->vxor_instruction,
                                                      i_micro_kernel_config->vector_name,
                                                      l_reg0, l_reg0, l_reg0 );
          }
        }
      } else {
        for ( l_n = 0; l_n < l_n_cols_kernel; l_n++ ) {
          for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
            unsigned int l_reg0 = (l_n*i_packed_blocking) + l_p;
            /* load [16m] for even column */
            libxsmm_x86_instruction_unified_vec_move( io_generated_code,
              l_c_move_instr,
              i_gp_reg_mapping->gp_reg_c,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              ((1ull * i_packed_processed * i_simd_packed_width) +
               (1ull * i_packed_width * l_n) + (l_p * i_simd_packed_width) + (l_n_block_id * l_n_cols_kernel * i_packed_width)) * (i_micro_kernel_config->datatype_size_out),
              l_c_vname, l_reg0,
              ((i_packed_remainder > 0) && (l_p == i_packed_blocking - 1)) ? 1 : 0,
              ((i_packed_remainder > 0) && (l_p == i_packed_blocking - 1)) ? l_output_bf16_mask : 0, 0 );
            /* up-convert bf16 to fp32 */
            if (l_c_bf16 > 0) {
              libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, 'z', l_reg0, l_reg0 );
            }
          }
        }
      }
    }

    if (l_beta_0 > 0) {
      /* Check if empty B column and beta == 0 and jump at the C store part of the kernel */
      libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, l_cur_column_gpr, l_next_column_gpr);
      libxsmm_x86_instruction_jump_to_label(io_generated_code, LIBXSMM_X86_INSTR_JE, EMPTY_BLOCK_COLUMN_LABEL_BETA0, i_jump_label_tracker);
    }

    /* k loop header */
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVD, l_row_idx_gpr, l_cur_column_gpr, 4, 0, i_gp_reg_mapping->gp_reg_kloop, 0 );

    if (l_is_AT_CT_kernel > 0) {
      /* Prep reg_a with "k" offset */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_kloop, (long long)i_bk * i_micro_kernel_config->datatype_size_in);
      libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_kloop);
    } else {
      /* Prep reg_a with "k" offset */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_kloop, (long long)i_bk * i_packed_width * i_micro_kernel_config->datatype_size_in);
      libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_kloop);
    }
    /* Prep reg_b with "k" offset */
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, l_cur_column_gpr, i_gp_reg_mapping->gp_reg_help_2);
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_help_2, i_bk * i_bn  * i_micro_kernel_config->datatype_size_in);
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_2);

    if (l_is_AT_CT_kernel > 0) {
      for (l_fma_i = 0; l_fma_i < l_fma_iters; l_fma_i++) {
        /* Load A registers  */
        for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            l_AT_CT_ab_vmove_instr,
            i_gp_reg_mapping->gp_reg_kloop,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            ((1ull * i_packed_processed * i_xgemm_desc->lda) + l_fma_i * l_vnni_block_size * i_simd_packed_width + (l_p * i_xgemm_desc->lda))* i_micro_kernel_config->datatype_size_in,
            l_AT_CT_vname, l_max_reg_block+l_p,
            ((l_simd_packed_remainder > 0) && (l_fma_i == l_fma_iters - 1) && (l_use_short_vec == 0)) ? 1 : 0,
            ((l_simd_packed_remainder > 0) && (l_fma_i == l_fma_iters - 1) && (l_use_short_vec == 0)) ? l_input_bf16_mask : 0, 0 );
        }

        /* Load B registers and perform the fmas  */
        for ( l_n_in_block = 0; l_n_in_block < l_n_cols_kernel; l_n_in_block++ ) {
          l_n = l_n_in_block;

          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            l_AT_CT_ab_vmove_instr,
            i_gp_reg_mapping->gp_reg_help_2,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            ((long long) i_bk * l_n_in_block + (long long)l_fma_i * l_vnni_block_size * i_simd_packed_width + (long long) l_n_block_id * l_n_cols_kernel * i_bk) * i_micro_kernel_config->datatype_size_in,
            l_AT_CT_vname, l_max_reg_block+i_packed_blocking,
            ((l_simd_packed_remainder > 0) && (l_fma_i == l_fma_iters - 1) && (l_use_short_vec == 0)) ? 1 : 0,
            ((l_simd_packed_remainder > 0) && (l_fma_i == l_fma_iters - 1) && (l_use_short_vec == 0)) ? l_input_bf16_mask : 0, 0 );

          if (l_pf_dist_B > 0) {
            if (l_fma_i == 0 && l_n_block_id == 0 && l_n_in_block == 0) {
             libxsmm_generator_packed_spgemm_bcsc_prefetch_B_block_in_L1( io_generated_code, i_bk*i_bn*i_micro_kernel_config->datatype_size_in, i_gp_reg_mapping->gp_reg_help_2, (long long)l_pf_dist_B*i_bk*i_bn*i_micro_kernel_config->datatype_size_in);
            }
          }

          for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              l_fma_instr,
                                              l_AT_CT_vname,
                                              (l_is_s8u8s32_kernel > 0) ? l_max_reg_block+l_p : l_max_reg_block+i_packed_blocking,
                                              (l_is_s8u8s32_kernel > 0) ? l_max_reg_block+i_packed_blocking : l_max_reg_block+l_p,
                                              (l_n * i_packed_blocking) + l_p );
          }
        }
      }
    } else {
      if ( l_assm_fma_iters > 1 ) {
        libxsmm_x86_instruction_push_reg( io_generated_code, l_k_loop_fma_reg );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, l_k_loop_fma_reg, 0 );
        libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, l_k_loop_fma_reg, 1 );
      }

      for (l_fma_i = 0; l_fma_i < l_fma_iters; l_fma_i++) {
        /* Load A registers  */
        for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            LIBXSMM_X86_INSTR_VMOVUPS,
            i_gp_reg_mapping->gp_reg_kloop,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            ((1ull * i_packed_processed * i_simd_packed_width * l_vnni_block_size) + l_fma_i * i_packed_width * l_vnni_block_size + (l_p * i_simd_packed_width * l_vnni_block_size))* i_micro_kernel_config->datatype_size_in,
            i_micro_kernel_config->vector_name, l_max_reg_block+l_p,
            ((i_packed_remainder > 0) && (l_p == i_packed_blocking - 1)) ? 1 : 0,
            ((i_packed_remainder > 0) && (l_p == i_packed_blocking - 1)) ? l_input_bf16_mask : 0, 0 );
        }

        /* Load B registers and perform the fmas  */
        for ( l_n_in_block = 0; l_n_in_block < l_n_cols_kernel; l_n_in_block++ ) {
          l_n = l_n_in_block;
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VBROADCASTSS,
              i_gp_reg_mapping->gp_reg_help_2,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              ((long long) i_bk * l_n_in_block + (long long)l_fma_i * l_vnni_block_size + (long long)l_n_block_id * l_n_cols_kernel * i_bk) * i_micro_kernel_config->datatype_size_in,
              i_micro_kernel_config->vector_name,
              l_max_reg_block+i_packed_blocking, 0, 1, 0 );

          if (l_pf_dist_B > 0) {
            if (l_fma_i == 0 && l_n_block_id == 0 && l_n_in_block == 0) {
             libxsmm_generator_packed_spgemm_bcsc_prefetch_B_block_in_L1( io_generated_code, i_bk*i_bn*i_micro_kernel_config->datatype_size_in, i_gp_reg_mapping->gp_reg_help_2, (long long)l_pf_dist_B*i_bk*i_bn*i_micro_kernel_config->datatype_size_in);
            }
          }

          for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              l_fma_instr,
                                              i_micro_kernel_config->vector_name,
                                              (l_is_s8u8s32_kernel > 0) ? l_max_reg_block+l_p : l_max_reg_block+i_packed_blocking,
                                              (l_is_s8u8s32_kernel > 0) ? l_max_reg_block+i_packed_blocking : l_max_reg_block+l_p,
                                              (l_n * i_packed_blocking) + l_p );
          }
        }
      }
      if ( l_assm_fma_iters > 1 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_2, (long long)l_fma_iters * l_vnni_block_size * i_micro_kernel_config->datatype_size_in);
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_kloop, (long long)l_fma_iters * i_packed_width * l_vnni_block_size * i_micro_kernel_config->datatype_size_in);
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, l_k_loop_fma_reg, (long long)l_assm_fma_iters );
        libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker);
        libxsmm_x86_instruction_pop_reg( io_generated_code, l_k_loop_fma_reg );
      }

    }

    /* k loop footer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, l_cur_column_gpr, 1 );
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, l_next_column_gpr, l_cur_column_gpr);
    libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker);

    if (l_is_AT_CT_kernel > 0) {
      /* Horizontal reduction of accumulators + adding original C value if beta = 1 */
      for ( l_n = 0; l_n < l_n_cols_kernel; l_n++ ) {
        for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
          unsigned int l_reg0 = (l_n*i_packed_blocking) + l_p;
          unsigned int l_reduction_instr = (LIBXSMM_DATATYPE_I32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) ? LIBXSMM_X86_INSTR_VPADDD : LIBXSMM_X86_INSTR_VADDPS  ;

          if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
            libxsmm_generator_hinstrps_avx( io_generated_code, l_reduction_instr, l_reg0, l_max_reg_block+1, l_max_reg_block);
          } else {
            libxsmm_generator_hinstrps_avx512( io_generated_code, l_reduction_instr, l_reg0, l_max_reg_block+1, l_max_reg_block);
          }

          if (l_beta_0 == 0) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                l_c_bcast_instr,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((1ull * i_packed_processed * i_xgemm_desc->ldc) +
                 (1ull * l_n) + (l_p * i_xgemm_desc->ldc) + (l_n_block_id * l_n_cols_kernel)) * (i_micro_kernel_config->datatype_size_out),
                l_c_vname,
                l_max_reg_block+i_packed_blocking, 0, 1, 0 );
            if (l_c_bf16 > 0) {
              libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, l_c_vname, l_max_reg_block+i_packed_blocking, l_max_reg_block+i_packed_blocking, 16);
            }
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, l_reduction_instr, l_c_vname, l_reg0, l_max_reg_block+i_packed_blocking, l_reg0);
          }

        }
      }
    }

    if (l_beta_0 > 0) {
      /* LABEL for empty column and beta == 0 */
      libxsmm_x86_instruction_register_jump_label(io_generated_code, EMPTY_BLOCK_COLUMN_LABEL_BETA0, i_jump_label_tracker);
    }

    /* store C accumulator */
    if (l_is_AT_CT_kernel > 0) {
      for ( l_n = 0; l_n < l_n_cols_kernel; l_n++ ) {
        for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
          unsigned int l_reg0 = (l_n*i_packed_blocking) + l_p;
          /* Cvt + store --> write even [16m] */
          if (l_c_bf16 > 0) {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, l_reg0, l_reg0 );
          }
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            l_c_move_instr,
            i_gp_reg_mapping->gp_reg_c,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            ((1ull * i_packed_processed * i_xgemm_desc->ldc) +
             (1ull * l_n) + (l_p * i_xgemm_desc->ldc) + (l_n_block_id * l_n_cols_kernel)) * (i_micro_kernel_config->datatype_size_out),
            l_c_vname, l_reg0,
            1,
            l_scalar_mask, 1 );
        }
      }
    } else {
      for ( l_n = 0; l_n < l_n_cols_kernel; l_n++ ) {
        for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
          unsigned int l_reg0 = (l_n*i_packed_blocking) + l_p;
          /* Cvt + store --> write even [16m] */
          if (l_c_bf16 > 0) {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, l_reg0, l_reg0 );
          }
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            l_c_move_instr,
            i_gp_reg_mapping->gp_reg_c,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            ((1ull * i_packed_processed * i_simd_packed_width) +
             (1ull * i_packed_width * l_n) + (l_p * i_simd_packed_width) + (l_n_block_id * l_n_cols_kernel * i_packed_width)) * (i_micro_kernel_config->datatype_size_out),
            l_c_vname, l_reg0,
            ((i_packed_remainder > 0) && (l_p == i_packed_blocking - 1)) ? 1 : 0,
            ((i_packed_remainder > 0) && (l_p == i_packed_blocking - 1)) ? l_output_bf16_mask : 0, 1 );
        }
      }
    }
    EMPTY_BLOCK_COLUMN_LABEL_BETA0++;
    if (l_n_blocks > 1) {
      libxsmm_x86_instruction_pop_reg( io_generated_code, l_cur_column_gpr );
    }
  }

  /* packed loop */
  if ( i_packed_range/i_packed_blocking > 1 ) {
    if (l_is_AT_CT_kernel > 0) {
      /* advance A and C pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, (long long)i_packed_blocking*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c, (long long)i_packed_blocking*i_xgemm_desc->ldc*i_micro_kernel_config->datatype_size_out );

      /* packed loop footer */
      libxsmm_x86_instruction_pop_reg( io_generated_code, l_cur_column_gpr );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_packed_range/i_packed_blocking );
      libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );

      /* reset A and C pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_a, ((long long)i_packed_range/i_packed_blocking)*i_packed_blocking*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_c, ((long long)i_packed_range/i_packed_blocking)*i_packed_blocking*i_xgemm_desc->ldc*i_micro_kernel_config->datatype_size_out );
    } else {
      /* advance A and C pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, (long long)i_packed_blocking*i_simd_packed_width*l_vnni_block_size*i_micro_kernel_config->datatype_size_in );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c, (long long)i_packed_blocking*i_simd_packed_width*i_micro_kernel_config->datatype_size_out );

      /* packed loop footer */
      libxsmm_x86_instruction_pop_reg( io_generated_code, l_cur_column_gpr );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_packed_range/i_packed_blocking );
      libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );

      /* reset A and C pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_a, ((long long)i_packed_range/i_packed_blocking)*i_packed_blocking*i_simd_packed_width*l_vnni_block_size*i_micro_kernel_config->datatype_size_in );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_c, ((long long)i_packed_range/i_packed_blocking)*i_packed_blocking*i_simd_packed_width*i_micro_kernel_config->datatype_size_out );
    }
  }

  if (l_beta_0 == 0) {
    /* LABEL for empty column and beta == 1 */
    libxsmm_x86_instruction_register_jump_label(io_generated_code, EMPTY_BLOCK_COLUMN_LABEL, i_jump_label_tracker);
  }

  libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_kloop );
}

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_bcsc_bsparse_kloop_amx(         libxsmm_generated_code*            io_generated_code,
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
                                                                     const unsigned int                 i_bn,
                                                                     unsigned int                       i_split_tiles,
                                                                     unsigned int*                      i_a_tile_id_starts ) {
  unsigned int l_n = 0;
  unsigned int l_p = 0;
  unsigned int l_n_tile = 0;
  unsigned int l_vnni_block_size = 2;
  unsigned int l_beta_0 = (0 != (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) ? 1 : 0;
  unsigned int l_c_tile_offset = ((i_split_tiles > 0) && (i_packed_remainder > 0)) ? 3 : 0;
  unsigned int l_output_bf16_mask = 1;
  unsigned int l_n_tiles_bn = (i_bn <= 16) ? 1 : 2;
  int l_n_cols0 = (l_n_tiles_bn == 1) ? i_bn : 16;
  int l_n_cols1 = (l_n_tiles_bn == 2) ? i_bn - 16 : 0;
  unsigned int l_a_tile_offset = l_n_tiles_bn * i_packed_blocking;
  unsigned int l_b_tmp_gpr = i_gp_reg_mapping->gp_reg_ldc;
  unsigned int l_row_idx_gpr = i_gp_reg_mapping->gp_reg_help_3;
  unsigned int l_cur_column_gpr = i_gp_reg_mapping->gp_reg_help_4;
  unsigned int l_next_column_gpr = i_gp_reg_mapping->gp_reg_help_5;
  unsigned int EMPTY_BLOCK_COLUMN_LABEL = (i_packed_processed == 0) ? 0 : 1;
  unsigned int EMPTY_BLOCK_COLUMN_LABEL_BETA0 = 0;
  unsigned int l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) );
  unsigned int l_k_elements = i_bk/l_k_pack_factor;
  unsigned int l_k_unroll_iters = 0, l_k_iter = 0;
  unsigned int l_bn_microkernel = i_bn;
  unsigned int l_bn_iters = 0, l_n_block_id = 0;
  unsigned int l_pf_dist_B = libxsmm_generator_x86_packed_spgemm_bcsc_pf_dist_B();

  while ((l_bn_microkernel > 32) || (i_bn % l_bn_microkernel != 0)) {
    l_bn_microkernel--;
  }
  if (l_bn_microkernel != i_bn) {
    l_n_tiles_bn = (l_bn_microkernel <= 16) ? 1 : 2;
    l_n_cols0 = (l_n_tiles_bn == 1) ? l_bn_microkernel : 16;
    l_n_cols1 = (l_n_tiles_bn == 2) ? l_bn_microkernel - 16 : 0;
  }

  l_bn_iters = i_bn / l_bn_microkernel;
  EMPTY_BLOCK_COLUMN_LABEL_BETA0 = (i_packed_processed == 0) ? 2 : 2 + l_bn_iters;

  while (l_k_elements > 16 || ((i_bk/l_k_pack_factor) % l_k_elements != 0)) {
    l_k_elements--;
  }
  l_k_unroll_iters = (i_bk/l_k_pack_factor)/l_k_elements;

  if (i_split_tiles > 0) {
    l_a_tile_offset = i_a_tile_id_starts[(i_packed_remainder == 0) ? 0 : 1];
  } else {
    l_a_tile_offset = i_a_tile_id_starts[0];
  }

  LIBXSMM_ASSERT( i_packed_blocking > 0 );

  if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
    l_vnni_block_size = 4;
  }

  if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
    libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 8 * 1024, i_gp_reg_mapping->gp_reg_kloop, 1 );
  } else if ( LIBXSMM_DATATYPE_I32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
    libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, i_gp_reg_mapping->gp_reg_kloop, 1 );
  } else {

  }
  if (l_beta_0 == 0) {
    /* Check if empty B column and beta == 1 and jump at the end of the kernel */
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, l_cur_column_gpr, l_next_column_gpr );
    libxsmm_x86_instruction_jump_to_label(io_generated_code, LIBXSMM_X86_INSTR_JE, EMPTY_BLOCK_COLUMN_LABEL, i_jump_label_tracker);
  }

  /* packed loop */
  if ( i_packed_range/i_packed_blocking > 1 ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_help_0, 0 );
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, 1 );
    if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
      libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 8 * 1024 + 64, l_cur_column_gpr, 1 );
    } else if ( LIBXSMM_DATATYPE_I32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
      libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 64, l_cur_column_gpr, 1 );
    } else {
    }
  }

  for (l_n_block_id = 0; l_n_block_id < l_bn_iters; l_n_block_id++) {
    if (l_bn_iters > 1) {
      if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
        libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 8 * 1024 + 64, l_cur_column_gpr, 1 );
      } else if ( LIBXSMM_DATATYPE_I32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
        libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 64, l_cur_column_gpr, 1 );
      } else {
      }
    }

    /* load C accumulator */
    if (l_beta_0 > 0) {
      for ( l_n = 0; l_n < l_n_tiles_bn; l_n++ ) {
        for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
          unsigned int l_tile0 = (l_n*i_packed_blocking) + l_p + l_c_tile_offset;
          libxsmm_x86_instruction_tile_move( io_generated_code, io_generated_code->arch,
              LIBXSMM_X86_INSTR_TILEZERO, LIBXSMM_X86_GP_REG_UNDEF,  LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
              l_tile0);
        }
      }
    } else {
      if ( LIBXSMM_DATATYPE_I32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
        for ( l_n_tile = 0; l_n_tile < l_n_tiles_bn; l_n_tile++) {
          for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
            unsigned int l_tile = (l_n_tile*i_packed_blocking) + l_p + l_c_tile_offset;
            libxsmm_x86_instruction_tile_move( io_generated_code,
                    io_generated_code->arch,
                    LIBXSMM_X86_INSTR_TILELOADD,
                    i_gp_reg_mapping->gp_reg_c,
                    i_gp_reg_mapping->gp_reg_ldc,
                    4,
                    ((1ull * i_packed_processed * i_simd_packed_width) +
                    (1ull * i_packed_width * l_n_tile * 16) + (l_p * i_simd_packed_width) + (l_n_block_id * l_bn_microkernel * i_packed_width)) * (i_micro_kernel_config->datatype_size_out),
                    l_tile);
          }
        }
      } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
        for ( l_n_tile = 0; l_n_tile < l_n_tiles_bn; l_n_tile++) {
          unsigned int cur_tile_cols = (l_n_tile == 0) ? l_n_cols0 : l_n_cols1 ;
          for ( l_n = 0; l_n < cur_tile_cols; l_n++) {
            for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
              unsigned int l_tile = (l_n_tile*i_packed_blocking) + l_p + l_c_tile_offset;
              unsigned int l_reg0 = l_n;
              /* load column */
              libxsmm_x86_instruction_vec_move( io_generated_code,
                  i_micro_kernel_config->instruction_set,
                  LIBXSMM_X86_INSTR_VMOVDQU16,
                  i_gp_reg_mapping->gp_reg_c,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  ((1ull * i_packed_processed * i_simd_packed_width ) +
                   (1ull * i_packed_width * (l_n + l_n_tile * 16)) + (l_p * i_simd_packed_width) + (l_n_block_id * l_bn_microkernel * i_packed_width)) * (i_micro_kernel_config->datatype_size_out),
                  'y', l_reg0,
                  ((i_packed_remainder > 0) && (l_p == i_packed_blocking - 1)) ? l_output_bf16_mask : 0, 1, 0 );
              /* up-convert bf16 to fp32 */
              libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, 'z', l_reg0, l_reg0 );
              /* Store column to scratch */
              libxsmm_x86_instruction_vec_move( io_generated_code,
                  i_micro_kernel_config->instruction_set,
                  LIBXSMM_X86_INSTR_VMOVDQU16,
                  LIBXSMM_X86_GP_REG_RSP,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  l_tile * 1024 + l_n * i_simd_packed_width * 4,
                  'z', l_reg0,
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
      } else {
      }
    }

    if (l_beta_0 > 0) {
      /* Check if empty B column and beta == 0 and jump at the C store part of the kernel */
      libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, l_cur_column_gpr, l_next_column_gpr);
      libxsmm_x86_instruction_jump_to_label(io_generated_code, LIBXSMM_X86_INSTR_JE, EMPTY_BLOCK_COLUMN_LABEL_BETA0, i_jump_label_tracker);
    }

    /* k loop header */
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVD, l_row_idx_gpr, l_cur_column_gpr, 4, 0, i_gp_reg_mapping->gp_reg_kloop, 0 );

    /* Prep reg_a with "k" offset */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_kloop, i_bk * i_packed_width * i_micro_kernel_config->datatype_size_in);
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_kloop);
    /* Prep reg_b with "k" offset */
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, l_cur_column_gpr, l_b_tmp_gpr);
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, l_b_tmp_gpr, i_bk * i_bn  * i_micro_kernel_config->datatype_size_in);
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_b, l_b_tmp_gpr);

    for (l_k_iter = 0; l_k_iter < l_k_unroll_iters; l_k_iter++) {
      /* Load A tile  */
      for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
        unsigned int tile_a = l_a_tile_offset + l_p;
        libxsmm_x86_instruction_tile_move( io_generated_code,
            io_generated_code->arch,
            LIBXSMM_X86_INSTR_TILELOADD,
            i_gp_reg_mapping->gp_reg_kloop,
            i_gp_reg_mapping->gp_reg_lda,
            4,
            ((1ull * i_packed_processed * i_simd_packed_width * l_vnni_block_size) + (l_k_iter * l_k_elements * i_packed_width * l_vnni_block_size) + (l_p * i_simd_packed_width * l_vnni_block_size))* i_micro_kernel_config->datatype_size_in,
            tile_a);
      }

      /* loop over the columns of B/C */
      for ( l_n = 0; l_n < l_n_tiles_bn; l_n++ ) {
        unsigned int tile_b = 6 + l_n % 2;
        libxsmm_x86_instruction_tile_move( io_generated_code,
            io_generated_code->arch,
            LIBXSMM_X86_INSTR_TILELOADD,
            l_b_tmp_gpr,
            i_gp_reg_mapping->gp_reg_ldb,
            4,
            ((long long) (l_n * 16) * i_bk + (l_k_iter * l_k_elements * l_vnni_block_size) + (l_n_block_id * l_bn_microkernel * i_bk)) * i_micro_kernel_config->datatype_size_in,
            tile_b);

        if (l_pf_dist_B > 0) {
          if (l_k_iter == 0 && l_n == 0 && l_n_block_id == 0) {
            libxsmm_generator_packed_spgemm_bcsc_prefetch_B_block_in_L1( io_generated_code, i_bk*i_bn*i_micro_kernel_config->datatype_size_in, l_b_tmp_gpr, (long long)l_pf_dist_B*i_bk*i_bn*i_micro_kernel_config->datatype_size_in);
          }
        }

        for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
          unsigned int tile_comp_instr = ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ? LIBXSMM_X86_INSTR_TDPBF16PS :
            ( (((LIBXSMM_GEMM_FLAG_A_UNSIGNED & i_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_B_UNSIGNED & i_xgemm_desc->flags) > 0)) ? LIBXSMM_X86_INSTR_TDPBUSD : LIBXSMM_X86_INSTR_TDPBSUD);
          unsigned int tile_a = l_a_tile_offset + l_p;
          unsigned int tile_c = l_n * i_packed_blocking + l_p + l_c_tile_offset;
          libxsmm_x86_instruction_tile_compute( io_generated_code,
              io_generated_code->arch,
              tile_comp_instr,
              tile_a,
              tile_b,
              tile_c);
        }
      }
    }

    /* k loop footer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, l_cur_column_gpr, 1 );
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, l_next_column_gpr, l_cur_column_gpr);
    libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker);
    if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_ldc, ((long long)16 * 4)/4);
    } else if (LIBXSMM_DATATYPE_I32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_ldc, ((long long)i_packed_width * 4)/4);
    } else {
    }

    if (l_beta_0 > 0) {
      /* LABEL for empty column and beta == 0 */
      libxsmm_x86_instruction_register_jump_label(io_generated_code, EMPTY_BLOCK_COLUMN_LABEL_BETA0, i_jump_label_tracker);
    }

    if ( LIBXSMM_DATATYPE_I32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
      for ( l_n_tile = 0; l_n_tile < l_n_tiles_bn; l_n_tile++) {
        for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
          unsigned int l_tile = (l_n_tile*i_packed_blocking) + l_p + l_c_tile_offset;
          libxsmm_x86_instruction_tile_move( io_generated_code,
                  io_generated_code->arch,
                  LIBXSMM_X86_INSTR_TILESTORED,
                  i_gp_reg_mapping->gp_reg_c,
                  i_gp_reg_mapping->gp_reg_ldc,
                  4,
                  ((1ull * i_packed_processed * i_simd_packed_width) +
                  (1ull * i_packed_width * l_n_tile * 16) + (l_p * i_simd_packed_width) + (l_n_block_id * l_bn_microkernel * i_packed_width)) * (i_micro_kernel_config->datatype_size_out),
                  l_tile);
        }
      }
    } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
      /* store C accumulator */
      for ( l_n_tile = 0; l_n_tile < l_n_tiles_bn; l_n_tile++) {
        /* Store accumulators to scratch */
        unsigned int cur_tile_cols = (l_n_tile == 0) ? l_n_cols0 : l_n_cols1 ;
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
        for ( l_n = 0; l_n < cur_tile_cols; l_n++) {
          for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
            unsigned int l_reg0 = l_n;
            /* Load reg0 from scratch */
            unsigned int l_tile = (l_n_tile*i_packed_blocking) + l_p + l_c_tile_offset;
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVDQU16,
                LIBXSMM_X86_GP_REG_RSP,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                l_tile * 1024 + l_n * i_simd_packed_width * 4,
                'z', l_reg0,
                0, 1, 0 );

            /* Cvt + store --> write to C */
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, l_reg0, l_reg0 );
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVDQU16,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((1ull * i_packed_processed * i_simd_packed_width) +
                 (1ull * i_packed_width * (l_n + l_n_tile * 16) ) + (l_p * i_simd_packed_width) + (l_n_block_id * l_bn_microkernel * i_packed_width)) * (i_micro_kernel_config->datatype_size_out),
                'y', l_reg0,
                ((i_packed_remainder > 0) && (l_p == i_packed_blocking - 1)) ? l_output_bf16_mask : 0, 0, 1 );
          }
        }
      }
    } else {

    }
    EMPTY_BLOCK_COLUMN_LABEL_BETA0++;
    if (l_bn_iters > 1) {
      if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
        libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 8 * 1024 + 64, l_cur_column_gpr, 0 );
      } else if ( LIBXSMM_DATATYPE_I32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
        libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 64, l_cur_column_gpr, 0 );
      } else {

      }
    }
  }

  /* packed loop */
  if ( i_packed_range/i_packed_blocking > 1 ) {
    /* advance A and C pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, (long long)i_packed_blocking*i_simd_packed_width*l_vnni_block_size*i_micro_kernel_config->datatype_size_in );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c, (long long)i_packed_blocking*i_simd_packed_width*i_micro_kernel_config->datatype_size_out );

    /* packed loop footer */
    if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
      libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 8 * 1024 + 64, l_cur_column_gpr, 0 );
    } else if ( LIBXSMM_DATATYPE_I32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
      libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 64, l_cur_column_gpr, 0 );
    } else {

    }
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_packed_range/i_packed_blocking );
    libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );

    /* reset A and C pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_a, ((long long)i_packed_range/i_packed_blocking)*i_packed_blocking*i_simd_packed_width*l_vnni_block_size*i_micro_kernel_config->datatype_size_in );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_c, ((long long)i_packed_range/i_packed_blocking)*i_packed_blocking*i_simd_packed_width*i_micro_kernel_config->datatype_size_out );
  }

  if (l_beta_0 == 0)  {
    /* LABEL for empty column and beta == 1 */
    libxsmm_x86_instruction_register_jump_label(io_generated_code, EMPTY_BLOCK_COLUMN_LABEL, i_jump_label_tracker);
  }

  if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
    libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 8 * 1024, i_gp_reg_mapping->gp_reg_kloop, 0 );
  } else if ( LIBXSMM_DATATYPE_I32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) {
    libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, i_gp_reg_mapping->gp_reg_kloop, 0 );
  } else {
  }
}

