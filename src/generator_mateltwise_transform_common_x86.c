/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke, Barukh Ziv, Menachem Adelmanm (Intel Corp.)
******************************************************************************/
#include "generator_mateltwise_transform_common_x86.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"


LIBXSMM_API_INTERN
void libxsmm_generator_transform_Xway_unpack_network_avx_avx512( libxsmm_generated_code* io_generated_code,
                                                                 const char              i_vector_name,
                                                                 const unsigned char*    i_in_idx,
                                                                 const unsigned int      i_vec_reg_src_start,
                                                                 const unsigned int      i_vec_reg_dst_start,
                                                                 const unsigned int      i_out_offset,
                                                                 const unsigned int      i_even_instr,
                                                                 const unsigned int      i_odd_instr,
                                                                 const unsigned int      i_ways ) {
  unsigned int l_i = 0;

  if ( (i_ways != 2) && (i_ways !=4) && (i_ways != 8) && (i_ways != 16) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  if ( (i_vec_reg_src_start % i_ways != 0) ||
       (i_vec_reg_dst_start % i_ways != 0)    ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  for ( l_i = 0 ; l_i < i_ways ; ++l_i ) {
    unsigned int l_shuffle_instr = ( l_i % 2 == 0 ) ? i_even_instr : i_odd_instr;
    unsigned int in1 = i_in_idx[l_i] + i_vec_reg_src_start;
    unsigned int in0 = in1           + i_out_offset;
    unsigned int dst = l_i           + i_vec_reg_dst_start;

    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, l_shuffle_instr, i_vector_name,
                                                            in0, in1, dst, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_Xway_full_load_avx_avx512( libxsmm_generated_code* io_generated_code,
                                                            const char              i_vector_name,
                                                            const unsigned int      i_gp_reg_in,
                                                            const unsigned int      i_vec_reg_dst_start,
                                                            const unsigned int      i_ld,
                                                            const unsigned int      i_ld_instr,
                                                            const unsigned int      i_ways,
                                                            const unsigned int      i_valid_ways,
                                                            const unsigned int      i_use_masking,
                                                            const unsigned int      i_mask_reg ) {
  unsigned int l_i = 0;

  if ( (i_ways != 8) && (i_ways != 16) && (i_ways != 32) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  if ( i_vec_reg_dst_start % i_ways != 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  if ( i_valid_ways > i_ways ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  for ( l_i = 0 ; l_i < i_ways ; ++l_i ) {
    unsigned int l_dst = l_i + i_vec_reg_dst_start;

    if ( l_i < i_valid_ways ) {
      libxsmm_x86_instruction_vex_evex_mask_mov( io_generated_code, i_ld_instr,
                                                 i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, l_i * i_ld,
                                                 i_vector_name, l_dst, i_use_masking, i_mask_reg, 0 );
    } else {
      libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD,
                                                i_vector_name, l_dst, l_dst, l_dst );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_Xway_full_store_avx_avx512( libxsmm_generated_code* io_generated_code,
                                                             const char              i_vector_name,
                                                             const unsigned int      i_gp_reg_out,
                                                             const unsigned int      i_vec_reg_src_start,
                                                             const unsigned int      i_ld,
                                                             const unsigned int      i_st_instr,
                                                             const unsigned int      i_use_masking,
                                                             const unsigned int      i_mask_reg,
                                                             const unsigned int      i_ways ) {
  unsigned int l_i = 0;

  for ( l_i = 0 ; l_i < i_ways ; ++l_i ) {
    unsigned int l_src = l_i + i_vec_reg_src_start;

    libxsmm_x86_instruction_vex_evex_mask_mov( io_generated_code, i_st_instr,
                                               i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, l_i * i_ld,
                                               i_vector_name, l_src, i_use_masking, i_mask_reg, 1 );
  }
}


