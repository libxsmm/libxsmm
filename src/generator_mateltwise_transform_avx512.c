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
#include "generator_mateltwise_sse_avx_avx512.h"
#include "generator_mateltwise_transform_common_x86.h"
#include "generator_mateltwise_transform_avx512.h"
#include "generator_mateltwise_transform_avx.h"
#include "generator_mateltwise_transform_sse.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"

#define USE_OPTIMIZED_AVX512_TRANSPOSE 1
#define COPY_UPPER_BOUND 0


LIBXSMM_API_INTERN
void libxsmm_generator_transform_Xway_shuffle_network_avx512( libxsmm_generated_code* io_generated_code,
                                                              const char              i_vector_name,
                                                              const unsigned char     i_in_idx[16],
                                                              const unsigned char     i_shuf_imm[16],
                                                              const unsigned int      i_vec_reg_src_start,
                                                              const unsigned int      i_vec_reg_dst_start,
                                                              const unsigned int      i_out_offset,
                                                              const unsigned int      i_shuffle_instr,
                                                              const unsigned int      i_ways ) {
  unsigned int l_i = 0;

  if ( (i_ways != 4) && (i_ways != 8) && (i_ways != 16) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  if ( (i_vec_reg_src_start % i_ways != 0) ||
       (i_vec_reg_dst_start % i_ways != 0)    ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  for ( l_i = 0 ; l_i < i_ways ; ++l_i ) {
    unsigned int in1 = i_in_idx[l_i] + i_vec_reg_src_start;
    unsigned int in0 = in1           + i_out_offset;
    unsigned int dst = l_i           + i_vec_reg_dst_start;

    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, i_shuffle_instr, i_vector_name,
                                                            in0, in1, dst, 0, 0, 0, i_shuf_imm[l_i] );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_Xway_byteshuffle_network_avx512( libxsmm_generated_code* io_generated_code,
                                                                  const char              i_vector_name,
                                                                  const unsigned char     i_in_idx[16],
                                                                  const unsigned int      i_vec_reg_suffle_cntl,
                                                                  const unsigned int      i_vec_reg_src_start,
                                                                  const unsigned int      i_vec_reg_dst_start,
                                                                  const unsigned int      i_shuffle_instr,
                                                                  const unsigned int      i_ways ) {
  unsigned int l_i = 0;

  if ( (i_ways != 2) && (i_ways != 4) && (i_ways != 8) && (i_ways != 16) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  if ( (i_vec_reg_src_start % i_ways != 0) ||
       (i_vec_reg_dst_start % i_ways != 0)    ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  for ( l_i = 0 ; l_i < i_ways ; ++l_i ) {
    unsigned int in0 = i_in_idx[l_i] + i_vec_reg_src_start;
    unsigned int dst = l_i           + i_vec_reg_dst_start;

    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, i_shuffle_instr, i_vector_name,
                                                            i_vec_reg_suffle_cntl, in0, dst, 0, 0, 0, 0 );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_16way_permute_network_avx512( libxsmm_generated_code* io_generated_code,
                                                               const char              i_vector_name,
                                                               const unsigned char     i_perm_mask[2],
                                                               const unsigned char     i_perm_imm[2],
                                                               const unsigned int      i_vec_reg_srcdst_start,
                                                               const unsigned int      i_perm_instr ) {
  unsigned int l_i = 0;

  if ( (i_vec_reg_srcdst_start != 0) && (i_vec_reg_srcdst_start != 16) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* copy registers as we use src/dst */
  for ( l_i = 0; l_i < 8; ++l_i ) {
    unsigned int in0 = l_i + i_vec_reg_srcdst_start;
    unsigned int dst = (in0 + 16)%32;

    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64_LD, i_vector_name,
                                                            in0, LIBXSMM_X86_VEC_REG_UNDEF, dst, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
  }

  for ( l_i = 0; l_i < 8; ++l_i ) {
    unsigned int in0 = l_i + i_vec_reg_srcdst_start + 8;
    unsigned int dst = l_i + i_vec_reg_srcdst_start;

    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, i_perm_instr, i_vector_name,
                                                            in0, LIBXSMM_X86_VEC_REG_UNDEF, dst, i_perm_mask[0], 0, 0, i_perm_imm[0] );
  }

  for ( l_i = 8; l_i < 16; ++l_i ) {
    unsigned int in0 = ((l_i-8) + i_vec_reg_srcdst_start + 16)%32;
    unsigned int dst = l_i + i_vec_reg_srcdst_start;

    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, i_perm_instr, i_vector_name,
                                                            in0, LIBXSMM_X86_VEC_REG_UNDEF, dst, i_perm_mask[1], 0, 0, i_perm_imm[1] );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_Xway_permutevar1_network_avx512( libxsmm_generated_code* io_generated_code,
                                                                  const char              i_vector_name,
                                                                  const unsigned int      i_vec_reg_perm_idx,
                                                                  const unsigned int      i_vec_reg_srcdst_start,
                                                                  const unsigned int      i_perm_instr,
                                                                  const unsigned int      i_ways ) {
  unsigned int l_i = 0;

  if ( i_vec_reg_srcdst_start + i_ways > 31 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  for ( l_i = 0; l_i < i_ways; ++l_i ) {
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, i_perm_instr, i_vector_name,
                                                            i_vec_reg_srcdst_start + l_i, i_vec_reg_perm_idx, i_vec_reg_srcdst_start + l_i,
                                                            0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_08way_permutevar_network_avx512( libxsmm_generated_code* io_generated_code,
                                                                  const char              i_vector_name,
                                                                  const unsigned int      i_vec_reg_perm_idx_lo,
                                                                  const unsigned int      i_vec_reg_perm_idx_hi,
                                                                  const unsigned int      i_vec_reg_srcdst_start,
                                                                  const unsigned int      i_perm_instr ) {
  unsigned int l_i = 0;

  if ( (i_vec_reg_srcdst_start != 0) && (i_vec_reg_srcdst_start != 8) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* copy registers as we use src/dst */
  for ( l_i = 0; l_i < 4; ++l_i ) {
    unsigned int in0 = l_i + i_vec_reg_srcdst_start;
    unsigned int dst = (((l_i/2)*4) + (l_i%2) + i_vec_reg_srcdst_start + 8) % 16;

    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64_LD, i_vector_name,
                                                            in0, LIBXSMM_X86_VEC_REG_UNDEF, dst, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );

    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64_LD, i_vector_name,
                                                            in0, LIBXSMM_X86_VEC_REG_UNDEF, dst+2, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
   }

  for ( l_i = 0; l_i < 4; ++l_i ) {
    unsigned int in0 = l_i + i_vec_reg_srcdst_start + 4;
    unsigned int dst = (((l_i/2)*4) + (l_i%2) + i_vec_reg_srcdst_start + 8) % 16;

    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, i_perm_instr, i_vector_name,
                                                            in0, i_vec_reg_perm_idx_lo, dst, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );

    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, i_perm_instr, i_vector_name,
                                                            in0, i_vec_reg_perm_idx_hi, dst+2, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_Xway_permute_network_avx512( libxsmm_generated_code* io_generated_code,
                                                              const char              i_vector_name,
                                                              const unsigned char     i_perm_mask[2],
                                                              const unsigned char     i_perm_imm[2],
                                                              const unsigned int      i_vec_reg_srcdst_start,
                                                              const unsigned int      i_perm_instr,
                                                              const unsigned int      i_ways ) {
  unsigned int l_i = 0;

  if ( (i_ways % 2 != 0) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* i_ways * 3 should be less than or equal to 32 */
  if ( (i_ways > 20) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* copy registers for odd quarters */
  for ( l_i = 0; l_i < i_ways/2; ++l_i ) {
    unsigned int in = l_i + i_vec_reg_srcdst_start;
    unsigned int dst = (l_i + i_vec_reg_srcdst_start + i_ways) % 32;

    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64_LD, i_vector_name,
                                                            in, LIBXSMM_X86_VEC_REG_UNDEF, dst, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
  }

  /* Even quarters */
  for ( l_i = 0; l_i < i_ways/2; ++l_i ) {
    unsigned int in = l_i + i_vec_reg_srcdst_start + i_ways/2;
    unsigned int dst = l_i + i_vec_reg_srcdst_start;

    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, i_perm_instr, i_vector_name,
                                                            in, LIBXSMM_X86_VEC_REG_UNDEF, dst, i_perm_mask[0], 0, 0, i_perm_imm[0] );
  }

  /* Odd quarters */
  for ( l_i = i_ways/2; l_i < i_ways; ++l_i ) {
    unsigned int in = (l_i + i_vec_reg_srcdst_start + i_ways/2) % 32;
    unsigned int dst = l_i + i_vec_reg_srcdst_start;

    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, i_perm_instr, i_vector_name,
                                                            in, LIBXSMM_X86_VEC_REG_UNDEF, dst, i_perm_mask[1], 0, 0, i_perm_imm[1] );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_Xway_half_load_blend_avx512( libxsmm_generated_code* io_generated_code,
                                                              const char              i_vector_name,
                                                              const unsigned int      i_gp_reg_in,
                                                              const unsigned int      i_vec_reg_dst_start,
                                                              const unsigned int      i_ld,
                                                              const unsigned int*     i_ld_idx,
                                                              const unsigned int      i_blend_mult,
                                                              const unsigned int      i_ld_instr,
                                                              const unsigned int      i_ways,
                                                              const unsigned int      i_mask_reg[2],
                                                              const unsigned int      i_n ) {
  unsigned int l_i = 0;
  unsigned int l_h = 0;
  unsigned int l_blend_offset = i_blend_mult * i_ld;

  /* supports only up to 32 registers */
  if (i_ways > 32) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  if (i_n > i_ways * 2) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
  }

  for ( l_i = 0 ; l_i < i_ways ; ++l_i ) {
    unsigned int l_dst = l_i + i_vec_reg_dst_start;
    unsigned int l_load_displ = i_ld * (i_ld_idx[l_i] / 2) + 32 * (i_ld_idx[l_i] % 2);
    unsigned int l_way_halves = (i_n / i_ways) + (l_i < (i_n % i_ways));

    for ( l_h = 0; l_h < l_way_halves; ++l_h ) {
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_ld_instr,
                                        i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, l_load_displ + l_h * l_blend_offset,
                                        i_vector_name, l_dst, i_mask_reg[l_h], (l_h == 0) , 0 );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_Xway_quarter_load_blend_avx512( libxsmm_generated_code* io_generated_code,
                                                                 const char              i_vector_name,
                                                                 const unsigned int      i_gp_reg_in,
                                                                 const unsigned int      i_vec_reg_dst_start,
                                                                 const unsigned int      i_ld,
                                                                 const unsigned int      i_ld_instr,
                                                                 const unsigned int      i_ways,
                                                                 const unsigned int      i_mask_reg[4],
                                                                 const unsigned int      i_n,
                                                                 const unsigned int      is_non32bit_ld ) {
  unsigned int l_i = 0;
  unsigned int l_q = 0;
  unsigned int l_stride_offset = i_ways * i_ld;
  unsigned int tmp_dst = ( i_vec_reg_dst_start + i_ways ) % 32;

  /* supports only up to 32 registers */
  if ( i_ways > 32 - is_non32bit_ld ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* These are *quarter* loads */
  if (i_n > i_ways * 4) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  for ( l_i = 0 ; l_i < i_ways ; ++l_i ) {
    unsigned int l_dst = l_i + i_vec_reg_dst_start;
    unsigned int l_way_quarters = (i_n / i_ways) + (l_i < (i_n % i_ways));

    if ( is_non32bit_ld ) {
      /* 16-bit quarter loads: use loads + insert */
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_ld_instr,
                                        i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, l_i * i_ld,
                                        i_vector_name, l_dst, i_mask_reg[0], 0, 0 );

      for ( l_q = 1; l_q < l_way_quarters; ++l_q ) {
        libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_ld_instr,
                                          i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, l_i * i_ld + l_stride_offset * l_q,
                                          i_vector_name, tmp_dst, i_mask_reg[0], 0, 0 );
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VINSERTI32X4, i_vector_name,
                                                                tmp_dst, l_dst, l_dst, 0, 0, 0, l_q );
      }
    } else {
      for ( l_q = 0; l_q < l_way_quarters; ++l_q ) {
          libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_ld_instr,
                                            i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, l_i * i_ld + l_stride_offset * l_q,
                                            i_vector_name, l_dst, i_mask_reg[l_q], 0, 0 );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_32way_half_store_avx512( libxsmm_generated_code* io_generated_code,
                                                          const char              i_vector_name,
                                                          const unsigned int      i_gp_reg_out,
                                                          const unsigned int      i_vec_reg_src_start,
                                                          const unsigned int      i_ld,
                                                          const unsigned int      i_st_instr ) {
  unsigned int l_i = 0;

  if ( (i_vec_reg_src_start != 0) && (i_vec_reg_src_start != 16) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  for ( l_i = 0 ; l_i < 32 ; ++l_i ) {
    unsigned int l_srcdst = (l_i/2) + i_vec_reg_src_start;

    if ( l_i % 2 == 1 ) {
      libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8( io_generated_code, LIBXSMM_X86_INSTR_VEXTRACTI64X4, i_vector_name,
          i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, l_i * i_ld, 0, LIBXSMM_X86_VEC_REG_UNDEF, l_srcdst, 0, 0, 0x1);
    } else {
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_st_instr,
                                        i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, l_i * i_ld,
                                        'y', l_srcdst, 0, 1, 1 );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_two_4x4_64bit_norm_to_normt_avx512( libxsmm_generated_code* io_generated_code,
                                                                     const char              i_vector_name,
                                                                     const unsigned int      i_vec_reg_src_start,
                                                                     const unsigned int      i_vec_reg_dst_start,
                                                                     const unsigned int      i_mask_reg_1,
                                                                     const unsigned int      i_mask_reg_2 )
{
  if ( (i_vec_reg_src_start + 4 > i_vec_reg_dst_start) && (i_vec_reg_dst_start + 4 > i_vec_reg_src_start) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* 1st stage: unpack network */
  {
    unsigned char l_in_idx[4] = { 0x0, 0x0, 0x2, 0x2};
    libxsmm_generator_transform_Xway_unpack_network_avx_avx512( io_generated_code, i_vector_name,
                                                                l_in_idx, i_vec_reg_src_start, i_vec_reg_dst_start, 1,
                                                                LIBXSMM_X86_INSTR_VPUNPCKLQDQ, LIBXSMM_X86_INSTR_VPUNPCKHQDQ, 4 );
  }

  /* 2nd stage: variable permute network */
  {
    unsigned char l_perm_imm[2] = { 0x44, 0xee };
    unsigned char l_perm_mask[2]; l_perm_mask[0] = (unsigned char)i_mask_reg_1; l_perm_mask[1] = (unsigned char)i_mask_reg_2;

    libxsmm_generator_transform_Xway_permute_network_avx512( io_generated_code, i_vector_name,
                                                             l_perm_mask, l_perm_imm, i_vec_reg_dst_start, LIBXSMM_X86_INSTR_VPERMQ_I, 4 );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_128bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                          libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                          const unsigned int                      i_gp_reg_in,
                                                                          const unsigned int                      i_gp_reg_out,
                                                                          const unsigned int                      i_gp_reg_m_loop,
                                                                          const unsigned int                      i_gp_reg_n_loop,
                                                                          const unsigned int                      i_gp_reg_mask,
                                                                          const unsigned int                      i_mask_reg_0,
                                                                          const unsigned int                      i_mask_reg_1,
                                                                          const unsigned int                      i_mask_reg_2,
                                                                          const unsigned int                      i_mask_reg_3,
                                                                          const unsigned int                      i_mask_reg_4,
                                                                          const unsigned int                      i_mask_reg_5,
                                                                          const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                          const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned long long l_mask = 0;

  /* Partial 8x4 blocks */
  const unsigned int l_n_4rem = i_mateltwise_desc->n % 4;
  const unsigned int l_m_4rem = i_mateltwise_desc->m % 4;

  const unsigned int l_n_4mul = i_mateltwise_desc->n - l_n_4rem;
  const unsigned int l_m_4mul = i_mateltwise_desc->m - l_m_4rem;

  const unsigned int l_datatype_size_in = 16;
  const unsigned int l_datatype_size_out = 16;

  /* set the masks for the load+blend stage */
  l_mask = 0xf0;
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                   i_gp_reg_mask, l_mask );
  libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVB_GPR_LD,
                                     i_gp_reg_mask, i_mask_reg_0 );

  /* @TODO: set the masks for the load+blend stage for partial 4x4 blocks */
  LIBXSMM_UNUSED( i_mask_reg_1 );
  LIBXSMM_UNUSED( i_mask_reg_2 );
  LIBXSMM_UNUSED( i_mask_reg_3 );
  LIBXSMM_UNUSED( i_mask_reg_4 );
  LIBXSMM_UNUSED( i_mask_reg_5 );

  if ( (l_m_4rem !=0) || (l_n_4rem !=0) ) {
    libxsmm_generator_transform_norm_to_normt_128bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                      i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                      i_micro_kernel_config, i_mateltwise_desc );
  } else {
    /* Transpose x4 blocks */
    if ( l_m_4mul > 0 ) {
      /* open m loop */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_m_loop, 4 );

      /* transpose 8x4 blocks */
      if ( l_n_4mul > 0 ) {
        /* open n loop */
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
        libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                         i_gp_reg_n_loop, 4 );

        /* load 4 registers with two half rows
           aX-dX: 128bit elements, or xmms
           zmm0: b1 b0 a1 a0
           zmm1: b3 b2 a3 a2
           zmm2: d1 d0 c1 c0
           zmm3: d3 d2 c3 c2  */
        {
          const unsigned int ld_idx[4] = { 0x0, 0x1, 0x4, 0x5};
          unsigned int l_mask_regs[2] = { 0 };
          l_mask_regs[0] = 0;    l_mask_regs[1] = i_mask_reg_0;

          libxsmm_generator_transform_Xway_half_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                   i_gp_reg_in, 0, i_mateltwise_desc->ldi * l_datatype_size_in,
                                                                   ld_idx, 1, LIBXSMM_X86_INSTR_VBROADCASTI64X4, 4, l_mask_regs, 8 );
        }

        /* advance input pointer */
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                         i_gp_reg_in, (long long)i_mateltwise_desc->ldi * l_datatype_size_in * 4 );

        /* transpose 4x4 blocks
          zmm4 = zmm0,zmm2 -> d0 c0 b0 a0
          zmm5 = zmm0,zmm2 -> d1 c1 b1 a1
          zmm6 = zmm1,zmm3 -> d2 c2 b2 a2
          zmm7 = zmm1,zmm3 -> d3 c3 b3 a3 */
        {
          unsigned char l_in_idx[16] = { 0x0, 0x0, 0x1, 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
          unsigned char l_shuf_imm[16] = { 0x88, 0xdd, 0x88, 0xdd, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
          libxsmm_generator_transform_Xway_shuffle_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                   l_in_idx, l_shuf_imm, 0, 4, 2,
                                                                   LIBXSMM_X86_INSTR_VSHUFI64X2, 4);
        }

        /* storing 4 registers */
        libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                i_gp_reg_out, 4, i_mateltwise_desc->ldo * l_datatype_size_out,
                                                                i_micro_kernel_config->vmove_instruction_out, 0, 0, 4 );

        /* advance output pointer */
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                         i_gp_reg_out, (long long)l_datatype_size_out * 4 );

        /* close n footer */
        libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                    i_gp_reg_n_loop, l_n_4mul );
      }

      /* transpose n_4rem x 4 block */
#if 0
      if ( l_n_4rem > 0 ) {
        /* @TODO: check if we need this path for performance */
      }
#endif

      /* advance output pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_out, (4LL * i_mateltwise_desc->ldo * l_datatype_size_out) - ((long long)l_datatype_size_in * l_n_4mul) );

      /* advance input pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                       i_gp_reg_in, ((long long)i_mateltwise_desc->ldi * l_datatype_size_in * l_n_4mul) - (4LL * l_datatype_size_in) );

      /* close m loop */
      libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                  i_gp_reg_m_loop, l_m_4mul );
    }

    /* Transpose m_4rem blocks */
#if 0
    if ( l_m_4rem > 0 ) {
      /* @TODO: check if we need this path for performance */
    }
#endif
  }
}

/*
 * Calling convention, this kernel assumes that all GPRs with are not in the argument list
 * and all xmm/ymm/zmm/tmm are caller save
 *
 * TODO; stack local variables....
 * */
LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_64bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                         libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                         const unsigned int                      i_gp_reg_in,
                                                                         const unsigned int                      i_gp_reg_out,
                                                                         const unsigned int                      i_gp_reg_m_loop,
                                                                         const unsigned int                      i_gp_reg_n_loop,
                                                                         const unsigned int                      i_gp_reg_mask,
                                                                         const unsigned int                      i_mask_reg_0,
                                                                         const unsigned int                      i_mask_reg_1,
                                                                         const unsigned int                      i_mask_reg_2,
                                                                         const unsigned int                      i_mask_reg_3,
                                                                         const unsigned int                      i_mask_reg_4,
                                                                         const unsigned int                      i_mask_reg_5,
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned long long l_mask = 0;

  /* Partial 8x4 blocks */
  const unsigned int l_n_8rem = i_mateltwise_desc->n % 8;
  const unsigned int l_m_4rem = i_mateltwise_desc->m % 4;

  const unsigned int l_n_8mul = i_mateltwise_desc->n - l_n_8rem;
  const unsigned int l_m_4mul = i_mateltwise_desc->m - l_m_4rem;

  /* set the masks for the load+blend stage */
  l_mask = 0xf0;
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                   i_gp_reg_mask, l_mask );
  libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVB_GPR_LD,
                                     i_gp_reg_mask, i_mask_reg_0 );

  /* set the masks for the load+blend stage for partial 8x4 blocks */
  if ( l_m_4rem > 0 ) {
    unsigned int l_mask_regs[2] = { 0 };
    const unsigned int l_m_4rem_mask = (1 << l_m_4rem ) - 1;
    unsigned int l_i = 0;
    l_mask_regs[0] = i_mask_reg_3;    l_mask_regs[1] = i_mask_reg_4;

    for (l_i = 0; l_i < 2; ++l_i) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask, (long long)l_m_4rem_mask << (l_i * 4) );
      libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mask, l_mask_regs[l_i] );
    }
  }

  if ( l_n_8rem > 0 ) {
    const unsigned int l_n_8rem_mask = (1 << l_n_8rem) - 1;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask, l_n_8rem_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
                                       i_gp_reg_mask, i_mask_reg_5 );
  }

  /* set the masks for the permute stage */
  /* even halves mask */
  l_mask = 0xcc;
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                   i_gp_reg_mask, l_mask );
  libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVB_GPR_LD,
                                     i_gp_reg_mask, i_mask_reg_1 );

  /* odd halves masks */
  l_mask = 0x33;
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                   i_gp_reg_mask, l_mask );
  libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVB_GPR_LD,
                                     i_gp_reg_mask, i_mask_reg_2 );

  /* Transpose x4 blocks */
  if ( l_m_4mul > 0 ) {

    /* open m loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_m_loop, 4 );

    /* transpose 8x4 blocks */
    if ( l_n_8mul > 0 ) {

      /* open n loop */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_n_loop, 8 );

      /* load 4 registers with two half rows */
      {
        const unsigned int ld_idx[4] = { 0x0, 0x2, 0x4, 0x6};
        unsigned int l_mask_regs[2] = { 0 };
        l_mask_regs[0] = 0;    l_mask_regs[1] = i_mask_reg_0;

        libxsmm_generator_transform_Xway_half_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                 i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                                 ld_idx, 4, LIBXSMM_X86_INSTR_VBROADCASTI64X4, 4, l_mask_regs, 8 );
      }

      /* advance input pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_in, (long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * 8 );

      /* transpose two 4x4 blocks */
      libxsmm_generator_transform_two_4x4_64bit_norm_to_normt_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 4, i_mask_reg_1, i_mask_reg_2);

      /* storing 4 registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 4, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 0, 0, 4 );

      /* advance output pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_out, (long long)i_micro_kernel_config->datatype_size_in * 8 );

      /* close n footer */
      libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                  i_gp_reg_n_loop, l_n_8mul );
    }

    /* transpose n_8_rem x 4 block */
    if ( l_n_8rem > 0 ) {
      /* load 4 registers with two half rows */
      {
        const unsigned int ld_idx[8] = { 0x0, 0x2, 0x4, 0x6};
        unsigned int l_mask_regs[2] = { 0 };
        l_mask_regs[0] = 0;    l_mask_regs[1] = i_mask_reg_0;
        libxsmm_generator_transform_Xway_half_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                 i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                                 ld_idx, 4, LIBXSMM_X86_INSTR_VBROADCASTI64X4, 4, l_mask_regs, l_n_8rem );
      }

      /* transpose two 4x4 blocks */
      libxsmm_generator_transform_two_4x4_64bit_norm_to_normt_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 4, i_mask_reg_1, i_mask_reg_2);

      /* storing 4 registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 4, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 1, i_mask_reg_5, 4 );
    }

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (4LL * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_in * l_n_8mul) );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                     i_gp_reg_in, ((long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * l_n_8mul) - (4LL * i_micro_kernel_config->datatype_size_in) );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, l_m_4mul );
  }

  /* Transpose m_4rem blocks */
  if ( l_m_4rem > 0 ) {

    /* transpose 8 x m_4rem blocks */
    if ( l_n_8mul > 0 ) {
      /* open n loop */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_n_loop, 8);

      /* load 4 registers with two half rows */
      {
        const unsigned int ld_idx[4] = { 0x0, 0x2, 0x4, 0x6};
        unsigned int l_mask_regs[2] = { 0 };
        l_mask_regs[0] = i_mask_reg_3;    l_mask_regs[1] = i_mask_reg_4;
        libxsmm_generator_transform_Xway_half_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                 i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                                 ld_idx, 4, LIBXSMM_X86_INSTR_VBROADCASTI64X4, 4, l_mask_regs, 8 );
      }

      /* advance input pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_in, (long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * 8 );

      /* transpose two 4x4 blocks */
      libxsmm_generator_transform_two_4x4_64bit_norm_to_normt_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 4, i_mask_reg_1, i_mask_reg_2);

      /* storing l_m_4rem registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 4, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 0, 0, l_m_4rem );
      /* advance output pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_out, (long long)i_micro_kernel_config->datatype_size_out * 8 );

      /* close n footer */
      libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                  i_gp_reg_n_loop, l_n_8mul );
    }

    /* transpose n_8rem x m_4rem block */
    if ( l_n_8rem > 0 ) {
      /* load 4 registers with two half rows */
      {
        const unsigned int ld_idx[4] = { 0x0, 0x2, 0x4, 0x6};
        unsigned int l_mask_regs[2] = { 0 };
        l_mask_regs[0] = i_mask_reg_3;    l_mask_regs[1] = i_mask_reg_4;
        libxsmm_generator_transform_Xway_half_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                 i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                                 ld_idx, 4, LIBXSMM_X86_INSTR_VBROADCASTI64X4, 4, l_mask_regs, l_n_8rem );
      }

      /* transpose two 4x4 blocks */
      libxsmm_generator_transform_two_4x4_64bit_norm_to_normt_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 4, i_mask_reg_1, i_mask_reg_2);

      /* storing l_m_4rem registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 4, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 1, i_mask_reg_5, l_m_4rem );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_two_8x8_32bit_norm_to_normt_avx512( libxsmm_generated_code* io_generated_code,
                                                                     const char              i_vector_name,
                                                                     const unsigned int      i_vec_reg_srcdst_start,
                                                                     const unsigned int      i_vec_reg_tmp_start,
                                                                     const unsigned int      i_mask_reg_1,
                                                                     const unsigned int      i_mask_reg_2 )
{
  if ( (i_vec_reg_srcdst_start + 8 > i_vec_reg_tmp_start) && (i_vec_reg_tmp_start + 8 > i_vec_reg_srcdst_start) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* first shuffle stage */
  {
    unsigned char l_in_idx[8] = { 0x0, 0x0, 0x2, 0x2, 0x4, 0x4, 0x6, 0x6};
    libxsmm_generator_transform_Xway_unpack_network_avx_avx512( io_generated_code, i_vector_name,
                                                            l_in_idx, i_vec_reg_srcdst_start, i_vec_reg_tmp_start, 1,
                                                            LIBXSMM_X86_INSTR_VPUNPCKLDQ, LIBXSMM_X86_INSTR_VPUNPCKHDQ, 8 );
  }

  /* second shuffle stage */
  {
    unsigned char l_in_idx[8] = { 0x0, 0x0, 0x1, 0x1, 0x4, 0x4, 0x5, 0x5};
    libxsmm_generator_transform_Xway_unpack_network_avx_avx512( io_generated_code, i_vector_name,
                                                            l_in_idx, i_vec_reg_tmp_start, i_vec_reg_srcdst_start, 2,
                                                            LIBXSMM_X86_INSTR_VPUNPCKLQDQ, LIBXSMM_X86_INSTR_VPUNPCKHQDQ, 8 );
  }

  /* 3rd stage: variable permute network */
  {
    unsigned char l_perm_imm[2] = { 0x44, 0xee };
    unsigned char l_perm_mask[2]; l_perm_mask[0] = (unsigned char)i_mask_reg_1; l_perm_mask[1] = (unsigned char)i_mask_reg_2;

    libxsmm_generator_transform_Xway_permute_network_avx512( io_generated_code, i_vector_name,
                                                             l_perm_mask, l_perm_imm, i_vec_reg_srcdst_start, LIBXSMM_X86_INSTR_VPERMQ_I, 8 );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_four_4x4_32bit_norm_to_normt_avx512( libxsmm_generated_code* io_generated_code,
                                                                      const char              i_vector_name,
                                                                      const unsigned int      i_vec_reg_srcdst_start,
                                                                      const unsigned int      i_vec_reg_tmp_start )
{
  if ( (i_vec_reg_srcdst_start + 4 > i_vec_reg_tmp_start) && (i_vec_reg_tmp_start + 4 > i_vec_reg_srcdst_start) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* first shuffle stage */
  {
    unsigned char l_in_idx[4] = { 0x0, 0x0, 0x2, 0x2};
    libxsmm_generator_transform_Xway_unpack_network_avx_avx512( io_generated_code, i_vector_name,
                                                            l_in_idx, i_vec_reg_srcdst_start, i_vec_reg_tmp_start, 1,
                                                            LIBXSMM_X86_INSTR_VPUNPCKLDQ, LIBXSMM_X86_INSTR_VPUNPCKHDQ, 4 );
  }

  /* second shuffle stage */
  {
    unsigned char l_in_idx[4] = { 0x0, 0x0, 0x1, 0x1};
    libxsmm_generator_transform_Xway_unpack_network_avx_avx512( io_generated_code, i_vector_name,
                                                            l_in_idx, i_vec_reg_tmp_start, i_vec_reg_srcdst_start, 2,
                                                            LIBXSMM_X86_INSTR_VPUNPCKLQDQ, LIBXSMM_X86_INSTR_VPUNPCKHQDQ, 4 );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_copy_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                          libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                          const unsigned int                      i_gp_reg_in,
                                                          const unsigned int                      i_gp_reg_out,
                                                          const unsigned int                      i_gp_reg_m_loop,
                                                          const unsigned int                      i_gp_reg_n_loop,
                                                          const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                          const unsigned int                      i_ldi,
                                                          const unsigned int                      i_ldo,
                                                          const unsigned int                      i_m,
                                                          const unsigned int                      i_n,
                                                          const unsigned int                      i_bsize ) {

  if ( (i_m * i_micro_kernel_config->datatype_size_in % 64 != 0 ) ||
       (i_n  % i_bsize != 0 ) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
  }

  /* open m loop */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_m_loop, 64 );

  /* open n loop */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_n_loop, i_bsize );

  /* load i_bsize registers full row */
  libxsmm_generator_transform_Xway_full_load_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                     i_gp_reg_in, 0, i_ldi * i_micro_kernel_config->datatype_size_in,
                                                     i_micro_kernel_config->vmove_instruction_in, i_bsize, i_bsize, 0, 0 );

  /* store i_bsize registers */
  libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                          i_gp_reg_out, 0, i_ldo * i_micro_kernel_config->datatype_size_out,
                                                          i_micro_kernel_config->vmove_instruction_out, 0, 0, i_bsize );

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_in, (long long)i_ldi * i_micro_kernel_config->datatype_size_in * i_bsize );


  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, 64 );

  /* close n footer */
  libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                              i_gp_reg_n_loop, i_n );

  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, ((long long)i_bsize * i_ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_in * i_n) );

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                   i_gp_reg_in, ((long long)i_ldi * i_micro_kernel_config->datatype_size_in * i_n) - 64LL );

  /* close m loop */
  libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                              i_gp_reg_m_loop, i_m*i_micro_kernel_config->datatype_size_in );
}


LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_32bit_avx512_spr_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                             libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                             const unsigned int                      i_gp_reg_in,
                                                                             const unsigned int                      i_gp_reg_out,
                                                                             const unsigned int                      i_gp_reg_m_loop,
                                                                             const unsigned int                      i_gp_reg_n_loop,
                                                                             const unsigned int                      i_gp_reg_mask,
                                                                             const unsigned int                      i_gp_reg_mask_2,
                                                                             const unsigned int                      i_mask_reg_0,
                                                                             const unsigned int                      i_mask_reg_1,
                                                                             const unsigned int                      i_mask_reg_2,
                                                                             const unsigned int                      i_mask_reg_3,
                                                                             const unsigned int                      i_mask_reg_4,
                                                                             const unsigned int                      i_mask_reg_5,
                                                                             const unsigned int                      i_mask_reg_6,
                                                                             const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                             const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {

  unsigned long long l_mask = 0;

  /* optimized shuffle network for SIMD aligned sizes */
  /* Partial 16x4 blocks */
  const unsigned int l_n_16rem = i_mateltwise_desc->n % 16;
  const unsigned int l_m_4rem = i_mateltwise_desc->m % 4;

  const unsigned int l_n_16mul = i_mateltwise_desc->n - l_n_16rem;
  const unsigned int l_m_4mul = i_mateltwise_desc->m - l_m_4rem;

  /* set the masks for the load+blend stage */
  l_mask = 0x00f0;
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                   i_gp_reg_mask, l_mask );
  libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
                                     i_gp_reg_mask, i_mask_reg_0 );

  l_mask = 0x0f00;
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                   i_gp_reg_mask, l_mask );
  libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
                                     i_gp_reg_mask, i_mask_reg_1 );

  l_mask = 0xf000;
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     i_gp_reg_mask, l_mask );
  libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
                                     i_gp_reg_mask, i_mask_reg_2 );

  /* set the masks for the load+blend stage for partial 8x8 blocks */
  if ( l_m_4rem > 0 ) {
    unsigned int l_mask_regs[4] = { 0 };
    const unsigned int l_m_4rem_mask = (1 << l_m_4rem) - 1;
    unsigned int l_i = 0;
    l_mask_regs[0] = i_mask_reg_3;    l_mask_regs[1] = i_mask_reg_4;
    l_mask_regs[2] = i_mask_reg_5;    l_mask_regs[3] = i_mask_reg_6;

    for (l_i = 0; l_i < 4; ++l_i) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask, (long long)l_m_4rem_mask << (l_i * 4) );
        libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mask, l_mask_regs[l_i] );
    }
  }

  if ( l_n_16rem > 0 ) {
      const unsigned int l_n_8rem_mask = (1 << l_n_16rem) - 1;
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask_2, l_n_8rem_mask );
  }

  /* Transpose x4 blocks */
  if ( l_m_4mul > 0 ) {

    /* open m loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_m_loop, 4 );

    /* transpose 16x4 blocks */
    if ( l_n_16mul > 0 ) {

      /* open n loop */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_n_loop, 16 );

      /* load 4 registers with four quarter rows */
      {
        unsigned int l_mask_regs[4] = { 0 };
        l_mask_regs[0] = 0;               l_mask_regs[1] = i_mask_reg_0;
        l_mask_regs[2] = i_mask_reg_1;    l_mask_regs[3] = i_mask_reg_2;
        libxsmm_generator_transform_Xway_quarter_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                    i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                                    LIBXSMM_X86_INSTR_VBROADCASTI32X4, 4, l_mask_regs, 16, 0 );
      }

      /* advance input pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_in, (long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * 16 );

      /* transpose four 4x4 blocks */
      libxsmm_generator_transform_four_4x4_32bit_norm_to_normt_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 4 );

      /* storing 4 registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 0, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 0, 0, 4 );

      /* advance output pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_out, (long long)i_micro_kernel_config->datatype_size_in * 16 );

      /* close n footer */
      libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                  i_gp_reg_n_loop, l_n_16mul );
    }

    /* transpose n_8rem x 8 block */
    if ( l_n_16rem > 0 ) {
      /* set store mask */
      libxsmm_x86_instruction_mask_move(io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mask_2, i_mask_reg_6);

      /* load 4 registers with four quarter rows */
      {
        unsigned int l_mask_regs[4] = { 0 };
        l_mask_regs[0] = 0;               l_mask_regs[1] = i_mask_reg_0;
        l_mask_regs[2] = i_mask_reg_1;    l_mask_regs[3] = i_mask_reg_2;
        libxsmm_generator_transform_Xway_quarter_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                    i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                                    LIBXSMM_X86_INSTR_VBROADCASTI32X4, 4, l_mask_regs, l_n_16rem, 0 );
      }

      /* transpose four 4x4 blocks */
      libxsmm_generator_transform_four_4x4_32bit_norm_to_normt_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 4 );

      /* storing 4 registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 0, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 1, i_mask_reg_6, 4 );

      /* restore quarter load masks */
      libxsmm_x86_instruction_mask_move(io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mask, i_mask_reg_6);
    }

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (4LL * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_in * l_n_16mul) );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                     i_gp_reg_in, ((long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * l_n_16mul) - (4LL * i_micro_kernel_config->datatype_size_in) );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, l_m_4mul );
  }

  /* Transpose m_8rem blocks */
  if ( l_m_4rem > 0 ) {

    /* transpose 32 x m_8rem blocks */
    if ( l_n_16mul > 0 ) {
      /* open n loop */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0 );
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_n_loop, 16 );

      /* load 4 registers with four quarter rows */
      {
        unsigned int l_mask_regs[4] = { 0 };
        l_mask_regs[0] = i_mask_reg_3;    l_mask_regs[1] = i_mask_reg_4;
        l_mask_regs[2] = i_mask_reg_5;    l_mask_regs[3] = i_mask_reg_6;
        libxsmm_generator_transform_Xway_quarter_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                    i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                                    LIBXSMM_X86_INSTR_VBROADCASTI32X4, 4, l_mask_regs, 16, 0 );
      }

      /* advance input pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_in, (long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * 16 );

      /* transpose four 4x4 blocks */
      libxsmm_generator_transform_four_4x4_32bit_norm_to_normt_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 4 );

      /* storing l_m_4rem registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 0, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 0, 0, l_m_4rem );

      /* advance output pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_out, (long long)i_micro_kernel_config->datatype_size_in * 16 );

      /* close n footer */
      libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                  i_gp_reg_n_loop, l_n_16mul );
    }

    /* transpose n_8rem x m_8rem block */
    if ( l_n_16rem > 0 ) {
      /* load 4 registers with four quarter rows */
      {
        unsigned int l_mask_regs[4] = { 0 };
        l_mask_regs[0] = i_mask_reg_3;    l_mask_regs[1] = i_mask_reg_4;
        l_mask_regs[2] = i_mask_reg_5;    l_mask_regs[3] = i_mask_reg_6;
        libxsmm_generator_transform_Xway_quarter_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                    i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                                    LIBXSMM_X86_INSTR_VBROADCASTI32X4, 4, l_mask_regs, l_n_16rem, 0 );
      }

      /* transpose four 4x4 blocks */
      libxsmm_generator_transform_four_4x4_32bit_norm_to_normt_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 4 );

      /* restore store mask */
      libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mask_2, i_mask_reg_6 );

      /* storing l_m_4rem registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 0, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 1, i_mask_reg_6, l_m_4rem );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_32bit_avx512_pre_spr_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                 libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                 const unsigned int                      i_gp_reg_in,
                                                                                 const unsigned int                      i_gp_reg_out,
                                                                                 const unsigned int                      i_gp_reg_m_loop,
                                                                                 const unsigned int                      i_gp_reg_n_loop,
                                                                                 const unsigned int                      i_gp_reg_mask,
                                                                                 const unsigned int                      i_mask_reg_0,
                                                                                 const unsigned int                      i_mask_reg_1,
                                                                                 const unsigned int                      i_mask_reg_2,
                                                                                 const unsigned int                      i_mask_reg_3,
                                                                                 const unsigned int                      i_mask_reg_4,
                                                                                 const unsigned int                      i_mask_reg_5,
                                                                                 const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned long long l_mask = 0;

  /* Partial 16x8 blocks */
  const unsigned int l_n_16rem = i_mateltwise_desc->n % 16;
  const unsigned int l_m_8rem = i_mateltwise_desc->m % 8;

  const unsigned int l_n_16mul = i_mateltwise_desc->n - l_n_16rem;
  const unsigned int l_m_8mul = i_mateltwise_desc->m - l_m_8rem;

  /* set the masks for the load+blend stage */
  l_mask = 0xff00;
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                   i_gp_reg_mask, l_mask );
  libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
                                     i_gp_reg_mask, i_mask_reg_0 );

  /* set the masks for the permute stage */
  /* even quarters mask */
  l_mask = 0xcc;
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                   i_gp_reg_mask, l_mask );
  libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVB_GPR_LD,
                                     i_gp_reg_mask, i_mask_reg_1 );

  /* odd quarter masks */
  l_mask = 0x33;
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                   i_gp_reg_mask, l_mask );
  libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVB_GPR_LD,
                                     i_gp_reg_mask, i_mask_reg_2 );

  /* set the masks for the load+blend stage for partial 16x8 blocks */
  if ( l_m_8rem > 0 ) {
    unsigned int l_mask_regs[2] = { 0 };
    const unsigned int l_m_8rem_mask = (1 << l_m_8rem ) - 1;
    unsigned int l_i = 0;
    l_mask_regs[0] = i_mask_reg_3;    l_mask_regs[1] = i_mask_reg_4;

    for (l_i = 0; l_i < 2; ++l_i) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask, (long long)l_m_8rem_mask << (l_i * 8) );
      libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mask, l_mask_regs[l_i] );
    }
  }

  if ( l_n_16rem > 0 ) {
    const unsigned int l_n_16rem_mask = (1 << l_n_16rem) - 1;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask, l_n_16rem_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
                                       i_gp_reg_mask, i_mask_reg_5 );
  }

  /* Transpose x8 blocks */
  if ( l_m_8mul > 0 ) {

    /* open m loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_m_loop, 8 );

    /* transpose 16x8 blocks */
    if ( l_n_16mul > 0 ) {

      /* open n loop */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_n_loop, 16 );

      /* load 8 registers with two half rows */
      {
        const unsigned int ld_idx[8] = { 0x0, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe };
        unsigned int l_mask_regs[2] = { 0 };
        l_mask_regs[0] = 0;    l_mask_regs[1] = i_mask_reg_0;

        libxsmm_generator_transform_Xway_half_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                 i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                                 ld_idx, 8, LIBXSMM_X86_INSTR_VBROADCASTI32X8, 8, l_mask_regs, 16 );
      }

      /* advance input pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_in, (long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * 16 );

      /* transpose two 8x8 blocks */
      libxsmm_generator_transform_two_8x8_32bit_norm_to_normt_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 8, i_mask_reg_1, i_mask_reg_2);

      /* storing 8 registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 0, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 0, 0, 8 );

      /* advance output pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_out, (long long)i_micro_kernel_config->datatype_size_in * 16 );

      /* close n footer */
      libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                  i_gp_reg_n_loop, l_n_16mul );
    }

    /* transpose n_16_rem x 8 block */
    if ( l_n_16rem > 0 ) {
      /* load 8 registers with two half rows */
      {
        const unsigned int ld_idx[8] = { 0x0, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe };
        unsigned int l_mask_regs[2] = { 0 };
        l_mask_regs[0] = 0;    l_mask_regs[1] = i_mask_reg_0;
        libxsmm_generator_transform_Xway_half_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                 i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                                 ld_idx, 8, LIBXSMM_X86_INSTR_VBROADCASTI32X8, 8, l_mask_regs, l_n_16rem );
      }

      /* transpose two 8x8 blocks */
      libxsmm_generator_transform_two_8x8_32bit_norm_to_normt_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 8, i_mask_reg_1, i_mask_reg_2);

      /* storing 8 registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 0, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 1, i_mask_reg_5, 8 );
    }

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (8LL * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_in * l_n_16mul) );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                     i_gp_reg_in, ((long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * l_n_16mul) - (8LL * i_micro_kernel_config->datatype_size_in) );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, l_m_8mul );
  }

  /* Transpose m_8rem blocks */
  if ( l_m_8rem > 0 ) {

    /* transpose 16 x m_8rem blocks */
    if ( l_n_16mul > 0 ) {
      /* open n loop */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_n_loop, 16);

      /* load 8 registers with two half rows */
      {
        const unsigned int ld_idx[8] = { 0x0, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe };
        unsigned int l_mask_regs[2] = { 0 };
        l_mask_regs[0] = i_mask_reg_3;    l_mask_regs[1] = i_mask_reg_4;
        libxsmm_generator_transform_Xway_half_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                 i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                                 ld_idx, 8, LIBXSMM_X86_INSTR_VBROADCASTI32X8, 8, l_mask_regs, 16 );
      }

      /* advance input pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_in, (long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * 16 );

      /* transpose two 8x8 blocks */
      libxsmm_generator_transform_two_8x8_32bit_norm_to_normt_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 8, i_mask_reg_1, i_mask_reg_2);

      /* storing l_m_8rem registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 0, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 0, 0, l_m_8rem );
      /* advance output pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_out, (long long)i_micro_kernel_config->datatype_size_out * 16 );

      /* close n footer */
      libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                  i_gp_reg_n_loop, l_n_16mul );
    }

    /* transpose n_16rem x m_8rem block */
    if ( l_n_16rem > 0 ) {
      /* load 8 registers with two half rows */
      {
        const unsigned int ld_idx[8] = { 0x0, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe };
        unsigned int l_mask_regs[2] = { 0 };
        l_mask_regs[0] = i_mask_reg_3;    l_mask_regs[1] = i_mask_reg_4;
        libxsmm_generator_transform_Xway_half_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                 i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                                 ld_idx, 8, LIBXSMM_X86_INSTR_VBROADCASTI32X8, 8, l_mask_regs, l_n_16rem );
      }

      /* transpose two 8x8 blocks */
      libxsmm_generator_transform_two_8x8_32bit_norm_to_normt_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 8, i_mask_reg_1, i_mask_reg_2);

      /* storing l_m_8rem registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 0, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 1, i_mask_reg_5, l_m_8rem );
    }
  }
}

/*
 * Calling convention, this kernel assumes that all GPRs with are not in the argument list
 * and all xmm/ymm/zmm/tmm are caller save
 *
 * TODO; stack local variables....
 * */
LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_32bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                         libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                         const unsigned int                      i_gp_reg_in,
                                                                         const unsigned int                      i_gp_reg_out,
                                                                         const unsigned int                      i_gp_reg_m_loop,
                                                                         const unsigned int                      i_gp_reg_n_loop,
                                                                         const unsigned int                      i_gp_reg_mask,
                                                                         const unsigned int                      i_gp_reg_mask_2,
                                                                         const unsigned int                      i_mask_reg_0,
                                                                         const unsigned int                      i_mask_reg_1,
                                                                         const unsigned int                      i_mask_reg_2,
                                                                         const unsigned int                      i_mask_reg_3,
                                                                         const unsigned int                      i_mask_reg_4,
                                                                         const unsigned int                      i_mask_reg_5,
                                                                         const unsigned int                      i_mask_reg_6,
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {


#if USE_OPTIMIZED_AVX512_TRANSPOSE == 1
  /* optimized shuffle network for SIMD aligned sizes */
  if ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) && (io_generated_code->arch < LIBXSMM_X86_ALLFEAT) ) {
    /* codepath optimized for SPR */
    libxsmm_generator_transform_norm_to_normt_32bit_avx512_spr_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_in, i_gp_reg_out,
                                                                            i_gp_reg_m_loop, i_gp_reg_n_loop, i_gp_reg_mask, i_gp_reg_mask_2,
                                                                            i_mask_reg_0, i_mask_reg_1, i_mask_reg_2, i_mask_reg_3, i_mask_reg_4,
                                                                            i_mask_reg_5, i_mask_reg_6, i_micro_kernel_config, i_mateltwise_desc );
  } else {
    /* codepath optimized for CLX */
    libxsmm_generator_transform_norm_to_normt_32bit_avx512_pre_spr_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_in, i_gp_reg_out,
                                                                                i_gp_reg_m_loop, i_gp_reg_n_loop, i_gp_reg_mask,
                                                                                i_mask_reg_0, i_mask_reg_1, i_mask_reg_2, i_mask_reg_3, i_mask_reg_4,
                                                                                i_mask_reg_5, i_micro_kernel_config, i_mateltwise_desc );
  }
#elif COPY_UPPER_BOUND == 1
  libxsmm_generator_transform_copy_avx512_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_in, i_gp_reg_out,
                                                       i_gp_reg_m_loop, i_gp_reg_n_loop, i_micro_kernel_config, i_mateltwise_desc->ldi,
                                                       i_mateltwise_desc->ldo, i_mateltwise_desc->m, i_mateltwise_desc->n, 16 );
#else
  libxsmm_generator_transform_norm_to_normt_mbit_scalar_avx512_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_in, i_gp_reg_out,
                                                                            i_gp_reg_m_loop, i_gp_reg_n_loop, i_gp_reg_mask, i_mask_reg_2, i_mask_reg_3,
                                                                            i_micro_kernel_config, i_mateltwise_desc );
#endif
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_four_16x16_08bit_norm_to_normt_avx512( libxsmm_generated_code* io_generated_code,
                                                                        const char              i_vector_name,
                                                                        const unsigned int      i_vec_reg_srcdst_start,
                                                                        const unsigned int      i_vec_reg_tmp_start )
{
  if ( (i_vec_reg_srcdst_start + 16 > i_vec_reg_tmp_start) && (i_vec_reg_tmp_start + 16 > i_vec_reg_srcdst_start) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* first shuffle stage */
  {
    unsigned char l_in_idx[16] = { 0x0, 0x0, 0x2, 0x2, 0x4, 0x4, 0x6, 0x6, 0x8, 0x8, 0xa, 0xa, 0xc, 0xc, 0xe, 0xe};
    libxsmm_generator_transform_Xway_unpack_network_avx_avx512( io_generated_code, i_vector_name,
                                                                l_in_idx, i_vec_reg_srcdst_start, i_vec_reg_tmp_start, 1,
                                                                LIBXSMM_X86_INSTR_VPUNPCKLBW, LIBXSMM_X86_INSTR_VPUNPCKHBW, 16 );
  }

  /* second shuffle stage */
  {
    unsigned char l_in_idx[16] = { 0x0, 0x0, 0x1, 0x1, 0x4, 0x4, 0x5, 0x5, 0x8, 0x8, 0x9, 0x9, 0xc, 0xc, 0xd, 0xd};
    libxsmm_generator_transform_Xway_unpack_network_avx_avx512( io_generated_code, i_vector_name,
                                                                l_in_idx, i_vec_reg_tmp_start, i_vec_reg_srcdst_start, 2,
                                                                LIBXSMM_X86_INSTR_VPUNPCKLWD, LIBXSMM_X86_INSTR_VPUNPCKHWD, 16 );
  }

  /* third shuffle stage */
  {
    unsigned char l_in_idx[16] = { 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x8, 0x8, 0x9, 0x9, 0xa, 0xa, 0xb, 0xb};
    libxsmm_generator_transform_Xway_unpack_network_avx_avx512( io_generated_code, i_vector_name,
                                                                l_in_idx, i_vec_reg_srcdst_start, i_vec_reg_tmp_start, 4,
                                                                LIBXSMM_X86_INSTR_VPUNPCKLDQ, LIBXSMM_X86_INSTR_VPUNPCKHDQ, 16 );
  }

  /* fourth shuffle stage */
  {
    unsigned char l_in_idx[16] = { 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7};
    libxsmm_generator_transform_Xway_unpack_network_avx_avx512( io_generated_code, i_vector_name,
                                                                l_in_idx, i_vec_reg_tmp_start, i_vec_reg_srcdst_start, 8,
                                                                LIBXSMM_X86_INSTR_VPUNPCKLQDQ, LIBXSMM_X86_INSTR_VPUNPCKHQDQ, 16 );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_four_8x8_16bit_norm_to_normt_avx512( libxsmm_generated_code* io_generated_code,
                                                              const char              i_vector_name,
                                                              const unsigned int      i_vec_reg_src_start,
                                                              const unsigned int      i_vec_reg_dst_start )
{
    if ( (i_vec_reg_src_start + 8 > i_vec_reg_dst_start) && (i_vec_reg_dst_start + 8 > i_vec_reg_src_start) ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
        return;
    }

    /* first shuffle stage */
    {
      unsigned char l_in_idx[8] = { 0x0, 0x0, 0x2, 0x2, 0x4, 0x4, 0x6, 0x6};
      libxsmm_generator_transform_Xway_unpack_network_avx_avx512( io_generated_code, i_vector_name,
                                                              l_in_idx, i_vec_reg_src_start, i_vec_reg_dst_start, 1,
                                                              LIBXSMM_X86_INSTR_VPUNPCKLWD, LIBXSMM_X86_INSTR_VPUNPCKHWD, 8 );
    }

    /* second shuffle stage */
    {
      unsigned char l_in_idx[8] = { 0x0, 0x0, 0x1, 0x1, 0x4, 0x4, 0x5, 0x5};
      libxsmm_generator_transform_Xway_unpack_network_avx_avx512( io_generated_code, i_vector_name,
                                                              l_in_idx, i_vec_reg_dst_start, i_vec_reg_src_start, 2,
                                                              LIBXSMM_X86_INSTR_VPUNPCKLDQ, LIBXSMM_X86_INSTR_VPUNPCKHDQ, 8 );
    }

    /* third shuffle stage */
    {
      unsigned char l_in_idx[8] = { 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3};
      libxsmm_generator_transform_Xway_unpack_network_avx_avx512( io_generated_code, i_vector_name,
                                                              l_in_idx, i_vec_reg_src_start, i_vec_reg_dst_start, 4,
                                                              LIBXSMM_X86_INSTR_VPUNPCKLQDQ, LIBXSMM_X86_INSTR_VPUNPCKHQDQ, 8 );
    }
}


/*
 * Calling convention, this kernel assumes that all GPRs with are not in the argument list
 * and all xmm/ymm/zmm/tmm are caller save
 *
 * TODO; stack local variables....
 * */
LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_16bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                         libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                         const unsigned int                      i_gp_reg_in,
                                                                         const unsigned int                      i_gp_reg_out,
                                                                         const unsigned int                      i_gp_reg_m_loop,
                                                                         const unsigned int                      i_gp_reg_n_loop,
                                                                         const unsigned int                      i_gp_reg_mask,
                                                                         const unsigned int                      i_gp_reg_mask_2,
                                                                         const unsigned int                      i_mask_reg_0,
                                                                         const unsigned int                      i_mask_reg_1,
                                                                         const unsigned int                      i_mask_reg_2,
                                                                         const unsigned int                      i_mask_reg_3,
                                                                         const unsigned int                      i_mask_reg_4,
                                                                         const unsigned int                      i_mask_reg_5,
                                                                         const unsigned int                      i_mask_reg_6,
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {

#if USE_OPTIMIZED_AVX512_TRANSPOSE == 1
  /* optimized shuffle network for SIMD aligned sizes */
  /* Partial 32x8 blocks */
  const unsigned int l_n_32rem = i_mateltwise_desc->n % 32;
  const unsigned int l_m_8rem  = i_mateltwise_desc->m % 8;

  const unsigned int l_n_32mul = i_mateltwise_desc->n - l_n_32rem;
  const unsigned int l_m_8mul =  i_mateltwise_desc->m - l_m_8rem;

  const unsigned int l_m_8rem_odd = (l_m_8rem % 2);

  unsigned int l_mask_regs[4] = { 0 };

  /* set the masks for the load+blend stage */
  unsigned long long l_mask = 0x00f0;
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                   i_gp_reg_mask, l_mask );
  libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
                                     i_gp_reg_mask, i_mask_reg_0 );

  l_mask = 0x0f00;
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                   i_gp_reg_mask, l_mask );
  libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
                                     i_gp_reg_mask, i_mask_reg_1 );
  l_mask = 0xf000;
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                   i_gp_reg_mask, l_mask );
  libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
                                     i_gp_reg_mask, i_mask_reg_2 );

  /* set the masks for the load+blend stage for partial 32x8 blocks */
  if ( l_m_8rem > 0 ) {
    const unsigned int l_m_8rem_mask  = (1 << l_m_8rem) - 1;
    const unsigned int l_m_8rem_masks = 4;
    unsigned int l_i = 0;

    l_mask_regs[0] = i_mask_reg_3; l_mask_regs[1] = i_mask_reg_4;
    l_mask_regs[2] = i_mask_reg_5; l_mask_regs[3] = i_mask_reg_6;
    for (l_i = 0; l_i < l_m_8rem_masks; ++l_i) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask, (long long)l_m_8rem_mask << (l_i * 4) );
      libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mask, l_mask_regs[l_i] );
    }
  }

  if ( l_n_32rem > 0 ) {
    const unsigned int l_n_32rem_mask = (1 << l_n_32rem) - 1;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask_2, l_n_32rem_mask );
    if ( l_m_8rem_odd ) {
      libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mask_2, i_mask_reg_6 );
    }
  }

  /* Transpose x8 blocks */
  if ( l_m_8mul > 0 ) {
    /* use full 128-bit loads */
    l_mask_regs[0] = 0;            l_mask_regs[1] = i_mask_reg_0;
    l_mask_regs[2] = i_mask_reg_1; l_mask_regs[3] = i_mask_reg_2;

    /* open m loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_m_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_m_loop, 8 );

    /* transpose 32x8 blocks */
    if ( l_n_32mul > 0 ) {
      /* open n loop */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_n_loop, 0);
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_n_loop, 32 );

      /* load 8 registers with four quarter rows */
      libxsmm_generator_transform_Xway_quarter_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                  i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                                  LIBXSMM_X86_INSTR_VBROADCASTI32X4, 8, l_mask_regs, 32, 0 );

      /* advance input pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_in, (long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * 32 );

      /* 3-stage shuffle */
      libxsmm_generator_transform_four_8x8_16bit_norm_to_normt_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 8 );

      /* storing 8 registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 8, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 0, 0, 8 );

      /* advance output pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_out, (long long)i_micro_kernel_config->datatype_size_in * 32 );

      /* close n footer */
      libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                  i_gp_reg_n_loop, l_n_32mul );
    }

    /* transpose n_32rem x 32 block */
    if ( l_n_32rem > 0 ) {
      /* set store mask */
      if ( !l_m_8rem_odd ) {
        libxsmm_x86_instruction_mask_move(io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mask_2, i_mask_reg_6);
      }

      /* load 8 registers with four quarter rows */
      libxsmm_generator_transform_Xway_quarter_load_blend_avx512( io_generated_code,
                                                                  i_micro_kernel_config->vector_name,
                                                                  i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                                  LIBXSMM_X86_INSTR_VBROADCASTI32X4, 8, l_mask_regs, l_n_32rem, 0 );

      /* 3-stage shuffle */
      libxsmm_generator_transform_four_8x8_16bit_norm_to_normt_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 8 );

      /* storing 8 registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 8, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 1, i_mask_reg_6, 8 );

      /* restore quarter load masks */
      if ( !l_m_8rem_odd ) {
        libxsmm_x86_instruction_mask_move(io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mask, i_mask_reg_6);
      }
    }

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (8LL * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_in * l_n_32mul) );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                     i_gp_reg_in, ((long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * l_n_32mul) - (8LL * i_micro_kernel_config->datatype_size_in) );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, l_m_8mul );
  }

  /* Transpose m_8rem blocks */
  if ( l_m_8rem > 0 ) {
    /* use partial 128-bit loads */
    l_mask_regs[0] = i_mask_reg_3; l_mask_regs[1] = i_mask_reg_4;
    l_mask_regs[2] = i_mask_reg_5; l_mask_regs[3] = i_mask_reg_6;

    /* transpose 32 x m_8rem blocks */
    if ( l_n_32mul > 0 ) {
      /* open n loop */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_n_loop, 0 );
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_n_loop, 32 );

      /* load 8 registers with four quarter rows */
      libxsmm_generator_transform_Xway_quarter_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                  i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                                  LIBXSMM_X86_INSTR_VMOVDQU16, 8, l_mask_regs, 32, 1 );

      /* advance input pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_in, (long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * 32 );

      /* 3-stage shuffle */
      libxsmm_generator_transform_four_8x8_16bit_norm_to_normt_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 8 );

      /* storing l_m_8rem registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 8, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 0, 0, l_m_8rem );

      /* advance output pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_out, (long long)i_micro_kernel_config->datatype_size_in * 32 );

      /* close n footer */
      libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                  i_gp_reg_n_loop, l_n_32mul );
    }

    /* transpose n_32rem x m_8rem block */
    if ( l_n_32rem > 0 ) {
      /* load 8 registers with four quarter rows */
      libxsmm_generator_transform_Xway_quarter_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                  i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                                  LIBXSMM_X86_INSTR_VMOVDQU16, LIBXSMM_MIN(8,l_n_32rem), l_mask_regs, l_n_32rem, 1 );

      /* 3-stage shuffle */
      libxsmm_generator_transform_four_8x8_16bit_norm_to_normt_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 8 );

      /* restore store mask */
      libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mask_2, i_mask_reg_6 );

      /* storing m_8rem registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 8, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 1, i_mask_reg_6, l_m_8rem );
    }
  }
#elif COPY_UPPER_BOUND == 1
  libxsmm_generator_transform_copy_avx512_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_in, i_gp_reg_out,
                                                       i_gp_reg_m_loop, i_gp_reg_n_loop, i_micro_kernel_config, i_mateltwise_desc->ldi,
                                                       i_mateltwise_desc->ldo, i_mateltwise_desc->m, i_mateltwise_desc->n, 32 );
#else
  libxsmm_generator_transform_norm_to_normt_mbit_scalar_avx512_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_in, i_gp_reg_out,
                                                                            i_gp_reg_m_loop, i_gp_reg_n_loop, i_gp_reg_mask, i_mask_reg_2, i_mask_reg_3,
                                                                            i_micro_kernel_config, i_mateltwise_desc );
#endif
}

/*
 * Calling convention, this kernel assumes that all GPRs with are not in the argument list
 * and all xmm/ymm/zmm/tmm are caller save
 *
 * TODO; stack local variables....
 * */
LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_08bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                         libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                         const unsigned int                      i_gp_reg_in,
                                                                         const unsigned int                      i_gp_reg_out,
                                                                         const unsigned int                      i_gp_reg_m_loop,
                                                                         const unsigned int                      i_gp_reg_n_loop,
                                                                         const unsigned int                      i_gp_reg_mask,
                                                                         const unsigned int                      i_gp_reg_mask_2,
                                                                         const unsigned int                      i_mask_reg_0,
                                                                         const unsigned int                      i_mask_reg_1,
                                                                         const unsigned int                      i_mask_reg_2,
                                                                         const unsigned int                      i_mask_reg_3,
                                                                         const unsigned int                      i_mask_reg_4,
                                                                         const unsigned int                      i_mask_reg_5,
                                                                         const unsigned int                      i_mask_reg_6,
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  /* optimized shuffle network for SIMD aligned sizes */
  /* Partial 64x16 blocks */
  const unsigned int l_n_64rem = i_mateltwise_desc->n % 64;
  const unsigned int l_m_16rem  = i_mateltwise_desc->m % 16;

  const unsigned int l_n_64mul = i_mateltwise_desc->n - l_n_64rem;
  const unsigned int l_m_16mul =  i_mateltwise_desc->m - l_m_16rem;

  const unsigned int l_m_16rem_odd = (l_m_16rem % 4);
  const unsigned int l_ld_instr = ( l_m_16rem_odd ) ? LIBXSMM_X86_INSTR_VMOVDQU8 : LIBXSMM_X86_INSTR_VBROADCASTI32X4;

  unsigned int l_mask_regs[4] = { 0 };

  /* set the masks for the load+blend stage */
  unsigned long long l_mask = 0x00f0;
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                   i_gp_reg_mask, l_mask );
  libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
                                     i_gp_reg_mask, i_mask_reg_0 );

  l_mask = 0x0f00;
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                   i_gp_reg_mask, l_mask );
  libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
                                     i_gp_reg_mask, i_mask_reg_1 );
  l_mask = 0xf000;
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                   i_gp_reg_mask, l_mask );
  libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
                                     i_gp_reg_mask, i_mask_reg_2 );

  /* set the masks for the load+blend stage for partial 64x16 blocks */
  if ( l_m_16rem > 0 ) {
    const unsigned int l_m_16rem_masks = ( l_m_16rem_odd ) ? 1 : 4;
    unsigned int l_i = 0;

    l_mask_regs[0] = i_mask_reg_3; l_mask_regs[1] = i_mask_reg_4;
    l_mask_regs[2] = i_mask_reg_5; l_mask_regs[3] = i_mask_reg_6;

    /* set mask with l_m_16rem_mask = (( l_m_16rem_odd ) ? (1 << l_m_16rem) - 1 : (1 << (l_m_16rem >> 2)) - 1) */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask, 1 );
    if ( l_m_16rem_odd ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SHLQ, i_gp_reg_mask, l_m_16rem );
    }
    else {
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SHLQ, i_gp_reg_mask, (long long)l_m_16rem >> 2 );
    }
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, i_gp_reg_mask, 1 );

    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mask, l_mask_regs[0] );
    for (l_i = 1; l_i < l_m_16rem_masks; ++l_i) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SHLQ, i_gp_reg_mask, 4 );
      libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mask, l_mask_regs[l_i] );
    }
  }

  if ( l_n_64rem > 0 ) {
    /* set mask with l_n_64rem_mask = (1 << l_n_64rem) - 1 */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask_2, 1 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SHLQ, i_gp_reg_mask_2, l_n_64rem );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, i_gp_reg_mask_2, 1 );

    if ( l_m_16rem_odd ) {
      libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVQ_GPR_LD, i_gp_reg_mask_2, i_mask_reg_6 );
    }
  }

  /* Transpose x16 blocks */
  if ( l_m_16mul > 0 ) {
    /* use full 128-bit loads */
    l_mask_regs[0] = 0;            l_mask_regs[1] = i_mask_reg_0;
    l_mask_regs[2] = i_mask_reg_1; l_mask_regs[3] = i_mask_reg_2;

    /* open m loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_m_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_m_loop, 16 );

    /* transpose 64x16 blocks */
    if ( l_n_64mul > 0 ) {
      /* open n loop */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_n_loop, 0);
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_n_loop, 64 );

      /* load 16 registers with four quarter rows */
      libxsmm_generator_transform_Xway_quarter_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                  i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                                  LIBXSMM_X86_INSTR_VBROADCASTI32X4, 16, l_mask_regs, 64, 0 );

      /* advance input pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_in, (long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * 64 );

      /* 4-stage shuffle */
      libxsmm_generator_transform_four_16x16_08bit_norm_to_normt_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 16 );

      /* storing 16 registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 0, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 0, 0, 16 );

      /* advance output pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_out, (long long)i_micro_kernel_config->datatype_size_in * 64 );

      /* close n footer */
      libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                  i_gp_reg_n_loop, l_n_64mul );
    }

    /* transpose n_64rem x 16 block */
    if ( l_n_64rem > 0 ) {
      /* set store mask */
      if ( !l_m_16rem_odd ) {
        libxsmm_x86_instruction_mask_move(io_generated_code, LIBXSMM_X86_INSTR_KMOVQ_GPR_LD, i_gp_reg_mask_2, i_mask_reg_6);
      }

      /* load 16 registers with four quarter rows */
      libxsmm_generator_transform_Xway_quarter_load_blend_avx512( io_generated_code,
                                                                  i_micro_kernel_config->vector_name,
                                                                  i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                                  LIBXSMM_X86_INSTR_VBROADCASTI32X4, 16, l_mask_regs, l_n_64rem, 0 );

      /* 4-stage shuffle */
      libxsmm_generator_transform_four_16x16_08bit_norm_to_normt_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 16 );

      /* storing 16 registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 0, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 1, i_mask_reg_6, 16 );

      /* restore quarter load masks */
      if ( !l_m_16rem_odd ) {
        libxsmm_x86_instruction_mask_move(io_generated_code, LIBXSMM_X86_INSTR_KMOVQ_GPR_LD, i_gp_reg_mask, i_mask_reg_6);
      }
    }

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (16LL * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_in * l_n_64mul) );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                     i_gp_reg_in, ((long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * l_n_64mul) - (16LL * i_micro_kernel_config->datatype_size_in) );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, l_m_16mul );
  }

  /* Transpose m_16rem blocks */
  if ( l_m_16rem > 0 ) {
    /* use partial 128-bit loads */
    l_mask_regs[0] = i_mask_reg_3; l_mask_regs[1] = i_mask_reg_4;
    l_mask_regs[2] = i_mask_reg_5; l_mask_regs[3] = i_mask_reg_6;

    /* transpose 64 x m_16rem blocks */
    if ( l_n_64mul > 0 ) {
      /* open n loop */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_n_loop, 0 );
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_n_loop, 64 );

      /* load 16 registers with four quarter rows */
      libxsmm_generator_transform_Xway_quarter_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                  i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                                  l_ld_instr, 16, l_mask_regs, 64, l_m_16rem_odd );

      /* advance input pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_in, (long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * 64 );

      /* 4-stage shuffle */
      libxsmm_generator_transform_four_16x16_08bit_norm_to_normt_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 16 );

      /* storing l_m_16rem registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 0, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 0, 0, l_m_16rem );

      /* advance output pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_out, (long long)i_micro_kernel_config->datatype_size_in * 64 );

      /* close n footer */
      libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                  i_gp_reg_n_loop, l_n_64mul );
    }

    /* transpose n_64rem x m_16rem block */
    if ( l_n_64rem > 0 ) {
      /* load 16 registers with four quarter rows */
      libxsmm_generator_transform_Xway_quarter_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                  i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                                  l_ld_instr, 16, l_mask_regs, l_n_64rem, l_m_16rem_odd );

      /* 4-stage shuffle */
      libxsmm_generator_transform_four_16x16_08bit_norm_to_normt_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 16 );

      /* restore store mask */
      libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVQ_GPR_LD, i_gp_reg_mask_2, i_mask_reg_6 );

      /* storing m_16rem registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 0, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 1, i_mask_reg_6, l_m_16rem );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_two_8x8_08bit_vnni4_to_vnni4t_avx512( libxsmm_generated_code* io_generated_code,
                                                                       const char              i_vector_name,
                                                                       const unsigned int      i_vec_reg_srcdst_start,
                                                                       const unsigned int      i_shuffle_op,
                                                                       const unsigned int      i_mask_reg_1,
                                                                       const unsigned int      i_mask_reg_2 )
{
  if ( i_vec_reg_srcdst_start + 2 > 30 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* first shuffle stage */
  {
    unsigned char l_in_idx[16] = { 0x0, 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
    libxsmm_generator_transform_Xway_byteshuffle_network_avx512( io_generated_code, i_vector_name,
                                                                 l_in_idx, i_shuffle_op, i_vec_reg_srcdst_start, i_vec_reg_srcdst_start, LIBXSMM_X86_INSTR_VPSHUFB, 2 );
  }

  /* 2rd stage: variable permute network */
  {
    unsigned char l_perm_imm[2] = { 0x44, 0xee };
    unsigned char l_perm_mask[2]; l_perm_mask[0] = (unsigned char)i_mask_reg_1; l_perm_mask[1] = (unsigned char)i_mask_reg_2;

    libxsmm_generator_transform_Xway_permute_network_avx512( io_generated_code, i_vector_name,
                                                             l_perm_mask, l_perm_imm, i_vec_reg_srcdst_start, LIBXSMM_X86_INSTR_VPERMQ_I, 2 );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_two_8x8_16bit_vnni2_to_vnni2t_avx512( libxsmm_generated_code* io_generated_code,
                                                                     const char              i_vector_name,
                                                                     const unsigned int      i_vec_reg_srcdst_start,
                                                                     const unsigned int      i_vec_reg_tmp_start,
                                                                     const unsigned int      i_shuffle_op,
                                                                     const unsigned int      i_mask_reg_1,
                                                                     const unsigned int      i_mask_reg_2 )
{
  if ( (i_vec_reg_srcdst_start + 4 > i_vec_reg_tmp_start) && (i_vec_reg_tmp_start + 4 > i_vec_reg_srcdst_start) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* first shuffle stage */
  {
    unsigned char l_in_idx[16] = { 0x0, 0x1, 0x2, 0x3, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
    libxsmm_generator_transform_Xway_byteshuffle_network_avx512( io_generated_code, i_vector_name,
                                                                 l_in_idx, i_shuffle_op, i_vec_reg_srcdst_start, i_vec_reg_tmp_start, LIBXSMM_X86_INSTR_VPSHUFB, 4 );
  }

  /* second shuffle stage */
  {
    unsigned char l_in_idx[16] = { 0x0, 0x0, 0x2, 0x2, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
    libxsmm_generator_transform_Xway_unpack_network_avx_avx512( io_generated_code, i_vector_name,
                                                            l_in_idx, i_vec_reg_tmp_start, i_vec_reg_srcdst_start, 1,
                                                            LIBXSMM_X86_INSTR_VPUNPCKLQDQ, LIBXSMM_X86_INSTR_VPUNPCKHQDQ, 4 );
  }

  /* 3rd stage: variable permute network */
  {
    unsigned char l_perm_imm[2] = { 0x44, 0xee };
    unsigned char l_perm_mask[2]; l_perm_mask[0] = (unsigned char)i_mask_reg_1; l_perm_mask[1] = (unsigned char)i_mask_reg_2;

    libxsmm_generator_transform_Xway_permute_network_avx512( io_generated_code, i_vector_name,
                                                             l_perm_mask, l_perm_imm, i_vec_reg_srcdst_start, LIBXSMM_X86_INSTR_VPERMQ_I, 4 );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_four_4x4_16bit_vnni2_to_vnni2t_avx512( libxsmm_generated_code* io_generated_code,
                                                                     const char              i_vector_name,
                                                                     const unsigned int      i_vec_reg_srcdst_start,
                                                                     const unsigned int      i_vec_reg_tmp_start,
                                                                     const unsigned int      i_shuffle_op )
{
  if ( (i_vec_reg_srcdst_start + 2 > i_vec_reg_tmp_start) && (i_vec_reg_tmp_start + 2 > i_vec_reg_srcdst_start) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* first shuffle stage */
  {
    unsigned char l_in_idx[16] = { 0x0, 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
    libxsmm_generator_transform_Xway_byteshuffle_network_avx512( io_generated_code, i_vector_name,
                                                                 l_in_idx, i_shuffle_op, i_vec_reg_srcdst_start, i_vec_reg_tmp_start, LIBXSMM_X86_INSTR_VPSHUFB, 2 );
  }

  /* second shuffle stage */
  {
    unsigned char l_in_idx[16] = { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
    libxsmm_generator_transform_Xway_unpack_network_avx_avx512( io_generated_code, i_vector_name,
                                                            l_in_idx, i_vec_reg_tmp_start, i_vec_reg_srcdst_start, 1,
                                                            LIBXSMM_X86_INSTR_VPUNPCKLQDQ, LIBXSMM_X86_INSTR_VPUNPCKHQDQ, 2 );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni4t_16bit_avx512_vl512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                             libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                             const unsigned int                      i_gp_reg_in,
                                                                             const unsigned int                      i_gp_reg_out,
                                                                             const unsigned int                      i_gp_reg_m_loop,
                                                                             const unsigned int                      i_gp_reg_n_loop,
                                                                             const unsigned int                      i_gp_reg_mask,
                                                                             const unsigned int                      i_mask_reg_0,
                                                                             const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                             const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int l_ldi = i_mateltwise_desc->ldi*4;
  unsigned int l_ldo = i_mateltwise_desc->ldo*4;
  unsigned int l_zmm = 0, i = 0;
  unsigned int l_perm_reg = 31;
  unsigned int m_unroll_factor = 16;
  unsigned int n4_chunks_odd = i_mateltwise_desc->n % 8;
  unsigned int N = i_mateltwise_desc->n;
  unsigned int n_blocks = ((n4_chunks_odd > 0) && (N > 4)) ? 2 : 1;
  unsigned long long l_mask = 0xff00;
  short perm_table[32] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 16, 20, 24, 28, 17, 21, 25, 29, 18, 22, 26, 30, 19, 23, 27, 31};

  /* Setup permute register */
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) perm_table, "perm_table_", i_micro_kernel_config->vector_name, l_perm_reg);

  /* set the masks for the load+blend stage */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                   i_gp_reg_mask, l_mask );
  libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
                                     i_gp_reg_mask, i_mask_reg_0 );

  while (m_unroll_factor * 4 > i_mateltwise_desc->m) {
    m_unroll_factor--;
  }

  while (i_mateltwise_desc->m % (m_unroll_factor * 4) != 0) {
    m_unroll_factor--;
  }

  for (i = 0; i < n_blocks; i++) {
    if (n4_chunks_odd > 0) {
      N = N - 4;
    }
    if ((i == 0) && (N >= 8)) {
      /* open n loop */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_n_loop, 4 );
    }

    /* open m loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_m_loop, 4 * m_unroll_factor );

    /* Load register */
    for (l_zmm = 0; l_zmm < m_unroll_factor; l_zmm++) {
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                        i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, l_zmm * 16 * i_micro_kernel_config->datatype_size_in,
                                        'y', l_zmm, 0, 1, 0 );
      if ((i == 0) && (N >= 8)) {
        libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, LIBXSMM_X86_INSTR_VBROADCASTI32X8,
                                          i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, (l_zmm * 16 + l_ldi) * i_micro_kernel_config->datatype_size_in,
                                          i_micro_kernel_config->vector_name, l_zmm, i_mask_reg_0, 0 , 0 );
      }
    }

    /* Permute register */
    for (l_zmm = 0; l_zmm < m_unroll_factor; l_zmm++) {
      libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMW, i_micro_kernel_config->vector_name,
                                                              l_zmm, l_perm_reg, l_zmm,  0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
    }

    /* Store register */
    for (l_zmm = 0; l_zmm < m_unroll_factor; l_zmm++) {
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                        i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, l_zmm * l_ldo * i_micro_kernel_config->datatype_size_out,
                                        ((i == 1) || (N <= 4)) ? 'y' : i_micro_kernel_config->vector_name, l_zmm, 0, 1, 1 );
    }
    /* Advance input ptr */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_in, (long long)16 * m_unroll_factor * i_micro_kernel_config->datatype_size_in );

    /* Advance output ptr */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_out, (long long)l_ldo * m_unroll_factor * i_micro_kernel_config->datatype_size_out );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_m_loop, i_mateltwise_desc->m );

    if ((i == 0) && (N >= 8)) {
      /* Advance input */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_in, (long long)(2 * l_ldi - 4 * i_mateltwise_desc->m) * i_micro_kernel_config->datatype_size_in );
      /* Advance output */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, i_gp_reg_out, (long long)(l_ldo * (i_mateltwise_desc->m/4) - 32) * i_micro_kernel_config->datatype_size_out );
      /* close n loop */
      libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_n_loop, N/2 );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni4t_16bit_avx512_vl256_microkernel( libxsmm_generated_code*             io_generated_code,
                                                                             libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                             const unsigned int                      i_gp_reg_in,
                                                                             const unsigned int                      i_gp_reg_out,
                                                                             const unsigned int                      i_gp_reg_m_loop,
                                                                             const unsigned int                      i_gp_reg_n_loop,
                                                                             const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                             const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int l_ldi = i_mateltwise_desc->ldi*4;
  unsigned int l_ldo = i_mateltwise_desc->ldo*4;
  unsigned int l_zmm = 0;
  unsigned int l_perm_reg = 31;
  unsigned int m_unroll_factor = 16;

  short perm_table[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};

  /* Setup permute register */
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) perm_table, "perm_table_", i_micro_kernel_config->vector_name, l_perm_reg);

  while (m_unroll_factor * 4 > i_mateltwise_desc->m) {
    m_unroll_factor--;
  }

  while (i_mateltwise_desc->m % (m_unroll_factor * 4) != 0) {
    m_unroll_factor--;
  }

  /* open n loop */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_n_loop, 4 );

  /* open m loop */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_m_loop, 4 * m_unroll_factor );

  /* Load register */
  for (l_zmm = 0; l_zmm < m_unroll_factor; l_zmm++) {
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, l_zmm * 16 * i_micro_kernel_config->datatype_size_in,
                                      i_micro_kernel_config->vector_name, l_zmm, 0, 1, 0 );
  }

  /* Permute register */
  for (l_zmm = 0; l_zmm < m_unroll_factor; l_zmm++) {
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMW, i_micro_kernel_config->vector_name,
                                                            l_zmm, l_perm_reg, l_zmm,  0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
  }

  /* Store register */
  for (l_zmm = 0; l_zmm < m_unroll_factor; l_zmm++) {
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                      i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, l_zmm * l_ldo * i_micro_kernel_config->datatype_size_out,
                                      i_micro_kernel_config->vector_name, l_zmm, 0, 1, 1 );
  }
  /* Advance input ptr */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_in, (long long)16 * m_unroll_factor * i_micro_kernel_config->datatype_size_in );

  /* Advance output ptr */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_out, (long long)l_ldo * m_unroll_factor * i_micro_kernel_config->datatype_size_out );

  /* close m loop */
  libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_m_loop, i_mateltwise_desc->m );

  /* Advance input */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_in, (long long)(l_ldi - 4 * i_mateltwise_desc->m) * i_micro_kernel_config->datatype_size_in );

  /* Advance output */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, i_gp_reg_out, (long long)(l_ldo * (i_mateltwise_desc->m/4) - 16) * i_micro_kernel_config->datatype_size_out );

  /* close n loop */
  libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_n_loop, i_mateltwise_desc->n );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni4t_16bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                             libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                             const unsigned int                      i_gp_reg_in,
                                                                             const unsigned int                      i_gp_reg_out,
                                                                             const unsigned int                      i_gp_reg_m_loop,
                                                                             const unsigned int                      i_gp_reg_n_loop,
                                                                             const unsigned int                      i_gp_reg_mask,
                                                                             const unsigned int                      i_mask_reg_0,
                                                                             const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                             const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  if ((io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX)) {
    libxsmm_generator_transform_vnni4_to_vnni4t_16bit_avx512_vl256_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop, i_micro_kernel_config, i_mateltwise_desc );
  } else {
    libxsmm_generator_transform_vnni4_to_vnni4t_16bit_avx512_vl512_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop, i_gp_reg_mask, i_mask_reg_0, i_micro_kernel_config, i_mateltwise_desc );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni2_to_vnni2t_16bit_avx512_spr_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                             libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                             const unsigned int                      i_gp_reg_in,
                                                                             const unsigned int                      i_gp_reg_out,
                                                                             const unsigned int                      i_gp_reg_m_loop,
                                                                             const unsigned int                      i_gp_reg_n_loop,
                                                                             const unsigned int                      i_gp_reg_mask,
                                                                             const unsigned int                      i_gp_reg_mask_2,
                                                                             const unsigned int                      i_mask_reg_0,
                                                                             const unsigned int                      i_mask_reg_1,
                                                                             const unsigned int                      i_mask_reg_2,
                                                                             const unsigned int                      i_mask_reg_3,
                                                                             const unsigned int                      i_mask_reg_4,
                                                                             const unsigned int                      i_mask_reg_5,
                                                                             const unsigned int                      i_mask_reg_6,
                                                                             const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                             const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int l_ldi = i_mateltwise_desc->ldi*2;
  unsigned int l_ldo = i_mateltwise_desc->ldo*2;

  /* byte shuffle operand */
  unsigned int  l_shuffle_cntl[16] = { 0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a, 0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a,
                                       0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a, 0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a };
  unsigned int  l_shuffle_op = 31;

  /* Partial 8x8 blocks */
  const unsigned int l_n_8rem = (i_mateltwise_desc->n/2) % 8;
  const unsigned int l_m_8rem = (i_mateltwise_desc->m*2) % 8;

  const unsigned int l_n_8mul = (i_mateltwise_desc->n/2) - l_n_8rem;
  const unsigned int l_m_8mul = (i_mateltwise_desc->m*2) - l_m_8rem;

  /* set the masks for the load+blend stage */
  unsigned long long l_mask = 0x00f0;

  libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char *) l_shuffle_cntl,
                                                      "vnni2_to_vnni2t_shufl_", i_micro_kernel_config->vector_name, l_shuffle_op);

  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                   i_gp_reg_mask, l_mask );
  libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
                                     i_gp_reg_mask, i_mask_reg_0 );

  l_mask = 0x0f00;
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                   i_gp_reg_mask, l_mask );
  libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
                                     i_gp_reg_mask, i_mask_reg_1 );
  l_mask = 0xf000;
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                   i_gp_reg_mask, l_mask );
  libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
                                     i_gp_reg_mask, i_mask_reg_2 );

  /* set the masks for the load+blend stage for partial 8x8 blocks */
  if ( l_m_8rem > 0 ) {
    unsigned int l_mask_regs[4] = { 0 };
    const unsigned int l_m_8rem_mask = (1 << (l_m_8rem >> 1)) - 1;
    unsigned int l_i = 0;
    l_mask_regs[0] = i_mask_reg_3;    l_mask_regs[1] = i_mask_reg_4;
    l_mask_regs[2] = i_mask_reg_5;    l_mask_regs[3] = i_mask_reg_6;

    for (l_i = 0; l_i < 4; ++l_i) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask, (long long)l_m_8rem_mask << (l_i * 4) );
      libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mask, l_mask_regs[l_i] );
    }
  }

  if ( l_n_8rem > 0 ) {
    const unsigned int l_n_8rem_mask = (1 << (l_n_8rem * 4)) - 1;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask_2, l_n_8rem_mask );
  }

  /* Transpose x8 blocks */
  if ( l_m_8mul > 0 ) {
    /* open m loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_m_loop, 8 );

    /* transpose 8x8 blocks */
    if ( l_n_8mul > 0 ) {
      /* open n loop */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_n_loop, 8 );

      /* load 2 registers with four quarter rows */
      {
        unsigned int l_mask_regs[4] = { 0 };
        l_mask_regs[0] = 0;               l_mask_regs[1] = i_mask_reg_0;
        l_mask_regs[2] = i_mask_reg_1;    l_mask_regs[3] = i_mask_reg_2;
        libxsmm_generator_transform_Xway_quarter_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                    i_gp_reg_in, 0, l_ldi * i_micro_kernel_config->datatype_size_in,
                                                                    LIBXSMM_X86_INSTR_VBROADCASTI32X4, 2, l_mask_regs, 8, 0 );
      }

      /* advance input pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_in, (long long)l_ldi * i_micro_kernel_config->datatype_size_in * 8 );

      /* transpose four 4x4 blocks */
      libxsmm_generator_transform_four_4x4_16bit_vnni2_to_vnni2t_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 2, l_shuffle_op);

      /* storing 2 registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 0, l_ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 0, 0, 2 );
      /* advance output pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_out, (long long)i_micro_kernel_config->datatype_size_out * 32 );

      /* close n footer */
      libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                  i_gp_reg_n_loop, l_n_8mul );
    }

    /* transpose n_8rem x 8 block */
    if ( l_n_8rem > 0 ) {
      /* set store mask */
      libxsmm_x86_instruction_mask_move(io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mask_2, i_mask_reg_6);

      /* load 2 registers with four quarter rows */
      {
        unsigned int l_mask_regs[4] = { 0 };
        l_mask_regs[0] = 0;               l_mask_regs[1] = i_mask_reg_0;
        l_mask_regs[2] = i_mask_reg_1;    l_mask_regs[3] = i_mask_reg_2;
        libxsmm_generator_transform_Xway_quarter_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                    i_gp_reg_in, 0, l_ldi * i_micro_kernel_config->datatype_size_in,
                                                                    LIBXSMM_X86_INSTR_VBROADCASTI32X4, 2, l_mask_regs, l_n_8rem, 0 );
      }

      /* transpose four 4x4 blocks */
      libxsmm_generator_transform_four_4x4_16bit_vnni2_to_vnni2t_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 2, l_shuffle_op);

      /* storing 2 registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 0, l_ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 1, i_mask_reg_6, 2 );
      /* restore quarter load masks */
      libxsmm_x86_instruction_mask_move(io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mask, i_mask_reg_6);
    }

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (2LL * l_ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_in * l_n_8mul * 4) );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                     i_gp_reg_in, ((long long)l_ldi * i_micro_kernel_config->datatype_size_in * l_n_8mul) - (8LL * i_micro_kernel_config->datatype_size_in) );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, l_m_8mul );
  }

  /* Transpose m_8rem blocks */
  if ( l_m_8rem > 0 ) {

    /* transpose 32 x m_8rem blocks */
    if ( l_n_8mul > 0 ) {
      /* open n loop */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0 );
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_n_loop, 8 );

      /* load 2 registers with four quarter rows */
      {
        unsigned int l_mask_regs[4] = { 0 };
        l_mask_regs[0] = i_mask_reg_3;    l_mask_regs[1] = i_mask_reg_4;
        l_mask_regs[2] = i_mask_reg_5;    l_mask_regs[3] = i_mask_reg_6;
        libxsmm_generator_transform_Xway_quarter_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                    i_gp_reg_in, 0, l_ldi * i_micro_kernel_config->datatype_size_in,
                                                                    LIBXSMM_X86_INSTR_VBROADCASTI32X4, 2, l_mask_regs, 8, 0 );
      }

      /* transpose four 4x4 blocks */
      libxsmm_generator_transform_four_4x4_16bit_vnni2_to_vnni2t_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 2, l_shuffle_op);

      /* storing l_m_8rem>>2 registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 0, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 0, 0, l_m_8rem>>2 );

      /* advance output pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_out, (long long)i_micro_kernel_config->datatype_size_in * 32 );

      /* advance input pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_in, (long long)l_ldi * i_micro_kernel_config->datatype_size_in * 8 );

      /* close n footer */
      libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                  i_gp_reg_n_loop, l_n_8mul );
    }

    /* transpose n_8rem x m_8rem block */
    if ( l_n_8rem > 0 ) {
      /* load 2 registers with four quarter rows */
      {
        unsigned int l_mask_regs[4] = { 0 };
        l_mask_regs[0] = i_mask_reg_3;    l_mask_regs[1] = i_mask_reg_4;
        l_mask_regs[2] = i_mask_reg_5;    l_mask_regs[3] = i_mask_reg_6;
        libxsmm_generator_transform_Xway_quarter_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                    i_gp_reg_in, 0, l_ldi * i_micro_kernel_config->datatype_size_in,
                                                                    LIBXSMM_X86_INSTR_VBROADCASTI32X4, 2, l_mask_regs, l_n_8rem, 0 );
      }

      /* transpose four 4x4 blocks */
      libxsmm_generator_transform_four_4x4_16bit_vnni2_to_vnni2t_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 2, l_shuffle_op);

      /* restore store mask */
      libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mask_2, i_mask_reg_6 );

      /* storing m_8rem registers>>2 */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 0, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 1, i_mask_reg_6, l_m_8rem>>2 );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni2_to_vnni2t_16bit_avx512_pre_spr_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                 libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                 const unsigned int                      i_gp_reg_in,
                                                                                 const unsigned int                      i_gp_reg_out,
                                                                                 const unsigned int                      i_gp_reg_m_loop,
                                                                                 const unsigned int                      i_gp_reg_n_loop,
                                                                                 const unsigned int                      i_gp_reg_mask,
                                                                                 const unsigned int                      i_mask_reg_0,
                                                                                 const unsigned int                      i_mask_reg_1,
                                                                                 const unsigned int                      i_mask_reg_2,
                                                                                 const unsigned int                      i_mask_reg_3,
                                                                                 const unsigned int                      i_mask_reg_4,
                                                                                 const unsigned int                      i_mask_reg_5,
                                                                                 const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int l_ldi = i_mateltwise_desc->ldi*2;
  unsigned int l_ldo = i_mateltwise_desc->ldo*2;

  /* byte shuffle operand */
  unsigned int  l_shuffle_cntl[16] = { 0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a, 0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a,
                                       0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a, 0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a };
  unsigned int  l_shuffle_op = 31;

  /* Partial 8x16 blocks */
  const unsigned int l_n_8rem = (i_mateltwise_desc->n/2) % 8;
  const unsigned int l_m_16rem = (i_mateltwise_desc->m*2) % 16;

  const unsigned int l_n_8mul = (i_mateltwise_desc->n/2) - l_n_8rem;
  const unsigned int l_m_16mul = (i_mateltwise_desc->m*2) - l_m_16rem;

  /* set the masks for the load+blend stage */
  unsigned long long l_mask = 0xff00;

  /* set the masks for the load+blend stage for partial 8x16 blocks */
  if ( l_m_16rem > 0 ) {
    unsigned int l_mask_regs[2] = { 0 };
    const unsigned int l_m_16rem_mask = (1 << (l_m_16rem >> 1)) - 1;
    unsigned int l_i = 0;
    l_mask_regs[0] = i_mask_reg_3;    l_mask_regs[1] = i_mask_reg_4;

    for (l_i = 0; l_i < 2; ++l_i) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask, (long long)l_m_16rem_mask << (l_i * 8) );
      libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mask, l_mask_regs[l_i] );
    }
  }

  if ( l_n_8rem > 0 ) {
    const unsigned int l_n_8rem_mask = (1 << (l_n_8rem * 4)) - 1;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask, l_n_8rem_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
                                       i_gp_reg_mask, i_mask_reg_5 );
  }

  libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char *) l_shuffle_cntl,
                                                      "vnni2_to_vnni2t_shufl_", i_micro_kernel_config->vector_name, l_shuffle_op);

  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                   i_gp_reg_mask, l_mask );
  libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
                                     i_gp_reg_mask, i_mask_reg_0 );

  /* set the masks for the permute stage */
  /* even quarters mask */
  l_mask = 0xcc;
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                   i_gp_reg_mask, l_mask );
  libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVB_GPR_LD,
                                     i_gp_reg_mask, i_mask_reg_1 );

  /* odd quarter masks */
  l_mask = 0x33;
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                   i_gp_reg_mask, l_mask );
  libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVB_GPR_LD,
                                     i_gp_reg_mask, i_mask_reg_2 );

  /* Transpose x16 blocks */
  if ( l_m_16mul > 0 ) {
    /* open m loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_m_loop, 16 );

    /* transpose 8x16 blocks */
    if ( l_n_8mul > 0 ) {

      /* open n loop */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_n_loop, 8 );

      /* load 4 registers with two half rows */
      {
        const unsigned int ld_idx[32] = { 0, 2, 4, 6 };
        unsigned int l_mask_regs[2] = { 0 };
        l_mask_regs[0] = 0;    l_mask_regs[1] = i_mask_reg_0;
        libxsmm_generator_transform_Xway_half_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                 i_gp_reg_in, 0, l_ldi * i_micro_kernel_config->datatype_size_in,
                                                                 ld_idx, 4, LIBXSMM_X86_INSTR_VBROADCASTI32X8, 4, l_mask_regs, 8 );
      }

      /* advance input pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_in, (long long)l_ldi * i_micro_kernel_config->datatype_size_in * 8 );

      /* transpose two 8x8 blocks */
      libxsmm_generator_transform_two_8x8_16bit_vnni2_to_vnni2t_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 4, l_shuffle_op, i_mask_reg_1, i_mask_reg_2);

      /* storing 4 registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 0, l_ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 0, 0, 4 );

      /* advance output pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_out, (long long)i_micro_kernel_config->datatype_size_out * 32 );

      /* close n footer */
      libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                  i_gp_reg_n_loop, l_n_8mul );
    }

    /* transpose n_8_rem x 16 block */
    if ( l_n_8rem > 0 ) {
      /* load 4 registers with two half rows */
      {
        const unsigned int ld_idx[32] = { 0, 2, 4, 6 };
        unsigned int l_mask_regs[2] = { 0 };
        l_mask_regs[0] = 0;    l_mask_regs[1] = i_mask_reg_0;
        libxsmm_generator_transform_Xway_half_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                 i_gp_reg_in, 0, l_ldi * i_micro_kernel_config->datatype_size_in,
                                                                 ld_idx, 4, LIBXSMM_X86_INSTR_VBROADCASTI32X8, 4, l_mask_regs, l_n_8rem );
      }

      /* transpose two 8x8 blocks */
      libxsmm_generator_transform_two_8x8_16bit_vnni2_to_vnni2t_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 4, l_shuffle_op, i_mask_reg_1, i_mask_reg_2);

      /* storing 4 registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 0, l_ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 1, i_mask_reg_5, 4 );
    }

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_out, (4LL * l_ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_out * l_n_8mul * 4) );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                     i_gp_reg_in, ((long long)l_ldi * i_micro_kernel_config->datatype_size_in * l_n_8mul) - (16LL * i_micro_kernel_config->datatype_size_in) );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, l_m_16mul );
  }

  /* Transpose m_8rem blocks */
  if ( l_m_16rem > 0 ) {

    /* transpose 8 x m_16rem blocks */
    if ( l_n_8mul > 0 ) {
      /* open n loop */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_n_loop, 8);

      /* load 4 registers with two half rows */
      {
        const unsigned int ld_idx[32] = { 0, 2, 4, 6 };
        unsigned int l_mask_regs[2] = { 0 };
        l_mask_regs[0] = i_mask_reg_3;    l_mask_regs[1] = i_mask_reg_4;
        libxsmm_generator_transform_Xway_half_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                 i_gp_reg_in, 0, l_ldi * i_micro_kernel_config->datatype_size_in,
                                                                 ld_idx, 4, LIBXSMM_X86_INSTR_VBROADCASTI32X8, 4, l_mask_regs, 8 );
      }

      /* advance input pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_in, (long long)l_ldi * i_micro_kernel_config->datatype_size_in * 8 );

      /* transpose two 8x8 blocks */
      libxsmm_generator_transform_two_8x8_16bit_vnni2_to_vnni2t_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 4, l_shuffle_op, i_mask_reg_1, i_mask_reg_2);

      /* storing l_m_16rem>>2 registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 0, l_ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 0, 0, l_m_16rem>>2 );

      /* advance output pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_out, (long long)i_micro_kernel_config->datatype_size_out * 32 );


      /* close n footer */
      libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                  i_gp_reg_n_loop, l_n_8mul );

    }

    /* transpose n_8rem x m_16rem block */
    if ( l_n_8rem > 0 ) {
      /* load 4 registers with two half rows */
      {
        const unsigned int ld_idx[32] = { 0, 2, 4, 6 };
        unsigned int l_mask_regs[2] = { 0 };
        l_mask_regs[0] = i_mask_reg_3;    l_mask_regs[1] = i_mask_reg_4;
        libxsmm_generator_transform_Xway_half_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                 i_gp_reg_in, 0, l_ldi * i_micro_kernel_config->datatype_size_in,
                                                                 ld_idx, 4, LIBXSMM_X86_INSTR_VBROADCASTI32X8, 4, l_mask_regs, l_n_8rem );
      }

      /* transpose two 8x8 blocks */
      libxsmm_generator_transform_two_8x8_16bit_vnni2_to_vnni2t_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, 4, l_shuffle_op, i_mask_reg_1, i_mask_reg_2);

      /* storing l_m_16rem>>2 registers */
      libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              i_gp_reg_out, 0, l_ldo * i_micro_kernel_config->datatype_size_out,
                                                              i_micro_kernel_config->vmove_instruction_out, 1, i_mask_reg_5, l_m_16rem>>2 );
    }
  }
}

/*
 * Calling convention, this kernel assumes that all GPRs with are not in the argument list
 * and all xmm/ymm/zmm/tmm are caller save
 *
 * TODO; stack local variables....
 * */
LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni2_to_vnni2t_16bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                         libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                         const unsigned int                      i_gp_reg_in,
                                                                         const unsigned int                      i_gp_reg_out,
                                                                         const unsigned int                      i_gp_reg_m_loop,
                                                                         const unsigned int                      i_gp_reg_n_loop,
                                                                         const unsigned int                      i_gp_reg_mask,
                                                                         const unsigned int                      i_gp_reg_mask_2,
                                                                         const unsigned int                      i_mask_reg_0,
                                                                         const unsigned int                      i_mask_reg_1,
                                                                         const unsigned int                      i_mask_reg_2,
                                                                         const unsigned int                      i_mask_reg_3,
                                                                         const unsigned int                      i_mask_reg_4,
                                                                         const unsigned int                      i_mask_reg_5,
                                                                         const unsigned int                      i_mask_reg_6,
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {

  /* optimized shuffle network for SIMD aligned sizes */
#if USE_OPTIMIZED_AVX512_TRANSPOSE == 1
  if ( (i_mateltwise_desc->m % 2 == 0) && (i_mateltwise_desc->n % 2 == 0) ) {
    if ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) && (io_generated_code->arch < LIBXSMM_X86_ALLFEAT) ) {
      /* codepath optimized for SPR */
      libxsmm_generator_transform_vnni2_to_vnni2t_16bit_avx512_spr_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_in, i_gp_reg_out,
                                                                              i_gp_reg_m_loop, i_gp_reg_n_loop, i_gp_reg_mask, i_gp_reg_mask_2,
                                                                              i_mask_reg_0, i_mask_reg_1, i_mask_reg_2, i_mask_reg_3, i_mask_reg_4,
                                                                              i_mask_reg_5, i_mask_reg_6, i_micro_kernel_config, i_mateltwise_desc );
    } else {
      /* codepath optimized for CLX */
      libxsmm_generator_transform_vnni2_to_vnni2t_16bit_avx512_pre_spr_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_in, i_gp_reg_out,
                                                                                  i_gp_reg_m_loop, i_gp_reg_n_loop, i_gp_reg_mask,
                                                                                  i_mask_reg_0, i_mask_reg_1, i_mask_reg_2, i_mask_reg_3, i_mask_reg_4,
                                                                                  i_mask_reg_5, i_micro_kernel_config, i_mateltwise_desc );
    }
  }
#elif COPY_UPPER_BOUND == 1
  if ( (i_mateltwise_desc->m % 2 == 0) && (i_mateltwise_desc->n % 2 == 0) ) {
    libxsmm_generator_transform_copy_avx512_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_in, i_gp_reg_out,
                                                         i_gp_reg_m_loop, i_gp_reg_n_loop, i_micro_kernel_config, i_mateltwise_desc->ldi*2,
                                                         i_mateltwise_desc->ldo*2, i_mateltwise_desc->m*2, i_mateltwise_desc->n/2, 8 );
  }
#else
  if ( (i_mateltwise_desc->m % 2 == 0) && (i_mateltwise_desc->n % 2 == 0) ) {

    unsigned int l_ldi = i_mateltwise_desc->ldi*2;
    unsigned int l_ldo = i_mateltwise_desc->ldo*2;

    /* input mask, scalar */
    unsigned long long l_mask = 0x1;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     i_gp_reg_mask, l_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
                                       i_gp_reg_mask, i_mask_reg_0 );

    /* output mask, scalar */
    l_mask = 0x1;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     i_gp_reg_mask, l_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
                                       i_gp_reg_mask, i_mask_reg_1 );

    /* n loop header */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_n_loop, 2 );

    /* m loop header */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_m_loop, 2 );

    /* actual transpose */
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                      'x', 0, i_mask_reg_0, 1, 0 );

    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in * 2,
                                      'x', 1, i_mask_reg_0, 1, 0 );

    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in,
                                      'x', 2, i_mask_reg_0, 1, 0 );

    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in * 3,
                                      'x', 3, i_mask_reg_0, 1, 0 );

    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                      i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                      'x', 0, i_mask_reg_1, 0, 1 );

    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                      i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out,
                                      'x', 1, i_mask_reg_1, 0, 1 );

    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                      i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out * 2,
                                      'x', 2, i_mask_reg_1, 0, 1 );

    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                      i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out * 3,
                                      'x', 3, i_mask_reg_1, 0, 1 );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, (long long)i_micro_kernel_config->datatype_size_in * 4 );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (long long)i_micro_kernel_config->datatype_size_out * l_ldo );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, i_mateltwise_desc->m );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, ((long long)l_ldi * i_micro_kernel_config->datatype_size_in) - ((long long)i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->m * 2) );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                     i_gp_reg_out, ((long long)i_micro_kernel_config->datatype_size_out * l_ldo * i_mateltwise_desc->m/2) - ((long long)i_micro_kernel_config->datatype_size_out * 4)  );

    /* close n loop */
    libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_n_loop, i_mateltwise_desc->n );
  }
#endif
  else if ( (i_mateltwise_desc->m == 1) && (i_mateltwise_desc->n % 2 == 0) ) {
    /* input mask, scalar */
    unsigned long long l_mask = 0x1;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     i_gp_reg_mask, l_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
                                       i_gp_reg_mask, i_mask_reg_0 );

    /* output mask, scalar */
    l_mask = 0x1;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     i_gp_reg_mask, l_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
                                       i_gp_reg_mask, i_mask_reg_1 );

    /* n loop header */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_n_loop, 1 );

    /* actual transpose */
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                      'x', 0, i_mask_reg_0, 1, 0 );

    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                      i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                      'x', 0, i_mask_reg_1, 0, 1 );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, i_micro_kernel_config->datatype_size_in );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, i_micro_kernel_config->datatype_size_out );

    /* close n loop */
    libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_n_loop, i_mateltwise_desc->n );
  } else {
    if ( i_mateltwise_desc->m % 2 != 0 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
      return;
    }
    if ( i_mateltwise_desc->n % 2 != 0 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
      return;
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni4_08bit_avx512_mnblock_micro_kernel( libxsmm_generated_code*                 io_generated_code,
                                                                                  const unsigned int                      i_gp_reg_in,
                                                                                  const unsigned int                      i_gp_reg_out,
                                                                                  const unsigned int                      i_mask_reg_0,
                                                                                  const unsigned int                      i_mask_reg_1,
                                                                                  const unsigned int                      i_perm_1st_stage_reg,
                                                                                  const unsigned int                      i_m_step,
                                                                                  const unsigned int                      i_n_step,
                                                                                  const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                  const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int l_zmm = 0;
  unsigned int l_n_full_block = ((i_n_step + 3) / 4) * 4;
  unsigned int l_zmm_tmp = l_n_full_block;
  unsigned int l_m_full = (i_m_step * 4) / 64;
  unsigned int l_m_rem = (i_m_step * 4) % 64;
  unsigned int l_i = 0;

  /* check for max unrolling */
  if ( l_zmm_tmp > 27 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* load i_n_step registers */
  for ( l_zmm = 0; l_zmm < i_n_step; l_zmm++ ) {
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, l_zmm * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                      i_micro_kernel_config->vector_name, l_zmm, i_mask_reg_0, 1, 0 );
  }

  /* complete last block with zero registers for a total number of registers divisible by 4 */
  for ( ; l_zmm < l_n_full_block; l_zmm++ ) {
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, i_micro_kernel_config->vector_name,
                                              l_zmm, l_zmm, l_zmm );
  }

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_in, (long long)i_m_step * i_micro_kernel_config->datatype_size_in );

  /* create VNNI interleaved format */
  for ( l_zmm = 0; l_zmm < i_n_step; l_zmm+=4 ) {

    /* 1st stage: variable permute network */
    libxsmm_generator_transform_Xway_permutevar1_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                 i_perm_1st_stage_reg, l_zmm, LIBXSMM_X86_INSTR_VPERMD, 4 );

    /* 2nd stage: shuffle network */
    {
      unsigned char l_in_idx[4] = { 0x0, 0x0, 0x2, 0x2};
      libxsmm_generator_transform_Xway_unpack_network_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                  l_in_idx, l_zmm, l_zmm_tmp, 1,
                                                                  LIBXSMM_X86_INSTR_VPUNPCKLBW, LIBXSMM_X86_INSTR_VPUNPCKHBW, 4 );
    }

    /* 3rd stage: second shuffle network */
    {
      unsigned char l_in_idx[4] = { 0x0, 0x0, 0x1, 0x1};
      libxsmm_generator_transform_Xway_unpack_network_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                  l_in_idx, l_zmm_tmp, l_zmm, 2,
                                                                  LIBXSMM_X86_INSTR_VPUNPCKLWD, LIBXSMM_X86_INSTR_VPUNPCKHWD, 4 );
    }

    /* store VNNI format */
    for ( l_i = 0; l_i < l_m_full; ++l_i) {
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                        i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, (l_zmm * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) + (l_i * 64 * i_micro_kernel_config->datatype_size_out),
                                        i_micro_kernel_config->vector_name, l_zmm + l_i, 0, 1, 1 );
    }
    if ( l_m_rem > 0 ) {
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                        i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, (l_zmm * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) + (l_i * 64 * i_micro_kernel_config->datatype_size_out),
                                        i_micro_kernel_config->vector_name, l_zmm + l_i, i_mask_reg_1, 0, 1 );
    }
  }

  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, (long long)i_m_step * 4 * i_micro_kernel_config->datatype_size_out );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni4_16bit_avx512_mnblock_micro_kernel( libxsmm_generated_code*                 io_generated_code,
                                                                                 const unsigned int                      i_gp_reg_in,
                                                                                 const unsigned int                      i_gp_reg_out,
                                                                                 const unsigned int                      i_mask_reg_0,
                                                                                 const unsigned int                      i_mask_reg_1,
                                                                                 const unsigned int                      i_vnni_lo_reg,
                                                                                 const unsigned int                      i_vnni_hi_reg,
                                                                                 const unsigned int                      i_vnni_lo_reg_2,
                                                                                 const unsigned int                      i_vnni_hi_reg_2,
                                                                                 const unsigned int                      i_m_step,
                                                                                 const unsigned int                      i_n_step,
                                                                                 const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int l_zmm = 0;
  unsigned int l_zmm_tmp0 = ((i_n_step + 3) / 4) * 4;
  unsigned int l_zmm_tmp1 = ((i_n_step + 3) / 4) * 4 + 1;
  unsigned int l_m_bound = 16;

  if ((io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX)) {
    l_m_bound = 8;
  }

  /* check for max unrolling */
  if ( l_zmm_tmp0 > 27 || l_zmm_tmp1 > 27 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* load 4 registers */
  for ( l_zmm = 0; l_zmm < i_n_step; l_zmm+=4 ) {
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, l_zmm * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                      i_micro_kernel_config->vector_name, l_zmm + 0, i_mask_reg_0, 1, 0 );
    if ( (i_n_step % 4 == 1) && (l_zmm + 3 >= i_n_step) ) {
      libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, i_micro_kernel_config->vector_name,
                                                l_zmm + 1, l_zmm + 1, l_zmm + 1 );
    } else {
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                        i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, (l_zmm + 1) * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                        i_micro_kernel_config->vector_name, l_zmm + 1, i_mask_reg_0, 1, 0 );
    }
    if ( ((i_n_step % 4 == 1) || (i_n_step % 4 == 2) ) && (l_zmm + 3 >= i_n_step) ) {
      libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, i_micro_kernel_config->vector_name,
                                                l_zmm + 2, l_zmm + 2, l_zmm + 2 );
    } else {
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                        i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, (l_zmm + 2) * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                        i_micro_kernel_config->vector_name, l_zmm + 2, i_mask_reg_0, 1, 0 );
    }
    if ( (i_n_step % 4 >= 1) && (l_zmm + 3 >= i_n_step) ) {
      libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, i_micro_kernel_config->vector_name,
                                                l_zmm + 3, l_zmm + 3, l_zmm + 3 );
    } else {
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                        i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, (l_zmm + 3) * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                        i_micro_kernel_config->vector_name, l_zmm + 3, i_mask_reg_0, 1, 0 );
    }
  }

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_in, (long long)i_m_step * i_micro_kernel_config->datatype_size_in );

  /* create VNNI interleaved format */
  for ( l_zmm = 0; l_zmm < i_n_step; l_zmm+=4 ) {
    unsigned int l_zmm_tmp2 = l_zmm;
    unsigned int l_zmm_tmp3 = l_zmm + 2;
    if ( i_m_step > l_m_bound ) {
      libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64_LD, i_micro_kernel_config->vector_name,
                                                              l_zmm + 1, LIBXSMM_X86_VEC_REG_UNDEF, l_zmm_tmp0, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
    }
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2W, i_micro_kernel_config->vector_name,
                                                            l_zmm, i_vnni_lo_reg, l_zmm + 1, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
    if ( i_m_step > l_m_bound ) {
      libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2W, i_micro_kernel_config->vector_name,
                                                              l_zmm, i_vnni_hi_reg, l_zmm_tmp0, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
    }

    if ( i_m_step > l_m_bound ) {
      libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64_LD, i_micro_kernel_config->vector_name,
                                                              l_zmm + 3, LIBXSMM_X86_VEC_REG_UNDEF, l_zmm_tmp1, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
    }
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2W, i_micro_kernel_config->vector_name,
                                                            l_zmm + 2, i_vnni_lo_reg, l_zmm + 3, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
    if ( i_m_step > l_m_bound ) {
      libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2W, i_micro_kernel_config->vector_name,
                                                              l_zmm + 2, i_vnni_hi_reg, l_zmm_tmp1, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
    }

    /* Now zip the vnni2 register (l_zmm + 1 with l_zmm + 3 and (if i_m_step > l_m_bound) tmp0 with tmp1 */
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64_LD, i_micro_kernel_config->vector_name,
                                                            l_zmm + 3, LIBXSMM_X86_VEC_REG_UNDEF, l_zmm_tmp2, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );

    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2D, i_micro_kernel_config->vector_name,
                                                            l_zmm + 1, i_vnni_lo_reg_2, l_zmm + 3, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );

    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2D, i_micro_kernel_config->vector_name,
                                                            l_zmm + 1, i_vnni_hi_reg_2, l_zmm_tmp2, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );

    /* Store 2 registers */
    if ( i_m_step > l_m_bound/2 ) {
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                        i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, l_zmm * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                        i_micro_kernel_config->vector_name, l_zmm + 3, 0, 1, 1 );
      if ( i_m_step > l_m_bound ) {
        libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                          i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, (l_zmm * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) + (l_m_bound * 2 * i_micro_kernel_config->datatype_size_out),
                                          i_micro_kernel_config->vector_name, l_zmm_tmp2, 0, 1, 1 );
      } else {
        libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                          i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, (l_zmm * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) + (l_m_bound * 2 * i_micro_kernel_config->datatype_size_out),
                                          i_micro_kernel_config->vector_name, l_zmm_tmp2, i_mask_reg_1, 0, 1 );
      }
    } else {
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                        i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, l_zmm * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                        i_micro_kernel_config->vector_name, l_zmm + 3, i_mask_reg_1, 0, 1 );
    }


    if ( i_m_step > l_m_bound ) {
      libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64_LD, i_micro_kernel_config->vector_name,
                                                              l_zmm_tmp1, LIBXSMM_X86_VEC_REG_UNDEF, l_zmm_tmp3, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );

      libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2D, i_micro_kernel_config->vector_name,
                                                              l_zmm_tmp0, i_vnni_lo_reg_2, l_zmm_tmp1, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );

      libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2D, i_micro_kernel_config->vector_name,
                                                              l_zmm_tmp0, i_vnni_hi_reg_2, l_zmm_tmp3, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );

      /* Store 2 registers */
      if ( i_m_step > l_m_bound + l_m_bound/2 ) {
        libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                          i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, l_zmm * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out + (l_m_bound * 4 * i_micro_kernel_config->datatype_size_out),
                                          i_micro_kernel_config->vector_name, l_zmm_tmp1, 0, 1, 1 );
        libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                          i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, (l_zmm * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) + (l_m_bound * 6 * i_micro_kernel_config->datatype_size_out),
                                          i_micro_kernel_config->vector_name, l_zmm_tmp3, i_mask_reg_1, 0, 1 );
      } else {
        libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                          i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, l_zmm * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out + (l_m_bound * 4 * i_micro_kernel_config->datatype_size_out) ,
                                          i_micro_kernel_config->vector_name, l_zmm_tmp1, i_mask_reg_1, 0, 1 );
      }
    }
  }

  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, (long long)i_m_step * 4 * i_micro_kernel_config->datatype_size_out );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni2_16bit_avx512_mnblock_micro_kernel( libxsmm_generated_code*                 io_generated_code,
                                                                                 const unsigned int                      i_gp_reg_in,
                                                                                 const unsigned int                      i_gp_reg_out,
                                                                                 const unsigned int                      i_mask_reg_0,
                                                                                 const unsigned int                      i_mask_reg_1,
                                                                                 const unsigned int                      i_vnni_lo_reg,
                                                                                 const unsigned int                      i_vnni_hi_reg,
                                                                                 const unsigned int                      i_m_step,
                                                                                 const unsigned int                      i_n_step,
                                                                                 const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int l_zmm = 0;
  unsigned int l_zmm_tmp = ((i_n_step + 1) / 2) * 2;

  /* check for max unrolling */
  if ( l_zmm_tmp > 29 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* load 2 registers */
  for ( l_zmm = 0; l_zmm < i_n_step; l_zmm+=2 ) {
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, l_zmm * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                      i_micro_kernel_config->vector_name, l_zmm + 0, i_mask_reg_0, 1, 0 );
    if ( (i_n_step % 2 == 1) && (l_zmm + 1 == i_n_step) ) {
      libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, i_micro_kernel_config->vector_name,
                                                l_zmm + 1, l_zmm + 1, l_zmm + 1 );
    } else {
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                        i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, (l_zmm + 1) * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                        i_micro_kernel_config->vector_name, l_zmm + 1, i_mask_reg_0, 1, 0 );
    }
  }

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_in, (long long)i_m_step * i_micro_kernel_config->datatype_size_in );

  /* create VNNI interleaved format */
  for ( l_zmm = 0; l_zmm < i_n_step; l_zmm+=2 ) {
    if ( i_m_step > 16 ) {
      libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64_LD, i_micro_kernel_config->vector_name,
                                                              l_zmm + 1, LIBXSMM_X86_VEC_REG_UNDEF, l_zmm_tmp, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
    }
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2W, i_micro_kernel_config->vector_name,
                                                            l_zmm, i_vnni_lo_reg, l_zmm + 1, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
    if ( i_m_step > 16 ) {
      libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2W, i_micro_kernel_config->vector_name,
                                                              l_zmm, i_vnni_hi_reg, l_zmm_tmp, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
    }

    /* storing 2 registers */
    if ( i_m_step > 16 ) {
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                        i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, l_zmm * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                        i_micro_kernel_config->vector_name, l_zmm + 1, 0, 1, 1 );
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                        i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, (l_zmm * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) + (32 * i_micro_kernel_config->datatype_size_out),
                                        i_micro_kernel_config->vector_name, l_zmm_tmp, i_mask_reg_1, 0, 1 );
    } else {
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                        i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, l_zmm * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                        i_micro_kernel_config->vector_name, l_zmm + 1, i_mask_reg_1, 0, 1 );
    }
  }

  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, (long long)i_m_step * 2 * i_micro_kernel_config->datatype_size_out );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni4_16bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                        libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                        const unsigned int                      i_gp_reg_in,
                                                                        const unsigned int                      i_gp_reg_out,
                                                                        const unsigned int                      i_gp_reg_m_loop,
                                                                        const unsigned int                      i_gp_reg_n_loop,
                                                                        const unsigned int                      i_gp_reg_mask,
                                                                        const unsigned int                      i_mask_reg_0,
                                                                        const unsigned int                      i_mask_reg_1,
                                                                        const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                        const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                        const unsigned int                      i_pad_vnni ) {
  unsigned int l_vnni_lo_reg = 31;
  unsigned int l_vnni_hi_reg = 30;
  unsigned int l_vnni_lo_reg_2 = 29;
  unsigned int l_vnni_hi_reg_2 = 28;
  unsigned int l_m_entries = ((io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX)) ? 16 : 32;
  unsigned int l_m_remainder = i_mateltwise_desc->m % l_m_entries;
  unsigned int l_m_full = i_mateltwise_desc->m / l_m_entries;
  unsigned int l_n_remainder = i_mateltwise_desc->n % 16;
  unsigned int l_n_full = i_mateltwise_desc->n / 16;
  short perm_table_vnni_lo[32] = {32, 0, 33, 1, 34, 2, 35, 3, 36, 4, 37, 5, 38, 6, 39, 7, 40, 8, 41, 9, 42, 10, 43, 11, 44, 12, 45, 13, 46, 14, 47, 15};
  short perm_table_vnni_hi[32] = {48, 16, 49, 17, 50, 18, 51, 19, 52, 20, 53, 21, 54, 22, 55, 23, 56, 24, 57, 25, 58, 26, 59, 27, 60, 28, 61, 29, 62, 30, 63, 31};
  int perm_table_vnni_lo_2[16] = {16, 0, 17, 1, 18, 2, 19, 3, 20, 4, 21, 5, 22, 6, 23, 7 };
  int perm_table_vnni_hi_2[16] = {24, 8, 25, 9, 26, 10, 27, 11, 28, 12, 29, 13, 30, 14, 31, 15};

  short _perm_table_vnni_lo[16] = {16, 0, 17, 1, 18, 2, 19, 3, 20, 4, 21, 5, 22, 6, 23, 7};
  short _perm_table_vnni_hi[16] = {24, 8, 25, 9, 26, 10, 27, 11, 28, 12, 29, 13, 30, 14, 31, 15};
  int _perm_table_vnni_lo_2[8] = {8, 0, 9, 1, 10, 2, 11, 3};
  int _perm_table_vnni_hi_2[8] = {12, 4, 13, 5, 14, 6, 15, 7};

  if (l_m_entries == 32) {
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) perm_table_vnni_hi, "perm_table_vnni_hi_", i_micro_kernel_config->vector_name, l_vnni_hi_reg);
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) perm_table_vnni_lo, "perm_table_vnni_lo_", i_micro_kernel_config->vector_name, l_vnni_lo_reg);
  } else {
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) _perm_table_vnni_hi, "perm_table_vnni_hi_", i_micro_kernel_config->vector_name, l_vnni_hi_reg);
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) _perm_table_vnni_lo, "perm_table_vnni_lo_", i_micro_kernel_config->vector_name, l_vnni_lo_reg);
  }
  if (l_m_entries == 32) {
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) perm_table_vnni_lo_2, "perm_table_vnni_lo_2_", i_micro_kernel_config->vector_name, l_vnni_lo_reg_2);
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) perm_table_vnni_hi_2, "perm_table_vnni_hi_2_", i_micro_kernel_config->vector_name, l_vnni_hi_reg_2);
  } else {
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) _perm_table_vnni_lo_2, "perm_table_vnni_lo_2_", i_micro_kernel_config->vector_name, l_vnni_lo_reg_2);
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) _perm_table_vnni_hi_2, "perm_table_vnni_hi_2_", i_micro_kernel_config->vector_name, l_vnni_hi_reg_2);
  }

  /* check if the right combination of knobs is provided */
  if ( (i_pad_vnni == 0) && (i_mateltwise_desc->n % 4 > 0) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* set masks */
  if ( l_m_remainder > 0 ) {
    /* load mask */
    unsigned int l_mask_instr = (l_m_entries == 32) ? LIBXSMM_X86_INSTR_KMOVD_GPR_LD : LIBXSMM_X86_INSTR_KMOVW_GPR_LD ;
    const unsigned long long l_load_mask = ( (unsigned long long)1 << l_m_remainder ) - 1;
    const unsigned long long l_store_mask = (l_m_remainder % (l_m_entries/4) == 0) ? (unsigned long long)0xffffffff : ( (unsigned long long)1 << ((l_m_remainder % (l_m_entries/4)) * 4) ) - 1;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask, l_load_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, l_mask_instr, i_gp_reg_mask, i_mask_reg_0 );

    /* create store masking */
    l_mask_instr = (l_m_entries == 32) ? LIBXSMM_X86_INSTR_KMOVD_GPR_LD : LIBXSMM_X86_INSTR_KMOVW_GPR_LD ;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask, l_store_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, l_mask_instr, i_gp_reg_mask, i_mask_reg_1 );
  }

  if ( l_n_full > 0 ) {
    /* open n loop */
    if ( l_n_full > 1 ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_n_loop, 16 );
    }

    /* full m iterations in a loop */
    if ( l_m_full > 0 ) {
      /* open m loop */
      if ( l_m_full > 1 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
        libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                         i_gp_reg_m_loop, l_m_entries );
      }

      libxsmm_generator_transform_norm_to_vnni4_16bit_avx512_mnblock_micro_kernel( io_generated_code, i_gp_reg_in, i_gp_reg_out, 0, 0, l_vnni_lo_reg, l_vnni_hi_reg, l_vnni_lo_reg_2, l_vnni_hi_reg_2,
          l_m_entries, 16, i_micro_kernel_config, i_mateltwise_desc );

      /* close m footer */
      if ( l_m_full > 1 ) {
        libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                    i_gp_reg_m_loop, l_m_full*l_m_entries );
      }
    }
    /* m remainder masked */
    if ( l_m_remainder > 0 ) {
      libxsmm_generator_transform_norm_to_vnni4_16bit_avx512_mnblock_micro_kernel( io_generated_code, i_gp_reg_in, i_gp_reg_out, i_mask_reg_0, i_mask_reg_1,
                                                                                  l_vnni_lo_reg, l_vnni_hi_reg, l_vnni_lo_reg_2, l_vnni_hi_reg_2, l_m_remainder, 16, i_micro_kernel_config, i_mateltwise_desc );
    }

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, (16LL * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in) - ((long long)i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->m) );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (16LL * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->m * 4) );

    /* close n loop */
    if ( l_n_full > 1 ) {
      libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                  i_gp_reg_n_loop, l_n_full*16  );
    }
  }

  if ( l_n_remainder > 0 ) {
    /* full m iterations in a loop */
    if ( l_m_full > 0 ) {
      /* open m loop */
      if ( l_m_full > 1 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
        libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                         i_gp_reg_m_loop, l_m_entries );
      }

      libxsmm_generator_transform_norm_to_vnni4_16bit_avx512_mnblock_micro_kernel( io_generated_code, i_gp_reg_in, i_gp_reg_out, 0, 0, l_vnni_lo_reg, l_vnni_hi_reg,  l_vnni_lo_reg_2, l_vnni_hi_reg_2,
          l_m_entries, l_n_remainder, i_micro_kernel_config, i_mateltwise_desc );

      /* close m footer */
      if ( l_m_full > 1 ) {
        libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                    i_gp_reg_m_loop, l_m_full*l_m_entries );
      }
    }
    /* m remainder masked */
    if ( l_m_remainder > 0 ) {
      libxsmm_generator_transform_norm_to_vnni4_16bit_avx512_mnblock_micro_kernel( io_generated_code, i_gp_reg_in, i_gp_reg_out, i_mask_reg_0, i_mask_reg_1,
                                                                                  l_vnni_lo_reg, l_vnni_hi_reg, l_vnni_lo_reg_2, l_vnni_hi_reg_2, l_m_remainder, l_n_remainder, i_micro_kernel_config, i_mateltwise_desc );
    }
  }
}

/*
 * Calling convention, this kernel assumes that all GPRs with are not in the argument list
 * and all xmm/ymm/zmm/tmm are caller save
 *
 * TODO; stack local variables....
 * */
LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni2_16bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                        libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                        const unsigned int                      i_gp_reg_in,
                                                                        const unsigned int                      i_gp_reg_out,
                                                                        const unsigned int                      i_gp_reg_m_loop,
                                                                        const unsigned int                      i_gp_reg_n_loop,
                                                                        const unsigned int                      i_gp_reg_mask,
                                                                        const unsigned int                      i_mask_reg_0,
                                                                        const unsigned int                      i_mask_reg_1,
                                                                        const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                        const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                        const unsigned int                      i_pad_vnni ) {
  unsigned int l_vnni_lo_reg = 31;
  unsigned int l_vnni_hi_reg = 30;
  unsigned int l_m_remainder = i_mateltwise_desc->m % 32;
  unsigned int l_m_full = i_mateltwise_desc->m / 32;
  unsigned int l_n_remainder = i_mateltwise_desc->n % 16;
  unsigned int l_n_full = i_mateltwise_desc->n / 16;
  short perm_table_vnni_lo[32] = {32, 0, 33, 1, 34, 2, 35, 3, 36, 4, 37, 5, 38, 6, 39, 7, 40, 8, 41, 9, 42, 10, 43, 11, 44, 12, 45, 13, 46, 14, 47, 15};
  short perm_table_vnni_hi[32] = {48, 16, 49, 17, 50, 18, 51, 19, 52, 20, 53, 21, 54, 22, 55, 23, 56, 24, 57, 25, 58, 26, 59, 27, 60, 28, 61, 29, 62, 30, 63, 31};
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) perm_table_vnni_lo, "perm_table_vnni_lo_", i_micro_kernel_config->vector_name, l_vnni_lo_reg);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) perm_table_vnni_hi, "perm_table_vnni_hi_", i_micro_kernel_config->vector_name, l_vnni_hi_reg);

  /* check if the right combination of knobs is provided */
  if ( (i_pad_vnni == 0) && (i_mateltwise_desc->n % 2 == 1) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* set masks */
  if ( l_m_remainder > 0 ) {
    /* load mask */
    const unsigned long long l_load_mask = ( (unsigned long long)1 << l_m_remainder ) - 1;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask, l_load_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mask, i_mask_reg_0 );

    /* create store masking */
    if ( l_m_remainder > 16 ) {
      /* load mask */
      const unsigned long long l_store_mask = ( (unsigned long long)1 << ((l_m_remainder - 16) * 2) ) - 1;
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask, l_store_mask );
      libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mask, i_mask_reg_1 );
    } else {
      /* load mask */
      const unsigned long long l_store_mask = ( l_m_remainder == 16 ) ? (unsigned long long)0xffffffff : (unsigned long long)(( (unsigned long long)1 << (l_m_remainder * 2) ) - 1);
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask, l_store_mask );
      libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mask, i_mask_reg_1 );
    }
  }

  if ( l_n_full > 0 ) {
    /* open n loop */
    if ( l_n_full > 1 ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_n_loop, 16 );
    }

    /* full m iterations in a loop */
    if ( l_m_full > 0 ) {
      /* open m loop */
      if ( l_m_full > 1 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
        libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                         i_gp_reg_m_loop, 32 );
      }

      libxsmm_generator_transform_norm_to_vnni2_16bit_avx512_mnblock_micro_kernel( io_generated_code, i_gp_reg_in, i_gp_reg_out, 0, 0, l_vnni_lo_reg, l_vnni_hi_reg,
                                                                                  32, 16, i_micro_kernel_config, i_mateltwise_desc );

      /* close m footer */
      if ( l_m_full > 1 ) {
        libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                    i_gp_reg_m_loop, l_m_full*32 );
      }
    }
    /* m remainder masked */
    if ( l_m_remainder > 0 ) {
      libxsmm_generator_transform_norm_to_vnni2_16bit_avx512_mnblock_micro_kernel( io_generated_code, i_gp_reg_in, i_gp_reg_out, i_mask_reg_0, i_mask_reg_1,
                                                                                  l_vnni_lo_reg, l_vnni_hi_reg, l_m_remainder, 16, i_micro_kernel_config, i_mateltwise_desc );
    }

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, (16LL * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in) - ((long long)i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->m) );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (16LL * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->m * 2) );

    /* close n loop */
    if ( l_n_full > 1 ) {
      libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                  i_gp_reg_n_loop, l_n_full*16  );
    }
  }

  if ( l_n_remainder > 0 ) {
    /* full m iterations in a loop */
    if ( l_m_full > 0 ) {
      /* open m loop */
      if ( l_m_full > 1 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
        libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                         i_gp_reg_m_loop, 32 );
      }

      libxsmm_generator_transform_norm_to_vnni2_16bit_avx512_mnblock_micro_kernel( io_generated_code, i_gp_reg_in, i_gp_reg_out, 0, 0, l_vnni_lo_reg, l_vnni_hi_reg,
                                                                                  32, l_n_remainder, i_micro_kernel_config, i_mateltwise_desc );

      /* close m footer */
      if ( l_m_full > 1 ) {
        libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                    i_gp_reg_m_loop, l_m_full*32 );
      }
    }
    /* m remainder masked */
    if ( l_m_remainder > 0 ) {
      libxsmm_generator_transform_norm_to_vnni2_16bit_avx512_mnblock_micro_kernel( io_generated_code, i_gp_reg_in, i_gp_reg_out, i_mask_reg_0, i_mask_reg_1,
                                                                                  l_vnni_lo_reg, l_vnni_hi_reg, l_m_remainder, l_n_remainder, i_micro_kernel_config, i_mateltwise_desc );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni4t_08bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                           libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                           const unsigned int                      i_gp_reg_in,
                                                                           const unsigned int                      i_gp_reg_out,
                                                                           const unsigned int                      i_gp_reg_m_loop,
                                                                           const unsigned int                      i_gp_reg_n_loop,
                                                                           const unsigned int                      i_gp_reg_mask,
                                                                           const unsigned int                      i_mask_reg_0,
                                                                           const unsigned int                      i_mask_reg_1,
                                                                           const unsigned int                      i_mask_reg_2,
                                                                           const unsigned int                      i_mask_reg_3,
                                                                           const unsigned int                      i_mask_reg_4,
                                                                           const unsigned int                      i_mask_reg_5,
                                                                           const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                           const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {

  if ( (i_mateltwise_desc->m % 4 == 0) && (i_mateltwise_desc->n % 4 == 0) ) {
    unsigned int l_ldi = i_mateltwise_desc->ldi*4;
    unsigned int l_ldo = i_mateltwise_desc->ldo*4;

    /* byte shuffle operand */
    unsigned int  l_shuffle_cntl[16] = { 0x0c080400, 0x0d090501, 0x0e0a0602, 0x0f0b0703, 0x1c181410, 0x1d191511, 0x1e1a1612, 0x1f1b1713,
                                         0x2c282420, 0x2d292521, 0x2e2a2622, 0x2f2b2723, 0x3c383430, 0x3d393531, 0x3e3a3632, 0x3f3b3733 };
    unsigned int  l_shuffle_op = 31;

    /* Partial 4x32 blocks */
    const unsigned int l_n_4rem = (i_mateltwise_desc->n/4) % 4;
    const unsigned int l_m_32rem = (i_mateltwise_desc->m*4) % 32;

    const unsigned int l_n_4mul = (i_mateltwise_desc->n/4) - l_n_4rem;
    const unsigned int l_m_32mul = (i_mateltwise_desc->m*4) - l_m_32rem;

    /* set the masks for the load+blend stage */
    unsigned long long l_mask = 0xff00;

    /* set the masks for the load+blend stage for partial 2x32 blocks */
    if ( l_m_32rem > 0 ) {
      /* set mask with l_m_32rem_mask = (1 << (l_m_32rem >> 1)) - 1 */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask, 1 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SHLQ, i_gp_reg_mask, (long long)l_m_32rem/2 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, i_gp_reg_mask, 1 );

      libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVQ_GPR_LD, i_gp_reg_mask, i_mask_reg_3 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SHLQ, i_gp_reg_mask, 8 );
      libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVQ_GPR_LD, i_gp_reg_mask, i_mask_reg_4 );

    }

    if ( l_n_4rem > 0 ) {
      /* set mask with l_n_4rem_mask = (1 << (l_n_4rem * 16)) - 1 */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask, 1 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SHLQ, i_gp_reg_mask, (long long)l_n_4rem * 16 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, i_gp_reg_mask, 1 );
      libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVQ_GPR_LD,
                                         i_gp_reg_mask, i_mask_reg_5 );
    }

    libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char *) l_shuffle_cntl,
                                                        "vnni4_to_vnni4t_shufl_", i_micro_kernel_config->vector_name, l_shuffle_op);

    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     i_gp_reg_mask, l_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVQ_GPR_LD,
                                       i_gp_reg_mask, i_mask_reg_0 );

    /* set the masks for the permute stage */
    /* even quarters mask */
    l_mask = 0xcc;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     i_gp_reg_mask, l_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVQ_GPR_LD,
                                       i_gp_reg_mask, i_mask_reg_1 );

    /* odd quarter masks */
    l_mask = 0x33;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     i_gp_reg_mask, l_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVQ_GPR_LD,
                                       i_gp_reg_mask, i_mask_reg_2 );

    /* Transpose x32 blocks */
    if ( l_m_32mul > 0 ) {
      /* open m loop */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_m_loop, 32 );

      /* transpose 4x32 blocks */
      if ( l_n_4mul > 0 ) {

        /* open n loop */
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
        libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                         i_gp_reg_n_loop, 4 );

        /* load 4 registers with two half rows */
        {
          const unsigned int ld_idx[32] = { 0, 2 };
          unsigned int l_mask_regs[2] = { 0 };
          l_mask_regs[0] = 0;    l_mask_regs[1] = i_mask_reg_0;
          libxsmm_generator_transform_Xway_half_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                   i_gp_reg_in, 0, l_ldi * i_micro_kernel_config->datatype_size_in,
                                                                   ld_idx, 2, LIBXSMM_X86_INSTR_VBROADCASTI32X8, 2, l_mask_regs, 4 );
        }

        /* advance input pointer */
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                         i_gp_reg_in, (long long)l_ldi * i_micro_kernel_config->datatype_size_in * 4 );

        /* transpose two 2x32 blocks */
        libxsmm_generator_transform_two_8x8_08bit_vnni4_to_vnni4t_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, l_shuffle_op, i_mask_reg_1, i_mask_reg_2);

        /* storing 2 registers */
        libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                i_gp_reg_out, 0, l_ldo * i_micro_kernel_config->datatype_size_out,
                                                                i_micro_kernel_config->vmove_instruction_out, 0, 0, 2 );

        /* advance output pointer */
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                         i_gp_reg_out, (long long)i_micro_kernel_config->datatype_size_out * 64 );

        /* close n footer */
        libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                    i_gp_reg_n_loop, l_n_4mul );
      }

      /* transpose n_4_rem x 32 block */
      if ( l_n_4rem > 0 ) {
        /* load 4 registers with two half rows */
        {
          const unsigned int ld_idx[32] = { 0, 2 };
          unsigned int l_mask_regs[2] = { 0 };
          l_mask_regs[0] = 0;    l_mask_regs[1] = i_mask_reg_0;
          libxsmm_generator_transform_Xway_half_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                   i_gp_reg_in, 0, l_ldi * i_micro_kernel_config->datatype_size_in,
                                                                   ld_idx, 2, LIBXSMM_X86_INSTR_VBROADCASTI32X8, 2, l_mask_regs, l_n_4rem );
        }

        /* transpose two 2x32 blocks */
        libxsmm_generator_transform_two_8x8_08bit_vnni4_to_vnni4t_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, l_shuffle_op, i_mask_reg_1, i_mask_reg_2);

        /* storing 2 registers */
        libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                i_gp_reg_out, 0, l_ldo * i_micro_kernel_config->datatype_size_out,
                                                                i_micro_kernel_config->vmove_instruction_out, 1, i_mask_reg_5, 2 );
      }

      /* advance output pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_out, (2LL * l_ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_out * l_n_4mul * 16) );

      /* advance input pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                       i_gp_reg_in, ((long long)l_ldi * i_micro_kernel_config->datatype_size_in * l_n_4mul) - (32LL * i_micro_kernel_config->datatype_size_in) );

      /* close m loop */
      libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                  i_gp_reg_m_loop, l_m_32mul );
    }

    /* Transpose m_32rem blocks */
    if ( l_m_32rem > 0 ) {

      /* transpose 4 x m_32rem blocks */
      if ( l_n_4mul > 0 ) {
        /* open n loop */
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
        libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                         i_gp_reg_n_loop, 4);

        /* load 4 registers with two half rows */
        {
          const unsigned int ld_idx[32] = { 0, 2 };
          unsigned int l_mask_regs[2] = { 0 };
          l_mask_regs[0] = i_mask_reg_3;    l_mask_regs[1] = i_mask_reg_4;
          libxsmm_generator_transform_Xway_half_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                   i_gp_reg_in, 0, l_ldi * i_micro_kernel_config->datatype_size_in,
                                                                   ld_idx, 2, LIBXSMM_X86_INSTR_VBROADCASTI32X8, 2, l_mask_regs, 4 );
        }

        /* advance input pointer */
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                         i_gp_reg_in, (long long)l_ldi * i_micro_kernel_config->datatype_size_in * 4 );

        /* transpose two 2x32 blocks */
        libxsmm_generator_transform_two_8x8_08bit_vnni4_to_vnni4t_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, l_shuffle_op, i_mask_reg_1, i_mask_reg_2);

        /* storing (l_m_32rem/16) registers */
        libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                i_gp_reg_out, 0, l_ldo * i_micro_kernel_config->datatype_size_out,
                                                                i_micro_kernel_config->vmove_instruction_out, 0, 0, l_m_32rem/16 );

        /* advance output pointer */
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                         i_gp_reg_out, (long long)i_micro_kernel_config->datatype_size_out * 64 );


        /* close n footer */
        libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                    i_gp_reg_n_loop, l_n_4mul );

      }

      /* transpose n_4rem x m_32rem block */
      if ( l_n_4rem > 0 ) {
        /* load 4 registers with two half rows */
        {
          const unsigned int ld_idx[32] = { 0, 2 };
          unsigned int l_mask_regs[2] = { 0 };
          l_mask_regs[0] = i_mask_reg_3;    l_mask_regs[1] = i_mask_reg_4;
          libxsmm_generator_transform_Xway_half_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                   i_gp_reg_in, 0, l_ldi * i_micro_kernel_config->datatype_size_in,
                                                                   ld_idx, 2, LIBXSMM_X86_INSTR_VBROADCASTI32X8, 2, l_mask_regs, l_n_4rem );
        }

        /* transpose two 2x32 blocks */
        libxsmm_generator_transform_two_8x8_08bit_vnni4_to_vnni4t_avx512( io_generated_code, i_micro_kernel_config->vector_name, 0, l_shuffle_op, i_mask_reg_1, i_mask_reg_2);

        /* storing (l_m_32rem/16) registers */
        libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                i_gp_reg_out, 0, l_ldo * i_micro_kernel_config->datatype_size_out,
                                                                i_micro_kernel_config->vmove_instruction_out, 1, i_mask_reg_5, l_m_32rem/16 );
      }
    }
  }

  else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_norm_08bit_avx512_mnblock_micro_kernel( libxsmm_generated_code*                 io_generated_code,
                                                                                  const unsigned int                      i_gp_reg_in,
                                                                                  const unsigned int                      i_gp_reg_out,
                                                                                  const unsigned int                      i_mask_reg_0,
                                                                                  const unsigned int                      i_perm_1st_stage_reg,
                                                                                  const unsigned int                      i_perm_2nd_stage_reg,
                                                                                  const unsigned int                      i_m_step,
                                                                                  const unsigned int                      i_n_step,
                                                                                  const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                  const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int l_zmm = 0;
  unsigned int l_zmm_tmp = i_n_step/4;
  unsigned int l_chunk = 0;

  /* load i_n_step registers */
  for ( l_zmm = 0; l_zmm < i_n_step/4; l_zmm++ ) {
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, LIBXSMM_X86_INSTR_VMOVUPS,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, l_zmm * i_mateltwise_desc->ldi * 4 * i_micro_kernel_config->datatype_size_in,
                                      i_micro_kernel_config->vector_name, l_zmm, i_mask_reg_0, 1, 0 );
  }

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_in, (long long)i_m_step * 4 * i_micro_kernel_config->datatype_size_in );

  /* create normal format from VNNI4 */
  for ( l_zmm = 0; l_zmm < i_n_step/4; l_zmm++) {
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSHUFB, i_micro_kernel_config->vector_name,
                                                            i_perm_1st_stage_reg, l_zmm, l_zmm, 0, 0, 0, 0 );


    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMD, i_micro_kernel_config->vector_name,
                                                            l_zmm, i_perm_2nd_stage_reg, l_zmm,  0, 0, 0, LIBXSMM_X86_IMM_UNDEF );

    if ( i_m_step % 16 == 0 ) {
      for (l_chunk = 0; l_chunk < 4; l_chunk++) {
        libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8 ( io_generated_code, LIBXSMM_X86_INSTR_VEXTRACTI32X4, i_micro_kernel_config->vector_name,
            i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, (l_chunk + 4 * l_zmm) * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out, 0, LIBXSMM_X86_VEC_REG_UNDEF, l_zmm, 0, 0, l_chunk );
      }
    } else {
      for (l_chunk = 0; l_chunk < 4; l_chunk++) {
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( io_generated_code, LIBXSMM_X86_INSTR_VEXTRACTI32X4, i_micro_kernel_config->vector_name,
            l_zmm, LIBXSMM_X86_VEC_REG_UNDEF, l_zmm_tmp + l_chunk, 0, 0, 0, l_chunk );
        libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, LIBXSMM_X86_INSTR_VMOVDQU8,
                                          i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, (l_chunk + 4 * l_zmm) * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                          'x', l_zmm_tmp + l_chunk, i_mask_reg_0, 0, 1 );
      }
    }
  }

  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, (long long)i_m_step * i_micro_kernel_config->datatype_size_out );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_norm_08bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                         libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                         const unsigned int                      i_gp_reg_in,
                                                                         const unsigned int                      i_gp_reg_out,
                                                                         const unsigned int                      i_gp_reg_m_loop,
                                                                         const unsigned int                      i_gp_reg_n_loop,
                                                                         const unsigned int                      i_gp_reg_mask,
                                                                         const unsigned int                      i_mask_reg_0,
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc) {
  unsigned int n_block = 4;
  unsigned int l_m_remainder = i_mateltwise_desc->m % 16;
  unsigned int l_m_full = i_mateltwise_desc->m / 16;
  unsigned int l_n_full = i_mateltwise_desc->n / n_block;
  unsigned int l_perm_1st_stage_reg = 31;
  unsigned char l_perm_table_1st_stage[64] ={0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                                            0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                                            0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                                            0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 };
  unsigned int l_perm_2nd_stage_reg = 30;
  unsigned int l_perm_table_2nd_stage[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};

  /* check if the right combination of knobs is provided */
  if (i_mateltwise_desc->n % 4 != 0) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  while (i_mateltwise_desc->n % n_block != 0) {
    n_block -= 4;
  }

  /* set masks */
  if ( l_m_remainder > 0 ) {
    /* load mask */
    /* set mask with (( (unsigned long long)1 << l_m_remainder ) - 1 ) */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask, 1 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SHLQ, i_gp_reg_mask, l_m_remainder );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, i_gp_reg_mask, 1 );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVW_GPR_LD, i_gp_reg_mask, i_mask_reg_0 );
  }

  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) l_perm_table_1st_stage, "perm_table_1st_stage_", i_micro_kernel_config->vector_name, l_perm_1st_stage_reg);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) l_perm_table_2nd_stage, "perm_table_2nd_stage_", i_micro_kernel_config->vector_name, l_perm_2nd_stage_reg);

  /* open n loop */
  if ( l_n_full > 1 ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
            i_gp_reg_n_loop, n_block );
  }

  /* full m iterations in a loop */
  if ( l_m_full > 0 ) {
    /* open m loop */
    if ( l_m_full > 1 ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_m_loop, 16 );
    }
    libxsmm_generator_transform_vnni4_to_norm_08bit_avx512_mnblock_micro_kernel( io_generated_code, i_gp_reg_in, i_gp_reg_out, 0, l_perm_1st_stage_reg, l_perm_2nd_stage_reg, 16, n_block, i_micro_kernel_config, i_mateltwise_desc );
    /* close m footer */
    if ( l_m_full > 1 ) {
      libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_m_loop, l_m_full*16 );
    }
  }

  /* m remainder masked */
  if ( l_m_remainder > 0 ) {
    libxsmm_generator_transform_vnni4_to_norm_08bit_avx512_mnblock_micro_kernel( io_generated_code, i_gp_reg_in, i_gp_reg_out, i_mask_reg_0, l_perm_1st_stage_reg, l_perm_2nd_stage_reg, l_m_remainder, n_block, i_micro_kernel_config, i_mateltwise_desc );
  }

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
          i_gp_reg_in, ((long long)n_block * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in) - ((long long)i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->m * 4) );

  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
          i_gp_reg_out, ((long long)n_block * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->m) );

  /* close n loop */
  if ( l_n_full > 1 ) {
    libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_n_loop, l_n_full*n_block  );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni2_08bit_avx512_mnblock_micro_kernel( libxsmm_generated_code*                io_generated_code,
                                                                                  const unsigned int                      i_gp_reg_in,
                                                                                  const unsigned int                      i_gp_reg_out,
                                                                                  const unsigned int                      i_mask_reg_0,
                                                                                  const unsigned int                      i_perm_1st_stage_reg,
                                                                                  const unsigned int                      i_m_step,
                                                                                  const unsigned int                      i_n_step,
                                                                                  const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                  const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int l_zmm = 0;
  unsigned int l_zmm_tmp = i_n_step/4;
  unsigned int l_chunk = 0;

  /* load i_n_step registers */
  for ( l_zmm = 0; l_zmm < i_n_step/4; l_zmm++ ) {
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, LIBXSMM_X86_INSTR_VMOVUPS,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, l_zmm * i_mateltwise_desc->ldi * 4 * i_micro_kernel_config->datatype_size_in,
                                      i_micro_kernel_config->vector_name, l_zmm, i_mask_reg_0, 1, 0 );
  }

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_in, (long long)i_m_step * 4 * i_micro_kernel_config->datatype_size_in );

  /* create VNNI2 format from VNNI4 */
  for ( l_zmm = 0; l_zmm < i_n_step/4; l_zmm++) {
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMW, i_micro_kernel_config->vector_name,
                                                            l_zmm, i_perm_1st_stage_reg, l_zmm,  0, 0, 0, LIBXSMM_X86_IMM_UNDEF );

    if ( i_m_step % 16 == 0 ) {
      for (l_chunk = 0; l_chunk < 2; l_chunk++) {
        libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8 ( io_generated_code, LIBXSMM_X86_INSTR_VEXTRACTI32X8, i_micro_kernel_config->vector_name,
            i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, (l_chunk + 2 * l_zmm) * 2 * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out, 0, LIBXSMM_X86_VEC_REG_UNDEF, l_zmm, 0, 0, l_chunk );
      }
    } else {
      for (l_chunk = 0; l_chunk < 2; l_chunk++) {
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( io_generated_code, LIBXSMM_X86_INSTR_VEXTRACTI32X8, i_micro_kernel_config->vector_name,
            l_zmm, LIBXSMM_X86_VEC_REG_UNDEF, l_zmm_tmp + l_chunk, 0, 0, 0, l_chunk );
        libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, LIBXSMM_X86_INSTR_VMOVDQU16,
                                          i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, (l_chunk + 2 * l_zmm) * 2 * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                          'y', l_zmm_tmp + l_chunk, i_mask_reg_0, 0, 1 );
      }
    }
  }

  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, 2LL * i_m_step * i_micro_kernel_config->datatype_size_out );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni2_08bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                         libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                         const unsigned int                      i_gp_reg_in,
                                                                         const unsigned int                      i_gp_reg_out,
                                                                         const unsigned int                      i_gp_reg_m_loop,
                                                                         const unsigned int                      i_gp_reg_n_loop,
                                                                         const unsigned int                      i_gp_reg_mask,
                                                                         const unsigned int                      i_mask_reg_0,
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc) {
  unsigned int n_block = 4;
  unsigned int l_m_remainder = i_mateltwise_desc->m % 16;
  unsigned int l_m_full = i_mateltwise_desc->m / 16;
  unsigned int l_n_full = i_mateltwise_desc->n / n_block;
  unsigned int l_perm_1st_stage_reg = 31;
  unsigned short l_perm_table_1st_stage[32] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
                                             1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
  /* check if the right combination of knobs is provided */
  if (i_mateltwise_desc->n % 4 != 0) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  while (i_mateltwise_desc->n % n_block != 0) {
    n_block -= 4;
  }

  /* set masks */
  if ( l_m_remainder > 0 ) {
    /* load mask */
    /* set mask with (( (unsigned long long)1 << l_m_remainder ) - 1 ) */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask, 1 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SHLQ, i_gp_reg_mask, l_m_remainder );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, i_gp_reg_mask, 1 );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVW_GPR_LD, i_gp_reg_mask, i_mask_reg_0 );
  }

  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) l_perm_table_1st_stage, "perm_table_1st_stage_", i_micro_kernel_config->vector_name, l_perm_1st_stage_reg);

  /* open n loop */
  if ( l_n_full > 1 ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
            i_gp_reg_n_loop, n_block );
  }

  /* full m iterations in a loop */
  if ( l_m_full > 0 ) {
    /* open m loop */
    if ( l_m_full > 1 ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_m_loop, 16 );
    }
    libxsmm_generator_transform_vnni4_to_vnni2_08bit_avx512_mnblock_micro_kernel( io_generated_code, i_gp_reg_in, i_gp_reg_out, 0, l_perm_1st_stage_reg, 16, n_block, i_micro_kernel_config, i_mateltwise_desc );
    /* close m footer */
    if ( l_m_full > 1 ) {
      libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_m_loop, l_m_full*16 );
    }
  }

  /* m remainder masked */
  if ( l_m_remainder > 0 ) {
    libxsmm_generator_transform_vnni4_to_vnni2_08bit_avx512_mnblock_micro_kernel( io_generated_code, i_gp_reg_in, i_gp_reg_out, i_mask_reg_0, l_perm_1st_stage_reg, l_m_remainder, n_block, i_micro_kernel_config, i_mateltwise_desc );
  }

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
          i_gp_reg_in, ((long long)n_block * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in) - ((long long)i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->m * 4) );

  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
          i_gp_reg_out, ((long long)n_block * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->m * 2) );

  /* close n loop */
  if ( l_n_full > 1 ) {
    libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_n_loop, l_n_full*n_block  );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni4_08bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                         libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                         const unsigned int                      i_gp_reg_in,
                                                                         const unsigned int                      i_gp_reg_out,
                                                                         const unsigned int                      i_gp_reg_m_loop,
                                                                         const unsigned int                      i_gp_reg_n_loop,
                                                                         const unsigned int                      i_gp_reg_mask,
                                                                         const unsigned int                      i_mask_reg_0,
                                                                         const unsigned int                      i_mask_reg_1,
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                         const unsigned int                      i_pad_vnni ) {
  unsigned int l_perm_1st_stage_reg = 31;
  unsigned int n_block = 8;
  unsigned int l_m_remainder = i_mateltwise_desc->m % 64;
  unsigned int l_m_full = i_mateltwise_desc->m / 64;
  unsigned int l_n_remainder = i_mateltwise_desc->n % n_block;
  unsigned int l_n_full = i_mateltwise_desc->n / n_block;
  unsigned int l_m_remainder_4store = (l_m_remainder * 4) % 64;
  int perm_table_1st_stage[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};

  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) perm_table_1st_stage, "perm_table_1st_stage_", i_micro_kernel_config->vector_name, l_perm_1st_stage_reg);

  /* check if the right combination of knobs is provided */
  if ( (i_pad_vnni == 0) && (i_mateltwise_desc->n % 4 != 0) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* set masks */
  if ( l_m_remainder > 0 ) {
    /* load mask */
    /* set mask with (( (unsigned long long)1 << l_m_remainder ) - 1 ) */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask, 1 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SHLQ, i_gp_reg_mask, l_m_remainder );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, i_gp_reg_mask, 1 );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVQ_GPR_LD, i_gp_reg_mask, i_mask_reg_0 );

    /* create store masking */
    if ( l_m_remainder_4store > 0 ) {
      /* store mask */
      /* set mask with (( (unsigned long long)1 << l_m_remainder_4store ) - 1 ) */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask, 1 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SHLQ, i_gp_reg_mask, l_m_remainder_4store );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, i_gp_reg_mask, 1 );
      libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVQ_GPR_LD, i_gp_reg_mask, i_mask_reg_1 );
    }
  }

  if ( l_n_full > 0 ) {
    /* open n loop */
    if ( l_n_full > 1 ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
              i_gp_reg_n_loop, n_block );
    }

    /* full m iterations in a loop */
    if ( l_m_full > 0 ) {
      /* open m loop */
      if ( l_m_full > 1 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
        libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                i_gp_reg_m_loop, 64 );
      }
      libxsmm_generator_transform_norm_to_vnni4_08bit_avx512_mnblock_micro_kernel( io_generated_code, i_gp_reg_in, i_gp_reg_out, 0, 0, l_perm_1st_stage_reg,
              64, n_block, i_micro_kernel_config, i_mateltwise_desc );

      /* close m footer */
      if ( l_m_full > 1 ) {
        libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                i_gp_reg_m_loop, l_m_full*64 );
      }
    }

    /* m remainder masked */
    if ( l_m_remainder > 0 ) {
      libxsmm_generator_transform_norm_to_vnni4_08bit_avx512_mnblock_micro_kernel( io_generated_code, i_gp_reg_in, i_gp_reg_out, i_mask_reg_0, i_mask_reg_1,
              l_perm_1st_stage_reg, l_m_remainder, n_block, i_micro_kernel_config, i_mateltwise_desc );
    }

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
            i_gp_reg_in, ((long long)n_block * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in) - ((long long)i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->m) );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
            i_gp_reg_out, ((long long)n_block * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->m * 4) );

    /* close n loop */
    if ( l_n_full > 1 ) {
      libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
              i_gp_reg_n_loop, l_n_full*n_block  );
    }
  }
  if ( l_n_remainder > 0 ) {
    /* full m iterations in a loop */
    if ( l_m_full > 0 ) {
      /* open m loop */
      if ( l_m_full > 1 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
        libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                         i_gp_reg_m_loop, 64 );
      }
      libxsmm_generator_transform_norm_to_vnni4_08bit_avx512_mnblock_micro_kernel( io_generated_code, i_gp_reg_in, i_gp_reg_out, 0, 0, l_perm_1st_stage_reg,
                                                                                   64, l_n_remainder, i_micro_kernel_config, i_mateltwise_desc );

      /* close m footer */
      if ( l_m_full > 1 ) {
        libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                    i_gp_reg_m_loop, l_m_full*64 );
      }
    }

    /* m remainder masked */
    if ( l_m_remainder > 0 ) {
      libxsmm_generator_transform_norm_to_vnni4_08bit_avx512_mnblock_micro_kernel( io_generated_code, i_gp_reg_in, i_gp_reg_out, i_mask_reg_0, i_mask_reg_1,
                                                                                   l_perm_1st_stage_reg, l_m_remainder, l_n_remainder, i_micro_kernel_config, i_mateltwise_desc );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni8_to_vnni8t_16bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                           libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                           const unsigned int                      i_gp_reg_in,
                                                                           const unsigned int                      i_gp_reg_out,
                                                                           const unsigned int                      i_gp_reg_m_loop,
                                                                           const unsigned int                      i_gp_reg_n_loop,
                                                                           const unsigned int                      i_gp_reg_mask,
                                                                           const unsigned int                      i_mask_reg_0,
                                                                           const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                           const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  LIBXSMM_UNUSED( i_gp_reg_mask );
  LIBXSMM_UNUSED( i_mask_reg_0 );

  libxsmm_generator_transform_vnni8_to_vnni8t_mbit_scalar_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                           i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                           i_micro_kernel_config, i_mateltwise_desc );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni8_16bit_avx512_mnblock_micro_kernel( libxsmm_generated_code*                 io_generated_code,
                                                                                 const unsigned int                      i_gp_reg_in,
                                                                                 const unsigned int                      i_gp_reg_out,
                                                                                 const unsigned int                      i_mask_reg_0,
                                                                                 const unsigned int                      i_mask_reg_1,
                                                                                 const unsigned int                      i_vnni_lo_reg,
                                                                                 const unsigned int                      i_vnni_hi_reg,
                                                                                 const unsigned int                      i_vnni_lo_reg_2,
                                                                                 const unsigned int                      i_vnni_hi_reg_2,
                                                                                 const unsigned int                      i_vnni_lo_reg_4,
                                                                                 const unsigned int                      i_vnni_hi_reg_4,
                                                                                 const unsigned int                      i_m_step,
                                                                                 const unsigned int                      i_n_step,
                                                                                 const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int l_zmm = 0;
  unsigned int l_m_1_2 = ((io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX)) ? 8 : 16;

#if 0
  if ((io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX)) {
    l_m_bound = 8;
  }
#endif

  /* load registers */
  /* zmm0: 0a, 1a, 2a, 3a, ..., 31a */
  /* zmm1: 0b, 1b,              31b */
  /* zmm2: 0c, ...,             31c */
  /* zmm3: 0d, ...,             31d */
  /* zmm4: 0e, ...,             31e */
  /* zmm5: 0f, ...,             31f */
  /* zmm6: 0g, ...,             31g */
  /* zmm7: 0h, ...,             31h */
  for ( l_zmm = 0; l_zmm < i_n_step; l_zmm++ ) {
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, l_zmm * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                      i_micro_kernel_config->vector_name, l_zmm, i_mask_reg_0, 1, 0 );
  }
  for ( ; l_zmm < LIBXSMM_UP( i_n_step, 8 ); l_zmm++ ) {
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, i_micro_kernel_config->vector_name,
                                              l_zmm, l_zmm, l_zmm );
  }

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_in, (long long)i_m_step * i_micro_kernel_config->datatype_size_in );

  /* create VNNI interleaved format */
  /* 16bit -> 32bit */
  /*  zmm1:  0a,  0b,  1a,  1b, ..., 15a, 15b -> lo0 */
  /*  zmm9: 16a, 16b, 17a, 17b, ..., 31a, 31b -> hi0 */
  /*  zmm3,  0c,  0d,  1c,  1d, ..., 15c, 15d -> lo1 */
  /* zmm11, 16c, 16d, 17c, 17d, ..., 31c, 31d -> hi1 */
  /*  zmm5,  0e,  0f, ...            15e, 15f -> lo2 */
  /* zmm13, 16e, 16f, ...            31e, 31f -> hi2 */
  /*  zmm7,  0g,  0h, ...            15g, 15g -> lo3 */
  /* zmm15  16g, 16g, ...            31g, 31h -> hi3 */
  {
    unsigned int l_in_reg[4]    = {  0,  2,  4,  6 };
    unsigned int l_inout_reg[4] = {  1,  3,  5,  7 };
    unsigned int l_tmp_reg[4]   = {  9, 11, 13, 15 };
    for ( l_zmm = 0; l_zmm < 4; l_zmm++ ) {
      if ( i_m_step >= l_m_1_2) {
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64_LD, i_micro_kernel_config->vector_name,
                                                                l_inout_reg[l_zmm], LIBXSMM_X86_VEC_REG_UNDEF, l_tmp_reg[l_zmm], 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
      }
      libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2W, i_micro_kernel_config->vector_name,
                                                              l_in_reg[l_zmm], i_vnni_lo_reg, l_inout_reg[l_zmm], 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
      if ( i_m_step >= l_m_1_2) {
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2W, i_micro_kernel_config->vector_name,
                                                                l_in_reg[l_zmm], i_vnni_hi_reg, l_tmp_reg[l_zmm], 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
      }
    }
  }

  /* 32bit -> 64bit */
  /*  zmm1,zmm3   ->  zmm3:  0a,  0b,  0c,  0d, ...,  7a,  7b,  7c,  7d -> (lo0,lo1)lo -> lolo0 */
  /*  zmm1,zmm3   ->  zmm4:  8a,  8b,  8c,  8d, ..., 15a, 15b, 15c, 15d -> (lo0,l01)hi -> lohi0 */
  /*  zmm5,zmm7   ->  zmm7:  0e,  0f,  0g,  0h, ...,  7e,  7f,  7g,  7h -> (lo2,lo3)lo -> lolo1 */
  /*  zmm5,zmm7   ->  zmm8:  8e,  8f,  8g,  8h, ..., 15e, 15f, 15g, 15h -> (lo2,lo3)hi -> lohi1 */
  /*  zmm9,zmm11  -> zmm11: 16a, 16b, 16c, 16d, ..., 23a, 23b, 23c, 23d -> (hi0,hi1)lo -> hilo0 */
  /*  zmm9,zmm11  -> zmm12: 24a, 24b. 24c, 24d, ..., 31a, 31b, 31c, 31d -> (hi0,hi1)hi -> hihi0 */
  /*  zmm13,zmm15 -> zmm15: 16e, 16f, 16g, 16h, ..., 23e, 23f, 23g, 23h -> (hi2,hi3)lo -> hilo1 */
  /*  zmm13,zmm15 -> zmm16: 24e, 24f. 24g, 24h, ..., 31e, 31f, 31g, 31h -> (hi2,hi3)hi -> hihi1 */
  {
    unsigned int l_in_reg[4]    = {  1,  5,  9, 13 };
    unsigned int l_inout_reg[4] = {  3,  7, 11, 15 };
    unsigned int l_tmp_reg[4]   = {  4,  8, 12, 16 };
    unsigned int l_m_step_loop = (i_m_step >= l_m_1_2) ? 4 : 2;
    unsigned int l_m_1_4 = ((io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX)) ? 4 : 8;
    for ( l_zmm = 0; l_zmm < l_m_step_loop; l_zmm++ ) {
      if ( i_m_step > l_m_1_4 ) {
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64_LD, i_micro_kernel_config->vector_name,
                                                                l_inout_reg[l_zmm], LIBXSMM_X86_VEC_REG_UNDEF, l_tmp_reg[l_zmm], 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
      }
      libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2D, i_micro_kernel_config->vector_name,
                                                              l_in_reg[l_zmm], i_vnni_lo_reg_2, l_inout_reg[l_zmm], 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
      if ( i_m_step > l_m_1_4 ) {
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2D, i_micro_kernel_config->vector_name,
                                                                l_in_reg[l_zmm], i_vnni_hi_reg_2, l_tmp_reg[l_zmm], 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
      }
    }
  }

  /* 64bit -> 128bit */
  /* zmm3,zmm7   ->   zmm7:   0a,  0b,  0c,  0d,  0e,  0f,  0g,  0h, ...,  4a -- 4h */
  /* zmm3,zmm7   ->   zmm9:   5a,  5b,  5c,  5d,  5e,  5f,  5g,  5h, ...,  7a -- 7h */
  /* zmm4,zmm8   ->   zmm8:   8a --  8h , ..., 11a -- 11h */
  /* zmm4,zmm8   ->  zmm10:  12a -- 12h , ..., 15a -- 15h */
  /* zmm11,zmm15 ->  zmm15:  16a -- 16h , ..., 19a -- 19h */
  /* zmm11,zmm15 ->  zmm17:  20a -- 20h , ..., 23a -- 23h */
  /* zmm12,zmm16 ->  zmm16:  24a -- 24h , ..., 27a -- 27h */
  /* zmm12,zmm16 ->  zmm18:  28a -- 28h , ..., 31a -- 31h */
  {
    unsigned int l_in_reg[4]    = {  3,  4, 11, 12 };
    unsigned int l_inout_reg[4] = {  7,  8, 15, 16 };
    unsigned int l_tmp_reg[4]   = {  9, 10, 17, 18 };
    unsigned int l_m_vlen = ((io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX)) ? 4 : 8;
    for ( l_zmm = 0; l_zmm < LIBXSMM_UPDIV(i_m_step, l_m_vlen); l_zmm++ ) {
      if ( i_m_step > ((l_m_vlen*(l_zmm+1)) - (l_m_vlen/2)) ) {
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64_LD, i_micro_kernel_config->vector_name,
                                                                l_inout_reg[l_zmm], LIBXSMM_X86_VEC_REG_UNDEF, l_tmp_reg[l_zmm], 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
      }
      libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2Q, i_micro_kernel_config->vector_name,
                                                              l_in_reg[l_zmm], i_vnni_lo_reg_4, l_inout_reg[l_zmm], 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
      if ( i_m_step > ((l_m_vlen*(l_zmm+1)) - (l_m_vlen/2)) ) {
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2Q, i_micro_kernel_config->vector_name,
                                                                l_in_reg[l_zmm], i_vnni_hi_reg_4, l_tmp_reg[l_zmm], 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
      }
    }
  }

  /* store VNNI packed vectors */
  {
    unsigned int l_out_reg[8] = { 7, 9, 8, 10, 15, 17, 16, 18 };
    unsigned int l_write_cnt = 0;
    unsigned int l_m_vlen = ((io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX)) ? 2 : 4;
    for ( l_zmm = 0; l_zmm < 8; l_zmm++ ) {
      if ( l_write_cnt < i_m_step ) {
        if ( l_write_cnt + l_m_vlen <= i_m_step ) {
          libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                            i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, l_zmm * l_m_vlen*8 * i_micro_kernel_config->datatype_size_out,
                                            i_micro_kernel_config->vector_name, l_out_reg[l_zmm], 0, 1, 1 );
        } else {
          libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                            i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, l_zmm * l_m_vlen*8 * i_micro_kernel_config->datatype_size_out,
                                            i_micro_kernel_config->vector_name, l_out_reg[l_zmm], i_mask_reg_1, 0, 1 );

        }
      }
      l_write_cnt += l_m_vlen;
    }
  }

  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, (long long)i_m_step * 8 * i_micro_kernel_config->datatype_size_out );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni8_16bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                         libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                         const unsigned int                      i_gp_reg_in,
                                                                         const unsigned int                      i_gp_reg_out,
                                                                         const unsigned int                      i_gp_reg_m_loop,
                                                                         const unsigned int                      i_gp_reg_n_loop,
                                                                         const unsigned int                      i_gp_reg_mask,
                                                                         const unsigned int                      i_mask_reg_0,
                                                                         const unsigned int                      i_mask_reg_1,
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                         const unsigned int                      i_pad_vnni ) {
  /* perm-table register mappings */
  unsigned int l_vnni_lo_reg = 31;
  unsigned int l_vnni_hi_reg = 30;
  unsigned int l_vnni_lo_reg_2 = 29;
  unsigned int l_vnni_hi_reg_2 = 28;
  unsigned int l_vnni_lo_reg_4 = 27;
  unsigned int l_vnni_hi_reg_4 = 26;
  /* m-blocking control */
  unsigned int l_m_entries = ((io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX)) ? 16 : 32;
  unsigned int l_m_remainder = i_mateltwise_desc->m % l_m_entries;
  unsigned int l_m_full = i_mateltwise_desc->m / l_m_entries;
  /* n-blocking control */
  unsigned int l_n_step = 8;
  unsigned int l_n_remainder = i_mateltwise_desc->n % l_n_step;
  unsigned int l_n_full = i_mateltwise_desc->n / l_n_step;

  if (l_m_entries == 32) {
    short perm_table_vnni_lo[32] = {32, 0, 33, 1, 34, 2, 35, 3, 36, 4, 37, 5, 38, 6, 39, 7, 40, 8, 41, 9, 42, 10, 43, 11, 44, 12, 45, 13, 46, 14, 47, 15};
    short perm_table_vnni_hi[32] = {48, 16, 49, 17, 50, 18, 51, 19, 52, 20, 53, 21, 54, 22, 55, 23, 56, 24, 57, 25, 58, 26, 59, 27, 60, 28, 61, 29, 62, 30, 63, 31};
    int perm_table_vnni_lo_2[16] = {16, 0, 17, 1, 18, 2, 19, 3, 20, 4, 21, 5, 22, 6, 23, 7 };
    int perm_table_vnni_hi_2[16] = {24, 8, 25, 9, 26, 10, 27, 11, 28, 12, 29, 13, 30, 14, 31, 15};
    long long perm_table_vnni_lo_4[8]  = {8, 0, 9, 1, 10, 2, 11, 3};
    long long perm_table_vnni_hi_4[8]  = {12, 4, 13, 5, 14, 6, 15, 7};

    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) perm_table_vnni_hi, "perm_table_vnni_hi_", i_micro_kernel_config->vector_name, l_vnni_hi_reg);
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) perm_table_vnni_lo, "perm_table_vnni_lo_", i_micro_kernel_config->vector_name, l_vnni_lo_reg);
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) perm_table_vnni_lo_2, "perm_table_vnni_lo_2_", i_micro_kernel_config->vector_name, l_vnni_lo_reg_2);
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) perm_table_vnni_hi_2, "perm_table_vnni_hi_2_", i_micro_kernel_config->vector_name, l_vnni_hi_reg_2);
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) perm_table_vnni_lo_4, "perm_table_vnni_lo_4_", i_micro_kernel_config->vector_name, l_vnni_lo_reg_4);
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) perm_table_vnni_hi_4, "perm_table_vnni_hi_4_", i_micro_kernel_config->vector_name, l_vnni_hi_reg_4);
  } else {
    short _perm_table_vnni_lo[16] = {16, 0, 17, 1, 18, 2, 19, 3, 20, 4, 21, 5, 22, 6, 23, 7};
    short _perm_table_vnni_hi[16] = {24, 8, 25, 9, 26, 10, 27, 11, 28, 12, 29, 13, 30, 14, 31, 15};
    int _perm_table_vnni_lo_2[8]  = {8, 0, 9, 1, 10, 2, 11, 3};
    int _perm_table_vnni_hi_2[8]  = {12, 4, 13, 5, 14, 6, 15, 7};
    long long _perm_table_vnni_lo_4[4]  = {4, 0, 5, 1};
    long long _perm_table_vnni_hi_4[4]  = {6, 2, 7, 3};

    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) _perm_table_vnni_hi, "perm_table_vnni_hi_", i_micro_kernel_config->vector_name, l_vnni_hi_reg);
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) _perm_table_vnni_lo, "perm_table_vnni_lo_", i_micro_kernel_config->vector_name, l_vnni_lo_reg);
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) _perm_table_vnni_lo_2, "perm_table_vnni_lo_2_", i_micro_kernel_config->vector_name, l_vnni_lo_reg_2);
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) _perm_table_vnni_hi_2, "perm_table_vnni_hi_2_", i_micro_kernel_config->vector_name, l_vnni_hi_reg_2);
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) _perm_table_vnni_lo_4, "perm_table_vnni_lo_4_", i_micro_kernel_config->vector_name, l_vnni_lo_reg_4);
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) _perm_table_vnni_hi_4, "perm_table_vnni_hi_4_", i_micro_kernel_config->vector_name, l_vnni_hi_reg_4);
  }

  /* check if the right combination of knobs is provided */
  if ( (i_pad_vnni == 0) && (i_mateltwise_desc->n % 8 > 0) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* set masks */
  if ( l_m_remainder > 0 ) {
    /* load mask */
    unsigned int l_mask_instr = (l_m_entries == 32) ? LIBXSMM_X86_INSTR_KMOVD_GPR_LD : LIBXSMM_X86_INSTR_KMOVW_GPR_LD ;
    const unsigned long long l_load_mask = ( (unsigned long long)1 << l_m_remainder ) - 1;
    const unsigned long long l_store_mask = (l_m_remainder % (l_m_entries/8) == 0) ? (unsigned long long)0xffffffff : ( (unsigned long long)1 << ((l_m_remainder % (l_m_entries/8)) * 8) ) - 1;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask, l_load_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, l_mask_instr, i_gp_reg_mask, i_mask_reg_0 );

    /* create store masking */
    l_mask_instr = (l_m_entries == 32) ? LIBXSMM_X86_INSTR_KMOVD_GPR_LD : LIBXSMM_X86_INSTR_KMOVW_GPR_LD ;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask, l_store_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, l_mask_instr, i_gp_reg_mask, i_mask_reg_1 );
  }

  if ( l_n_full > 0 ) {
    /* open n loop */
    if ( l_n_full > 1 ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_n_loop, l_n_step );
    }

    /* full m iterations in a loop */
    if ( l_m_full > 0 ) {
      /* open m loop */
      if ( l_m_full > 1 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
        libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                         i_gp_reg_m_loop, l_m_entries );
      }

      libxsmm_generator_transform_norm_to_vnni8_16bit_avx512_mnblock_micro_kernel( io_generated_code, i_gp_reg_in, i_gp_reg_out, 0, 0, l_vnni_lo_reg, l_vnni_hi_reg, l_vnni_lo_reg_2, l_vnni_hi_reg_2,
          l_vnni_lo_reg_4, l_vnni_hi_reg_4, l_m_entries, l_n_step, i_micro_kernel_config, i_mateltwise_desc );

      /* close m footer */
      if ( l_m_full > 1 ) {
        libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                    i_gp_reg_m_loop, l_m_full*l_m_entries );
      }
    }
    /* m remainder masked */
    if ( l_m_remainder > 0 ) {
      libxsmm_generator_transform_norm_to_vnni8_16bit_avx512_mnblock_micro_kernel( io_generated_code, i_gp_reg_in, i_gp_reg_out, i_mask_reg_0, i_mask_reg_1,
                                                                                  l_vnni_lo_reg, l_vnni_hi_reg, l_vnni_lo_reg_2, l_vnni_hi_reg_2, l_vnni_lo_reg_4, l_vnni_hi_reg_4, l_m_remainder, l_n_step, i_micro_kernel_config, i_mateltwise_desc );
    }

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, (l_n_step * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in) - ((long long)i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->m) );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (l_n_step * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->m * 8) );

    /* close n loop */
    if ( l_n_full > 1 ) {
      libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                  i_gp_reg_n_loop, l_n_full * l_n_step  );
    }
  }

  if ( l_n_remainder > 0 ) {
    /* full m iterations in a loop */
    if ( l_m_full > 0 ) {
      /* open m loop */
      if ( l_m_full > 1 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
        libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                         i_gp_reg_m_loop, l_m_entries );
      }

      libxsmm_generator_transform_norm_to_vnni8_16bit_avx512_mnblock_micro_kernel( io_generated_code, i_gp_reg_in, i_gp_reg_out, 0, 0, l_vnni_lo_reg, l_vnni_hi_reg,  l_vnni_lo_reg_2, l_vnni_hi_reg_2,
          l_vnni_lo_reg_4, l_vnni_hi_reg_4, l_m_entries, l_n_remainder, i_micro_kernel_config, i_mateltwise_desc );

      /* close m footer */
      if ( l_m_full > 1 ) {
        libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                    i_gp_reg_m_loop, l_m_full*l_m_entries );
      }
    }
    /* m remainder masked */
    if ( l_m_remainder > 0 ) {
      libxsmm_generator_transform_norm_to_vnni8_16bit_avx512_mnblock_micro_kernel( io_generated_code, i_gp_reg_in, i_gp_reg_out, i_mask_reg_0, i_mask_reg_1,
                                                                                  l_vnni_lo_reg, l_vnni_hi_reg, l_vnni_lo_reg_2, l_vnni_hi_reg_2, l_vnni_lo_reg_4, l_vnni_hi_reg_4, l_m_remainder, l_n_remainder, i_micro_kernel_config, i_mateltwise_desc );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni8_to_vnni8t_08bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                           libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                           const unsigned int                      i_gp_reg_in,
                                                                           const unsigned int                      i_gp_reg_out,
                                                                           const unsigned int                      i_gp_reg_m_loop,
                                                                           const unsigned int                      i_gp_reg_n_loop,
                                                                           const unsigned int                      i_gp_reg_mask,
                                                                           const unsigned int                      i_mask_reg_0,
                                                                           const unsigned int                      i_mask_reg_1,
                                                                           const unsigned int                      i_mask_reg_2,
                                                                           const unsigned int                      i_mask_reg_3,
                                                                           const unsigned int                      i_mask_reg_4,
                                                                           const unsigned int                      i_mask_reg_5,
                                                                           const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                           const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  LIBXSMM_UNUSED( i_gp_reg_mask );
  LIBXSMM_UNUSED( i_mask_reg_0 );
  LIBXSMM_UNUSED( i_mask_reg_1 );
  LIBXSMM_UNUSED( i_mask_reg_2 );
  LIBXSMM_UNUSED( i_mask_reg_3 );
  LIBXSMM_UNUSED( i_mask_reg_4 );
  LIBXSMM_UNUSED( i_mask_reg_5 );

  libxsmm_generator_transform_vnni8_to_vnni8t_mbit_scalar_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                           i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                           i_micro_kernel_config, i_mateltwise_desc );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni8_08bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                         libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                         const unsigned int                      i_gp_reg_in,
                                                                         const unsigned int                      i_gp_reg_out,
                                                                         const unsigned int                      i_gp_reg_m_loop,
                                                                         const unsigned int                      i_gp_reg_n_loop,
                                                                         const unsigned int                      i_gp_reg_mask,
                                                                         const unsigned int                      i_mask_reg_0,
                                                                         const unsigned int                      i_mask_reg_1,
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                         const unsigned int                      i_pad_vnni ) {
  LIBXSMM_UNUSED( i_gp_reg_mask );
  LIBXSMM_UNUSED( i_mask_reg_0 );
  LIBXSMM_UNUSED( i_mask_reg_1 );

  libxsmm_generator_transform_norm_to_vnni8_mbit_scalar_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                         i_micro_kernel_config, i_mateltwise_desc, i_pad_vnni );
}


LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_padnm_mod2_16bit_avx512_mnblock_micro_kernel( libxsmm_generated_code*                 io_generated_code,
                                                                                    const unsigned int                      i_gp_reg_in,
                                                                                    const unsigned int                      i_gp_reg_out,
                                                                                    const unsigned int                      i_mask_reg_0,
                                                                                    const unsigned int                      i_mask_reg_1,
                                                                                    const unsigned int                      i_m_step_in,
                                                                                    const unsigned int                      i_m_step_out,
                                                                                    const unsigned int                      i_n_step,
                                                                                    const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                    const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int l_zmm = 0;

  /* load registers */
  for ( l_zmm = 0; l_zmm < i_n_step; ++l_zmm ) {
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, l_zmm * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                      i_micro_kernel_config->vector_name, l_zmm, i_mask_reg_0, 1, 0 );
  }
  if ( (i_n_step % 2 == 1) && ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD2) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2)) ) {
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, i_micro_kernel_config->vector_name,
                                              l_zmm, l_zmm, l_zmm );
  }

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_in, (long long)i_m_step_in * i_micro_kernel_config->datatype_size_in );

  /* storing registers */
  for ( l_zmm = 0; l_zmm < i_n_step; ++l_zmm ) {
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                      i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, l_zmm * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                      i_micro_kernel_config->vector_name, l_zmm, i_mask_reg_1, 0, 1 );
  }
  if ( (i_n_step % 2 == 1) && ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD2) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2)) ) {
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                      i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, l_zmm * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                      i_micro_kernel_config->vector_name, l_zmm, i_mask_reg_1, 0, 1 );
  }

  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, (long long)i_m_step_out * i_micro_kernel_config->datatype_size_out );
}

/*
 * Calling convention, this kernel assumes that all GPRs with are not in the argument list
 * and all xmm/ymm/zmm/tmm are caller save
 *
 * TODO; stack local variables....
 * */
LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_padnm_mod2_16bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                           libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                           const unsigned int                      i_gp_reg_in,
                                                                           const unsigned int                      i_gp_reg_out,
                                                                           const unsigned int                      i_gp_reg_m_loop,
                                                                           const unsigned int                      i_gp_reg_n_loop,
                                                                           const unsigned int                      i_gp_reg_mask,
                                                                           const unsigned int                      i_mask_reg_0,
                                                                           const unsigned int                      i_mask_reg_1,
                                                                           const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                           const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int l_m_remainder_in  = i_mateltwise_desc->m % 32;
  unsigned int l_m_remainder_out = ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD2) || (l_m_remainder_in%2 == 0) ) ? l_m_remainder_in : l_m_remainder_in + 1;
  unsigned int l_m_full = i_mateltwise_desc->m / 32;
  unsigned int l_n_remainder = i_mateltwise_desc->n % 16;
  unsigned int l_n_full = i_mateltwise_desc->n / 16;

  /* set masks */
  if ( l_m_remainder_in > 0 ) {
    /* load mask */
    const unsigned long long l_load_mask = ((unsigned long long)1 << l_m_remainder_in ) - 1;
    const unsigned long long l_store_mask = ( l_m_remainder_out == 32 ) ? (unsigned long long)0xffffffff : (unsigned long long)(((unsigned long long)1 << l_m_remainder_out ) - 1);
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask, l_load_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mask, i_mask_reg_0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mask, l_store_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mask, i_mask_reg_1 );
  }

  if ( l_n_full > 0 ) {
    /* open n loop */
    if ( l_n_full > 1 ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_n_loop, 16 );
    }

    /* full m iterations in a loop */
    if ( l_m_full > 0 ) {
      /* open m loop */
      if ( l_m_full > 1 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
        libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                         i_gp_reg_m_loop, 32 );
      }

      libxsmm_generator_transform_norm_padnm_mod2_16bit_avx512_mnblock_micro_kernel( io_generated_code, i_gp_reg_in, i_gp_reg_out, 0, 0,
                                                                                     32, 32, 16, i_micro_kernel_config, i_mateltwise_desc );

      /* close m footer */
      if ( l_m_full > 1 ) {
        libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                    i_gp_reg_m_loop, l_m_full*32 );
      }
    }
    /* m remainder masked */
    if ( l_m_remainder_in > 0 ) {
      libxsmm_generator_transform_norm_padnm_mod2_16bit_avx512_mnblock_micro_kernel( io_generated_code, i_gp_reg_in, i_gp_reg_out, i_mask_reg_0, i_mask_reg_1,
                                                                                     l_m_remainder_in, l_m_remainder_out, 16, i_micro_kernel_config, i_mateltwise_desc );
    }

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, (16LL * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in) - ((long long)i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->m) );

    /* advance input pointer */
    if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD2 ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_out, (16LL * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->m) );
    } else {
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_out, (16LL * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->m + i_micro_kernel_config->datatype_size_out * (i_mateltwise_desc->m % 2)) );
    }

    /* close n loop */
    if ( l_n_full > 1 ) {
      libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                  i_gp_reg_n_loop, l_n_full*16  );
    }
  }

  if ( l_n_remainder > 0 ) {
    /* full m iterations in a loop */
    if ( l_m_full > 0 ) {
      /* open m loop */
      if ( l_m_full > 1 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
        libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                         i_gp_reg_m_loop, 32 );
      }

      libxsmm_generator_transform_norm_padnm_mod2_16bit_avx512_mnblock_micro_kernel( io_generated_code, i_gp_reg_in, i_gp_reg_out, 0, 0,
                                                                                     32, 32, l_n_remainder, i_micro_kernel_config, i_mateltwise_desc );

      /* close m footer */
      if ( l_m_full > 1 ) {
        libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                    i_gp_reg_m_loop, l_m_full*32 );
      }
    }
    /* m remainder masked */
    if ( l_m_remainder_in > 0 ) {
      libxsmm_generator_transform_norm_padnm_mod2_16bit_avx512_mnblock_micro_kernel( io_generated_code, i_gp_reg_in, i_gp_reg_out, i_mask_reg_0, i_mask_reg_1,
                                                                                     l_m_remainder_in, l_m_remainder_out, l_n_remainder, i_micro_kernel_config, i_mateltwise_desc );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_padnm_mod4_08bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                           libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                           const unsigned int                      i_gp_reg_in,
                                                                           const unsigned int                      i_gp_reg_out,
                                                                           const unsigned int                      i_gp_reg_m_loop,
                                                                           const unsigned int                      i_gp_reg_n_loop,
                                                                           const unsigned int                      i_gp_reg_mask,
                                                                           const unsigned int                      i_mask_reg_0,
                                                                           const unsigned int                      i_mask_reg_1,
                                                                           const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                           const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  LIBXSMM_UNUSED( i_gp_reg_mask );
  LIBXSMM_UNUSED( i_mask_reg_0 );
  LIBXSMM_UNUSED( i_mask_reg_1 );

  libxsmm_generator_transform_norm_padnm_mod4_mbit_scalar_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                           i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                           i_micro_kernel_config, i_mateltwise_desc );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
                                                     libxsmm_loop_label_tracker*                    io_loop_label_tracker,
                                                     libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
                                                     const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
                                                     const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {
  unsigned int l_gp_reg_in  = LIBXSMM_X86_GP_REG_R8;
  unsigned int l_gp_reg_out = LIBXSMM_X86_GP_REG_R9;
  unsigned int l_gp_reg_mloop = LIBXSMM_X86_GP_REG_RAX;
  unsigned int l_gp_reg_nloop = LIBXSMM_X86_GP_REG_RDX;
  unsigned int l_gp_reg_mask = LIBXSMM_X86_GP_REG_R10;
  unsigned int l_gp_reg_mask_2 = LIBXSMM_X86_GP_REG_R11;
  unsigned int l_mask_reg_0 = 1;
  unsigned int l_mask_reg_1 = 2;
  unsigned int l_mask_reg_2 = 3;
  unsigned int l_mask_reg_3 = 4;
  unsigned int l_mask_reg_4 = 5;
  unsigned int l_mask_reg_5 = 6;
  unsigned int l_mask_reg_6 = 7;

  /* load pointers from struct */
  libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_param_struct,
                                   LIBXSMM_X86_GP_REG_UNDEF, 0, 32,
                                   l_gp_reg_in, 0 );

  libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_param_struct,
                                   LIBXSMM_X86_GP_REG_UNDEF, 0, 64,
                                   l_gp_reg_out, 0 );

  /* check leading dimnesions and sizes */
  if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) ||
       (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2_TO_VNNI2T)    ) {
    /* coverity[copy_paste_error] */
    if ( (i_mateltwise_desc->m > i_mateltwise_desc->ldi) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
      return;
    }
    if ( (i_mateltwise_desc->n > i_mateltwise_desc->ldo) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB );
      return;
    }
  } else if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2)     ||
              (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD) ||
              (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD2)        ||
              (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD2)        ||
              (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2)          ) {
    if ( (i_mateltwise_desc->m > i_mateltwise_desc->ldi) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
      return;
    }
    if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD2)  ||
         (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2)    ) {
      if ( (i_mateltwise_desc->m + i_mateltwise_desc->m%2) > i_mateltwise_desc->ldo ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB );
        return;
      }
    } else {
      if ( (i_mateltwise_desc->m > i_mateltwise_desc->ldo) ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB );
        return;
      }
    }
  } else {
    /* should not happen */
  }

  if ( ( LIBXSMM_DATATYPE_I64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
         LIBXSMM_DATATYPE_I64 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) ||
       ( LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
         LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) )  ) {
    if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT ) {
      libxsmm_generator_transform_norm_to_normt_64bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                          l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                          l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1, l_mask_reg_2,
                                                                          l_mask_reg_3, l_mask_reg_4, l_mask_reg_5,
                                                                          i_micro_kernel_config, i_mateltwise_desc );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else if ( ( LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
                LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) ||
              ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
                LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) ) {
    if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT ) {
      libxsmm_generator_transform_norm_to_normt_32bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                          l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                          l_gp_reg_mask, l_gp_reg_mask_2, l_mask_reg_0, l_mask_reg_1, l_mask_reg_2, l_mask_reg_3,
                                                                          l_mask_reg_4, l_mask_reg_5, l_mask_reg_6,
                                                                          i_micro_kernel_config, i_mateltwise_desc );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else if ( ( LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
                LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) ||
              ( LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
                LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) ||
              ( LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
                LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) )  ) {
    if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT ) {
      libxsmm_generator_transform_norm_to_normt_16bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                          l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                          l_gp_reg_mask, l_gp_reg_mask_2,
                                                                          l_mask_reg_0, l_mask_reg_1, l_mask_reg_2, l_mask_reg_3,
                                                                          l_mask_reg_4, l_mask_reg_5, l_mask_reg_6,
                                                                          i_micro_kernel_config, i_mateltwise_desc );
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8T ) {
      /* Call 64bit normal transpose */
      libxsmm_descriptor_blob blob;
      const libxsmm_meltw_descriptor *const mock_desc = libxsmm_meltw_descriptor_init2(&blob,
        LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, i_mateltwise_desc->m/8, i_mateltwise_desc->n,
        i_mateltwise_desc->ldi/8, i_mateltwise_desc->ldo, 0, 0,
        (unsigned short)LIBXSMM_MELTW_FLAG_UNARY_NONE, (unsigned short)LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, LIBXSMM_MELTW_OPERATION_UNARY);
      libxsmm_mateltwise_kernel_config l_trans_config;
      libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_trans_config, mock_desc);
      libxsmm_generator_transform_norm_to_normt_128bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                           l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                           l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1, l_mask_reg_2,
                                                                           l_mask_reg_3, l_mask_reg_4, l_mask_reg_5,
                                                                           &l_trans_config, mock_desc );
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4T ) {
      /* Call 64bit normal transpose */
      libxsmm_descriptor_blob blob;
      const libxsmm_meltw_descriptor *const mock_desc = libxsmm_meltw_descriptor_init2(&blob,
        LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, i_mateltwise_desc->m/4, i_mateltwise_desc->n,
        i_mateltwise_desc->ldi/4, i_mateltwise_desc->ldo, 0, 0,
        (unsigned short)LIBXSMM_MELTW_FLAG_UNARY_NONE, (unsigned short)LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, LIBXSMM_MELTW_OPERATION_UNARY);
      libxsmm_mateltwise_kernel_config l_trans_config;
      libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_trans_config, mock_desc);
      libxsmm_generator_transform_norm_to_normt_64bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                          l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                          l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1, l_mask_reg_2,
                                                                          l_mask_reg_3, l_mask_reg_4, l_mask_reg_5,
                                                                          &l_trans_config, mock_desc );
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2T ) {
      /* Call 32bit normal transpose */
      libxsmm_descriptor_blob blob;
      const libxsmm_meltw_descriptor *const mock_desc = libxsmm_meltw_descriptor_init2(&blob,
        LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, i_mateltwise_desc->m/2, i_mateltwise_desc->n,
        i_mateltwise_desc->ldi/2, i_mateltwise_desc->ldo, 0, 0,
        (unsigned short)LIBXSMM_MELTW_FLAG_UNARY_NONE, (unsigned short)LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, LIBXSMM_MELTW_OPERATION_UNARY);
      libxsmm_mateltwise_kernel_config l_trans_config;
      libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_trans_config, mock_desc);
      libxsmm_generator_transform_norm_to_normt_32bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                          l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                          l_gp_reg_mask, l_gp_reg_mask_2, l_mask_reg_0, l_mask_reg_1, l_mask_reg_2, l_mask_reg_3,
                                                                          l_mask_reg_4, l_mask_reg_5, l_mask_reg_6,
                                                                          &l_trans_config, mock_desc );
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI8T_TO_NORM ) {
      /* Call 64bit normal transpose */
      libxsmm_descriptor_blob blob;
      const libxsmm_meltw_descriptor *const mock_desc = libxsmm_meltw_descriptor_init2(&blob,
        LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, i_mateltwise_desc->m, i_mateltwise_desc->n/8,
        i_mateltwise_desc->ldi, i_mateltwise_desc->ldo/8, 0, 0,
        (unsigned short)LIBXSMM_MELTW_FLAG_UNARY_NONE, (unsigned short)LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, LIBXSMM_MELTW_OPERATION_UNARY);
      libxsmm_mateltwise_kernel_config l_trans_config;
      libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_trans_config, mock_desc);
      libxsmm_generator_transform_norm_to_normt_128bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                           l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                           l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1, l_mask_reg_2,
                                                                           l_mask_reg_3, l_mask_reg_4, l_mask_reg_5,
                                                                           &l_trans_config, mock_desc );
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4T_TO_NORM ) {
      /* Call 64bit normal transpose */
      libxsmm_descriptor_blob blob;
      const libxsmm_meltw_descriptor *const mock_desc = libxsmm_meltw_descriptor_init2(&blob,
        LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, i_mateltwise_desc->m, i_mateltwise_desc->n/4,
        i_mateltwise_desc->ldi, i_mateltwise_desc->ldo/4, 0, 0,
        (unsigned short)LIBXSMM_MELTW_FLAG_UNARY_NONE, (unsigned short)LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, LIBXSMM_MELTW_OPERATION_UNARY);
      libxsmm_mateltwise_kernel_config l_trans_config;
      libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_trans_config, mock_desc);
      libxsmm_generator_transform_norm_to_normt_64bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                          l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                          l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1, l_mask_reg_2,
                                                                          l_mask_reg_3, l_mask_reg_4, l_mask_reg_5,
                                                                          &l_trans_config, mock_desc );
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2T_TO_NORM ) {
      /* Call 32bit normal transpose */
      libxsmm_descriptor_blob blob;
      const libxsmm_meltw_descriptor *const mock_desc = libxsmm_meltw_descriptor_init2(&blob,
        LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, i_mateltwise_desc->m, i_mateltwise_desc->n/2,
        i_mateltwise_desc->ldi, i_mateltwise_desc->ldo/2, 0, 0,
        (unsigned short)LIBXSMM_MELTW_FLAG_UNARY_NONE, (unsigned short)LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, LIBXSMM_MELTW_OPERATION_UNARY);
      libxsmm_mateltwise_kernel_config l_trans_config;
      libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_trans_config, mock_desc);
      libxsmm_generator_transform_norm_to_normt_32bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                          l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                          l_gp_reg_mask, l_gp_reg_mask_2, l_mask_reg_0, l_mask_reg_1, l_mask_reg_2, l_mask_reg_3,
                                                                          l_mask_reg_4, l_mask_reg_5, l_mask_reg_6,
                                                                          &l_trans_config, mock_desc );
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2_TO_VNNI2T ) {
      libxsmm_generator_transform_vnni2_to_vnni2t_16bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                          l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                          l_gp_reg_mask, l_gp_reg_mask_2, l_mask_reg_0, l_mask_reg_1, l_mask_reg_2,
                                                                          l_mask_reg_3, l_mask_reg_4, l_mask_reg_5, l_mask_reg_6,
                                                                          i_micro_kernel_config, i_mateltwise_desc );
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2 ) {
      libxsmm_generator_transform_norm_to_vnni2_16bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                         l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1,
                                                                         i_micro_kernel_config, i_mateltwise_desc, 0 );
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD ) {
      libxsmm_generator_transform_norm_to_vnni2_16bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                         l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1,
                                                                         i_micro_kernel_config, i_mateltwise_desc, 1 );
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4 ) {
      libxsmm_generator_transform_norm_to_vnni4_16bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                         l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1,
                                                                         i_micro_kernel_config, i_mateltwise_desc, 0 );
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4_PAD ) {
      libxsmm_generator_transform_norm_to_vnni4_16bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                         l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1,
                                                                         i_micro_kernel_config, i_mateltwise_desc, 1 );
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI4T ) {
      libxsmm_generator_transform_vnni4_to_vnni4t_16bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                            l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                            l_gp_reg_mask, l_mask_reg_0, i_micro_kernel_config, i_mateltwise_desc );
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8 ) {
      libxsmm_generator_transform_norm_to_vnni8_16bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                         l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1,
                                                                         i_micro_kernel_config, i_mateltwise_desc, 0 );
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8_PAD ) {
      libxsmm_generator_transform_norm_to_vnni8_16bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                         l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1,
                                                                         i_micro_kernel_config, i_mateltwise_desc, 1 );
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI8_TO_VNNI8T ) {
      libxsmm_generator_transform_vnni8_to_vnni8t_16bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                            l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                            l_gp_reg_mask, l_mask_reg_0, i_micro_kernel_config, i_mateltwise_desc );
    } else if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD2)  ||
                (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD2)  ||
                (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2)    ) {
      libxsmm_generator_transform_norm_padnm_mod2_16bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                            l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                            l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1,
                                                                            i_micro_kernel_config, i_mateltwise_desc );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else if ( ( LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
                LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) ||
              ( LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
                LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) ||
              ( LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
                LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) ) {
    if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT ) {
      libxsmm_generator_transform_norm_to_normt_08bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                          l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                          l_gp_reg_mask, l_gp_reg_mask_2,
                                                                          l_mask_reg_0, l_mask_reg_1, l_mask_reg_2, l_mask_reg_3,
                                                                          l_mask_reg_4, l_mask_reg_5, l_mask_reg_6,
                                                                          i_micro_kernel_config, i_mateltwise_desc );
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI4T ) {
      libxsmm_generator_transform_vnni4_to_vnni4t_08bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                            l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                            l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1, l_mask_reg_2,
                                                                            l_mask_reg_3, l_mask_reg_4, l_mask_reg_5,
                                                                            i_micro_kernel_config, i_mateltwise_desc );
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_NORM ) {
      libxsmm_generator_transform_vnni4_to_norm_08bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                          l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                          l_gp_reg_mask, l_mask_reg_0,
                                                                          i_micro_kernel_config, i_mateltwise_desc);
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI2 ) {
      libxsmm_generator_transform_vnni4_to_vnni2_08bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                          l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                          l_gp_reg_mask, l_mask_reg_0,
                                                                          i_micro_kernel_config, i_mateltwise_desc);
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4 ) {
      libxsmm_generator_transform_norm_to_vnni4_08bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                          l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                          l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1,
                                                                          i_micro_kernel_config, i_mateltwise_desc, 0 );
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4_PAD ) {
      libxsmm_generator_transform_norm_to_vnni4_08bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                          l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                          l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1,
                                                                          i_micro_kernel_config, i_mateltwise_desc, 1 );
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8 ) {
      libxsmm_generator_transform_norm_to_vnni8_08bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                          l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                          l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1,
                                                                          i_micro_kernel_config, i_mateltwise_desc, 0 );
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8_PAD ) {
      libxsmm_generator_transform_norm_to_vnni8_08bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                          l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                          l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1,
                                                                          i_micro_kernel_config, i_mateltwise_desc, 1 );
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI8_TO_VNNI8T ) {
      libxsmm_generator_transform_vnni8_to_vnni8t_08bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                            l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                            l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1, l_mask_reg_2,
                                                                            l_mask_reg_3, l_mask_reg_4, l_mask_reg_5,
                                                                            i_micro_kernel_config, i_mateltwise_desc );
    } else if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD4)  ||
                (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD4)  ||
                (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD4)    ) {
      libxsmm_generator_transform_norm_padnm_mod4_08bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                            l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                            l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1,
                                                                            i_micro_kernel_config, i_mateltwise_desc );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
}

