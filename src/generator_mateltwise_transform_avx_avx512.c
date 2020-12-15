/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke, Barukh Ziv, Menachem Adelmanm (Intel Corp.)
******************************************************************************/
#include "generator_mateltwise_avx_avx512.h"
#include "generator_mateltwise_transform_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
void libxsmm_generator_transform_Xway_unpack_network_avx512( libxsmm_generated_code* io_generated_code,
                                                             const char              i_vector_name,
                                                             const unsigned char     i_in_idx[16],
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

  if ( (i_ways != 8) && (i_ways != 16) ) {
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
                                                                  const unsigned char     i_vec_reg_suffle_cntl,
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
void libxsmm_generator_transform_Xway_full_load_avx512( libxsmm_generated_code* io_generated_code,
                                                        const char              i_vector_name,
                                                        const unsigned int      i_gp_reg_in,
                                                        const unsigned int      i_vec_reg_dst_start,
                                                        const unsigned int      i_ld,
                                                        const unsigned int      i_ld_instr,
                                                        const unsigned int      i_ways ) {
  unsigned int l_i = 0;

  if ( (i_ways != 8) && (i_ways != 16) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  if ( i_vec_reg_dst_start % i_ways != 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  for ( l_i = 0 ; l_i < i_ways ; ++l_i ) {
    unsigned int l_dst = l_i + i_vec_reg_dst_start;

    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_ld_instr,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, l_i * i_ld,
                                      i_vector_name, l_dst, 0, 1, 0 );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_Xway_half_load_blend_avx512( libxsmm_generated_code* io_generated_code,
                                                              const char              i_vector_name,
                                                              const unsigned int      i_gp_reg_in,
                                                              const unsigned int      i_vec_reg_dst_start,
                                                              const unsigned int      i_ld,
                                                              const unsigned int      i_ld_idx[32],
                                                              const unsigned int      i_blend_mult,
                                                              const unsigned int      i_ld_instr,
                                                              const unsigned int      i_ways,
                                                              const unsigned int      i_mask_reg ) {
  unsigned int l_i = 0;
  unsigned int l_blend_offset = i_blend_mult * i_ld;

  /* supports only up to 32 registers */
  if (i_ways > 32) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  for ( l_i = 0 ; l_i < i_ways ; ++l_i ) {
    unsigned int l_dst = l_i + i_vec_reg_dst_start;
    unsigned int l_load_displ = i_ld * (i_ld_idx[l_i] / 2) + 32 * (i_ld_idx[l_i] % 2);

    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_ld_instr,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, l_load_displ,
                                      i_vector_name, l_dst, 0, 1, 0 );

    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_ld_instr,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, l_load_displ + l_blend_offset,
                                      i_vector_name, l_dst, i_mask_reg, 0, 0 );
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
                                                                 const unsigned int      i_mask_reg_0,
                                                                 const unsigned int      i_mask_reg_1,
                                                                 const unsigned int      i_mask_reg_2) {
  unsigned int l_i = 0;
  unsigned int l_stride_offset = i_ways * i_ld;

  /* supports only up to 32 registers */
  if (i_ways > 32) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  for ( l_i = 0 ; l_i < i_ways ; ++l_i ) {
    unsigned int l_dst = l_i + i_vec_reg_dst_start;

    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_ld_instr,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, l_i*i_ld,
                                      i_vector_name, l_dst, 0, 1, 0 );

    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_ld_instr,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, l_i*i_ld + l_stride_offset,
                                      i_vector_name, l_dst, i_mask_reg_0, 0, 0 );

    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_ld_instr,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, l_i*i_ld + l_stride_offset*2,
                                      i_vector_name, l_dst, i_mask_reg_1, 0, 0 );

    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_ld_instr,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, l_i*i_ld + l_stride_offset*3,
                                      i_vector_name, l_dst, i_mask_reg_2, 0, 0 );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_Xway_full_store_avx512( libxsmm_generated_code* io_generated_code,
                                                         const char              i_vector_name,
                                                         const unsigned int      i_gp_reg_out,
                                                         const unsigned int      i_vec_reg_src_start,
                                                         const unsigned int      i_ld,
                                                         const unsigned int      i_st_instr,
                                                         const unsigned int      i_ways ) {
  unsigned int l_i = 0;

  if ( (i_ways != 2) && (i_ways != 4) && (i_ways != 8) && (i_ways != 16) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  if ( i_vec_reg_src_start % i_ways != 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  for ( l_i = 0 ; l_i < i_ways ; ++l_i ) {
    unsigned int l_src = l_i + i_vec_reg_src_start;

    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_st_instr,
                                      i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, l_i * i_ld,
                                      i_vector_name, l_src, 0, 1, 1 );
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
void libxsmm_generator_transform_norm_to_normt_mbit_scalar_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
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
  unsigned long long l_mask = 0;

  /* input mask, scalar */
  l_mask = 0x1;
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

  /* m loop header */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_m_loop, 1 );

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
                                   i_gp_reg_in, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in );

  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, i_micro_kernel_config->datatype_size_out );

  /* close n loop */
  libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                              i_gp_reg_n_loop, i_mateltwise_desc->n );

  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, (i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - (i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->n) );

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                   i_gp_reg_in, (i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) - (i_micro_kernel_config->datatype_size_in) );

  /* close m loop */
  libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                              i_gp_reg_m_loop, i_mateltwise_desc->m );
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
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int l_ldi = i_mateltwise_desc->ldi;
  unsigned int l_ldo = i_mateltwise_desc->ldo;
  unsigned long long l_mask = 0;

  LIBXSMM_UNUSED( i_mask_reg_1 );

  /* optimized shuffle network for SIMD aligned sizes */
  if ( (i_mateltwise_desc->m % 8 == 0) && (i_mateltwise_desc->n % 8 == 0) ) {
    const unsigned long long l_perm_lo[8] = { 0, 1, 4, 5,  8,  9, 12, 13 };
    const unsigned long long l_perm_hi[8] = { 2, 3, 6, 7, 10, 11, 14, 15 };
    l_mask = 0xf0;

    /* set the masks for the permute stage */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     i_gp_reg_mask, l_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVB_GPR_LD,
                                       i_gp_reg_mask, i_mask_reg_0 );

    /* load permute vectors to zmm31 and zmm30 */
    libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (unsigned char*)l_perm_lo, "i64_perm_lo",
                                                        i_micro_kernel_config->vector_name, 31 );
    libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (unsigned char*)l_perm_hi, "i64_perm_hi",
                                                        i_micro_kernel_config->vector_name, 30 );

    /* open m loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_m_loop, 8 );

    /* open n loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_n_loop, 8 );

    /* load 8 registes which shuffle at 256bit granularity */
    {
        const unsigned int ld_idx[32] = { 0, 1, 2, 3, 8, 9, 10, 11 };
        libxsmm_generator_transform_Xway_half_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                 i_gp_reg_in, 0, l_ldi * i_micro_kernel_config->datatype_size_in,
                                                                 ld_idx, 2, LIBXSMM_X86_INSTR_VBROADCASTI64X4, 8, i_mask_reg_0 );
    }

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * 8 );

    /* 2nd stage: unpack network */
    {
      unsigned char l_in_idx[16] = { 0x0, 0x0, 0x1, 0x1, 0x4, 0x4, 0x5, 0x5, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
      unsigned int  l_src_start = 0;
      unsigned int  l_dst_start = 8;
      libxsmm_generator_transform_Xway_unpack_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              l_in_idx, l_src_start, l_dst_start, 2,
                                                              LIBXSMM_X86_INSTR_VPUNPCKLQDQ, LIBXSMM_X86_INSTR_VPUNPCKHQDQ, 8 );
    }

    /* 3rd stage: variable permute network */
    libxsmm_generator_transform_08way_permutevar_network_avx512( io_generated_code, i_micro_kernel_config->vector_name, 31, 30,
                                                                 8, LIBXSMM_X86_INSTR_VPERMT2Q );

    /* storing 8 registers */
    libxsmm_generator_transform_Xway_full_store_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                        i_gp_reg_out, 0, l_ldo * i_micro_kernel_config->datatype_size_out,
                                                        i_micro_kernel_config->vmove_instruction_out, 8 );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, i_micro_kernel_config->datatype_size_out * 8 );

    /* close n footer */
    libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_n_loop, i_mateltwise_desc->n );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (8 * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - (i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->n) );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                     i_gp_reg_in, (l_ldi * i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) - (8 * i_micro_kernel_config->datatype_size_in) );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, i_mateltwise_desc->m );
  } else {
    libxsmm_generator_transform_norm_to_normt_mbit_scalar_avx512_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_in, i_gp_reg_out,
                                                                              i_gp_reg_m_loop, i_gp_reg_n_loop, i_gp_reg_mask, i_mask_reg_2, i_mask_reg_3,
                                                                              i_micro_kernel_config, i_mateltwise_desc );
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
                                                                         const unsigned int                      i_mask_reg_0,
                                                                         const unsigned int                      i_mask_reg_1,
                                                                         const unsigned int                      i_mask_reg_2,
                                                                         const unsigned int                      i_mask_reg_3,
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {

  unsigned long long l_mask = 0;

  /* optimized shuffle network for SIMD aligned sizes */
  /* codepath optimized for SPR */
  if ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) && (i_mateltwise_desc->m % 16 == 0) && (i_mateltwise_desc->n % 4 == 0) ) {
    /* set the masks for the load+blend stage */
    l_mask = 0x0c;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     i_gp_reg_mask, l_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVB_GPR_LD,
                                       i_gp_reg_mask, i_mask_reg_0 );

    l_mask = 0x30;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     i_gp_reg_mask, l_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVB_GPR_LD,
                                       i_gp_reg_mask, i_mask_reg_1 );

    l_mask = 0xc0;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     i_gp_reg_mask, l_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVB_GPR_LD,
                                       i_gp_reg_mask, i_mask_reg_2 );

    /* open m loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_m_loop, 4 );

    /* open n loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_n_loop, 16 );

    /* load 4 registers with four quarter rows */
    libxsmm_generator_transform_Xway_quarter_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                                LIBXSMM_X86_INSTR_VBROADCASTI64X2, 4, i_mask_reg_0, i_mask_reg_1, i_mask_reg_2 );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * 16 );

    /* first shuffle stage */
    {
        unsigned char l_in_idx[4] = { 0x0, 0x0, 0x2, 0x2};
        unsigned int  l_src_start = 0;
        unsigned int  l_dst_start = 4;
        libxsmm_generator_transform_Xway_unpack_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                l_in_idx, l_src_start, l_dst_start, 1,
                                                                LIBXSMM_X86_INSTR_VPUNPCKLDQ, LIBXSMM_X86_INSTR_VPUNPCKHDQ, 4 );
    }

    /* second shuffle stage */
    {
        unsigned char l_in_idx[8] = { 0x0, 0x0, 0x1, 0x1};
        unsigned int  l_src_start = 4;
        unsigned int  l_dst_start = 0;
        libxsmm_generator_transform_Xway_unpack_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                l_in_idx, l_src_start, l_dst_start, 2,
                                                                LIBXSMM_X86_INSTR_VPUNPCKLQDQ, LIBXSMM_X86_INSTR_VPUNPCKHQDQ, 4 );
    }

    /* storing 4 registers */
    libxsmm_generator_transform_Xway_full_store_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                        i_gp_reg_out, 0, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                        i_micro_kernel_config->vmove_instruction_out, 4 );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, i_micro_kernel_config->datatype_size_in * 16 );

    /* close n footer */
    libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_n_loop, i_mateltwise_desc->n );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (4 * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - (i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                     i_gp_reg_in, (i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) - (4 * i_micro_kernel_config->datatype_size_in) );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, i_mateltwise_desc->m );

  /* codepath optimized for CLX */
  } else if ( (i_mateltwise_desc->m % 16 == 0) && (i_mateltwise_desc->n % 8 == 0) ) {

    /* set the masks for the load+blend stage */
    unsigned long long l_mask = 0xf0;

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

    /* open m loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_m_loop, 8 );

    /* open n loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_n_loop, 16 );

    /* load 8 registers with two half rows */
    {
        const unsigned int ld_idx[32] = { 0, 2, 4, 6, 8, 10, 12, 14 };
        libxsmm_generator_transform_Xway_half_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                 i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                                 ld_idx, 8, LIBXSMM_X86_INSTR_VBROADCASTI64X4, 8, i_mask_reg_0 );
    }

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * 16 );

    /* first shuffle stage */
    {
        unsigned char l_in_idx[8] = { 0x0, 0x0, 0x2, 0x2, 0x4, 0x4, 0x6, 0x6};
        unsigned int  l_src_start = 0;
        unsigned int  l_dst_start = 8;
        libxsmm_generator_transform_Xway_unpack_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                l_in_idx, l_src_start, l_dst_start, 1,
                                                                LIBXSMM_X86_INSTR_VPUNPCKLDQ, LIBXSMM_X86_INSTR_VPUNPCKHDQ, 8 );
    }

    /* second shuffle stage */
    {
        unsigned char l_in_idx[8] = { 0x0, 0x0, 0x1, 0x1, 0x4, 0x4, 0x5, 0x5};
        unsigned int  l_src_start = 8;
        unsigned int  l_dst_start = 0;
        libxsmm_generator_transform_Xway_unpack_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                l_in_idx, l_src_start, l_dst_start, 2,
                                                                LIBXSMM_X86_INSTR_VPUNPCKLQDQ, LIBXSMM_X86_INSTR_VPUNPCKHQDQ, 8 );
    }

    /* 3rd stage: variable permute network */
    {
        unsigned int  l_srcdst_start = 0;
        unsigned char l_perm_imm[2] = { 0x44, 0xee };
        unsigned char l_perm_mask[2]; l_perm_mask[0] = (unsigned char)i_mask_reg_1; l_perm_mask[1] = (unsigned char)i_mask_reg_2;

        libxsmm_generator_transform_Xway_permute_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                 l_perm_mask, l_perm_imm, l_srcdst_start, LIBXSMM_X86_INSTR_VPERMQ_I, 8 );
    }

    /* storing 8 registers */
    libxsmm_generator_transform_Xway_full_store_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                        i_gp_reg_out, 0, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                        i_micro_kernel_config->vmove_instruction_out, 8 );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, i_micro_kernel_config->datatype_size_in * 16 );

    /* close n footer */
    libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_n_loop, i_mateltwise_desc->n );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (8 * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - (i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                     i_gp_reg_in, (i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) - (8 * i_micro_kernel_config->datatype_size_in) );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, i_mateltwise_desc->m );

  } else {

    libxsmm_generator_transform_norm_to_normt_mbit_scalar_avx512_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_in, i_gp_reg_out,
                                                                              i_gp_reg_m_loop, i_gp_reg_n_loop, i_gp_reg_mask, i_mask_reg_2, i_mask_reg_3,
                                                                              i_micro_kernel_config, i_mateltwise_desc );

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
                                                                         const unsigned int                      i_mask_reg_0,
                                                                         const unsigned int                      i_mask_reg_1,
                                                                         const unsigned int                      i_mask_reg_2,
                                                                         const unsigned int                      i_mask_reg_3,
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned long long l_mask = 0;

  /* optimized shuffle network for SIMD aligned sizes */
#if 1
  if ( (i_mateltwise_desc->m % 32 == 0) && (i_mateltwise_desc->n % 8 == 0) ) {
    /* set the masks for the load+blend stage */
    l_mask = 0x0c;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     i_gp_reg_mask, l_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVB_GPR_LD,
                                       i_gp_reg_mask, i_mask_reg_0 );

    l_mask = 0x30;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     i_gp_reg_mask, l_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVB_GPR_LD,
                                       i_gp_reg_mask, i_mask_reg_1 );
    l_mask = 0xc0;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     i_gp_reg_mask, l_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVB_GPR_LD,
                                       i_gp_reg_mask, i_mask_reg_2 );

    /* open m loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_m_loop, 8 );

    /* open n loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_n_loop, 32 );

    /* load 8 registers with four quarter rows */
    libxsmm_generator_transform_Xway_quarter_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                       i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                       LIBXSMM_X86_INSTR_VBROADCASTI64X2, 8, i_mask_reg_0, i_mask_reg_1, i_mask_reg_2 );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * 32 );

    /* first shuffle stage */
    {
      unsigned char l_in_idx[8] = { 0x0, 0x0, 0x2, 0x2, 0x4, 0x4, 0x6, 0x6};
      unsigned int  l_src_start = 0;
      unsigned int  l_dst_start = 8;
      libxsmm_generator_transform_Xway_unpack_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              l_in_idx, l_src_start, l_dst_start, 1,
                                                              LIBXSMM_X86_INSTR_VPUNPCKLWD, LIBXSMM_X86_INSTR_VPUNPCKHWD, 8 );
    }

    /* second shuffle stage */
    {
      unsigned char l_in_idx[8] = { 0x0, 0x0, 0x1, 0x1, 0x4, 0x4, 0x5, 0x5};
      unsigned int  l_src_start = 8;
      unsigned int  l_dst_start = 0;
      libxsmm_generator_transform_Xway_unpack_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              l_in_idx, l_src_start, l_dst_start, 2,
                                                              LIBXSMM_X86_INSTR_VPUNPCKLDQ, LIBXSMM_X86_INSTR_VPUNPCKHDQ, 8 );
    }

    /* third shuffle stage */
    {
      unsigned char l_in_idx[8] = { 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3};
      unsigned int  l_src_start = 0;
      unsigned int  l_dst_start = 8;
      libxsmm_generator_transform_Xway_unpack_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              l_in_idx, l_src_start, l_dst_start, 4,
                                                              LIBXSMM_X86_INSTR_VPUNPCKLQDQ, LIBXSMM_X86_INSTR_VPUNPCKHQDQ, 8 );
    }

    /* storing 8 registers */
    libxsmm_generator_transform_Xway_full_store_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                        i_gp_reg_out, 8, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                        i_micro_kernel_config->vmove_instruction_out, 8 );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, i_micro_kernel_config->datatype_size_in * 32 );

    /* close n footer */
    libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_n_loop, i_mateltwise_desc->n );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (8 * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - (i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                     i_gp_reg_in, (i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) - (8 * i_micro_kernel_config->datatype_size_in) );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, i_mateltwise_desc->m );
#else
  if ( (i_mateltwise_desc->m % 32 == 0) && (i_mateltwise_desc->n % 16 == 0) ) {
    /* set the masks for the permute stage */
    l_mask = 0xcc;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     i_gp_reg_mask, l_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVB_GPR_LD,
                                       i_gp_reg_mask, i_mask_reg_0 );

    l_mask = 0x33;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     i_gp_reg_mask, l_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVB_GPR_LD,
                                       i_gp_reg_mask, i_mask_reg_1 );

    /* open m loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_m_loop, 32 );

    /* open n loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_n_loop, 16 );

    /* load 16 registers */
    libxsmm_generator_transform_Xway_full_load_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                       i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                       i_micro_kernel_config->vmove_instruction_in, 16 );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * 16 );

    /* first shuffle stage */
    {
      unsigned char l_in_idx[16] = { 0x0, 0x0, 0x2, 0x2, 0x4, 0x4, 0x6, 0x6, 0x8, 0x8, 0xa, 0xa, 0xc, 0xc, 0xe, 0xe };
      unsigned int  l_src_start = 0;
      unsigned int  l_dst_start = 16;
      libxsmm_generator_transform_Xway_unpack_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              l_in_idx, l_src_start, l_dst_start, 1,
                                                              LIBXSMM_X86_INSTR_VPUNPCKLWD, LIBXSMM_X86_INSTR_VPUNPCKHWD, 16 );
    }

    /* second shuffle stage */
    {
      unsigned char l_in_idx[16] = { 0x0, 0x0, 0x1, 0x1, 0x4, 0x4, 0x5, 0x5, 0x8, 0x8, 0x9, 0x9, 0xc, 0xc, 0xd, 0xd };
      unsigned int  l_src_start = 16;
      unsigned int  l_dst_start = 0;
      libxsmm_generator_transform_Xway_unpack_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              l_in_idx, l_src_start, l_dst_start, 2,
                                                              LIBXSMM_X86_INSTR_VPUNPCKLDQ, LIBXSMM_X86_INSTR_VPUNPCKHDQ, 16 );
    }

    /* third shuffle stage */
    {
      unsigned char l_in_idx[16] = { 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x8, 0x8, 0x9, 0x9, 0xa, 0xa, 0xb, 0xb };
      unsigned int  l_src_start = 0;
      unsigned int  l_dst_start = 16;
      libxsmm_generator_transform_Xway_unpack_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              l_in_idx, l_src_start, l_dst_start, 4,
                                                              LIBXSMM_X86_INSTR_VPUNPCKLQDQ, LIBXSMM_X86_INSTR_VPUNPCKHQDQ, 16 );
    }

    /* fourth shuffle stage */
    {
      unsigned char l_in_idx[16] = { 0x0, 0x2, 0x4, 0x6, 0x0, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe, 0x8, 0xa, 0xc, 0xe };
      unsigned char l_shuf_imm[16] = { 0x88, 0x88, 0x88, 0x88, 0xdd, 0xdd, 0xdd, 0xdd, 0x88, 0x88, 0x88, 0x88, 0xdd, 0xdd, 0xdd, 0xdd };
      unsigned int  l_src_start = 16;
      unsigned int  l_dst_start = 0;
      libxsmm_generator_transform_Xway_shuffle_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                               l_in_idx, l_shuf_imm, l_src_start, l_dst_start, 1,
                                                               LIBXSMM_X86_INSTR_VSHUFI32X4, 16 );
    }

    /* fifth shuffle stage */
    {
      unsigned int  l_dst_start = 0;
      unsigned char l_perm_imm[2] = { 0x40, 0x0e };
      unsigned char l_perm_mask[2]; l_perm_mask[0] = (unsigned char)i_mask_reg_0; l_perm_mask[1] = (unsigned char)i_mask_reg_1;
      libxsmm_generator_transform_Xway_permute_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                               l_perm_mask, l_perm_imm, l_dst_start, LIBXSMM_X86_INSTR_VPERMQ_I, 16 );
    }

    /* storing 32x 32byte */
    libxsmm_generator_transform_32way_half_store_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                         i_gp_reg_out, 0, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                         i_micro_kernel_config->vmove_instruction_out );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, i_micro_kernel_config->datatype_size_in * 16 );

    /* close n footer */
    libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_n_loop, i_mateltwise_desc->n );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (32 * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - (i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                     i_gp_reg_in, (i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) - (32 * i_micro_kernel_config->datatype_size_in) );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, i_mateltwise_desc->m );
#endif
  } else {
    libxsmm_generator_transform_norm_to_normt_mbit_scalar_avx512_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_in, i_gp_reg_out,
                                                                              i_gp_reg_m_loop, i_gp_reg_n_loop, i_gp_reg_mask, i_mask_reg_2, i_mask_reg_3,
                                                                              i_micro_kernel_config, i_mateltwise_desc );
  }
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
                                                                         const unsigned int                      i_mask_reg_0,
                                                                         const unsigned int                      i_mask_reg_1,
                                                                         const unsigned int                      i_mask_reg_2,
                                                                         const unsigned int                      i_mask_reg_3,
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  LIBXSMM_UNUSED( i_mask_reg_0 );
  LIBXSMM_UNUSED( i_mask_reg_1 );

#if 0
  /* optimized shuffle network for SIMD aligned sizes */
  if ( (i_mateltwise_desc->m % 8 == 0) && (i_mateltwise_desc->n % 8 == 0) ) {
  } else {
#endif
    libxsmm_generator_transform_norm_to_normt_mbit_scalar_avx512_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_in, i_gp_reg_out,
                                                                              i_gp_reg_m_loop, i_gp_reg_n_loop, i_gp_reg_mask, i_mask_reg_2, i_mask_reg_3,
                                                                              i_micro_kernel_config, i_mateltwise_desc );
#if 0
  }
#endif
}

/*
 * Calling convention, this kernel assumes that all GPRs with are not in the argument list
 * and all xmm/ymm/zmm/tmm are caller save
 *
 * TODO; stack local variables....
 * */
LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni_to_vnnit_16bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                         libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                         const unsigned int                      i_gp_reg_in,
                                                                         const unsigned int                      i_gp_reg_out,
                                                                         const unsigned int                      i_gp_reg_m_loop,
                                                                         const unsigned int                      i_gp_reg_n_loop,
                                                                         const unsigned int                      i_gp_reg_mask,
                                                                         const unsigned int                      i_mask_reg_0,
                                                                         const unsigned int                      i_mask_reg_1,
                                                                         const unsigned int                      i_mask_reg_2,
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int l_ldi = i_mateltwise_desc->ldi*2;
  unsigned int l_ldo = i_mateltwise_desc->ldo*2;

  /* optimized shuffle network for SIMD aligned sizes */
  #if 1
  /* codepath optimized for SPR */
  if ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) && (i_mateltwise_desc->m % 4 == 0) && (i_mateltwise_desc->n % 16 == 0) ) {
    /* byte shuffle operand */
    unsigned int  l_shuffle_cntl[16] = { 0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a, 0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a,
                                         0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a, 0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a };
    unsigned int  l_shuffle_op = 31;
    /* set the masks for the load+blend stage */
    unsigned long long l_mask = 0x0c;

    libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char *) l_shuffle_cntl,
                                                        "vnni_to_vnnit_shufl_", i_micro_kernel_config->vector_name, l_shuffle_op);

    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     i_gp_reg_mask, l_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVB_GPR_LD,
                                       i_gp_reg_mask, i_mask_reg_0 );

    l_mask = 0x30;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     i_gp_reg_mask, l_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVB_GPR_LD,
                                       i_gp_reg_mask, i_mask_reg_1 );
    l_mask = 0xc0;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     i_gp_reg_mask, l_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVB_GPR_LD,
                                       i_gp_reg_mask, i_mask_reg_2 );

    /* open m loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_m_loop, 4 );

    /* open n loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_n_loop, 16 );

    /* load 2 registers with four quarter rows */
    libxsmm_generator_transform_Xway_quarter_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                i_gp_reg_in, 0, l_ldi * i_micro_kernel_config->datatype_size_in,
                                                                LIBXSMM_X86_INSTR_VBROADCASTI64X2, 2, i_mask_reg_0, i_mask_reg_1, i_mask_reg_2 );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, l_ldi * i_micro_kernel_config->datatype_size_in * 8 );

    /* first shuffle stage */
    {
      unsigned char l_in_idx[16] = { 0x0, 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
      unsigned int  l_src_start = 0;
      unsigned int  l_dst_start = 0;
      libxsmm_generator_transform_Xway_byteshuffle_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                   l_in_idx, l_shuffle_op, l_src_start, l_dst_start, LIBXSMM_X86_INSTR_VPSHUFB, 2 );
    }

    /* second shuffle stage */
    {
      unsigned char l_in_idx[16] = { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
      unsigned int  l_src_start = 0;
      unsigned int  l_dst_start = 2;
      libxsmm_generator_transform_Xway_unpack_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              l_in_idx, l_src_start, l_dst_start, 1,
                                                              LIBXSMM_X86_INSTR_VPUNPCKLQDQ, LIBXSMM_X86_INSTR_VPUNPCKHQDQ, 2 );
    }

    /* storing 2 registers */
    libxsmm_generator_transform_Xway_full_store_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                        i_gp_reg_out, 2, l_ldo * i_micro_kernel_config->datatype_size_out,
                                                        i_micro_kernel_config->vmove_instruction_out, 2 );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, i_micro_kernel_config->datatype_size_out * 32 );

    /* close n footer */
    libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_n_loop, i_mateltwise_desc->n );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (2 * l_ldo * i_micro_kernel_config->datatype_size_out) - (i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->n*2) );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                     i_gp_reg_in, (i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) - (8 * i_micro_kernel_config->datatype_size_in) );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, i_mateltwise_desc->m );

  }

  /* codepath optimized for CLX */
  else if ( (i_mateltwise_desc->m % 8 == 0) && (i_mateltwise_desc->n % 16 == 0) ) {
    /* byte shuffle operand */
    unsigned int  l_shuffle_cntl[16] = { 0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a, 0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a,
                                         0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a, 0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a };
    unsigned int  l_shuffle_op = 31;
    /* set the masks for the load+blend stage */
    unsigned long long l_mask = 0xf0;

    libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char *) l_shuffle_cntl,
                                                        "vnni_to_vnnit_shufl_", i_micro_kernel_config->vector_name, l_shuffle_op);

    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     i_gp_reg_mask, l_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVB_GPR_LD,
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

    /* open m loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_m_loop, 8 );

    /* open n loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_n_loop, 16 );

    /* load 4 registers with two half rows */
    {
        const unsigned int ld_idx[32] = { 0, 2, 4, 6 };
        libxsmm_generator_transform_Xway_half_load_blend_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                 i_gp_reg_in, 0, l_ldi * i_micro_kernel_config->datatype_size_in,
                                                                 ld_idx, 4, LIBXSMM_X86_INSTR_VBROADCASTI64X4, 4, i_mask_reg_0 );
    }

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, l_ldi * i_micro_kernel_config->datatype_size_in * 8 );

    /* first shuffle stage */
    {
      unsigned char l_in_idx[16] = { 0x0, 0x1, 0x2, 0x3, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
      unsigned int  l_src_start = 0;
      unsigned int  l_dst_start = 0;
      libxsmm_generator_transform_Xway_byteshuffle_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                   l_in_idx, l_shuffle_op, l_src_start, l_dst_start, LIBXSMM_X86_INSTR_VPSHUFB, 4 );
    }

    /* second shuffle stage */
    {
      unsigned char l_in_idx[16] = { 0x0, 0x0, 0x2, 0x2, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
      unsigned int  l_src_start = 0;
      unsigned int  l_dst_start = 4;
      libxsmm_generator_transform_Xway_unpack_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              l_in_idx, l_src_start, l_dst_start, 1,
                                                              LIBXSMM_X86_INSTR_VPUNPCKLQDQ, LIBXSMM_X86_INSTR_VPUNPCKHQDQ, 4 );
    }

    /* 3rd stage: variable permute network */
    {
      unsigned int  l_srcdst_start = 4;
      unsigned char l_perm_imm[2] = { 0x44, 0xee };
      unsigned char l_perm_mask[2]; l_perm_mask[0] = (unsigned char)i_mask_reg_1; l_perm_mask[1] = (unsigned char)i_mask_reg_2;

      libxsmm_generator_transform_Xway_permute_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                l_perm_mask, l_perm_imm, l_srcdst_start, LIBXSMM_X86_INSTR_VPERMQ_I, 4 );
    }

    /* storing 4 registers */
    libxsmm_generator_transform_Xway_full_store_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                        i_gp_reg_out, 4, l_ldo * i_micro_kernel_config->datatype_size_out,
                                                        i_micro_kernel_config->vmove_instruction_out, 4 );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, i_micro_kernel_config->datatype_size_out * 32 );

    /* close n footer */
    libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_n_loop, i_mateltwise_desc->n );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (4 * l_ldo * i_micro_kernel_config->datatype_size_out) - (i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->n*2) );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                     i_gp_reg_in, (i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) - (16 * i_micro_kernel_config->datatype_size_in) );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, i_mateltwise_desc->m );
  }

  #else
  if ( (i_mateltwise_desc->m % 16 == 0) && (i_mateltwise_desc->n % 16 == 0) ) {
    /* byte shuffle operand */
    unsigned int  l_shuffle_cntl[16] = { 0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a, 0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a,
                                         0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a, 0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a };
    unsigned int  l_shuffle_op = 31;
    libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char *) l_shuffle_cntl,
                                                        "vnni_to_vnnit_shufl_", i_micro_kernel_config->vector_name, l_shuffle_op);

    /* open m loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_m_loop, 16 );

    /* open n loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_n_loop, 16 );

    /* load 8 registers */
    libxsmm_generator_transform_Xway_full_load_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                       i_gp_reg_in, 0, l_ldi * i_micro_kernel_config->datatype_size_in,
                                                       i_micro_kernel_config->vmove_instruction_in, 8 );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, l_ldi * i_micro_kernel_config->datatype_size_in * 8 );

    /* first shuffle stage */
    {
      unsigned char l_in_idx[16] = { 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
      unsigned int  l_src_start = 0;
      unsigned int  l_dst_start = 0;
      libxsmm_generator_transform_Xway_byteshuffle_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                   l_in_idx, l_shuffle_op, l_src_start, l_dst_start, LIBXSMM_X86_INSTR_VPSHUFB, 8 );
    }

    /* second shuffle stage */
    {
      unsigned char l_in_idx[16] = { 0x0, 0x0, 0x2, 0x2, 0x4, 0x4, 0x6, 0x6, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
      unsigned int  l_src_start = 0;
      unsigned int  l_dst_start = 8;
      libxsmm_generator_transform_Xway_unpack_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                              l_in_idx, l_src_start, l_dst_start, 1,
                                                              LIBXSMM_X86_INSTR_VPUNPCKLQDQ, LIBXSMM_X86_INSTR_VPUNPCKHQDQ, 8 );
    }

    /* third shuffle stage */
    {
      unsigned char l_in_idx[16] = { 0x0, 0x1, 0x0, 0x1, 0x4, 0x5, 0x4, 0x5, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
      unsigned char l_shuf_imm[16] = { 0x88, 0x88, 0xdd, 0xdd, 0x88, 0x88, 0xdd, 0xdd, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
      unsigned int  l_src_start = 8;
      unsigned int  l_dst_start = 0;
      libxsmm_generator_transform_Xway_shuffle_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                               l_in_idx, l_shuf_imm, l_src_start, l_dst_start, 2,
                                                               LIBXSMM_X86_INSTR_VSHUFI32X4, 8 );
    }

    /* fourth shuffle stage */
    {
      unsigned char l_in_idx[16] = { 0x0, 0x1, 0x2, 0x3, 0x0, 0x1, 0x2, 0x3, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
      unsigned char l_shuf_imm[16] = { 0x88, 0x88, 0x88, 0x88, 0xdd, 0xdd, 0xdd, 0xdd, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
      unsigned int  l_src_start = 0;
      unsigned int  l_dst_start = 8;
      libxsmm_generator_transform_Xway_shuffle_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                               l_in_idx, l_shuf_imm, l_src_start, l_dst_start, 4,
                                                               LIBXSMM_X86_INSTR_VSHUFI32X4, 8 );
    }

    /* storing 8 registers */
    libxsmm_generator_transform_Xway_full_store_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                         i_gp_reg_out, 8, l_ldo * i_micro_kernel_config->datatype_size_out,
                                                         i_micro_kernel_config->vmove_instruction_out, 8 );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, i_micro_kernel_config->datatype_size_out * 32 );

    /* close n footer */
    libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_n_loop, i_mateltwise_desc->n );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (16 * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - (i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->n * 2) );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                     i_gp_reg_in, (l_ldi * i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n/2) - (32 * i_micro_kernel_config->datatype_size_in) );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, i_mateltwise_desc->m );
  }
  #endif
  else if ( (i_mateltwise_desc->m % 2 == 0) && (i_mateltwise_desc->n % 2 == 0) ) {
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
                                     i_gp_reg_in, i_micro_kernel_config->datatype_size_in * 4 );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, i_micro_kernel_config->datatype_size_out * l_ldo );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, i_mateltwise_desc->m );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, (l_ldi * i_micro_kernel_config->datatype_size_in) - (i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->m * 2) );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                     i_gp_reg_out, ( i_micro_kernel_config->datatype_size_out * l_ldo * i_mateltwise_desc->m/2 ) - (i_micro_kernel_config->datatype_size_out * 4)  );

    /* close n loop */
    libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_n_loop, i_mateltwise_desc->n );


  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
}

/*
 * Calling convention, this kernel assumes that all GPRs with are not in the argument list
 * and all xmm/ymm/zmm/tmm are caller save
 *
 * TODO; stack local variables....
 * */
LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni_16bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
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
  /* optimized shuffle network for SIMD aligned sizes */
  if ( (i_mateltwise_desc->m % 32 == 0) && (i_mateltwise_desc->n % 8 == 0) && (i_pad_vnni == 0) ) {
    unsigned int l_vnni_lo_reg = 31;
    unsigned int l_vnni_hi_reg = 30;
    short perm_table_vnni_lo[32] = {32, 0, 33, 1, 34, 2, 35, 3, 36, 4, 37, 5, 38, 6, 39, 7, 40, 8, 41, 9, 42, 10, 43, 11, 44, 12, 45, 13, 46, 14, 47, 15};
    short perm_table_vnni_hi[32] = {48, 16, 49, 17, 50, 18, 51, 19, 52, 20, 53, 21, 54, 22, 55, 23, 56, 24, 57, 25, 58, 26, 59, 27, 60, 28, 61, 29, 62, 30, 63, 31};
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) perm_table_vnni_lo, "perm_table_vnni_lo_", i_micro_kernel_config->vector_name, l_vnni_lo_reg);
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) perm_table_vnni_hi, "perm_table_vnni_hi_", i_micro_kernel_config->vector_name, l_vnni_hi_reg);


    /* open n loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_n_loop, 8 );

    /* open m loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_m_loop, 32 );

    /* load 2 registers */
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                      i_micro_kernel_config->vector_name, 0, 0, 1, 0 );
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                      i_micro_kernel_config->vector_name, 1, 0, 1, 0 );
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, 2*i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                      i_micro_kernel_config->vector_name, 2, 0, 1, 0 );
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, 3*i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                      i_micro_kernel_config->vector_name, 3, 0, 1, 0 );
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, 4*i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                      i_micro_kernel_config->vector_name, 4, 0, 1, 0 );
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, 5*i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                      i_micro_kernel_config->vector_name, 5, 0, 1, 0 );
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, 6*i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                      i_micro_kernel_config->vector_name, 6, 0, 1, 0 );
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                      i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, 7*i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                      i_micro_kernel_config->vector_name, 7, 0, 1, 0 );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, 32 * i_micro_kernel_config->datatype_size_in );

    /* create VNNI interleaved format */
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64_LD, i_micro_kernel_config->vector_name,
                                                            1, LIBXSMM_X86_VEC_REG_UNDEF, 8, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2W, i_micro_kernel_config->vector_name,
                                                            0, l_vnni_lo_reg, 1, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2W, i_micro_kernel_config->vector_name,
                                                            0, l_vnni_hi_reg, 8, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
    /* storing 2 registers */
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                      i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                      i_micro_kernel_config->vector_name, 1, 0, 1, 1 );
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                      i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, 32 * i_micro_kernel_config->datatype_size_out,
                                      i_micro_kernel_config->vector_name, 8, 0, 1, 1 );
    /* create VNNI interleaved format */
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64_LD, i_micro_kernel_config->vector_name,
                                                            3, LIBXSMM_X86_VEC_REG_UNDEF, 8, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2W, i_micro_kernel_config->vector_name,
                                                            2, l_vnni_lo_reg, 3, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2W, i_micro_kernel_config->vector_name,
                                                            2, l_vnni_hi_reg, 8, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
    /* storing 2 registers */
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                      i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, 2 * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                      i_micro_kernel_config->vector_name, 3, 0, 1, 1 );
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                      i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, (2 * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) + (32 * i_micro_kernel_config->datatype_size_out),
                                      i_micro_kernel_config->vector_name, 8, 0, 1, 1 );

    /* create VNNI interleaved format */
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64_LD, i_micro_kernel_config->vector_name,
                                                            5, LIBXSMM_X86_VEC_REG_UNDEF, 8, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2W, i_micro_kernel_config->vector_name,
                                                            4, l_vnni_lo_reg, 5, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2W, i_micro_kernel_config->vector_name,
                                                            4, l_vnni_hi_reg, 8, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
    /* storing 2 registers */
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                      i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, 4 * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                      i_micro_kernel_config->vector_name, 5, 0, 1, 1 );
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                      i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, (4 * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) + (32 * i_micro_kernel_config->datatype_size_out),
                                      i_micro_kernel_config->vector_name, 8, 0, 1, 1 );
    /* create VNNI interleaved format */
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64_LD, i_micro_kernel_config->vector_name,
                                                            7, LIBXSMM_X86_VEC_REG_UNDEF, 8, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2W, i_micro_kernel_config->vector_name,
                                                            6, l_vnni_lo_reg, 7, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERMT2W, i_micro_kernel_config->vector_name,
                                                            6, l_vnni_hi_reg, 8, 0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
    /* storing 2 registers */
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                      i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, 6 * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                      i_micro_kernel_config->vector_name, 7, 0, 1, 1 );
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                      i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, (6 * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) + (32 * i_micro_kernel_config->datatype_size_out),
                                      i_micro_kernel_config->vector_name, 8, 0, 1, 1 );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, 64 * i_micro_kernel_config->datatype_size_out );

    /* close m footer */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, i_mateltwise_desc->m );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, (8 * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in) - (i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->m) );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (8 * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - (i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->m * 2) );

    /* close n loop */
    libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_n_loop, i_mateltwise_desc->n );
  } else {
    /* input mask, scalar */
    unsigned long long l_mask = 0x1;

    if ( (i_pad_vnni == 0) && (i_mateltwise_desc->n % 2 == 1) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }

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

    if ( i_mateltwise_desc->n >= 2 ) {
      /* n loop header */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_n_loop, 2 );

      /* m loop header */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_m_loop, 1 );

      /* actual transpose */
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                        i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                        'x', 0, i_mask_reg_0, 1, 0 );

      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                        i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->ldi,
                                        'x', 1, i_mask_reg_0, 1, 0 );

      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                        i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                        'x', 0, i_mask_reg_1, 0, 1 );

      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                        i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out,
                                        'x', 1, i_mask_reg_1, 0, 1 );

      /* advance input pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_in, i_micro_kernel_config->datatype_size_in );

      /* advance output pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_out, i_micro_kernel_config->datatype_size_out * 2 );

      /* close m loop */
      libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                  i_gp_reg_m_loop, i_mateltwise_desc->m );

      /* advance output pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_in, (2 * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in) - (i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->m) );

      /* advance input pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_out, (2 * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - (i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->m * 2) );

      /* close n loop */
      libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                  i_gp_reg_n_loop, (i_mateltwise_desc->n/2)*2 );
    }

    if ( (i_mateltwise_desc->n % 2 == 1) && (i_pad_vnni == 1) ) {
      /* set zmm1 to 0 */
      libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, 'x',
                                                              1, 1, 1, 0, 0, 0, 0 );

      /* m loop header */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_m_loop, 1 );

      /* actual transpose */
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_in,
                                        i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                        'x', 0, i_mask_reg_0, 1, 0 );

      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                        i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                        'x', 0, i_mask_reg_1, 0, 1 );

      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                        i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out,
                                        'x', 1, i_mask_reg_1, 0, 1 );

      /* advance input pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_in, i_micro_kernel_config->datatype_size_in );

      /* advance output pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_out, i_micro_kernel_config->datatype_size_out * 2 );

      /* close m loop */
      libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                  i_gp_reg_m_loop, i_mateltwise_desc->m );
    }
  }
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
  unsigned int l_mask_reg_0 = 1;
  unsigned int l_mask_reg_1 = 2;
  unsigned int l_mask_reg_2 = 3;
  unsigned int l_mask_reg_3 = 4;

  /* load pointers from struct */
  libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_param_struct,
                                   LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                   l_gp_reg_in, 0 );

  libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_param_struct,
                                   LIBXSMM_X86_GP_REG_UNDEF, 0, 8,
                                   l_gp_reg_out, 0 );

  /* check leading dimnesions and sizes */
  if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_NORMT) > 0) ||
       ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TRANSFORM_VNNI_TO_VNNIT) > 0)    ) {
    if ( (i_mateltwise_desc->m > i_mateltwise_desc->ldi) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
      return;
    }
    if ( (i_mateltwise_desc->n > i_mateltwise_desc->ldo) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB );
      return;
    }
  } else if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_VNNI) > 0)     ||
              ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_VNNI_PAD) > 0)    ) {
    if ( (i_mateltwise_desc->m > i_mateltwise_desc->ldi) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
      return;
    }
    if ( (i_mateltwise_desc->m > i_mateltwise_desc->ldo) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB );
      return;
    }
  } else {
    /* should not happen */
  }

  if ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
       LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
    if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_NORMT) > 0 ) {
      libxsmm_generator_transform_norm_to_normt_64bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                          l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                          l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1, l_mask_reg_2, l_mask_reg_3,
                                                                          i_micro_kernel_config, i_mateltwise_desc );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else if ( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
              LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
    if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_NORMT) > 0 ) {
      libxsmm_generator_transform_norm_to_normt_32bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                          l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                          l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1, l_mask_reg_2, l_mask_reg_3,
                                                                          i_micro_kernel_config, i_mateltwise_desc );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else if ( LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
              LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
    if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_NORMT) > 0 ) {
      libxsmm_generator_transform_norm_to_normt_16bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                          l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                          l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1, l_mask_reg_2, l_mask_reg_3,
                                                                          i_micro_kernel_config, i_mateltwise_desc );
    } else if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TRANSFORM_VNNI_TO_VNNIT) > 0 ) {
      libxsmm_generator_transform_vnni_to_vnnit_16bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                          l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                          l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1, l_mask_reg_2,
                                                                          i_micro_kernel_config, i_mateltwise_desc );
    } else if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_VNNI) > 0 ) {
      libxsmm_generator_transform_norm_to_vnni_16bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                         l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1,
                                                                         i_micro_kernel_config, i_mateltwise_desc, 0 );
    } else if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_VNNI_PAD) > 0 ) {
      libxsmm_generator_transform_norm_to_vnni_16bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                         l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1,
                                                                         i_micro_kernel_config, i_mateltwise_desc, 1 );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else if ( LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
              LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
    if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_NORMT) > 0 ) {
      libxsmm_generator_transform_norm_to_normt_08bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                          l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                          l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1, l_mask_reg_2, l_mask_reg_3,
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

