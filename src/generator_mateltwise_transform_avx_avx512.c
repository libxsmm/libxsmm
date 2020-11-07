/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
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
    unsigned int l_shuffle_instr = ( l_i % 2 == 0 ) ? i_even_instr : i_odd_instr;
    unsigned int in1 = i_in_idx[l_i] + i_vec_reg_src_start;
    unsigned int in0 = in1           + i_out_offset;
    unsigned int dst = l_i           + i_vec_reg_dst_start;

    libxsmm_x86_instruction_vec_compute_3reg_mask_imm8( io_generated_code, l_shuffle_instr, i_vector_name,
                                                        in0, in1, dst, 0, 0, LIBXSMM_X86_IMM_UNDEF );
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

    libxsmm_x86_instruction_vec_compute_3reg_mask_imm8( io_generated_code, i_shuffle_instr, i_vector_name,
                                                        in0, in1, dst, 0, 0, i_shuf_imm[l_i] );
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
    unsigned int in0 = i_in_idx[l_i] + i_vec_reg_src_start;
    unsigned int dst = l_i           + i_vec_reg_dst_start;

    libxsmm_x86_instruction_vec_compute_3reg_mask_imm8( io_generated_code, i_shuffle_instr, i_vector_name,
                                                        i_vec_reg_suffle_cntl, in0, dst, 0, 0, 0 );
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

    libxsmm_x86_instruction_vec_compute_3reg_mask_imm8( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64_LD, i_vector_name,
                                                        in0, LIBXSMM_X86_VEC_REG_UNDEF, dst, 0, 0, LIBXSMM_X86_IMM_UNDEF );
  }

  for ( l_i = 0; l_i < 8; ++l_i ) {
    unsigned int in0 = l_i + i_vec_reg_srcdst_start + 8;
    unsigned int dst = l_i + i_vec_reg_srcdst_start;

    libxsmm_x86_instruction_vec_compute_3reg_mask_imm8( io_generated_code, i_perm_instr, i_vector_name,
                                                        in0, LIBXSMM_X86_VEC_REG_UNDEF, dst, i_perm_mask[0], 0, i_perm_imm[0] );
  }

  for ( l_i = 8; l_i < 16; ++l_i ) {
    unsigned int in0 = ((l_i-8) + i_vec_reg_srcdst_start + 16)%32;
    unsigned int dst = l_i + i_vec_reg_srcdst_start;

    libxsmm_x86_instruction_vec_compute_3reg_mask_imm8( io_generated_code, i_perm_instr, i_vector_name,
                                                        in0, LIBXSMM_X86_VEC_REG_UNDEF, dst, i_perm_mask[1], 0, i_perm_imm[1] );
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
void libxsmm_generator_transform_Xway_full_store_avx512( libxsmm_generated_code* io_generated_code,
                                                         const char              i_vector_name,
                                                         const unsigned int      i_gp_reg_out,
                                                         const unsigned int      i_vec_reg_src_start,
                                                         const unsigned int      i_ld,
                                                         const unsigned int      i_st_instr,
                                                         const unsigned int      i_ways ) {
  unsigned int l_i = 0;

  if ( (i_ways != 8) && (i_ways != 16) ) {
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
  if ( (i_mateltwise_desc->m % 32 == 0) && (i_mateltwise_desc->n % 16 == 0) ) {
    /* set the masks for the permute stage */
    l_mask = 0xcc;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     i_gp_reg_mask, l_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVB,
                                       i_gp_reg_mask, i_mask_reg_0, 0 );

    l_mask = 0x33;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     i_gp_reg_mask, l_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVB,
                                       i_gp_reg_mask, i_mask_reg_1, 0 );

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
      libxsmm_generator_transform_16way_permute_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                l_perm_mask, l_perm_imm, l_dst_start, LIBXSMM_X86_INSTR_VPERMQ_I );
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
  } else {
    /* input mask, scalar */
    l_mask = 0x1;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     i_gp_reg_mask, l_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD,
                                       i_gp_reg_mask, i_mask_reg_2, 0 );

    /* output mask, scalar */
    l_mask = 0x1;
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     i_gp_reg_mask, l_mask );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD,
                                       i_gp_reg_mask, i_mask_reg_3, 0 );

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
                                      'x', 0, i_mask_reg_2, 1, 0 );

    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_micro_kernel_config->vmove_instruction_out,
                                      i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                      'x', 0, i_mask_reg_3, 0, 1 );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, i_micro_kernel_config->datatype_size_in );

    /* close n loop */
    libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_n_loop, i_mateltwise_desc->n );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - (i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                     i_gp_reg_in, (i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) - (i_micro_kernel_config->datatype_size_in) );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, i_mateltwise_desc->m );
  }
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
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned long long l_mask = 0;
  unsigned int  l_shuffle_cntl[16] = { 0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a, 0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a,
                                       0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a, 0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a };

  /* optimized shuffle network for SIMD aligned sizes */
  if ( (i_mateltwise_desc->m % 16 == 0) && (i_mateltwise_desc->n % 16 == 0) ) {
#if 0
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
#endif
    /* load 8 registers */
    libxsmm_generator_transform_Xway_full_load_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                       i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                       i_micro_kernel_config->vmove_instruction_in, 8 );

#if 0
    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * 16 );
#endif
    /* first shuffle stage */
    {
      unsigned char l_in_idx[16] = { 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
      unsigned int  l_src_start = 0;
      unsigned int  l_dst_start = 8;
      unsigned int  l_shuffle_op = 31;
      libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char *) l_shuffle_cntl,
                                                          "vnni_to_vnnit_shufl_", i_micro_kernel_config->vector_name, l_shuffle_op);
      libxsmm_generator_transform_Xway_byteshuffle_network_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                   l_in_idx, l_shuffle_op, 0, 0, LIBXSMM_X86_INSTR_VSHUFB, 8 );
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
                                                         i_gp_reg_out, 8, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                         i_micro_kernel_config->vmove_instruction_out, 8 );

#if 0
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
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
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

  if ( LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
       LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) &&
       ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_NORMT) > 0) ) {
    libxsmm_generator_transform_norm_to_normt_16bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                        l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                        l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1, l_mask_reg_2, l_mask_reg_3,
                                                                        i_micro_kernel_config, i_mateltwise_desc );
  } else if ( LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
              LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) &&
              ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TRANSFORM_VNNI_TO_VNNIT) > 0) ) {
    libxsmm_generator_transform_vnni_to_vnnit_16bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                                        l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                        l_gp_reg_mask, l_mask_reg_0, l_mask_reg_1,
                                                                        i_micro_kernel_config, i_mateltwise_desc );
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
}

