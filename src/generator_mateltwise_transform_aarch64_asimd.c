/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Evangelos Georganas (Intel Corp.)
******************************************************************************/

#include "generator_mateltwise_transform_aarch64_asimd.h"
#include "generator_common_aarch64.h"
#include "generator_aarch64_instructions.h"
#include "generator_common.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
void libxsmm_generator_transform_Xway_unpack_network_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                                    const unsigned char*    i_in_idx,
                                                                    const unsigned char*    i_out_idx,
                                                                    const unsigned int      i_vec_reg_src_start,
                                                                    const unsigned int      i_vec_reg_dst_start,
                                                                    const unsigned int      i_in_offset,
                                                                    const unsigned int      i_even_instr,
                                                                    const unsigned int      i_odd_instr,
                                                                    const unsigned int      i_ways,
                                                                    const libxsmm_aarch64_asimd_tupletype i_tupletype ) {
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
    unsigned int in0 = i_in_idx[l_i]  + i_vec_reg_src_start;
    unsigned int in1 = in0            + i_in_offset;
    unsigned int dst = i_out_idx[l_i] + i_vec_reg_dst_start;

    libxsmm_aarch64_instruction_asimd_compute( io_generated_code, l_shuffle_instr,
                                               in0, in1, 0, dst, i_tupletype );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_64bit_4x4_shufflenetwork_aarch64_asimd( libxsmm_generated_code*                 io_generated_code,
                                                                                       const unsigned int                      i_gp_reg_in,
                                                                                       const unsigned int                      i_gp_reg_out,
                                                                                       const unsigned int                      i_gp_reg_scratch,
                                                                                       const unsigned int                      i_m_valid,
                                                                                       const unsigned int                      i_n_valid,
                                                                                       const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                       const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  LIBXSMM_UNUSED( i_m_valid );
  LIBXSMM_UNUSED( i_n_valid );

  /* load 8 vectors */
  libxsmm_generator_load_2dregblock_aarch64_asimd( io_generated_code, i_gp_reg_in, i_gp_reg_scratch, 2, 32, 4, 4, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in, 0 );

  /* shuffle */
  {
    unsigned char l_in_idx[16]  = { 0x0, 0x0, 0x4, 0x4, 0x1, 0x1, 0x5, 0x5, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
    unsigned char l_out_idx[16] = { 0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
    unsigned int  l_src_start = 24;
    unsigned int  l_dst_start = 16;
    libxsmm_generator_transform_Xway_unpack_network_aarch64_asimd( io_generated_code, l_in_idx, l_out_idx, l_src_start, l_dst_start, 2,
                                                                   LIBXSMM_AARCH64_INSTR_ASIMD_TRN1, LIBXSMM_AARCH64_INSTR_ASIMD_TRN2, 8, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
  }

  /* store 8 vecotrs */
  libxsmm_generator_store_2dregblock_aarch64_asimd( io_generated_code, i_gp_reg_out, i_gp_reg_scratch, 2, 24, 4, 4, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_32bit_4x4_shufflenetwork_aarch64_asimd( libxsmm_generated_code*                 io_generated_code,
                                                                                       const unsigned int                      i_gp_reg_in,
                                                                                       const unsigned int                      i_gp_reg_out,
                                                                                       const unsigned int                      i_gp_reg_scratch,
                                                                                       const unsigned int                      i_m_valid,
                                                                                       const unsigned int                      i_n_valid,
                                                                                       const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                       const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  LIBXSMM_UNUSED( i_m_valid );
  LIBXSMM_UNUSED( i_n_valid );

  /* load 8 vectors */
  libxsmm_generator_load_2dregblock_aarch64_asimd( io_generated_code, i_gp_reg_in, i_gp_reg_scratch, 4, 32, 4, 4, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in, 0 );

  /* shuffle 64 bit */
  {
    unsigned char l_in_idx[16]  = { 0x0, 0x0, 0x1, 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
    unsigned char l_out_idx[16] = { 0x0, 0x1, 0x2, 0x3, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
    unsigned int  l_src_start = 28;
    unsigned int  l_dst_start = 24;
    libxsmm_generator_transform_Xway_unpack_network_aarch64_asimd( io_generated_code, l_in_idx, l_out_idx, l_src_start, l_dst_start, 2,
                                                                   LIBXSMM_AARCH64_INSTR_ASIMD_TRN1, LIBXSMM_AARCH64_INSTR_ASIMD_TRN2, 4, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
  }

  /* shuffle 32 bit */
  {
    unsigned char l_in_idx[16]  = { 0x0, 0x0, 0x1, 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
    unsigned char l_out_idx[16] = { 0x0, 0x1, 0x2, 0x3, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
    unsigned int  l_src_start = 24;
    unsigned int  l_dst_start = 28;
    libxsmm_generator_transform_Xway_unpack_network_aarch64_asimd( io_generated_code, l_in_idx, l_out_idx, l_src_start, l_dst_start, 2,
                                                                   LIBXSMM_AARCH64_INSTR_ASIMD_TRN1, LIBXSMM_AARCH64_INSTR_ASIMD_TRN2, 4, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
  }

  /* store 8 vecotrs */
  libxsmm_generator_store_2dregblock_aarch64_asimd( io_generated_code, i_gp_reg_out, i_gp_reg_scratch, 4, 32, 4, 4, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_32bit_8x8_shufflenetwork_aarch64_asimd( libxsmm_generated_code*                 io_generated_code,
                                                                                       const unsigned int                      i_gp_reg_in,
                                                                                       const unsigned int                      i_gp_reg_out,
                                                                                       const unsigned int                      i_gp_reg_scratch,
                                                                                       const unsigned int                      i_m_valid,
                                                                                       const unsigned int                      i_n_valid,
                                                                                       const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                       const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  LIBXSMM_UNUSED( i_m_valid );
  LIBXSMM_UNUSED( i_n_valid );

  /* load 8 vectors */
  libxsmm_generator_load_2dregblock_aarch64_asimd( io_generated_code, i_gp_reg_in, i_gp_reg_scratch, 4, 32, 8, 8, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in, 0 );

  /* shuffle 64 bit */
  {
    unsigned char l_in_idx[16]  = { 0x0, 0x0, 0x2, 0x2, 0x8, 0x8, 0xa, 0xa, 0x1, 0x1, 0x3, 0x3, 0x9, 0x9, 0xb, 0xb };
    unsigned char l_out_idx[16] = { 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf };
    unsigned int  l_src_start = 16;
    unsigned int  l_dst_start = 0;
    libxsmm_generator_transform_Xway_unpack_network_aarch64_asimd( io_generated_code, l_in_idx, l_out_idx, l_src_start, l_dst_start, 4,
                                                                   LIBXSMM_AARCH64_INSTR_ASIMD_TRN1, LIBXSMM_AARCH64_INSTR_ASIMD_TRN2, 16, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
  }

  /* shuffle 32 bit */
  {
    unsigned char l_in_idx[16]  = { 0x0, 0x0, 0x1, 0x1, 0x4, 0x4, 0x5, 0x5, 0x8, 0x8, 0x9, 0x9, 0xc, 0xc, 0xd, 0xd };
    unsigned char l_out_idx[16] = { 0x0, 0x2, 0x4, 0x6, 0x1, 0x3, 0x5, 0x7, 0x8, 0xa, 0xc, 0xe, 0x9, 0xb, 0xd, 0xf };
    unsigned int  l_src_start = 0;
    unsigned int  l_dst_start = 16;
    libxsmm_generator_transform_Xway_unpack_network_aarch64_asimd( io_generated_code, l_in_idx, l_out_idx, l_src_start, l_dst_start, 2,
                                                                   LIBXSMM_AARCH64_INSTR_ASIMD_TRN1, LIBXSMM_AARCH64_INSTR_ASIMD_TRN2, 16, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
  }

  /* store 8 vecotrs */
  libxsmm_generator_store_2dregblock_aarch64_asimd( io_generated_code, i_gp_reg_out, i_gp_reg_scratch, 4, 32, 8, 8, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_mbit_scalar_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                      const unsigned int                      i_gp_reg_in,
                                                                                      const unsigned int                      i_gp_reg_out,
                                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                                      const unsigned int                      i_gp_reg_scratch,
                                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  libxsmm_aarch64_asimd_width l_load_instr_width;
  libxsmm_aarch64_asimd_width l_store_instr_width;

  /* select load and store instructions */
  if ( i_micro_kernel_config->datatype_size_in == 8 ) {
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_D;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_D;
  } else if ( i_micro_kernel_config->datatype_size_in == 4 ) {
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_S;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_S;
  } else if ( i_micro_kernel_config->datatype_size_in == 2 ) {
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_H;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_H;
  } else if ( i_micro_kernel_config->datatype_size_in == 1 ) {
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_B;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_B;
  } else {
    /* should not happen */
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_B;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_B;
  }

  /* m loop header */
  libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, i_mateltwise_desc->m );

  /* n loop header */
  libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, i_mateltwise_desc->n );

  /* actual transpose */
  libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                          i_gp_reg_in, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 0, l_load_instr_width );

  libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF,
                                          i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 0, l_store_instr_width );

  /* advance input pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in,
                                                 i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in );

  /* advance output pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out,
                                                 i_micro_kernel_config->datatype_size_out );

  /* close n loop */
  libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, 1 );

  /* advance output pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out,
                                                 (i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - (i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->n) );

  /* advance input pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in,
                                                 (i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) - (i_micro_kernel_config->datatype_size_in) );

  /* close m loop */
  libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, 1 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni4_mbit_scalar_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                      const unsigned int                      i_gp_reg_in,
                                                                                      const unsigned int                      i_gp_reg_out,
                                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                                      const unsigned int                      i_gp_reg_scratch,
                                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                                      const unsigned int                      i_pad_vnni ) {
  libxsmm_aarch64_asimd_width l_load_instr_width;
  libxsmm_aarch64_asimd_width l_store_instr_width;
  unsigned int l_remainder_n = i_mateltwise_desc->n % 4, l_r = 0;

  /* select load and store instructions */
  if ( i_micro_kernel_config->datatype_size_in == 8 ) {
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_D;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_D;
  } else if ( i_micro_kernel_config->datatype_size_in == 4 ) {
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_S;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_S;
  } else if ( i_micro_kernel_config->datatype_size_in == 2 ) {
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_H;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_H;
  } else if ( i_micro_kernel_config->datatype_size_in == 1 ) {
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_B;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_B;
  } else {
    /* should not happen */
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_B;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_B;
  }

  if ( (i_pad_vnni == 0) && (i_mateltwise_desc->n % 4 != 0) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  if ( i_mateltwise_desc->n >= 4 ) {
    /* n loop header */
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, (i_mateltwise_desc->n/4)*4 );

    /* m loop header */
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, i_mateltwise_desc->m );

    /* actual transpose */
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                            i_gp_reg_in, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 0, l_load_instr_width );

    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                            i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->datatype_size_out, 0, l_store_instr_width );

    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in,
                                                   i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->ldi );

    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                            i_gp_reg_in, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 0, l_load_instr_width );

    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                            i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->datatype_size_out, 0, l_store_instr_width );

    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in,
                                                   i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->ldi );

    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                            i_gp_reg_in, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 0, l_load_instr_width );

    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                            i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->datatype_size_out, 0, l_store_instr_width );

    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in,
                                                   i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->ldi );

    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                            i_gp_reg_in, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 0, l_load_instr_width );

    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                            i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->datatype_size_out, 0, l_store_instr_width );


    /* advance input pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in,
                                                   (3 * i_mateltwise_desc->ldi - 1) * i_micro_kernel_config->datatype_size_in );

    /* close m loop */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, 1 );

    /* advance output pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in,
                                                   (4 * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in) - (i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->m) );

    /* advance input pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out,
                                                   (4 * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - (i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->m * 4) );

    /* close n loop */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, 4 );
  }

  if ( (i_mateltwise_desc->n % 4 != 0) && (i_pad_vnni == 1) ) {
    /* reset v1 to 0 */
    libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                               1, 1, 0, 1, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );

    /* m loop header */
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, i_mateltwise_desc->m );

    /* actual transpose */
    for (l_r = 0; l_r < l_remainder_n; l_r++) {
      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                              i_gp_reg_in, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 0, l_load_instr_width );

      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                              i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->datatype_size_out, 0, l_store_instr_width );

      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in,
                                                     i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->ldi );
    }

    for (l_r = 0; l_r < (4 - l_remainder_n); l_r++) {
      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                              i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->datatype_size_out, 1, l_store_instr_width );
    }

    /* advance input pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in,
                                                   (l_remainder_n * i_mateltwise_desc->ldi - 1) * i_micro_kernel_config->datatype_size_in );


    /* close m loop */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, 1 );
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni2_mbit_scalar_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                      const unsigned int                      i_gp_reg_in,
                                                                                      const unsigned int                      i_gp_reg_out,
                                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                                      const unsigned int                      i_gp_reg_scratch,
                                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                                      const unsigned int                      i_pad_vnni ) {
  libxsmm_aarch64_asimd_width l_load_instr_width;
  libxsmm_aarch64_asimd_width l_store_instr_width;

  /* select load and store instructions */
  if ( i_micro_kernel_config->datatype_size_in == 8 ) {
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_D;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_D;
  } else if ( i_micro_kernel_config->datatype_size_in == 4 ) {
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_S;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_S;
  } else if ( i_micro_kernel_config->datatype_size_in == 2 ) {
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_H;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_H;
  } else if ( i_micro_kernel_config->datatype_size_in == 1 ) {
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_B;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_B;
  } else {
    /* should not happen */
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_B;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_B;
  }

  if ( (i_pad_vnni == 0) && (i_mateltwise_desc->n % 2 == 1) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  if ( i_mateltwise_desc->n >= 2 ) {
    /* n loop header */
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, (i_mateltwise_desc->n/2)*2 );

    /* m loop header */
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, i_mateltwise_desc->m );

    /* actual transpose */
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                            i_gp_reg_in, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 0, l_load_instr_width );

    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                            i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->datatype_size_out, 0, l_store_instr_width );

    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in,
                                                   i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->ldi );

    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                            i_gp_reg_in, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 0, l_load_instr_width );

    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                            i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->datatype_size_out, 0, l_store_instr_width );


    /* advance input pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in,
                                                   i_micro_kernel_config->datatype_size_in * (i_mateltwise_desc->ldi-1) );

    /* close m loop */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, 1 );

    /* advance output pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in,
                                                   (2 * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in) - (i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->m) );

    /* advance input pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out,
                                                   (2 * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - (i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->m * 2) );

    /* close n loop */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, 2 );
  }

  if ( (i_mateltwise_desc->n % 2 == 1) && (i_pad_vnni == 1) ) {
    /* reset v1 to 0 */
    libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                               1, 1, 0, 1,LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );

    /* m loop header */
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, i_mateltwise_desc->m );

    /* actual transpose */
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                            i_gp_reg_in, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->datatype_size_in, 0, l_load_instr_width );

    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                            i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->datatype_size_out, 0, l_store_instr_width );

    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                            i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->datatype_size_out, 1, l_store_instr_width );

    /* close m loop */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, 1 );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni2_to_vnni2t_mbit_scalar_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                        libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                        const unsigned int                      i_gp_reg_in,
                                                                                        const unsigned int                      i_gp_reg_out,
                                                                                        const unsigned int                      i_gp_reg_m_loop,
                                                                                        const unsigned int                      i_gp_reg_n_loop,
                                                                                        const unsigned int                      i_gp_reg_scratch,
                                                                                        const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                        const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int l_ldi = i_mateltwise_desc->ldi*2;
  unsigned int l_ldo = i_mateltwise_desc->ldo*2;
  libxsmm_aarch64_asimd_width l_load_instr_width;
  libxsmm_aarch64_asimd_width l_store_instr_width;

  /* select load and store instructions */
  if ( i_micro_kernel_config->datatype_size_in == 8 ) {
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_D;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_D;
  } else if ( i_micro_kernel_config->datatype_size_in == 4 ) {
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_S;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_S;
  } else if ( i_micro_kernel_config->datatype_size_in == 2 ) {
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_H;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_H;
  } else if ( i_micro_kernel_config->datatype_size_in == 1 ) {
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_B;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_B;
  } else {
    /* should not happen */
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_B;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_B;
  }

  if ( (i_mateltwise_desc->m % 2 == 0) && (i_mateltwise_desc->n % 2 == 0) ) {
    /* n loop header */
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, i_mateltwise_desc->n );

    /* m loop header */
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, i_mateltwise_desc->m );

    /* actual transpose */
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                            i_gp_reg_in, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->datatype_size_in, 0, l_load_instr_width );
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                            i_gp_reg_in, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->datatype_size_in, 1, l_load_instr_width );
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                            i_gp_reg_in, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->datatype_size_in, 2, l_load_instr_width );
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                            i_gp_reg_in, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->datatype_size_in, 3, l_load_instr_width );

    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                            i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->datatype_size_out, 0, l_store_instr_width );
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                            i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->datatype_size_out, 2, l_store_instr_width );
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                            i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->datatype_size_out, 1, l_store_instr_width );
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                            i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->datatype_size_out, 3, l_store_instr_width );

    /* advance output pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out,
                                                   (i_micro_kernel_config->datatype_size_out * l_ldo) - (i_micro_kernel_config->datatype_size_out * 4) );

    /* close m loop */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, 2 );

    /* advance output pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in,
                                                   (l_ldi * i_micro_kernel_config->datatype_size_in) - (i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->m * 2) );

    /* advance input pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out,
                                                   ( i_micro_kernel_config->datatype_size_out * l_ldo * i_mateltwise_desc->m/2 ) - (i_micro_kernel_config->datatype_size_out * 4) );

    /* close n loop */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, 2 );
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni4t_mbit_scalar_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                        libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                        const unsigned int                      i_gp_reg_in,
                                                                                        const unsigned int                      i_gp_reg_out,
                                                                                        const unsigned int                      i_gp_reg_m_loop,
                                                                                        const unsigned int                      i_gp_reg_n_loop,
                                                                                        const unsigned int                      i_gp_reg_scratch,
                                                                                        const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                        const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int l_ldi = i_mateltwise_desc->ldi*4;
  unsigned int l_ldo = i_mateltwise_desc->ldo*4;
  libxsmm_aarch64_asimd_width l_load_instr_width;
  libxsmm_aarch64_asimd_width l_store_instr_width;
  unsigned int l_m = 0, l_n = 0;

  /* select load and store instructions */
  if ( i_micro_kernel_config->datatype_size_in == 8 ) {
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_D;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_D;
  } else if ( i_micro_kernel_config->datatype_size_in == 4 ) {
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_S;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_S;
  } else if ( i_micro_kernel_config->datatype_size_in == 2 ) {
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_H;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_H;
  } else if ( i_micro_kernel_config->datatype_size_in == 1 ) {
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_B;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_B;
  } else {
    /* should not happen */
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_B;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_B;
  }

  if ( (i_mateltwise_desc->m % 4 == 0) && (i_mateltwise_desc->n % 4 == 0) ) {
    /* n loop header */
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, i_mateltwise_desc->n );

    /* m loop header */
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, i_mateltwise_desc->m );

    /* actual transpose */
    for (l_m = 0; l_m < 4; l_m++) {
      for (l_n = 0; l_n < 4; l_n++) {
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                i_gp_reg_in, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->datatype_size_in, l_m * 4 + l_n, l_load_instr_width );
      }
    }

    for (l_n = 0; l_n < 4; l_n++) {
      for (l_m = 0; l_m < 4; l_m++) {
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                                i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->datatype_size_out, l_m * 4 + l_n, l_store_instr_width );
      }
    }


    /* advance output pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out,
                                                   (i_micro_kernel_config->datatype_size_out * l_ldo) - (i_micro_kernel_config->datatype_size_out * 16) );

    /* close m loop */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, 4 );

    /* advance input pointer if need be */
    if (l_ldi * i_micro_kernel_config->datatype_size_in > i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->m * 4) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in,
                                                     (l_ldi * i_micro_kernel_config->datatype_size_in) - (i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->m * 4) );
    }


    /* advance output pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out,
                                                   ( i_micro_kernel_config->datatype_size_out * l_ldo * i_mateltwise_desc->m/4 ) - (i_micro_kernel_config->datatype_size_out * 16) );

    /* close n loop */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, 4 );
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_padnm_mod2_mbit_scalar_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                        libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                        const unsigned int                      i_gp_reg_in,
                                                                                        const unsigned int                      i_gp_reg_out,
                                                                                        const unsigned int                      i_gp_reg_m_loop,
                                                                                        const unsigned int                      i_gp_reg_n_loop,
                                                                                        const unsigned int                      i_gp_reg_scratch,
                                                                                        const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                        const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  libxsmm_aarch64_asimd_width l_load_instr_width;
  libxsmm_aarch64_asimd_width l_store_instr_width;

  /* select load and store instructions */
  if ( i_micro_kernel_config->datatype_size_in == 8 ) {
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_D;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_D;
  } else if ( i_micro_kernel_config->datatype_size_in == 4 ) {
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_S;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_S;
  } else if ( i_micro_kernel_config->datatype_size_in == 2 ) {
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_H;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_H;
  } else if ( i_micro_kernel_config->datatype_size_in == 1 ) {
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_B;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_B;
  } else {
    /* should not happen */
    l_load_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_B;
    l_store_instr_width = LIBXSMM_AARCH64_ASIMD_WIDTH_B;
  }

  /* n loop header */
  libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, i_mateltwise_desc->n );

  /* m loop header */
  libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, i_mateltwise_desc->m );

  /* actual copy / padding */
  libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                          i_gp_reg_in, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->datatype_size_in, 0, l_load_instr_width );

  libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                          i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->datatype_size_out, 0, l_store_instr_width );

  /* close m loop */
  libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, 1 );

  /* pad in M dimension during regular N loop */
  if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD2 ||
        i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2) &&
       (i_mateltwise_desc->m % 2 == 1) ) {
    /* reset v1 to 0 */
    libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                               1, 1, 0, 1,LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );

    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF,
                                            i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 1, l_store_instr_width );
  }

  /* advance output pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in,
                                                 (i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in) - (i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->m) );

  /* advance input pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out,
                                                 (i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - (i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->m) );

  /* close n loop */
  libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, 1 );

  /* pad in M dimension during regular N loop */
  if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD2 ||
        i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2) &&
        (i_mateltwise_desc->n % 2 == 1) ) {
    /* reset v1 to 0 */
    libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                               1, 1, 0, 1,LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );

    /* m loop header */
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, i_mateltwise_desc->m );

    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                            i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->datatype_size_out, 1, l_store_instr_width );

    /* close m loop */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, 1 );

    /* pad in M dimension during regular N loop */
    if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2) &&
         (i_mateltwise_desc->m % 2 == 1) ) {
      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF,
                                              i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 1, l_store_instr_width );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_64bit_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                const unsigned int                      i_gp_reg_in,
                                                                                const unsigned int                      i_gp_reg_out,
                                                                                const unsigned int                      i_gp_reg_m_loop,
                                                                                const unsigned int                      i_gp_reg_n_loop,
                                                                                const unsigned int                      i_gp_reg_scratch,
                                                                                const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  if ( i_mateltwise_desc->m % 4 == 0 && i_mateltwise_desc->n % 4 == 0 ) {
    /* m loop header */
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, i_mateltwise_desc->m );

    /* n loop header */
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, i_mateltwise_desc->n );

    /* shuffle network */
    libxsmm_generator_transform_norm_to_normt_64bit_4x4_shufflenetwork_aarch64_asimd( io_generated_code, i_gp_reg_in, i_gp_reg_out, i_gp_reg_scratch,
                                                                                      2, 2, i_micro_kernel_config, i_mateltwise_desc );

    /* advance input pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in,
                                                   4 * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in );

    /* advance output pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out,
                                                   4 * i_micro_kernel_config->datatype_size_out );

    /* close n loop */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, 4 );

    /* advance output pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out,
                                                   (4 * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - (i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->n) );

    /* advance input pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in,
                                                   (i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) - (i_micro_kernel_config->datatype_size_in * 4) );

    /* close m loop */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, 4 );
  } else {
    libxsmm_generator_transform_norm_to_normt_mbit_scalar_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                     i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                                     i_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_32bit_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                const unsigned int                      i_gp_reg_in,
                                                                                const unsigned int                      i_gp_reg_out,
                                                                                const unsigned int                      i_gp_reg_m_loop,
                                                                                const unsigned int                      i_gp_reg_n_loop,
                                                                                const unsigned int                      i_gp_reg_scratch,
                                                                                const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  if ( i_mateltwise_desc->m % 8 == 0 && i_mateltwise_desc->n % 8 == 0 ) {
    /* m loop header */
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, i_mateltwise_desc->m );

    /* n loop header */
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, i_mateltwise_desc->n );

    /* shuffle network */
    libxsmm_generator_transform_norm_to_normt_32bit_8x8_shufflenetwork_aarch64_asimd( io_generated_code, i_gp_reg_in, i_gp_reg_out, i_gp_reg_scratch,
                                                                                      8, 8, i_micro_kernel_config, i_mateltwise_desc );

    /* advance input pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in,
                                                   8 * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in );

    /* advance output pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out,
                                                   8 * i_micro_kernel_config->datatype_size_out );

    /* close n loop */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, 8 );

    /* advance output pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out,
                                                   (8 * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - (i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->n) );

    /* advance input pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in,
                                                   (i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) - (i_micro_kernel_config->datatype_size_in * 8) );

    /* close m loop */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, 8 );
  } else  if ( i_mateltwise_desc->m % 4 == 0 && i_mateltwise_desc->n % 4 == 0 ) {
    /* m loop header */
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, i_mateltwise_desc->m );

    /* n loop header */
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, i_mateltwise_desc->n );

    /* shuffle network */
    libxsmm_generator_transform_norm_to_normt_32bit_4x4_shufflenetwork_aarch64_asimd( io_generated_code, i_gp_reg_in, i_gp_reg_out, i_gp_reg_scratch,
                                                                                      4, 4, i_micro_kernel_config, i_mateltwise_desc );

    /* advance input pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in,
                                                   4 * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in );

    /* advance output pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out,
                                                   4 * i_micro_kernel_config->datatype_size_out );

    /* close n loop */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, 4 );

    /* advance output pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out,
                                                   (4 * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - (i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->n) );

    /* advance input pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in,
                                                   (i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) - (i_micro_kernel_config->datatype_size_in * 4) );

    /* close m loop */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, 4 );
  } else {
    libxsmm_generator_transform_norm_to_normt_mbit_scalar_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                     i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                                     i_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_16bit_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                const unsigned int                      i_gp_reg_in,
                                                                                const unsigned int                      i_gp_reg_out,
                                                                                const unsigned int                      i_gp_reg_m_loop,
                                                                                const unsigned int                      i_gp_reg_n_loop,
                                                                                const unsigned int                      i_gp_reg_scratch,
                                                                                const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  libxsmm_generator_transform_norm_to_normt_mbit_scalar_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                   i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                                   i_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_08bit_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                const unsigned int                      i_gp_reg_in,
                                                                                const unsigned int                      i_gp_reg_out,
                                                                                const unsigned int                      i_gp_reg_m_loop,
                                                                                const unsigned int                      i_gp_reg_n_loop,
                                                                                const unsigned int                      i_gp_reg_scratch,
                                                                                const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  libxsmm_generator_transform_norm_to_normt_mbit_scalar_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                   i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                                   i_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni2_16bit_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                const unsigned int                      i_gp_reg_in,
                                                                                const unsigned int                      i_gp_reg_out,
                                                                                const unsigned int                      i_gp_reg_m_loop,
                                                                                const unsigned int                      i_gp_reg_n_loop,
                                                                                const unsigned int                      i_gp_reg_scratch,
                                                                                const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                                const unsigned int                      i_pad_vnni ) {
  libxsmm_generator_transform_norm_to_vnni2_mbit_scalar_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                  i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                                  i_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc, i_pad_vnni );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni4_16bit_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                const unsigned int                      i_gp_reg_in,
                                                                                const unsigned int                      i_gp_reg_out,
                                                                                const unsigned int                      i_gp_reg_m_loop,
                                                                                const unsigned int                      i_gp_reg_n_loop,
                                                                                const unsigned int                      i_gp_reg_scratch,
                                                                                const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                                const unsigned int                      i_pad_vnni ) {
  libxsmm_generator_transform_norm_to_vnni4_mbit_scalar_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                  i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                                  i_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc, i_pad_vnni );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni4_08bit_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                const unsigned int                      i_gp_reg_in,
                                                                                const unsigned int                      i_gp_reg_out,
                                                                                const unsigned int                      i_gp_reg_m_loop,
                                                                                const unsigned int                      i_gp_reg_n_loop,
                                                                                const unsigned int                      i_gp_reg_scratch,
                                                                                const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                                const unsigned int                      i_pad_vnni ) {
  libxsmm_generator_transform_norm_to_vnni4_mbit_scalar_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                  i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                                  i_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc, i_pad_vnni );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni2_to_vnni2t_16bit_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                  libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                  const unsigned int                      i_gp_reg_in,
                                                                                  const unsigned int                      i_gp_reg_out,
                                                                                  const unsigned int                      i_gp_reg_m_loop,
                                                                                  const unsigned int                      i_gp_reg_n_loop,
                                                                                  const unsigned int                      i_gp_reg_scratch,
                                                                                  const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                  const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  libxsmm_generator_transform_vnni2_to_vnni2t_mbit_scalar_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                   i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                                   i_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni4t_16bit_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                  libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                  const unsigned int                      i_gp_reg_in,
                                                                                  const unsigned int                      i_gp_reg_out,
                                                                                  const unsigned int                      i_gp_reg_m_loop,
                                                                                  const unsigned int                      i_gp_reg_n_loop,
                                                                                  const unsigned int                      i_gp_reg_scratch,
                                                                                  const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                  const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  libxsmm_generator_transform_vnni4_to_vnni4t_mbit_scalar_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                   i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                                   i_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_padnm_mod2_16bit_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                  libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                  const unsigned int                      i_gp_reg_in,
                                                                                  const unsigned int                      i_gp_reg_out,
                                                                                  const unsigned int                      i_gp_reg_m_loop,
                                                                                  const unsigned int                      i_gp_reg_n_loop,
                                                                                  const unsigned int                      i_gp_reg_scratch,
                                                                                  const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                  const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  libxsmm_generator_transform_norm_padnm_mod2_mbit_scalar_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                     i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                                     i_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_aarch64_asimd_microkernel( libxsmm_generated_code*                        io_generated_code,
                                                            libxsmm_loop_label_tracker*                    io_loop_label_tracker,
                                                            libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
                                                            const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
                                                            const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {
  i_gp_reg_mapping->gp_reg_in        = LIBXSMM_AARCH64_GP_REG_X8;
  i_gp_reg_mapping->gp_reg_out       = LIBXSMM_AARCH64_GP_REG_X9;
  i_gp_reg_mapping->gp_reg_m_loop    = LIBXSMM_AARCH64_GP_REG_X10;
  i_gp_reg_mapping->gp_reg_n_loop    = LIBXSMM_AARCH64_GP_REG_X11;
  i_gp_reg_mapping->gp_reg_scratch_0 = LIBXSMM_AARCH64_GP_REG_X6;
  i_gp_reg_mapping->gp_reg_scratch_1 = LIBXSMM_AARCH64_GP_REG_X7;

  /* load pointers from struct */
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 32,
                                        i_gp_reg_mapping->gp_reg_in );
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 64,
                                        i_gp_reg_mapping->gp_reg_out );

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

  if ( ( LIBXSMM_DATATYPE_I64 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
         LIBXSMM_DATATYPE_I64 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) ||
       ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
         LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) ) {
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) {
      libxsmm_generator_transform_norm_to_normt_64bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else if ( ( LIBXSMM_DATATYPE_I32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
                LIBXSMM_DATATYPE_I32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) ||
              ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
                LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) ) {
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) {
      libxsmm_generator_transform_norm_to_normt_32bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else if ( ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
                LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) ||
              ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
                LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) ||
              ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
                LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) ) {
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) {
      libxsmm_generator_transform_norm_to_normt_16bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2_TO_VNNI2T) {
      libxsmm_generator_transform_vnni2_to_vnni2t_16bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                   i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                   i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                   i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc );
    }  else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI4T) {
      libxsmm_generator_transform_vnni4_to_vnni4t_16bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                   i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                   i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                   i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2) {
      libxsmm_generator_transform_norm_to_vnni2_16bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc, 0 );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD) {
      libxsmm_generator_transform_norm_to_vnni2_16bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc, 1 );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4) {
      libxsmm_generator_transform_norm_to_vnni4_16bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc, 0 );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4_PAD) {
      libxsmm_generator_transform_norm_to_vnni4_16bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc, 1 );
    } else if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD2)  ||
                (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD2)  ||
                (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2)    ) {
      libxsmm_generator_transform_norm_padnm_mod2_16bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                   i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                   i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                   i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else if ( ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
                LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) ||
              ( LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
                LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) ) {
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) {
      libxsmm_generator_transform_norm_to_normt_08bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4) {
      libxsmm_generator_transform_norm_to_vnni4_08bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc, 0 );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4_PAD) {
      libxsmm_generator_transform_norm_to_vnni4_08bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc, 1 );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
}

