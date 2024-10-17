/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_mateltwise_sse_avx_avx512.h"
#include "generator_mateltwise_transform_common_x86.h"
#include "generator_mateltwise_transform_avx.h"
#include "generator_mateltwise_transform_sse.h"
#include "generator_common_x86.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"


LIBXSMM_API_INTERN
void libxsmm_generator_transform_08way_permute128_network_avx( libxsmm_generated_code* io_generated_code,
                                                               const char              i_vector_name,
                                                               const unsigned int      i_vec_reg_src_start,
                                                               const unsigned char     i_in_idx[8],
                                                               const unsigned int      i_sec_op_offset,
                                                               const unsigned int      i_vec_reg_dst_start,
                                                               const unsigned int      i_perm_instr,
                                                               const unsigned char     i_perm_imm[8] ) {
  unsigned int l_i = 0;

  if ( ((i_vec_reg_src_start != 0) && (i_vec_reg_src_start != 8)) ||
       ((i_vec_reg_dst_start != 0) && (i_vec_reg_dst_start != 8))    ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  for ( l_i = 0; l_i < 8; ++l_i ) {
    unsigned int in0 = i_vec_reg_src_start + i_in_idx[l_i];
    unsigned int in1 = in0 + i_sec_op_offset;

    libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, i_perm_instr, i_vector_name,
                                                   in1, in0, i_vec_reg_dst_start + l_i, i_perm_imm[l_i] );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_mbit_scalar_avx_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                            libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                            const unsigned int                      i_gp_reg_in,
                                                                            const unsigned int                      i_gp_reg_out,
                                                                            const unsigned int                      i_gp_reg_m_loop,
                                                                            const unsigned int                      i_gp_reg_n_loop,
                                                                            const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                            const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  const unsigned int l_load_instr  = ( i_micro_kernel_config->datatype_size_in == 4 ) ? LIBXSMM_X86_INSTR_VMOVSS_LD : LIBXSMM_X86_INSTR_VMOVSD_LD;
  const unsigned int l_store_instr = ( i_micro_kernel_config->datatype_size_in == 4 ) ? LIBXSMM_X86_INSTR_VMOVSS_ST : LIBXSMM_X86_INSTR_VMOVSD_ST;

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
  libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, l_load_instr,
                                    i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                    'x', 0, 0, 1, 0 );

  libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, l_store_instr,
                                    i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                    'x', 0, 0, 0, 1 );

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_in, (long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in );

  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, i_micro_kernel_config->datatype_size_out );

  /* close n loop */
  libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                              i_gp_reg_n_loop, i_mateltwise_desc->n );

  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, ((long long)i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->n) );

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                   i_gp_reg_in, ((long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) - (i_micro_kernel_config->datatype_size_in) );

  /* close m loop */
  libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                              i_gp_reg_m_loop, i_mateltwise_desc->m );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_64bit_4x8_shufflenetwork_avx( libxsmm_generated_code*                 io_generated_code,
                                                                             const unsigned int                      i_gp_reg_in,
                                                                             const unsigned int                      i_gp_reg_out,
                                                                             const unsigned int                      i_m_valid,
                                                                             const unsigned int                      i_n_valid,
                                                                             const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                             const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int l_m_masking = (i_m_valid < 4) ? 1 : 0;
  unsigned int l_n0_masking = (i_n_valid < 4) ? 1 : 0;
  unsigned int l_n1_masking = ((i_n_valid < 8) && (i_n_valid > 4)) ? 1 : 0;
  unsigned int l_ld_instr = (l_m_masking == 0) ? i_micro_kernel_config->vmove_instruction_in  : LIBXSMM_X86_INSTR_VMASKMOVPD_LD;
  unsigned int l_st0_instr = (l_n0_masking == 0) ? i_micro_kernel_config->vmove_instruction_out : LIBXSMM_X86_INSTR_VMASKMOVPD_ST;
  unsigned int l_st1_instr = (l_n1_masking == 0) ? i_micro_kernel_config->vmove_instruction_out : LIBXSMM_X86_INSTR_VMASKMOVPD_ST;
  unsigned int l_n0_advance = (i_n_valid < 4) ? i_n_valid : 4;
  unsigned int l_n1_advance = ((i_n_valid <= 8) && (i_n_valid >= 4)) ? i_n_valid - 4 : 0;

  /* load mask register */
  if ( l_m_masking != 0 ) {
    unsigned int l_m = 0;
    unsigned long long l_data[4] = { 0 };

    for ( l_m = 0; l_m < 4; ++l_m ) {
      l_data[l_m] = (l_m < i_m_valid) ? 0xFFFFFFFFFFFFFFFF : 0x0;
    }

    /* load register with constants from code */
    libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char*)l_data, "load", 'y', 8 );
  }

  /* load 8 registers */
  libxsmm_generator_transform_Xway_full_load_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                         i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                         l_ld_instr, 8, i_n_valid, l_m_masking, 8 );

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_in, (long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * i_n_valid );

  /* first shuffle stage */
  {
    unsigned char l_in_idx[16] = { 0x0, 0x0, 0x2, 0x2, 0x4, 0x4, 0x6, 0x6, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
    unsigned int  l_src_start = 0;
    unsigned int  l_dst_start = 8;
    libxsmm_generator_transform_Xway_unpack_network_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                l_in_idx, l_src_start, l_dst_start, 1,
                                                                LIBXSMM_X86_INSTR_VUNPCKLPD, LIBXSMM_X86_INSTR_VUNPCKHPD, 8 );
  }

  /* second shuffle stage */
  {
    unsigned char l_in_idx[8] = { 0x00, 0x01, 0x00, 0x01, 0x04, 0x05, 0x04, 0x05 };
    unsigned int  l_src_start = 8;
    unsigned int  l_dst_start = 0;
    unsigned char l_perm_imm[8] = { 0x20, 0x20, 0x31, 0x31, 0x20, 0x20, 0x31, 0x31 };
    libxsmm_generator_transform_08way_permute128_network_avx( io_generated_code, i_micro_kernel_config->vector_name,
                                                              l_src_start, l_in_idx, 2, l_dst_start, LIBXSMM_X86_INSTR_VPERM2F128, l_perm_imm );
  }

  /* load mask register */
  if ( (l_n0_masking != 0) || (l_n1_masking != 0) ) {
    unsigned int l_n = 0;
    unsigned long long l_data[4] = { 0 };
    unsigned int l_n_valid = ( i_n_valid < 4 ) ? i_n_valid : (i_n_valid - 4);
    for ( l_n = 0; l_n < 4; ++l_n ) {
      l_data[l_n] = (l_n < l_n_valid) ? 0xFFFFFFFFFFFFFFFF : 0x0;
    }

    /* load register with constants from code */
    libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char*)l_data, "store", 'y', 8 );
  }

  /* storing 8x 32byte */
  libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                          i_gp_reg_out, 0, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                          l_st0_instr, l_n0_masking, 8, i_m_valid );
  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, (long long)i_micro_kernel_config->datatype_size_out * l_n0_advance );

  if ( l_n1_advance > 0 ) {
    libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                            i_gp_reg_out, 4, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                            l_st1_instr, l_n1_masking, 8, i_m_valid );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (long long)i_micro_kernel_config->datatype_size_out * l_n1_advance );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_128bit_avx_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                       libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                       const unsigned int                      i_gp_reg_in,
                                                                       const unsigned int                      i_gp_reg_out,
                                                                       const unsigned int                      i_gp_reg_m_loop,
                                                                       const unsigned int                      i_gp_reg_n_loop,
                                                                       const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                       const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int l_load_instr  = LIBXSMM_X86_INSTR_VMOVUPS;
  unsigned int l_store_instr = LIBXSMM_X86_INSTR_VMOVUPS;
  unsigned int l_datasize_in = 16;
  unsigned int l_datasize_out = 16;

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
  libxsmm_x86_instruction_unified_vec_move( io_generated_code, l_load_instr,
                                            i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                            'x', 0, 0, 1, 0);

  libxsmm_x86_instruction_unified_vec_move( io_generated_code, l_store_instr,
                                            i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                            'x', 0, 0, 1, 1);

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_in, (long long)i_mateltwise_desc->ldi * l_datasize_in );

  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, l_datasize_out );

  /* close n loop */
  libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                              i_gp_reg_n_loop, i_mateltwise_desc->n );

  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, ((long long)i_mateltwise_desc->ldo * l_datasize_out) - ((long long)l_datasize_out * i_mateltwise_desc->n) );

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                   i_gp_reg_in, ((long long)i_mateltwise_desc->ldi * l_datasize_in * i_mateltwise_desc->n) - ((long long)l_datasize_in) );

  /* close m loop */
  libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                              i_gp_reg_m_loop, i_mateltwise_desc->m );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_64bit_avx_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  if ( (i_mateltwise_desc->m < 2) && (i_mateltwise_desc->n < 4) ) {
    libxsmm_generator_transform_norm_to_normt_mbit_scalar_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                           i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                           i_micro_kernel_config, i_mateltwise_desc );
  } else {
    /* optimized shuffle network for SIMD aligned sizes */
    if ( i_mateltwise_desc->m >= 4 ) {
      /* open m loop */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_m_loop, 4 );

      if ( i_mateltwise_desc->n >= 8 ) {
        /* open n loop */
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
        libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                         i_gp_reg_n_loop, 8 );

        /* call shuffle network */
        libxsmm_generator_transform_norm_to_normt_64bit_4x8_shufflenetwork_avx( io_generated_code, i_gp_reg_in, i_gp_reg_out, 4, 8,
                                                                                i_micro_kernel_config, i_mateltwise_desc );

        /* close n footer */
        libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                    i_gp_reg_n_loop, (i_mateltwise_desc->n/8)*8 );
      }
      if ( i_mateltwise_desc->n % 8 != 0 ) {
        /* call shuffle network */
        libxsmm_generator_transform_norm_to_normt_64bit_4x8_shufflenetwork_avx( io_generated_code, i_gp_reg_in, i_gp_reg_out, 4, i_mateltwise_desc->n % 8,
                                                                                i_micro_kernel_config, i_mateltwise_desc );
      }

      /* advance output pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_out, (4LL * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) );

      /* advance input pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                       i_gp_reg_in, ((long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) - (4LL * i_micro_kernel_config->datatype_size_in) );

      /* close m loop */
      libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                  i_gp_reg_m_loop, (i_mateltwise_desc->m/4)*4 );
    }
    if ( i_mateltwise_desc->m % 4 != 0 ) {
      if ( i_mateltwise_desc->n >= 8 ) {
        /* open n loop */
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
        libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                         i_gp_reg_n_loop, 8 );

        /* call shuffle network */
        libxsmm_generator_transform_norm_to_normt_64bit_4x8_shufflenetwork_avx( io_generated_code, i_gp_reg_in, i_gp_reg_out, i_mateltwise_desc->m % 4, 8,
                                                                                i_micro_kernel_config, i_mateltwise_desc );

        /* close n footer */
        libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                    i_gp_reg_n_loop, (i_mateltwise_desc->n/8)*8 );
      }
      if ( i_mateltwise_desc->n % 8 != 0 ) {
        /* call shuffle network */
        libxsmm_generator_transform_norm_to_normt_64bit_4x8_shufflenetwork_avx( io_generated_code, i_gp_reg_in, i_gp_reg_out, i_mateltwise_desc->m % 4, i_mateltwise_desc->n % 8,
                                                                                i_micro_kernel_config, i_mateltwise_desc );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_32bit_8x8_shufflenetwork_avx( libxsmm_generated_code*                 io_generated_code,
                                                                             const unsigned int                      i_gp_reg_in,
                                                                             const unsigned int                      i_gp_reg_out,
                                                                             const unsigned int                      i_m_valid,
                                                                             const unsigned int                      i_n_valid,
                                                                             const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                             const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int l_m_masking = (i_m_valid < 8) ? 1 : 0;
  unsigned int l_n_masking = (i_n_valid < 8) ? 1 : 0;
  unsigned int l_ld_instr = (l_m_masking == 0) ? i_micro_kernel_config->vmove_instruction_in : LIBXSMM_X86_INSTR_VMASKMOVPS_LD;
  unsigned int l_st_instr = (l_n_masking == 0) ? i_micro_kernel_config->vmove_instruction_in : LIBXSMM_X86_INSTR_VMASKMOVPS_ST;

  /* load mask register */
  if ( l_m_masking != 0 ) {
    unsigned int l_m = 0;
    unsigned int l_data[8] = { 0 };

    for ( l_m = 0; l_m < 8; ++l_m ) {
      l_data[l_m] = (l_m < i_m_valid) ? 0xFFFFFFFF : 0x0;
    }

    /* load register with constants from code */
    libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char*)l_data, "load", 'y', 8 );
  }

  /* load 8 registers */
  libxsmm_generator_transform_Xway_full_load_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                         i_gp_reg_in, 0, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                                                         l_ld_instr, 8, i_n_valid, l_m_masking, 8 );

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_in, (long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * i_n_valid );

  /* first shuffle stage */
  {
    unsigned char l_in_idx[16] = { 0x0, 0x0, 0x2, 0x2, 0x4, 0x4, 0x6, 0x6, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
    unsigned int  l_src_start = 0;
    unsigned int  l_dst_start = 8;
    libxsmm_generator_transform_Xway_unpack_network_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                l_in_idx, l_src_start, l_dst_start, 1,
                                                                LIBXSMM_X86_INSTR_VUNPCKLPS, LIBXSMM_X86_INSTR_VUNPCKHPS, 8 );
  }

  /* second shuffle stage */
  {
    unsigned char l_in_idx[16] = { 0x0, 0x0, 0x1, 0x1, 0x4, 0x4, 0x5, 0x5, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
    unsigned int  l_src_start = 8;
    unsigned int  l_dst_start = 0;
    libxsmm_generator_transform_Xway_unpack_network_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                                l_in_idx, l_src_start, l_dst_start, 2,
                                                                LIBXSMM_X86_INSTR_VUNPCKLPD, LIBXSMM_X86_INSTR_VUNPCKHPD, 8 );
  }

  /* third shuffle stage */
  {
    unsigned char l_in_idx[8] = { 0x00, 0x01, 0x02, 0x03, 0x00, 0x01, 0x02, 0x03 };
    unsigned int  l_src_start = 0;
    unsigned int  l_dst_start = 8;
    unsigned char l_perm_imm[8] = { 0x20, 0x20, 0x20, 0x20, 0x31, 0x31, 0x31, 0x31 };
    libxsmm_generator_transform_08way_permute128_network_avx( io_generated_code, i_micro_kernel_config->vector_name,
                                                              l_src_start, l_in_idx, 4, l_dst_start, LIBXSMM_X86_INSTR_VPERM2F128, l_perm_imm );
  }

  /* load mask register */
  if ( l_n_masking != 0 ) {
    unsigned int l_n = 0;
    unsigned int l_data[8] = { 0 };

    for ( l_n = 0; l_n < 8; ++l_n ) {
      l_data[l_n] = (l_n < i_n_valid) ? 0xFFFFFFFF : 0x0;
    }

    /* load register with constants from code */
    libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char*)l_data, "store", 'y', 0 );
  }

  /* storing 8x 32byte */
  libxsmm_generator_transform_Xway_full_store_avx_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                          i_gp_reg_out, 8, i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                                          l_st_instr, l_n_masking, 0, i_m_valid );

  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, (long long)i_micro_kernel_config->datatype_size_out * i_n_valid );

}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_32bit_avx_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  if ( (i_mateltwise_desc->m < 4) && (i_mateltwise_desc->n < 4) ) {
    libxsmm_generator_transform_norm_to_normt_mbit_scalar_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                           i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                           i_micro_kernel_config, i_mateltwise_desc );
  } else {
    /* optimized shuffle network for SIMD aligned sizes */
    if ( i_mateltwise_desc->m >= 8 ) {
      /* open m loop */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_m_loop, 8 );

      if ( i_mateltwise_desc->n >= 8 ) {
        /* open n loop */
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
        libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                         i_gp_reg_n_loop, 8 );

        /* call shuffle network */
        libxsmm_generator_transform_norm_to_normt_32bit_8x8_shufflenetwork_avx( io_generated_code, i_gp_reg_in, i_gp_reg_out, 8, 8,
                                                                                i_micro_kernel_config, i_mateltwise_desc );

        /* close n footer */
        libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                    i_gp_reg_n_loop, (i_mateltwise_desc->n/8)*8 );
      }
      if ( i_mateltwise_desc->n % 8 != 0 ) {
        /* call shuffle network */
        libxsmm_generator_transform_norm_to_normt_32bit_8x8_shufflenetwork_avx( io_generated_code, i_gp_reg_in, i_gp_reg_out, 8, i_mateltwise_desc->n % 8,
                                                                                i_micro_kernel_config, i_mateltwise_desc );
      }

      /* advance output pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_out, (8LL * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) );

      /* advance input pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                       i_gp_reg_in, ((long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) - (8LL * i_micro_kernel_config->datatype_size_in) );

      /* close m loop */
      libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                  i_gp_reg_m_loop, (i_mateltwise_desc->m/8)*8 );
    }
    if ( i_mateltwise_desc->m % 8 != 0 ) {
      if ( i_mateltwise_desc->n >= 8 ) {
        /* open n loop */
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
        libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                         i_gp_reg_n_loop, 8 );

        /* call shuffle network */
        libxsmm_generator_transform_norm_to_normt_32bit_8x8_shufflenetwork_avx( io_generated_code, i_gp_reg_in, i_gp_reg_out, i_mateltwise_desc->m % 8, 8,
                                                                                i_micro_kernel_config, i_mateltwise_desc );

        /* close n footer */
        libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                    i_gp_reg_n_loop, (i_mateltwise_desc->n/8)*8 );
      }
      if ( i_mateltwise_desc->n % 8 != 0 ) {
        /* call shuffle network */
        libxsmm_generator_transform_norm_to_normt_32bit_8x8_shufflenetwork_avx( io_generated_code, i_gp_reg_in, i_gp_reg_out, i_mateltwise_desc->m % 8, i_mateltwise_desc->n % 8,
                                                                                i_micro_kernel_config, i_mateltwise_desc );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_16bit_avx_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  libxsmm_generator_transform_norm_to_normt_mbit_scalar_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                         i_micro_kernel_config, i_mateltwise_desc );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_08bit_avx_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  libxsmm_generator_transform_norm_to_normt_mbit_scalar_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                         i_micro_kernel_config, i_mateltwise_desc );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni2_16bit_avx_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                      const unsigned int                      i_pad_vnni ) {
  libxsmm_generator_transform_norm_to_vnni2_mbit_scalar_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                        i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                        i_micro_kernel_config, i_mateltwise_desc, i_pad_vnni );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni2_to_vnni2t_16bit_avx_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                        libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                        const unsigned int                      i_gp_reg_in,
                                                                        const unsigned int                      i_gp_reg_out,
                                                                        const unsigned int                      i_gp_reg_m_loop,
                                                                        const unsigned int                      i_gp_reg_n_loop,
                                                                        const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                        const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  libxsmm_generator_transform_vnni2_to_vnni2t_mbit_scalar_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                         i_micro_kernel_config, i_mateltwise_desc );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni4_16bit_avx_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                      const unsigned int                      i_pad_vnni ) {
  libxsmm_generator_transform_norm_to_vnni4_mbit_scalar_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                         i_micro_kernel_config, i_mateltwise_desc, i_pad_vnni );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni4_08bit_avx_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                      const unsigned int                      i_pad_vnni ) {
  libxsmm_generator_transform_norm_to_vnni4_mbit_scalar_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                         i_micro_kernel_config, i_mateltwise_desc, i_pad_vnni );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_norm_08bit_avx_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc) {
  libxsmm_generator_transform_vnni4_to_norm_mbit_scalar_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                         i_micro_kernel_config, i_mateltwise_desc);
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni2_08bit_avx_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc) {
  libxsmm_generator_transform_vnni4_to_vnni2_mbit_scalar_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                         i_micro_kernel_config, i_mateltwise_desc);
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni4t_16bit_avx_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                        libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                        const unsigned int                      i_gp_reg_in,
                                                                        const unsigned int                      i_gp_reg_out,
                                                                        const unsigned int                      i_gp_reg_m_loop,
                                                                        const unsigned int                      i_gp_reg_n_loop,
                                                                        const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                        const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  libxsmm_generator_transform_vnni4_to_vnni4t_mbit_scalar_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                           i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                           i_micro_kernel_config, i_mateltwise_desc );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni4t_08bit_avx_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                        libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                        const unsigned int                      i_gp_reg_in,
                                                                        const unsigned int                      i_gp_reg_out,
                                                                        const unsigned int                      i_gp_reg_m_loop,
                                                                        const unsigned int                      i_gp_reg_n_loop,
                                                                        const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                        const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  libxsmm_generator_transform_vnni4_to_vnni4t_mbit_scalar_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                           i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                           i_micro_kernel_config, i_mateltwise_desc );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni8_16bit_avx_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                      const unsigned int                      i_pad_vnni ) {
  libxsmm_generator_transform_norm_to_vnni8_mbit_scalar_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                         i_micro_kernel_config, i_mateltwise_desc, i_pad_vnni );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni8_08bit_avx_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                      const unsigned int                      i_pad_vnni ) {
  libxsmm_generator_transform_norm_to_vnni8_mbit_scalar_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                         i_micro_kernel_config, i_mateltwise_desc, i_pad_vnni );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni8_to_norm_08bit_avx_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc) {
  libxsmm_generator_transform_vnni8_to_norm_mbit_scalar_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                         i_micro_kernel_config, i_mateltwise_desc);
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni8_to_vnni8t_16bit_avx_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                        libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                        const unsigned int                      i_gp_reg_in,
                                                                        const unsigned int                      i_gp_reg_out,
                                                                        const unsigned int                      i_gp_reg_m_loop,
                                                                        const unsigned int                      i_gp_reg_n_loop,
                                                                        const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                        const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  libxsmm_generator_transform_vnni8_to_vnni8t_mbit_scalar_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                           i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                           i_micro_kernel_config, i_mateltwise_desc );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni8_to_vnni8t_08bit_avx_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                        libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                        const unsigned int                      i_gp_reg_in,
                                                                        const unsigned int                      i_gp_reg_out,
                                                                        const unsigned int                      i_gp_reg_m_loop,
                                                                        const unsigned int                      i_gp_reg_n_loop,
                                                                        const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                        const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  libxsmm_generator_transform_vnni8_to_vnni8t_mbit_scalar_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                           i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                           i_micro_kernel_config, i_mateltwise_desc );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_padnm_mod2_16bit_avx_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                        libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                        const unsigned int                      i_gp_reg_in,
                                                                        const unsigned int                      i_gp_reg_out,
                                                                        const unsigned int                      i_gp_reg_m_loop,
                                                                        const unsigned int                      i_gp_reg_n_loop,
                                                                        const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                        const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  libxsmm_generator_transform_norm_padnm_mod2_mbit_scalar_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                           i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                           i_micro_kernel_config, i_mateltwise_desc );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_padnm_mod4_08bit_avx_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                        libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                        const unsigned int                      i_gp_reg_in,
                                                                        const unsigned int                      i_gp_reg_out,
                                                                        const unsigned int                      i_gp_reg_m_loop,
                                                                        const unsigned int                      i_gp_reg_n_loop,
                                                                        const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                        const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  libxsmm_generator_transform_norm_padnm_mod4_mbit_scalar_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                           i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                           i_micro_kernel_config, i_mateltwise_desc );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_avx_microkernel( libxsmm_generated_code*                        io_generated_code,
                                                  libxsmm_loop_label_tracker*                    io_loop_label_tracker,
                                                  libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
                                                  const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
                                                  const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {
  unsigned int l_gp_reg_in  = LIBXSMM_X86_GP_REG_R8;
  unsigned int l_gp_reg_out = LIBXSMM_X86_GP_REG_R9;
  unsigned int l_gp_reg_mloop = LIBXSMM_X86_GP_REG_RAX;
  unsigned int l_gp_reg_nloop = LIBXSMM_X86_GP_REG_RDX;

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
         LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) ) {
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) {
      libxsmm_generator_transform_norm_to_normt_64bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else if ( ( LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
                LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) ||
              ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
                LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) ) {
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) {
      libxsmm_generator_transform_norm_to_normt_32bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
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
                LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) )   ) {
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) {
      libxsmm_generator_transform_norm_to_normt_16bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
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
      libxsmm_generator_transform_norm_to_normt_128bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                 l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
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
      libxsmm_generator_transform_norm_to_normt_64bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                 l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
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
      libxsmm_generator_transform_norm_to_normt_32bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                 l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
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
      libxsmm_generator_transform_norm_to_normt_128bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                 l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
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
      libxsmm_generator_transform_norm_to_normt_64bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                 l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
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
      libxsmm_generator_transform_norm_to_normt_32bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                 l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                 &l_trans_config, mock_desc );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2_TO_VNNI2T) {
      libxsmm_generator_transform_vnni2_to_vnni2t_16bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                         i_micro_kernel_config, i_mateltwise_desc );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI4T) {
      libxsmm_generator_transform_vnni4_to_vnni4t_16bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                         i_micro_kernel_config, i_mateltwise_desc );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI8_TO_VNNI8T) {
      libxsmm_generator_transform_vnni8_to_vnni8t_16bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                         i_micro_kernel_config, i_mateltwise_desc );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2) {
      libxsmm_generator_transform_norm_to_vnni2_16bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc, 0 );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD) {
      libxsmm_generator_transform_norm_to_vnni2_16bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc, 1 );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4) {
      libxsmm_generator_transform_norm_to_vnni4_16bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc, 0 );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4_PAD) {
      libxsmm_generator_transform_norm_to_vnni4_16bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc, 1 );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8) {
      libxsmm_generator_transform_norm_to_vnni8_16bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc, 0 );
    } else if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD2)  ||
                (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD2)  ||
                (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2)    ) {
      libxsmm_generator_transform_norm_padnm_mod2_16bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
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
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) {
      libxsmm_generator_transform_norm_to_normt_08bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI4T) {
      libxsmm_generator_transform_vnni4_to_vnni4t_08bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                         i_micro_kernel_config, i_mateltwise_desc );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_NORM) {
      libxsmm_generator_transform_vnni4_to_norm_08bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc);
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI8_TO_VNNI8T) {
      libxsmm_generator_transform_vnni8_to_vnni8t_08bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                         i_micro_kernel_config, i_mateltwise_desc );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI8_TO_NORM) {
      libxsmm_generator_transform_vnni8_to_norm_08bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc);
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI2) {
      libxsmm_generator_transform_vnni4_to_vnni2_08bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc);
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4) {
      libxsmm_generator_transform_norm_to_vnni4_08bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc, 0 );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4_PAD) {
      libxsmm_generator_transform_norm_to_vnni4_08bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc, 1 );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8) {
      libxsmm_generator_transform_norm_to_vnni8_08bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc, 0 );
    } else if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD4)  ||
                (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD4)  ||
                (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD4)    ) {
      libxsmm_generator_transform_norm_padnm_mod4_08bit_avx_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
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

