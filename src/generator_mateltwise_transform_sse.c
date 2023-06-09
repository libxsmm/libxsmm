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
#include "generator_mateltwise_transform_sse.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"


LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_mbit_scalar_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                            libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                            const unsigned int                      i_gp_reg_in,
                                                                            const unsigned int                      i_gp_reg_out,
                                                                            const unsigned int                      i_gp_reg_m_loop,
                                                                            const unsigned int                      i_gp_reg_n_loop,
                                                                            const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                            const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int l_load_instr  = LIBXSMM_X86_INSTR_UNDEF;
  unsigned int l_store_instr = LIBXSMM_X86_INSTR_UNDEF;
  unsigned int l_gp_temp = LIBXSMM_X86_GP_REG_R15;

  /* GP temp registers: check against loop and address registers */
  if ( (l_gp_temp == i_gp_reg_m_loop) || (l_gp_temp == i_gp_reg_n_loop) ||
       (l_gp_temp == i_gp_reg_in)     || (l_gp_temp == i_gp_reg_out) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GP_TEMP_MAPPING );
    return;
  }

  /* select load and store instructions */
  if ( i_micro_kernel_config->datatype_size_in == 8 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVQ;
    l_store_instr = LIBXSMM_X86_INSTR_MOVQ;
  } else if ( i_micro_kernel_config->datatype_size_in == 4 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVD;
    l_store_instr = LIBXSMM_X86_INSTR_MOVD;
  } else if ( i_micro_kernel_config->datatype_size_in == 2 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVW;
    l_store_instr = LIBXSMM_X86_INSTR_MOVW;
  } else if ( i_micro_kernel_config->datatype_size_in == 1 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVB;
    l_store_instr = LIBXSMM_X86_INSTR_MOVB;
  } else {
    /* should not happen */
  }

  /* save l_gp_temp to stack */
  libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_temp );

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
  libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                   i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                   l_gp_temp, 0 );

  libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                   i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                   l_gp_temp, 1 );

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
                                   i_gp_reg_in, ((long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) - ((long long)i_micro_kernel_config->datatype_size_in) );

  /* close m loop */
  libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                              i_gp_reg_m_loop, i_mateltwise_desc->m );

  /* restore l_gp_temp */
  libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_temp );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni2_mbit_scalar_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                            libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                            const unsigned int                      i_gp_reg_in,
                                                                            const unsigned int                      i_gp_reg_out,
                                                                            const unsigned int                      i_gp_reg_m_loop,
                                                                            const unsigned int                      i_gp_reg_n_loop,
                                                                            const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                            const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                            const unsigned int                      i_pad_vnni ) {
  unsigned int l_load_instr  = LIBXSMM_X86_INSTR_UNDEF;
  unsigned int l_store_instr = LIBXSMM_X86_INSTR_UNDEF;
  unsigned int l_gp_temp = LIBXSMM_X86_GP_REG_R15;

  /* GP temp registers: check against loop and address registers */
  if ( (l_gp_temp == i_gp_reg_m_loop) || (l_gp_temp == i_gp_reg_n_loop) ||
       (l_gp_temp == i_gp_reg_in)     || (l_gp_temp == i_gp_reg_out) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GP_TEMP_MAPPING );
    return;
  }

  if ( (i_pad_vnni == 0) && (i_mateltwise_desc->n % 2 == 1) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* select load and store instructions */
  if ( i_micro_kernel_config->datatype_size_in == 8 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVQ;
    l_store_instr = LIBXSMM_X86_INSTR_MOVQ;
  } else if ( i_micro_kernel_config->datatype_size_in == 4 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVD;
    l_store_instr = LIBXSMM_X86_INSTR_MOVD;
  } else if ( i_micro_kernel_config->datatype_size_in == 2 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVW;
    l_store_instr = LIBXSMM_X86_INSTR_MOVW;
  } else if ( i_micro_kernel_config->datatype_size_in == 1 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVB;
    l_store_instr = LIBXSMM_X86_INSTR_MOVB;
  } else {
    /* should not happen */
  }

  /* save l_gp_temp to stack */
  libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_temp );

  if ( i_mateltwise_desc->n >= 2 ) {
    /* n loop header */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_n_loop, 2 );

    /* m loop header */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_m_loop, 1 );

    /* actual transpose */
    libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                     i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                     l_gp_temp, 0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                     l_gp_temp, 1 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                     i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->ldi,
                                     l_gp_temp, 0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out,
                                     l_gp_temp, 1 );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, i_micro_kernel_config->datatype_size_in );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (long long)i_micro_kernel_config->datatype_size_out * 2 );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, i_mateltwise_desc->m );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, (2LL * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in) - (i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->m) );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (2LL * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - (i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->m * 2) );

    /* close n loop */
    libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_n_loop, (i_mateltwise_desc->n/2)*2 );
  }

  if ( (i_mateltwise_desc->n % 2 == 1) && (i_pad_vnni == 1) ) {
    /* m loop header */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
                                     libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
                                     libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_m_loop, 1 );

    /* actual transpose */
    libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                     i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                     l_gp_temp, 0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                     l_gp_temp, 1 );

    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     l_gp_temp, 0x0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out,
                                     l_gp_temp, 1 );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, i_micro_kernel_config->datatype_size_in );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (long long)i_micro_kernel_config->datatype_size_out * 2 );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, i_mateltwise_desc->m );
  }

  /* restore l_gp_temp */
  libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_temp );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni2_to_vnni2t_mbit_scalar_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                              libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                              const unsigned int                      i_gp_reg_in,
                                                                              const unsigned int                      i_gp_reg_out,
                                                                              const unsigned int                      i_gp_reg_m_loop,
                                                                              const unsigned int                      i_gp_reg_n_loop,
                                                                              const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                              const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int l_load_instr  = LIBXSMM_X86_INSTR_UNDEF;
  unsigned int l_store_instr = LIBXSMM_X86_INSTR_UNDEF;
  unsigned int l_gp_temp = LIBXSMM_X86_GP_REG_R15;
  unsigned int l_ldi = i_mateltwise_desc->ldi*2;
  unsigned int l_ldo = i_mateltwise_desc->ldo*2;

  /* GP temp registers: check against loop and address registers */
  if ( (l_gp_temp == i_gp_reg_m_loop) || (l_gp_temp == i_gp_reg_n_loop) ||
       (l_gp_temp == i_gp_reg_in)     || (l_gp_temp == i_gp_reg_out) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GP_TEMP_MAPPING );
    return;
  }

  /* select load and store instructions */
  if ( i_micro_kernel_config->datatype_size_in == 8 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVQ;
    l_store_instr = LIBXSMM_X86_INSTR_MOVQ;
  } else if ( i_micro_kernel_config->datatype_size_in == 4 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVD;
    l_store_instr = LIBXSMM_X86_INSTR_MOVD;
  } else if ( i_micro_kernel_config->datatype_size_in == 2 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVW;
    l_store_instr = LIBXSMM_X86_INSTR_MOVW;
  } else if ( i_micro_kernel_config->datatype_size_in == 1 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVB;
    l_store_instr = LIBXSMM_X86_INSTR_MOVB;
  } else {
    /* should not happen */
  }

  /* save l_gp_temp to stack */
  libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_temp );

  if ( (i_mateltwise_desc->m % 2 == 0) && (i_mateltwise_desc->n % 2 == 0) ) {
    /* n loop header */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_n_loop, 2 );

    /* m loop header */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_m_loop, 2 );

    /* actual transpose */
    libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                     i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                     l_gp_temp, 0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                     l_gp_temp, 1 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                     i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in,
                                     l_gp_temp, 0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out * 2,
                                     l_gp_temp, 1 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                     i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in * 2,
                                     l_gp_temp, 0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out,
                                     l_gp_temp, 1 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                     i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in * 3,
                                     l_gp_temp, 0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out * 3,
                                     l_gp_temp, 1 );

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
                                     i_gp_reg_out, ((long long)i_micro_kernel_config->datatype_size_out * l_ldo * i_mateltwise_desc->m/2) - ((long long)i_micro_kernel_config->datatype_size_out * 4) );

    /* close n loop */
    libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_n_loop, i_mateltwise_desc->n );
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* restore l_gp_temp */
  libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_temp );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni4_mbit_scalar_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                            libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                            const unsigned int                      i_gp_reg_in,
                                                                            const unsigned int                      i_gp_reg_out,
                                                                            const unsigned int                      i_gp_reg_m_loop,
                                                                            const unsigned int                      i_gp_reg_n_loop,
                                                                            const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                            const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                            const unsigned int                      i_pad_vnni ) {
  unsigned int l_load_instr  = LIBXSMM_X86_INSTR_UNDEF;
  unsigned int l_store_instr = LIBXSMM_X86_INSTR_UNDEF;
  unsigned int l_gp_temp = LIBXSMM_X86_GP_REG_R15;
  unsigned int l_n = 0;

  /* GP temp registers: check against loop and address registers */
  if ( (l_gp_temp == i_gp_reg_m_loop) || (l_gp_temp == i_gp_reg_n_loop) ||
       (l_gp_temp == i_gp_reg_in)     || (l_gp_temp == i_gp_reg_out) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GP_TEMP_MAPPING );
    return;
  }

  if ( (i_pad_vnni == 0) && (i_mateltwise_desc->n % 4 != 0) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* select load and store instructions */
  if ( i_micro_kernel_config->datatype_size_in == 8 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVQ;
    l_store_instr = LIBXSMM_X86_INSTR_MOVQ;
  } else if ( i_micro_kernel_config->datatype_size_in == 4 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVD;
    l_store_instr = LIBXSMM_X86_INSTR_MOVD;
  } else if ( i_micro_kernel_config->datatype_size_in == 2 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVW;
    l_store_instr = LIBXSMM_X86_INSTR_MOVW;
  } else if ( i_micro_kernel_config->datatype_size_in == 1 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVB;
    l_store_instr = LIBXSMM_X86_INSTR_MOVB;
  } else {
    /* should not happen */
  }

  /* save l_gp_temp to stack */
  libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_temp );

  if ( i_mateltwise_desc->n >= 4 ) {
    /* n loop header */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_n_loop, 4 );

    /* m loop header */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_m_loop, 1 );

    /* actual vnni formatting */
    for ( l_n = 0; l_n < 4; ++l_n ) {
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                       i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->ldi * l_n,
                                       l_gp_temp, 0 );

      libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                       i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out * l_n,
                                       l_gp_temp, 1 );
    }

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, i_micro_kernel_config->datatype_size_in );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (long long)i_micro_kernel_config->datatype_size_out * 4 );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, i_mateltwise_desc->m );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, (4LL * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in) - ((long long)i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->m) );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (4LL * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->m * 4) );

    /* close n loop */
    libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_n_loop, (i_mateltwise_desc->n/4)*4 );
  }

  if ( (i_mateltwise_desc->n % 4 != 0) && (i_pad_vnni == 1) ) {
    /* m loop header */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
                                     libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
                                     libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_m_loop, 1 );

    /* actual vnni formatting */
    for ( l_n = 0; l_n < 4; ++l_n ) {
      if ( l_n < (i_mateltwise_desc->n % 4) ) {
        libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                         i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->ldi * l_n,
                                         l_gp_temp, 0 );
      } else {
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                         l_gp_temp, 0x0 );
      }

      libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                       i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out * l_n,
                                       l_gp_temp, 1 );
    }

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, i_micro_kernel_config->datatype_size_in );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (long long)i_micro_kernel_config->datatype_size_out * 4 );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, i_mateltwise_desc->m );
  }

  /* restore l_gp_temp */
  libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_temp );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_norm_mbit_scalar_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                            libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                            const unsigned int                      i_gp_reg_in,
                                                                            const unsigned int                      i_gp_reg_out,
                                                                            const unsigned int                      i_gp_reg_m_loop,
                                                                            const unsigned int                      i_gp_reg_n_loop,
                                                                            const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                            const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int l_load_instr  = LIBXSMM_X86_INSTR_UNDEF;
  unsigned int l_store_instr = LIBXSMM_X86_INSTR_UNDEF;
  unsigned int l_gp_temp = LIBXSMM_X86_GP_REG_R15;
  unsigned int l_n = 0;

  /* GP temp registers: check against loop and address registers */
  if ( (l_gp_temp == i_gp_reg_m_loop) || (l_gp_temp == i_gp_reg_n_loop) ||
       (l_gp_temp == i_gp_reg_in)     || (l_gp_temp == i_gp_reg_out) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GP_TEMP_MAPPING );
    return;
  }

  if ( i_mateltwise_desc->n % 4 != 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* select load and store instructions */
  if ( i_micro_kernel_config->datatype_size_in == 8 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVQ;
    l_store_instr = LIBXSMM_X86_INSTR_MOVQ;
  } else if ( i_micro_kernel_config->datatype_size_in == 4 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVD;
    l_store_instr = LIBXSMM_X86_INSTR_MOVD;
  } else if ( i_micro_kernel_config->datatype_size_in == 2 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVW;
    l_store_instr = LIBXSMM_X86_INSTR_MOVW;
  } else if ( i_micro_kernel_config->datatype_size_in == 1 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVB;
    l_store_instr = LIBXSMM_X86_INSTR_MOVB;
  } else {
    /* should not happen */
  }

  /* save l_gp_temp to stack */
  libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_temp );

  /* n loop header */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_n_loop, 4 );

  /* m loop header */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_m_loop, 1 );

  /* actual vnni reverse formatting */
  for ( l_n = 0; l_n < 4; ++l_n ) {
    libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                     i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in * l_n,
                                     l_gp_temp, 0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->ldo * l_n,
                                     l_gp_temp, 1 );
  }

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_in, (long long)i_micro_kernel_config->datatype_size_in * 4 );

  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, i_micro_kernel_config->datatype_size_out );

  /* close m loop */
  libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                              i_gp_reg_m_loop, i_mateltwise_desc->m );

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_in, (4LL * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in) - ((long long)i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->m * 4) );
  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, (4LL * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->m) );

  /* close n loop */
  libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                              i_gp_reg_n_loop, i_mateltwise_desc->n );

  /* restore l_gp_temp */
  libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_temp );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni2_mbit_scalar_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                            libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                            const unsigned int                      i_gp_reg_in,
                                                                            const unsigned int                      i_gp_reg_out,
                                                                            const unsigned int                      i_gp_reg_m_loop,
                                                                            const unsigned int                      i_gp_reg_n_loop,
                                                                            const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                            const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int l_load_instr  = LIBXSMM_X86_INSTR_UNDEF;
  unsigned int l_store_instr = LIBXSMM_X86_INSTR_UNDEF;
  unsigned int l_gp_temp = LIBXSMM_X86_GP_REG_R15;
  unsigned int l_n = 0;

  /* GP temp registers: check against loop and address registers */
  if ( (l_gp_temp == i_gp_reg_m_loop) || (l_gp_temp == i_gp_reg_n_loop) ||
       (l_gp_temp == i_gp_reg_in)     || (l_gp_temp == i_gp_reg_out) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GP_TEMP_MAPPING );
    return;
  }

  if ( i_mateltwise_desc->n % 4 != 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* select load and store instructions */
  if ( i_micro_kernel_config->datatype_size_in == 8 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVQ;
    l_store_instr = LIBXSMM_X86_INSTR_MOVQ;
  } else if ( i_micro_kernel_config->datatype_size_in == 4 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVD;
    l_store_instr = LIBXSMM_X86_INSTR_MOVD;
  } else if ( i_micro_kernel_config->datatype_size_in == 2 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVW;
    l_store_instr = LIBXSMM_X86_INSTR_MOVW;
  } else if ( i_micro_kernel_config->datatype_size_in == 1 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVB;
    l_store_instr = LIBXSMM_X86_INSTR_MOVB;
  } else {
    /* should not happen */
  }

  /* save l_gp_temp to stack */
  libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_temp );

  /* n loop header */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_n_loop, 4 );

  /* m loop header */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_m_loop, 1 );

  /* actual vnni reverse formatting */
  for ( l_n = 0; l_n < 4; ++l_n ) {
    libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                     i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in * l_n,
                                     l_gp_temp, 0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out * (l_n % 2) + i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->ldo * 2 * (l_n / 2),
                                     l_gp_temp, 1 );
  }

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_in, (long long)i_micro_kernel_config->datatype_size_in * 4 );

  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, (long long)i_micro_kernel_config->datatype_size_out * 2 );

  /* close m loop */
  libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                              i_gp_reg_m_loop, i_mateltwise_desc->m );

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_in, (4LL * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in) - ((long long)i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->m * 4) );
  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, (4LL * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->m * 2) );

  /* close n loop */
  libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                              i_gp_reg_n_loop, i_mateltwise_desc->n );

  /* restore l_gp_temp */
  libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_temp );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni4t_mbit_scalar_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                              libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                              const unsigned int                      i_gp_reg_in,
                                                                              const unsigned int                      i_gp_reg_out,
                                                                              const unsigned int                      i_gp_reg_m_loop,
                                                                              const unsigned int                      i_gp_reg_n_loop,
                                                                              const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                              const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int l_load_instr  = LIBXSMM_X86_INSTR_UNDEF;
  unsigned int l_store_instr = LIBXSMM_X86_INSTR_UNDEF;
  unsigned int l_gp_temp = LIBXSMM_X86_GP_REG_R15;
  unsigned int l_ldi = i_mateltwise_desc->ldi*4;
  unsigned int l_ldo = i_mateltwise_desc->ldo*4;

  /* GP temp registers: check against loop and address registers */
  if ( (l_gp_temp == i_gp_reg_m_loop) || (l_gp_temp == i_gp_reg_n_loop) ||
       (l_gp_temp == i_gp_reg_in)     || (l_gp_temp == i_gp_reg_out) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GP_TEMP_MAPPING );
    return;
  }

  /* select load and store instructions */
  if ( i_micro_kernel_config->datatype_size_in == 8 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVQ;
    l_store_instr = LIBXSMM_X86_INSTR_MOVQ;
  } else if ( i_micro_kernel_config->datatype_size_in == 4 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVD;
    l_store_instr = LIBXSMM_X86_INSTR_MOVD;
  } else if ( i_micro_kernel_config->datatype_size_in == 2 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVW;
    l_store_instr = LIBXSMM_X86_INSTR_MOVW;
  } else if ( i_micro_kernel_config->datatype_size_in == 1 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVB;
    l_store_instr = LIBXSMM_X86_INSTR_MOVB;
  } else {
    /* should not happen */
  }

  /* save l_gp_temp to stack */
  libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_temp );

  if ( (i_mateltwise_desc->m % 4 == 0) && (i_mateltwise_desc->n % 4 == 0) ) {
    /* n loop header */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_n_loop, 4 );

    /* m loop header */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_m_loop, 4 );

    /* actual transpose */
    libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                     i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in * 0,
                                     l_gp_temp, 0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out * 0,
                                     l_gp_temp, 1 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                     i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in * 1,
                                     l_gp_temp, 0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out * 4,
                                     l_gp_temp, 1 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                     i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in * 2,
                                     l_gp_temp, 0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out * 8,
                                     l_gp_temp, 1 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                     i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in * 3,
                                     l_gp_temp, 0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out * 12,
                                     l_gp_temp, 1 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                     i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in * 4,
                                     l_gp_temp, 0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out * 1,
                                     l_gp_temp, 1 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                     i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in * 5,
                                     l_gp_temp, 0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out * 5,
                                     l_gp_temp, 1 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                     i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in * 6,
                                     l_gp_temp, 0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out * 9,
                                     l_gp_temp, 1 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                     i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in * 7,
                                     l_gp_temp, 0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out * 13,
                                     l_gp_temp, 1 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                     i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in * 8,
                                     l_gp_temp, 0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out * 2,
                                     l_gp_temp, 1 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                     i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in * 9,
                                     l_gp_temp, 0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out * 6,
                                     l_gp_temp, 1 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                     i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in * 10,
                                     l_gp_temp, 0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out * 10,
                                     l_gp_temp, 1 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                     i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in * 11,
                                     l_gp_temp, 0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out * 14,
                                     l_gp_temp, 1 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                     i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in * 12,
                                     l_gp_temp, 0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out * 3,
                                     l_gp_temp, 1 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                     i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in * 13,
                                     l_gp_temp, 0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out * 7,
                                     l_gp_temp, 1 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                     i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in * 14,
                                     l_gp_temp, 0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out * 11,
                                     l_gp_temp, 1 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                     i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_in * 15,
                                     l_gp_temp, 0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, i_micro_kernel_config->datatype_size_out * 15,
                                     l_gp_temp, 1 );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, (long long)i_micro_kernel_config->datatype_size_in * 16 );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, (long long)i_micro_kernel_config->datatype_size_out * l_ldo );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, i_mateltwise_desc->m );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_in, ((long long)l_ldi * i_micro_kernel_config->datatype_size_in) - ((long long)i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->m * 4) );

    /* advance input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ,
                                     i_gp_reg_out, ((long long)i_micro_kernel_config->datatype_size_out * l_ldo * i_mateltwise_desc->m/4) - ((long long)i_micro_kernel_config->datatype_size_out * 16) );

    /* close n loop */
    libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_n_loop, i_mateltwise_desc->n );
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* restore l_gp_temp */
  libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_temp );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_padnm_mod2_mbit_scalar_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                              libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                              const unsigned int                      i_gp_reg_in,
                                                                              const unsigned int                      i_gp_reg_out,
                                                                              const unsigned int                      i_gp_reg_m_loop,
                                                                              const unsigned int                      i_gp_reg_n_loop,
                                                                              const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                              const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int l_load_instr  = LIBXSMM_X86_INSTR_UNDEF;
  unsigned int l_store_instr = LIBXSMM_X86_INSTR_UNDEF;
  unsigned int l_gp_temp = LIBXSMM_X86_GP_REG_R15;

  /* GP temp registers: check against loop and address registers */
  if ( (l_gp_temp == i_gp_reg_m_loop) || (l_gp_temp == i_gp_reg_n_loop) ||
       (l_gp_temp == i_gp_reg_in)     || (l_gp_temp == i_gp_reg_out) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GP_TEMP_MAPPING );
    return;
  }

  /* select load and store instructions */
  if ( i_micro_kernel_config->datatype_size_in == 8 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVQ;
    l_store_instr = LIBXSMM_X86_INSTR_MOVQ;
  } else if ( i_micro_kernel_config->datatype_size_in == 4 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVD;
    l_store_instr = LIBXSMM_X86_INSTR_MOVD;
  } else if ( i_micro_kernel_config->datatype_size_in == 2 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVW;
    l_store_instr = LIBXSMM_X86_INSTR_MOVW;
  } else if ( i_micro_kernel_config->datatype_size_in == 1 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVB;
    l_store_instr = LIBXSMM_X86_INSTR_MOVB;
  } else {
    /* should not happen */
  }

  /* save l_gp_temp to stack */
  libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_temp );

  /* n loop header */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_n_loop, 1 );

  /* m loop header */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_m_loop, 1 );

  /* actual copy / padding */
  libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                   i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                   l_gp_temp, 0 );

  libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                   i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                   l_gp_temp, 1 );

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_in, i_micro_kernel_config->datatype_size_in );

  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, i_micro_kernel_config->datatype_size_out );

  /* close m loop */
  libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                              i_gp_reg_m_loop, i_mateltwise_desc->m );

  /* pad in M dimension during regular N loop */
  if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD2 ||
        i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2) &&
       (i_mateltwise_desc->m % 2 == 1) ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     l_gp_temp, 0x0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                     l_gp_temp, 1 );
  }

  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_in, ((long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in) - ((long long)i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->m) );

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, ((long long)i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->m) );

  /* close n loop */
  libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                              i_gp_reg_n_loop, i_mateltwise_desc->n );

  /* pad in M dimension during regular N loop */
  if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD2 ||
        i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2) &&
        (i_mateltwise_desc->n % 2 == 1) ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     l_gp_temp, 0x0 );

    /* m loop header */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_m_loop, 1 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                     l_gp_temp, 1 );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, i_micro_kernel_config->datatype_size_out );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, i_mateltwise_desc->m );

    /* pad in M dimension during regular N loop */
    if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2) &&
         (i_mateltwise_desc->m % 2 == 1) ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                       l_gp_temp, 0x0 );

      libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                       i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                       l_gp_temp, 1 );
    }
  }

  /* restore l_gp_temp */
  libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_temp );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_padnm_mod4_mbit_scalar_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                              libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                              const unsigned int                      i_gp_reg_in,
                                                                              const unsigned int                      i_gp_reg_out,
                                                                              const unsigned int                      i_gp_reg_m_loop,
                                                                              const unsigned int                      i_gp_reg_n_loop,
                                                                              const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                              const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int l_load_instr  = LIBXSMM_X86_INSTR_UNDEF;
  unsigned int l_store_instr = LIBXSMM_X86_INSTR_UNDEF;
  unsigned int l_gp_temp = LIBXSMM_X86_GP_REG_R15;
  unsigned int l_gp_out_fixup = 0;

  /* GP temp registers: check against loop and address registers */
  if ( (l_gp_temp == i_gp_reg_m_loop) || (l_gp_temp == i_gp_reg_n_loop) ||
       (l_gp_temp == i_gp_reg_in)     || (l_gp_temp == i_gp_reg_out) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GP_TEMP_MAPPING );
    return;
  }

  /* select load and store instructions */
  if ( i_micro_kernel_config->datatype_size_in == 8 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVQ;
    l_store_instr = LIBXSMM_X86_INSTR_MOVQ;
  } else if ( i_micro_kernel_config->datatype_size_in == 4 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVD;
    l_store_instr = LIBXSMM_X86_INSTR_MOVD;
  } else if ( i_micro_kernel_config->datatype_size_in == 2 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVW;
    l_store_instr = LIBXSMM_X86_INSTR_MOVW;
  } else if ( i_micro_kernel_config->datatype_size_in == 1 ) {
    l_load_instr = LIBXSMM_X86_INSTR_MOVB;
    l_store_instr = LIBXSMM_X86_INSTR_MOVB;
  } else {
    /* should not happen */
  }

  /* save l_gp_temp to stack */
  libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_temp );

  /* n loop header */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_n_loop, 1 );

  /* m loop header */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_m_loop, 1 );

  /* actual copy / padding */
  libxsmm_x86_instruction_alu_mem( io_generated_code, l_load_instr,
                                   i_gp_reg_in, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                   l_gp_temp, 0 );

  libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                   i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                   l_gp_temp, 1 );

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_in, i_micro_kernel_config->datatype_size_in );

  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, i_micro_kernel_config->datatype_size_out );

  /* close m loop */
  libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                              i_gp_reg_m_loop, i_mateltwise_desc->m );

  /* pad in M dimension during regular N loop */
  if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD4 ||
        i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD4) &&
       (i_mateltwise_desc->m % 4 != 0) ) {
    unsigned int l_m = 0;

    l_gp_out_fixup = LIBXSMM_UP(i_mateltwise_desc->m, 4);

    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     l_gp_temp, 0x0 );

    for ( l_m = (i_mateltwise_desc->m % 4); l_m < 4; ++l_m ) {
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                       i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                       l_gp_temp, 1 );

      /* advance output pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                       i_gp_reg_out, i_micro_kernel_config->datatype_size_out );
    }
  } else {
    l_gp_out_fixup = i_mateltwise_desc->m;
  }

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_in, ((long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in) - ((long long)i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->m) );

  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, ((long long)i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_out * l_gp_out_fixup) );

  /* close n loop */
  libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                              i_gp_reg_n_loop, i_mateltwise_desc->n );

  /* pad in M dimension during regular N loop */
  if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD4 ||
        i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD4) &&
        (i_mateltwise_desc->n % 4 != 0) ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                     l_gp_temp, 0x0 );

    /* n loop header */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_n_loop, 1 );

    /* m loop header */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_m_loop, 1 );

    libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                     i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                     l_gp_temp, 1 );

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                     i_gp_reg_out, i_micro_kernel_config->datatype_size_out );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, i_mateltwise_desc->m );

    /* pad in M dimension during regular N loop */
    if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD4) &&
         (i_mateltwise_desc->m % 4 != 0) ) {
      unsigned int l_m = 0;

      l_gp_out_fixup = LIBXSMM_UP(i_mateltwise_desc->m, 4);

      for ( l_m = (i_mateltwise_desc->m % 4); l_m < 4; ++l_m ) {
        libxsmm_x86_instruction_alu_mem( io_generated_code, l_store_instr,
                                         i_gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                         l_gp_temp, 1 );

        /* advance output pointer */
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                         i_gp_reg_out, i_micro_kernel_config->datatype_size_out );
      }
    } else {
      l_gp_out_fixup = i_mateltwise_desc->m;
    }

    /* advance output pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, ((long long)i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_out * l_gp_out_fixup) );

    /* close n loop */
    libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_n_loop, 4 - (i_mateltwise_desc->n % 4) );
  }

  /* restore l_gp_temp */
  libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_temp );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_64bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
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
void libxsmm_generator_transform_norm_to_normt_32bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
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
void libxsmm_generator_transform_norm_to_normt_16bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
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
void libxsmm_generator_transform_norm_to_normt_08bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
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
void libxsmm_generator_transform_norm_to_vnni2_16bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
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
void libxsmm_generator_transform_vnni2_to_vnni2t_16bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
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
void libxsmm_generator_transform_norm_to_vnni4_16bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
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
void libxsmm_generator_transform_norm_to_vnni4_08bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
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
void libxsmm_generator_transform_vnni4_to_norm_08bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  libxsmm_generator_transform_vnni4_to_norm_mbit_scalar_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                         i_micro_kernel_config, i_mateltwise_desc);
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni2_08bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  libxsmm_generator_transform_vnni4_to_vnni2_mbit_scalar_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                         i_micro_kernel_config, i_mateltwise_desc);
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni4t_16bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
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
void libxsmm_generator_transform_vnni4_to_vnni4t_08bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
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
void libxsmm_generator_transform_norm_padnm_mod2_16bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
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
void libxsmm_generator_transform_norm_padnm_mod4_08bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
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
void libxsmm_generator_transform_sse_microkernel( libxsmm_generated_code*                        io_generated_code,
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
              (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4)     ||
              (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4_PAD) ||
              (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD2)         ||
              (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD2)         ||
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
      libxsmm_generator_transform_norm_to_normt_64bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else if ( ( LIBXSMM_DATATYPE_I32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
                LIBXSMM_DATATYPE_I32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) ||
              ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
                LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) ) {
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) {
      libxsmm_generator_transform_norm_to_normt_32bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else if ( ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
                LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) ||
              ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
                LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) ||
              ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
                LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) )   ) {
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) {
      libxsmm_generator_transform_norm_to_normt_16bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc );
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4T ) {
      /* Call 64bit normal transpose */
      libxsmm_descriptor_blob blob;
      const libxsmm_meltw_descriptor *const mock_desc = libxsmm_meltw_descriptor_init2(&blob,
        LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, i_mateltwise_desc->m/4, i_mateltwise_desc->n,
        i_mateltwise_desc->ldi/4, i_mateltwise_desc->ldo, 0, 0,
        (unsigned short)LIBXSMM_MELTW_FLAG_UNARY_NONE, (unsigned short)LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, LIBXSMM_MELTW_OPERATION_UNARY);
      libxsmm_mateltwise_kernel_config l_trans_config;
      libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_trans_config, mock_desc);
      libxsmm_generator_transform_norm_to_normt_64bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
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
      libxsmm_generator_transform_norm_to_normt_32bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
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
      libxsmm_generator_transform_norm_to_normt_64bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
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
      libxsmm_generator_transform_norm_to_normt_32bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                 l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                 &l_trans_config, mock_desc );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2_TO_VNNI2T) {
      libxsmm_generator_transform_vnni2_to_vnni2t_16bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                         i_micro_kernel_config, i_mateltwise_desc );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI4T) {
      libxsmm_generator_transform_vnni4_to_vnni4t_16bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                         i_micro_kernel_config, i_mateltwise_desc );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2) {
      libxsmm_generator_transform_norm_to_vnni2_16bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc, 0 );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD) {
      libxsmm_generator_transform_norm_to_vnni2_16bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc, 1 );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4) {
      libxsmm_generator_transform_norm_to_vnni4_16bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc, 0 );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4_PAD) {
      libxsmm_generator_transform_norm_to_vnni4_16bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc, 1 );
    } else if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD2)  ||
                (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD2)  ||
                (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2)    ) {
      libxsmm_generator_transform_norm_padnm_mod2_16bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                         i_micro_kernel_config, i_mateltwise_desc );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else if ( ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
                LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) ||
              ( LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
                LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) ||
              ( LIBXSMM_DATATYPE_HF8 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
                LIBXSMM_DATATYPE_HF8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) ) {
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) {
      libxsmm_generator_transform_norm_to_normt_08bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI4T) {
      libxsmm_generator_transform_vnni4_to_vnni4t_08bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                         l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                         i_micro_kernel_config, i_mateltwise_desc );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_NORM) {
      libxsmm_generator_transform_vnni4_to_norm_08bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc);
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI2) {
      libxsmm_generator_transform_vnni4_to_vnni2_08bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc);
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4) {
      libxsmm_generator_transform_norm_to_vnni4_08bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc, 0 );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4_PAD) {
      libxsmm_generator_transform_norm_to_vnni4_08bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc, 1 );
    } else if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD4)  ||
                (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD4)  ||
                (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD4)    ) {
      libxsmm_generator_transform_norm_padnm_mod4_08bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
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

