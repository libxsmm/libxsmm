/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include "generator_mateltwise_sse_avx_avx512.h"
#include "generator_mateltwise_transform_sse.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "libxsmm_main.h"

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

  /* restore l_gp_temp */
  libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_temp );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni_mbit_scalar_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
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
                                     i_gp_reg_out, i_micro_kernel_config->datatype_size_out * 2 );

    /* close m loop */
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config,
                                                i_gp_reg_m_loop, i_mateltwise_desc->m );
  }

  /* restore l_gp_temp */
  libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_temp );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni_to_vnnit_mbit_scalar_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
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
                                   i_gp_reg_in, (i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in) - (i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->m) );

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ,
                                   i_gp_reg_out, (i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - (i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->m) );

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
void libxsmm_generator_transform_norm_to_vnni_16bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                     libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                     const unsigned int                      i_gp_reg_in,
                                                                     const unsigned int                      i_gp_reg_out,
                                                                     const unsigned int                      i_gp_reg_m_loop,
                                                                     const unsigned int                      i_gp_reg_n_loop,
                                                                     const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                     const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                     const unsigned int                      i_pad_vnni ) {
  libxsmm_generator_transform_norm_to_vnni_mbit_scalar_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                        i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                        i_micro_kernel_config, i_mateltwise_desc, i_pad_vnni );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni_to_vnnit_16bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  libxsmm_generator_transform_vnni_to_vnnit_mbit_scalar_sse_microkernel( io_generated_code, io_loop_label_tracker,
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
                                   LIBXSMM_X86_GP_REG_UNDEF, 0, 56,
                                   l_gp_reg_out, 0 );

  /* check leading dimnesions and sizes */
  if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) ||
       (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI_TO_VNNIT)    ) {
    /* coverity[copy_paste_error] */
    if ( (i_mateltwise_desc->m > i_mateltwise_desc->ldi) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
      return;
    }
    if ( (i_mateltwise_desc->n > i_mateltwise_desc->ldo) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB );
      return;
    }
  } else if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI)     ||
              (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI_PAD) ||
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

  if ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
       LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) {
      libxsmm_generator_transform_norm_to_normt_64bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else if ( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
              LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) {
      libxsmm_generator_transform_norm_to_normt_32bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else if ( (LIBXSMM_GEMM_PRECISION_I16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )  &&
               LIBXSMM_GEMM_PRECISION_I16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) ||
              (LIBXSMM_GEMM_PRECISION_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )  &&
               LIBXSMM_GEMM_PRECISION_F16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) ||
              (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
               LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ))   ) {
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) {
      libxsmm_generator_transform_norm_to_normt_16bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI_TO_VNNIT) {
      libxsmm_generator_transform_vnni_to_vnnit_16bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                       l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                       i_micro_kernel_config, i_mateltwise_desc );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI) {
      libxsmm_generator_transform_norm_to_vnni_16bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
                                                                      l_gp_reg_in, l_gp_reg_out, l_gp_reg_mloop, l_gp_reg_nloop,
                                                                      i_micro_kernel_config, i_mateltwise_desc, 0 );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI_PAD) {
      libxsmm_generator_transform_norm_to_vnni_16bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
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
  } else if ( LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
              LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) {
      libxsmm_generator_transform_norm_to_normt_08bit_sse_microkernel( io_generated_code, io_loop_label_tracker,
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

