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

#include "generator_mateltwise_transform_aarch64_asimd.h"
#include "generator_common_aarch64.h"
#include "generator_aarch64_instructions.h"
#include "generator_common.h"
#include "libxsmm_main.h"

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
void libxsmm_generator_transform_norm_to_vnni_mbit_scalar_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
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
void libxsmm_generator_transform_vnni_to_vnnit_mbit_scalar_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
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
  libxsmm_generator_transform_norm_to_normt_mbit_scalar_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                   i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                                   i_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc );
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
  libxsmm_generator_transform_norm_to_normt_mbit_scalar_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                   i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                                   i_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc );
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
void libxsmm_generator_transform_norm_to_vnni_16bit_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                               libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                               const unsigned int                      i_gp_reg_in,
                                                                               const unsigned int                      i_gp_reg_out,
                                                                               const unsigned int                      i_gp_reg_m_loop,
                                                                               const unsigned int                      i_gp_reg_n_loop,
                                                                               const unsigned int                      i_gp_reg_scratch,
                                                                               const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                               const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                               const unsigned int                      i_pad_vnni ) {
  libxsmm_generator_transform_norm_to_vnni_mbit_scalar_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                  i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                                  i_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc, i_pad_vnni );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni_to_vnnit_16bit_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                const unsigned int                      i_gp_reg_in,
                                                                                const unsigned int                      i_gp_reg_out,
                                                                                const unsigned int                      i_gp_reg_m_loop,
                                                                                const unsigned int                      i_gp_reg_n_loop,
                                                                                const unsigned int                      i_gp_reg_scratch,
                                                                                const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  libxsmm_generator_transform_vnni_to_vnnit_mbit_scalar_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
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
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 56,
                                        i_gp_reg_mapping->gp_reg_out );

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
      libxsmm_generator_transform_norm_to_normt_64bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else if ( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
              LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) {
      libxsmm_generator_transform_norm_to_normt_32bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc );
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
      libxsmm_generator_transform_norm_to_normt_16bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI_TO_VNNIT) {
      libxsmm_generator_transform_vnni_to_vnnit_16bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI) {
      libxsmm_generator_transform_norm_to_vnni_16bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc, 0 );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI_PAD) {
      libxsmm_generator_transform_norm_to_vnni_16bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
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
  } else if ( LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) &&
              LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) {
      libxsmm_generator_transform_norm_to_normt_08bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
}

