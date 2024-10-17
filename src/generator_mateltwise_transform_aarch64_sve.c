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
#include "generator_mateltwise_transform_aarch64_sve.h"
#include "generator_common_aarch64.h"
#include "generator_aarch64_instructions.h"
#include "generator_common.h"
#include "generator_mateltwise_aarch64.h"


LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni2_16bit_aarch64_sve_microkernel_block( libxsmm_generated_code*                 io_generated_code,
                                                                                    libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                    const unsigned int                      i_gp_reg_in,
                                                                                    const unsigned int                      i_gp_reg_out,
                                                                                    const unsigned int                      i_gp_reg_m_loop,
                                                                                    const unsigned int                      i_gp_reg_n_loop,
                                                                                    const unsigned int                      i_gp_reg_scratch,
                                                                                    const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                    const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                                    unsigned int                            i_m_remainder,
                                                                                    unsigned int                            i_m_mask_in,
                                                                                    unsigned int                            i_m_mask_out,
                                                                                    unsigned int                            i_n_padding_vregs ) {
  libxsmm_aarch64_sve_type l_sve_type2 = LIBXSMM_AARCH64_SVE_TYPE_H;
  unsigned int l_n = 0;
  unsigned int l_load_instruction = ( i_m_remainder == 0 ) ? LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF : LIBXSMM_AARCH64_INSTR_SVE_LD1H_I_OFF;
  unsigned int l_load_mask = ( i_m_remainder == 0 ) ? LIBXSMM_AARCH64_SVE_REG_UNDEF : i_m_mask_in;

  LIBXSMM_UNUSED( io_loop_label_tracker );
  LIBXSMM_UNUSED( i_gp_reg_m_loop );
  LIBXSMM_UNUSED( i_gp_reg_n_loop );

  for (l_n = 0; l_n < 2 - i_n_padding_vregs; l_n++) {
    libxsmm_aarch64_instruction_sve_move( io_generated_code, l_load_instruction,
                                          i_gp_reg_in, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, l_n, l_load_mask );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, (long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in );
  }
  for (l_n = 2 - i_n_padding_vregs; l_n < 2 ; l_n++) {
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V, l_n, l_n, 0, l_n, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
  }
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, (2LL - i_n_padding_vregs) * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP1_V, 0, 1, 0, 2, 0, l_sve_type2 );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP2_V, 0, 1, 0, 3, 0, l_sve_type2 );

  if (i_m_remainder == 0) {
    libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                          i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 2, LIBXSMM_AARCH64_SVE_REG_UNDEF );
    libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                          i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 1, 3, LIBXSMM_AARCH64_SVE_REG_UNDEF );
  } else {
    if (i_m_remainder <= 7) {
      /* 1 masked store*/
      libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ST1H_I_OFF,
                                            i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 2, i_m_mask_out );
    } else if (i_m_remainder > 7 && i_m_remainder <= 15){
      /* 1 full store, 1 masked_store (potentially) */
      libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                            i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 2, LIBXSMM_AARCH64_SVE_REG_UNDEF );
      if (i_m_remainder != 8) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, 16LL * i_micro_kernel_config->datatype_size_out );
        libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ST1H_I_OFF,
                                              i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 3, i_m_mask_out );
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                       i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, 16LL * i_micro_kernel_config->datatype_size_out );
      }
    }
  }
}


/* @TODO: check if this code can be joined with the vnni4 code */
/* @TODO: cehck if this code can follow the some logic as the x86 counterpart */
LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni2_16bit_aarch64_sve_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                              libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                              const unsigned int                      i_gp_reg_in,
                                                                              const unsigned int                      i_gp_reg_out,
                                                                              const unsigned int                      i_gp_reg_m_loop,
                                                                              const unsigned int                      i_gp_reg_n_loop,
                                                                              const unsigned int                      i_gp_reg_scratch,
                                                                              const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                              const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                              const unsigned int                      i_pad_vnni ) {
  unsigned int l_m_remainder = i_mateltwise_desc->m % 16;
  unsigned int l_n_remainder = i_mateltwise_desc->n % 2;
  unsigned int l_inp_mask = 1;
  unsigned int l_out_mask = 2;

  if ( (i_pad_vnni == 0) && (l_n_remainder != 0) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  if (l_m_remainder > 0) {
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, l_inp_mask, l_m_remainder * i_micro_kernel_config->datatype_size_in, i_gp_reg_scratch );
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, l_out_mask, (l_m_remainder%8) * 2 * i_micro_kernel_config->datatype_size_out, i_gp_reg_scratch );
  }

  if ( i_mateltwise_desc->m >= 16) {
    unsigned int l_effective_m = (i_mateltwise_desc->m/16)*16;
    if (i_mateltwise_desc->n >= 2) {
      /* n loop header */
      libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, (i_mateltwise_desc->n/2)*2);

      /* m loop header */
      libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, l_effective_m );

      libxsmm_generator_transform_norm_to_vnni2_16bit_aarch64_sve_microkernel_block( io_generated_code, io_loop_label_tracker, i_gp_reg_in, i_gp_reg_out,
          i_gp_reg_m_loop, i_gp_reg_n_loop, i_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc, 0, 0, 0, 0 );

      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, 16LL * i_micro_kernel_config->datatype_size_in );

      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, 32LL * i_micro_kernel_config->datatype_size_out );

      /* close m loop */
      libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, 16 );

      if ((2 * i_mateltwise_desc->ldi) > l_effective_m) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, (2LL * i_mateltwise_desc->ldi - l_effective_m) * i_micro_kernel_config->datatype_size_in );
      }

      if (i_mateltwise_desc->ldo > l_effective_m) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, (2LL * i_mateltwise_desc->ldo - 2LL * l_effective_m) * i_micro_kernel_config->datatype_size_out );
      }
      /* close n loop */
      libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, 2 );
    }
    if (l_n_remainder != 0) {
      /* m loop header */
      libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, l_effective_m );

      libxsmm_generator_transform_norm_to_vnni2_16bit_aarch64_sve_microkernel_block( io_generated_code, io_loop_label_tracker, i_gp_reg_in, i_gp_reg_out,
          i_gp_reg_m_loop, i_gp_reg_n_loop, i_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc, 0, 0, 0, 2-l_n_remainder );

      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, 16LL * i_micro_kernel_config->datatype_size_in );

      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, 32LL * i_micro_kernel_config->datatype_size_out );
      /* close m loop */
      libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, 16 );
    }
  }
  if (l_m_remainder > 0) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, LIBXSMM_AARCH64_GP_REG_X0, LIBXSMM_AARCH64_GP_REG_UNDEF, 32,
                                          i_gp_reg_in );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, LIBXSMM_AARCH64_GP_REG_X0, LIBXSMM_AARCH64_GP_REG_UNDEF, 64,
                                          i_gp_reg_out );
    if ((i_mateltwise_desc->m/16) > 0) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, ((long long)i_mateltwise_desc->m/16) * 16 * i_micro_kernel_config->datatype_size_in );
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, ((long long)i_mateltwise_desc->m/16) * 16 * 2 * i_micro_kernel_config->datatype_size_out );
    }

    if (i_mateltwise_desc->n >= 2) {
      /* n loop header */
      libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, (i_mateltwise_desc->n/2)*2);

      libxsmm_generator_transform_norm_to_vnni2_16bit_aarch64_sve_microkernel_block( io_generated_code, io_loop_label_tracker, i_gp_reg_in, i_gp_reg_out,
          i_gp_reg_m_loop, i_gp_reg_n_loop, i_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc, l_m_remainder, l_inp_mask, l_out_mask, 0 );

      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, (2LL *i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in );

      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, (2LL *i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out );
      /* close n loop */
      libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, 2 );
    }
    if (l_n_remainder != 0) {
      libxsmm_generator_transform_norm_to_vnni2_16bit_aarch64_sve_microkernel_block( io_generated_code, io_loop_label_tracker, i_gp_reg_in, i_gp_reg_out,
          i_gp_reg_m_loop, i_gp_reg_n_loop, i_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc, l_m_remainder, l_inp_mask, l_out_mask, 2-l_n_remainder );
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni4_16bit_aarch64_sve_microkernel_block( libxsmm_generated_code*                 io_generated_code,
                                                                                    libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                    const unsigned int                      i_gp_reg_in,
                                                                                    const unsigned int                      i_gp_reg_out,
                                                                                    const unsigned int                      i_gp_reg_m_loop,
                                                                                    const unsigned int                      i_gp_reg_n_loop,
                                                                                    const unsigned int                      i_gp_reg_scratch,
                                                                                    const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                    const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                                    unsigned int                            i_m_remainder,
                                                                                    unsigned int                            i_m_mask_in,
                                                                                    unsigned int                            i_m_mask_out,
                                                                                    unsigned int                            i_n_padding_vregs ) {
  libxsmm_aarch64_sve_type l_sve_type4 = libxsmm_generator_aarch64_get_sve_type(LIBXSMM_CAST_UCHAR(4));
  libxsmm_aarch64_sve_type l_sve_type2 = libxsmm_generator_aarch64_get_sve_type(LIBXSMM_CAST_UCHAR(2));
  unsigned int l_n = 0;

  LIBXSMM_UNUSED( io_loop_label_tracker );
  LIBXSMM_UNUSED( i_gp_reg_m_loop );
  LIBXSMM_UNUSED( i_gp_reg_n_loop );

  if (i_m_remainder == 0) {
    for (l_n = 0; l_n < 4 - i_n_padding_vregs; l_n++) {
      libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF,
                                            i_gp_reg_in, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, l_n, LIBXSMM_AARCH64_SVE_REG_UNDEF );
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, (long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in );
    }
    for (l_n = 4 - i_n_padding_vregs; l_n < 4 ; l_n++) {
      libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V, l_n, l_n, 0, l_n, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
    }
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, (4LL - i_n_padding_vregs) * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in );
  } else {
    for (l_n = 0; l_n < 4 - i_n_padding_vregs; l_n++) {
      libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1H_I_OFF,
                                            i_gp_reg_in, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, l_n, i_m_mask_in );
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, (long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in );
    }
    for (l_n = 4 - i_n_padding_vregs; l_n < 4 ; l_n++) {
      libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V, l_n, l_n, 0, l_n, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
    }
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, (4LL - i_n_padding_vregs) * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in );
  }

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP1_V, 0, 1, 0, 4, 0, l_sve_type2 );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP1_V, 2, 3, 0, 5, 0, l_sve_type2 );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP2_V, 0, 1, 0, 6, 0, l_sve_type2 );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP2_V, 2, 3, 0, 7, 0, l_sve_type2 );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP1_V, 4, 5, 0, 8, 0, l_sve_type4 ); /* M0 - M3 [N0N1N2N3] */
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP2_V, 4, 5, 0, 9, 0, l_sve_type4 ); /* M4 - M7 [N0N1N2N3] */
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP1_V, 6, 7, 0, 10, 0, l_sve_type4 ); /* M8 - M11[N0N1N2N3] */
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP2_V, 6, 7, 0, 11, 0, l_sve_type4 ); /* M12 -M15[N0N1N2N3] */

  if (i_m_remainder == 0) {
    libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                          i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 8, LIBXSMM_AARCH64_SVE_REG_UNDEF );
    libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                          i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 1, 9, LIBXSMM_AARCH64_SVE_REG_UNDEF );
    libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                          i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 2, 10, LIBXSMM_AARCH64_SVE_REG_UNDEF );
    libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                          i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 3, 11, LIBXSMM_AARCH64_SVE_REG_UNDEF );
  } else {
    if (i_m_remainder <= 3) {
      /* 1 masked store*/
      libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ST1H_I_OFF,
                                            i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 8, i_m_mask_out );
    } else if (i_m_remainder > 3 && i_m_remainder <= 7){
      /* 1 full store, 1 masked_store (potentially) */
      libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                            i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 8, LIBXSMM_AARCH64_SVE_REG_UNDEF );
      if (i_m_remainder != 4) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, 16LL * i_micro_kernel_config->datatype_size_out );
        libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ST1H_I_OFF,
                                              i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 9, i_m_mask_out );
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                       i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, 16LL * i_micro_kernel_config->datatype_size_out );
      }
    } else if (i_m_remainder > 7 && i_m_remainder <= 11){
      /* 2 full store, 1 masked_store (potentially) */
      libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                            i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 8, LIBXSMM_AARCH64_SVE_REG_UNDEF );
      libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                            i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 1, 9, LIBXSMM_AARCH64_SVE_REG_UNDEF );
      if (i_m_remainder != 8) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, 32LL * i_micro_kernel_config->datatype_size_out );
        libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ST1H_I_OFF,
                                              i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 10, i_m_mask_out );
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                       i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, 32LL * i_micro_kernel_config->datatype_size_out );
      }
    } else {
      /* 3 full stores, 1 masked store (potentially) */
      libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                            i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 8, LIBXSMM_AARCH64_SVE_REG_UNDEF );
      libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                            i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 1, 9, LIBXSMM_AARCH64_SVE_REG_UNDEF );
      libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                            i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 2, 10, LIBXSMM_AARCH64_SVE_REG_UNDEF );
      if (i_m_remainder != 12) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, 48LL * i_micro_kernel_config->datatype_size_out );
        libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ST1H_I_OFF,
                                              i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 11, i_m_mask_out );
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                              i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, 48LL * i_micro_kernel_config->datatype_size_out );
      }
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni4_16bit_aarch64_sve_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                              libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                              const unsigned int                      i_gp_reg_in,
                                                                              const unsigned int                      i_gp_reg_out,
                                                                              const unsigned int                      i_gp_reg_m_loop,
                                                                              const unsigned int                      i_gp_reg_n_loop,
                                                                              const unsigned int                      i_gp_reg_scratch,
                                                                              const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                              const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                              const unsigned int                      i_pad_vnni ) {
  unsigned int l_m_remainder = i_mateltwise_desc->m % 16;
  unsigned int l_n_remainder = i_mateltwise_desc->n % 4;
  unsigned int l_inp_mask = 1;
  unsigned int l_out_mask = 2;

  if ( (i_pad_vnni == 0) && (l_n_remainder != 0) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  if (l_m_remainder > 0) {
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, l_inp_mask, l_m_remainder * i_micro_kernel_config->datatype_size_in, i_gp_reg_scratch );
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, l_out_mask, (l_m_remainder%4) * 4 * i_micro_kernel_config->datatype_size_out, i_gp_reg_scratch );
  }

  if ( i_mateltwise_desc->m >= 16) {
    unsigned int l_effective_m = (i_mateltwise_desc->m/16)*16;
    if (i_mateltwise_desc->n >= 4) {
      /* n loop header */
      libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, (i_mateltwise_desc->n/4)*4);

      /* m loop header */
      libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, l_effective_m );

      libxsmm_generator_transform_norm_to_vnni4_16bit_aarch64_sve_microkernel_block( io_generated_code, io_loop_label_tracker, i_gp_reg_in, i_gp_reg_out,
          i_gp_reg_m_loop, i_gp_reg_n_loop, i_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc, 0, 0, 0, 0 );

      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, 16LL * i_micro_kernel_config->datatype_size_in );

      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, 64LL * i_micro_kernel_config->datatype_size_out );

      /* close m loop */
      libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, 16 );

      if ((4 * i_mateltwise_desc->ldi) > l_effective_m) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, (4LL * i_mateltwise_desc->ldi - l_effective_m) * i_micro_kernel_config->datatype_size_in );
      }

      if (i_mateltwise_desc->ldo > l_effective_m) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, (4LL * i_mateltwise_desc->ldo - 4LL * l_effective_m) * i_micro_kernel_config->datatype_size_out );
      }
      /* close n loop */
      libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, 4 );
    }
    if (l_n_remainder != 0) {
      /* m loop header */
      libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, l_effective_m );

      libxsmm_generator_transform_norm_to_vnni4_16bit_aarch64_sve_microkernel_block( io_generated_code, io_loop_label_tracker, i_gp_reg_in, i_gp_reg_out,
          i_gp_reg_m_loop, i_gp_reg_n_loop, i_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc, 0, 0, 0, 4-l_n_remainder );

      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, 16LL * i_micro_kernel_config->datatype_size_in );

      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, 64LL * i_micro_kernel_config->datatype_size_out );
      /* close m loop */
      libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, 16 );
    }
  }
  if (l_m_remainder > 0) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, LIBXSMM_AARCH64_GP_REG_X0, LIBXSMM_AARCH64_GP_REG_UNDEF, 32,
                                          i_gp_reg_in );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, LIBXSMM_AARCH64_GP_REG_X0, LIBXSMM_AARCH64_GP_REG_UNDEF, 64,
                                          i_gp_reg_out );
    if ((i_mateltwise_desc->m/16) > 0) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, ((long long)i_mateltwise_desc->m/16) * 16 * i_micro_kernel_config->datatype_size_in );
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, ((long long)i_mateltwise_desc->m/16) * 16 * 4 * i_micro_kernel_config->datatype_size_out );
    }

    if (i_mateltwise_desc->n >= 4) {
      /* n loop header */
      libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, (i_mateltwise_desc->n/4)*4);

      libxsmm_generator_transform_norm_to_vnni4_16bit_aarch64_sve_microkernel_block( io_generated_code, io_loop_label_tracker, i_gp_reg_in, i_gp_reg_out,
          i_gp_reg_m_loop, i_gp_reg_n_loop, i_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc, l_m_remainder, l_inp_mask, l_out_mask, 0 );

      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, (4LL *i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in );

      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, (4LL *i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out );
      /* close n loop */
      libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, 4 );
    }
    if (l_n_remainder != 0) {
      libxsmm_generator_transform_norm_to_vnni4_16bit_aarch64_sve_microkernel_block( io_generated_code, io_loop_label_tracker, i_gp_reg_in, i_gp_reg_out,
          i_gp_reg_m_loop, i_gp_reg_n_loop, i_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc, l_m_remainder, l_inp_mask, l_out_mask, 4-l_n_remainder );
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni2_to_vnni2t_Nmod8_16bit_aarch64_sve_microkernel( libxsmm_generated_code*                 io_generated_code,
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
  libxsmm_aarch64_sve_type l_sve_type8 = libxsmm_generator_aarch64_get_sve_type(LIBXSMM_CAST_UCHAR(8));
  libxsmm_aarch64_sve_type l_sve_type4 = libxsmm_generator_aarch64_get_sve_type(LIBXSMM_CAST_UCHAR(4));
  libxsmm_aarch64_sve_type l_sve_type2 = libxsmm_generator_aarch64_get_sve_type(LIBXSMM_CAST_UCHAR(2));

  /* n loop header */
  libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, i_mateltwise_desc->n );

  /* m loop header */
  libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, i_mateltwise_desc->m );


  libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF,
                                        i_gp_reg_in, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 0, LIBXSMM_AARCH64_SVE_REG_UNDEF );
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, (long long)l_ldi * i_micro_kernel_config->datatype_size_in );
  libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF,
                                        i_gp_reg_in, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 1, LIBXSMM_AARCH64_SVE_REG_UNDEF );
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, (long long)l_ldi * i_micro_kernel_config->datatype_size_in );
  libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF,
                                        i_gp_reg_in, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 2, LIBXSMM_AARCH64_SVE_REG_UNDEF );
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, (long long)l_ldi * i_micro_kernel_config->datatype_size_in );
  libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF,
                                        i_gp_reg_in, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 3, LIBXSMM_AARCH64_SVE_REG_UNDEF );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TRN1_V, 0, 1, 0,  4, 0, l_sve_type4 );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TRN2_V, 0, 1, 0,  5, 0, l_sve_type4 );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TRN1_V, 2, 3, 0,  6, 0, l_sve_type4 );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TRN2_V, 2, 3, 0,  7, 0, l_sve_type4 );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP1_V, 4, 6, 0,  8, 0, l_sve_type8 );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP2_V, 4, 6, 0,  9, 0, l_sve_type8 );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP1_V, 5, 7, 0, 10, 0, l_sve_type8 );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP2_V, 5, 7, 0, 11, 0, l_sve_type8 );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP1_V, 8, 10, 0,  0, 0, l_sve_type2 );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP2_V, 8, 10, 0,  1, 0, l_sve_type2 );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP1_V, 9, 11, 0,  2, 0, l_sve_type2 );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP2_V, 9, 11, 0,  3, 0, l_sve_type2 );

  libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                        i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 0, LIBXSMM_AARCH64_SVE_REG_UNDEF );
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, (long long)l_ldo * i_micro_kernel_config->datatype_size_out );
  libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                        i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 1, LIBXSMM_AARCH64_SVE_REG_UNDEF );
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, (long long)l_ldo * i_micro_kernel_config->datatype_size_out );
  libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                        i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 2, LIBXSMM_AARCH64_SVE_REG_UNDEF );
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, (long long)l_ldo * i_micro_kernel_config->datatype_size_out );
  libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                        i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 3, LIBXSMM_AARCH64_SVE_REG_UNDEF );
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, (long long)l_ldo * i_micro_kernel_config->datatype_size_out );

  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, (3LL * l_ldi - 16) * i_micro_kernel_config->datatype_size_in );

  /* close m loop */
  libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, 8 );

  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, (4LL * l_ldi - 2LL * i_mateltwise_desc->m) * i_micro_kernel_config->datatype_size_in );

  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, (((long long)i_mateltwise_desc->m/8) * 4 * l_ldo - 8LL * 2LL) * i_micro_kernel_config->datatype_size_out );

  /* close n loop */
  libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, 8 );
}


LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni2_to_vnni2t_16bit_aarch64_sve_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                const unsigned int                      i_gp_reg_in,
                                                                                const unsigned int                      i_gp_reg_out,
                                                                                const unsigned int                      i_gp_reg_m_loop,
                                                                                const unsigned int                      i_gp_reg_n_loop,
                                                                                const unsigned int                      i_gp_reg_scratch,
                                                                                const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  if ( (i_mateltwise_desc->n % 8 == 0) && (i_mateltwise_desc->m % 8 == 0) ) {
    libxsmm_generator_transform_vnni2_to_vnni2t_Nmod8_16bit_aarch64_sve_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                     i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                                     i_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc );
  } else {
    libxsmm_generator_transform_vnni2_to_vnni2t_mbit_scalar_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                       i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                                       i_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc );
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni4t_Nmod16_16bit_aarch64_sve_microkernel(  libxsmm_generated_code*                 io_generated_code,
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
  libxsmm_aarch64_sve_type l_sve_type8 = libxsmm_generator_aarch64_get_sve_type(LIBXSMM_CAST_UCHAR(8));
  libxsmm_aarch64_sve_type l_sve_type4 = libxsmm_generator_aarch64_get_sve_type(LIBXSMM_CAST_UCHAR(4));
  libxsmm_aarch64_sve_type l_sve_type2 = libxsmm_generator_aarch64_get_sve_type(LIBXSMM_CAST_UCHAR(2));

  /* n loop header */
  libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, i_mateltwise_desc->n );

  /* m loop header */
  libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, i_mateltwise_desc->m );

  libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF,
                                        i_gp_reg_in, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 0, LIBXSMM_AARCH64_SVE_REG_UNDEF );
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, (long long)l_ldi * i_micro_kernel_config->datatype_size_in );
  libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF,
                                        i_gp_reg_in, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 1, LIBXSMM_AARCH64_SVE_REG_UNDEF );
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, (long long)l_ldi * i_micro_kernel_config->datatype_size_in );
  libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF,
                                        i_gp_reg_in, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 2, LIBXSMM_AARCH64_SVE_REG_UNDEF );
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, (long long)l_ldi * i_micro_kernel_config->datatype_size_in );
  libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF,
                                        i_gp_reg_in, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 3, LIBXSMM_AARCH64_SVE_REG_UNDEF );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TRN1_V, 0, 1, 0, 4, 0, l_sve_type8 );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TRN2_V, 0, 1, 0, 5, 0, l_sve_type8 );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TRN1_V, 2, 3, 0, 6, 0, l_sve_type8 );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TRN2_V, 2, 3, 0, 7, 0, l_sve_type8 );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP1_V, 4, 5, 0, 8, 0, l_sve_type2 );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP2_V, 4, 5, 0, 9, 0, l_sve_type2 );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP1_V, 8, 9, 0, 10, 0, l_sve_type4 ); /* Computed result for N0N1N3N3*/
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP2_V, 8, 9, 0, 11, 0, l_sve_type4 ); /* Computed result for N4N5N6N7*/
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP1_V, 6, 7, 0, 14, 0, l_sve_type2 );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP2_V, 6, 7, 0, 15, 0, l_sve_type2 );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP1_V, 14, 15, 0, 12, 0, l_sve_type4 ); /* Computed result for N8N9N10N11*/
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP2_V, 14, 15, 0, 13, 0, l_sve_type4 ); /* Computed result for N12N13N14N15*/

  libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                        i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 10, LIBXSMM_AARCH64_SVE_REG_UNDEF );
  libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                        i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 1, 11, LIBXSMM_AARCH64_SVE_REG_UNDEF );
  libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                        i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 2, 12, LIBXSMM_AARCH64_SVE_REG_UNDEF );
  libxsmm_aarch64_instruction_sve_move( io_generated_code,  LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                        i_gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF, 3, 13, LIBXSMM_AARCH64_SVE_REG_UNDEF );

  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, (3LL * l_ldi - 16) * i_micro_kernel_config->datatype_size_in );

  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, (long long)l_ldo * i_micro_kernel_config->datatype_size_out );

  /* close m loop */
  libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, 4 );

  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, (4LL * l_ldi - 4LL * i_mateltwise_desc->m) * i_micro_kernel_config->datatype_size_in );

  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, (((long long)i_mateltwise_desc->m/4) * l_ldo - 16LL * 4LL) * i_micro_kernel_config->datatype_size_out );

  /* close n loop */
  libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, 16 );
}


LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni4t_16bit_aarch64_sve_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                const unsigned int                      i_gp_reg_in,
                                                                                const unsigned int                      i_gp_reg_out,
                                                                                const unsigned int                      i_gp_reg_m_loop,
                                                                                const unsigned int                      i_gp_reg_n_loop,
                                                                                const unsigned int                      i_gp_reg_scratch,
                                                                                const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  if ( (i_mateltwise_desc->n % 16 == 0) && (i_mateltwise_desc->m % 4 == 0) ) {
    libxsmm_generator_transform_vnni4_to_vnni4t_Nmod16_16bit_aarch64_sve_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                      i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                                      i_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc );
  } else {
    libxsmm_meltw_descriptor l_new_desc = *i_mateltwise_desc;
    if ( i_mateltwise_desc->n > 16 ) {
      l_new_desc.n = i_mateltwise_desc->n - (i_mateltwise_desc->n % 16);
      libxsmm_generator_transform_vnni4_to_vnni4t_Nmod16_16bit_aarch64_sve_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                        i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                                        i_gp_reg_scratch, i_micro_kernel_config, &l_new_desc  );
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, LIBXSMM_AARCH64_GP_REG_X0, LIBXSMM_AARCH64_GP_REG_UNDEF, 32,
                                            i_gp_reg_in );
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, LIBXSMM_AARCH64_GP_REG_X0, LIBXSMM_AARCH64_GP_REG_UNDEF, 64,
                                            i_gp_reg_out );
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, (long long)l_new_desc.n * l_new_desc.ldi * i_micro_kernel_config->datatype_size_in );
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, (long long)l_new_desc.n * 4 * i_micro_kernel_config->datatype_size_out );
      l_new_desc.n = i_mateltwise_desc->n % 16;
    }

    libxsmm_generator_transform_vnni4_to_vnni4t_mbit_scalar_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                     i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                                     i_gp_reg_scratch, i_micro_kernel_config, &l_new_desc );
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_transform_aarch64_sve_microkernel( libxsmm_generated_code*                        io_generated_code,
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

  if ( ( LIBXSMM_DATATYPE_I64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
         LIBXSMM_DATATYPE_I64 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) ||
       ( LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
         LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) ) {
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) {
      libxsmm_generator_transform_norm_to_normt_64bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else if ( ( LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
                LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) ||
              ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
                LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) ) {
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) {
      libxsmm_generator_transform_norm_to_normt_32bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else if ( ( LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
                LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) ||
              ( LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
                LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) ||
              ( LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
                LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) ) {
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) {
      libxsmm_generator_transform_norm_to_normt_16bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc );
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4T ) {
      /* Call 64bit normal transpose */
      libxsmm_descriptor_blob blob;
      const libxsmm_meltw_descriptor *const mock_desc = libxsmm_meltw_descriptor_init2(&blob,
        LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, i_mateltwise_desc->m/4, i_mateltwise_desc->n,
        i_mateltwise_desc->ldi/4, i_mateltwise_desc->ldo, 0, 0,
        (unsigned short)LIBXSMM_MELTW_FLAG_UNARY_NONE, (unsigned short)LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, LIBXSMM_MELTW_OPERATION_UNARY);
      libxsmm_mateltwise_kernel_config l_trans_config;
      libxsmm_generator_mateltwise_aarch64_init_micro_kernel_config_fullvector( io_generated_code, &l_trans_config, mock_desc);
      libxsmm_generator_transform_norm_to_normt_64bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, &l_trans_config, mock_desc);
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2T ) {
      /* Call 32bit normal transpose */
      libxsmm_descriptor_blob blob;
      const libxsmm_meltw_descriptor *const mock_desc = libxsmm_meltw_descriptor_init2(&blob,
        LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, i_mateltwise_desc->m/2, i_mateltwise_desc->n,
        i_mateltwise_desc->ldi/2, i_mateltwise_desc->ldo, 0, 0,
        (unsigned short)LIBXSMM_MELTW_FLAG_UNARY_NONE, (unsigned short)LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, LIBXSMM_MELTW_OPERATION_UNARY);
      libxsmm_mateltwise_kernel_config l_trans_config;
      libxsmm_generator_mateltwise_aarch64_init_micro_kernel_config_fullvector( io_generated_code, &l_trans_config, mock_desc);
      libxsmm_generator_transform_norm_to_normt_32bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, &l_trans_config, mock_desc);
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4T_TO_NORM ) {
      /* Call 64bit normal transpose */
      libxsmm_descriptor_blob blob;
      const libxsmm_meltw_descriptor *const mock_desc = libxsmm_meltw_descriptor_init2(&blob,
        LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, i_mateltwise_desc->m, i_mateltwise_desc->n/4,
        i_mateltwise_desc->ldi, i_mateltwise_desc->ldo/4, 0, 0,
        (unsigned short)LIBXSMM_MELTW_FLAG_UNARY_NONE, (unsigned short)LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, LIBXSMM_MELTW_OPERATION_UNARY);
      libxsmm_mateltwise_kernel_config l_trans_config;
      libxsmm_generator_mateltwise_aarch64_init_micro_kernel_config_fullvector( io_generated_code, &l_trans_config, mock_desc);
      libxsmm_generator_transform_norm_to_normt_64bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, &l_trans_config, mock_desc);
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2T_TO_NORM ) {
      /* Call 32bit normal transpose */
      libxsmm_descriptor_blob blob;
      const libxsmm_meltw_descriptor *const mock_desc = libxsmm_meltw_descriptor_init2(&blob,
        LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, i_mateltwise_desc->m, i_mateltwise_desc->n/2,
        i_mateltwise_desc->ldi, i_mateltwise_desc->ldo/2, 0, 0,
        (unsigned short)LIBXSMM_MELTW_FLAG_UNARY_NONE, (unsigned short)LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, LIBXSMM_MELTW_OPERATION_UNARY);
      libxsmm_mateltwise_kernel_config l_trans_config;
      libxsmm_generator_mateltwise_aarch64_init_micro_kernel_config_fullvector( io_generated_code, &l_trans_config, mock_desc);
      libxsmm_generator_transform_norm_to_normt_32bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, &l_trans_config, mock_desc);
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2_TO_VNNI2T) {
      /* TODO: check for SVE128 */
      if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE256) && (io_generated_code->arch < LIBXSMM_AARCH64_SVE512) ) {
        libxsmm_generator_transform_vnni2_to_vnni2t_16bit_aarch64_sve_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                   i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                   i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                   i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc );
      } else {
        libxsmm_generator_transform_vnni2_to_vnni2t_16bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                     i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                     i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                     i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc );
      }
    }  else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI4T) {
      if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE256) && (io_generated_code->arch < LIBXSMM_AARCH64_SVE512) ) {
        libxsmm_generator_transform_vnni4_to_vnni4t_16bit_aarch64_sve_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                   i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                   i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                   i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc );
      } else {
        libxsmm_generator_transform_vnni4_to_vnni4t_16bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                     i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                     i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                     i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc );
      }
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2) {
      if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE256) && (io_generated_code->arch < LIBXSMM_AARCH64_SVE512) ) {
        libxsmm_generator_transform_norm_to_vnni2_16bit_aarch64_sve_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc, 0 );
      } else {
        libxsmm_generator_transform_norm_to_vnni2_16bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                   i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                   i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                   i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc, 0 );
      }
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD) {
      if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE256) && (io_generated_code->arch < LIBXSMM_AARCH64_SVE512) ) {
        libxsmm_generator_transform_norm_to_vnni2_16bit_aarch64_sve_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc, 1 );
      } else {
        libxsmm_generator_transform_norm_to_vnni2_16bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                   i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                   i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                   i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc, 1 );
      }
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4) {
      if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE256) && (io_generated_code->arch < LIBXSMM_AARCH64_SVE512) ) {
        libxsmm_generator_transform_norm_to_vnni4_16bit_aarch64_sve_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc, 0 );
      } else {
        libxsmm_generator_transform_norm_to_vnni4_16bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                   i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                   i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                   i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc, 0 );
      }
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4_PAD) {
      if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE256) && (io_generated_code->arch < LIBXSMM_AARCH64_SVE512) ) {
        libxsmm_generator_transform_norm_to_vnni4_16bit_aarch64_sve_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc, 1 );
      } else {
        libxsmm_generator_transform_norm_to_vnni4_16bit_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                   i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                   i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                   i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc, 1 );
      }
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
  } else if ( ( LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
                LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) ||
              ( LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
                LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) ) {
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

