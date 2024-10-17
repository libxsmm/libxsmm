/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include "generator_common_aarch64.h"
#include "generator_mateltwise_aarch64.h"
#include "generator_mateltwise_misc_aarch64.h"
#include "generator_aarch64_instructions.h"
#include "generator_common.h"


LIBXSMM_API_INTERN
void libxsmm_generator_mn_code_block_replicate_col_var_aarch64( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    libxsmm_mateltwise_kernel_config*              i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc,
    unsigned int                                   vlen,
    unsigned int                                   m_trips_loop,
    unsigned int                                   m_unroll_factor,
    unsigned int                                   peeled_m_trips,
    unsigned int                                   i_use_masking,
    unsigned int                                   mask_inout ) {
  unsigned int im;
  int extra_bytes_ldo = 0;
  unsigned char pred_reg_mask_use = 0;
  unsigned int l_masked_elements = 0;
  unsigned int l_is_inp_bf16 = (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ? 1 : 0;
  unsigned int l_is_out_bf16 = (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ? 1 : 0;
  unsigned char pred_reg_all = 0;
  unsigned char pred_reg_mask_f32 = 1;
  unsigned char pred_reg_all_bf16 = 2;
  unsigned char pred_reg_mask_bf16= 3;
  unsigned int upconvert_input_bf16f32 = 0;
  unsigned int downconvert_input_f32bf16 = 0;
  unsigned int mask_count2 = 0;

  if ((LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) && (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))) {
    upconvert_input_bf16f32 = 1;
  }

  if ((LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) && (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0))) {
    downconvert_input_f32bf16 = 1;
  }

  if (m_trips_loop > 1) {
    libxsmm_generator_loop_header_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, m_trips_loop);
  }

  for (im = 0; im < m_unroll_factor; im++) {
    unsigned int use_masking = ((im == m_unroll_factor - 1) && (i_use_masking == 1)) ? 1 : 0;
    mask_count2 = (use_masking > 0) ? mask_inout : 0;
    l_masked_elements = (l_is_inp_bf16 == 0) ? mask_count2 : (mask_count2 > 0) ? mask_count2 : vlen;
    pred_reg_mask_use = (l_is_inp_bf16 == 0) ? ((mask_count2 > 0) ? pred_reg_mask_f32 : pred_reg_all) : ((mask_count2 > 0) ? pred_reg_mask_bf16 : pred_reg_all_bf16);

    libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_in,
        i_gp_reg_mapping->gp_reg_scratch_0, im, i_micro_kernel_config->datatype_size_in,
        l_masked_elements, 1, 0, pred_reg_mask_use);
    if (upconvert_input_bf16f32 > 0) {
      libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, im, 0);
    }
    if (downconvert_input_f32bf16 > 0) {
      libxsmm_generator_vcvt_f32bf16_aarch64( io_generated_code, im, 0);
    }
  }

  libxsmm_generator_loop_header_gp_reg_bound_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, i_gp_reg_mapping->gp_reg_n );

  for (im = 0; im < m_unroll_factor; im++) {
    unsigned int use_masking = ((im == m_unroll_factor - 1) && (i_use_masking == 1)) ? 1 : 0;

    mask_count2 = (use_masking > 0) ? mask_inout : 0;
    l_masked_elements = (l_is_out_bf16 == 0) ? mask_count2 : (mask_count2 > 0) ? mask_count2 : vlen;
    pred_reg_mask_use = (l_is_out_bf16 == 0) ? ((mask_count2 > 0) ? pred_reg_mask_f32 : pred_reg_all) : ((mask_count2 > 0) ? pred_reg_mask_bf16 : pred_reg_all_bf16);

    libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_out,
        i_gp_reg_mapping->gp_reg_scratch_0, im, i_micro_kernel_config->datatype_size_out,
        l_masked_elements, 1, 1, pred_reg_mask_use);
  }

  extra_bytes_ldo = i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out - (vlen * i_micro_kernel_config->datatype_size_out * m_unroll_factor - i_use_masking * (vlen-mask_inout) * i_micro_kernel_config->datatype_size_out);
  if (extra_bytes_ldo > 0) {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out, extra_bytes_ldo );
  }

  libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, 1);

  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_scratch_0, (long long)i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out );
  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MUL, i_gp_reg_mapping->gp_reg_n, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_scratch_0, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR, i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );

  if ((m_trips_loop > 1) || ((m_trips_loop == 1) && (peeled_m_trips > 0))) {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out, (long long)m_unroll_factor * i_micro_kernel_config->datatype_size_out * vlen );
    if (m_trips_loop > 1) {
      libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, 1);
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_replicate_col_var_aarch64_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    libxsmm_mateltwise_kernel_config*              i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {

  unsigned int m, use_m_masking, m_trips, m_unroll_factor, m_trips_loop, peeled_m_trips, max_m_unrolling;
  unsigned int END_LABEL = 1;
  unsigned int mask_inout = 1;
  unsigned int vlen = libxsmm_cpuid_vlen32(i_micro_kernel_config->instruction_set);
  unsigned char is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  unsigned char pred_reg_all = 0;
  unsigned char pred_reg_mask_f32 = 1;
  unsigned char pred_reg_all_bf16 = 2;
  unsigned char pred_reg_mask_bf16= 3;
  unsigned int l_is_inp_bf16 = (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ? 1 : 0;
  unsigned int l_is_out_bf16 = (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ? 1 : 0;
  libxsmm_jump_label_tracker l_jump_label_tracker;
  libxsmm_reset_jump_label_tracker(&l_jump_label_tracker);

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in     = LIBXSMM_AARCH64_GP_REG_X9;
  i_gp_reg_mapping->gp_reg_out    = LIBXSMM_AARCH64_GP_REG_X10;
  i_gp_reg_mapping->gp_reg_n_loop = LIBXSMM_AARCH64_GP_REG_X11;
  i_gp_reg_mapping->gp_reg_m_loop = LIBXSMM_AARCH64_GP_REG_X12;
  i_gp_reg_mapping->gp_reg_n      = LIBXSMM_AARCH64_GP_REG_X13;
  i_gp_reg_mapping->gp_reg_scratch_0  = LIBXSMM_AARCH64_GP_REG_X14;

  /* load the input pointer and output pointer and the variable N */
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_gp_reg_mapping->gp_reg_n );
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_n, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_gp_reg_mapping->gp_reg_n );
  libxsmm_aarch64_instruction_cond_jump_to_label( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_CBZ, i_gp_reg_mapping->gp_reg_n, END_LABEL, &l_jump_label_tracker);

  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 32, i_gp_reg_mapping->gp_reg_in );
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 64, i_gp_reg_mapping->gp_reg_out );

  max_m_unrolling = 32;

  m                 = i_mateltwise_desc->m;
  use_m_masking     = (m % vlen == 0) ? 0 : 1;
  m_trips           = (m + vlen - 1) / vlen;
  m_unroll_factor   = m_trips;
  m_trips_loop      = 1;
  peeled_m_trips    = 0;

  /* set pred reg 0 to true for sve */
  if ( is_sve ) {
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, pred_reg_all, -1, i_gp_reg_mapping->gp_reg_scratch_0 );
  }

  if (use_m_masking > 0) {
    mask_inout = m % vlen;
    if ((is_sve > 0) && (l_is_inp_bf16 > 0 || l_is_out_bf16 > 0)) {
      libxsmm_generator_set_p_register_aarch64_sve(io_generated_code, pred_reg_mask_bf16, 2 * mask_inout, i_gp_reg_mapping->gp_reg_scratch_0);
    }
    if ((is_sve > 0) && (l_is_inp_bf16 == 0 || l_is_out_bf16 == 0)) {
      libxsmm_generator_set_p_register_aarch64_sve(io_generated_code, pred_reg_mask_f32, 4 * mask_inout, i_gp_reg_mapping->gp_reg_scratch_0);
    }
  }

  if ((is_sve > 0) && (l_is_inp_bf16 > 0 || l_is_out_bf16 > 0)) {
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, pred_reg_all_bf16, i_micro_kernel_config->datatype_size_in * vlen, i_gp_reg_mapping->gp_reg_scratch_0 );
  }

  /* In this case we have to generate a loop for m */
  if (m_unroll_factor > max_m_unrolling) {
    m_unroll_factor = max_m_unrolling;
    m_trips_loop = m_trips/m_unroll_factor;
    peeled_m_trips = m_trips  - m_unroll_factor * m_trips_loop;
    if ((use_m_masking > 0) && (peeled_m_trips == 0)) {
      m_trips_loop--;
      peeled_m_trips = m_trips  - m_unroll_factor * m_trips_loop;
    }
  } else {
    if ((use_m_masking > 0) && (peeled_m_trips == 0)) {
      m_trips_loop--;
      peeled_m_trips = m_trips  - m_unroll_factor * m_trips_loop;
    }
  }

  if (m_trips_loop >= 1) {
    libxsmm_generator_mn_code_block_replicate_col_var_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        vlen, m_trips_loop, m_unroll_factor, peeled_m_trips, 0, mask_inout );
  }
  if (peeled_m_trips > 0) {
    libxsmm_generator_mn_code_block_replicate_col_var_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        vlen, 0, peeled_m_trips, peeled_m_trips, use_m_masking, mask_inout );
  }

  libxsmm_aarch64_instruction_register_jump_label( io_generated_code, END_LABEL, &l_jump_label_tracker );
}
