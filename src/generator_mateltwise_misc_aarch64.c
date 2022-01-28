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
#include "libxsmm_main.h"

#if !defined(LIBXSMM_GENERATOR_MATELTWISE_MISC_AARCH64_JUMP_LABEL_TRACKER_MALLOC)
# define LIBXSMM_GENERATOR_MATELTWISE_MISC_AARCH64_JUMP_LABEL_TRACKER_MALLOC
#endif

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

  if (m_trips_loop > 1) {
    libxsmm_generator_loop_header_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, m_trips_loop);
  }

  for (im = 0; im < m_unroll_factor; im++) {
    unsigned int use_masking = ((im == m_unroll_factor - 1) && (i_use_masking == 1)) ? 1 : 0;
    libxsmm_generator_vloadstore_masked_vreg_aarch64_asimd( io_generated_code, i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, im, i_micro_kernel_config->datatype_size_in, (use_masking > 0) ? mask_inout : 0, 1, 0);
  }

  libxsmm_generator_loop_header_gp_reg_bound_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, i_gp_reg_mapping->gp_reg_n );

  for (im = 0; im < m_unroll_factor; im++) {
    unsigned int use_masking = ((im == m_unroll_factor - 1) && (i_use_masking == 1)) ? 1 : 0;
    libxsmm_generator_vloadstore_masked_vreg_aarch64_asimd( io_generated_code, i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, im, i_micro_kernel_config->datatype_size_out, (use_masking > 0) ? mask_inout : 0, 1, 1);
  }

  extra_bytes_ldo = i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out - (16 * m_unroll_factor - i_use_masking * (vlen-mask_inout) *4);
  if (extra_bytes_ldo > 0) {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out, (unsigned long long)extra_bytes_ldo );
  }

  libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, 1);

  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_scratch_0, (unsigned long long)i_mateltwise_desc->ldo * (unsigned long long)i_micro_kernel_config->datatype_size_out );
  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MUL, i_gp_reg_mapping->gp_reg_n, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_scratch_0, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR, i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );

  if ((m_trips_loop > 1) || ((m_trips_loop == 1) && (peeled_m_trips > 0))) {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out, (unsigned long long)m_unroll_factor * (unsigned long long)16 );
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

  unsigned int m, use_m_masking, m_trips, m_unroll_factor, m_trips_loop, peeled_m_trips, vlen, max_m_unrolling;
  unsigned int END_LABEL = 1;
  unsigned int mask_inout = 1;
#if defined(LIBXSMM_GENERATOR_MATELTWISE_MISC_AARCH64_JUMP_LABEL_TRACKER_MALLOC)
  libxsmm_jump_label_tracker* const p_jump_label_tracker = (libxsmm_jump_label_tracker*)malloc(sizeof(libxsmm_jump_label_tracker));
#else
  libxsmm_jump_label_tracker l_jump_label_tracker;
  libxsmm_jump_label_tracker* const p_jump_label_tracker = &l_jump_label_tracker;
#endif
  libxsmm_reset_jump_label_tracker(p_jump_label_tracker);

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
  libxsmm_aarch64_instruction_cond_jump_to_label( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_CBZ, i_gp_reg_mapping->gp_reg_n, END_LABEL, p_jump_label_tracker );

  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 32, i_gp_reg_mapping->gp_reg_in );
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 56, i_gp_reg_mapping->gp_reg_out );

  vlen = 4;
  max_m_unrolling = 32;

  m                 = i_mateltwise_desc->m;
  use_m_masking     = (m % vlen == 0) ? 0 : 1;
  m_trips           = (m + vlen - 1) / vlen;
  m_unroll_factor   = m_trips;
  m_trips_loop      = 1;
  peeled_m_trips    = 0;

  if (use_m_masking == 1) {
    mask_inout = m % vlen;
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

  libxsmm_aarch64_instruction_register_jump_label( io_generated_code, END_LABEL, p_jump_label_tracker );
#if defined(LIBXSMM_GENERATOR_MATELTWISE_MISC_AARCH64_JUMP_LABEL_TRACKER_MALLOC)
  free(p_jump_label_tracker);
#endif
}


