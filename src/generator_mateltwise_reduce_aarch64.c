/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
*               Friedrich Schiller University Jena - All rights reserved.     *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.), Antonio Noack (FSU Jena)
******************************************************************************/
#include "generator_common_aarch64.h"
#include "generator_mateltwise_aarch64.h"
#include "generator_mateltwise_reduce_aarch64.h"
#include "generator_aarch64_instructions.h"
#include "generator_common_aarch64.h"
#include "generator_common.h"

#if !defined(LIBXSMM_GENERATOR_MATELTWISE_REDUCE_AARCH64_JUMP_LABEL_TRACKER_MALLOC)
# define LIBXSMM_GENERATOR_MATELTWISE_REDUCE_AARCH64_JUMP_LABEL_TRACKER_MALLOC
#endif
#if 0
#define USE_ENV_TUNING
#endif


LIBXSMM_API_INTERN
void libxsmm_generator_reduce_cols_ncnc_aarch64_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {

  unsigned int bn, bc, N, C, Nb, iM, im, in, vreg0;
  unsigned int use_m_masking, m_trips, m_outer_trips, m_inner_trips;
  unsigned int cur_acc0;
  unsigned int vlen = libxsmm_cpuid_vlen32(i_micro_kernel_config->instruction_set);
  unsigned int m_unroll_factor = 4;
  unsigned int mask_count = 0, mask_count2;
  unsigned char is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  unsigned char pred_reg_all = 0;
  unsigned char pred_reg_mask = 1;
  unsigned char pred_reg_all_bf16 = 2;
  unsigned char pred_reg_mask_use = 0;
  unsigned int l_masked_elements = 0;
  unsigned int l_is_inp_bf16 = (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ? 1 : 0;
  unsigned int l_is_out_bf16 = (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ? 1 : 0;
  unsigned int l_bn_loop_reg = 0;
  unsigned int max_bn_unroll = 32;
  unsigned int bn_unroll_iters = 0;

  bc  = i_mateltwise_desc->m;
  bn  = i_mateltwise_desc->n;
  C   = i_mateltwise_desc->ldi;
  N   = i_mateltwise_desc->ldo;
  bn_unroll_iters = bn;

  Nb  = N/bn;

  if ( (N % bn != 0)  || (C % bc != 0) ) {
    /* This should not happen */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in     = LIBXSMM_AARCH64_GP_REG_X9;
  i_gp_reg_mapping->gp_reg_out    = LIBXSMM_AARCH64_GP_REG_X10;
  i_gp_reg_mapping->gp_reg_n_loop = LIBXSMM_AARCH64_GP_REG_X11;
  i_gp_reg_mapping->gp_reg_scratch_0  = LIBXSMM_AARCH64_GP_REG_X12;
  l_bn_loop_reg = LIBXSMM_AARCH64_GP_REG_X13;

  /* load the input pointer and output pointer */
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 32, i_gp_reg_mapping->gp_reg_in );

  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 64, i_gp_reg_mapping->gp_reg_out );

  use_m_masking     = (bc % vlen == 0) ? 0 : 1;
  m_unroll_factor   = (use_m_masking == 0) ? 8 : 4;
  m_trips           = (bc + vlen - 1)/vlen;
  m_outer_trips     = (m_trips + m_unroll_factor - 1)/m_unroll_factor;

  /* set pred reg 0 to true for sve */
  if ( is_sve ) {
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, 0, -1, 0 );
  }

  if (use_m_masking > 0) {
    mask_count = bc % vlen;
    if (is_sve) libxsmm_generator_set_p_register_aarch64_sve(io_generated_code, pred_reg_mask, i_micro_kernel_config->datatype_size_in * mask_count, i_gp_reg_mapping->gp_reg_scratch_0);
  }

  if ((is_sve > 0) && (l_is_inp_bf16 > 0 || l_is_out_bf16 > 0)) {
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, pred_reg_all_bf16,  (l_is_inp_bf16 > 0) ? i_micro_kernel_config->datatype_size_in * vlen : i_micro_kernel_config->datatype_size_out * vlen, i_gp_reg_mapping->gp_reg_scratch_0 );
  }

  /* Register allocation: Registers zmm8-zmm15 are accumulators, zmm0-zmm7 are used for loading input */
  for (iM = 0; iM < m_outer_trips; iM++) {
    m_inner_trips = (iM == m_outer_trips - 1) ? m_trips - iM * m_unroll_factor : m_unroll_factor;
    if (bn * m_inner_trips > 100) {
      max_bn_unroll = LIBXSMM_UPDIV(100, m_inner_trips);
    } else {
      max_bn_unroll = bn;
    }
    for (im = 0; im < m_inner_trips; im++) {
      cur_acc0 = m_unroll_factor + im;
      if (is_sve){
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                cur_acc0, cur_acc0, 0, cur_acc0, pred_reg_all, LIBXSMM_AARCH64_SVE_TYPE_B );
      } else {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                cur_acc0, cur_acc0, 0, cur_acc0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
      }
    }

    if (Nb > 1) {
      /* open n loop */
      libxsmm_generator_loop_header_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, Nb);
    }

    if (bn > max_bn_unroll) {
      bn_unroll_iters = max_bn_unroll;
      while (bn % bn_unroll_iters != 0) {
        bn_unroll_iters--;
      }
      libxsmm_generator_loop_header_aarch64(io_generated_code, io_loop_label_tracker, l_bn_loop_reg, bn/bn_unroll_iters);
    } else {
      bn_unroll_iters = bn;
    }

    for (in = 0; in < bn_unroll_iters; in++ ) {
      int extra_bytes = bc * i_micro_kernel_config->datatype_size_in - vlen * i_micro_kernel_config->datatype_size_in * m_inner_trips;
      if ((use_m_masking == 1) && (iM == m_outer_trips-1)) {
        extra_bytes = extra_bytes + (vlen - mask_count) * i_micro_kernel_config->datatype_size_in;
      }
      for (im = 0; im < m_inner_trips; im++) {
        cur_acc0 = m_unroll_factor + im;
        vreg0    = im;
        mask_count2 = ((use_m_masking == 1) && (iM == m_outer_trips-1) && (im == m_inner_trips-1)) ? mask_count : 0;
        l_masked_elements = (l_is_inp_bf16 == 0) ? mask_count2 : (mask_count2 > 0) ? mask_count2 : vlen;
        pred_reg_mask_use = (l_is_inp_bf16 == 0) ? ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all) : ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all_bf16);
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_in,
            i_gp_reg_mapping->gp_reg_scratch_0, vreg0, i_micro_kernel_config->datatype_size_in,
            l_masked_elements, 1, 0, pred_reg_mask_use);
        if (l_is_inp_bf16 > 0) {
          libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, vreg0, 0);
        }
        if (is_sve){
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FADD_V,
                                                   vreg0, cur_acc0, 0, cur_acc0, pred_reg_all, LIBXSMM_AARCH64_SVE_TYPE_S );
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V,
                                                     vreg0, cur_acc0, 0, cur_acc0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
        }
      }
      if (extra_bytes > 0) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                        i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in, extra_bytes );
      }
    }

    if (bn > max_bn_unroll) {
      libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, l_bn_loop_reg, 1);
    }

    if (Nb > 1) {
      unsigned int offset = C * bn * i_micro_kernel_config->datatype_size_in - bc * bn * i_micro_kernel_config->datatype_size_in;
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                        i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in, offset );
      /* close n loop */
      libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, 1);

      /* Readjust reg_in */
      if (m_outer_trips > 1) {
        unsigned int _offset = C * N * i_micro_kernel_config->datatype_size_in - vlen * m_unroll_factor * i_micro_kernel_config->datatype_size_in ;
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                          i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in, _offset );
      }
    }

    if (Nb == 1) {
      if (m_outer_trips > 1) {
        unsigned int offset = bc * bn * i_micro_kernel_config->datatype_size_in - vlen * m_unroll_factor * i_micro_kernel_config->datatype_size_in ;
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                       i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in, offset );
      }
    }

    for (im = 0; im < m_inner_trips; im++) {
      unsigned char _mask_count2 = LIBXSMM_CAST_UCHAR(((use_m_masking == 1) && (iM == m_outer_trips-1) && (im == m_inner_trips-1)) ? mask_count : 0);
      l_masked_elements = (l_is_out_bf16 == 0) ? _mask_count2 : (_mask_count2 > 0) ? _mask_count2 : vlen;
      pred_reg_mask_use = (l_is_out_bf16 == 0) ? ((_mask_count2 > 0) ? pred_reg_mask : pred_reg_all) : ((_mask_count2 > 0) ? pred_reg_mask : pred_reg_all_bf16);
      cur_acc0 = m_unroll_factor + im;
      if (l_is_inp_bf16 > 0) {
        libxsmm_generator_vcvt_f32bf16_aarch64( io_generated_code, cur_acc0, 0);
      }
      libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, cur_acc0, i_micro_kernel_config->datatype_size_out,
          l_masked_elements, 1, 1, pred_reg_mask_use );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_reduce_cols_aarch64_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {

  unsigned int _in, im, m, n, m_trips, use_m_masking, mask_count = 0, mask_count2, compute_squared_vals_reduce, compute_plain_vals_reduce;
  unsigned int start_vreg_sum = 0;
  unsigned int start_vreg_sum2 = 0;
  unsigned int reduce_instr = 0;
  unsigned int vmove_instruction_in = i_micro_kernel_config->vmove_instruction_in;
  unsigned int vmove_instruction_out = i_micro_kernel_config->vmove_instruction_out;

  unsigned int vlen = libxsmm_cpuid_vlen32(i_micro_kernel_config->instruction_set);
  unsigned int tmp_vreg = 31;
  unsigned int max_m_unrolling = 30;
  unsigned int m_unroll_factor = 30;
  unsigned int peeled_m_trips = 0;
  unsigned int m_trips_loop = 0;
  unsigned int accs_used = 0;
  unsigned int split_factor = 0;
  unsigned int n_trips = 0;
  unsigned int n_remainder = 0;
  unsigned int peeled_n_trips = 0;
  unsigned int flag_reduce_elts = ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD) ||
                                   (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD) ||
                                   (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX)) ? 1 : 0;
  unsigned int flag_reduce_elts_sq = ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X2_OP_ADD) ||
                                   (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD)) ? 1 : 0;
  unsigned int flag_reduce_op_add =((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD) ||
                                   (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD) ||
                                   (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X2_OP_ADD)) ? 1 : 0;
  unsigned int flag_reduce_op_max = (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX) ? 1 : 0;
  unsigned int reduce_on_output   = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_INIT_ACC) > 0 ) ? 1 : 0;

  libxsmm_aarch64_sve_type sve_type = libxsmm_generator_aarch64_get_sve_type(LIBXSMM_TYPESIZE(libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)));
  libxsmm_aarch64_asimd_tupletype asimd_type = (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D;
  unsigned char is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  unsigned char pred_reg_all = 0; /* set by caller */
  unsigned char pred_reg_mask_in = 1;
  unsigned char pred_reg_mask_out = 3;
  unsigned char pred_reg_all_bf16 = 2;
  unsigned char pred_reg_mask_use = 0;
  unsigned int l_masked_elements = 0;
  unsigned int l_is_inp_bf16 = (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ? 1 : 0;
  unsigned int l_is_out_bf16 = (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ? 1 : 0;

  if (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
    vlen = vlen/2;
  }

  LIBXSMM_UNUSED(vmove_instruction_in);
  LIBXSMM_UNUSED(vmove_instruction_out);
  /* Some rudimentary checking of M, N and LDs*/
  if ( i_mateltwise_desc->m > i_mateltwise_desc->ldi ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
    return;
  }

  if ( flag_reduce_elts > 0 ) {
    compute_plain_vals_reduce= 1;
  } else {
    compute_plain_vals_reduce= 0;
  }

  if ( flag_reduce_elts_sq > 0 ) {
    compute_squared_vals_reduce = 1;
  } else {
    compute_squared_vals_reduce = 0;
  }

  if ( flag_reduce_op_add > 0 ) {
    reduce_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FADD_V : LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V;
  } else if ( flag_reduce_op_max > 0 ) {
    reduce_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FMAX_V_P : LIBXSMM_AARCH64_INSTR_ASIMD_FMAX_V;
  } else {
    /* This should not happen */
    printf("Only supported reduction OPs are ADD and MAX for this reduce kernel\n");
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  if ((compute_squared_vals_reduce > 0) && (flag_reduce_op_add == 0)) {
    /* This should not happen */
    printf("Support for squares's reduction only when reduction OP is ADD\n");
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in                   = LIBXSMM_AARCH64_GP_REG_X9;
  i_gp_reg_mapping->gp_reg_reduced_elts         = LIBXSMM_AARCH64_GP_REG_X10;
  i_gp_reg_mapping->gp_reg_reduced_elts_squared = LIBXSMM_AARCH64_GP_REG_X11;
  i_gp_reg_mapping->gp_reg_m_loop               = LIBXSMM_AARCH64_GP_REG_X12;
  i_gp_reg_mapping->gp_reg_n_loop               = LIBXSMM_AARCH64_GP_REG_X13;
  i_gp_reg_mapping->gp_reg_scratch_0            = LIBXSMM_AARCH64_GP_REG_X14;

  /* load the input pointer and output pointer */
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
    i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 32, i_gp_reg_mapping->gp_reg_in );

  if ( compute_plain_vals_reduce > 0 ) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
      i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 64, i_gp_reg_mapping->gp_reg_reduced_elts );
    if ( compute_squared_vals_reduce > 0 ) {
      unsigned int result_size = i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out;
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
        i_gp_reg_mapping->gp_reg_reduced_elts, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_reduced_elts_squared, result_size );
    }
  } else {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
      i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 64, i_gp_reg_mapping->gp_reg_reduced_elts_squared );
  }

  /* We fully unroll in N dimension, calculate m-mask if there is remainder */
  m = i_mateltwise_desc->m;
  n = i_mateltwise_desc->n;
  assert(0 != vlen);
  use_m_masking = ( m % vlen == 0 ) ? 0 : 1;

  /* set pred reg 0 to true for sve */
  if ( is_sve ) {
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, 0, -1, 0 );
  }

  /* Calculate input mask in case we see m_masking */
  if (use_m_masking == 1) {
    mask_count = m % vlen;
    if (is_sve) {
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, pred_reg_mask_in,
      i_micro_kernel_config->datatype_size_in * mask_count, i_gp_reg_mapping->gp_reg_scratch_0 );
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, pred_reg_mask_out,
      i_micro_kernel_config->datatype_size_out * mask_count, i_gp_reg_mapping->gp_reg_scratch_0 );
    }
  }

  if ((is_sve > 0) && (l_is_inp_bf16 > 0 || l_is_out_bf16 > 0)) {
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, pred_reg_all_bf16,
    (l_is_inp_bf16 > 0) ? i_micro_kernel_config->datatype_size_in * vlen : i_micro_kernel_config->datatype_size_out * vlen, i_gp_reg_mapping->gp_reg_scratch_0 );
  }

  m_trips           = (m + vlen - 1) / vlen;
  m_unroll_factor   = m_trips;
  m_trips_loop      = 1;
  peeled_m_trips    = 0;

  if ( (compute_plain_vals_reduce > 0) && (compute_squared_vals_reduce > 0) ) {
    max_m_unrolling = max_m_unrolling/2;
    start_vreg_sum2 = max_m_unrolling;
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

  accs_used = m_unroll_factor;
  if (max_m_unrolling/accs_used > 1) {
    split_factor = max_m_unrolling/accs_used;
  } else {
    split_factor = 1;
  }

  n_trips = (n+split_factor-1)/split_factor;
  n_remainder = n % split_factor;
  peeled_n_trips = 0;
  if (n_remainder > 0) {
    n_trips--;
    peeled_n_trips = n - n_trips * split_factor;
  }

  if ( m_trips_loop >= 1 ) {
    if ( m_trips_loop > 1 ) {
      libxsmm_generator_loop_header_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, m_trips_loop);
    }
    /* Initialize accumulators to zero */
    for (_in = 0; _in < split_factor; _in++) {
      for (im = 0; im < m_unroll_factor; im++) {
        if ( compute_plain_vals_reduce > 0 ) {
          if ( flag_reduce_op_add > 0 ) {
            if (is_sve){
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                start_vreg_sum + im + _in * m_unroll_factor, start_vreg_sum + im + _in * m_unroll_factor,
                0, start_vreg_sum + im + _in * m_unroll_factor, 0, sve_type );
            } else {
              libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                start_vreg_sum + im + _in * m_unroll_factor, start_vreg_sum + im + _in * m_unroll_factor,
                0, start_vreg_sum + im + _in * m_unroll_factor, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
            }
          } else if ( flag_reduce_op_max > 0 ) {
            l_masked_elements = (l_is_inp_bf16 == 0) ? 0 : vlen;
            pred_reg_mask_use = (l_is_inp_bf16 == 0) ? pred_reg_all : pred_reg_all_bf16;
            libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0,
                                                             start_vreg_sum + im + _in * m_unroll_factor, i_micro_kernel_config->datatype_size_in, l_masked_elements, 1, 0, pred_reg_mask_use );
            if (l_is_inp_bf16 > 0) {
              libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, start_vreg_sum + im + _in * m_unroll_factor, 0);
            }
          }
        }

        if ( compute_squared_vals_reduce > 0 ) {
          if (is_sve){
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
              start_vreg_sum2 + im + _in * m_unroll_factor, start_vreg_sum2 + im + _in * m_unroll_factor,
              0, start_vreg_sum2 + im + _in * m_unroll_factor, 0, sve_type );
          } else {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
              start_vreg_sum2 + im + _in * m_unroll_factor, start_vreg_sum2 + im + _in * m_unroll_factor,
              0, start_vreg_sum2 + im + _in * m_unroll_factor, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
          }
        }
      }
      if ( flag_reduce_op_max > 0 ) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                      i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in,
                                                      (long long)m_unroll_factor * i_micro_kernel_config->datatype_size_in * vlen );
      }
    }

    if (n_trips >= 1) {
      if (n_trips > 1) {
        libxsmm_generator_loop_header_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, n_trips);
      }

      for (_in = 0; _in < split_factor; _in++) {
        int extra_ld_bytes  = i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in - m_unroll_factor * i_micro_kernel_config->datatype_size_in * vlen;
        for (im = 0; im < m_unroll_factor; im++) {
          l_masked_elements = (l_is_inp_bf16 == 0) ? 0 : vlen;
          pred_reg_mask_use = (l_is_inp_bf16 == 0) ? pred_reg_all : pred_reg_all_bf16;
          libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0,
                                                            tmp_vreg, i_micro_kernel_config->datatype_size_in, l_masked_elements, 1, 0, pred_reg_mask_use );
          if (l_is_inp_bf16 > 0) {
            libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, tmp_vreg, 0);
          }

          if ( compute_plain_vals_reduce > 0 ) {
            if (is_sve){
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, reduce_instr,
                                                       tmp_vreg, start_vreg_sum + im + _in * m_unroll_factor,
                                                       0, start_vreg_sum + im + _in * m_unroll_factor, pred_reg_all,
                                                       sve_type );
            } else {
              libxsmm_aarch64_instruction_asimd_compute( io_generated_code, reduce_instr,
                                                        tmp_vreg, start_vreg_sum + im + _in * m_unroll_factor,
                                                        0, start_vreg_sum + im + _in * m_unroll_factor, asimd_type );
            }
          }

          if ( compute_squared_vals_reduce > 0 ) {
            if (is_sve){
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_P,
                                                       tmp_vreg, tmp_vreg, 0, start_vreg_sum2 + im + _in * m_unroll_factor, pred_reg_all, sve_type);
            } else {
              libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,
                                                        tmp_vreg, tmp_vreg, 0, start_vreg_sum2 + im + _in * m_unroll_factor, asimd_type );
            }
          }
        }
        if (extra_ld_bytes > 0) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                        i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in, extra_ld_bytes );
        }
      }

      if (n_trips > 1) {
        libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, 1);
      }
    }

    if (peeled_n_trips > 0) {
      for (_in = 0; _in < peeled_n_trips; _in++) {
        int  extra_ld_bytes  = i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in - m_unroll_factor * i_micro_kernel_config->datatype_size_in * vlen;
        for (im = 0; im < m_unroll_factor; im++) {
          l_masked_elements = (l_is_inp_bf16 == 0) ? 0 : vlen;
          pred_reg_mask_use = (l_is_inp_bf16 == 0) ? pred_reg_all : pred_reg_all_bf16;
          libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0,
                                                            tmp_vreg, i_micro_kernel_config->datatype_size_in, l_masked_elements, 1, 0, pred_reg_mask_use );
          if (l_is_inp_bf16 > 0) {
            libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, tmp_vreg, 0);
          }
          if ( compute_plain_vals_reduce > 0 ) {
            if (is_sve){
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, reduce_instr,
                                                       tmp_vreg, start_vreg_sum + im + _in * m_unroll_factor,
                                                       0, start_vreg_sum + im + _in * m_unroll_factor, pred_reg_all, sve_type );
            } else {
              libxsmm_aarch64_instruction_asimd_compute( io_generated_code, reduce_instr,
                                                         tmp_vreg, start_vreg_sum + im + _in * m_unroll_factor,
                                                         0, start_vreg_sum + im + _in * m_unroll_factor, asimd_type );
            }
          }

          if ( compute_squared_vals_reduce > 0 ) {
            if (is_sve){
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_P,
                                                       tmp_vreg, tmp_vreg, 0, start_vreg_sum2 + im + _in * m_unroll_factor, pred_reg_all, sve_type );
            } else {
              libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,
                                                         tmp_vreg, tmp_vreg, 0, start_vreg_sum2 + im + _in * m_unroll_factor, asimd_type );
            }
          }
        }
        if (extra_ld_bytes > 0) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                        i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0,
                                                        i_gp_reg_mapping->gp_reg_in, extra_ld_bytes );
        }
      }
    }

    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in,
                                                   (long long)n * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in - (long long)m_unroll_factor * vlen * i_micro_kernel_config->datatype_size_in );

    for (_in = 1; _in < split_factor; _in++) {
      for (im = 0; im < m_unroll_factor; im++) {
        if ( compute_plain_vals_reduce > 0 ) {
          if ( is_sve ) {
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, reduce_instr,
                                                     start_vreg_sum + im, start_vreg_sum + im + _in * m_unroll_factor,
                                                     0, start_vreg_sum + im, pred_reg_all, sve_type );
          } else {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, reduce_instr,
                                                       start_vreg_sum + im, start_vreg_sum + im + _in * m_unroll_factor,
                                                       0, start_vreg_sum + im, asimd_type );
          }
        }

        if ( compute_squared_vals_reduce > 0 ) {
          if ( is_sve ) {
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, reduce_instr,
                                                     start_vreg_sum2 + im, start_vreg_sum2 + im + _in * m_unroll_factor,
                                                     0, start_vreg_sum2 + im, pred_reg_all, sve_type );
          } else {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, reduce_instr,
                                                       start_vreg_sum2 + im, start_vreg_sum2 + im + _in * m_unroll_factor,
                                                       0, start_vreg_sum2 + im, asimd_type );
          }
        }
      }
    }

    /* Store computed results */
    for (im = 0; im < m_unroll_factor; im++) {
      l_masked_elements = (l_is_out_bf16 == 0) ? 0 : vlen;
      pred_reg_mask_use = (l_is_out_bf16 == 0) ? pred_reg_all : pred_reg_all_bf16;
      if ( compute_plain_vals_reduce > 0 ) {
        if (reduce_on_output > 0) {
          libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_reduced_elts, i_gp_reg_mapping->gp_reg_scratch_0,
                                                            tmp_vreg, i_micro_kernel_config->datatype_size_out, l_masked_elements, 0, 0, pred_reg_mask_use );
          if (l_is_out_bf16 > 0) {
            libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, tmp_vreg, 0);
          }
          if (is_sve){
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, reduce_instr,
                                                     tmp_vreg, start_vreg_sum + im,
                                                     0, start_vreg_sum + im, pred_reg_all,
                                                     sve_type );
          } else {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, reduce_instr,
                                                      tmp_vreg, start_vreg_sum + im,
                                                      0, start_vreg_sum + im, asimd_type );
          }
        }
        if (l_is_out_bf16 > 0) {
          libxsmm_generator_vcvt_f32bf16_aarch64( io_generated_code, start_vreg_sum + im, 0);
        }
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_reduced_elts, i_gp_reg_mapping->gp_reg_scratch_0,
                                                           start_vreg_sum + im, i_micro_kernel_config->datatype_size_out, l_masked_elements, 1, 1, pred_reg_mask_use );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        if (reduce_on_output > 0) {
          libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_reduced_elts_squared, i_gp_reg_mapping->gp_reg_scratch_0,
                                                            tmp_vreg, i_micro_kernel_config->datatype_size_out, l_masked_elements, 0, 0, pred_reg_mask_use );
          if (l_is_out_bf16 > 0) {
            libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, tmp_vreg, 0);
          }
          if (is_sve){
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, reduce_instr,
                                                     tmp_vreg, start_vreg_sum2 + im,
                                                     0, start_vreg_sum2 + im, pred_reg_all,
                                                     sve_type );
          } else {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, reduce_instr,
                                                      tmp_vreg, start_vreg_sum2 + im,
                                                      0, start_vreg_sum2 + im, asimd_type );
          }
        }
        if (l_is_out_bf16 > 0) {
          libxsmm_generator_vcvt_f32bf16_aarch64( io_generated_code, start_vreg_sum2 + im, 0);
        }
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_reduced_elts_squared, i_gp_reg_mapping->gp_reg_scratch_0,
                                                           start_vreg_sum2 + im, i_micro_kernel_config->datatype_size_out, l_masked_elements, 1, 1, pred_reg_mask_use );
      }
    }

    if (m_trips_loop > 1) {
      libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, 1);
    }
  }

  if (peeled_m_trips > 0) {
    for (_in = 0; _in < split_factor; _in++) {
      for (im = 0; im < peeled_m_trips; im++) {
        if ( compute_plain_vals_reduce > 0 ) {
          if ( flag_reduce_op_add > 0 ) {
            if (is_sve){
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                                       start_vreg_sum + im + _in * m_unroll_factor,
                                                       start_vreg_sum + im + _in * m_unroll_factor, 0,
                                                       start_vreg_sum + im + _in * m_unroll_factor, pred_reg_all, sve_type );
            } else {
              libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                      start_vreg_sum + im + _in * m_unroll_factor,
                                                      start_vreg_sum + im + _in * m_unroll_factor, 0,
                                                      start_vreg_sum + im + _in * m_unroll_factor, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
            }
          } else if ( flag_reduce_op_max > 0 ) {
            mask_count2 = ((im == peeled_m_trips -1) && (use_m_masking > 0)) ? mask_count : 0;
            l_masked_elements = (l_is_inp_bf16 == 0) ? mask_count2 : (mask_count2 > 0) ? mask_count2 : vlen;
            pred_reg_mask_use = (l_is_inp_bf16 == 0) ? ((mask_count2 > 0) ? pred_reg_mask_in : pred_reg_all) : ((mask_count2 > 0) ? pred_reg_mask_in : pred_reg_all_bf16);
            libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0,
                                                              start_vreg_sum + im + _in * m_unroll_factor, i_micro_kernel_config->datatype_size_in, l_masked_elements, 1, 0, pred_reg_mask_use );
            if (l_is_inp_bf16 > 0) {
              libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, start_vreg_sum + im + _in * m_unroll_factor, 0);
            }
          }
        }

        if ( compute_squared_vals_reduce > 0 ) {
          if (is_sve){
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                                     start_vreg_sum2 + im + _in * m_unroll_factor,
                                                     start_vreg_sum2 + im + _in * m_unroll_factor, 0,
                                                     start_vreg_sum2 + im + _in * m_unroll_factor, pred_reg_all, sve_type );
          } else {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                       start_vreg_sum2 + im + _in * m_unroll_factor,
                                                       start_vreg_sum2 + im + _in * m_unroll_factor, 0,
                                                       start_vreg_sum2 + im + _in * m_unroll_factor, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
          }
        }
      }
      if ( flag_reduce_op_max > 0 ) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                      i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in,
                                                      (long long)peeled_m_trips * i_micro_kernel_config->datatype_size_in * vlen - (long long)use_m_masking * ((long long)vlen - mask_count) * i_micro_kernel_config->datatype_size_in );
      }
    }

    if (n_trips >= 1) {
      if (n_trips > 1) {
        libxsmm_generator_loop_header_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, n_trips);
      }

      for (_in = 0; _in < split_factor; _in++) {
        int extra_ld_bytes  = i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in - (peeled_m_trips * i_micro_kernel_config->datatype_size_in * vlen  - use_m_masking * (vlen - mask_count) * i_micro_kernel_config->datatype_size_in);
        for (im = 0; im < peeled_m_trips; im++) {
          mask_count2 = ((im == peeled_m_trips -1) && (use_m_masking > 0)) ? mask_count : 0;
          l_masked_elements = (l_is_inp_bf16 == 0) ? mask_count2 : (mask_count2 > 0) ? mask_count2 : vlen;
          pred_reg_mask_use = (l_is_inp_bf16 == 0) ? ((mask_count2 > 0) ? pred_reg_mask_in : pred_reg_all) : ((mask_count2 > 0) ? pred_reg_mask_in : pred_reg_all_bf16);
          libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0,
                                                            tmp_vreg, i_micro_kernel_config->datatype_size_in, l_masked_elements, 1, 0, pred_reg_mask_use );
          if (l_is_inp_bf16 > 0) {
            libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, tmp_vreg, 0);
          }

          if ( compute_plain_vals_reduce > 0 ) {
            if ( is_sve ) {
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, reduce_instr,
                                                       tmp_vreg,
                                                       start_vreg_sum + im + _in * m_unroll_factor, 0,
                                                       start_vreg_sum + im + _in * m_unroll_factor,
                                                       pred_reg_all, sve_type );
            } else {
              libxsmm_aarch64_instruction_asimd_compute( io_generated_code, reduce_instr,
                                                         tmp_vreg,
                                                         start_vreg_sum + im + _in * m_unroll_factor, 0,
                                                         start_vreg_sum + im + _in * m_unroll_factor,
                                                         asimd_type );
            }
          }

          if ( compute_squared_vals_reduce > 0 ) {
            if ( is_sve ) {
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_P,
                                                       tmp_vreg, tmp_vreg, 0, start_vreg_sum2 + im + _in * m_unroll_factor,
                                                       pred_reg_all, sve_type );
            } else {
              libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,
                                                         tmp_vreg, tmp_vreg, 0, start_vreg_sum2 + im + _in * m_unroll_factor,
                                                         asimd_type );
            }
          }
        }
        if (extra_ld_bytes > 0) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                        i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in, extra_ld_bytes );
        }
      }

      if (n_trips > 1) {
        libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, 1);
      }
    }

    if (peeled_n_trips > 0) {
      for (_in = 0; _in < peeled_n_trips; _in++) {
        int  extra_ld_bytes  = i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in - (peeled_m_trips * i_micro_kernel_config->datatype_size_in * vlen  - use_m_masking * (vlen - mask_count) * i_micro_kernel_config->datatype_size_in);
        for (im = 0; im < peeled_m_trips; im++) {
          mask_count2 = ((im == peeled_m_trips -1) && (use_m_masking > 0)) ? mask_count : 0;
          l_masked_elements = (l_is_inp_bf16 == 0) ? mask_count2 : (mask_count2 > 0) ? mask_count2 : vlen;
          pred_reg_mask_use = (l_is_inp_bf16 == 0) ? ((mask_count2 > 0) ? pred_reg_mask_in : pred_reg_all) : ((mask_count2 > 0) ? pred_reg_mask_in : pred_reg_all_bf16);
          libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0,
                                                            tmp_vreg, i_micro_kernel_config->datatype_size_in, l_masked_elements, 1, 0, pred_reg_mask_use );
          if (l_is_inp_bf16 > 0) {
            libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, tmp_vreg, 0);
          }
          if ( compute_plain_vals_reduce > 0 ) {
            if (is_sve){
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, reduce_instr,
                                                       tmp_vreg,
                                                       start_vreg_sum + im + _in * m_unroll_factor, 0,
                                                       start_vreg_sum + im + _in * m_unroll_factor,
                                                       pred_reg_all, sve_type );
            } else {
              libxsmm_aarch64_instruction_asimd_compute( io_generated_code, reduce_instr,
                                                         tmp_vreg,
                                                         start_vreg_sum + im + _in * m_unroll_factor, 0,
                                                         start_vreg_sum + im + _in * m_unroll_factor,
                                                         asimd_type );
            }
          }

          if ( compute_squared_vals_reduce > 0 ) {
            if (is_sve){
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_P,
                                                       tmp_vreg, tmp_vreg, 0, start_vreg_sum2 + im + _in * m_unroll_factor,
                                                       pred_reg_all, sve_type );
            } else {
              libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,
                                                         tmp_vreg, tmp_vreg, 0, start_vreg_sum2 + im + _in * m_unroll_factor,
                                                         asimd_type );
            }
          }
        }
        if (extra_ld_bytes > 0) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                        i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in, extra_ld_bytes );
        }
      }
    }

    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                  i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in,
                                                  (long long)n * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in - (long long)peeled_m_trips * i_micro_kernel_config->datatype_size_in * vlen  - use_m_masking * ((long long)vlen - mask_count) * i_micro_kernel_config->datatype_size_in );

    for (_in = 1; _in < split_factor; _in++) {
      for (im = 0; im < peeled_m_trips; im++) {
        if ( compute_plain_vals_reduce > 0 ) {
          if (is_sve){
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, reduce_instr,
                                                     start_vreg_sum + im, start_vreg_sum + im + _in * m_unroll_factor,
                                                     0, start_vreg_sum + im, pred_reg_all, sve_type );
          } else {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, reduce_instr,
                                                       start_vreg_sum + im, start_vreg_sum + im + _in * m_unroll_factor,
                                                       0, start_vreg_sum + im, asimd_type );
          }
        }

        if ( compute_squared_vals_reduce > 0 ) {
          if (is_sve){
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, reduce_instr,
                                                     start_vreg_sum2 + im, start_vreg_sum2 + im + _in * m_unroll_factor, 0, start_vreg_sum2 + im,
                                                     pred_reg_all, sve_type );
          } else {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, reduce_instr,
                                                       start_vreg_sum2 + im, start_vreg_sum2 + im + _in * m_unroll_factor, 0, start_vreg_sum2 + im,
                                                       asimd_type );
          }
        }
      }
    }

    /* Store computed results */
    for (im = 0; im < peeled_m_trips; im++) {
      if ( compute_plain_vals_reduce > 0 ) {
        mask_count2 = ((im == peeled_m_trips -1) && (use_m_masking > 0)) ? mask_count : 0;
        l_masked_elements = (l_is_out_bf16 == 0) ? mask_count2 : (mask_count2 > 0) ? mask_count2 : vlen;
        pred_reg_mask_use = (l_is_out_bf16 == 0) ? ((mask_count2 > 0) ? pred_reg_mask_out : pred_reg_all) : ((mask_count2 > 0) ? pred_reg_mask_out : pred_reg_all_bf16);
        if (reduce_on_output > 0) {
          libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_reduced_elts, i_gp_reg_mapping->gp_reg_scratch_0,
                                                            tmp_vreg, i_micro_kernel_config->datatype_size_out, l_masked_elements, 0, 0, pred_reg_mask_use );
          if (l_is_out_bf16 > 0) {
            libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, tmp_vreg, 0);
          }
          if (is_sve){
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, reduce_instr,
                                                     tmp_vreg, start_vreg_sum + im,
                                                     0, start_vreg_sum + im, pred_reg_all,
                                                     sve_type );
          } else {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, reduce_instr,
                                                      tmp_vreg, start_vreg_sum + im,
                                                      0, start_vreg_sum + im, asimd_type );
          }
        }
        if (l_is_out_bf16 > 0) {
          libxsmm_generator_vcvt_f32bf16_aarch64( io_generated_code, start_vreg_sum + im, 0);
        }
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_reduced_elts,
          i_gp_reg_mapping->gp_reg_scratch_0, start_vreg_sum + im, i_micro_kernel_config->datatype_size_out,  l_masked_elements, 1, 1, pred_reg_mask_use );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        mask_count2 = ((im == peeled_m_trips -1) && (use_m_masking > 0)) ? mask_count : 0;
        l_masked_elements = (l_is_out_bf16 == 0) ? mask_count2 : (mask_count2 > 0) ? mask_count2 : vlen;
        pred_reg_mask_use = (l_is_out_bf16 == 0) ? ((mask_count2 > 0) ? pred_reg_mask_out : pred_reg_all) : ((mask_count2 > 0) ? pred_reg_mask_out : pred_reg_all_bf16);
        if (reduce_on_output > 0) {
          libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_reduced_elts_squared, i_gp_reg_mapping->gp_reg_scratch_0, tmp_vreg, i_micro_kernel_config->datatype_size_out, l_masked_elements, 0, 0, pred_reg_mask_use );
          if (l_is_out_bf16 > 0) {
            libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, tmp_vreg, 0);
          }
          if (is_sve){
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, reduce_instr,
                                                     tmp_vreg, start_vreg_sum2 + im,
                                                     0, start_vreg_sum2 + im, pred_reg_all,
                                                     sve_type );
          } else {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, reduce_instr,
                                                      tmp_vreg, start_vreg_sum2 + im,
                                                      0, start_vreg_sum2 + im, asimd_type );
          }
        }
        if (l_is_out_bf16 > 0) {
          libxsmm_generator_vcvt_f32bf16_aarch64( io_generated_code, start_vreg_sum2 + im, 0);
        }
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_reduced_elts_squared, i_gp_reg_mapping->gp_reg_scratch_0, start_vreg_sum2 + im, i_micro_kernel_config->datatype_size_out, l_masked_elements, 1, 1, pred_reg_mask_use );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_reduce_rows_aarch64_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {

  unsigned int im, m, n, m_trips, use_m_masking, compute_squared_vals_reduce, compute_plain_vals_reduce;
  unsigned int reduce_instr = 0, hreduce_instr = 0;
  unsigned int reg_sum = 31, reg_sum_squared = 30;
  unsigned int aux_vreg = 0;
  unsigned int cur_vreg = 0;
  unsigned int vlen = libxsmm_cpuid_vlen32(i_micro_kernel_config->instruction_set);
  unsigned int available_vregs = 30;
  unsigned int mask_count = 0, mask_count2;
  unsigned int flag_reduce_elts = ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD) ||
                                   (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD) ||
                                   (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX)) ? 1 : 0;
  unsigned int flag_reduce_elts_sq = ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X2_OP_ADD) ||
                                   (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD)) ? 1 : 0;
  unsigned int flag_reduce_op_add =((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD) ||
                                   (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD) ||
                                   (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X2_OP_ADD)) ? 1 : 0;
  unsigned int flag_reduce_op_max = (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX) ? 1 : 0;
  unsigned int reduce_on_output   = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_INIT_ACC) > 0 ) ? 1 : 0;
  unsigned char is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  libxsmm_aarch64_sve_type sve_type = libxsmm_generator_aarch64_get_sve_type(LIBXSMM_TYPESIZE(libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)));
  libxsmm_aarch64_asimd_tupletype asimd_type = (i_micro_kernel_config->datatype_size_in == 8) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S;
  unsigned char pred_reg_all = 0; /* defined by caller */
  unsigned char pred_reg_mask = 1;
  unsigned char pred_reg_all_bf16 = 2;
  unsigned char pred_reg_mask_compute_f32 = 3;
  unsigned char pred_reg_mask_use = 0;
  unsigned int l_masked_elements = 0;
  unsigned int l_is_inp_bf16 = (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ? 1 : 0;
  unsigned int l_is_out_bf16 = (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ? 1 : 0;
  unsigned int reduce_on_output_instr = 0;

  if (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
    vlen = vlen/2;
  }

  /* Some rudimentary checking of M, N and LDs*/
  if ( i_mateltwise_desc->m > i_mateltwise_desc->ldi ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
    return;
  }

  if ( flag_reduce_elts > 0 ) {
    compute_plain_vals_reduce= 1;
  } else {
    compute_plain_vals_reduce= 0;
  }

  if ( flag_reduce_elts_sq > 0 ) {
    compute_squared_vals_reduce = 1;
  } else {
    compute_squared_vals_reduce = 0;
  }

  if ( flag_reduce_op_add > 0 ) {
    reduce_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FADD_V : LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V;
    hreduce_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FADDV_V_P : LIBXSMM_AARCH64_INSTR_ASIMD_FADDP_V; /* sve todo: what is that instruction? probably reduce... */
  } else if ( flag_reduce_op_max > 0 ) {
    reduce_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FMAX_V_P : LIBXSMM_AARCH64_INSTR_ASIMD_FMAX_V;
    hreduce_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FMAXV_V_P : LIBXSMM_AARCH64_INSTR_ASIMD_FMAXP_V;
  } else {
    /* This should not happen */
    printf("Only supported reduction OPs are ADD and MAX for this reduce kernel\n");
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  if ( flag_reduce_op_add > 0 ) {
    reduce_on_output_instr =LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V;
  } else {
    reduce_on_output_instr =LIBXSMM_AARCH64_INSTR_ASIMD_FMAX_V;
  }

  if ((compute_squared_vals_reduce > 0) && (flag_reduce_op_add == 0)) {
    /* This should not happen */
    printf("Support for squares's reduction only when reduction OP is ADD\n");
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in                   = LIBXSMM_AARCH64_GP_REG_X9;
  i_gp_reg_mapping->gp_reg_reduced_elts         = LIBXSMM_AARCH64_GP_REG_X10;
  i_gp_reg_mapping->gp_reg_reduced_elts_squared = LIBXSMM_AARCH64_GP_REG_X11;
  i_gp_reg_mapping->gp_reg_m_loop               = LIBXSMM_AARCH64_GP_REG_X12;
  i_gp_reg_mapping->gp_reg_n_loop               = LIBXSMM_AARCH64_GP_REG_X13;
  i_gp_reg_mapping->gp_reg_scratch_0            = LIBXSMM_AARCH64_GP_REG_X14;

  /* load the input pointer and output pointer */
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
    i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 32, i_gp_reg_mapping->gp_reg_in );

  if ( compute_plain_vals_reduce > 0 ) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
      i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 64, i_gp_reg_mapping->gp_reg_reduced_elts );
    if ( compute_squared_vals_reduce > 0 ) {
      unsigned int result_size = i_mateltwise_desc->n * i_micro_kernel_config->datatype_size_out;
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
        i_gp_reg_mapping->gp_reg_reduced_elts, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_reduced_elts_squared, result_size );
    }
  } else {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
      i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 64, i_gp_reg_mapping->gp_reg_reduced_elts_squared );
  }

  /* In this case we do not support the algorithm with "on the fly transpose" */
  m = i_mateltwise_desc->m;
  n = i_mateltwise_desc->n;
  assert(0 != vlen);
  use_m_masking = ( m % vlen == 0 ) ? 0 : 1;
  m_trips = ( m+vlen-1 )/ vlen;
  im = 0;

  /* set pred reg 0 to true for sve */
  if ( is_sve ) {
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, pred_reg_all, -1, i_gp_reg_mapping->gp_reg_scratch_0 );
  }

  if (use_m_masking == 1) {
    mask_count = m % vlen;
    if ( is_sve ) {
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, pred_reg_mask, i_micro_kernel_config->datatype_size_in * mask_count, i_gp_reg_mapping->gp_reg_scratch_0 );
      if (l_is_inp_bf16 > 0 || l_is_out_bf16 > 0) {
        libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, pred_reg_mask_compute_f32, 4 * mask_count, i_gp_reg_mapping->gp_reg_scratch_0 );
      } else {
        pred_reg_mask_compute_f32 = pred_reg_mask;
      }
    }
  }

  if ((is_sve > 0) && (l_is_inp_bf16 > 0 || l_is_out_bf16 > 0)) {
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, pred_reg_all_bf16, (l_is_inp_bf16 > 0) ? i_micro_kernel_config->datatype_size_in * vlen : i_micro_kernel_config->datatype_size_out * vlen, i_gp_reg_mapping->gp_reg_scratch_0 );
  }

  if ((use_m_masking > 0) && ( flag_reduce_op_max > 0 )) {
    aux_vreg = available_vregs - 1;
    available_vregs--;
    if (!is_sve){/* vreg masks are not used in sve */
      if (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
        if (mask_count == 1) {
          unsigned int mask_array[4] = { 0x0, 0x0, 0x0, 0xfff00000};
          libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(aux_vreg), i_gp_reg_mapping->gp_reg_scratch_0, LIBXSMM_AARCH64_GP_REG_X12, mask_array, 0 );
        }
      } else {
        if (mask_count == 1) {
          unsigned int mask_array[4] = { 0x0, 0xff800000, 0xff800000, 0xff800000};
          libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(aux_vreg), i_gp_reg_mapping->gp_reg_scratch_0, LIBXSMM_AARCH64_GP_REG_X12, mask_array, 0 );
        } else if (mask_count == 2) {
          unsigned int mask_array[4] = { 0x0, 0x0, 0xff800000, 0xff800000};
          libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(aux_vreg), i_gp_reg_mapping->gp_reg_scratch_0, LIBXSMM_AARCH64_GP_REG_X12, mask_array, 0 );
        } else if (mask_count == 3) {
          unsigned int mask_array[4] = { 0x0, 0x0, 0x0, 0xff800000};
          libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(aux_vreg), i_gp_reg_mapping->gp_reg_scratch_0, LIBXSMM_AARCH64_GP_REG_X12, mask_array, 0 );
        }
      }
    }
  }

  if (n > 1) {
    /* open n loop */
    libxsmm_generator_loop_header_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, n);
  }

  /* Initialize accumulators to zero */
  if ( compute_plain_vals_reduce > 0 ) {
    if ( flag_reduce_op_add > 0 ) {
      if (is_sve){
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                reg_sum, reg_sum, 0, reg_sum, 0, sve_type );
      } else {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                reg_sum, reg_sum, 0, reg_sum, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
      }
    } else if ( flag_reduce_op_max > 0 ) {
      mask_count2 = ((im == m_trips-1) && (use_m_masking > 0)) ? mask_count : 0;
      l_masked_elements = (l_is_inp_bf16 == 0) ? mask_count2 : (mask_count2 > 0) ? mask_count2 : vlen;
      pred_reg_mask_use = (l_is_inp_bf16 == 0) ? ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all) : ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all_bf16);
      libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_in,
        i_gp_reg_mapping->gp_reg_scratch_0, reg_sum, i_micro_kernel_config->datatype_size_in,
        l_masked_elements, 0, 0, pred_reg_mask_use );
      if (l_is_inp_bf16 > 0) {
        libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, reg_sum, 0);
      }
      if ((use_m_masking == 1) && (im == m_trips - 1)) {
        if (is_sve) {
          /* not needed in sve */
          /*libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
            aux_vreg, reg_sum, 0, reg_sum, 0, sve_type );*/
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
            aux_vreg, reg_sum, 0, reg_sum, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
        }
      }
    }
  }

  if ( compute_squared_vals_reduce > 0 ) {
    if ( is_sve ) {
      libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                reg_sum_squared, reg_sum_squared, 0, reg_sum_squared, 0, sve_type );
    } else {
      libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                reg_sum_squared, reg_sum_squared, 0, reg_sum_squared, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
    }
  }

  if ( is_sve ) {
    for (im = 0; im < m_trips; im++) {
      cur_vreg = im % available_vregs;
      mask_count2 = ((im == m_trips-1) && (use_m_masking > 0)) ? mask_count : 0;
      l_masked_elements = (l_is_inp_bf16 == 0) ? mask_count2 : (mask_count2 > 0) ? mask_count2 : vlen;
      pred_reg_mask_use = (l_is_inp_bf16 == 0) ? ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all) : ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all_bf16);
      libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0,
          cur_vreg, i_micro_kernel_config->datatype_size_in, l_masked_elements, 1, 0, pred_reg_mask_use );
      if (l_is_inp_bf16 > 0) {
        libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, cur_vreg, 0);
      }
      if ((use_m_masking == 1) && (im == m_trips - 1) && (flag_reduce_op_max > 0)) {
        /* not needed in sve */
        /*libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
          aux_vreg, cur_vreg, 0, cur_vreg, 0, LIBXSMM_AARCH64_SVE_TYPE_S );*/
      }

      if ( compute_plain_vals_reduce > 0 ) {
        unsigned int pred_reg_compute = pred_reg_all;
        if ((flag_reduce_op_max > 0) && (im == m_trips-1) && (use_m_masking > 0)) {
          if (l_is_inp_bf16 > 0 || l_is_out_bf16 > 0) {
            pred_reg_compute = pred_reg_mask_compute_f32;
          } else {
            pred_reg_compute = pred_reg_mask;
          }
        }

        libxsmm_aarch64_instruction_sve_compute( io_generated_code, reduce_instr,
          cur_vreg, reg_sum, 0, reg_sum,
          pred_reg_compute, sve_type );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_P,
          cur_vreg, cur_vreg, 0, reg_sum_squared,
          pred_reg_all, sve_type );
      }
    }
  } else {
    for (im = 0; im < m_trips; im++) {
      cur_vreg = im % available_vregs;
      mask_count2 = ((im == m_trips-1) && (use_m_masking > 0)) ? mask_count : 0;
      l_masked_elements = (l_is_inp_bf16 == 0) ? mask_count2 : (mask_count2 > 0) ? mask_count2 : vlen;
      pred_reg_mask_use = (l_is_inp_bf16 == 0) ? ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all) : ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all_bf16);
      libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0,
          cur_vreg, i_micro_kernel_config->datatype_size_in, l_masked_elements, 1, 0, pred_reg_mask_use );
      if (l_is_inp_bf16 > 0) {
        libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, cur_vreg, 0);
      }
      if ((use_m_masking == 1) && (im == m_trips - 1) && (flag_reduce_op_max > 0)) {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
          aux_vreg, cur_vreg, 0, cur_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
      }

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, reduce_instr,
          cur_vreg, reg_sum, 0, reg_sum, asimd_type );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,
          cur_vreg, cur_vreg, 0, reg_sum_squared, asimd_type );
      }
    }
  }

  /* Now last horizontal reduction and store of the result... */
  if ( compute_plain_vals_reduce > 0 ) {
    if ( is_sve ) {/* reduces all values into one; places result into asimd register */
      /* if m_trips is 1 && use_m_masking, then the end of the vreg is set to zero */
      /* that's ok for add, but deadly for max of negative numbers */
      libxsmm_aarch64_instruction_sve_compute(io_generated_code, hreduce_instr,
        reg_sum, reg_sum, 0, reg_sum,
        m_trips == 1 && use_m_masking > 0 ? pred_reg_mask_compute_f32 : pred_reg_all, sve_type );
    } else {
      libxsmm_generator_hinstrps_aarch64(io_generated_code, hreduce_instr, reg_sum, asimd_type);
    }
    if (reduce_on_output > 0) {
      cur_vreg = 0;
      if (l_is_out_bf16 == 0) {
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
            i_gp_reg_mapping->gp_reg_reduced_elts, LIBXSMM_AARCH64_GP_REG_UNDEF,
            0, cur_vreg, (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ? LIBXSMM_AARCH64_ASIMD_WIDTH_D : LIBXSMM_AARCH64_ASIMD_WIDTH_S );
      } else {
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
            i_gp_reg_mapping->gp_reg_reduced_elts, LIBXSMM_AARCH64_GP_REG_UNDEF,
            0, cur_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_H );
        libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, cur_vreg, 0);
      }
      libxsmm_aarch64_instruction_asimd_compute( io_generated_code, reduce_on_output_instr,
        cur_vreg, reg_sum, 0, reg_sum, asimd_type );
    }
    if (l_is_out_bf16 == 0) {
      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
          i_gp_reg_mapping->gp_reg_reduced_elts, LIBXSMM_AARCH64_GP_REG_UNDEF,
          i_micro_kernel_config->datatype_size_out, reg_sum, (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ? LIBXSMM_AARCH64_ASIMD_WIDTH_D : LIBXSMM_AARCH64_ASIMD_WIDTH_S  );
    } else {
      libxsmm_generator_vcvt_f32bf16_aarch64( io_generated_code, reg_sum, 0);
      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
          i_gp_reg_mapping->gp_reg_reduced_elts, LIBXSMM_AARCH64_GP_REG_UNDEF,
          i_micro_kernel_config->datatype_size_out, reg_sum, LIBXSMM_AARCH64_ASIMD_WIDTH_H );
    }
  }

  if ( compute_squared_vals_reduce > 0 ) {
    if ( is_sve ) {/* reduces all values into one; places result into asimd register */
      libxsmm_aarch64_instruction_sve_compute(io_generated_code, hreduce_instr,
        reg_sum_squared, reg_sum_squared, 0, reg_sum_squared,
        m_trips == 1 && use_m_masking > 0 ? pred_reg_mask_compute_f32 : pred_reg_all, sve_type );
    } else {
      libxsmm_generator_hinstrps_aarch64(io_generated_code, hreduce_instr, reg_sum_squared, asimd_type);
    }
    if (reduce_on_output > 0) {
      cur_vreg = 0;
      if (l_is_out_bf16 == 0) {
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
            i_gp_reg_mapping->gp_reg_reduced_elts_squared, LIBXSMM_AARCH64_GP_REG_UNDEF,
            0, cur_vreg, (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ? LIBXSMM_AARCH64_ASIMD_WIDTH_D : LIBXSMM_AARCH64_ASIMD_WIDTH_S  );
      } else {
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
            i_gp_reg_mapping->gp_reg_reduced_elts_squared, LIBXSMM_AARCH64_GP_REG_UNDEF,
            0, cur_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_H );
        libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, cur_vreg, 0);
      }
      libxsmm_aarch64_instruction_asimd_compute( io_generated_code, reduce_on_output_instr,
        cur_vreg, reg_sum_squared, 0, reg_sum_squared, asimd_type );
    }
    if (l_is_out_bf16 == 0) {
      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
          i_gp_reg_mapping->gp_reg_reduced_elts_squared, LIBXSMM_AARCH64_GP_REG_UNDEF,
          i_micro_kernel_config->datatype_size_out, reg_sum_squared, (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ? LIBXSMM_AARCH64_ASIMD_WIDTH_D : LIBXSMM_AARCH64_ASIMD_WIDTH_S  );
    } else {
      libxsmm_generator_vcvt_f32bf16_aarch64( io_generated_code, reg_sum_squared, 0);
      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
          i_gp_reg_mapping->gp_reg_reduced_elts_squared, LIBXSMM_AARCH64_GP_REG_UNDEF,
          i_micro_kernel_config->datatype_size_out, reg_sum_squared, LIBXSMM_AARCH64_ASIMD_WIDTH_H );
    }
  }

  if (n > 1) {
    /* Adjust input and output pointer */
    int extra_ld_bytes  = i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in - (m_trips * vlen * i_micro_kernel_config->datatype_size_in - use_m_masking * (vlen - mask_count) * i_micro_kernel_config->datatype_size_in);
    if (extra_ld_bytes > 0) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
          i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in, extra_ld_bytes );
    }
    /* close n loop */
    libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, 1);
  }

  return;
}

LIBXSMM_API_INTERN
void libxsmm_generator_getval_stack_var_aarch64( libxsmm_generated_code* io_generated_code,
    int                                 offset,
    unsigned int                        i_gp_reg );
LIBXSMM_API_INTERN
void libxsmm_generator_getval_stack_var_aarch64( libxsmm_generated_code* io_generated_code,
    int                                 offset,
    unsigned int                        i_gp_reg ) {
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I, LIBXSMM_AARCH64_GP_REG_X29, i_gp_reg, -offset, 0 );
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_gp_reg );
}

LIBXSMM_API_INTERN
void libxsmm_generator_setval_stack_var_aarch64( libxsmm_generated_code* io_generated_code,
    int                                 offset,
    unsigned int                        i_gp_reg,
    unsigned int                        i_gp_reg_aux);
LIBXSMM_API_INTERN
void libxsmm_generator_setval_stack_var_aarch64( libxsmm_generated_code* io_generated_code,
    int                                 offset,
    unsigned int                        i_gp_reg,
    unsigned int                        i_gp_reg_aux) {
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I, LIBXSMM_AARCH64_GP_REG_X29, i_gp_reg_aux, -offset, 0 );
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF, i_gp_reg_aux, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_gp_reg );
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_extract_mask2_from_mask4( libxsmm_generated_code* io_generated_code,
    unsigned int input_vec_mask,
    unsigned int output_vec_mask,
    unsigned int lohi);
LIBXSMM_API_INTERN
void libxsmm_aarch64_extract_mask2_from_mask4( libxsmm_generated_code* io_generated_code,
    unsigned int input_vec_mask,
    unsigned int output_vec_mask,
    unsigned int lohi) {
  unsigned char is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  if (is_sve > 0){
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, lohi == 0 ? LIBXSMM_AARCH64_INSTR_SVE_ZIP_P_L : LIBXSMM_AARCH64_INSTR_SVE_ZIP_P_H,
                                             input_vec_mask, input_vec_mask, 0, output_vec_mask, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
  } else {
    libxsmm_aarch64_instruction_asimd_compute( io_generated_code, lohi == 0 ? LIBXSMM_AARCH64_INSTR_ASIMD_ZIP1 : LIBXSMM_AARCH64_INSTR_ASIMD_ZIP2,
                                               input_vec_mask, input_vec_mask, 0, output_vec_mask, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_opreduce_vecs_index_aarch64_microkernel_block( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    libxsmm_meltw_descriptor*                      i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_opreduce_vecs_index_aarch64_microkernel_block( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    libxsmm_meltw_descriptor*                      i_mateltwise_desc ) {

  unsigned int m, im, _im, use_m_masking, m_trips, max_m_unrolling = 4, m_unroll_factor = 1, m_trips_loop = 0, peeled_m_trips = 0, vecin_offset = 0, vecidxin_offset = 0, vecout_offset = 0, temp_vreg = 31, use_stack_vars = 0;
  unsigned char is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  unsigned int vlen = libxsmm_cpuid_vlen32(i_micro_kernel_config->instruction_set);
  unsigned int mask_count = 1, mask_count2;
  unsigned int mask_argidx64 = 2;
  unsigned int use_indexed_vec = 0, use_indexed_vecidx = 1;
  unsigned int use_implicitly_indexed_vec = 0;
  unsigned int use_implicitly_indexed_vecidx = 0;
  unsigned int idx_tsize = i_mateltwise_desc->n;
  unsigned int in_tsize = 4;
  unsigned int vecidx_ind_base_param_offset = 8;
  unsigned int vecidx_in_base_param_offset = 16;
  unsigned int temp_vreg_argop0 = 30, temp_vreg_argop1 = 29;
  unsigned int record_argop_off_vec0 = 0, record_argop_off_vec1 = 0;
  unsigned int vecargop_vec0_offset = 0, vecargop_vec1_offset = 0;
  int rbp_offset_argop_ptr0 = -16;
  int rbp_offset_argop_ptr1 = -24;
  int rbp_offset_neg_inf    = -32;
  int pos_rbp_offset_neg_inf = 32;
  /* for sve, the mask registers are predicate registers */
  unsigned int argop_mask = is_sve ? 1 : 28;
  unsigned int argop_mask_aux = is_sve ? 2 : 31;
  unsigned int argop_cmp_instr = 0;
  unsigned int argop_blend_instr = 0;
  unsigned int bcast_loops = 0;
  unsigned int gp_reg_bcast_loop = 0;
  unsigned int temp_gpr = 0;
  unsigned int scratch_gpr = 0;
  unsigned int max_m_unrolling_index = (idx_tsize == 8) ? 2 * max_m_unrolling : max_m_unrolling;
  int load_acc = 1;
  int apply_op = 0, apply_redop = 0, op_order = -1, op_instr = 0, reduceop_instr = 0, scale_op_result = 0;
  unsigned int END_LABEL = 0;
  const int LIBXSMM_AARCH64_INSTR_DOTPS = -1;
  unsigned int  bcast_param = 0;
  unsigned int gp_reg_index_64bit = 0, gp_reg_index = 0, gp_reg_ldi = 0, gp_reg_impl_index = 0;
  unsigned char pred_reg_all = 0; /* sve predicate registers with all elements enabled; set by caller */
  unsigned char pred_reg_mask = 3;
  unsigned char pred_reg_argidx = 4;
  unsigned char pred_reg_all_bf16 = 5;
  unsigned char pred_reg_argidx32 = 6;
  unsigned char pred_reg_mask_use = 0;
  unsigned int l_masked_elements = 0;
  unsigned int l_is_inp_bf16 = (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ? 1 : 0;
  unsigned int l_is_out_bf16 = (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ? 1 : 0;
  const char *const env_load_acc= getenv("LOAD_ACCS_OPREDUCE_VECS_IDX");

#if defined(LIBXSMM_GENERATOR_MATELTWISE_REDUCE_AARCH64_JUMP_LABEL_TRACKER_MALLOC)
  libxsmm_jump_label_tracker* const p_jump_label_tracker = (libxsmm_jump_label_tracker*)malloc(sizeof(libxsmm_jump_label_tracker));
#else
  libxsmm_jump_label_tracker l_jump_label_tracker;
  libxsmm_jump_label_tracker* const p_jump_label_tracker = &l_jump_label_tracker;
#endif
  libxsmm_reset_jump_label_tracker(p_jump_label_tracker);

  if ((i_mateltwise_desc->param & 0x1) > 0) {
    record_argop_off_vec0 = 1;
    use_stack_vars = 1;
  }

  if ((i_mateltwise_desc->param & 0x2) > 0) {
    record_argop_off_vec1 = 1;
    use_stack_vars = 1;
  }

  bcast_param = (unsigned int) (i_mateltwise_desc->param >> 2);

  if ((i_mateltwise_desc->param & 0x1) > 0) {
    vecargop_vec0_offset = 0;
  }
  if ((i_mateltwise_desc->param & 0x2) > 0) {
    vecargop_vec1_offset = (record_argop_off_vec0 == 1) ? vecargop_vec0_offset + max_m_unrolling_index : 0;
  }
  vecout_offset   = ((record_argop_off_vec0 == 1) || (record_argop_off_vec1 == 1)) ? LIBXSMM_MAX(vecargop_vec0_offset, vecargop_vec1_offset) + max_m_unrolling_index : 0;
  vecin_offset    = vecout_offset + max_m_unrolling;
  vecidxin_offset = vecin_offset + max_m_unrolling;

  if ( 0 == env_load_acc ) {
  } else {
    load_acc = atoi(env_load_acc);
  }
  if (i_micro_kernel_config->opreduce_avoid_acc_load > 0) {
    load_acc = 0;
  }

  /* TODO: Add validation checks for various JIT params and error out... */
  if (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) {
    in_tsize = 4;
  } else if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) {
    in_tsize = 2;
  } else {
#if defined(LIBXSMM_GENERATOR_MATELTWISE_REDUCE_AARCH64_JUMP_LABEL_TRACKER_MALLOC)
    free(p_jump_label_tracker);
#endif
    /* This should not happen */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }

  if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_INDEXED_VEC) > 0) {
    use_indexed_vec = 1;
  }

  if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_IMPLICIT_INDEXED_VEC) > 0) {
    use_indexed_vec = 1;
    use_implicitly_indexed_vec = 1;
  }

  if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_IMPLICIT_INDEXED_VECIDX) > 0) {
    use_implicitly_indexed_vecidx = 1;
  }

  i_gp_reg_mapping->gp_reg_n        = LIBXSMM_AARCH64_GP_REG_X9;
  i_gp_reg_mapping->gp_reg_ind_base = LIBXSMM_AARCH64_GP_REG_X10;
  i_gp_reg_mapping->gp_reg_in_base  = LIBXSMM_AARCH64_GP_REG_X11;
  i_gp_reg_mapping->gp_reg_out      = LIBXSMM_AARCH64_GP_REG_X12;
  i_gp_reg_mapping->gp_reg_m_loop   = LIBXSMM_AARCH64_GP_REG_X13;
  i_gp_reg_mapping->gp_reg_n_loop   = LIBXSMM_AARCH64_GP_REG_X14;
  i_gp_reg_mapping->gp_reg_in       = LIBXSMM_AARCH64_GP_REG_X2;
  i_gp_reg_mapping->gp_reg_invec    = LIBXSMM_AARCH64_GP_REG_X15;

  i_gp_reg_mapping->gp_reg_in_base2  = LIBXSMM_AARCH64_GP_REG_X15;
  i_gp_reg_mapping->gp_reg_ind_base2 = LIBXSMM_AARCH64_GP_REG_X3;
  i_gp_reg_mapping->gp_reg_in2       = LIBXSMM_AARCH64_GP_REG_X4;
  gp_reg_bcast_loop                  = LIBXSMM_AARCH64_GP_REG_X5;

  gp_reg_index                        = ( idx_tsize == 8 ) ? LIBXSMM_AARCH64_GP_REG_X16 : LIBXSMM_AARCH64_GP_REG_W16;
  gp_reg_index_64bit                  = LIBXSMM_AARCH64_GP_REG_X16;
  scratch_gpr                         = LIBXSMM_AARCH64_GP_REG_X17;
  i_gp_reg_mapping->gp_reg_scratch_0  = scratch_gpr;
  temp_gpr                            = LIBXSMM_AARCH64_GP_REG_X6;
  gp_reg_ldi                          = LIBXSMM_AARCH64_GP_REG_X1;
  i_gp_reg_mapping->gp_reg_scale_base = LIBXSMM_AARCH64_GP_REG_X7;

  if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_IMPLICIT_INDEXED_VEC) > 0) {
    gp_reg_impl_index = i_gp_reg_mapping->gp_reg_ind_base2;
  }

  if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_IMPLICIT_INDEXED_VECIDX) > 0) {
    gp_reg_impl_index = i_gp_reg_mapping->gp_reg_ind_base;
  }

  if (i_micro_kernel_config->opreduce_use_unary_arg_reading > 0) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 48, i_gp_reg_mapping->gp_reg_n );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_n, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_gp_reg_mapping->gp_reg_n );
  } else {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_gp_reg_mapping->gp_reg_n );
  }
  libxsmm_aarch64_instruction_cond_jump_to_label( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_CBZ, i_gp_reg_mapping->gp_reg_n, END_LABEL, p_jump_label_tracker );

  if (use_stack_vars > 0) {
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_X29,  0, 0 );
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 40, 0 );
  }

  if ( record_argop_off_vec0 == 1 ) {
    if (i_micro_kernel_config->opreduce_use_unary_arg_reading > 0) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 72, temp_gpr );
    } else {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 64, temp_gpr );
    }
    libxsmm_generator_setval_stack_var_aarch64( io_generated_code, rbp_offset_argop_ptr0, temp_gpr, scratch_gpr);
  }

  if ( record_argop_off_vec1 == 1 ) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 72, temp_gpr );
    libxsmm_generator_setval_stack_var_aarch64( io_generated_code, rbp_offset_argop_ptr1, temp_gpr, scratch_gpr);
  }

  if (((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_COPY) > 0) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OPORDER_VECIN_VECIDX) > 0)) {
    vecidx_ind_base_param_offset = 48;
    vecidx_in_base_param_offset = 56;
    if (record_argop_off_vec1 > 0) {
      record_argop_off_vec0 = 1;
      record_argop_off_vec1 = 0;
      rbp_offset_argop_ptr0 = rbp_offset_argop_ptr1;
    }
  }

  if (use_implicitly_indexed_vecidx == 0) {
    if (i_micro_kernel_config->opreduce_use_unary_arg_reading > 0) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                            i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF,
                                            40, i_gp_reg_mapping->gp_reg_ind_base );
    } else {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                            i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF,
                                            vecidx_ind_base_param_offset, i_gp_reg_mapping->gp_reg_ind_base );
    }
  }

  if (i_micro_kernel_config->opreduce_use_unary_arg_reading > 0) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                          i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF,
                                          32, i_gp_reg_mapping->gp_reg_in_base );
  } else {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                          i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF,
                                          vecidx_in_base_param_offset, i_gp_reg_mapping->gp_reg_in_base );
  }


  if (i_micro_kernel_config->opreduce_use_unary_arg_reading > 0) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                          i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF,
                                          64, i_gp_reg_mapping->gp_reg_out );
  } else {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                          i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF,
                                          32, i_gp_reg_mapping->gp_reg_out );
  }

  if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_SCALE_OP_RESULT) > 0) {
    scale_op_result = 1;
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                          i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF,
                                          40, i_gp_reg_mapping->gp_reg_scale_base );
  }

  if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_COPY) == 0) {
    apply_op = 1;
    if (use_indexed_vec > 0) {
      if (use_implicitly_indexed_vec == 0) {
        libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                              i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF,
                                              48, i_gp_reg_mapping->gp_reg_ind_base2 );
      }
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                            i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF,
                                            56, i_gp_reg_mapping->gp_reg_in_base2 );
    } else {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                            i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF,
                                            24, i_gp_reg_mapping->gp_reg_invec );
    }
  }

  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_EOR_SR, gp_reg_index_64bit, gp_reg_index_64bit, gp_reg_index_64bit, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, gp_reg_ldi, (long long)i_mateltwise_desc->ldi * in_tsize);

  if (apply_op == 1) {
    if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OPORDER_VECIN_VECIDX) > 0) {
      op_order = 1;
    } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OPORDER_VECIDX_VECIN) > 0) {
      op_order = 0;
    } else {
#if defined(LIBXSMM_GENERATOR_MATELTWISE_REDUCE_AARCH64_JUMP_LABEL_TRACKER_MALLOC)
      free(p_jump_label_tracker);
#endif
      /* This should not happen */
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }

    if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_ADD) > 0) {
      op_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FADD_V : LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V;
    } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_SUB) > 0) {
      op_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FSUB_V : LIBXSMM_AARCH64_INSTR_ASIMD_FSUB_V;
    } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_MUL) > 0) {
      op_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FMUL_V : LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V;
    } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_DIV) > 0) {
      op_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FDIV_V_P : LIBXSMM_AARCH64_INSTR_ASIMD_FDIV_V;
    } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_DOT) > 0) {
      /* not supported currently */
      op_instr = LIBXSMM_AARCH64_INSTR_DOTPS;
    } else {
#if defined(LIBXSMM_GENERATOR_MATELTWISE_REDUCE_AARCH64_JUMP_LABEL_TRACKER_MALLOC)
      free(p_jump_label_tracker);
#endif
      /* This should not happen */
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  }

  if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_NONE) == 0) {
    apply_redop = 1;
    if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_SUM) > 0) {
      reduceop_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FADD_V : LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V;
    } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_MAX) > 0) {
      reduceop_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FMAX_V_P : LIBXSMM_AARCH64_INSTR_ASIMD_FMAX_V;
      argop_cmp_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FCMLE_P_V : LIBXSMM_AARCH64_INSTR_ASIMD_FCMGE_R_V;
      argop_blend_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_SEL_V_P : LIBXSMM_AARCH64_INSTR_ASIMD_BIT_V;
    } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_MIN) > 0) {
      reduceop_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FMIN_V_P : LIBXSMM_AARCH64_INSTR_ASIMD_FMIN_V;
      argop_cmp_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FCMGT_P_V : LIBXSMM_AARCH64_INSTR_ASIMD_FCMGT_R_V;
      argop_blend_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_SEL_V_P : LIBXSMM_AARCH64_INSTR_ASIMD_BIF_V;
    } else {
#if defined(LIBXSMM_GENERATOR_MATELTWISE_REDUCE_AARCH64_JUMP_LABEL_TRACKER_MALLOC)
      free(p_jump_label_tracker);
#endif
      /* This should not happen */
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else {
    vecin_offset = vecout_offset;
    if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_COPY) > 0) {
      vecidxin_offset = vecout_offset;
    }
  }

  m = i_mateltwise_desc->m;
  if (bcast_param > 0) {
    if (bcast_param > m) {
      bcast_param = m;
    }
    m = bcast_param;
    bcast_loops = i_mateltwise_desc->m / bcast_param;
  }
  use_m_masking     = ( m % vlen == 0 ) ? 0 : 1;
  m_trips           = LIBXSMM_UPDIV(m, vlen);
  m_unroll_factor   = m_trips;
  m_trips_loop      = 1;
  peeled_m_trips    = 0;

  /* set pred reg 0 to true for sve */
  if (is_sve) {
    libxsmm_generator_set_p_register_aarch64_sve(io_generated_code, 0, -1, 0);
  }

  if (use_m_masking == 1) {
    mask_count = m % vlen;
    if (is_sve) libxsmm_generator_set_p_register_aarch64_sve(io_generated_code, pred_reg_mask, i_micro_kernel_config->datatype_size_in * mask_count, i_gp_reg_mapping->gp_reg_scratch_0);
    if (idx_tsize == 4) {
      if (is_sve) libxsmm_generator_set_p_register_aarch64_sve(io_generated_code, pred_reg_argidx32, idx_tsize * mask_count, i_gp_reg_mapping->gp_reg_scratch_0);
    }
  }

  /* todo sve: why is the default for mask_argidx64 == 2 ? */
  LIBXSMM_ASSERT(0 != (vlen / 2));
  if ((idx_tsize == 8) && (m % (vlen/2) != 0) && ((record_argop_off_vec0 > 0) || (record_argop_off_vec1 > 0))) {
    mask_argidx64 = m % (vlen/2);
  }
  if (is_sve) libxsmm_generator_set_p_register_aarch64_sve(io_generated_code, pred_reg_argidx, idx_tsize * mask_argidx64, i_gp_reg_mapping->gp_reg_scratch_0);

  if ((is_sve > 0) && (l_is_inp_bf16 > 0 || l_is_out_bf16 > 0)) {
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, pred_reg_all_bf16, (l_is_inp_bf16 > 0) ? i_micro_kernel_config->datatype_size_in * vlen : i_micro_kernel_config->datatype_size_out * vlen, i_gp_reg_mapping->gp_reg_scratch_0 );
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

  if (apply_redop == 1) {
    if (load_acc == 0) {
      if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_MAX) > 0) {
        libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, temp_gpr, (long long)0xff800000);
        libxsmm_generator_setval_stack_var_aarch64( io_generated_code, rbp_offset_neg_inf, temp_gpr, scratch_gpr);
      }
    }
  }

  if (bcast_loops > 1) {
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, gp_reg_bcast_loop, bcast_loops );
  }

  if (m_trips_loop > 1) {
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, m_trips_loop );
  }

  if (m_trips_loop >= 1) {
    for (im = 0; im < m_unroll_factor; im++) {
      /* Load output for reduction */
      if (apply_redop == 1) {
        if (load_acc == 0) {
          if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_MAX) > 0) {
            if (im == 0) {
              libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I, LIBXSMM_AARCH64_GP_REG_X29, i_gp_reg_mapping->gp_reg_scratch_0, pos_rbp_offset_neg_inf, 0 );
              if (is_sve){
                libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF,
                                                       i_gp_reg_mapping->gp_reg_scratch_0, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, vecout_offset, pred_reg_all );
              } else {
                libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R,
                                                                 i_gp_reg_mapping->gp_reg_scratch_0, LIBXSMM_AARCH64_GP_REG_UNDEF, vecout_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
              }
            } else {
              if (is_sve){
                libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                                           vecout_offset, vecout_offset, 0, im + vecout_offset, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
              } else {
                libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                                           vecout_offset, vecout_offset, 0, im + vecout_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
              }
            }
          } else {
            if (is_sve){
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                im + vecout_offset, im + vecout_offset, 0, im + vecout_offset, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
            } else {
              libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                im + vecout_offset, im + vecout_offset, 0, im + vecout_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
            }
          }
        } else {
          l_masked_elements = (l_is_out_bf16 == 0) ? 0 : vlen;
          pred_reg_mask_use = (l_is_out_bf16 == 0) ? pred_reg_all : pred_reg_all_bf16;
          libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0,
              im + vecout_offset, i_micro_kernel_config->datatype_size_out, l_masked_elements, 1, 0, pred_reg_mask_use );
          if (l_is_out_bf16 > 0) {
            libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, im + vecout_offset, 0);
          }
        }
      }

      /* Initialize argop vectors if need be */
      if (record_argop_off_vec0 == 1) {
        if (is_sve){
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                                   im + vecargop_vec0_offset,
                                                   im + vecargop_vec0_offset, 0,
                                                   im + vecargop_vec0_offset, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
          if (idx_tsize == 8) {
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                                     im + max_m_unrolling + vecargop_vec0_offset,
                                                     im + max_m_unrolling + vecargop_vec0_offset, 0,
                                                     im + max_m_unrolling + vecargop_vec0_offset, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
          }
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                     im + vecargop_vec0_offset,
                                                     im + vecargop_vec0_offset, 0,
                                                     im + vecargop_vec0_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
          if (idx_tsize == 8) {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                       im + max_m_unrolling + vecargop_vec0_offset,
                                                       im + max_m_unrolling + vecargop_vec0_offset, 0,
                                                       im + max_m_unrolling + vecargop_vec0_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
          }
        }
      }

      /* Initialize argop vectors if need be */
      if (record_argop_off_vec1 == 1) {
        if (is_sve){
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                                   im + vecargop_vec1_offset,
                                                   im + vecargop_vec1_offset, 0,
                                                   im + vecargop_vec1_offset, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
          if (idx_tsize == 8) {
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                                     im + max_m_unrolling + vecargop_vec1_offset,
                                                     im + max_m_unrolling + vecargop_vec1_offset, 0,
                                                     im + max_m_unrolling + vecargop_vec1_offset, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
          }
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                     im + vecargop_vec1_offset,
                                                     im + vecargop_vec1_offset, 0,
                                                     im + vecargop_vec1_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
          if (idx_tsize == 8) {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                       im + max_m_unrolling + vecargop_vec1_offset,
                                                       im + max_m_unrolling + vecargop_vec1_offset, 0,
                                                       im + max_m_unrolling + vecargop_vec1_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
          }
        }
      }

      /* Load input vector in case we have to apply op */
      if (apply_op == 1) {
        if (use_indexed_vec == 0) {
          if (bcast_param == 0) {
            l_masked_elements = (l_is_inp_bf16 == 0) ? 0 : vlen;
            pred_reg_mask_use = (l_is_inp_bf16 == 0) ? pred_reg_all : pred_reg_all_bf16;
            libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_invec, scratch_gpr,
                im + vecin_offset, i_micro_kernel_config->datatype_size_out, l_masked_elements, 1, 0, pred_reg_mask_use );
            if (l_is_inp_bf16 > 0) {
              libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, im + vecin_offset, 0);
            }
          } else {
            if (im == 0) {
              libxsmm_generator_bcastload_masked_vreg_aarch64_asimd( io_generated_code, i_gp_reg_mapping->gp_reg_invec, scratch_gpr,
                  im + vecin_offset, i_micro_kernel_config->datatype_size_in, 0, 0 );
              if (l_is_inp_bf16 > 0) {
                libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, im + vecin_offset, 0);
              }
              if (is_sve){
                for (_im = 1; _im < m_unroll_factor; _im++) {
                  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                                           vecin_offset, vecin_offset, 0, _im + vecin_offset, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
                }
              } else {
                for (_im = 1; _im < m_unroll_factor; _im++) {
                  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                                             vecin_offset, vecin_offset, 0, _im + vecin_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
                }
              }
            }
          }
        }
      }
    }

    /* Adjust post advancements if need be */
    if (apply_redop == 1) {
      if (load_acc == 1) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
          i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out,
          (long long)m_unroll_factor * i_micro_kernel_config->datatype_size_out * vlen );
      }
    }
    if (apply_op == 1) {
      if (use_indexed_vec == 0) {
        if (bcast_param == 0) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
            i_gp_reg_mapping->gp_reg_invec, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_invec,
            (long long)m_unroll_factor * i_micro_kernel_config->datatype_size_out * vlen );
        }
      }
    }

    /* Perform the reductions for all columns */
    if (gp_reg_impl_index > 0) {
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_EOR_SR,
          gp_reg_impl_index, gp_reg_impl_index, gp_reg_impl_index, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    }

    libxsmm_generator_loop_header_gp_reg_bound_aarch64( io_generated_code, io_loop_label_tracker,
          i_gp_reg_mapping->gp_reg_n_loop, i_gp_reg_mapping->gp_reg_n );

    if (use_indexed_vecidx > 0) {
      if (use_implicitly_indexed_vecidx == 0) {
        libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_POST,
          i_gp_reg_mapping->gp_reg_ind_base, LIBXSMM_AARCH64_GP_REG_UNDEF, idx_tsize, gp_reg_index );
      } else {
        libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
          gp_reg_impl_index, gp_reg_impl_index, gp_reg_index_64bit, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
      }
      if (record_argop_off_vec0 == 1) {
        /* load 1 value from a gp reg, and store into vreg */
        if (is_sve){
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_DUP_GP_V,
              gp_reg_index_64bit, gp_reg_index_64bit, 0, temp_vreg_argop0, 0, (idx_tsize == 8) ? LIBXSMM_AARCH64_SVE_TYPE_D : LIBXSMM_AARCH64_SVE_TYPE_S );
        } else {
          libxsmm_aarch64_instruction_asimd_gpr_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_DUP_FULL,
              gp_reg_index_64bit, temp_vreg_argop0, 0, (idx_tsize == 8) ? LIBXSMM_AARCH64_ASIMD_WIDTH_D : LIBXSMM_AARCH64_ASIMD_WIDTH_S );
        }
      }
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MUL, gp_reg_index_64bit, gp_reg_ldi, i_gp_reg_mapping->gp_reg_in, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_in_base, i_gp_reg_mapping->gp_reg_in, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    }

    if (use_indexed_vec > 0) {
      if (use_implicitly_indexed_vec == 0) {
        libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_POST, i_gp_reg_mapping->gp_reg_ind_base2, LIBXSMM_AARCH64_GP_REG_UNDEF, idx_tsize, gp_reg_index );
      } else {
        libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ORR_SR, gp_reg_impl_index, gp_reg_impl_index, gp_reg_index_64bit, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
      }
      if (record_argop_off_vec1 == 1) {
        /* load 1 value from a gp reg, and store into vreg */
        if (is_sve){
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_DUP_GP_V,
              gp_reg_index_64bit, gp_reg_index_64bit, 0, temp_vreg_argop1, 0, (idx_tsize == 8) ? LIBXSMM_AARCH64_SVE_TYPE_D : LIBXSMM_AARCH64_SVE_TYPE_S );
        } else {
          libxsmm_aarch64_instruction_asimd_gpr_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_DUP_FULL,
              gp_reg_index_64bit, temp_vreg_argop1, 0, (idx_tsize == 8) ? LIBXSMM_AARCH64_ASIMD_WIDTH_D : LIBXSMM_AARCH64_ASIMD_WIDTH_S );
        }
      }
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MUL, gp_reg_index_64bit, gp_reg_ldi, i_gp_reg_mapping->gp_reg_in2, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_in_base2, i_gp_reg_mapping->gp_reg_in2, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    }

    for (im = 0; im < m_unroll_factor; im++) {
      /* First load the indexed vector */
      if (bcast_param == 0) {
        l_masked_elements = (l_is_inp_bf16 == 0) ? 0 : vlen;
        pred_reg_mask_use = (l_is_inp_bf16 == 0) ? pred_reg_all : pred_reg_all_bf16;
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_in, scratch_gpr,
            im + vecidxin_offset, i_micro_kernel_config->datatype_size_in, l_masked_elements, 1, 0, pred_reg_mask_use );
        if (l_is_inp_bf16 > 0) {
          libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, im + vecidxin_offset, 0);
        }
      } else {
        if (im == 0) {
          libxsmm_generator_bcastload_masked_vreg_aarch64_asimd( io_generated_code, i_gp_reg_mapping->gp_reg_in, scratch_gpr,
              im + vecidxin_offset, i_micro_kernel_config->datatype_size_in, 0, 0 );
          if (l_is_inp_bf16 > 0) {
            libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, im + vecidxin_offset, 0);
          }
          if (is_sve){
            for (_im = 1; _im < m_unroll_factor; _im++) {
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                                       vecidxin_offset, vecidxin_offset, 0, _im + vecidxin_offset, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
            }
          } else {
            for (_im = 1; _im < m_unroll_factor; _im++) {
              libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                                         vecidxin_offset, vecidxin_offset, 0, _im + vecidxin_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
            }
          }
        }
      }

      /* Now apply the OP among the indexed vector and the input vector */
      if (apply_op == 1) {
        if (use_indexed_vec > 0) {
          if (bcast_param == 0) {
            l_masked_elements = (l_is_inp_bf16 == 0) ? 0 : vlen;
            pred_reg_mask_use = (l_is_inp_bf16 == 0) ? pred_reg_all : pred_reg_all_bf16;
            libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_in2, scratch_gpr,
                im + vecin_offset, i_micro_kernel_config->datatype_size_in, l_masked_elements, 1, 0, pred_reg_mask_use );
            if (l_is_inp_bf16 > 0) {
              libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, im + vecin_offset, 0);
            }
          } else {
            if (im == 0) {
              libxsmm_generator_bcastload_masked_vreg_aarch64_asimd( io_generated_code, i_gp_reg_mapping->gp_reg_in2, scratch_gpr,
                  im + vecin_offset, i_micro_kernel_config->datatype_size_in, 0, 0 );
              if (l_is_inp_bf16 > 0) {
                libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, im + vecin_offset, 0);
              }
              if (is_sve){
                for (_im = 1; _im < m_unroll_factor; _im++) {
                  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                                           vecin_offset, vecin_offset, 0, _im + vecin_offset, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
                }
              } else {
                for (_im = 1; _im < m_unroll_factor; _im++) {
                  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                                             vecin_offset, vecin_offset, 0, _im + vecin_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
                }
              }
            }
          }
        }

        if (op_instr == LIBXSMM_AARCH64_INSTR_DOTPS) {
          /* TODO: Add DOT op sequence here */
        } else {
          if (is_sve){
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, op_instr,
              (op_order == 0)    ? im + vecidxin_offset : im + vecin_offset,
              (op_order == 0)    ? im + vecin_offset    : im + vecidxin_offset, 0,
              (apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
          } else {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, op_instr,
              (op_order == 0)    ? im + vecidxin_offset : im + vecin_offset,
              (op_order == 0)    ? im + vecin_offset    : im + vecidxin_offset, 0,
              (apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
          }
        }
      }

      if (scale_op_result == 1) {
        if ( is_sve ) {
          /* ld1r = load 1 elements, and broadcast to all */
          if (l_is_inp_bf16 == 0) {
            libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF,
                                                  i_gp_reg_mapping->gp_reg_scale_base, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, temp_vreg, pred_reg_all );
          } else {
            libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1RH_I_OFF,
                                                  i_gp_reg_mapping->gp_reg_scale_base, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, temp_vreg, pred_reg_all );
            libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, temp_vreg, 0);
          }
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
              temp_vreg,
              (apply_op == 1) ? ((apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset) : im + vecidxin_offset, 0,
              (apply_op == 1) ? ((apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset) : im + vecidxin_offset, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
        } else {
          if (l_is_inp_bf16 == 0) {
            libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R,
                                                             i_gp_reg_mapping->gp_reg_scale_base, LIBXSMM_AARCH64_GP_REG_UNDEF, temp_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
          } else {
            libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R,
                                                             i_gp_reg_mapping->gp_reg_scale_base, LIBXSMM_AARCH64_GP_REG_UNDEF, temp_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
            libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, temp_vreg, 0);
          }
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
              temp_vreg,
              (apply_op == 1) ? ((apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset) : im + vecidxin_offset, 0,
              (apply_op == 1) ? ((apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset) : im + vecidxin_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
        }
      }

      /* Now apply the Reduce OP */
      if (apply_redop == 1) {
        if (is_sve){
          if ((record_argop_off_vec0 == 1) || (record_argop_off_vec1 == 1)) {
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, argop_cmp_instr,
                im + vecidxin_offset, im + vecout_offset, 0, argop_mask, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
            if (idx_tsize == 4) {
              if (record_argop_off_vec0 == 1) {
                libxsmm_aarch64_instruction_sve_compute( io_generated_code, argop_blend_instr,
                    im + vecargop_vec0_offset, temp_vreg_argop0, 0, im + vecargop_vec0_offset, argop_mask, LIBXSMM_AARCH64_SVE_TYPE_S );
              }
              if (record_argop_off_vec1 == 1) {
                libxsmm_aarch64_instruction_sve_compute( io_generated_code, argop_blend_instr,
                    im + vecargop_vec1_offset, temp_vreg_argop1, 0, im + vecargop_vec1_offset, argop_mask, LIBXSMM_AARCH64_SVE_TYPE_S );
              }
            } else {
              /* Extract lo mask to aux */
              libxsmm_aarch64_extract_mask2_from_mask4(io_generated_code, argop_mask, argop_mask_aux, 0);
              if (record_argop_off_vec0 == 1) {/* todo sve: mix old value with new value (index), by argop_mask */
                libxsmm_aarch64_instruction_sve_compute( io_generated_code, argop_blend_instr,
                    im + vecargop_vec0_offset, temp_vreg_argop0, 0, im + vecargop_vec0_offset, argop_mask_aux, LIBXSMM_AARCH64_SVE_TYPE_D );
              }
              if (record_argop_off_vec1 == 1) {/* todo sve: mix old value with new value (index), by argop_mask */
                libxsmm_aarch64_instruction_sve_compute( io_generated_code, argop_blend_instr,
                    im + vecargop_vec0_offset, temp_vreg_argop1, 0, im + vecargop_vec1_offset, argop_mask_aux, LIBXSMM_AARCH64_SVE_TYPE_D );
              }
              /* Extract hi mask to aux */
              libxsmm_aarch64_extract_mask2_from_mask4(io_generated_code, argop_mask, argop_mask_aux, 1);
              if (record_argop_off_vec0 == 1) {/* todo sve: mix old value with new value (index), by argop_mask */
                libxsmm_aarch64_instruction_sve_compute( io_generated_code, argop_blend_instr,
                    im + max_m_unrolling + vecargop_vec0_offset, temp_vreg_argop0, 0, im + max_m_unrolling + vecargop_vec0_offset, argop_mask_aux, LIBXSMM_AARCH64_SVE_TYPE_D );
              }
              if (record_argop_off_vec1 == 1) {/* todo sve: mix old value with new value (index), by argop_mask */
                libxsmm_aarch64_instruction_sve_compute( io_generated_code, argop_blend_instr,
                    im + max_m_unrolling + vecargop_vec0_offset, temp_vreg_argop1, 0, im + max_m_unrolling + vecargop_vec1_offset, argop_mask_aux, LIBXSMM_AARCH64_SVE_TYPE_D );
              }
            }
          }

          libxsmm_aarch64_instruction_sve_compute( io_generated_code, reduceop_instr,
              im + vecidxin_offset, im + vecout_offset, 0, im + vecout_offset, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
        } else {
          if ((record_argop_off_vec0 == 1) || (record_argop_off_vec1 == 1)) {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, argop_cmp_instr,
                im + vecidxin_offset, im + vecout_offset, 0, argop_mask, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
            if (idx_tsize == 4) {
              if (record_argop_off_vec0 == 1) {
                libxsmm_aarch64_instruction_asimd_compute( io_generated_code, argop_blend_instr,
                    temp_vreg_argop0, argop_mask, 0, im + vecargop_vec0_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
              }
              if (record_argop_off_vec1 == 1) {
                libxsmm_aarch64_instruction_asimd_compute( io_generated_code, argop_blend_instr,
                    temp_vreg_argop1, argop_mask, 0, im + vecargop_vec1_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
              }
            } else {
              /* Extract lo mask to aux */
              libxsmm_aarch64_extract_mask2_from_mask4(io_generated_code, argop_mask, argop_mask_aux, 0);
              if (record_argop_off_vec0 == 1) {
                libxsmm_aarch64_instruction_asimd_compute( io_generated_code, argop_blend_instr,
                    temp_vreg_argop0, argop_mask_aux, 0, im + vecargop_vec0_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
              }
              if (record_argop_off_vec1 == 1) {
                libxsmm_aarch64_instruction_asimd_compute( io_generated_code, argop_blend_instr,
                    temp_vreg_argop1, argop_mask_aux, 0, im + vecargop_vec1_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
              }
              /* Extract hi mask to aux */
              libxsmm_aarch64_extract_mask2_from_mask4(io_generated_code, argop_mask, argop_mask_aux, 1);
              if (record_argop_off_vec0 == 1) {
                libxsmm_aarch64_instruction_asimd_compute( io_generated_code, argop_blend_instr,
                    temp_vreg_argop0, argop_mask_aux, 0, im + max_m_unrolling + vecargop_vec0_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
              }
              if (record_argop_off_vec1 == 1) {
                libxsmm_aarch64_instruction_asimd_compute( io_generated_code, argop_blend_instr,
                    temp_vreg_argop1, argop_mask_aux, 0, im + max_m_unrolling + vecargop_vec1_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
              }
            }
          }

          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, reduceop_instr,
              im + vecidxin_offset, im + vecout_offset, 0, im + vecout_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S);
        }
      }
    }

    if (gp_reg_impl_index > 0) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, gp_reg_impl_index, i_gp_reg_mapping->gp_reg_scratch_0, gp_reg_impl_index, 1 );
    }
    if (scale_op_result == 1) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, i_gp_reg_mapping->gp_reg_scale_base,
          i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_scale_base, i_micro_kernel_config->datatype_size_in);
    }

    libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, 1);

    /* Adjust indexed vecs */
    if (use_indexed_vecidx > 0) {
      if (use_implicitly_indexed_vecidx == 0) {
        libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, scratch_gpr, idx_tsize );
        libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MUL, scratch_gpr, i_gp_reg_mapping->gp_reg_n, scratch_gpr, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
        libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR, i_gp_reg_mapping->gp_reg_ind_base, scratch_gpr, i_gp_reg_mapping->gp_reg_ind_base, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
      }
    }
    if (use_indexed_vec > 0) {
      if (use_implicitly_indexed_vec == 0) {
        if (!((use_indexed_vecidx > 0) && (use_implicitly_indexed_vecidx == 0))) {
          libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, scratch_gpr, idx_tsize );
          libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MUL, scratch_gpr, i_gp_reg_mapping->gp_reg_n, scratch_gpr, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
        }
        libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR, i_gp_reg_mapping->gp_reg_ind_base2, scratch_gpr, i_gp_reg_mapping->gp_reg_ind_base2, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
      }
    }

    /* Adjust scale base reg */
    if (scale_op_result == 1) {
      libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, scratch_gpr, i_micro_kernel_config->datatype_size_in );
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MUL, scratch_gpr, i_gp_reg_mapping->gp_reg_n, scratch_gpr, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR, i_gp_reg_mapping->gp_reg_scale_base, scratch_gpr, i_gp_reg_mapping->gp_reg_scale_base, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    }

    /* Now store accumulators */
    for (im = 0; im < m_unroll_factor; im++) {
      l_masked_elements = (l_is_out_bf16 == 0) ? 0 : vlen;
      pred_reg_mask_use = (l_is_out_bf16 == 0) ? pred_reg_all : pred_reg_all_bf16;
      if (l_is_out_bf16 > 0) {
        libxsmm_generator_vcvt_f32bf16_aarch64( io_generated_code, im + vecout_offset, 0);
      }
      libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_out,
        i_gp_reg_mapping->gp_reg_scratch_0, im + vecout_offset, i_micro_kernel_config->datatype_size_out, l_masked_elements, 1, 1, pred_reg_mask_use );
    }

    if ( record_argop_off_vec0 == 1 ) {
      libxsmm_generator_getval_stack_var_aarch64( io_generated_code, rbp_offset_argop_ptr0, temp_gpr);
      for (im = 0; im < m_unroll_factor; im++) {
        unsigned int vreg_id = (idx_tsize == 8) ? ((im % 2 == 0) ? im/2 + vecargop_vec0_offset : im/2 + max_m_unrolling + vecargop_vec0_offset)  : im + vecargop_vec0_offset;
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, temp_gpr,
          i_gp_reg_mapping->gp_reg_scratch_0, vreg_id, 4, 0, 1, 1, pred_reg_all );
      }
      if (idx_tsize == 8) {
        for (im = m_unroll_factor; im < 2*m_unroll_factor; im++) {
          unsigned int vreg_id = (im % 2 == 0) ? im/2 + vecargop_vec0_offset : im/2 + max_m_unrolling + vecargop_vec0_offset;
          libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, temp_gpr,
            i_gp_reg_mapping->gp_reg_scratch_0, vreg_id, 4, 0, 1, 1, pred_reg_all );
        }
      }
    }

    if ( record_argop_off_vec1 == 1 ) {
      libxsmm_generator_getval_stack_var_aarch64( io_generated_code, rbp_offset_argop_ptr1, temp_gpr);
      for (im = 0; im < m_unroll_factor; im++) {
        unsigned int vreg_id = (idx_tsize == 8) ? ((im % 2 == 0) ? im/2 + vecargop_vec1_offset : im/2 + max_m_unrolling + vecargop_vec1_offset)  : im + vecargop_vec1_offset;
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, temp_gpr,
          i_gp_reg_mapping->gp_reg_scratch_0, vreg_id, 4, 0, 1, 1, pred_reg_all );
      }
      if (idx_tsize == 8) {
        for (im = m_unroll_factor; im < 2*m_unroll_factor; im++) {
          unsigned int vreg_id = (im % 2 == 0) ? im/2 + vecargop_vec1_offset : im/2 + max_m_unrolling + vecargop_vec1_offset;
          libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, temp_gpr,
            i_gp_reg_mapping->gp_reg_scratch_0, vreg_id, 4, 0, 1, 1, pred_reg_all );
        }
      }
    }
  }

  if (m_trips_loop > 1) {
    if ( record_argop_off_vec0 == 1 ) {
      libxsmm_generator_getval_stack_var_aarch64( io_generated_code, rbp_offset_argop_ptr0, temp_gpr);
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
          temp_gpr, i_gp_reg_mapping->gp_reg_scratch_0, temp_gpr, (long long)m_unroll_factor * vlen * idx_tsize );
      libxsmm_generator_setval_stack_var_aarch64( io_generated_code, rbp_offset_argop_ptr0, temp_gpr, scratch_gpr);
    }
    if ( record_argop_off_vec1== 1 ) {
      libxsmm_generator_getval_stack_var_aarch64( io_generated_code, rbp_offset_argop_ptr1, temp_gpr);
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
          temp_gpr, i_gp_reg_mapping->gp_reg_scratch_0, temp_gpr, (long long)m_unroll_factor * vlen * idx_tsize );
      libxsmm_generator_setval_stack_var_aarch64( io_generated_code, rbp_offset_argop_ptr1, temp_gpr, scratch_gpr);
    }
    if (apply_op == 1) {
      if (bcast_param == 0) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
            i_gp_reg_mapping->gp_reg_invec, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_invec, (long long)m_unroll_factor * vlen * i_micro_kernel_config->datatype_size_out );
      }
    }
    if (bcast_param == 0) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
          i_gp_reg_mapping->gp_reg_in_base, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in_base, (long long)m_unroll_factor * vlen * i_micro_kernel_config->datatype_size_in );
    }
    libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, 1);
  }

  if (peeled_m_trips > 0) {
    if (m_trips_loop == 1) {
      if ( record_argop_off_vec0 == 1 ) {
        libxsmm_generator_getval_stack_var_aarch64( io_generated_code, rbp_offset_argop_ptr0, temp_gpr);
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
            temp_gpr, i_gp_reg_mapping->gp_reg_scratch_0, temp_gpr, (long long)m_unroll_factor * vlen * idx_tsize );
        libxsmm_generator_setval_stack_var_aarch64( io_generated_code, rbp_offset_argop_ptr0, temp_gpr, scratch_gpr);
      }
      if ( record_argop_off_vec1== 1 ) {
        libxsmm_generator_getval_stack_var_aarch64( io_generated_code, rbp_offset_argop_ptr1, temp_gpr);
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
            temp_gpr, i_gp_reg_mapping->gp_reg_scratch_0, temp_gpr, (long long)m_unroll_factor * vlen * idx_tsize );
        libxsmm_generator_setval_stack_var_aarch64( io_generated_code, rbp_offset_argop_ptr1, temp_gpr, scratch_gpr);
      }
      if (apply_op == 1) {
        if (bcast_param == 0) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
              i_gp_reg_mapping->gp_reg_invec, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_invec, (long long)m_unroll_factor * vlen * i_micro_kernel_config->datatype_size_out );
        }
      }
      if (bcast_param == 0) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
            i_gp_reg_mapping->gp_reg_in_base, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in_base, (long long)m_unroll_factor * vlen * i_micro_kernel_config->datatype_size_in );
      }
    }

    /* Perform the reductions for all columns */
    for (im = 0; im < peeled_m_trips; im++) {
      /* Load output for reduction */
      if (apply_redop == 1) {
        if (load_acc == 0) {
          if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_MAX) > 0) {
            if (im == 0) {
              libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I, LIBXSMM_AARCH64_GP_REG_X29, i_gp_reg_mapping->gp_reg_scratch_0, pos_rbp_offset_neg_inf, 0 );
              if (is_sve){
                libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF,
                                                      i_gp_reg_mapping->gp_reg_scratch_0, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, vecout_offset, pred_reg_all );
              } else {
                libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R,
                                                                 i_gp_reg_mapping->gp_reg_scratch_0, LIBXSMM_AARCH64_GP_REG_UNDEF, vecout_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
              }
            } else {
              if (is_sve){
                libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                                           vecout_offset, vecout_offset, 0, im + vecout_offset, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
              } else {
                libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                                           vecout_offset, vecout_offset, 0, im + vecout_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
              }
            }
          } else {
            if (is_sve){
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                im + vecout_offset, im + vecout_offset, 0, im + vecout_offset, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
            } else {
              libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                im + vecout_offset, im + vecout_offset, 0, im + vecout_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
            }
          }
        } else {
          mask_count2 = ((im == peeled_m_trips-1) && (use_m_masking > 0)) ? mask_count: 0;
          l_masked_elements = (l_is_out_bf16 == 0) ? mask_count2 : (mask_count2 > 0) ? mask_count2 : vlen;
          pred_reg_mask_use = (l_is_out_bf16 == 0) ? ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all) : ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all_bf16);
          libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0,
              im + vecout_offset, i_micro_kernel_config->datatype_size_out, l_masked_elements, 1, 0, pred_reg_mask_use );
          if (l_is_out_bf16 > 0) {
            libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, im + vecout_offset, 0);
          }
        }
      }

      /* Initialize argop vectors if need be */
      if (record_argop_off_vec0 == 1) {
        if (is_sve){
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
              im + vecargop_vec0_offset, im + vecargop_vec0_offset, 0, im + vecargop_vec0_offset, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
          if (idx_tsize == 8) {
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                im + max_m_unrolling + vecargop_vec0_offset, im + max_m_unrolling + vecargop_vec0_offset, 0, im + max_m_unrolling + vecargop_vec0_offset, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
          }
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
              im + vecargop_vec0_offset, im + vecargop_vec0_offset, 0, im + vecargop_vec0_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
          if (idx_tsize == 8) {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                im + max_m_unrolling + vecargop_vec0_offset, im + max_m_unrolling + vecargop_vec0_offset, 0, im + max_m_unrolling + vecargop_vec0_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
          }
        }
      }

      /* Initialize argop vectors if need be */
      if (record_argop_off_vec1 == 1) {
        if (is_sve){
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
              im + vecargop_vec1_offset, im + vecargop_vec1_offset, 0, im + vecargop_vec1_offset, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
          if (idx_tsize == 8) {
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                im + max_m_unrolling + vecargop_vec1_offset, im + max_m_unrolling + vecargop_vec1_offset, 0, im + max_m_unrolling + vecargop_vec1_offset, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
          }
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
              im + vecargop_vec1_offset, im + vecargop_vec1_offset, 0, im + vecargop_vec1_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
          if (idx_tsize == 8) {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                im + max_m_unrolling + vecargop_vec1_offset, im + max_m_unrolling + vecargop_vec1_offset, 0, im + max_m_unrolling + vecargop_vec1_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
          }
        }
      }

      /* Load input vector in case we have to apply op */
      if (apply_op == 1) {
        if (use_indexed_vec == 0) {
          if (bcast_param == 0) {
            mask_count2 = ((im == peeled_m_trips-1) && (use_m_masking > 0)) ? mask_count : 0;
            l_masked_elements = (l_is_inp_bf16 == 0) ? mask_count2 : (mask_count2 > 0) ? mask_count2 : vlen;
            pred_reg_mask_use = (l_is_inp_bf16 == 0) ? ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all) : ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all_bf16);
            libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_invec, scratch_gpr,
                im + vecin_offset, i_micro_kernel_config->datatype_size_out, l_masked_elements, 1, 0, pred_reg_mask_use );
            if (l_is_inp_bf16 > 0) {
              libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, im + vecin_offset, 0);
            }
          } else {
            if (im == 0) {
              libxsmm_generator_bcastload_masked_vreg_aarch64_asimd( io_generated_code, i_gp_reg_mapping->gp_reg_invec, scratch_gpr,
                  im + vecin_offset, i_micro_kernel_config->datatype_size_in, 0, 0 );
              if (l_is_inp_bf16 > 0) {
                libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, im + vecin_offset, 0);
              }
              if ( is_sve){
                for (_im = 1; _im < m_unroll_factor; _im++) {
                  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                                           vecin_offset, vecin_offset, 0, _im + vecin_offset, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
                }
              } else {
                for (_im = 1; _im < m_unroll_factor; _im++) {
                  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                                          vecin_offset, vecin_offset, 0, _im + vecin_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
                }
              }
            }
          }
        }
      }
    }

    /* Adjust post advancements if need be */
    if (apply_redop == 1) {
      if (load_acc == 1) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB, i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out,
            (long long)peeled_m_trips * vlen * i_micro_kernel_config->datatype_size_out - (long long)use_m_masking * ((long long)vlen - mask_count) * i_micro_kernel_config->datatype_size_out );
      }
    }
    if (apply_op == 1) {
      if (use_indexed_vec == 0) {
        if (bcast_param == 0) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB, i_gp_reg_mapping->gp_reg_invec, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_invec,
              (long long)peeled_m_trips * vlen * i_micro_kernel_config->datatype_size_in - (long long)use_m_masking * ((long long)vlen - mask_count) * i_micro_kernel_config->datatype_size_in );
        }
      }
    }

    /* Perform the reductions for all columns */
    if (gp_reg_impl_index > 0) {
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_EOR_SR,
          gp_reg_impl_index, gp_reg_impl_index, gp_reg_impl_index, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    }

    libxsmm_generator_loop_header_gp_reg_bound_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, i_gp_reg_mapping->gp_reg_n );

    if (use_indexed_vecidx > 0) {
      if (use_implicitly_indexed_vecidx == 0) {
        libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_POST, i_gp_reg_mapping->gp_reg_ind_base, LIBXSMM_AARCH64_GP_REG_UNDEF, idx_tsize, gp_reg_index );
      } else {
        libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ORR_SR, gp_reg_impl_index, gp_reg_impl_index, gp_reg_index_64bit, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
      }
      if (record_argop_off_vec0 == 1) {
        /* broadcast gp reg into vreg */
        if (is_sve){
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_DUP_GP_V,
            gp_reg_index_64bit, gp_reg_index_64bit, 0, temp_vreg_argop0, 0, (idx_tsize == 8) ? LIBXSMM_AARCH64_SVE_TYPE_D : LIBXSMM_AARCH64_SVE_TYPE_S );
        } else {
          libxsmm_aarch64_instruction_asimd_gpr_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_DUP_FULL,
            gp_reg_index_64bit, temp_vreg_argop0, 0, (idx_tsize == 8) ? LIBXSMM_AARCH64_ASIMD_WIDTH_D : LIBXSMM_AARCH64_ASIMD_WIDTH_S );
        }
      }
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MUL, gp_reg_index_64bit, gp_reg_ldi, i_gp_reg_mapping->gp_reg_in, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_in_base, i_gp_reg_mapping->gp_reg_in, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    }

    if (use_indexed_vec > 0) {
      if (use_implicitly_indexed_vec == 0) {
        libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_POST, i_gp_reg_mapping->gp_reg_ind_base2, LIBXSMM_AARCH64_GP_REG_UNDEF, idx_tsize, gp_reg_index );
      } else {
        libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ORR_SR, gp_reg_impl_index, gp_reg_impl_index, gp_reg_index_64bit, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
      }
      if (record_argop_off_vec1 == 1) {
        if (is_sve){
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_DUP_GP_V,
            gp_reg_index_64bit, gp_reg_index_64bit, 0, temp_vreg_argop1, 0, (idx_tsize == 8) ? LIBXSMM_AARCH64_SVE_TYPE_D : LIBXSMM_AARCH64_SVE_TYPE_S );
        } else {
          libxsmm_aarch64_instruction_asimd_gpr_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_DUP_FULL,
              gp_reg_index_64bit, temp_vreg_argop1, 0, (idx_tsize == 8) ? LIBXSMM_AARCH64_ASIMD_WIDTH_D : LIBXSMM_AARCH64_ASIMD_WIDTH_S);
        }
      }
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MUL, gp_reg_index_64bit, gp_reg_ldi, i_gp_reg_mapping->gp_reg_in2, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_in_base2, i_gp_reg_mapping->gp_reg_in2, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    }

    for (im = 0; im < peeled_m_trips; im++) {
      /* First load the indexed vector */
      if (bcast_param == 0) {
        mask_count2 = ((im == peeled_m_trips-1) && (use_m_masking > 0)) ? mask_count : 0;
        l_masked_elements = (l_is_inp_bf16 == 0) ? mask_count2 : (mask_count2 > 0) ? mask_count2 : vlen;
        pred_reg_mask_use = (l_is_inp_bf16 == 0) ? ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all) : ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all_bf16);
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_in, scratch_gpr,
            im + vecidxin_offset, i_micro_kernel_config->datatype_size_in, l_masked_elements, 1, 0, pred_reg_mask_use );
        if (l_is_inp_bf16 > 0) {
          libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, im + vecidxin_offset, 0);
        }
      } else {
        if (im == 0) {
          libxsmm_generator_bcastload_masked_vreg_aarch64_asimd( io_generated_code, i_gp_reg_mapping->gp_reg_in, scratch_gpr,
              im + vecidxin_offset, i_micro_kernel_config->datatype_size_in, 0, 0 );
          if (l_is_inp_bf16 > 0) {
            libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, im + vecidxin_offset, 0);
          }
          if (is_sve){
            for (_im = 1; _im < m_unroll_factor; _im++) {
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                                       vecidxin_offset, vecidxin_offset, 0, _im + vecidxin_offset, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
            }
          } else {
            for (_im = 1; _im < m_unroll_factor; _im++) {
              libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                                      vecidxin_offset, vecidxin_offset, 0, _im + vecidxin_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
            }
          }
        }
      }

      /* Now apply the OP among the indexed vector and the input vector */
      if (apply_op == 1) {
        if (use_indexed_vec > 0) {
          if (bcast_param == 0) {
            mask_count2 = ((im == peeled_m_trips-1) && (use_m_masking > 0)) ? mask_count : 0;
            l_masked_elements = (l_is_inp_bf16 == 0) ? mask_count2 : (mask_count2 > 0) ? mask_count2 : vlen;
            pred_reg_mask_use = (l_is_inp_bf16 == 0) ? ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all) : ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all_bf16);
            libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_in2, scratch_gpr,
                im + vecin_offset, i_micro_kernel_config->datatype_size_in, l_masked_elements, 1, 0, pred_reg_mask_use );
            if (l_is_inp_bf16 > 0) {
              libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, im + vecin_offset, 0);
            }
          } else {
            if (im == 0) {
              libxsmm_generator_bcastload_masked_vreg_aarch64_asimd( io_generated_code, i_gp_reg_mapping->gp_reg_in2, scratch_gpr,
                  im + vecin_offset, i_micro_kernel_config->datatype_size_in, 0, 0 );
              if (l_is_inp_bf16 > 0) {
                libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, im + vecin_offset, 0);
              }
              if (is_sve){
                for (_im = 1; _im < m_unroll_factor; _im++) {
                  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                                          vecin_offset, vecin_offset, 0, _im + vecin_offset, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
                }
              } else {
                for (_im = 1; _im < m_unroll_factor; _im++) {
                  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                                          vecin_offset, vecin_offset, 0, _im + vecin_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
                }
              }
            }
          }
        }

        if (op_instr == LIBXSMM_AARCH64_INSTR_DOTPS) {
          /* TODO: Add DOT op sequence here */
        } else {
          if (is_sve){
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, op_instr,
              (op_order == 0)    ? im + vecidxin_offset : im + vecin_offset,
              (op_order == 0)    ? im + vecin_offset    : im + vecidxin_offset, 0,
              (apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
          } else {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, op_instr,
              (op_order == 0)    ? im + vecidxin_offset : im + vecin_offset,
              (op_order == 0)    ? im + vecin_offset    : im + vecidxin_offset, 0,
              (apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
          }
        }
      }

      if (scale_op_result == 1) {
        if (is_sve){
          if (l_is_inp_bf16 == 0) {
            libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF,
                                                  i_gp_reg_mapping->gp_reg_scale_base, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, temp_vreg, pred_reg_all );
          } else {
            libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1RH_I_OFF,
                                                  i_gp_reg_mapping->gp_reg_scale_base, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, temp_vreg, pred_reg_all );
            libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, temp_vreg, 0);
          }
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
              temp_vreg,
              (apply_op == 1) ? ((apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset) : im + vecidxin_offset, 0,
              (apply_op == 1) ? ((apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset) : im + vecidxin_offset, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
        } else {
          if (l_is_inp_bf16 == 0) {
            libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R,
                                                             i_gp_reg_mapping->gp_reg_scale_base, LIBXSMM_AARCH64_GP_REG_UNDEF, temp_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
          } else {
            libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R,
                                                             i_gp_reg_mapping->gp_reg_scale_base, LIBXSMM_AARCH64_GP_REG_UNDEF, temp_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
            libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, temp_vreg, 0);
          }
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
              temp_vreg,
              (apply_op == 1) ? ((apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset) : im + vecidxin_offset, 0,
              (apply_op == 1) ? ((apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset) : im + vecidxin_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
        }
      }

      /* Now apply the Reduce OP */
      if (apply_redop == 1) {
        if (is_sve){
          if ((record_argop_off_vec0 == 1) || (record_argop_off_vec1 == 1)) {
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, argop_cmp_instr,
                im + vecidxin_offset, im + vecout_offset, 0, argop_mask, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
            if (idx_tsize == 4) {
              if (record_argop_off_vec0 == 1) {
                libxsmm_aarch64_instruction_sve_compute( io_generated_code, argop_blend_instr,
                    im + vecargop_vec0_offset, temp_vreg_argop0, 0, im + vecargop_vec0_offset, argop_mask, LIBXSMM_AARCH64_SVE_TYPE_S );
              }
              if (record_argop_off_vec1 == 1) {
                libxsmm_aarch64_instruction_sve_compute( io_generated_code, argop_blend_instr,
                    im + vecargop_vec1_offset, temp_vreg_argop1, 0, im + vecargop_vec1_offset, argop_mask, LIBXSMM_AARCH64_SVE_TYPE_S );
              }
            } else {
              /* Extract lo mask to aux */
              libxsmm_aarch64_extract_mask2_from_mask4(io_generated_code, argop_mask, argop_mask_aux, 0);
              if (record_argop_off_vec0 == 1) {
                libxsmm_aarch64_instruction_sve_compute( io_generated_code, argop_blend_instr,
                    im + vecargop_vec0_offset, temp_vreg_argop0, 0, im + vecargop_vec0_offset, argop_mask_aux, LIBXSMM_AARCH64_SVE_TYPE_D );
              }
              if (record_argop_off_vec1 == 1) {
                libxsmm_aarch64_instruction_sve_compute( io_generated_code, argop_blend_instr,
                    im + vecargop_vec1_offset, temp_vreg_argop1, 0, im + vecargop_vec1_offset, argop_mask_aux, LIBXSMM_AARCH64_SVE_TYPE_D );
              }
              /* Extract hi mask to aux */
              libxsmm_aarch64_extract_mask2_from_mask4(io_generated_code, argop_mask, argop_mask_aux, 1);
              if (record_argop_off_vec0 == 1) {
                libxsmm_aarch64_instruction_sve_compute( io_generated_code, argop_blend_instr,
                    im + max_m_unrolling + vecargop_vec0_offset, temp_vreg_argop0, 0, im + max_m_unrolling + vecargop_vec0_offset, argop_mask_aux, LIBXSMM_AARCH64_SVE_TYPE_D );
              }
              if (record_argop_off_vec1 == 1) {
                libxsmm_aarch64_instruction_sve_compute( io_generated_code, argop_blend_instr,
                    im + max_m_unrolling + vecargop_vec1_offset, temp_vreg_argop1, 0, im + max_m_unrolling + vecargop_vec1_offset, argop_mask_aux, LIBXSMM_AARCH64_SVE_TYPE_D );
              }
            }
          }

          libxsmm_aarch64_instruction_sve_compute( io_generated_code, reduceop_instr,
              im + vecidxin_offset, im + vecout_offset, 0, im + vecout_offset, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
        } else {
          if ((record_argop_off_vec0 == 1) || (record_argop_off_vec1 == 1)) {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, argop_cmp_instr,
                im + vecidxin_offset, im + vecout_offset, 0, argop_mask, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
            if (idx_tsize == 4) {
              if (record_argop_off_vec0 == 1) {
                libxsmm_aarch64_instruction_asimd_compute( io_generated_code, argop_blend_instr,
                    temp_vreg_argop0, argop_mask, 0, im + vecargop_vec0_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
              }
              if (record_argop_off_vec1 == 1) {
                libxsmm_aarch64_instruction_asimd_compute( io_generated_code, argop_blend_instr,
                    temp_vreg_argop1, argop_mask, 0, im + vecargop_vec1_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
              }
            } else {
              /* Extract lo mask to aux */
              libxsmm_aarch64_extract_mask2_from_mask4(io_generated_code, argop_mask, argop_mask_aux, 0);
              if (record_argop_off_vec0 == 1) {
                libxsmm_aarch64_instruction_asimd_compute( io_generated_code, argop_blend_instr,
                    temp_vreg_argop0, argop_mask_aux, 0, im + vecargop_vec0_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
              }
              if (record_argop_off_vec1 == 1) {
                libxsmm_aarch64_instruction_asimd_compute( io_generated_code, argop_blend_instr,
                    temp_vreg_argop1, argop_mask_aux, 0, im + vecargop_vec1_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
              }
              /* Extract hi mask to aux */
              libxsmm_aarch64_extract_mask2_from_mask4(io_generated_code, argop_mask, argop_mask_aux, 1);
              if (record_argop_off_vec0 == 1) {
                libxsmm_aarch64_instruction_asimd_compute( io_generated_code, argop_blend_instr,
                    temp_vreg_argop0, argop_mask_aux, 0, im + max_m_unrolling + vecargop_vec0_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
              }
              if (record_argop_off_vec1 == 1) {
                libxsmm_aarch64_instruction_asimd_compute( io_generated_code, argop_blend_instr,
                    temp_vreg_argop1, argop_mask_aux, 0, im + max_m_unrolling + vecargop_vec1_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
              }
            }
          }

          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, reduceop_instr,
              im + vecidxin_offset, im + vecout_offset, 0, im + vecout_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S);
        }
      }
    }

    if (gp_reg_impl_index > 0) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, gp_reg_impl_index, i_gp_reg_mapping->gp_reg_scratch_0, gp_reg_impl_index, 1 );
    }
    if (scale_op_result == 1) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, i_gp_reg_mapping->gp_reg_scale_base,
          i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_scale_base, i_micro_kernel_config->datatype_size_in);
    }

    libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, 1);

    /* Adjust indexed vecs */
    if (use_indexed_vecidx > 0) {
      if (use_implicitly_indexed_vecidx == 0) {
        libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, scratch_gpr, idx_tsize );
        libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MUL, scratch_gpr, i_gp_reg_mapping->gp_reg_n, scratch_gpr, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
        libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR, i_gp_reg_mapping->gp_reg_ind_base, scratch_gpr, i_gp_reg_mapping->gp_reg_ind_base, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
      }
    }
    if (use_indexed_vec > 0) {
      if (use_implicitly_indexed_vec == 0) {
        if (!((use_indexed_vecidx > 0) && (use_implicitly_indexed_vecidx == 0))) {
          libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, scratch_gpr, idx_tsize );
          libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MUL, scratch_gpr, i_gp_reg_mapping->gp_reg_n, scratch_gpr, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
        }
        libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR, i_gp_reg_mapping->gp_reg_ind_base2, scratch_gpr, i_gp_reg_mapping->gp_reg_ind_base2, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
      }
    }

    if (scale_op_result == 1) {
      libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, scratch_gpr, i_micro_kernel_config->datatype_size_in );
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MUL, scratch_gpr, i_gp_reg_mapping->gp_reg_n, scratch_gpr, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR, i_gp_reg_mapping->gp_reg_scale_base, scratch_gpr, i_gp_reg_mapping->gp_reg_scale_base, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    }

    /* Now store accumulators */
    for (im = 0; im < peeled_m_trips; im++) {
      mask_count2 = ((im == peeled_m_trips-1) && (use_m_masking > 0)) ? mask_count : 0;
      l_masked_elements = (l_is_out_bf16 == 0) ? mask_count2 : (mask_count2 > 0) ? mask_count2 : vlen;
      pred_reg_mask_use = (l_is_out_bf16 == 0) ? ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all) : ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all_bf16);
      if (l_is_out_bf16 > 0) {
        libxsmm_generator_vcvt_f32bf16_aarch64( io_generated_code, im + vecout_offset, 0);
      }
      libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, im + vecout_offset,
          i_micro_kernel_config->datatype_size_out, l_masked_elements, 1, 1, pred_reg_mask_use );
    }

    if ( record_argop_off_vec0 == 1 ) {
      libxsmm_generator_getval_stack_var_aarch64( io_generated_code, rbp_offset_argop_ptr0, temp_gpr);
      if (idx_tsize == 4) {
        for (im = 0; im < peeled_m_trips; im++) {
          unsigned int vreg_id = im + vecargop_vec0_offset;
          mask_count2 = ((im == peeled_m_trips-1) && (use_m_masking > 0)) ? mask_count : 0;
          libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, temp_gpr,
            i_gp_reg_mapping->gp_reg_scratch_0, vreg_id, 4, mask_count2, 1, 1, mask_count2 ? pred_reg_argidx32 : pred_reg_all );
        }
      } else {
        unsigned int use_idx_masking = 0;
        unsigned int idx_peeled_m_trips = 0;
        if (m % vlen != 0) {
          idx_peeled_m_trips = (peeled_m_trips-1) * vlen + m % vlen;
        } else {
          idx_peeled_m_trips = peeled_m_trips * vlen;
        }
        if (idx_peeled_m_trips % (vlen/2) != 0) {
          use_idx_masking = 1;
        }
        idx_peeled_m_trips = (idx_peeled_m_trips + (vlen/2)-1)/(vlen/2);
        for (im = 0; im < idx_peeled_m_trips; im++) {
          unsigned int vreg_id = (im % 2 == 0) ? im/2 + vecargop_vec0_offset : im/2 + max_m_unrolling + vecargop_vec0_offset;
          mask_count2 = ((im == idx_peeled_m_trips-1) && (use_idx_masking > 0)) ? mask_argidx64 : 0;
          libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, temp_gpr, i_gp_reg_mapping->gp_reg_scratch_0,
              vreg_id, 4, mask_count2, 1, 1, mask_count2 == 0 ? pred_reg_all : pred_reg_argidx );
        }
      }
    }

    if ( record_argop_off_vec1 == 1 ) {
      libxsmm_generator_getval_stack_var_aarch64( io_generated_code, rbp_offset_argop_ptr1, temp_gpr);
      if (idx_tsize == 4) {
        for (im = 0; im < peeled_m_trips; im++) {
          unsigned int vreg_id = im + vecargop_vec1_offset;
          mask_count2 = ((im == peeled_m_trips-1) && (use_m_masking > 0)) ? mask_count : 0;
          libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, temp_gpr,
            i_gp_reg_mapping->gp_reg_scratch_0, vreg_id, 4, mask_count2, 1, 1, mask_count2 == 0 ? pred_reg_all : pred_reg_argidx32 );
        }
      } else {
        unsigned int use_idx_masking = 0;
        unsigned int idx_peeled_m_trips = 0;
        if (m % vlen != 0) {
          idx_peeled_m_trips = (peeled_m_trips-1) * vlen + m % vlen;
        } else {
          idx_peeled_m_trips = peeled_m_trips * vlen;
        }
        if (idx_peeled_m_trips % (vlen/2) != 0) {
          use_idx_masking = 1;
        }
        idx_peeled_m_trips = (idx_peeled_m_trips + (vlen/2)-1)/(vlen/2);
        for (im = 0; im < idx_peeled_m_trips; im++) {
          unsigned int vreg_id = (im % 2 == 0) ? im/2 + vecargop_vec1_offset : im/2 + max_m_unrolling + vecargop_vec1_offset;
          mask_count2 = ((im == idx_peeled_m_trips-1) && (use_idx_masking > 0)) ? mask_argidx64 : 0;
          libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, temp_gpr, i_gp_reg_mapping->gp_reg_scratch_0,
              vreg_id, 4, mask_count2, 1, 1, mask_count2 == 0 ? pred_reg_all : pred_reg_argidx );
        }
      }
    }
    if (bcast_loops > 1) {
      if ( record_argop_off_vec0 == 1 ) {
        libxsmm_generator_getval_stack_var_aarch64( io_generated_code, rbp_offset_argop_ptr0, temp_gpr);
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
            temp_gpr, i_gp_reg_mapping->gp_reg_scratch_0, temp_gpr, (((long long)peeled_m_trips-1) * vlen + m % vlen) * idx_tsize );
        libxsmm_generator_setval_stack_var_aarch64( io_generated_code, rbp_offset_argop_ptr0, temp_gpr, scratch_gpr);
      }
      if ( record_argop_off_vec1 == 1 ) {
        libxsmm_generator_getval_stack_var_aarch64( io_generated_code, rbp_offset_argop_ptr1, temp_gpr);
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
            temp_gpr, i_gp_reg_mapping->gp_reg_scratch_0, temp_gpr, (((long long)peeled_m_trips-1) * vlen + m % vlen) * idx_tsize );
        libxsmm_generator_setval_stack_var_aarch64( io_generated_code, rbp_offset_argop_ptr1, temp_gpr, scratch_gpr);
      }
    }
  } else {
    if (m_trips_loop == 1) {
      if (bcast_loops > 1) {
        if ( record_argop_off_vec0 == 1 ) {
          libxsmm_generator_getval_stack_var_aarch64( io_generated_code, rbp_offset_argop_ptr0, temp_gpr);
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
              temp_gpr, i_gp_reg_mapping->gp_reg_scratch_0, temp_gpr, (long long)m_unroll_factor * vlen * idx_tsize );
          libxsmm_generator_setval_stack_var_aarch64( io_generated_code, rbp_offset_argop_ptr0, temp_gpr, scratch_gpr);
        }
        if ( record_argop_off_vec1 == 1 ) {
          libxsmm_generator_getval_stack_var_aarch64( io_generated_code, rbp_offset_argop_ptr1, temp_gpr);
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
              temp_gpr, i_gp_reg_mapping->gp_reg_scratch_0, temp_gpr, (long long)m_unroll_factor * vlen * idx_tsize );
          libxsmm_generator_setval_stack_var_aarch64( io_generated_code, rbp_offset_argop_ptr1, temp_gpr, scratch_gpr);
        }
      }
    }
  }

  if (bcast_loops > 1) {
    if (apply_op == 1) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, i_gp_reg_mapping->gp_reg_invec, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_invec, i_micro_kernel_config->datatype_size_out);
    }
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, i_gp_reg_mapping->gp_reg_in_base, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in_base, i_micro_kernel_config->datatype_size_in);
    libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, gp_reg_bcast_loop, 1);
  }

  if (use_stack_vars > 0) {
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_X29, LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );
  }

  libxsmm_aarch64_instruction_register_jump_label( io_generated_code, END_LABEL, p_jump_label_tracker );

#if defined(LIBXSMM_GENERATOR_MATELTWISE_REDUCE_AARCH64_JUMP_LABEL_TRACKER_MALLOC)
  free(p_jump_label_tracker);
#endif
}

LIBXSMM_API_INTERN
void libxsmm_generator_opreduce_vecs_index_aarch64_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {

  libxsmm_meltw_descriptor  i_mateltwise_desc_copy = *i_mateltwise_desc;
  unsigned int bcast_param = (unsigned int) (i_mateltwise_desc->param >> 2);

  if (bcast_param == 0) {
    libxsmm_generator_opreduce_vecs_index_aarch64_microkernel_block( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, &i_mateltwise_desc_copy );
  } else {
    if ((bcast_param >= i_mateltwise_desc->m) || ((i_mateltwise_desc->m % bcast_param) == 0)) {
      libxsmm_generator_opreduce_vecs_index_aarch64_microkernel_block( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, &i_mateltwise_desc_copy );
    } else {
      /* In this case we stamp out two different microkernels back to back... */
      unsigned int temp_gpr = LIBXSMM_AARCH64_GP_REG_X9;
      unsigned int scratch_gpr = LIBXSMM_AARCH64_GP_REG_X10;
      int aux = 0;

      i_mateltwise_desc_copy.m = i_mateltwise_desc->m - (i_mateltwise_desc->m % bcast_param);
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 16, 0 );
      libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 0, i_gp_reg_mapping->gp_reg_param_struct, scratch_gpr );
      libxsmm_generator_opreduce_vecs_index_aarch64_microkernel_block( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, &i_mateltwise_desc_copy );
      libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 0, i_gp_reg_mapping->gp_reg_param_struct, scratch_gpr);
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 16, 0 );

      /* Store and adjust contents of input param struct before stamping out the second microkernel*/
      for (aux = 0; aux <= 72; aux += 8) {
        libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, aux, temp_gpr );
        libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 16, 0 );
        libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 0, temp_gpr, scratch_gpr );

        /* Adjusting Output ptr */
        if (aux == 32) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
              temp_gpr, scratch_gpr, temp_gpr, (long long)i_mateltwise_desc_copy.m * i_micro_kernel_config->datatype_size_out );
          libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, aux, temp_gpr );
        }

        /* Adjusting Input ptrs */
        if ((aux == 16) || (aux == 24) || (aux == 56)) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
              temp_gpr, scratch_gpr, temp_gpr, ((long long)i_mateltwise_desc_copy.m/bcast_param) * i_micro_kernel_config->datatype_size_in );
          libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, aux, temp_gpr );
        }

        /* Adjusting Argop ptrs */
        if ((aux == 64) || (aux == 72)) {
          unsigned int idx_tsize =  i_mateltwise_desc->n;
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
              temp_gpr, scratch_gpr, temp_gpr, (long long)i_mateltwise_desc_copy.m * idx_tsize );
          libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, aux, temp_gpr );
        }
      }

      i_mateltwise_desc_copy.m = i_mateltwise_desc->m % bcast_param;
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 16, 0 );
      libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 0, i_gp_reg_mapping->gp_reg_param_struct, scratch_gpr );
      libxsmm_generator_opreduce_vecs_index_aarch64_microkernel_block( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, &i_mateltwise_desc_copy );
      libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 0, i_gp_reg_mapping->gp_reg_param_struct, scratch_gpr);
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 16, 0 );

      /* Recover contents of input param struct */
      for (aux = 72; aux >= 0; aux -= 8) {
        libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 0, temp_gpr, scratch_gpr );
        libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 16, 0 );
        libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, aux, temp_gpr );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_reduce_cols_index_aarch64_microkernel( libxsmm_generated_code*                        io_generated_code,
                                                              libxsmm_loop_label_tracker*                    io_loop_label_tracker,
                                                              libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
                                                              const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
                                                              const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {

  unsigned int m, im, use_m_masking, m_trips, max_m_unrolling = 4, m_unroll_factor = 1, m_trips_loop = 0;
  unsigned int peeled_m_trips = 0, mask_count = 0, mask_count2;
  unsigned int gp_reg_index_64bit = 0, gp_reg_index = 0, gp_reg_ldi = 0;
  unsigned int idx_tsize = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES) > 0 ) ? 8 : 4;
  unsigned int load_acc = 1;
  unsigned int END_LABEL = 1;
  unsigned int vlen = libxsmm_cpuid_vlen32(i_micro_kernel_config->instruction_set);
  unsigned char pred_reg_all = 0; /* set by caller */
  unsigned char pred_reg_mask = 1;
  unsigned char pred_reg_all_bf16 = 2;
  unsigned char pred_reg_mask_use = 0;
  unsigned int l_masked_elements = 0;
  unsigned int l_is_inp_bf16 = (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ? 1 : 0;
  unsigned int l_is_out_bf16 = (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ? 1 : 0;
  unsigned int aux_vreg_offset = LIBXSMM_AARCH64_GP_REG_X16 & 31; /* register number 16 */
  unsigned char is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
#if defined(USE_ENV_TUNING)
  const char *const env_max_m_unroll = getenv("MAX_M_UNROLL_REDUCE_COLS_IDX");
#endif
  const char *const env_load_acc= getenv("LOAD_ACCS_REDUCE_COLS_IDX");
#if defined(LIBXSMM_GENERATOR_MATELTWISE_REDUCE_AARCH64_JUMP_LABEL_TRACKER_MALLOC)
  libxsmm_jump_label_tracker* const p_jump_label_tracker = (libxsmm_jump_label_tracker*)malloc(sizeof(libxsmm_jump_label_tracker));
#else
  libxsmm_jump_label_tracker l_jump_label_tracker;
  libxsmm_jump_label_tracker* const p_jump_label_tracker = &l_jump_label_tracker;
#endif
  libxsmm_reset_jump_label_tracker(p_jump_label_tracker);

  /* intercept codegen and call specialized opreduce kernel */
  if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_MAX) {
    libxsmm_descriptor_blob blob;
    libxsmm_mateltwise_kernel_config new_config = *i_micro_kernel_config;
    libxsmm_blasint idx_dtype_size = idx_tsize;
    unsigned short bcast_param = 0;
    libxsmm_meltw_opreduce_vecs_flags flags = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_RECORD_ARGOP) > 0) ? LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDUCE_MAX_IDX_COLS_ARGOP : LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDUCE_MAX_IDX_COLS;
    unsigned short argidx_params = (unsigned short) (((flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_RECORD_ARGOP_OFF_VEC_0) | (flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_RECORD_ARGOP_OFF_VEC_1)) >> 16);
    unsigned short bcast_shifted_params = (unsigned short) (bcast_param << 2);
    unsigned short combined_params = argidx_params | bcast_shifted_params;
    const libxsmm_meltw_descriptor *const new_desc = libxsmm_meltw_descriptor_init(&blob, (libxsmm_datatype)LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ), (libxsmm_datatype)LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ),
        i_mateltwise_desc->m, idx_dtype_size, i_mateltwise_desc->ldi, i_mateltwise_desc->ldo, (unsigned short)flags, (unsigned short) combined_params, LIBXSMM_MELTW_OPERATION_OPREDUCE_VECS_IDX);
#if defined(LIBXSMM_GENERATOR_MATELTWISE_REDUCE_AARCH64_JUMP_LABEL_TRACKER_MALLOC)
    free(p_jump_label_tracker);
#endif
    new_config.opreduce_use_unary_arg_reading = 1;
    if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_NEG_INF_ACC) > 0 ) {
      new_config.opreduce_avoid_acc_load = 1;
    }
    libxsmm_generator_opreduce_vecs_index_aarch64_microkernel( io_generated_code,io_loop_label_tracker, i_gp_reg_mapping, &new_config, new_desc );
    return;
  }

#if defined(USE_ENV_TUNING)
  if ( 0 == env_max_m_unroll ) {
  } else {
    max_m_unrolling = LIBXSMM_MAX(1,atoi(env_max_m_unroll));
  }
#endif
  if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_INIT_ACC) > 0 ) {
    load_acc = 1;
  } else {
    load_acc = 0;
  }
  if ( 0 == env_load_acc ) {
  } else {
    load_acc = atoi(env_load_acc);
  }

  i_gp_reg_mapping->gp_reg_n        = LIBXSMM_AARCH64_GP_REG_X9;
  i_gp_reg_mapping->gp_reg_ind_base = LIBXSMM_AARCH64_GP_REG_X10;
  i_gp_reg_mapping->gp_reg_in_base  = LIBXSMM_AARCH64_GP_REG_X11;
  i_gp_reg_mapping->gp_reg_out      = LIBXSMM_AARCH64_GP_REG_X12;
  i_gp_reg_mapping->gp_reg_m_loop   = LIBXSMM_AARCH64_GP_REG_X13;
  i_gp_reg_mapping->gp_reg_n_loop   = LIBXSMM_AARCH64_GP_REG_X14;
  i_gp_reg_mapping->gp_reg_in       = LIBXSMM_AARCH64_GP_REG_X15;
  gp_reg_index                        = ( idx_tsize == 8 ) ? LIBXSMM_AARCH64_GP_REG_X16 : LIBXSMM_AARCH64_GP_REG_W16;
  gp_reg_index_64bit                  = LIBXSMM_AARCH64_GP_REG_X16;
  i_gp_reg_mapping->gp_reg_scratch_0  = LIBXSMM_AARCH64_GP_REG_X17;
  gp_reg_ldi                          = LIBXSMM_AARCH64_GP_REG_X1;

  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 48, i_gp_reg_mapping->gp_reg_n );

  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_n, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_gp_reg_mapping->gp_reg_n );

  libxsmm_aarch64_instruction_cond_jump_to_label( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_CBZ, i_gp_reg_mapping->gp_reg_n, END_LABEL, p_jump_label_tracker );

  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 40, i_gp_reg_mapping->gp_reg_ind_base );
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 32, i_gp_reg_mapping->gp_reg_in_base );
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 64, i_gp_reg_mapping->gp_reg_out );
  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_EOR_SR, gp_reg_index_64bit, gp_reg_index_64bit, gp_reg_index_64bit, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, gp_reg_ldi, (long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in );

  m                 = i_mateltwise_desc->m;
  use_m_masking     = (m % vlen == 0) ? 0 : 1;
  m_trips           = (m + vlen - 1) / vlen;
  m_unroll_factor   = m_trips;
  m_trips_loop      = 1;
  peeled_m_trips    = 0;

  /* set pred reg 0 to true for sve */
  if ( is_sve ) {
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, pred_reg_all, -1, i_gp_reg_mapping->gp_reg_scratch_0);
  }

  if (use_m_masking == 1) {
    mask_count = m % vlen;
    if (is_sve) libxsmm_generator_set_p_register_aarch64_sve(io_generated_code, pred_reg_mask, i_micro_kernel_config->datatype_size_in * mask_count, i_gp_reg_mapping->gp_reg_scratch_0);
  }

  if ((is_sve > 0) && (l_is_inp_bf16 > 0 || l_is_out_bf16 > 0)) {
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, pred_reg_all_bf16,
    (l_is_inp_bf16 > 0) ? i_micro_kernel_config->datatype_size_in * vlen : i_micro_kernel_config->datatype_size_out * vlen, i_gp_reg_mapping->gp_reg_scratch_0 );
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

  if (m_trips_loop > 1) {
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, m_trips_loop );
  }

  if (m_trips_loop >= 1) {
    for (im = 0; im < m_unroll_factor; im++) {
      if (load_acc == 0) {
        if (is_sve){
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                                   im, im, 0, im, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                     im, im, 0, im, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
        }
      } else {
        l_masked_elements = (l_is_out_bf16 == 0) ? 0 : vlen;
        pred_reg_mask_use = (l_is_out_bf16 == 0) ? pred_reg_all : pred_reg_all_bf16;
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0,
                                                         im, i_micro_kernel_config->datatype_size_out, l_masked_elements, 1, 0, pred_reg_mask_use );
        if (l_is_out_bf16 > 0) {
          libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, im, 0);
        }
      }
    }

    if (load_acc == 1) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB, i_gp_reg_mapping->gp_reg_out,
        i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out, (long long)m_unroll_factor * vlen * i_micro_kernel_config->datatype_size_out );
    }

    /* Perform the reductions for all columns */
    libxsmm_generator_loop_header_gp_reg_bound_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, i_gp_reg_mapping->gp_reg_n );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_POST, i_gp_reg_mapping->gp_reg_ind_base, LIBXSMM_AARCH64_GP_REG_UNDEF, idx_tsize, gp_reg_index );
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MUL, gp_reg_index_64bit, gp_reg_ldi, i_gp_reg_mapping->gp_reg_in, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_in_base, i_gp_reg_mapping->gp_reg_in, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );

    for (im = 0; im < m_unroll_factor; im++) {
      l_masked_elements = (l_is_inp_bf16 == 0) ? 0 : vlen;
      pred_reg_mask_use = (l_is_inp_bf16 == 0) ? pred_reg_all : pred_reg_all_bf16;
      libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_in,
        i_gp_reg_mapping->gp_reg_scratch_0, aux_vreg_offset+im, i_micro_kernel_config->datatype_size_in, l_masked_elements, 1, 0, pred_reg_mask_use );
      if (l_is_inp_bf16 > 0) {
          libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, aux_vreg_offset+im, 0);
      }
      if (is_sve){
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FADD_V,
                                                   aux_vreg_offset+im, im, 0, im, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
      } else {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V,
                                                   aux_vreg_offset+im, im, 0, im, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
      }
    }

    libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, 1);

    /* Now store accumulators */
    for (im = 0; im < m_unroll_factor; im++) {
      l_masked_elements = (l_is_out_bf16 == 0) ? 0 : vlen;
      pred_reg_mask_use = (l_is_out_bf16 == 0) ? pred_reg_all : pred_reg_all_bf16;
      if (l_is_out_bf16 > 0) {
        libxsmm_generator_vcvt_f32bf16_aarch64( io_generated_code, im, 0);
      }
      libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_out,
        i_gp_reg_mapping->gp_reg_scratch_0, im, i_micro_kernel_config->datatype_size_out, l_masked_elements, 1, 1, pred_reg_mask_use );
    }
  }

  if (m_trips_loop > 1) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 40, i_gp_reg_mapping->gp_reg_ind_base );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, i_gp_reg_mapping->gp_reg_in_base,
      i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in_base, (long long)i_micro_kernel_config->datatype_size_in * vlen * m_unroll_factor );
    libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, 1);
  }

  if (peeled_m_trips > 0) {
    if (m_trips_loop >= 1) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 40, i_gp_reg_mapping->gp_reg_ind_base );
    }
    if (m_trips_loop == 1) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
        i_gp_reg_mapping->gp_reg_in_base, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in_base,
        (long long)i_micro_kernel_config->datatype_size_in * vlen * m_unroll_factor );
    }

    /* Perform the reductions for all columns */
    for (im = 0; im < peeled_m_trips; im++) {
      if (load_acc == 0) {
        if (is_sve){
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
            im, im, 0, im, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
            im, im, 0, im, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
        }
      } else {
        mask_count2 = ((im == peeled_m_trips-1) && (use_m_masking > 0 )) ? mask_count : 0;
        l_masked_elements = (l_is_out_bf16 == 0) ? mask_count2 : (mask_count2 > 0) ? mask_count2 : vlen;
        pred_reg_mask_use = (l_is_out_bf16 == 0) ? ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all) : ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all_bf16);
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_out,
          i_gp_reg_mapping->gp_reg_scratch_0, im, i_micro_kernel_config->datatype_size_out, l_masked_elements, 1, 0, pred_reg_mask_use);
        if (l_is_out_bf16 > 0) {
          libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, im, 0);
        }
      }
    }

    if (load_acc == 1) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
        i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out,
        (long long)peeled_m_trips * i_micro_kernel_config->datatype_size_out * vlen - (long long)use_m_masking * ((long long)vlen - mask_count) * i_micro_kernel_config->datatype_size_out );
    }

    libxsmm_generator_loop_header_gp_reg_bound_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, i_gp_reg_mapping->gp_reg_n );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_POST, i_gp_reg_mapping->gp_reg_ind_base, LIBXSMM_AARCH64_GP_REG_UNDEF, idx_tsize, gp_reg_index );
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MUL, gp_reg_index_64bit, gp_reg_ldi, i_gp_reg_mapping->gp_reg_in, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_in_base, i_gp_reg_mapping->gp_reg_in, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );

    for (im = 0; im < peeled_m_trips; im++) {
      mask_count2 = ((im == peeled_m_trips-1) && (use_m_masking > 0 )) ? mask_count : 0;
      l_masked_elements = (l_is_inp_bf16 == 0) ? mask_count2 : (mask_count2 > 0) ? mask_count2 : vlen;
      pred_reg_mask_use = (l_is_inp_bf16 == 0) ? ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all) : ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all_bf16);
      libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_in,
        i_gp_reg_mapping->gp_reg_scratch_0, aux_vreg_offset+im, i_micro_kernel_config->datatype_size_in,
        l_masked_elements, 1, 0, pred_reg_mask_use );
      if (l_is_inp_bf16 > 0) {
        libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, aux_vreg_offset+im, 0);
      }
      if (is_sve){
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FADD_V,
          aux_vreg_offset+im, im, 0, im, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
      } else {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V,
          aux_vreg_offset+im, im, 0, im, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
      }
    }

    libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, 1);

    /* Now store accumulators */
    for (im = 0; im < peeled_m_trips; im++) {
      mask_count2 = ((im == peeled_m_trips-1) && (use_m_masking > 0 )) ? mask_count : 0;
      l_masked_elements = (l_is_out_bf16 == 0) ? mask_count2 : (mask_count2 > 0) ? mask_count2 : vlen;
      pred_reg_mask_use = (l_is_out_bf16 == 0) ? ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all) : ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all_bf16);
      if (l_is_out_bf16 > 0) {
        libxsmm_generator_vcvt_f32bf16_aarch64( io_generated_code, im, 0);
      }
      libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_out,
        i_gp_reg_mapping->gp_reg_scratch_0, im, i_micro_kernel_config->datatype_size_out,
        l_masked_elements, 1, 1, pred_reg_mask_use );
    }
  }

  libxsmm_aarch64_instruction_register_jump_label( io_generated_code, END_LABEL, p_jump_label_tracker );

#if defined(LIBXSMM_GENERATOR_MATELTWISE_REDUCE_AARCH64_JUMP_LABEL_TRACKER_MALLOC)
  free(p_jump_label_tracker);
#endif
}
