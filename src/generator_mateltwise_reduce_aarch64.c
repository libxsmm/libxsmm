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
  unsigned int reduce_instr = 0, absmax_instr = 0;
  unsigned int vmove_instruction_in = i_micro_kernel_config->vmove_instruction_in;
  unsigned int vmove_instruction_out = i_micro_kernel_config->vmove_instruction_out;

  unsigned int vlen = libxsmm_cpuid_vlen32(i_micro_kernel_config->instruction_set);
  unsigned int tmp_vreg = 31;
  unsigned int aux_vreg = 0;
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
                                   (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ABSMAX) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MIN)) ? 1 : 0;
  unsigned int flag_reduce_elts_sq = ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X2_OP_ADD) ||
                                   (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD)) ? 1 : 0;
  unsigned int flag_reduce_op_add =((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD) ||
                                   (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD) ||
                                   (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X2_OP_ADD)) ? 1 : 0;
  unsigned int flag_reduce_op_max = (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX) ? 1 : 0;
  unsigned int flag_reduce_op_absmax = (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ABSMAX) ? 1 : 0;
  unsigned int flag_reduce_op_min = (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MIN) ? 1 : 0;
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
  } else if ( flag_reduce_op_min > 0 ) {
    reduce_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FMIN_V_P : LIBXSMM_AARCH64_INSTR_ASIMD_FMIN_V;
  } else if ( flag_reduce_op_absmax > 0 ) {
    reduce_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FMAX_V_P : LIBXSMM_AARCH64_INSTR_ASIMD_FMAX_V;
    absmax_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_AND_V : LIBXSMM_AARCH64_INSTR_ASIMD_AND_V;
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

  if (flag_reduce_op_absmax > 0) {
    /* Load the signmask in aux vreg  */
    aux_vreg = max_m_unrolling;
    max_m_unrolling--;
    if (is_sve) {
      libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(aux_vreg), i_gp_reg_mapping->gp_reg_scratch_0,
                                                            sve_type, pred_reg_all, (i_micro_kernel_config->datatype_size_in == 8) ? 0x7fffffffffffffff : 0x7fffffff  );
    } else {
      libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(aux_vreg), i_gp_reg_mapping->gp_reg_scratch_0,
                                                            asimd_type, (i_micro_kernel_config->datatype_size_in == 8) ? 0x7fffffffffffffff : 0x7fffffff );
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
          if ( flag_reduce_op_add > 0 || flag_reduce_op_absmax > 0 ) {
            if (is_sve){
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                start_vreg_sum + im + _in * m_unroll_factor, start_vreg_sum + im + _in * m_unroll_factor,
                0, start_vreg_sum + im + _in * m_unroll_factor, 0, sve_type );
            } else {
              libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                start_vreg_sum + im + _in * m_unroll_factor, start_vreg_sum + im + _in * m_unroll_factor,
                0, start_vreg_sum + im + _in * m_unroll_factor, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
            }
          } else if ( flag_reduce_op_max > 0 || flag_reduce_op_min > 0) {
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
      if ( flag_reduce_op_max > 0 || flag_reduce_op_min > 0 ) {
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
              if (flag_reduce_op_absmax > 0) {
                libxsmm_aarch64_instruction_sve_compute( io_generated_code, absmax_instr, tmp_vreg, aux_vreg, 0, tmp_vreg,  pred_reg_all, sve_type );
              }
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, reduce_instr,
                                                       tmp_vreg, start_vreg_sum + im + _in * m_unroll_factor,
                                                       0, start_vreg_sum + im + _in * m_unroll_factor, pred_reg_all,
                                                       sve_type );
            } else {
              if (flag_reduce_op_absmax > 0) {
                libxsmm_aarch64_instruction_asimd_compute( io_generated_code, absmax_instr, tmp_vreg, aux_vreg, 0, tmp_vreg, asimd_type );
              }
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
              if (flag_reduce_op_absmax > 0) {
                libxsmm_aarch64_instruction_sve_compute( io_generated_code, absmax_instr, tmp_vreg, aux_vreg, 0, tmp_vreg,  pred_reg_all, sve_type );
              }
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, reduce_instr,
                                                       tmp_vreg, start_vreg_sum + im + _in * m_unroll_factor,
                                                       0, start_vreg_sum + im + _in * m_unroll_factor, pred_reg_all, sve_type );
            } else {
              if (flag_reduce_op_absmax > 0) {
                libxsmm_aarch64_instruction_asimd_compute( io_generated_code, absmax_instr, tmp_vreg, aux_vreg, 0, tmp_vreg, asimd_type );
              }
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
            if (flag_reduce_op_absmax > 0) {
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, absmax_instr, tmp_vreg, aux_vreg, 0, tmp_vreg,  pred_reg_all, sve_type );
            }
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, reduce_instr,
                                                     tmp_vreg, start_vreg_sum + im,
                                                     0, start_vreg_sum + im, pred_reg_all,
                                                     sve_type );
          } else {
            if (flag_reduce_op_absmax > 0) {
              libxsmm_aarch64_instruction_asimd_compute( io_generated_code, absmax_instr, tmp_vreg, aux_vreg, 0, tmp_vreg, asimd_type );
            }
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
          if ( flag_reduce_op_add > 0 || flag_reduce_op_absmax > 0  ) {
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
          } else if ( flag_reduce_op_max > 0 || flag_reduce_op_min > 0) {
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
      if ( flag_reduce_op_max > 0 || flag_reduce_op_min > 0) {
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
              if (flag_reduce_op_absmax > 0) {
                libxsmm_aarch64_instruction_sve_compute( io_generated_code, absmax_instr, tmp_vreg, aux_vreg, 0, tmp_vreg,  pred_reg_all, sve_type );
              }
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, reduce_instr,
                                                       tmp_vreg,
                                                       start_vreg_sum + im + _in * m_unroll_factor, 0,
                                                       start_vreg_sum + im + _in * m_unroll_factor,
                                                       pred_reg_all, sve_type );
            } else {
              if (flag_reduce_op_absmax > 0) {
                libxsmm_aarch64_instruction_asimd_compute( io_generated_code, absmax_instr, tmp_vreg, aux_vreg, 0, tmp_vreg, asimd_type );
              }
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
              if (flag_reduce_op_absmax > 0) {
                libxsmm_aarch64_instruction_sve_compute( io_generated_code, absmax_instr, tmp_vreg, aux_vreg, 0, tmp_vreg,  pred_reg_all, sve_type );
              }
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, reduce_instr,
                                                       tmp_vreg,
                                                       start_vreg_sum + im + _in * m_unroll_factor, 0,
                                                       start_vreg_sum + im + _in * m_unroll_factor,
                                                       pred_reg_all, sve_type );
            } else {
              if (flag_reduce_op_absmax > 0) {
                libxsmm_aarch64_instruction_asimd_compute( io_generated_code, absmax_instr, tmp_vreg, aux_vreg, 0, tmp_vreg, asimd_type );
              }
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
            if (flag_reduce_op_absmax > 0) {
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, absmax_instr, tmp_vreg, aux_vreg, 0, tmp_vreg,  pred_reg_all, sve_type );
            }
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, reduce_instr,
                                                     tmp_vreg, start_vreg_sum + im,
                                                     0, start_vreg_sum + im, pred_reg_all,
                                                     sve_type );
          } else {
            if (flag_reduce_op_absmax > 0) {
              libxsmm_aarch64_instruction_asimd_compute( io_generated_code, absmax_instr, tmp_vreg, aux_vreg, 0, tmp_vreg, asimd_type );
            }
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
  unsigned int reduce_instr = 0, hreduce_instr = 0, absmax_instr = 0;
  unsigned int reg_sum = 31, reg_sum_squared = 30;
  unsigned int aux_vreg = 0;
  unsigned int cur_vreg = 0;
  unsigned int vlen = libxsmm_cpuid_vlen32(i_micro_kernel_config->instruction_set);
  unsigned int available_vregs = 30;
  unsigned int mask_count = 0, mask_count2;
  unsigned int flag_reduce_elts = ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD) ||
                                   (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD) ||
                                   (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ABSMAX) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MIN)) ? 1 : 0;
  unsigned int flag_reduce_elts_sq = ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X2_OP_ADD) ||
                                   (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD)) ? 1 : 0;
  unsigned int flag_reduce_op_add =((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD) ||
                                   (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD) ||
                                   (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X2_OP_ADD)) ? 1 : 0;
  unsigned int flag_reduce_op_max = (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX) ? 1 : 0;
  unsigned int flag_reduce_op_absmax = (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ABSMAX) ? 1 : 0;
  unsigned int flag_reduce_op_min = (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MIN) ? 1 : 0;
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
  unsigned int reduce_on_output_absmax_instr = 0;

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
  } else if ( flag_reduce_op_min > 0 ) {
    reduce_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FMIN_V_P : LIBXSMM_AARCH64_INSTR_ASIMD_FMIN_V;
    hreduce_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FMINV_V_P : LIBXSMM_AARCH64_INSTR_ASIMD_FMINP_V;
  } else if ( flag_reduce_op_absmax > 0 ) {
    reduce_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FMAX_V_P : LIBXSMM_AARCH64_INSTR_ASIMD_FMAX_V;
    hreduce_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FMAXV_V_P : LIBXSMM_AARCH64_INSTR_ASIMD_FMAXP_V;
    absmax_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_AND_V : LIBXSMM_AARCH64_INSTR_ASIMD_AND_V;
    reduce_on_output_absmax_instr = LIBXSMM_AARCH64_INSTR_ASIMD_AND_V;
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

  if ((use_m_masking > 0) && ( flag_reduce_op_max > 0 || flag_reduce_op_min > 0 )) {
    aux_vreg = available_vregs - 1;
    available_vregs--;
    if (!is_sve){/* vreg masks are not used in sve */
      if (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
        if (mask_count == 1) {
          unsigned int mask_array_neg_inf[4] = { 0x0, 0x0, 0x0, 0xfff00000 };
          unsigned int mask_array_pos_inf[4] = { 0x0, 0x0, 0x0, 0x7ff00000 };
          libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(aux_vreg), i_gp_reg_mapping->gp_reg_scratch_0, LIBXSMM_AARCH64_GP_REG_X12, (flag_reduce_op_max > 0) ? mask_array_neg_inf : mask_array_pos_inf, 0 );
        }
      } else {
        if (mask_count == 1) {
          unsigned int mask_array_neg_inf[4] = { 0x0, 0xff800000, 0xff800000, 0xff800000 };
          unsigned int mask_array_pos_inf[4] = { 0x0, 0x7f800000, 0x7f800000, 0x7f800000 };
          libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(aux_vreg), i_gp_reg_mapping->gp_reg_scratch_0, LIBXSMM_AARCH64_GP_REG_X12, (flag_reduce_op_max > 0) ? mask_array_neg_inf : mask_array_pos_inf, 0 );
        } else if (mask_count == 2) {
          unsigned int mask_array_neg_inf[4] = { 0x0, 0x0, 0xff800000, 0xff800000 };
          unsigned int mask_array_pos_inf[4] = { 0x0, 0x0, 0x7f800000, 0x7f800000 };
          libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(aux_vreg), i_gp_reg_mapping->gp_reg_scratch_0, LIBXSMM_AARCH64_GP_REG_X12, (flag_reduce_op_max > 0) ? mask_array_neg_inf : mask_array_pos_inf, 0 );
        } else if (mask_count == 3) {
          unsigned int mask_array_neg_inf[4] = { 0x0, 0x0, 0x0, 0xff800000 };
          unsigned int mask_array_pos_inf[4] = { 0x0, 0x0, 0x0, 0x7f800000 };
          libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(aux_vreg), i_gp_reg_mapping->gp_reg_scratch_0, LIBXSMM_AARCH64_GP_REG_X12, (flag_reduce_op_max > 0) ? mask_array_neg_inf : mask_array_pos_inf, 0 );
        }
      }
    }
  } else if (flag_reduce_op_absmax > 0) {
    /* Load the signmask in aux vreg  */
    aux_vreg = available_vregs - 1;
    available_vregs--;
    if (is_sve) {
      libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(aux_vreg), i_gp_reg_mapping->gp_reg_scratch_0,
                                                            sve_type, pred_reg_all, (i_micro_kernel_config->datatype_size_in == 8) ? 0x7fffffffffffffff : 0x7fffffff  );
    } else {
      libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(aux_vreg), i_gp_reg_mapping->gp_reg_scratch_0,
                                                            asimd_type, (i_micro_kernel_config->datatype_size_in == 8) ? 0x7fffffffffffffff : 0x7fffffff );
    }
  }

  if (n > 1) {
    /* open n loop */
    libxsmm_generator_loop_header_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, n);
  }

  /* Initialize accumulators to zero */
  if ( compute_plain_vals_reduce > 0 ) {
    if ( flag_reduce_op_add > 0 || flag_reduce_op_absmax > 0 ) {
      if (is_sve){
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                reg_sum, reg_sum, 0, reg_sum, 0, sve_type );
      } else {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                reg_sum, reg_sum, 0, reg_sum, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
      }
    } else if ( flag_reduce_op_max > 0 || flag_reduce_op_min > 0) {
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
      if ((use_m_masking == 1) && (im == m_trips - 1) && (flag_reduce_op_max > 0 || flag_reduce_op_min > 0)) {
        /* not needed in sve */
        /*libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
          aux_vreg, cur_vreg, 0, cur_vreg, 0, LIBXSMM_AARCH64_SVE_TYPE_S );*/
      }

      if ( compute_plain_vals_reduce > 0 ) {
        unsigned int pred_reg_compute = pred_reg_all;
        if ((flag_reduce_op_max > 0 || flag_reduce_op_min > 0) && (im == m_trips-1) && (use_m_masking > 0)) {
          if (l_is_inp_bf16 > 0 || l_is_out_bf16 > 0) {
            pred_reg_compute = pred_reg_mask_compute_f32;
          } else {
            pred_reg_compute = pred_reg_mask;
          }
        }

        if (flag_reduce_op_absmax > 0) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, absmax_instr,
            cur_vreg, aux_vreg, 0, cur_vreg,
            pred_reg_compute, sve_type );
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
      if ((use_m_masking == 1) && (im == m_trips - 1) && (flag_reduce_op_max > 0 || flag_reduce_op_min > 0)) {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
          aux_vreg, cur_vreg, 0, cur_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
      }

      if ( compute_plain_vals_reduce > 0 ) {
        if (flag_reduce_op_absmax > 0) {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, absmax_instr,
            cur_vreg, aux_vreg, 0, cur_vreg, asimd_type );
        }
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
      if (flag_reduce_op_absmax > 0) {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, reduce_on_output_absmax_instr,
          cur_vreg, aux_vreg, 0, cur_vreg, asimd_type );
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
  unsigned int mask_argidx64 = 0;
  unsigned char pred_reg_all = 0; /* set by caller */
  unsigned char pred_reg_mask = 1;
  unsigned char pred_reg_all_bf16 = 2;
  unsigned char pred_reg_argidx32 = 3;
  unsigned char pred_reg_argidx64 = 4;
  unsigned int l_idx0_vreg_offset = 0, l_idx1_vreg_offset = 0;
  unsigned char pred_reg_mask_use = 0;
  unsigned int l_masked_elements = 0;
  unsigned int l_m_code_blocks = 0;
  unsigned int l_m_code_block_id = 0;
  unsigned int l_record_argop = 0;
  unsigned int l_is_reduce_max = 0;
  unsigned int l_is_reduce_min = 0;
  unsigned int temp_vreg_argop = 31;
  int rbp_offset_inf = -32;
  int pos_rbp_offset_inf = 32;
  unsigned int gp_reg_argop = 0, temp_gpr = 0;
  unsigned int l_reduce_instr = 0, l_argop_cmp_instr = 0, l_argop_blend_instr = 0, l_use_stack_vars = 0;
  unsigned int l_is_inp_bf16 = (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ? 1 : 0;
  unsigned int l_is_out_bf16 = (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ? 1 : 0;
  unsigned int aux_vreg_offset = 0;
  unsigned char is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  unsigned int argop_mask = is_sve ? 5 : 30;
  unsigned int argop_mask_aux = is_sve ? 6 : 29;
#if defined(USE_ENV_TUNING)
  const char *const env_max_m_unroll = getenv("MAX_M_UNROLL_REDUCE_COLS_IDX");
#endif
  const char *const env_load_acc= getenv("LOAD_ACCS_REDUCE_COLS_IDX");
  libxsmm_jump_label_tracker l_jump_label_tracker;
  libxsmm_reset_jump_label_tracker(&l_jump_label_tracker);

  l_reduce_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FADD_V : LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V;
  if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_MAX || i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_MIN) {
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_MAX) {
      l_is_reduce_max = 1;
      l_reduce_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FMAX_V_P : LIBXSMM_AARCH64_INSTR_ASIMD_FMAX_V;
      l_argop_cmp_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FCMLE_P_V : LIBXSMM_AARCH64_INSTR_ASIMD_FCMGE_R_V;
      l_argop_blend_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_SEL_V_P : LIBXSMM_AARCH64_INSTR_ASIMD_BIT_V;
    }
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_MIN) {
      l_is_reduce_min = 1;
      l_reduce_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FMIN_V_P : LIBXSMM_AARCH64_INSTR_ASIMD_FMIN_V;
      l_argop_cmp_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FCMGT_P_V : LIBXSMM_AARCH64_INSTR_ASIMD_FCMGT_R_V;
      l_argop_blend_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_SEL_V_P : LIBXSMM_AARCH64_INSTR_ASIMD_BIF_V;
    }
    if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_RECORD_ARGOP) > 0) {
      l_record_argop = 1;
      l_use_stack_vars = 1;
    }
    if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_INF_ACC) > 0) {
      l_use_stack_vars = 1;
    }
  }

  max_m_unrolling = 16;
  if (l_record_argop > 0) {
    if (idx_tsize == 4) {
      max_m_unrolling = 10;
    } else {
      max_m_unrolling = 7;
    }
  }

#if defined(USE_ENV_TUNING)
  if ( 0 == env_max_m_unroll ) {
  } else {
    max_m_unrolling = LIBXSMM_MAX(1,atoi(env_max_m_unroll));
  }
#endif
  if ( 0 == env_load_acc ) {
  } else {
    load_acc = atoi(env_load_acc);
  }
  if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_INIT_ACC) > 0 ) {
    load_acc = 1;
  } else {
    load_acc = 0;
  }
  if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_INF_ACC) > 0) {
    load_acc = 0;
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
  gp_reg_argop                        = LIBXSMM_AARCH64_GP_REG_X7;
  temp_gpr                            = LIBXSMM_AARCH64_GP_REG_X6;

  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 48, i_gp_reg_mapping->gp_reg_n );

  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_n, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_gp_reg_mapping->gp_reg_n );

  libxsmm_aarch64_instruction_cond_jump_to_label( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_CBZ, i_gp_reg_mapping->gp_reg_n, END_LABEL, &l_jump_label_tracker );

  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 40, i_gp_reg_mapping->gp_reg_ind_base );
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 32, i_gp_reg_mapping->gp_reg_in_base );
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 64, i_gp_reg_mapping->gp_reg_out );
  if (l_record_argop) libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 72, gp_reg_argop );

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
    if (idx_tsize == 4) {
      if (is_sve) libxsmm_generator_set_p_register_aarch64_sve(io_generated_code, pred_reg_argidx32, idx_tsize * mask_count, i_gp_reg_mapping->gp_reg_scratch_0);
    }
  }

  if (l_use_stack_vars > 0) {
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_X29,  0, 0 );
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 40, 0 );
  }

  if (l_is_reduce_max || l_is_reduce_min) {
    if (load_acc == 0) {
      if (l_is_reduce_max) {
        libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, temp_gpr, (long long)0xff800000);
      }
      if (l_is_reduce_min) {
        libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, temp_gpr, (long long)0x7f800000);
      }
      libxsmm_generator_setval_stack_var_aarch64( io_generated_code, rbp_offset_inf, temp_gpr, i_gp_reg_mapping->gp_reg_scratch_0);
    }
  }

  LIBXSMM_ASSERT(0 != (vlen / 2));
  if ((idx_tsize == 8) && (m % (vlen/2) != 0) && (l_record_argop > 0)) {
    mask_argidx64 = m % (vlen/2);
    if (is_sve) libxsmm_generator_set_p_register_aarch64_sve(io_generated_code, pred_reg_argidx64, idx_tsize * mask_argidx64, i_gp_reg_mapping->gp_reg_scratch_0);
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

  if (m_trips_loop >= 1) l_m_code_blocks++;
  if (peeled_m_trips > 0) l_m_code_blocks++;

  for (l_m_code_block_id = 0; l_m_code_block_id < l_m_code_blocks; l_m_code_block_id++) {
    unsigned int l_is_peeled_loop = 0;
    if (m_trips_loop > 1 && l_m_code_block_id == 0) {
      libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, m_trips_loop );
    }
    if (peeled_m_trips > 0 && l_m_code_block_id == l_m_code_blocks - 1) {
      if (m_trips_loop >= 1) {
        libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 40, i_gp_reg_mapping->gp_reg_ind_base );
      }
      if (m_trips_loop == 1) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
          i_gp_reg_mapping->gp_reg_in_base, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in_base,
          (long long)i_micro_kernel_config->datatype_size_in * vlen * m_unroll_factor );
      }
    }
    if ((m_trips_loop == 0) || ((m_trips_loop >= 1) && (peeled_m_trips > 0) && (l_m_code_block_id == l_m_code_blocks - 1))) {
      m_unroll_factor = peeled_m_trips;
      l_is_peeled_loop = 1;
    }
    aux_vreg_offset = m_unroll_factor;
    l_idx0_vreg_offset = 2*aux_vreg_offset;
    l_idx1_vreg_offset = 3*aux_vreg_offset;

    for (im = 0; im < m_unroll_factor; im++) {
      if (load_acc == 0) {
        if (l_is_reduce_max || l_is_reduce_min) {
          if (im == 0) {
            libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I, LIBXSMM_AARCH64_GP_REG_X29, i_gp_reg_mapping->gp_reg_scratch_0, pos_rbp_offset_inf, 0 );
            if (is_sve){
              libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF,
                                                     i_gp_reg_mapping->gp_reg_scratch_0, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, im, pred_reg_all );
            } else {
              libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R,
                                                               i_gp_reg_mapping->gp_reg_scratch_0, LIBXSMM_AARCH64_GP_REG_UNDEF, im, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
            }
          } else {
            if (is_sve){
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                                         0, 0, 0, im, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
            } else {
              libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                                         0, 0, 0, im, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
            }
          }
        } else {
          if (is_sve){
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                                     im, im, 0, im, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
          } else {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                       im, im, 0, im, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
          }
        }
      } else {
        mask_count2 = ((im == peeled_m_trips-1) && (use_m_masking > 0) && (l_is_peeled_loop > 0)) ? mask_count : 0;
        l_masked_elements = (l_is_out_bf16 == 0) ? mask_count2 : (mask_count2 > 0) ? mask_count2 : vlen;
        pred_reg_mask_use = (l_is_out_bf16 == 0) ? ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all) : ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all_bf16);
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0,
                                                         im, i_micro_kernel_config->datatype_size_out, l_masked_elements, 1, 0, pred_reg_mask_use );
        if (l_is_out_bf16 > 0) {
          libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, im, 0);
        }
      }

      if (l_record_argop > 0) {
        if (is_sve){
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                                   im + l_idx0_vreg_offset,
                                                   im + l_idx0_vreg_offset, 0,
                                                   im + l_idx0_vreg_offset, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
          if (idx_tsize == 8) {
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                                     im + l_idx1_vreg_offset,
                                                     im + l_idx1_vreg_offset, 0,
                                                     im + l_idx1_vreg_offset, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
          }
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                     im + l_idx0_vreg_offset,
                                                     im + l_idx0_vreg_offset, 0,
                                                     im + l_idx0_vreg_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
          if (idx_tsize == 8) {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                       im + l_idx1_vreg_offset,
                                                       im + l_idx1_vreg_offset, 0,
                                                       im + l_idx1_vreg_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
          }
        }
      }
    }

    if (load_acc == 1) {
      long long l_adj_offset = m_unroll_factor * vlen * i_micro_kernel_config->datatype_size_out;
      if (l_is_peeled_loop) {
        l_adj_offset = (long long)peeled_m_trips * i_micro_kernel_config->datatype_size_out * vlen - (long long)use_m_masking * ((long long)vlen - mask_count) * i_micro_kernel_config->datatype_size_out;
      }
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB, i_gp_reg_mapping->gp_reg_out,
        i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out, l_adj_offset );
    }

    /* Perform the reductions for all columns */
    libxsmm_generator_loop_header_gp_reg_bound_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, i_gp_reg_mapping->gp_reg_n );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_POST, i_gp_reg_mapping->gp_reg_ind_base, LIBXSMM_AARCH64_GP_REG_UNDEF, idx_tsize, gp_reg_index );
    if (l_record_argop > 0) {
      /* load 1 value from a gp reg, and store into vreg */
      if (is_sve){
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_DUP_GP_V,
            gp_reg_index_64bit, gp_reg_index_64bit, 0, temp_vreg_argop, 0, (idx_tsize == 8) ? LIBXSMM_AARCH64_SVE_TYPE_D : LIBXSMM_AARCH64_SVE_TYPE_S );
      } else {
        libxsmm_aarch64_instruction_asimd_gpr_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_DUP_FULL,
            gp_reg_index_64bit, temp_vreg_argop, 0, (idx_tsize == 8) ? LIBXSMM_AARCH64_ASIMD_WIDTH_D : LIBXSMM_AARCH64_ASIMD_WIDTH_S );
      }
    }
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MUL, gp_reg_index_64bit, gp_reg_ldi, i_gp_reg_mapping->gp_reg_in, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_in_base, i_gp_reg_mapping->gp_reg_in, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );

    for (im = 0; im < m_unroll_factor; im++) {
      mask_count2 = ((im == peeled_m_trips-1) && (use_m_masking > 0 ) && (l_is_peeled_loop > 0)) ? mask_count : 0;
      l_masked_elements = (l_is_inp_bf16 == 0) ? mask_count2 : (mask_count2 > 0) ? mask_count2 : vlen;
      pred_reg_mask_use = (l_is_inp_bf16 == 0) ? ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all) : ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all_bf16);
      libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_in,
        i_gp_reg_mapping->gp_reg_scratch_0, aux_vreg_offset+im, i_micro_kernel_config->datatype_size_in, l_masked_elements, 1, 0, pred_reg_mask_use );
      if (l_is_inp_bf16 > 0) {
        libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, aux_vreg_offset+im, 0);
      }

      if (is_sve){
        if (l_record_argop) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, l_argop_cmp_instr, im + aux_vreg_offset, im, 0, argop_mask, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
          if (idx_tsize == 4) {
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, l_argop_blend_instr, im + l_idx0_vreg_offset, temp_vreg_argop, 0, im + l_idx0_vreg_offset, argop_mask, LIBXSMM_AARCH64_SVE_TYPE_S );
          } else {
            /* Extract lo mask to aux */
            libxsmm_aarch64_extract_mask2_from_mask4(io_generated_code, argop_mask, argop_mask_aux, 0);
             libxsmm_aarch64_instruction_sve_compute( io_generated_code, l_argop_blend_instr, im + l_idx0_vreg_offset, temp_vreg_argop, 0, im + l_idx0_vreg_offset, argop_mask_aux, LIBXSMM_AARCH64_SVE_TYPE_D );
            /* Extract hi mask to aux */
            libxsmm_aarch64_extract_mask2_from_mask4(io_generated_code, argop_mask, argop_mask_aux, 1);
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, l_argop_blend_instr, im + l_idx1_vreg_offset, temp_vreg_argop, 0, im + l_idx1_vreg_offset, argop_mask_aux, LIBXSMM_AARCH64_SVE_TYPE_D );
          }
        }
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, l_reduce_instr, im + aux_vreg_offset, im, 0, im, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
      } else {
        if (l_record_argop) {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, l_argop_cmp_instr, im + aux_vreg_offset, im, 0, argop_mask, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
          if (idx_tsize == 4) {
              libxsmm_aarch64_instruction_asimd_compute( io_generated_code, l_argop_blend_instr, temp_vreg_argop, argop_mask, 0, im + l_idx0_vreg_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
          } else {
            /* Extract lo mask to aux */
            libxsmm_aarch64_extract_mask2_from_mask4(io_generated_code, argop_mask, argop_mask_aux, 0);
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, l_argop_blend_instr, temp_vreg_argop, argop_mask_aux, 0, im + l_idx0_vreg_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
            /* Extract hi mask to aux */
            libxsmm_aarch64_extract_mask2_from_mask4(io_generated_code, argop_mask, argop_mask_aux, 1);
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, l_argop_blend_instr, temp_vreg_argop, argop_mask_aux, 0, im + l_idx1_vreg_offset, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
          }
        }
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, l_reduce_instr, im + aux_vreg_offset, im, 0, im, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S);
      }
    }

    libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, 1);

    /* Now store accumulators */
    for (im = 0; im < m_unroll_factor; im++) {
      mask_count2 = ((im == peeled_m_trips-1) && (use_m_masking > 0 ) && (l_is_peeled_loop > 0)) ? mask_count : 0;
      l_masked_elements = (l_is_out_bf16 == 0) ? mask_count2 : (mask_count2 > 0) ? mask_count2 : vlen;
      pred_reg_mask_use = (l_is_out_bf16 == 0) ? ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all) : ((mask_count2 > 0) ? pred_reg_mask : pred_reg_all_bf16);
      if (l_is_out_bf16 > 0) {
        libxsmm_generator_vcvt_f32bf16_aarch64( io_generated_code, im, 0);
      }
      libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_out,
        i_gp_reg_mapping->gp_reg_scratch_0, im, i_micro_kernel_config->datatype_size_out, l_masked_elements, 1, 1, pred_reg_mask_use );
    }

    if ( l_record_argop > 0 ) {
      if (idx_tsize == 4) {
        for (im = 0; im < m_unroll_factor; im++) {
          unsigned int vreg_id = im + l_idx0_vreg_offset;
          mask_count2 = ((im == peeled_m_trips-1) && (use_m_masking > 0) && (l_is_peeled_loop > 0)) ? mask_count : 0;
          libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, gp_reg_argop, i_gp_reg_mapping->gp_reg_scratch_0, vreg_id, 4, mask_count2, 1, 1, mask_count2 ? pred_reg_argidx32 : pred_reg_all );
        }
      } else {
        for (im = 0; im < m_unroll_factor; im++) {
          unsigned int use_mask_0 = ((im == peeled_m_trips-1) && (l_is_peeled_loop > 0) && (m % vlen < (vlen/2) && (m % vlen != 0))) ? 1 : 0;
          unsigned int use_mask_1 = ((im == peeled_m_trips-1) && (l_is_peeled_loop > 0) && (m % vlen > (vlen/2) && (m % vlen != 0))) ? 1 : 0;
          unsigned int vreg_id0 = im + l_idx0_vreg_offset;
          unsigned int vreg_id1 = im + l_idx1_vreg_offset;

          mask_count2 = (use_mask_0) ? mask_argidx64 : 0;
          libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, gp_reg_argop, i_gp_reg_mapping->gp_reg_scratch_0,
              vreg_id0, 8, mask_count2, 1, 1, mask_count2 == 0 ? pred_reg_all : pred_reg_argidx64 );
          if (use_mask_0 == 0 && !((m % vlen == vlen/2) && (im == peeled_m_trips-1) && (l_is_peeled_loop > 0))) {
            mask_count2 = (use_mask_1) ? mask_argidx64 : 0;
            libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, gp_reg_argop, i_gp_reg_mapping->gp_reg_scratch_0,
                vreg_id1, 8, mask_count2, 1, 1, mask_count2 == 0 ? pred_reg_all : pred_reg_argidx64 );
          }
        }
      }
    }

    if (m_trips_loop > 1 && l_m_code_block_id == 0) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 40, i_gp_reg_mapping->gp_reg_ind_base );
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, i_gp_reg_mapping->gp_reg_in_base,
        i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in_base, (long long)i_micro_kernel_config->datatype_size_in * vlen * m_unroll_factor );
      libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, 1);
    }
  }

  if (l_use_stack_vars > 0) {
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_X29, LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );
  }

  libxsmm_aarch64_instruction_register_jump_label( io_generated_code, END_LABEL, &l_jump_label_tracker );
}
