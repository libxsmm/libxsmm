/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/

#include "generator_common_x86.h"
#include "generator_mateltwise_avx_avx512.h"
#include "generator_mateltwise_cvtfp32bf16_act_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
void libxsmm_generator_cvtfp32bf16_vnni_format_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {

  unsigned int m, n, im, in, unroll_iter, use_m_masking = 0, m_trips = 0, n_trips = 0, vlen_in = 16, vlen_out = 32, mask_in_count = 0;
  unsigned int reserved_zmms = 1, max_nm_unrolling = 31, n_unroll_factor = 2;
  unsigned int mask_reg_in = 1, mask_reg_out = 2;
  unsigned int perm_vreg = 0, vreg_in_0 = 0, vreg_in_1 = 0;
  short vnni_perm_array[32] = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};

  /* Some rudimentary checking of M, N and LDs*/
  if ( (i_mateltwise_desc->m > i_mateltwise_desc->ldi) || (i_mateltwise_desc->m > i_mateltwise_desc->ldo) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
    return;
  }

  /* n % 2 should be zero in order to support vnni reformating  */
  if ( i_mateltwise_desc->n % 2 != 0  ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }

  vlen_in = 16;
  vlen_out = 32;

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in     = LIBXSMM_X86_GP_REG_R8;
  i_gp_reg_mapping->gp_reg_out    = LIBXSMM_X86_GP_REG_R9;
  i_gp_reg_mapping->gp_reg_n_loop = LIBXSMM_X86_GP_REG_R10;

  /* Load permute register */
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) vnni_perm_array, "vnni_perm_array_", i_micro_kernel_config->vector_name, perm_vreg);

  /* Load the input pointer and output pointer */
  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      0,
      i_gp_reg_mapping->gp_reg_in,
      0 );

  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      8,
      i_gp_reg_mapping->gp_reg_out,
      0 );

  /* We fully unroll in M dimension, calculate mask if there is remainder */
  m                 = i_mateltwise_desc->m;
  n                 = i_mateltwise_desc->n;
  use_m_masking     = ( m % vlen_in == 0 ) ? 0 : 1;
  m_trips           = (m + vlen_in - 1) / vlen_in;

  max_nm_unrolling = max_nm_unrolling - reserved_zmms;

  if (m_trips > max_nm_unrolling) {
    n_unroll_factor = 2;
  } else {
    /* Explore n unrolling opportunities... We unroll only by factors that divide N  */
    n_unroll_factor = n;
    while (m_trips * n_unroll_factor > max_nm_unrolling) {
      n_unroll_factor = n_unroll_factor - 2;
    }
    while (n % n_unroll_factor > 0) {
      n_unroll_factor = n_unroll_factor - 2;
    }
  }
  n_trips = n / n_unroll_factor;

  /* Calculate input and output masks in case we see m_masking */
  if (use_m_masking == 1) {
    mask_in_count = vlen_in - m % vlen_in;
    libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_R11, mask_reg_in, mask_in_count, LIBXSMM_GETENUM_INP(i_mateltwise_desc->datatype));
    libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_R11, mask_reg_out, 2*mask_in_count, LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype));
  }

  if (n_trips > 1) {
    /* open n loop */
    libxsmm_generator_mateltwise_header_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop );
  }

  for (in = 0; in < n_unroll_factor; in+=2) {
    for (im = 0; im < m_trips; im++) {
      unroll_iter = in * m_trips + im * 2;
      vreg_in_0 = unroll_iter % (max_nm_unrolling - reserved_zmms) + reserved_zmms;
      vreg_in_1 = vreg_in_0 + 1;

      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_in,
          i_gp_reg_mapping->gp_reg_in,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * vlen_in + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
          i_micro_kernel_config->vector_name,
          vreg_in_1, ((im == (m_trips-1)) && (use_m_masking == 1)) ? mask_reg_in : 0, 1, 0 );

      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_in,
          i_gp_reg_mapping->gp_reg_in,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * vlen_in + (in+1) * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
          i_micro_kernel_config->vector_name,
          vreg_in_0, ((im == (m_trips-1)) && (use_m_masking == 1)) ? mask_reg_in : 0, 1, 0 );

      libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
          LIBXSMM_X86_INSTR_VCVTNE2PS2BF16,
          i_micro_kernel_config->vector_name,
          vreg_in_1, vreg_in_0, vreg_in_0 );

      libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               LIBXSMM_X86_INSTR_VPERMW,
                                               i_micro_kernel_config->vector_name,
                                               vreg_in_0,
                                               perm_vreg,
                                               vreg_in_0);


      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_out,
          i_gp_reg_mapping->gp_reg_out,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * vlen_out + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
          i_micro_kernel_config->vector_name,
          vreg_in_0, ((im == (m_trips-1)) && (use_m_masking == 1)) ? mask_reg_out : 0, 0, 1 );

    }
  }

  if (n_trips > 1) {
    libxsmm_x86_instruction_alu_imm(  io_generated_code,
        i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_in,
        i_mateltwise_desc->ldi * n_unroll_factor * i_micro_kernel_config->datatype_size_in);

    libxsmm_x86_instruction_alu_imm(  io_generated_code,
        i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_out,
        i_mateltwise_desc->ldo * n_unroll_factor * i_micro_kernel_config->datatype_size_out);

    /* close n loop */
    libxsmm_generator_mateltwise_footer_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, n_trips);
  }

}

LIBXSMM_API_INTERN
void libxsmm_generator_cvtfp32bf16_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {

  unsigned int i = 0, im, in, m, n, n_trips, m_trips, use_m_masking, mask_in_count, mask_out_count, reg_0, reg_1, unroll_iter = 0, zero_reg = 0;
  unsigned int reserved_zmms = 0, max_nm_unrolling = 31, reserved_mask_regs = 1, current_mask_reg = 1;
  unsigned int n_unroll_factor = 1, eager_result_store = 0;
  unsigned int vec_x2 = 0, vec_nom = 0, vec_denom = 0, vec_c0 = 0, vec_c1 = 0, vec_c2 = 0, vec_c3 = 0, vec_c1_d = 0, vec_c2_d = 0, vec_c3_d = 0, vec_hi_bound = 0, vec_lo_bound = 0, vec_ones = 0, vec_neg_ones = 0, vec_halves = 0, mask_hi = 0, mask_lo = 0;
  unsigned int acvt_flags = (i_mateltwise_desc->operation == (unsigned char)LIBXSMM_MELTW_OPERATION_ACT_CVTFP32BF16) ? (unsigned int)i_mateltwise_desc->flags : (unsigned int)LIBXSMM_MELTW_FLAG_ACVT_NONE;
  unsigned int cvta_flags = (i_mateltwise_desc->operation == (unsigned char)LIBXSMM_MELTW_OPERATION_CVTFP32BF16_ACT) ? (unsigned int)i_mateltwise_desc->flags : (unsigned int)LIBXSMM_MELTW_FLAG_CVTA_NONE;
  unsigned int fuse_tanh_before_cvt = ( (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_ACT_CVTFP32BF16) && ((acvt_flags & LIBXSMM_MELTW_FLAG_ACVT_FUSE_TANH) > 0) ) ? 1 : 0;
  unsigned int fuse_sigmoid_before_cvt = ( (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_ACT_CVTFP32BF16) &&  ((acvt_flags & LIBXSMM_MELTW_FLAG_ACVT_FUSE_SIGM) > 0) ) ? 1 : 0;
  unsigned int fuse_relu_after_cvt = ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_CVTFP32BF16_ACT) && ((cvta_flags & LIBXSMM_MELTW_FLAG_CVTA_FUSE_RELU) > 0) ) ? 1 : 0;
  unsigned int fuse_tanh_after_cvt = ( (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_CVTFP32BF16_ACT) && ((cvta_flags & LIBXSMM_MELTW_FLAG_CVTA_FUSE_TANH) > 0) ) ? 1 : 0;
  unsigned int fuse_sigmoid_after_cvt = ( (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_CVTFP32BF16_ACT) &&  ((cvta_flags & LIBXSMM_MELTW_FLAG_CVTA_FUSE_SIGM) > 0) ) ? 1 : 0;

  if ((fuse_tanh_before_cvt == 1) && (fuse_sigmoid_before_cvt == 1)) {
    /* This should not happen  */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* For now the below options are not supported in JITer */
  if ((fuse_tanh_after_cvt == 1) || (fuse_sigmoid_after_cvt == 1)) {
    /* This should not happen  */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* Some rudimentary checking of M, N and LDs*/
  if ( (i_mateltwise_desc->m > i_mateltwise_desc->ldi) || (i_mateltwise_desc->m > i_mateltwise_desc->ldo) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
    return;
  }

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in     = LIBXSMM_X86_GP_REG_R8;
  i_gp_reg_mapping->gp_reg_out    = LIBXSMM_X86_GP_REG_R9;
  i_gp_reg_mapping->gp_reg_m_loop = LIBXSMM_X86_GP_REG_R10;
  i_gp_reg_mapping->gp_reg_n_loop = LIBXSMM_X86_GP_REG_R11;
  if ( fuse_relu_after_cvt == 1 ) {
    i_gp_reg_mapping->gp_reg_relumask = LIBXSMM_X86_GP_REG_R13;
  }

  /* load the input pointer and output pointer */
  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      0,
      i_gp_reg_mapping->gp_reg_in,
      0 );

  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      8,
      i_gp_reg_mapping->gp_reg_out,
      0 );

  if ( fuse_relu_after_cvt == 1 ) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        16,
        i_gp_reg_mapping->gp_reg_relumask,
        0 );
  }

  if (io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) {
    reserved_zmms += 3;
    reserved_mask_regs += 2;
  }

  /* Determine the names of the reserved registers and load with constants when applicable... */
  if ( (fuse_tanh_before_cvt == 1) || (fuse_sigmoid_before_cvt == 1) ) {
    float c0_array[16] = { 2027025.0f, 2027025.0f, 2027025.0f, 2027025.0f, 2027025.0f, 2027025.0f, 2027025.0f, 2027025.0f, 2027025.0f, 2027025.0f, 2027025.0f, 2027025.0f, 2027025.0f, 2027025.0f, 2027025.0f, 2027025.0f };
    float c1_array[16] = { 270270.0f, 270270.0f, 270270.0f, 270270.0f, 270270.0f, 270270.0f, 270270.0f, 270270.0f, 270270.0f, 270270.0f, 270270.0f, 270270.0f, 270270.0f, 270270.0f, 270270.0f, 270270.0f };
    float c2_array[16] = { 6930.0f, 6930.0f, 6930.0f, 6930.0f, 6930.0f, 6930.0f, 6930.0f, 6930.0f, 6930.0f, 6930.0f, 6930.0f, 6930.0f, 6930.0f, 6930.0f, 6930.0f, 6930.0f };
    float c3_array[16] = { 36.0f, 36.0f, 36.0f, 36.0f, 36.0f, 36.0f, 36.0f, 36.0f, 36.0f, 36.0f, 36.0f, 36.0f, 36.0f, 36.0f, 36.0f, 36.0f };
    float c1_d_array[16] = { 945945.0f, 945945.0f, 945945.0f, 945945.0f, 945945.0f, 945945.0f, 945945.0f, 945945.0f, 945945.0f, 945945.0f, 945945.0f, 945945.0f, 945945.0f, 945945.0f, 945945.0f, 945945.0f };
    float c2_d_array[16] = { 51975.0f, 51975.0f, 51975.0f, 51975.0f, 51975.0f, 51975.0f, 51975.0f, 51975.0f, 51975.0f, 51975.0f, 51975.0f, 51975.0f, 51975.0f, 51975.0f, 51975.0f, 51975.0f };
    float c3_d_array[16] = { 630.0f, 630.0f, 630.0f, 630.0f, 630.0f, 630.0f, 630.0f, 630.0f, 630.0f, 630.0f, 630.0f, 630.0f, 630.0f, 630.0f, 630.0f, 630.0f };
    float hi_b_array[16] = { 4.97f, 4.97f, 4.97f, 4.97f, 4.97f, 4.97f, 4.97f, 4.97f, 4.97f, 4.97f, 4.97f, 4.97f, 4.97f, 4.97f, 4.97f, 4.97f };
    float lo_b_array[16] = { -4.97f, -4.97f, -4.97f, -4.97f, -4.97f, -4.97f, -4.97f, -4.97f, -4.97f, -4.97f, -4.97f, -4.97f, -4.97f, -4.97f, -4.97f, -4.97f };
    float ones_array[16] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
    float neg_ones_array[16] = { -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f };

    reserved_zmms += 14;
    if (fuse_sigmoid_before_cvt == 1) {
      reserved_zmms++;
    }
    vec_x2        = reserved_zmms - 1;
    vec_nom       = reserved_zmms - 2;
    vec_denom     = reserved_zmms - 3;
    vec_c0        = reserved_zmms - 4;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c0_array, "c0_array_", i_micro_kernel_config->vector_name, vec_c0);
    vec_c1        = reserved_zmms - 5;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c1_array, "c1_array_", i_micro_kernel_config->vector_name, vec_c1);
    vec_c2        = reserved_zmms - 6;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c2_array, "c2_array_", i_micro_kernel_config->vector_name, vec_c2);
    vec_c3        = reserved_zmms - 7;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c3_array, "c3_array_", i_micro_kernel_config->vector_name, vec_c3);
    vec_c1_d      = reserved_zmms - 8;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c1_d_array, "c1_d_array_", i_micro_kernel_config->vector_name, vec_c1_d);
    vec_c2_d      = reserved_zmms - 9;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c2_d_array, "c2_d_array_", i_micro_kernel_config->vector_name, vec_c2_d);
    vec_c3_d      = reserved_zmms - 10;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c3_d_array, "c3_d_array_", i_micro_kernel_config->vector_name, vec_c3_d);
    vec_hi_bound  = reserved_zmms - 11;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) hi_b_array, "hi_b_array_", i_micro_kernel_config->vector_name, vec_hi_bound);
    vec_lo_bound  = reserved_zmms - 12;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) lo_b_array, "lo_b_array_", i_micro_kernel_config->vector_name, vec_lo_bound);
    vec_ones      = reserved_zmms - 13;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) ones_array, "ones_array_", i_micro_kernel_config->vector_name, vec_ones);
    vec_neg_ones  = reserved_zmms - 14;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) neg_ones_array, "neg_ones_array_", i_micro_kernel_config->vector_name, vec_neg_ones);
    if (fuse_sigmoid_before_cvt == 1) {
      float halves_array[16] = { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };
      vec_halves  = reserved_zmms - 15;
      libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) halves_array, "halves_array_", i_micro_kernel_config->vector_name, vec_halves);
    }
  }

  /* Set zero register neede for relu  */
  if ( fuse_relu_after_cvt == 1 ) {
    reserved_zmms++;
    zero_reg = reserved_zmms - 1;
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             LIBXSMM_X86_INSTR_VPXORD,
                                             i_micro_kernel_config->vector_name,
                                             zero_reg, zero_reg, zero_reg );
  }

  /* We fully unroll in M dimension, calculate mask if there is remainder */
  m                 = i_mateltwise_desc->m;
  n                 = i_mateltwise_desc->n;
  use_m_masking     = ( m % 32 == 0 ) ? 0 : 1;
  m_trips           = (m + 31) / 32;

  max_nm_unrolling = max_nm_unrolling - reserved_zmms;

  if (m_trips > max_nm_unrolling) {
    eager_result_store = 1;
    n_unroll_factor = 1;
  } else {
    /* Explore n unrolling opportunities... We unroll only by factors that divide N  */
    eager_result_store = 0;
    n_unroll_factor = n;
    while (m_trips * n_unroll_factor > max_nm_unrolling) {
      n_unroll_factor--;
    }
    while (n % n_unroll_factor > 0) {
      n_unroll_factor--;
    }
  }
  n_trips = n / n_unroll_factor;

  /* Calculate input and output masks in case we see m_masking */
  if (use_m_masking == 1) {
    /* If the remaining elements are < 16, then we read a full vector and a partial one at the last m trip */
    /* If the remaining elements are >= 16, then we read a partial vector at the last m trip  */
    /* Calculate mask reg 1 for input-reading */
    mask_in_count = ( (m % 32) > 16) ? 32 - (m % 32) : 16 - (m % 32);
    libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_R12, 1, mask_in_count, LIBXSMM_GEMM_PRECISION_F32);
    /* Calculate mask reg 2 for output-writing */
    mask_out_count = 32 - (m % 32);
    libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_R12, 2, mask_out_count, LIBXSMM_GEMM_PRECISION_BF16);
    reserved_mask_regs += 2;
  }

  /* Determine the names of the reserved registers... */
  if ( (fuse_tanh_before_cvt == 1) || (fuse_sigmoid_before_cvt == 1) ) {
    reserved_mask_regs += 2;
    mask_hi            = reserved_mask_regs - 1;
    mask_lo            = reserved_mask_regs - 2;
  }

  /* In this case we have to use CPX replacement sequence for downconverts... */
  if (io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) {
    /* init stack with helper variables for SW-based RNE rounding */
    /* push 0x7f800000 on the stack, naninf masking */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_R12, 0x7f800000);
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R12 );

    /* push 0x00010000 on the stack, fixup masking */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_R12, 0x00010000);
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R12 );

    /* push 0x00007fff on the stack, rneadd */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_R12, 0x00007fff);
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R12);

    /* push 0x00000001 on the stack, fixup */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_R12, 0x00000001);
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R12 );

    /* If we are using the 3 operant convert variant, then generate the proper permute table in zmm2 for the replacement code */
    if (m > 16) {
      short perm_array[32] = { 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
      short selector_array[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 };
      for (i = 0; i < 32; i++) {
        perm_array[i] = (short)(perm_array[i] | selector_array[i]);
      }
      libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
          (const unsigned char *) perm_array,
          "perm_arrray_",
          i_micro_kernel_config->vector_name,
          2);
    }
  }

  if (n_trips > 1) {
    /* open n loop */
    libxsmm_generator_mateltwise_header_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop );
  }

  for (in = 0; in < n_unroll_factor; in++) {
    for (im = 0; im < m_trips; im++) {
      unroll_iter = in * m_trips + im;
      if (unroll_iter + reserved_zmms < 16) {
        reg_0 = unroll_iter % (16-reserved_zmms) + reserved_zmms;
        reg_1 = unroll_iter % (16-reserved_zmms) + reserved_zmms + 16;
      } else {
        reg_0 = 16 + ((unroll_iter-16+reserved_zmms) % 15);
        reg_1 = reg_0 + 1;
      }

      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_in,
          i_gp_reg_mapping->gp_reg_in,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * 32 + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
          i_micro_kernel_config->vector_name,
          reg_0, ((im == (m_trips-1)) && (m % 32 < 16)) ? use_m_masking : 0, 1, 0 );

      /* If last iteration and remainder is less than 16, do not load anything  */
      if (!((use_m_masking == 1) && (im == m_trips-1) && (m % 32 <= 16))) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * 32 + 16 + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
            i_micro_kernel_config->vector_name,
            reg_1, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );
      }

      if ( (fuse_tanh_before_cvt == 1) || (fuse_sigmoid_before_cvt == 1) ) {
        if (fuse_sigmoid_before_cvt == 1) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                LIBXSMM_X86_INSTR_VMULPS,
                                                i_micro_kernel_config->vector_name,
                                                reg_0, vec_halves, reg_0 );
        }

        libxsmm_generator_tanh_ps_rational_78_avx512(io_generated_code, reg_0, vec_x2, vec_nom, vec_denom, mask_hi, mask_lo,
                                                     vec_c0, vec_c1, vec_c2, vec_c3, vec_c1_d, vec_c2_d, vec_c3_d, vec_hi_bound, vec_lo_bound, vec_ones, vec_neg_ones);

        if (fuse_sigmoid_before_cvt == 1) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                LIBXSMM_X86_INSTR_VADDPS,
                                                i_micro_kernel_config->vector_name,
                                                reg_0, vec_ones, reg_0 );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                LIBXSMM_X86_INSTR_VMULPS,
                                                i_micro_kernel_config->vector_name,
                                                reg_0, vec_halves, reg_0 );
        }

        if (!((use_m_masking == 1) && (im == m_trips-1) && (m % 32 <= 16))) {
          if (fuse_sigmoid_before_cvt == 1) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VMULPS,
                                                  i_micro_kernel_config->vector_name,
                                                  reg_1, vec_halves, reg_1 );
          }

          libxsmm_generator_tanh_ps_rational_78_avx512(io_generated_code, reg_1, vec_x2, vec_nom, vec_denom, mask_hi, mask_lo,
                                                       vec_c0, vec_c1, vec_c2, vec_c3, vec_c1_d, vec_c2_d, vec_c3_d, vec_hi_bound, vec_lo_bound, vec_ones, vec_neg_ones);

          if (fuse_sigmoid_before_cvt == 1) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VADDPS,
                                                  i_micro_kernel_config->vector_name,
                                                  reg_1, vec_ones, reg_1 );

            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VMULPS,
                                                  i_micro_kernel_config->vector_name,
                                                  reg_1, vec_halves, reg_1 );

          }
        }
      }

      /* Downconvert to BF16  */
      if (io_generated_code->arch >= LIBXSMM_X86_AVX512_CPX) {
        if (!((use_m_masking == 1) && (im == m_trips-1) && (m % 32 <= 16))) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
              LIBXSMM_X86_INSTR_VCVTNE2PS2BF16,
              i_micro_kernel_config->vector_name,
              reg_0, reg_1, reg_0 );
        } else {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
              LIBXSMM_X86_INSTR_VCVTNEPS2BF16,
              i_micro_kernel_config->vector_name,
              reg_0, reg_0 );
        }
      } else {
        if (!((use_m_masking == 1) && (im == m_trips-1) && (m % 32 <= 16))) {
          /* RNE convert reg_0 and reg_1 */
          libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, 'z', reg_0, reg_0, 0, 1, 6, 7 );
          libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, 'z', reg_1, reg_1, 0, 1, 6, 7 );
          /* Properly interleave reg_0 and reg_1 into reg_0  */
          libxsmm_x86_instruction_vec_compute_3reg(io_generated_code,
              LIBXSMM_X86_INSTR_VPERMT2W,
              i_micro_kernel_config->vector_name,
              reg_1,
              2,
              reg_0);
        } else {
          /* RNE convert reg_0 */
          libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, 'z', reg_0, reg_0, 0, 1, 6, 7 );
          /* shift FP32 by 16bit to right */
          libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code,
              LIBXSMM_X86_INSTR_VPSRAD_I,
              i_micro_kernel_config->vector_name,
              reg_0,
              reg_0,
              16);
          /* store 16 bit values into lower portion of reg_0 */
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
              LIBXSMM_X86_INSTR_VPMOVDW,
              i_micro_kernel_config->vector_name,
              reg_0, reg_0 );
        }
      }

      if ( fuse_relu_after_cvt == 1 ) {
        /* Compute relu mask */
        if (io_generated_code->arch >= LIBXSMM_X86_AVX512_CPX) {
          current_mask_reg = reserved_mask_regs + (unroll_iter % (8-reserved_mask_regs));
        } else {
          current_mask_reg = reserved_mask_regs - 2 + (unroll_iter % (8-reserved_mask_regs));
        }
        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
            LIBXSMM_X86_INSTR_VPCMPW,
            i_micro_kernel_config->vector_name,
            zero_reg,
            reg_0,
            current_mask_reg,
            6 );

        /* Store relu mask */
        libxsmm_x86_instruction_mask_move_mem( io_generated_code,
            LIBXSMM_X86_INSTR_KMOVD_ST,
            i_gp_reg_mapping->gp_reg_relumask,
            LIBXSMM_X86_GP_REG_UNDEF,
            0,
            (im * 32 + in * i_mateltwise_desc->ldo)/8,
            current_mask_reg );

        /* Blend output result with zero reg based on relu mask */
        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
            LIBXSMM_X86_INSTR_VPBLENDMW,
            i_micro_kernel_config->vector_name,
            reg_0,
            zero_reg,
            reg_0,
            current_mask_reg,
            0 );
      }

      /* Store the result here if have "eager store" decision */
      if (eager_result_store == 1) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_out,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * 32 + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
            i_micro_kernel_config->vector_name,
            reg_0, (im == (m_trips-1)) ? use_m_masking * 2 : 0, 0, 1 );
      }
    }
  }

  /* Store computed results... */
  if (eager_result_store == 0) {
    for (in = 0; in < n_unroll_factor; in++) {
      for (im = 0; im < m_trips; im++) {
        unroll_iter = in * m_trips + im;
        if (unroll_iter + reserved_zmms < 16) {
          reg_0 = unroll_iter % (16-reserved_zmms) + reserved_zmms;
        } else {
          reg_0 = 16 + ((unroll_iter-16+reserved_zmms) % 15);
        }
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_out,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * 32 + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
            i_micro_kernel_config->vector_name,
            reg_0, (im == (m_trips-1)) ? use_m_masking * 2 : 0, 0, 1 );
      }
    }
  }

  if (n_trips > 1) {
    /* Adjust input and output pointer */
    libxsmm_x86_instruction_alu_imm(  io_generated_code,
        i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_in,
        i_mateltwise_desc->ldi * n_unroll_factor * i_micro_kernel_config->datatype_size_in);

    libxsmm_x86_instruction_alu_imm(  io_generated_code,
        i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_out,
        i_mateltwise_desc->ldo *  n_unroll_factor * i_micro_kernel_config->datatype_size_out);

    /* In case of fused relu adjust also relu ptr, datatype for relumask tensor is "bit" and also it has always the same shape as output  */
    if ( fuse_relu_after_cvt == 1 ) {
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_relumask,
          (i_mateltwise_desc->ldo * n_unroll_factor)/8);
    }

    /* close n loop */
    libxsmm_generator_mateltwise_footer_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, n_trips);
  }

  if (io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R12 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R12 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R12 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R12 );
  }
}

