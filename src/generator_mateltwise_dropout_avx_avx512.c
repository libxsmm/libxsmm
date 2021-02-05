/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include "generator_common_x86.h"
#include "generator_mateltwise_avx_avx512.h"
#include "generator_mateltwise_dropout_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "libxsmm_main.h"

/*
 * Calling convention, this kernel assumes that all GPRs with are not in the argument list
 * and all xmm/ymm/zmm/tmm are caller save
 *
 * TODO; stack local variables....
 * */
LIBXSMM_API_INTERN
void libxsmm_generator_dropout_fwd_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                       libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                       const unsigned int                      i_gp_reg_in,
                                                       const unsigned int                      i_gp_reg_out,
                                                       const unsigned int                      i_gp_reg_dropmask,
                                                       const unsigned int                      i_gp_reg_rng_state,
                                                       const unsigned int                      i_gp_reg_prob,
                                                       const unsigned int                      i_gp_reg_m_loop,
                                                       const unsigned int                      i_gp_reg_n_loop,
                                                       const unsigned int                      i_gp_reg_tmp,
                                                       const unsigned int                      i_mask_reg_0,
                                                       const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                       const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int in, im, m, n, use_m_masking, m_trips, n_unroll_factor, n_trips, mask_out_count = 0, unroll_iter = 0;
  unsigned int reserved_mask_regs = 1, n_available_zmms = 22, n_available_mask_regs = 7, max_nm_unrolling = 16;
  unsigned int prob_vreg = 31, state0_vreg = 30, state1_vreg = 29, state2_vreg = 28, state3_vreg = 27;
  unsigned int rng_vreg_tmp0 = 26, rng_vreg_tmp1 = 25, rng_vreg_one = 24, rng_vreg_res = 23, invprob_vreg = 22;
  unsigned int cur_vreg, cur_mask_reg;
  LIBXSMM_UNUSED(i_gp_reg_m_loop);

  /* We fully unroll in M dimension, calculate mask if there is remainder */
  m                 = i_mateltwise_desc->m;
  n                 = i_mateltwise_desc->n;
  use_m_masking     = ( m % 16 == 0 ) ? 0 : 1;
  m_trips           = (m + 15) / 16;

  /* Calculate mask reg 1 for reading/output-writing */
  if (use_m_masking == 1) {
    mask_out_count = 16 - (m % 16);
    libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, i_gp_reg_tmp, i_mask_reg_0, mask_out_count, LIBXSMM_GEMM_PRECISION_F32);
    n_available_mask_regs--;
    reserved_mask_regs++;
  }

  if (m_trips > max_nm_unrolling) {
    n_unroll_factor = 1;
  } else {
    /* Explore n unrolling opportunities... We unroll only by factors that divide N  */
    n_unroll_factor = n;
    while (m_trips * n_unroll_factor > max_nm_unrolling) {
      n_unroll_factor--;
    }
    while (n % n_unroll_factor > 0) {
      n_unroll_factor--;
    }
  }
  n_trips = n / n_unroll_factor;

  /* load RNG state */
  libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->vmove_instruction_in,
                                    i_gp_reg_rng_state, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                    i_micro_kernel_config->vector_name, state0_vreg, 0, 1, 0 );
  libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->vmove_instruction_in,
                                    i_gp_reg_rng_state, LIBXSMM_X86_GP_REG_UNDEF, 0, 64,
                                    i_micro_kernel_config->vector_name, state1_vreg, 0, 1, 0 );
  libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->vmove_instruction_in,
                                    i_gp_reg_rng_state, LIBXSMM_X86_GP_REG_UNDEF, 0, 128,
                                    i_micro_kernel_config->vector_name, state2_vreg, 0, 1, 0 );
  libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->vmove_instruction_in,
                                    i_gp_reg_rng_state, LIBXSMM_X86_GP_REG_UNDEF, 0, 192,
                                    i_micro_kernel_config->vector_name, state3_vreg, 0, 1, 0 );

  /* load constant register */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_tmp, 0x3f800000);
  libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_tmp );
  libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set,
                                    LIBXSMM_X86_INSTR_VBROADCASTSS,
                                    LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                    i_micro_kernel_config->vector_name, rng_vreg_one, 0, 1, 0 );
  libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_tmp );

  /* load probability */
  libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set,
                                    LIBXSMM_X86_INSTR_VBROADCASTSS,
                                    i_gp_reg_prob, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                    i_micro_kernel_config->vector_name, prob_vreg, 0, 1, 0 );
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VSUBPS, 'z',
                                            prob_vreg, rng_vreg_one, prob_vreg );

  /* load 1/prob */
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VDIVPS, 'z',
                                            prob_vreg, rng_vreg_one, invprob_vreg );

  /* In this case we have to use CPX replacement sequence for downconverts... */
  if ( (io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) &&
       (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) ) {
    libxsmm_generator_vcvtneps2bf16_avx512_prep_stack( io_generated_code, i_gp_reg_tmp );

    n_available_mask_regs -= 2;
    reserved_mask_regs += 2;
  }

  /* open n loop, if needed */
  if (n_trips > 1) {
    libxsmm_generator_mateltwise_header_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_n_loop );
  }

  for (in = 0; in < n_unroll_factor; in++) {
    for (im = 0; im < m_trips; im++) {
      unroll_iter = in * m_trips + im;
      cur_vreg = unroll_iter % n_available_zmms;
      cur_mask_reg = reserved_mask_regs + unroll_iter % n_available_mask_regs;

      /* load input */
      if ( LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * 16 + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
            'y',
            cur_vreg, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );

        libxsmm_generator_cvtbf16ps_avx512( io_generated_code, 'z', cur_vreg, cur_vreg );
      } else {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * 16 + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
            i_micro_kernel_config->vector_name,
            cur_vreg, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );
      }

      /* draw a random number */
      libxsmm_generator_xoshiro128p_f32_avx512( io_generated_code, state0_vreg, state1_vreg, state2_vreg, state3_vreg,
                                                rng_vreg_tmp0, rng_vreg_tmp1, rng_vreg_one, rng_vreg_res );

      /* compare with p */
      libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCMPPS, 'z',
                                                     rng_vreg_res, prob_vreg, cur_mask_reg, 0x06  );

      /* weight and zero input */
      libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code, LIBXSMM_X86_INSTR_VMULPS, 'z',
                                                     cur_vreg, invprob_vreg, cur_vreg, cur_mask_reg, 1 );

      /* Store dropout mask */
      if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_DROPOUT_BITMASK) > 0 ) {
        libxsmm_x86_instruction_mask_move_mem( io_generated_code,
            LIBXSMM_X86_INSTR_KMOVW_ST,
            i_gp_reg_dropmask,
            LIBXSMM_X86_GP_REG_UNDEF,
            0,
            (im * 16 + in * i_mateltwise_desc->ldo)/8,
            cur_mask_reg );
      }

      /* Store result  */
      if ( LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
        if ( io_generated_code->arch >= LIBXSMM_X86_AVX512_CPX ) {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16,
                                                    i_micro_kernel_config->vector_name,
                                                    cur_vreg, cur_vreg );
        } else {
          libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                                                               cur_vreg, cur_vreg,
                                                               rng_vreg_tmp0, rng_vreg_tmp1, reserved_mask_regs-1, reserved_mask_regs-2 );
        }
        libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_out,
          i_gp_reg_out,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * 16 + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
          'y',
          cur_vreg, (im == (m_trips-1)) ? use_m_masking : 0, 0, 1 );
      } else {
        libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_out,
          i_gp_reg_out,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * 16 + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
          i_micro_kernel_config->vector_name,
          cur_vreg, (im == (m_trips-1)) ? use_m_masking : 0, 0, 1 );
      }
    }
  }

  if (n_trips > 1) {
    /* Adjust input and output pointer */
    libxsmm_x86_instruction_alu_imm(  io_generated_code,
        i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_in,
        i_mateltwise_desc->ldi * n_unroll_factor * i_micro_kernel_config->datatype_size_in);

    libxsmm_x86_instruction_alu_imm(  io_generated_code,
        i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_out,
        i_mateltwise_desc->ldo * n_unroll_factor * i_micro_kernel_config->datatype_size_out);

    /* Adjust also relu ptr, datatype for relumask tensor is "bit" and also it has always the same shape as output  */
    if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_DROPOUT_BITMASK) > 0 ) {
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_dropmask,
          (i_mateltwise_desc->ldo * n_unroll_factor)/8);
    }

    /* close n loop */
    libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_n_loop, n_trips );
  }

  /* In this case we have to use CPX replacement sequence for downconverts... */
  if ( (io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) &&
       (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) ) {
    libxsmm_generator_vcvtneps2bf16_avx512_clean_stack( io_generated_code, i_gp_reg_tmp );
  }

  /* store RNG state */
  libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->vmove_instruction_in,
                                    i_gp_reg_rng_state, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                    i_micro_kernel_config->vector_name, state0_vreg, 0, 0, 1 );
  libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->vmove_instruction_in,
                                    i_gp_reg_rng_state, LIBXSMM_X86_GP_REG_UNDEF, 0, 64,
                                    i_micro_kernel_config->vector_name, state1_vreg, 0, 0, 1 );
  libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->vmove_instruction_in,
                                    i_gp_reg_rng_state, LIBXSMM_X86_GP_REG_UNDEF, 0, 128,
                                    i_micro_kernel_config->vector_name, state2_vreg, 0, 0, 1 );
  libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->vmove_instruction_in,
                                    i_gp_reg_rng_state, LIBXSMM_X86_GP_REG_UNDEF, 0, 192,
                                    i_micro_kernel_config->vector_name, state3_vreg, 0, 0, 1 );
}

/*
 * Calling convention, this kernel assumes that all GPRs with are not in the argument list
 * and all xmm/ymm/zmm/tmm are caller save
 *
 * TODO; stack local variables....
 * */
LIBXSMM_API_INTERN
void libxsmm_generator_dropout_bwd_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                       libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                       const unsigned int                      i_gp_reg_in,
                                                       const unsigned int                      i_gp_reg_out,
                                                       const unsigned int                      i_gp_reg_dropmask,
                                                       const unsigned int                      i_gp_reg_prob,
                                                       const unsigned int                      i_gp_reg_m_loop,
                                                       const unsigned int                      i_gp_reg_n_loop,
                                                       const unsigned int                      i_gp_reg_tmp,
                                                       const unsigned int                      i_mask_reg_0,
                                                       const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                       const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int in, im, m, n, use_m_masking, m_trips, n_unroll_factor, n_trips, mask_out_count = 0, unroll_iter = 0;
  unsigned int reserved_mask_regs = 1, n_available_zmms = 26, n_available_mask_regs = 7, max_nm_unrolling = 16;
  unsigned int prob_vreg = 31, vreg_one = 30, vreg_tmp0 = 29, vreg_tmp1 = 28, vreg_tmp2 = 27, vreg_zero = 26;
  unsigned int cur_vreg, cur_mask_reg;
  LIBXSMM_UNUSED(i_gp_reg_m_loop);

  /* We fully unroll in M dimension, calculate mask if there is remainder */
  m                 = i_mateltwise_desc->m;
  n                 = i_mateltwise_desc->n;
  use_m_masking     = ( m % 16 == 0 ) ? 0 : 1;
  m_trips           = (m + 15) / 16;

  /* Calculate mask reg 1 for reading/output-writing */
  if (use_m_masking == 1) {
    mask_out_count = 16 - (m % 16);
    libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, i_gp_reg_tmp, i_mask_reg_0, mask_out_count, LIBXSMM_GEMM_PRECISION_F32);
    n_available_mask_regs--;
    reserved_mask_regs++;
  }

  if (m_trips > max_nm_unrolling) {
    n_unroll_factor = 1;
  } else {
    /* Explore n unrolling opportunities... We unroll only by factors that divide N  */
    n_unroll_factor = n;
    while (m_trips * n_unroll_factor > max_nm_unrolling) {
      n_unroll_factor--;
    }
    while (n % n_unroll_factor > 0) {
      n_unroll_factor--;
    }
  }
  n_trips = n / n_unroll_factor;

  /* load constant register */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_tmp, 0x3f800000);
  libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_tmp );
  libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set,
                                    LIBXSMM_X86_INSTR_VBROADCASTSS,
                                    LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                    i_micro_kernel_config->vector_name, vreg_one, 0, 1, 0 );
  libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_tmp );

  /* load probability */
  libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set,
                                    LIBXSMM_X86_INSTR_VBROADCASTSS,
                                    i_gp_reg_prob, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                    i_micro_kernel_config->vector_name, prob_vreg, 0, 1, 0 );
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VSUBPS, 'z',
                                            prob_vreg, vreg_one, prob_vreg );

  /* load 1/prob */
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VDIVPS, 'z',
                                            prob_vreg, vreg_one, prob_vreg );

  /* load zero, for masking */
  if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_DROPOUT_BITMASK) > 0 ) {
     n_available_zmms = 28;
  } else {
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, 'z',
                                              vreg_zero, vreg_zero, vreg_zero );
  }

  /* In this case we have to use CPX replacement sequence for downconverts... */
  if ( (io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) &&
       (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) ) {
    libxsmm_generator_vcvtneps2bf16_avx512_prep_stack( io_generated_code, i_gp_reg_tmp );

    n_available_mask_regs -= 2;
    reserved_mask_regs += 2;
  }

  /* open n loop, if needed */
  if (n_trips > 1) {
    libxsmm_generator_mateltwise_header_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_n_loop );
  }

  for (in = 0; in < n_unroll_factor; in++) {
    for (im = 0; im < m_trips; im++) {
      unroll_iter = in * m_trips + im;
      cur_vreg = unroll_iter % n_available_zmms;
      cur_mask_reg = reserved_mask_regs + unroll_iter % n_available_mask_regs;

      /* load input */
      if ( LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * 16 + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
            'y',
            cur_vreg, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );

        libxsmm_generator_cvtbf16ps_avx512( io_generated_code, 'z', cur_vreg, cur_vreg );
      } else {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * 16 + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
            i_micro_kernel_config->vector_name,
            cur_vreg, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );
      }

      /* load dropout mask */
      if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_DROPOUT_BITMASK) > 0 ) {
        libxsmm_x86_instruction_mask_move_mem( io_generated_code,
            LIBXSMM_X86_INSTR_KMOVW_LD,
            i_gp_reg_dropmask,
            LIBXSMM_X86_GP_REG_UNDEF,
            0,
            (im * 16 + in * i_mateltwise_desc->ldi)/8,
            cur_mask_reg );
      } else {
        if ( LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              i_micro_kernel_config->vmove_instruction_in,
              i_gp_reg_dropmask,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (im * 16 + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
              'y',
              vreg_tmp2, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );

          libxsmm_generator_cvtbf16ps_avx512( io_generated_code, 'z', vreg_tmp2, vreg_tmp2 );
        } else {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              i_micro_kernel_config->vmove_instruction_in,
              i_gp_reg_dropmask,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (im * 16 + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
              i_micro_kernel_config->vector_name,
              vreg_tmp2, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );
        }
        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCMPPS, 'z',
                                                       vreg_tmp2, vreg_zero, cur_mask_reg, 0x04  );
      }

      /* weight and zero input */
      libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code, LIBXSMM_X86_INSTR_VMULPS, 'z',
                                                     cur_vreg, prob_vreg, cur_vreg, cur_mask_reg, 1 );

      /* Store result  */
      if ( LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
        if ( io_generated_code->arch >= LIBXSMM_X86_AVX512_CPX ) {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16,
                                                    i_micro_kernel_config->vector_name,
                                                    cur_vreg, cur_vreg );
        } else {
          libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                                                               cur_vreg, cur_vreg,
                                                               vreg_tmp0, vreg_tmp1, reserved_mask_regs-1, reserved_mask_regs-2 );
        }
        libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_out,
          i_gp_reg_out,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * 16 + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
          'y',
          cur_vreg, (im == (m_trips-1)) ? use_m_masking : 0, 0, 1 );
      } else {
        libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_out,
          i_gp_reg_out,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * 16 + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
          i_micro_kernel_config->vector_name,
          cur_vreg, (im == (m_trips-1)) ? use_m_masking : 0, 0, 1 );
      }
    }
  }

  if (n_trips > 1) {
    /* Adjust input and output pointer */
    libxsmm_x86_instruction_alu_imm(  io_generated_code,
        i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_in,
        i_mateltwise_desc->ldi * n_unroll_factor * i_micro_kernel_config->datatype_size_in);

    libxsmm_x86_instruction_alu_imm(  io_generated_code,
        i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_out,
        i_mateltwise_desc->ldo * n_unroll_factor * i_micro_kernel_config->datatype_size_out);

    /* Adjust also mask ptr, datatype for mask tensor might be "bit" and also it has always the same shape as input  */
    if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_DROPOUT_BITMASK) > 0 ) {
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_dropmask,
          (i_mateltwise_desc->ldi * n_unroll_factor)/8);
    } else {
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_dropmask,
          i_mateltwise_desc->ldi * n_unroll_factor * i_micro_kernel_config->datatype_size_in);
    }

    /* close n loop */
    libxsmm_generator_mateltwise_footer_n_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_n_loop, n_trips );
  }

  /* In this case we have to use CPX replacement sequence for downconverts... */
  if ( (io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) &&
       (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) ) {
    libxsmm_generator_vcvtneps2bf16_avx512_clean_stack( io_generated_code, i_gp_reg_tmp );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_dropout_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
                                                   libxsmm_loop_label_tracker*                    io_loop_label_tracker,
                                                   libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
                                                   const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
                                                   const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {
  unsigned int l_gp_reg_in  = LIBXSMM_X86_GP_REG_R8;
  unsigned int l_gp_reg_out = LIBXSMM_X86_GP_REG_R9;
  unsigned int l_gp_reg_mloop = LIBXSMM_X86_GP_REG_RAX;
  unsigned int l_gp_reg_nloop = LIBXSMM_X86_GP_REG_RDX;
  unsigned int l_gp_reg_dropmask = LIBXSMM_X86_GP_REG_R10;
  unsigned int l_gp_reg_prob = LIBXSMM_X86_GP_REG_R11;
  unsigned int l_gp_reg_rngstate = LIBXSMM_X86_GP_REG_R12;
  unsigned int l_gp_reg_tmp = LIBXSMM_X86_GP_REG_R13;
  unsigned int l_mask_reg_0 = 1;

  /* load pointers from struct */
  libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_param_struct,
                                   LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                   l_gp_reg_in, 0 );

  libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_param_struct,
                                   LIBXSMM_X86_GP_REG_UNDEF, 0, 8,
                                   l_gp_reg_out, 0 );

  libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_param_struct,
                                   LIBXSMM_X86_GP_REG_UNDEF, 0, 16,
                                   l_gp_reg_dropmask, 0 );

  libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_param_struct,
                                   LIBXSMM_X86_GP_REG_UNDEF, 0, 24,
                                   l_gp_reg_prob, 0 );

  libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_param_struct,
                                   LIBXSMM_X86_GP_REG_UNDEF, 0, 32,
                                   l_gp_reg_rngstate, 0 );

  /* check leading dimnesions and sizes */
  if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_DROPOUT_FWD) > 0) ||
       ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_DROPOUT_BWD) > 0)    ) {
    if ( i_mateltwise_desc->m > i_mateltwise_desc->ldi ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
      return;
    }
    if ( i_mateltwise_desc->m > i_mateltwise_desc->ldo ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB );
      return;
    }
    if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_DROPOUT_BITMASK) > 0) &&
         ((i_mateltwise_desc->ldi % 16 != 0) || (i_mateltwise_desc->ldo % 16 != 0)) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDC );
      return;
    }
  } else {
    /* should not happen */
  }

  if ( ( LIBXSMM_GEMM_PRECISION_F32  == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ||
         LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )    )
       &&
       ( LIBXSMM_GEMM_PRECISION_F32  == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ||
         LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )    ) ) {
    if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_DROPOUT_FWD) > 0 ) {
      libxsmm_generator_dropout_fwd_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                        l_gp_reg_in, l_gp_reg_out, l_gp_reg_dropmask, l_gp_reg_rngstate,
                                                        l_gp_reg_prob, l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_tmp,
                                                        l_mask_reg_0, i_micro_kernel_config, i_mateltwise_desc );
    } else if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_DROPOUT_BWD) > 0 ) {
      libxsmm_generator_dropout_bwd_avx512_microkernel( io_generated_code, io_loop_label_tracker,
                                                        l_gp_reg_in, l_gp_reg_out, l_gp_reg_dropmask,
                                                        l_gp_reg_prob, l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_tmp,
                                                        l_mask_reg_0, i_micro_kernel_config, i_mateltwise_desc );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }
}

