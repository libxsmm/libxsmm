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

#include "generator_mateltwise_relu_avx_avx512.h"
#include "generator_mateltwise_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
void libxsmm_generator_relu_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int in, im, m, n, use_m_masking, m_trips, n_unroll_factor, n_trips, mask_out_count = 0, unroll_iter = 0;
  unsigned int reserved_mask_regs = 1, n_available_zmms = 30, n_available_mask_regs = 7, max_nm_unrolling = 16;
  unsigned int zero_vreg = 31, tmp_vreg = 30, cur_vreg = 0, cur_mask_reg = 0;
  unsigned int gpr_mask_regs[8] = {LIBXSMM_X86_GP_REG_R8, LIBXSMM_X86_GP_REG_R9, LIBXSMM_X86_GP_REG_R10, LIBXSMM_X86_GP_REG_R11, LIBXSMM_X86_GP_REG_R12, LIBXSMM_X86_GP_REG_R13, LIBXSMM_X86_GP_REG_R14, LIBXSMM_X86_GP_REG_R15};
  unsigned int aggregate_mask_loads = 1;
  unsigned int l_is_pure_bf16 = ( (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) && (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) ) ? 1 : 0;
  unsigned int l_vlen = ( l_is_pure_bf16 != 0 ) ? 32 : 16;
  unsigned int l_vcmp_instr = ( l_is_pure_bf16 != 0 ) ? LIBXSMM_X86_INSTR_VPCMPW : LIBXSMM_X86_INSTR_VCMPPS;
  unsigned int l_vblend_instr = ( l_is_pure_bf16 != 0 ) ? LIBXSMM_X86_INSTR_VPBLENDMW : LIBXSMM_X86_INSTR_VPBLENDMD;
  unsigned int l_mask_ld_instr = ( l_is_pure_bf16 != 0 ) ? LIBXSMM_X86_INSTR_KMOVD_LD : LIBXSMM_X86_INSTR_KMOVW_LD;
  unsigned int l_mask_st_instr = ( l_is_pure_bf16 != 0 ) ? LIBXSMM_X86_INSTR_KMOVD_ST : LIBXSMM_X86_INSTR_KMOVW_ST;

  /* Determine what relu (fwd/bwd) to perform */
  if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_RELU_FWD) == 0) &&
       ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_RELU_BWD) == 0) ) {
    /* This should not happen  */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* Some rudimentary checking of M, N and LDs*/
  if ( i_mateltwise_desc->m > i_mateltwise_desc->ldi ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
    return;
  }

  /* check datatype */
  if ( ( LIBXSMM_GEMM_PRECISION_F32  == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ||
         LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )    )
       &&
       ( LIBXSMM_GEMM_PRECISION_F32  == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ||
         LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )    ) ) {
    /* fine */
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in     = LIBXSMM_X86_GP_REG_RAX;
  i_gp_reg_mapping->gp_reg_out    = LIBXSMM_X86_GP_REG_RBX;
  i_gp_reg_mapping->gp_reg_m_loop = LIBXSMM_X86_GP_REG_RCX;
  i_gp_reg_mapping->gp_reg_n_loop = LIBXSMM_X86_GP_REG_RSI;
  i_gp_reg_mapping->gp_reg_relumask = LIBXSMM_X86_GP_REG_RDX;

  /* Set zero register needed for relu  */
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                           LIBXSMM_X86_INSTR_VPXORD,
                                           i_micro_kernel_config->vector_name,
                                           zero_vreg, zero_vreg, zero_vreg );

  /* We fully unroll in M dimension, calculate mask if there is remainder */
  m                 = i_mateltwise_desc->m;
  n                 = i_mateltwise_desc->n;
  use_m_masking     = ( m % l_vlen == 0 ) ? 0 : 1;
  m_trips           = (m + (l_vlen-1)) / l_vlen;

  if (use_m_masking == 1) {
    /* Calculate mask reg 1 for reading/output-writing */
    mask_out_count = l_vlen - (m % l_vlen);
    libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_R12, 1, mask_out_count, LIBXSMM_GEMM_PRECISION_BF16);
    reserved_mask_regs++;
    n_available_mask_regs--;
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
      16,
      i_gp_reg_mapping->gp_reg_out,
      0 );

  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      8,
      i_gp_reg_mapping->gp_reg_relumask,
      0 );

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

  if (((n_unroll_factor*m_trips)%2 != 0) || (i_mateltwise_desc->ldo != m)) {
    aggregate_mask_loads = 0;
  }

  /* FWD and non MASK variants for sure disabled aggreaged mask load */
  if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_RELU_FWD) > 0)      ||
       ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_RELU_BITMASK) == 0) ||
       (LIBXSMM_GEMM_PRECISION_F32  == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) ||
       (LIBXSMM_GEMM_PRECISION_F32  == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) ) {
    aggregate_mask_loads = 0;
  }

  if (n_trips > 1) {
    /* open n loop */
    libxsmm_generator_mateltwise_header_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop );
  }

  if (aggregate_mask_loads > 0) {
    /* First load all the masks in the auxiliary gprs  */
    for (unroll_iter = 0; unroll_iter < m_trips * n_unroll_factor; unroll_iter += 2) {
      in = unroll_iter/m_trips;
      im = unroll_iter%m_trips;

      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_relumask,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * l_vlen + in * i_mateltwise_desc->ldo)/8,
          gpr_mask_regs[unroll_iter/2],
          0 );
    }

    /* Now read the input, apply the relu and store the mask */
    for (in = 0; in < n_unroll_factor; in++) {
      for (im = 0; im < m_trips; im++) {
        unroll_iter = in * m_trips + im;
        cur_vreg = unroll_iter % n_available_zmms;
        cur_mask_reg = reserved_mask_regs + unroll_iter % n_available_mask_regs;

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * l_vlen + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
            i_micro_kernel_config->vector_name,
            cur_vreg, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );

        /* Extract the mask from the proper auxiliry gpr */
        libxsmm_x86_instruction_mask_move( io_generated_code,
            LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
            gpr_mask_regs[unroll_iter/2],
            cur_mask_reg );

        /* Shift current mask gpr in order to br able to extract the next mask  */
        if (unroll_iter % 2 == 0) {
          libxsmm_x86_instruction_alu_imm( io_generated_code,
              LIBXSMM_X86_INSTR_SHRQ,
              gpr_mask_regs[unroll_iter/2],
              l_vlen );
        }

        /* Blend output result with zero reg based on relu mask */
        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
            LIBXSMM_X86_INSTR_VPBLENDMW,
            i_micro_kernel_config->vector_name,
            cur_vreg,
            zero_vreg,
            cur_vreg,
            cur_mask_reg,
            0 );

        /* Store result  */
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_out,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * l_vlen + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
            i_micro_kernel_config->vector_name,
            cur_vreg, (im == (m_trips-1)) ? use_m_masking : 0, 0, 1 );
      }
    }
  } else {
    for (in = 0; in < n_unroll_factor; in++) {
      for (im = 0; im < m_trips; im++) {
        unroll_iter = in * m_trips + im;
        cur_vreg = unroll_iter % n_available_zmms;
        cur_mask_reg = reserved_mask_regs + unroll_iter % n_available_mask_regs;

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * l_vlen + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
            i_micro_kernel_config->vector_name,
            cur_vreg, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );

        if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_RELU_FWD) > 0 ) {
          /* Compare to generate mask  */
          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
              l_vcmp_instr,
              i_micro_kernel_config->vector_name,
              zero_vreg,
              cur_vreg,
              cur_mask_reg,
              6 );

          /* Store mask relu  */
          if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_RELU_BITMASK) > 0 ) {
            libxsmm_x86_instruction_mask_move_mem( io_generated_code,
                l_mask_st_instr,
                i_gp_reg_mapping->gp_reg_relumask,
                LIBXSMM_X86_GP_REG_UNDEF,
                0,
                (im * l_vlen + in * i_mateltwise_desc->ldo)/8,
                cur_mask_reg );
          }

          /* Blend output result with zero reg based on relu mask */
          libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
              l_vblend_instr,
              i_micro_kernel_config->vector_name,
              cur_vreg,
              zero_vreg,
              cur_vreg,
              cur_mask_reg,
              0 );
        } else if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_RELU_BWD) > 0 ) {
          /* Load relu mask */
          if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_RELU_BITMASK) > 0 ) {
            libxsmm_x86_instruction_mask_move_mem( io_generated_code,
                l_mask_ld_instr,
                i_gp_reg_mapping->gp_reg_relumask,
                LIBXSMM_X86_GP_REG_UNDEF,
                0,
                (im * l_vlen + in * i_mateltwise_desc->ldi)/8,
                cur_mask_reg );
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_relumask,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * l_vlen + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
            i_micro_kernel_config->vector_name,
            tmp_vreg, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );

            /* Compare to generate mask  */
            libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
              l_vcmp_instr,
              i_micro_kernel_config->vector_name,
              zero_vreg,
              tmp_vreg,
              cur_mask_reg,
              6 );
          }

          /* Blend output result with zero reg based on relu mask */
          libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
              l_vblend_instr,
              i_micro_kernel_config->vector_name,
              cur_vreg,
              zero_vreg,
              cur_vreg,
              cur_mask_reg,
              0 );
        } else {
          /* shouldn't happen */
        }

        /* Store result  */
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_out,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * l_vlen + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
            i_micro_kernel_config->vector_name,
            cur_vreg, (im == (m_trips-1)) ? use_m_masking : 0, 0, 1 );
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

    /* Adjust also relu ptr, datatype for relumask tensor is "bit" and also it has always the same shape as output  */
    if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_RELU_BITMASK) > 0 ) {
      if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_RELU_BWD) > 0 ) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
            i_micro_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_relumask,
            (i_mateltwise_desc->ldi * n_unroll_factor)/8);
      } else {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
            i_micro_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_relumask,
            (i_mateltwise_desc->ldo * n_unroll_factor)/8);
      }
    } else if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_RELU_BITMASK) == 0) &&
                ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_RELU_BWD) > 0) ) {
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_relumask,
          i_mateltwise_desc->ldi * n_unroll_factor * i_micro_kernel_config->datatype_size_in);
    }

    /* close n loop */
    libxsmm_generator_mateltwise_footer_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, n_trips);
  }
}

