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
#include "generator_mateltwise_sse_avx_avx512.h"
#include "generator_mateltwise_scale_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
void libxsmm_generator_scale_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {

  unsigned int in, im, m, n, m_full_trips, m_trips, use_m_masking, mask_count, reg_n, reg_m;
  unsigned int scale_rows = 0, scale_cols = 0, scale_rows_cols = 0, perform_scale = 0, perform_shift = 0, perform_addbias = 0;
  unsigned int reg_shift = 31, reg_bias = 30, reg_scale = 29;
  unsigned int reg_shift2 = 28, reg_bias2 = 27, reg_scale2 = 26;
  unsigned int n_available_zmms = 29;
  unsigned int scale_rows_bcastval_accumulate = 0;

  /* Some rudimentary checking of M, N and LDs*/
  if ( i_mateltwise_desc->m > i_mateltwise_desc->ldi ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
    return;
  }

  /* Determine what operations to perform */
  scale_rows_cols = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_SCALE_ROWS_COLS) > 0) ? 1 : 0;
  scale_rows    = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_SCALE_ROWS) > 0) ? 1 : 0;
  scale_cols    = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_SCALE_COLS) > 0) ? 1 : 0;
  perform_scale = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_SCALE_MULT) > 0) ? 1 : 0;
  perform_shift = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_SCALE_SHIFT) > 0) ? 1 : 0;
  perform_addbias = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_SCALE_ADD_BIAS) > 0) ? 1 : 0;
  scale_rows_bcastval_accumulate = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_SCALE_ROWS_BCASTVAL_ACCUMULATE) > 0) ? 1 : 0;

  if (((scale_rows > 0) && (scale_cols > 0)) ||
      ((scale_cols > 0) && (scale_rows_cols > 0)) ||
      ((scale_rows > 0) && (scale_rows_cols > 0)) ||
      ((scale_rows == 0) && (scale_cols == 0) && (scale_rows_cols == 0) && (scale_rows_bcastval_accumulate == 0))) {
    /* This should not happen  */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* When we request to JIT for rows-cols order, then honor (for code gen purposes) the scale_rows flag */
  if (scale_rows_cols > 0) {
    scale_rows = 1;
    n_available_zmms = 26;
  }

  if (scale_rows_bcastval_accumulate > 0) {
    n_available_zmms = 30;
    reg_scale = 31;
  }

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in                     = LIBXSMM_X86_GP_REG_R8;
  i_gp_reg_mapping->gp_reg_out                    = LIBXSMM_X86_GP_REG_R9;
  i_gp_reg_mapping->gp_reg_shift_vals             = LIBXSMM_X86_GP_REG_R10;
  i_gp_reg_mapping->gp_reg_scale_vals             = LIBXSMM_X86_GP_REG_R11;
  i_gp_reg_mapping->gp_reg_bias_vals              = LIBXSMM_X86_GP_REG_R12;
  i_gp_reg_mapping->gp_reg_shift_vals2            = LIBXSMM_X86_GP_REG_R14;
  i_gp_reg_mapping->gp_reg_scale_vals2            = LIBXSMM_X86_GP_REG_RBX;
  i_gp_reg_mapping->gp_reg_bias_vals2             = LIBXSMM_X86_GP_REG_RCX;
  i_gp_reg_mapping->gp_reg_m_loop                 = LIBXSMM_X86_GP_REG_R13;
  i_gp_reg_mapping->gp_reg_n_loop                 = LIBXSMM_X86_GP_REG_RAX;

  /* We fully unroll in N dimension, calculate m-mask if there is remainder */
  m                 = i_mateltwise_desc->m;
  n                 = i_mateltwise_desc->n;
  use_m_masking     = ( m % 16 == 0 ) ? 0 : 1;

  /* Calculate input mask in case we see m_masking */
  if (use_m_masking == 1) {
    /* Calculate mask reg 1 for input-reading */
    mask_count =  16 - (m % 16);
    libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, 1, mask_count, LIBXSMM_GEMM_PRECISION_F32);
  }

  /* load the input pointer(s) and output pointer */
  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      0,
      i_gp_reg_mapping->gp_reg_in,
      0 );

  if ( perform_shift > 0 ) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
       i_micro_kernel_config->alu_mov_instruction,
       i_gp_reg_mapping->gp_reg_param_struct,
       LIBXSMM_X86_GP_REG_UNDEF, 0,
       8,
       i_gp_reg_mapping->gp_reg_shift_vals,
       0 );
  }

  if ( (perform_scale > 0) || (scale_rows_bcastval_accumulate > 0) ) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
       i_micro_kernel_config->alu_mov_instruction,
       i_gp_reg_mapping->gp_reg_param_struct,
       LIBXSMM_X86_GP_REG_UNDEF, 0,
       16,
       i_gp_reg_mapping->gp_reg_scale_vals,
       0 );
  }

  if ( perform_addbias > 0 ) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
       i_micro_kernel_config->alu_mov_instruction,
       i_gp_reg_mapping->gp_reg_param_struct,
       LIBXSMM_X86_GP_REG_UNDEF, 0,
       24,
       i_gp_reg_mapping->gp_reg_bias_vals,
       0 );
  }

  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      32,
      i_gp_reg_mapping->gp_reg_out,
      0 );

  if (scale_rows_cols > 0) {
    if ( perform_shift > 0 ) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
         i_micro_kernel_config->alu_mov_instruction,
         i_gp_reg_mapping->gp_reg_param_struct,
         LIBXSMM_X86_GP_REG_UNDEF, 0,
         40,
         i_gp_reg_mapping->gp_reg_shift_vals2,
         0 );
    }

    if ( perform_scale > 0 ) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
         i_micro_kernel_config->alu_mov_instruction,
         i_gp_reg_mapping->gp_reg_param_struct,
         LIBXSMM_X86_GP_REG_UNDEF, 0,
         48,
         i_gp_reg_mapping->gp_reg_scale_vals2,
         0 );
    }

    if ( perform_addbias > 0 ) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
         i_micro_kernel_config->alu_mov_instruction,
         i_gp_reg_mapping->gp_reg_param_struct,
         LIBXSMM_X86_GP_REG_UNDEF, 0,
         56,
         i_gp_reg_mapping->gp_reg_bias_vals2,
         0 );
    }
  }

  /* If scaling cols: follow an MN loop order with fully unrolled N loop */
  if (scale_cols == 1) {
    m_full_trips = m / 16;

    if ( m_full_trips >= 1 ) {
      if (m_full_trips > 1) {
        /* open m loop */
        libxsmm_generator_mateltwise_header_m_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_m_loop );
      }

      /* Load the correspodning columns to be used for scaling */
      if ( perform_shift > 0 ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_shift_vals,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            i_micro_kernel_config->vector_name,
            reg_shift, 0, 1, 0 );
      }

      if ( perform_scale > 0 ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_scale_vals,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            i_micro_kernel_config->vector_name,
            reg_scale, 0, 1, 0 );
      }


      if ( perform_addbias > 0 ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_bias_vals,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            i_micro_kernel_config->vector_name,
            reg_bias, 0, 1, 0 );
      }

      for (in = 0; in < n; in++) {
        reg_n = in % n_available_zmms;

        /* Load part of the column  */
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            in * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
            i_micro_kernel_config->vector_name,
            reg_n, 0, 1, 0 );

        /* Perform transformations */
        if ( perform_shift > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               LIBXSMM_X86_INSTR_VADDPS,
                                               i_micro_kernel_config->vector_name,
                                               reg_n, reg_shift, reg_n );
        }

        if ( perform_scale > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               LIBXSMM_X86_INSTR_VMULPS,
                                               i_micro_kernel_config->vector_name,
                                               reg_n, reg_scale, reg_n );
        }

        if ( perform_addbias> 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               LIBXSMM_X86_INSTR_VADDPS,
                                               i_micro_kernel_config->vector_name,
                                               reg_n, reg_bias, reg_n );
        }

        /* Store part of the column */
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_micro_kernel_config->instruction_set,
                                          i_micro_kernel_config->vmove_instruction_out,
                                          i_gp_reg_mapping->gp_reg_out,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          in * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                          i_micro_kernel_config->vector_name,
                                          reg_n, 0, 0, 1 );
      }

      if ((m_full_trips > 1) || (use_m_masking == 1)) {
        /* Adjust input and output pointer */
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
            i_micro_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_in,
            16 * i_micro_kernel_config->datatype_size_in);

        libxsmm_x86_instruction_alu_imm(  io_generated_code,
            i_micro_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_out,
            16 * i_micro_kernel_config->datatype_size_out);

        if ( perform_shift > 0 ) {
          libxsmm_x86_instruction_alu_imm(  io_generated_code,
              i_micro_kernel_config->alu_add_instruction,
              i_gp_reg_mapping->gp_reg_shift_vals,
              16 * i_micro_kernel_config->datatype_size_in);
        }

        if ( perform_scale > 0 ) {
          libxsmm_x86_instruction_alu_imm(  io_generated_code,
              i_micro_kernel_config->alu_add_instruction,
              i_gp_reg_mapping->gp_reg_scale_vals,
              16 * i_micro_kernel_config->datatype_size_in);
        }

        if ( perform_addbias > 0 ) {
          libxsmm_x86_instruction_alu_imm(  io_generated_code,
              i_micro_kernel_config->alu_add_instruction,
              i_gp_reg_mapping->gp_reg_bias_vals,
              16 * i_micro_kernel_config->datatype_size_in);
        }
      }

      if (m_full_trips > 1) {
        /* close m loop */
        libxsmm_generator_mateltwise_footer_m_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_m_loop, m_full_trips);
      }
    }

    if (use_m_masking == 1) {
      /* Load the correspodning columns to be used for scaling */
      if ( perform_shift > 0 ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_shift_vals,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            i_micro_kernel_config->vector_name,
            reg_shift, 0, 1, 0 );
      }

      if ( perform_scale > 0 ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_scale_vals,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            i_micro_kernel_config->vector_name,
            reg_scale, 0, 1, 0 );
      }


      if ( perform_addbias > 0 ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_bias_vals,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            i_micro_kernel_config->vector_name,
            reg_bias, 0, 1, 0 );
      }

      for (in = 0; in < n; in++) {
        reg_n = in % n_available_zmms;

        /* Load part of the column  */
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            in * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
            i_micro_kernel_config->vector_name,
            reg_n, use_m_masking, 1, 0 );

        /* Perform transformations */
        if ( perform_shift > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               LIBXSMM_X86_INSTR_VADDPS,
                                               i_micro_kernel_config->vector_name,
                                               reg_n, reg_shift, reg_n );
        }

        if ( perform_scale > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               LIBXSMM_X86_INSTR_VMULPS,
                                               i_micro_kernel_config->vector_name,
                                               reg_n, reg_scale, reg_n );
        }

        if ( perform_addbias> 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               LIBXSMM_X86_INSTR_VADDPS,
                                               i_micro_kernel_config->vector_name,
                                               reg_n, reg_bias, reg_n );
        }

        /* Store part of the column */
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_micro_kernel_config->instruction_set,
                                          i_micro_kernel_config->vmove_instruction_out,
                                          i_gp_reg_mapping->gp_reg_out,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          in * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                          i_micro_kernel_config->vector_name,
                                          reg_n, use_m_masking, 0, 1 );
      }
    }
  }

  /* If scaling rows: follow an NM loop order with fully unrolled M loop */
  if ((scale_rows == 1) || (scale_rows_bcastval_accumulate == 1)) {
    m_trips = (m + 15) / 16;
    if (n > 1) {
      /* open n loop */
      libxsmm_generator_mateltwise_header_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop );
    }

    /* Load the correspodning columns to be used for scaling */
    if ( perform_shift > 0 ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VBROADCASTSS,
          i_gp_reg_mapping->gp_reg_shift_vals,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          0,
          i_micro_kernel_config->vector_name,
          reg_shift, 0, 1, 0 );
    }

    if ( (perform_scale > 0) || (scale_rows_bcastval_accumulate == 1) ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VBROADCASTSS,
          i_gp_reg_mapping->gp_reg_scale_vals,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          0,
          i_micro_kernel_config->vector_name,
          reg_scale, 0, 1, 0 );
    }

    if ( perform_addbias > 0 ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VBROADCASTSS,
          i_gp_reg_mapping->gp_reg_bias_vals,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          0,
          i_micro_kernel_config->vector_name,
          reg_bias, 0, 1, 0 );
    }

    for (im = 0; im < m_trips; im++) {
      if (scale_rows_bcastval_accumulate == 1) {
        reg_m = (im*2) % n_available_zmms;

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * 16 * i_micro_kernel_config->datatype_size_in,
            i_micro_kernel_config->vector_name,
            reg_m, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_out,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * 16 * i_micro_kernel_config->datatype_size_out,
            i_micro_kernel_config->vector_name,
            reg_m+1, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );

        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             LIBXSMM_X86_INSTR_VFMADD231PS,
                                             i_micro_kernel_config->vector_name,
                                             reg_m, reg_scale, reg_m+1 );

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_out,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * 16 * i_micro_kernel_config->datatype_size_out,
            i_micro_kernel_config->vector_name,
            reg_m+1, (im == (m_trips-1)) ? use_m_masking : 0, 0, 1 );
      } else {
        reg_m = im % n_available_zmms;

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * 16 * i_micro_kernel_config->datatype_size_in,
            i_micro_kernel_config->vector_name,
            reg_m, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );

        if (scale_rows_cols > 0) {
          /* Load the correspodning columns to be used for scaling */
          if ( perform_shift > 0 ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                i_micro_kernel_config->vmove_instruction_in,
                i_gp_reg_mapping->gp_reg_shift_vals2,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                im * 16 * i_micro_kernel_config->datatype_size_in,
                i_micro_kernel_config->vector_name,
                reg_shift2, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );
          }

          if ( perform_scale > 0 ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                i_micro_kernel_config->vmove_instruction_in,
                i_gp_reg_mapping->gp_reg_scale_vals2,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                im * 16 * i_micro_kernel_config->datatype_size_in,
                i_micro_kernel_config->vector_name,
                reg_scale2, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );
          }

          if ( perform_addbias > 0 ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                i_micro_kernel_config->vmove_instruction_in,
                i_gp_reg_mapping->gp_reg_bias_vals2,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                im * 16 * i_micro_kernel_config->datatype_size_in,
                i_micro_kernel_config->vector_name,
                reg_bias2, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );
          }
        }

        /* Perform transformations on the rows*/
        if ((perform_scale > 0) && (perform_addbias > 0) && (perform_shift == 0)) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             LIBXSMM_X86_INSTR_VFMADD213PS,
                                             i_micro_kernel_config->vector_name,
                                             reg_bias, reg_scale, reg_m );
        } else {
          if ( perform_shift > 0 ) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VADDPS,
                                                 i_micro_kernel_config->vector_name,
                                                 reg_m, reg_shift, reg_m );
          }

          if ( perform_scale > 0 ) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VMULPS,
                                                 i_micro_kernel_config->vector_name,
                                                 reg_m, reg_scale, reg_m );
          }

          if ( perform_addbias > 0 ) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VADDPS,
                                                 i_micro_kernel_config->vector_name,
                                                 reg_m, reg_bias, reg_m );
          }
        }

        if (scale_rows_cols > 0) {
          /* Perform transformations on the columns */
          if ((perform_scale > 0) && (perform_addbias > 0) && (perform_shift == 0)) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               LIBXSMM_X86_INSTR_VFMADD213PS,
                                               i_micro_kernel_config->vector_name,
                                               reg_bias2, reg_scale2, reg_m );
          } else {
            if ( perform_shift > 0 ) {
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                   LIBXSMM_X86_INSTR_VADDPS,
                                                   i_micro_kernel_config->vector_name,
                                                   reg_m, reg_shift2, reg_m );
            }

            if ( perform_scale > 0 ) {
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                   LIBXSMM_X86_INSTR_VMULPS,
                                                   i_micro_kernel_config->vector_name,
                                                   reg_m, reg_scale2, reg_m );
            }

            if ( perform_addbias > 0 ) {
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                   LIBXSMM_X86_INSTR_VADDPS,
                                                   i_micro_kernel_config->vector_name,
                                                   reg_m, reg_bias2, reg_m );
            }
          }
        }

        /* Store the result  */
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_out,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * 16 * i_micro_kernel_config->datatype_size_out,
            i_micro_kernel_config->vector_name,
            reg_m, (im == (m_trips-1)) ? use_m_masking : 0, 0, 1 );
      }
    }

    if (n > 1) {
      /* Adjust input and output pointer */
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_in,
          i_mateltwise_desc->ldi *  i_micro_kernel_config->datatype_size_in);

      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_out,
          i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out);

      if ( perform_shift > 0 ) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
            i_micro_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_shift_vals,
            i_micro_kernel_config->datatype_size_in);
      }

      if ( (perform_scale > 0) || (scale_rows_bcastval_accumulate == 1) ) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
            i_micro_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_scale_vals,
            i_micro_kernel_config->datatype_size_in);
      }

      if ( perform_addbias > 0 ) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
            i_micro_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_bias_vals,
            i_micro_kernel_config->datatype_size_in);
      }

      /* close n loop */
      libxsmm_generator_mateltwise_footer_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, n);
    }

  }
}

