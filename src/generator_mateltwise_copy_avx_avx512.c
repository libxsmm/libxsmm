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
#include "generator_mateltwise_copy_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
void libxsmm_generator_copy_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {
  unsigned int m, n, im, in, in_load, unroll_iter, use_m_masking = 0, m_trips = 0, n_trips = 0, vlen_in = 32, vlen_out = 32, mask_in_count = 0, mask_out_count = 0;
  unsigned int reserved_zmms = 0, max_nm_unrolling = 32, n_unroll_factor = 1;
  unsigned int mask_reg_in = 1, mask_reg_out = 2;
  unsigned int zero_vreg = 0, vreg_in = 0;
  unsigned int read_input = 1;
  unsigned int colbcast = 0;
  unsigned int hoist_colbcast = 0;
  char     input_vname = 'z', output_vname = 'z';

  /* Some rudimentary checking of M, N and LDs*/
  if ( ((i_mateltwise_desc->m > i_mateltwise_desc->ldi) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_COPY_ZERO) == 0) ) || (i_mateltwise_desc->m > i_mateltwise_desc->ldo) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
    return;
  }

  /* Specify vlens depending on precision  */
  if ( (LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP(i_mateltwise_desc->datatype)) || (LIBXSMM_DATATYPE_I64 == LIBXSMM_GETENUM_INP(i_mateltwise_desc->datatype))) {
    vlen_in = 8;
  } else if ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP(i_mateltwise_desc->datatype)) || (LIBXSMM_DATATYPE_I32 == LIBXSMM_GETENUM_INP(i_mateltwise_desc->datatype))) {
    vlen_in = 16;
  } else if ( (LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_INP(i_mateltwise_desc->datatype)) || (LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP(i_mateltwise_desc->datatype)) || (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP(i_mateltwise_desc->datatype))) {
    vlen_in = 32;
  } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP(i_mateltwise_desc->datatype)) {
    vlen_in = 64;
  }

  if ( (LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype)) || (LIBXSMM_DATATYPE_I64 == LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype))) {
    vlen_out = 8;
  } else if ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype)) || (LIBXSMM_DATATYPE_I32 == LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype))) {
    vlen_out = 16;
  } else if ( (LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype)) || (LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype)) || (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype))) {
    vlen_out = 32;
  } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype)) {
    vlen_out = 64;
  }

  if (vlen_in != vlen_out) {
    if ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype)) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP(i_mateltwise_desc->datatype)) ) {
      vlen_in = 16;
      input_vname = 'y';
    } else {
      /* This should not happen  */
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  }

  /* Check if we have to hoist column broadcast */
  if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_COPY_COLBCAST) > 0) {
    colbcast = 1;
  }

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in     = LIBXSMM_X86_GP_REG_R8;
  i_gp_reg_mapping->gp_reg_out    = LIBXSMM_X86_GP_REG_R9;
  i_gp_reg_mapping->gp_reg_n_loop = LIBXSMM_X86_GP_REG_R10;

  /* Load the input pointer and output pointer */
  /* Load input only if we dont have zerobeta  */
  if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_COPY_ZERO) > 0) {
    zero_vreg = reserved_zmms++;
    reserved_zmms++;
    read_input = 0;
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             LIBXSMM_X86_INSTR_VPXORD,
                                             i_micro_kernel_config->vector_name,
                                             zero_vreg, zero_vreg, zero_vreg );
  } else {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      0,
      i_gp_reg_mapping->gp_reg_in,
      0 );
  }

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

  /* Calculate input and output masks in case we see m_masking */
  if (use_m_masking == 1) {
    /* Calculate mask reg 1 for input-reading */
    if (read_input > 0) {
      mask_in_count = vlen_in - m % vlen_in;
      libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_R11, mask_reg_in, mask_in_count, LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype));
    }
    /* Calculate mask reg 2 for output-writing */
    if (((read_input > 0) && (vlen_in != vlen_out)) || (read_input == 0)){
      mask_out_count = vlen_out - m % vlen_out;
      libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_R11, mask_reg_out, mask_out_count, LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype));
    } else {
      mask_reg_out = mask_reg_in;
    }
  }

  /* Check if we can hoist column brodcast to avoid redundant loads  */
  if ((colbcast == 1) && (m_trips <= max_nm_unrolling) && (n_trips >= 1)) {
    hoist_colbcast = 1;
  }

  if (hoist_colbcast == 1) {
    for (im = 0; im < m_trips; im++) {
      unroll_iter = im;
      vreg_in = unroll_iter % (max_nm_unrolling - reserved_zmms) + reserved_zmms;
      if (read_input == 0) {
        vreg_in = zero_vreg;
      } else {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * vlen_in * i_micro_kernel_config->datatype_size_in,
            input_vname,
            vreg_in, ((im == (m_trips-1)) && (use_m_masking == 1)) ? mask_reg_in : 0, 1, 0 );
      }

      if ((read_input > 0) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype)) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP(i_mateltwise_desc->datatype)) ) {
        /* convert 16 bit values into 32 bit (integer convert) */
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
            LIBXSMM_X86_INSTR_VPMOVSXWD,
            i_micro_kernel_config->vector_name,
            vreg_in, vreg_in );

        /* shift 16 bits to the left to generate valid FP32 numbers */
        libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code,
            LIBXSMM_X86_INSTR_VPSLLD_I,
            i_micro_kernel_config->vector_name,
            vreg_in,
            vreg_in,
            16);
      }
    }
  }

  if (n_trips > 1) {
    /* open n loop */
    libxsmm_generator_mateltwise_header_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop );
  }

  for (in = 0; in < n_unroll_factor; in++) {
    for (im = 0; im < m_trips; im++) {
      in_load = (colbcast == 0) ? in : 0;
      unroll_iter = in_load * m_trips + im;
      vreg_in = unroll_iter % (max_nm_unrolling - reserved_zmms) + reserved_zmms;
      if (read_input == 0) {
        vreg_in = zero_vreg;
      } else {
        if (hoist_colbcast == 0) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              i_micro_kernel_config->vmove_instruction_in,
              i_gp_reg_mapping->gp_reg_in,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (im * vlen_in + in_load * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
              input_vname,
              vreg_in, ((im == (m_trips-1)) && (use_m_masking == 1)) ? mask_reg_in : 0, 1, 0 );
        }
      }

      if ((read_input > 0) && (hoist_colbcast == 0) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype)) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP(i_mateltwise_desc->datatype)) ) {
        /* convert 16 bit values into 32 bit (integer convert) */
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
            LIBXSMM_X86_INSTR_VPMOVSXWD,
            i_micro_kernel_config->vector_name,
            vreg_in, vreg_in );

        /* shift 16 bits to the left to generate valid FP32 numbers */
        libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code,
            LIBXSMM_X86_INSTR_VPSLLD_I,
            i_micro_kernel_config->vector_name,
            vreg_in,
            vreg_in,
            16);
      }

      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_out,
          i_gp_reg_mapping->gp_reg_out,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * vlen_out + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
          output_vname,
          vreg_in, ((im == (m_trips-1)) && (use_m_masking == 1)) ? mask_reg_out : 0, 0, 1 );

    }
  }

  if (n_trips > 1) {
    /* Adjust input and output pointer */
    if ((read_input > 0) && (colbcast == 0)) {
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_in,
          i_mateltwise_desc->ldi * n_unroll_factor * i_micro_kernel_config->datatype_size_in);
    }

    libxsmm_x86_instruction_alu_imm(  io_generated_code,
        i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_out,
        i_mateltwise_desc->ldo *  n_unroll_factor * i_micro_kernel_config->datatype_size_out);

    /* close n loop */
    libxsmm_generator_mateltwise_footer_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, n_trips);
  }
}



