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
#include "generator_common_x86.h"
#include "generator_mateltwise_sse_avx_avx512.h"
#include "generator_mateltwise_misc_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"

#if !defined(LIBXSMM_GENERATOR_MATELTWISE_MISC_AVX_AVX512_JUMP_LABEL_TRACKER_MALLOC)
# define LIBXSMM_GENERATOR_MATELTWISE_MISC_AVX_AVX512_JUMP_LABEL_TRACKER_MALLOC
#endif


LIBXSMM_API_INTERN
void libxsmm_generator_mn_code_block_replicate_col_var_avx_avx512( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    libxsmm_mateltwise_kernel_config*              i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc,
    unsigned int                                   vlen,
    unsigned int                                   m_trips_loop,
    unsigned int                                   m_unroll_factor,
    unsigned int                                   peeled_m_trips,
    unsigned int                                   i_use_masking,
    unsigned int                                   mask_in,
    unsigned int                                   mask_out) {
  unsigned int im;
  char vname_in   = (io_generated_code->arch < LIBXSMM_X86_AVX512) ? 'y' : 'z';
  char vname_out  = (io_generated_code->arch < LIBXSMM_X86_AVX512) ? 'y' : 'z';
  unsigned int upconvert_input_bf16f32 = 0;
  unsigned int upconvert_input_bf8f32 = 0;
  unsigned int upconvert_input_hf8f32 = 0;
  unsigned int upconvert_input_f16f32 = 0;
  unsigned int downconvert_input_f32bf16 = 0;
  unsigned int downconvert_input_f32bf8 = 0;
  unsigned int downconvert_input_f32hf8 = 0;
  unsigned int downconvert_input_f32f16 = 0;

  if (io_generated_code->arch < LIBXSMM_X86_AVX512) {
    if ((LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ))) {
      vname_in = 'x';
      upconvert_input_bf16f32 = 1;
    }
    if ((LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ))) {
      vname_in = 'x';
      upconvert_input_bf8f32 = 1;
    }
    if ((LIBXSMM_DATATYPE_HF8 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ))) {
      vname_in = 'x';
      upconvert_input_hf8f32 = 1;
    }

    if ((LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ))) {
      vname_in = 'x';
      upconvert_input_f16f32 = 1;
    }

    if ((LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ))) {
      vname_out = 'x';
      downconvert_input_f32bf16 = 1;
    }

    if ((LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ))) {
      vname_out = 'x';
      downconvert_input_f32f16 = 1;
    }

    if ((LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ))) {
      vname_out = 'x';
      downconvert_input_f32bf8 = 1;
    }
    if ((LIBXSMM_DATATYPE_HF8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ))) {
      vname_out = 'x';
      downconvert_input_f32hf8 = 1;
    }
  } else {
    if ((LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ))) {
      vname_in = 'y';
      upconvert_input_bf16f32 = 1;
    }

    if ((LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ))) {
      vname_in = 'y';
      upconvert_input_f16f32 = 1;
    }

    if ((LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ))) {
      vname_in = 'x';
      upconvert_input_bf8f32 = 1;
    }

    if ((LIBXSMM_DATATYPE_HF8 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ))) {
      vname_in = 'x';
      upconvert_input_hf8f32 = 1;
    }

    if ((LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ))) {
      vname_out = 'y';
      downconvert_input_f32bf16 = 1;
    }

    if ((LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ))) {
      vname_out = 'y';
      downconvert_input_f32f16 = 1;
    }

    if ((LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ))) {
      vname_out = 'x';
      downconvert_input_f32bf8 = 1;
    }
    if ((LIBXSMM_DATATYPE_HF8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ))) {
      vname_out = 'x';
      downconvert_input_f32hf8 = 1;
    }
  }
  if (m_trips_loop > 1) {
    libxsmm_generator_mateltwise_header_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_m_loop );
  }

  for (im = 0; im < m_unroll_factor; im++) {
    unsigned int use_masking = ((im == m_unroll_factor - 1) && (i_use_masking == 1)) ? 1 : 0;
    if ((upconvert_input_f16f32 > 0) && (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256)) {
      libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8(io_generated_code,
                                                         LIBXSMM_X86_INSTR_VCVTPH2PS,
                                                         i_micro_kernel_config->vector_name,
                                                         i_gp_reg_mapping->gp_reg_in,
                                                         LIBXSMM_X86_GP_REG_UNDEF,
                                                         0,
                                                         im * vlen * i_micro_kernel_config->datatype_size_in,
                                                         0,
                                                         LIBXSMM_X86_VEC_REG_UNDEF,
                                                         im,
                                                         (use_masking > 0) ? mask_in : 0,
                                                         0,
                                                         0);
    } else {
      libxsmm_x86_instruction_unified_vec_move( io_generated_code,
        i_micro_kernel_config->vmove_instruction_in,
        i_gp_reg_mapping->gp_reg_in,
        LIBXSMM_X86_GP_REG_UNDEF,
        0,
        im * vlen * i_micro_kernel_config->datatype_size_in,
        vname_in,
        im,
        use_masking,
        mask_in,
        0 );

      if (upconvert_input_bf16f32 > 0) {
        libxsmm_generator_cvtbf16ps_avx2_avx512(io_generated_code, i_micro_kernel_config->vector_name, im, im);
      } else if (upconvert_input_f16f32 > 0) {
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, im, im );
      } else if (upconvert_input_bf8f32 > 0) {
        libxsmm_generator_cvtbf8ps_avx512(io_generated_code, i_micro_kernel_config->vector_name, im, im);
      } else if (upconvert_input_hf8f32 > 0) {
        libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
            im, im, i_micro_kernel_config->dcvt_zmm_aux0, i_micro_kernel_config->dcvt_zmm_aux1, i_micro_kernel_config->dcvt_mask_aux0, i_micro_kernel_config->dcvt_mask_aux1 );
      }

      if (downconvert_input_f32bf16 > 0) {
        if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX)) {
          libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              im, im,
              i_micro_kernel_config->dcvt_zmm_aux0, i_micro_kernel_config->dcvt_zmm_aux1,
              i_micro_kernel_config->dcvt_mask_aux0, i_micro_kernel_config->dcvt_mask_aux1, 0);
        } else {
         libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, im, im );
        }
      } else if (downconvert_input_f32bf8 > 0) {
        libxsmm_generator_vcvtneps2bf8_avx512_preppedstack( io_generated_code, 'z',
            im, im,
            i_micro_kernel_config->dcvt_zmm_aux0, i_micro_kernel_config->dcvt_zmm_aux1,
            i_micro_kernel_config->dcvt_mask_aux0, i_micro_kernel_config->dcvt_mask_aux1,
            0, 0);
      } else if (downconvert_input_f32hf8 > 0) {
        libxsmm_generator_vcvtf32_to_hf8_avx512_preppedstack( io_generated_code, 'z',
            im, im,
            i_micro_kernel_config->dcvt_zmm_aux0, i_micro_kernel_config->dcvt_zmm_aux1, i_micro_kernel_config->dcvt_zmm_aux2, i_micro_kernel_config->dcvt_zmm_aux3,
            i_micro_kernel_config->dcvt_mask_aux0, i_micro_kernel_config->dcvt_mask_aux1, i_micro_kernel_config->dcvt_mask_aux2 );
      }

      if (downconvert_input_f32f16 > 0) {
         libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, im, im, 0x0 );
      }
    }
  }

  libxsmm_generator_mateltwise_header_n_dyn_loop(io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, 0);

  for (im = 0; im < m_unroll_factor; im++) {
    unsigned int use_masking = ((im == m_unroll_factor - 1) && (i_use_masking == 1)) ? 1 : 0;
    libxsmm_x86_instruction_unified_vec_move( io_generated_code,
        i_micro_kernel_config->vmove_instruction_out,
        i_gp_reg_mapping->gp_reg_out,
        LIBXSMM_X86_GP_REG_UNDEF,
        0,
        im * vlen * i_micro_kernel_config->datatype_size_out,
        vname_out,
        im,
        use_masking,
        mask_out,
        1 );
  }

  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_out, (long long)i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out);
  libxsmm_generator_mateltwise_footer_n_dyn_loop(io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, i_gp_reg_mapping->gp_reg_n);
  libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_n_loop, (long long)i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out);
  libxsmm_x86_instruction_alu_reg(io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_n_loop, i_gp_reg_mapping->gp_reg_out);

  if ((m_trips_loop > 1) || ((m_trips_loop == 1) && (peeled_m_trips > 0))) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in, (long long)m_unroll_factor * vlen * i_micro_kernel_config->datatype_size_in);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_out, (long long)m_unroll_factor * vlen * i_micro_kernel_config->datatype_size_out);
    if (m_trips_loop > 1) {
      libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_m_loop, m_trips_loop);
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_replicate_col_var_avx_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    libxsmm_mateltwise_kernel_config*              i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {

  unsigned int m, use_m_masking, m_trips, m_unroll_factor, m_trips_loop, peeled_m_trips, vlen, max_m_unrolling;
  unsigned int in_tsize, out_tsize, tsize;
  unsigned int mask_in = 1, mask_out = 1, mask_out_count;
  unsigned int END_LABEL = 1;
#if defined(LIBXSMM_GENERATOR_MATELTWISE_MISC_AVX_AVX512_JUMP_LABEL_TRACKER_MALLOC)
  libxsmm_jump_label_tracker* const p_jump_label_tracker = (libxsmm_jump_label_tracker*)malloc(sizeof(libxsmm_jump_label_tracker));
#else
  libxsmm_jump_label_tracker l_jump_label_tracker;
  libxsmm_jump_label_tracker* const p_jump_label_tracker = &l_jump_label_tracker;
#endif
  libxsmm_reset_jump_label_tracker(p_jump_label_tracker);

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in     = LIBXSMM_X86_GP_REG_R8;
  i_gp_reg_mapping->gp_reg_out    = LIBXSMM_X86_GP_REG_R9;
  i_gp_reg_mapping->gp_reg_n_loop = LIBXSMM_X86_GP_REG_R10;
  i_gp_reg_mapping->gp_reg_m_loop = LIBXSMM_X86_GP_REG_RAX;
  i_gp_reg_mapping->gp_reg_n      = LIBXSMM_X86_GP_REG_R11;

  /* load the input pointer and output pointer and the variable N */
  libxsmm_x86_instruction_alu_mem( io_generated_code,
    i_micro_kernel_config->alu_mov_instruction,
    i_gp_reg_mapping->gp_reg_param_struct,
    LIBXSMM_X86_GP_REG_UNDEF, 0,
    0,
    i_gp_reg_mapping->gp_reg_n,
    0 );

  libxsmm_x86_instruction_alu_mem( io_generated_code,
    i_micro_kernel_config->alu_mov_instruction,
    i_gp_reg_mapping->gp_reg_n,
    LIBXSMM_X86_GP_REG_UNDEF, 0,
    0,
    i_gp_reg_mapping->gp_reg_n,
    0 );

  libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_n, 0);
  libxsmm_x86_instruction_jump_to_label(io_generated_code, LIBXSMM_X86_INSTR_JLE, END_LABEL, p_jump_label_tracker);

  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      32,
      i_gp_reg_mapping->gp_reg_in,
      0 );

  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      64,
      i_gp_reg_mapping->gp_reg_out,
      0 );

  if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) {
    in_tsize = 2;
  } else if (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) {
    in_tsize = 1;
  } else if (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) {
    in_tsize = 1;
  } else if (LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) {
    in_tsize = 2;
  } else if (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) {
    in_tsize = 4;
  } else {
#if defined(LIBXSMM_GENERATOR_MATELTWISE_MISC_AVX_AVX512_JUMP_LABEL_TRACKER_MALLOC)
    free(p_jump_label_tracker);
#endif
    /* This should not happen */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }

  if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {
    out_tsize = 2;
  } else if (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {
    out_tsize = 1;
  } else if (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {
    out_tsize = 1;
  } else if (LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {
    out_tsize = 2;
  } else if (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {
    out_tsize = 4;
  } else {
#if defined(LIBXSMM_GENERATOR_MATELTWISE_MISC_AVX_AVX512_JUMP_LABEL_TRACKER_MALLOC)
    free(p_jump_label_tracker);
#endif
    /* This should not happen */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }

  tsize = LIBXSMM_MAX(in_tsize, out_tsize);
  vlen = (io_generated_code->arch < LIBXSMM_X86_AVX512) ? 32/tsize : 64/tsize;
  max_m_unrolling = (io_generated_code->arch < LIBXSMM_X86_AVX512) ? 16 : 32;

  m                 = i_mateltwise_desc->m;
  use_m_masking     = (m % vlen == 0) ? 0 : 1;
  m_trips           = (m + vlen - 1) / vlen;
  m_unroll_factor   = m_trips;
  m_trips_loop      = 1;
  peeled_m_trips    = 0;

  /* In this case we have to use CPX replacement sequence for downconverts... */
 if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX)
      && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) ) {
    libxsmm_generator_vcvtneps2bf16_avx512_prep_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
    i_micro_kernel_config->dcvt_zmm_aux0 = 31;
    i_micro_kernel_config->dcvt_zmm_aux1 = 30;
    if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256) {
      i_micro_kernel_config->dcvt_zmm_aux0 = max_m_unrolling - 1;
      i_micro_kernel_config->dcvt_zmm_aux1 = max_m_unrolling - 2;
    }
    /*i_micro_kernel_config->dcvt_mask_aux0 = 2;*/
    i_micro_kernel_config->dcvt_mask_aux0 = 3;
    max_m_unrolling -= 2;
  } else if ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) && (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) ) {
    libxsmm_generator_vcvtneps2bf8_avx512_prep_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
    i_micro_kernel_config->dcvt_zmm_aux0 = 31;
    i_micro_kernel_config->dcvt_zmm_aux1 = 30;
    /*i_micro_kernel_config->dcvt_mask_aux0 = 2;*/
    i_micro_kernel_config->dcvt_mask_aux0 = 3;
    max_m_unrolling -= 2;
  } else if ( (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) || (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) ) {
    libxsmm_generator_vcvt_hf8_tofrom_f32_avx512_prep_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
    i_micro_kernel_config->dcvt_zmm_aux0 = 31;
    i_micro_kernel_config->dcvt_zmm_aux1 = 30;
    i_micro_kernel_config->dcvt_mask_aux0 = 3;
    i_micro_kernel_config->dcvt_mask_aux1 = 4;
    max_m_unrolling -= 2;
    if (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {
      i_micro_kernel_config->dcvt_zmm_aux2 = 29;
      i_micro_kernel_config->dcvt_zmm_aux3 = 28;
      i_micro_kernel_config->dcvt_mask_aux2 = 5;
      max_m_unrolling -= 2;
    }
  }

  if (use_m_masking == 1) {
    if (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256) {
      const libxsmm_datatype precision = (libxsmm_datatype)(
          (LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) )
        ? LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) /* treat signed and unsigned types as equal */
        : LIBXSMM_DATATYPE_F32);
      mask_out = 1;
      mask_in = 1;
      mask_out_count = vlen - (m % vlen);
      libxsmm_generator_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, mask_out, mask_out_count, precision);
    } else {
      if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) || LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) {
        mask_in = m % vlen;
        if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) || LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {
          mask_out = mask_in;
        } else {
          mask_out = max_m_unrolling - 1;
          libxsmm_generator_initialize_avx_mask(io_generated_code, mask_out, m % vlen, LIBXSMM_DATATYPE_F32);
          max_m_unrolling--;
        }
      } else {
        mask_in = max_m_unrolling - 1;
        libxsmm_generator_initialize_avx_mask(io_generated_code, mask_in, m % vlen, LIBXSMM_DATATYPE_F32);
        max_m_unrolling--;
        if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) || LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {
          mask_out = m % vlen;
        } else {
          mask_out = mask_in;
        }
      }
    }
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
    libxsmm_generator_mn_code_block_replicate_col_var_avx_avx512( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        vlen, m_trips_loop, m_unroll_factor, peeled_m_trips, 0, mask_in, mask_out );
  }
  if (peeled_m_trips > 0) {
    libxsmm_generator_mn_code_block_replicate_col_var_avx_avx512( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        vlen, 0, peeled_m_trips, peeled_m_trips, use_m_masking, mask_in, mask_out );
  }

  /* In this case we have to use CPX replacement sequence for downconverts... */
  if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (LIBXSMM_DATATYPE_BF16 != LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) ) {
    libxsmm_generator_vcvtneps2bf16_avx512_clean_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
  } else if ( (LIBXSMM_DATATYPE_BF8 != LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) && (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) ) {
    libxsmm_generator_vcvtneps2bf8_avx512_clean_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
  } else if ( (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) || (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) ) {
    libxsmm_generator_vcvt_hf8_tofrom_f32_avx512_clean_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
  }
  libxsmm_x86_instruction_register_jump_label(io_generated_code, END_LABEL, p_jump_label_tracker);
#if defined(LIBXSMM_GENERATOR_MATELTWISE_MISC_AVX_AVX512_JUMP_LABEL_TRACKER_MALLOC)
  free(p_jump_label_tracker);
#endif
}


