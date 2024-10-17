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
#include "generator_mateltwise_reduce_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"

#if 0
#define USE_ENV_TUNING
#endif


LIBXSMM_API_INTERN
void libxsmm_generator_reduce_set_lp_vlen_vname_vmove_x86( libxsmm_generated_code*                        io_generated_code,
                                                           const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
                                                           const libxsmm_meltw_descriptor*                i_mateltwise_desc,
                                                           unsigned int*                                  io_vlen,
                                                           char*                                          io_vname_in,
                                                           char*                                          io_vname_out,
                                                           unsigned int*                                  io_vmove_instruction_in,
                                                           unsigned int*                                  io_vmove_instruction_out ) {
  char  vname_in = i_micro_kernel_config->vector_name;
  char  vname_out = i_micro_kernel_config->vector_name;
  unsigned int vmove_instruction_in = i_micro_kernel_config->vmove_instruction_in;
  unsigned int vmove_instruction_out = i_micro_kernel_config->vmove_instruction_out;
  unsigned int vlen = 16;

  if ( (io_generated_code->arch >= LIBXSMM_X86_AVX2) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ) {
    vlen = 8;
    if ((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ) {
      vname_in = 'x';
      vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU8;
    }

    if ((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ) {
      vname_out = 'x';
      vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVDQU8;
    }

    if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
      vname_in = 'x';
      vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU16;
    }

    if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
      vname_out = 'x';
      vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVDQU16;
    }

    if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
      vname_in = 'x';
      vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU16;
    }

    if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
      vname_out = 'x';
      vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVDQU16;
    }
  } else {
    if ((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ) {
      vname_in = 'x';
      vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU8;
    }

    if ((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ) {
      vname_out = 'x';
      vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVDQU8;
    }

    if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
      vname_in = 'y';
      vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU16;
    }

    if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
      vname_out = 'y';
      vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVDQU16;
    }

    if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
      vname_in = 'y';
      vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU16;
    }

    if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
      vname_out = 'y';
      vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVDQU16;
    }
  }

  *io_vlen = vlen;
  *io_vname_in = vname_in;
  *io_vname_out = vname_out;
  *io_vmove_instruction_in = vmove_instruction_in;
  *io_vmove_instruction_out = vmove_instruction_out;
}

LIBXSMM_API_INTERN
void libxsmm_generator_reduce_cols_ncnc_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {

  unsigned int bn, bc, N, C, Nb, iM, im, in, vreg0, vreg1;
  unsigned int use_m_masking, m_trips, m_outer_trips, m_inner_trips, mask_in_count, mask_out_count;
  unsigned int cur_acc0, cur_acc1, mask_load_0, mask_load_1, mask_store;
  unsigned int vlen = 16;
  unsigned int m_unroll_factor = 4;
  unsigned int mask_inout = 1;
  char vname_in;
  unsigned int cvt_vreg_aux0 = 30, cvt_vreg_aux1 = 31, cvt_vreg_aux2 = 27, cvt_vreg_aux3 = 28;
  unsigned int cvt_mask_aux0 = 7, cvt_mask_aux1 = 6, cvt_mask_aux2 = 5;

  bc  = i_mateltwise_desc->m;
  bn  = i_mateltwise_desc->n;
  C   = i_mateltwise_desc->ldi;
  N   = i_mateltwise_desc->ldo;

  Nb  = N/bn;

  if ( (N % bn != 0)  || (C % bc != 0) ) {
    /* This should not happen */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in     = LIBXSMM_X86_GP_REG_R8;
  i_gp_reg_mapping->gp_reg_out    = LIBXSMM_X86_GP_REG_R9;
  i_gp_reg_mapping->gp_reg_n_loop = LIBXSMM_X86_GP_REG_R10;

  /* load the input pointer and output pointer */
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

  if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
    vlen = 8;
    vname_in = 'y';
    mask_inout = 15;
    use_m_masking     = (bc % vlen == 0) ? 0 : 1;
    m_unroll_factor   = (use_m_masking == 0) ? 8 : 4;
    m_trips           = (bc + vlen - 1)/vlen;
    m_outer_trips     = (m_trips + m_unroll_factor - 1)/m_unroll_factor;


    if (use_m_masking > 0) {
      libxsmm_generator_initialize_avx_mask(io_generated_code, mask_inout, bc % vlen, (libxsmm_datatype)libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0));
    }

    /* Register allocation: Registers zmm8-zmm15 are accumulators, zmm0-zmm7 are used for loading input */
    for (iM = 0; iM < m_outer_trips; iM++) {
      m_inner_trips = (iM == m_outer_trips - 1) ? m_trips - iM * m_unroll_factor : m_unroll_factor;
      for (im = 0; im < m_inner_trips; im++) {
        cur_acc0 = m_unroll_factor + im;
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VXORPS,
                                                 i_micro_kernel_config->vector_name,
                                                 cur_acc0, cur_acc0, cur_acc0 );
      }

      if (Nb > 1) {
        /* open n loop */
        libxsmm_generator_mateltwise_header_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop );
      }

      for (in = 0; in < bn; in++ ) {
        for (im = 0; im < m_inner_trips; im++) {
          cur_acc0 = m_unroll_factor + im;
          vreg0    = im;
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
              i_micro_kernel_config->vmove_instruction_in,
              i_gp_reg_mapping->gp_reg_in,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (in * bc + im * vlen + iM * vlen * m_unroll_factor) * i_micro_kernel_config->datatype_size_in,
              vname_in,
              vreg0, ((use_m_masking == 1) && (iM == m_outer_trips-1) && (im == m_inner_trips-1)) ? 1 : 0, ((use_m_masking == 1) && (iM == m_outer_trips-1) && (im == m_inner_trips-1)) ? mask_inout : 0, 0 );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               LIBXSMM_X86_INSTR_VADDPS,
                                               i_micro_kernel_config->vector_name,
                                               vreg0, cur_acc0, cur_acc0 );
        }
      }

      if (Nb > 1) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
            i_micro_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_in,
            (long long)C * bn * i_micro_kernel_config->datatype_size_in);

        /* close n loop */
        libxsmm_generator_mateltwise_footer_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, Nb);

        /* Readjust reg_in */
        if (m_outer_trips > 1) {
          libxsmm_x86_instruction_alu_imm(  io_generated_code,
              LIBXSMM_X86_INSTR_SUBQ,
              i_gp_reg_mapping->gp_reg_in,
              (long long)C * N * i_micro_kernel_config->datatype_size_in);
        }
      }

      for (im = 0; im < m_inner_trips; im++) {
        cur_acc0 = m_unroll_factor + im;
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            LIBXSMM_X86_INSTR_VMOVUPS,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * vlen + iM * vlen * m_unroll_factor) * i_micro_kernel_config->datatype_size_out,
            i_micro_kernel_config->vector_name,
            cur_acc0, ((use_m_masking == 1) && (iM == m_outer_trips-1) && (im == m_inner_trips-1)) ? 1 : 0, ((use_m_masking == 1) && (iM == m_outer_trips-1) && (im == m_inner_trips-1)) ? mask_inout : 0, 1 );
      }
    }
    return;
  }

  if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
    if ((LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ||
        (LIBXSMM_DATATYPE_F16  == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0))    ) {
      vname_in = 'x';
    } else {
      vname_in = 'y';
    }
  } else {
    if ((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ) {
      vname_in = 'x';
    } else if ((LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ||
               (LIBXSMM_DATATYPE_F16  == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0))    ) {
      vname_in = 'y';
    } else {
      vname_in = 'z';
    }
    vlen = 32;
  }

  use_m_masking     = (bc % vlen == 0) ? 0 : 1;
  m_trips           = (bc + vlen - 1)/vlen;
  m_outer_trips     = (m_trips + m_unroll_factor - 1)/m_unroll_factor;

  if ((LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )))  {
    libxsmm_generator_vcvt_hf8_tofrom_f32_avx512_prep_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
  }

  /* Calculate input mask in case we see m_masking */
  if (use_m_masking == 1) {
    /* If the remaining elements are < 16, then we read a full vector and a partial one at the last m trip */
    /* If the remaining elements are >= 16, then we read a partial vector at the last m trip */
    /* Calculate mask reg 1 for input-reading */
    mask_in_count = ( (bc % vlen) > (vlen>>1)) ? vlen - (bc % vlen) : (vlen>>1) - (bc % vlen);
    libxsmm_generator_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, 1, mask_in_count, LIBXSMM_DATATYPE_F32);
    /* Calculate mask reg 2 for output-writing */
    if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
      mask_out_count = vlen - (bc % vlen);
      libxsmm_generator_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, 2, mask_out_count, LIBXSMM_DATATYPE_BF8);
    } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
      mask_out_count = vlen - (bc % vlen);
      libxsmm_generator_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, 2, mask_out_count, LIBXSMM_DATATYPE_HF8);
    } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
      mask_out_count = vlen - (bc % vlen);
      libxsmm_generator_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, 2, mask_out_count, LIBXSMM_DATATYPE_BF16);
    } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
      mask_out_count = vlen - (bc % vlen);
      libxsmm_generator_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, 2, mask_out_count, LIBXSMM_DATATYPE_F16);
    } else {
      mask_out_count = (vlen>>1) - (bc % (vlen>>1));
      libxsmm_generator_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, 2, mask_out_count, LIBXSMM_DATATYPE_F32);
    }
  }
  if ( (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))) {
    if ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX)  ) {
      unsigned int i = 0;
      char perm_array[32] = { 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61};
      char selector_array[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64 };
      for (i = 0; i < 32; i++) {
        perm_array[i] = (char)(perm_array[i] | selector_array[i]);
      }
      libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
          (const unsigned char *) perm_array,
          "perm_arrray_",
          i_micro_kernel_config->vector_name,
          29);
      if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))  {
        libxsmm_generator_vcvtneps2bf8_avx512_prep_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
      }
    }
  } else if ( (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) &&
       (io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
    if ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX)  ) {
      unsigned int i = 0;
      short perm_array[32] = { 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
      short selector_array[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 };
      for (i = 0; i < 32; i++) {
        perm_array[i] = (short)(perm_array[i] | selector_array[i]);
      }
      libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
          (const unsigned char *) perm_array,
          "perm_arrray_",
          i_micro_kernel_config->vector_name,
          29);
      libxsmm_generator_vcvtneps2bf16_avx512_prep_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
    } else {
      unsigned int i = 0;
      short perm_array[16] = { 1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15};
      short selector_array[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16, 16 };
      for (i = 0; i < 16; i++) {
        perm_array[i] = (short)(perm_array[i] | selector_array[i]);
      }
      libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
          (const unsigned char *) perm_array,
          "perm_arrray_",
          i_micro_kernel_config->vector_name,
          29);
      libxsmm_generator_vcvtneps2bf16_avx512_prep_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
    }
  } else if ( LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) {
    if ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX)  ) {
      unsigned int i = 0;
      short perm_array[32] = { 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
      short selector_array[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 };
      for (i = 0; i < 32; i++) {
        perm_array[i] = (short)(perm_array[i] | selector_array[i]);
      }
      libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
          (const unsigned char *) perm_array,
          "perm_arrray_",
          i_micro_kernel_config->vector_name,
          29);
    } else {
      unsigned int i = 0;
      short perm_array[16] = { 1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15};
      short selector_array[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16, 16 };
      for (i = 0; i < 16; i++) {
        perm_array[i] = (short)(perm_array[i] | selector_array[i]);
      }
      libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
          (const unsigned char *) perm_array,
          "perm_arrray_",
          i_micro_kernel_config->vector_name,
          29);
    }
  }

  /* Register allocation: Registers zmm8-zmm15 are accumulators, zmm0-zmm7 are used for loading input */
  for (iM = 0; iM < m_outer_trips; iM++) {
    m_inner_trips = (iM == m_outer_trips - 1) ? m_trips - iM * 4 : 4;
    for (im = 0; im < m_inner_trips; im++) {
      cur_acc0 = 8 + im * 2;
      cur_acc1 = 8 + im * 2 + 1;
      libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               LIBXSMM_X86_INSTR_VXORPS,
                                               i_micro_kernel_config->vector_name,
                                               cur_acc0, cur_acc0, cur_acc0 );
      libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               LIBXSMM_X86_INSTR_VXORPS,
                                               i_micro_kernel_config->vector_name,
                                               cur_acc1, cur_acc1, cur_acc1 );
    }

    if (Nb > 1) {
      /* open n loop */
      libxsmm_generator_mateltwise_header_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop );
    }

    for (in = 0; in < bn; in++ ) {
      for (im = 0; im < m_inner_trips; im++) {
        unsigned int m_done = 0;
        cur_acc0 = 8 + im * 2;
        cur_acc1 = 8 + im * 2 + 1;
        vreg0    = im * 2;
        vreg1    = im * 2 + 1;
        mask_load_0 = ((use_m_masking == 1) && (bc % vlen < (vlen>>1)) && (iM == m_outer_trips-1) && (im == m_inner_trips - 1)) ? 1 : 0;
        mask_load_1 = ((mask_load_0 == 0) && (use_m_masking == 1) && (iM == m_outer_trips-1) && (im == m_inner_trips - 1)) ? 1 : 0;

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (in * bc + im * vlen + iM * vlen * 4) * i_micro_kernel_config->datatype_size_in,
            vname_in,
            vreg0, mask_load_0, 1, 0 );

        if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          /* convert 16 bit values into 32 bit (integer convert) */
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
              LIBXSMM_X86_INSTR_VPMOVSXWD,
              i_micro_kernel_config->vector_name,
              vreg0, vreg0 );

          /* shift 16 bits to the left to generate valid FP32 numbers */
          libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code,
              LIBXSMM_X86_INSTR_VPSLLD_I,
              i_micro_kernel_config->vector_name,
              vreg0,
              vreg0,
              16);
        } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, vreg0, vreg0 );
        } else if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          /* convert 8 bit values into 32 bit floats */
          libxsmm_generator_cvtbf8ps_avx512( io_generated_code,
              i_micro_kernel_config->vector_name,
              vreg0,
              vreg0);
        } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          /* convert 8 bit values into 32 bit floats */
          libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              vreg0, vreg0, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
        }
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             vreg0, cur_acc0, cur_acc0 );

        m_done = iM * vlen * 4 + im * vlen + (vlen>>1);

        if ((mask_load_0 == 0) && (m_done < bc)) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              i_micro_kernel_config->vmove_instruction_in,
              i_gp_reg_mapping->gp_reg_in,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (in * bc + im * vlen + (vlen>>1) + iM * vlen * 4) * i_micro_kernel_config->datatype_size_in,
              vname_in,
              vreg1, mask_load_1, 1, 0 );

          if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
            /* convert 16 bit values into 32 bit (integer convert) */
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                LIBXSMM_X86_INSTR_VPMOVSXWD,
                i_micro_kernel_config->vector_name,
                vreg1, vreg1 );

            /* shift 16 bits to the left to generate valid FP32 numbers */
            libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code,
                LIBXSMM_X86_INSTR_VPSLLD_I,
                i_micro_kernel_config->vector_name,
                vreg1,
                vreg1,
                16);
          } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, vreg1, vreg1 );
          } else if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
            /* convert 8 bit values into 32 bit floats */
            libxsmm_generator_cvtbf8ps_avx512( io_generated_code,
                i_micro_kernel_config->vector_name,
                vreg1,
                vreg1);
          } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
            /* convert 8 bit values into 32 bit floats */
            libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                vreg1, vreg1, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
          }
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               LIBXSMM_X86_INSTR_VADDPS,
                                               i_micro_kernel_config->vector_name,
                                               vreg1, cur_acc1, cur_acc1 );
        }
      }
    }

    if (Nb > 1) {
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_in,
          (long long)C * bn * i_micro_kernel_config->datatype_size_in);

      /* close n loop */
      libxsmm_generator_mateltwise_footer_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, Nb);

      /* Readjust reg_in */
      if (m_outer_trips > 1) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
            LIBXSMM_X86_INSTR_SUBQ,
            i_gp_reg_mapping->gp_reg_in,
            (long long)C * N * i_micro_kernel_config->datatype_size_in);
      }
    }

    for (im = 0; im < m_inner_trips; im++) {
      cur_acc0 = 8 + im * 2;
      cur_acc1 = 8 + im * 2 + 1;
      mask_store = ((use_m_masking == 1) && (iM == m_outer_trips-1) && (im == m_inner_trips - 1)) ? 1 : 0;

      if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
        if ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) ) {
          /* RNE convert reg_0 and reg_1 */
          libxsmm_generator_vcvtneps2bf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name, cur_acc0, cur_acc0, 30, 31, 6, 7, 0, 0 );
          libxsmm_generator_vcvtneps2bf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name, cur_acc1, cur_acc1, 30, 31, 6, 7, 0, 0 );
          /* Properly interleave reg_0 and reg_1 into reg_0 */
          libxsmm_x86_instruction_vec_compute_3reg(io_generated_code,
              LIBXSMM_X86_INSTR_VPERMT2B,
              i_micro_kernel_config->vector_name,
              cur_acc1,
              29,
              cur_acc0);
        }

        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_micro_kernel_config->instruction_set,
                                          i_micro_kernel_config->vmove_instruction_out,
                                          i_gp_reg_mapping->gp_reg_out,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (im * vlen + iM * vlen * 4) * i_micro_kernel_config->datatype_size_out,
                                          i_micro_kernel_config->vector_name,
                                          cur_acc0, mask_store * 2, 0, 1 );
      } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
        if ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) ) {
          /* RNE convert reg_0 and reg_1 */
          libxsmm_generator_vcvtf32_to_hf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              cur_acc0, cur_acc0, cvt_vreg_aux0, cvt_vreg_aux1, cvt_vreg_aux2, cvt_vreg_aux3, cvt_mask_aux0, cvt_mask_aux1, cvt_mask_aux2);
          libxsmm_generator_vcvtf32_to_hf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              cur_acc1, cur_acc1, cvt_vreg_aux0, cvt_vreg_aux1, cvt_vreg_aux2, cvt_vreg_aux3, cvt_mask_aux0, cvt_mask_aux1, cvt_mask_aux2);
         /* Properly interleave reg_0 and reg_1 into reg_0 */
          libxsmm_x86_instruction_vec_compute_3reg(io_generated_code,
              LIBXSMM_X86_INSTR_VPERMT2B,
              i_micro_kernel_config->vector_name,
              cur_acc1,
              29,
              cur_acc0);
        }

        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_micro_kernel_config->instruction_set,
                                          i_micro_kernel_config->vmove_instruction_out,
                                          i_gp_reg_mapping->gp_reg_out,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (im * vlen + iM * vlen * 4) * i_micro_kernel_config->datatype_size_out,
                                          i_micro_kernel_config->vector_name,
                                          cur_acc0, mask_store * 2, 0, 1 );
      } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
        if ((io_generated_code->arch >= LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX)) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
              LIBXSMM_X86_INSTR_VCVTNE2PS2BF16,
              i_micro_kernel_config->vector_name,
              cur_acc0, cur_acc1, cur_acc0 );
        } else {
          /* RNE convert reg_0 and reg_1 */
          libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name, cur_acc0, cur_acc0, 30, 31, 6, 7, 1 );
          libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name, cur_acc1, cur_acc1, 30, 31, 6, 7, 1 );
          /* Properly interleave reg_0 and reg_1 into reg_0 */
          libxsmm_x86_instruction_vec_compute_3reg(io_generated_code,
              LIBXSMM_X86_INSTR_VPERMT2W,
              i_micro_kernel_config->vector_name,
              cur_acc1,
              29,
              cur_acc0);
        }

        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_micro_kernel_config->instruction_set,
                                          i_micro_kernel_config->vmove_instruction_out,
                                          i_gp_reg_mapping->gp_reg_out,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (im * vlen + iM * vlen * 4) * i_micro_kernel_config->datatype_size_out,
                                          i_micro_kernel_config->vector_name,
                                          cur_acc0, mask_store * 2, 0, 1 );
      } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
        /* RNE convert reg_0 and reg_1 */
        libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, cur_acc0, cur_acc0, 0,
                                                                (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
        libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, cur_acc1, cur_acc1, 0,
                                                                (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
        /* Properly interleave reg_0 and reg_1 into reg_0 */
        libxsmm_x86_instruction_vec_compute_3reg(io_generated_code,
            LIBXSMM_X86_INSTR_VPERMT2W,
            i_micro_kernel_config->vector_name,
            cur_acc1,
            29,
            cur_acc0);

        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_micro_kernel_config->instruction_set,
                                          i_micro_kernel_config->vmove_instruction_out,
                                          i_gp_reg_mapping->gp_reg_out,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (im * vlen + iM * vlen * 4) * i_micro_kernel_config->datatype_size_out,
                                          i_micro_kernel_config->vector_name,
                                          cur_acc0, mask_store * 2, 0, 1 );
      } else {
        unsigned int m_done = 0;
        mask_load_0 = ((use_m_masking == 1) && (bc % vlen < (vlen>>1)) && (iM == m_outer_trips-1) && (im == m_inner_trips - 1)) ? 1 : 0;
        mask_load_1 = ((mask_load_0 == 0) && (use_m_masking == 1) && (iM == m_outer_trips-1) && (im == m_inner_trips - 1)) ? 1 : 0;

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMOVUPS,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * vlen + iM * vlen * 4) * i_micro_kernel_config->datatype_size_out,
            i_micro_kernel_config->vector_name,
            cur_acc0, mask_load_0 * 2, 0, 1 );

        m_done = iM * vlen * 4 + im * vlen + (vlen>>1);

        if ((mask_load_0 == 0) && (m_done < bc)) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVUPS,
              i_gp_reg_mapping->gp_reg_out,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (im * vlen + iM * vlen * 4 + (vlen>>1)) * i_micro_kernel_config->datatype_size_out,
              i_micro_kernel_config->vector_name,
              cur_acc1, mask_load_1 * 2, 0, 1 );
        }
      }
    }
  }

  if ( (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) && (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX)  ) {
    libxsmm_generator_vcvtneps2bf8_avx512_clean_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
  } else if ((LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) && (io_generated_code->arch < LIBXSMM_X86_AVX512_CPX)
      && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
    libxsmm_generator_vcvtneps2bf16_avx512_clean_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
  } else if ((LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )))  {
    libxsmm_generator_vcvt_hf8_tofrom_f32_avx512_clean_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_reduce_cols_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {

  unsigned int _in, in_use, im, m, n, m_trips, use_m_masking, mask_count, compute_squared_vals_reduce, compute_plain_vals_reduce;
  unsigned int start_vreg_sum = 0;
  unsigned int start_vreg_sum2 = 0;
  unsigned int reduce_instr = 0, reduce_instr_squared = 0, absmax_instr = 0;
  char  vname_in = i_micro_kernel_config->vector_name;
  char  vname_out = i_micro_kernel_config->vector_name;
  unsigned int vmove_instruction_in = i_micro_kernel_config->vmove_instruction_in;
  unsigned int vmove_instruction_out = i_micro_kernel_config->vmove_instruction_out;
  unsigned int mask_reg = 0, mask_reg_in = 0, mask_reg_out = 0;
  unsigned int vlen = 16;
  unsigned int aux_vreg = 0;
  unsigned int tmp_vreg = 31;
  unsigned int cvt_vreg_aux0 = 30, cvt_vreg_aux1 = 29, cvt_vreg_aux2 = 27, cvt_vreg_aux3 = 28;
  unsigned int cvt_mask_aux0 = 7, cvt_mask_aux1 = 6, cvt_mask_aux2 = 5;
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

  if ((LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))) {
    max_m_unrolling = 28;
    m_unroll_factor = 28;
  }

  libxsmm_generator_reduce_set_lp_vlen_vname_vmove_x86( io_generated_code, i_micro_kernel_config, i_mateltwise_desc, &vlen, &vname_in, &vname_out, &vmove_instruction_in, &vmove_instruction_out );

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
    if (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)) {
      reduce_instr = LIBXSMM_X86_INSTR_VADDPD;
      reduce_instr_squared = LIBXSMM_X86_INSTR_VFMADD231PD;
    } else {
      reduce_instr = LIBXSMM_X86_INSTR_VADDPS;
      reduce_instr_squared = LIBXSMM_X86_INSTR_VFMADD231PS;
    }
  } else if ( flag_reduce_op_max > 0 || flag_reduce_op_absmax > 0) {
    if (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)) {
      reduce_instr = LIBXSMM_X86_INSTR_VMAXPD;
      if (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) {
        absmax_instr = LIBXSMM_X86_INSTR_VRANGEPD;
      } else {
        absmax_instr = LIBXSMM_X86_INSTR_VPANDQ;
      }
    } else {
      reduce_instr = LIBXSMM_X86_INSTR_VMAXPS;
      if (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) {
        absmax_instr = LIBXSMM_X86_INSTR_VRANGEPS;
      } else {
        absmax_instr = LIBXSMM_X86_INSTR_VPANDQ;
      }
    }
  } else if ( flag_reduce_op_min > 0 ) {
    if (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)) {
      reduce_instr = LIBXSMM_X86_INSTR_VMINPD;
    } else {
      reduce_instr = LIBXSMM_X86_INSTR_VMINPS;
    }
  } else {
    /* This should not happen */
    printf("Only supported reduction OPs are ADD and MAX for this reduce kernel\n");
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  if ((compute_squared_vals_reduce > 0) && !(reduce_instr == LIBXSMM_X86_INSTR_VADDPS || reduce_instr == LIBXSMM_X86_INSTR_VADDPD)) {
    /* This should not happen */
    printf("Support for squares's reduction only when reduction OP is ADD\n");
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
    vlen = 8;
    tmp_vreg = 15;
    max_m_unrolling = 15;
    if (flag_reduce_op_absmax > 0) {
      aux_vreg  = 14;
      max_m_unrolling--;
      if (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)) {
        libxsmm_generator_load_vreg_signmask(io_generated_code, i_micro_kernel_config->vector_name, LIBXSMM_X86_GP_REG_RAX, aux_vreg, LIBXSMM_DATATYPE_F64);
      } else {
        libxsmm_generator_load_vreg_signmask(io_generated_code, i_micro_kernel_config->vector_name, LIBXSMM_X86_GP_REG_RAX, aux_vreg, LIBXSMM_DATATYPE_F32);
      }
    }
    if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
      max_m_unrolling--;
      cvt_vreg_aux0 = max_m_unrolling;
    }
  }

  if (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)) {
    vlen = vlen/2;
  }

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in                   = LIBXSMM_X86_GP_REG_R8;
  i_gp_reg_mapping->gp_reg_reduced_elts         = LIBXSMM_X86_GP_REG_R9;
  i_gp_reg_mapping->gp_reg_reduced_elts_squared = LIBXSMM_X86_GP_REG_R10;
  i_gp_reg_mapping->gp_reg_m_loop               = LIBXSMM_X86_GP_REG_R11;
  i_gp_reg_mapping->gp_reg_n_loop               = LIBXSMM_X86_GP_REG_RAX;

  /* load the input pointer and output pointer */
  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      32,
      i_gp_reg_mapping->gp_reg_in,
      0 );

  if ( compute_plain_vals_reduce > 0 ) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
       i_gp_reg_mapping->gp_reg_param_struct,
       LIBXSMM_X86_GP_REG_UNDEF, 0,
       64,
       i_gp_reg_mapping->gp_reg_reduced_elts,
       0 );
    if ( compute_squared_vals_reduce > 0 ) {
      unsigned int result_size = i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out;
      libxsmm_x86_instruction_alu_reg(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduced_elts, i_gp_reg_mapping->gp_reg_reduced_elts_squared);
      libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_mapping->gp_reg_reduced_elts_squared, result_size);
    }
  } else {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
       i_gp_reg_mapping->gp_reg_param_struct,
       LIBXSMM_X86_GP_REG_UNDEF, 0,
       64,
       i_gp_reg_mapping->gp_reg_reduced_elts_squared,
       0 );
  }

  /* We fully unroll in N dimension, calculate m-mask if there is remainder */
  m                 = i_mateltwise_desc->m;
  n                 = i_mateltwise_desc->n;
  use_m_masking     = ( m % vlen == 0 ) ? 0 : 1;

  /* Calculate input mask in case we see m_masking */
  if (use_m_masking == 1) {
    const int datatype = libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP);
    /* Calculate mask reg 1 for input-reading */
    if (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) {
      mask_count =  vlen - (m % vlen);
      mask_reg = 1;
      libxsmm_generator_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, mask_reg, mask_count, (libxsmm_datatype)datatype);
      mask_reg_in = mask_reg;
      mask_reg_out = mask_reg;
    } else {
      max_m_unrolling--;
      mask_reg = max_m_unrolling;
      libxsmm_generator_initialize_avx_mask(io_generated_code, mask_reg, m % vlen, (libxsmm_datatype)datatype);
      mask_reg_in = mask_reg;
      mask_reg_out = mask_reg;
      if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) || LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
        mask_reg_in = m % vlen;
      }
      if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) || LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
        mask_reg_out = m % vlen;
      }
    }
  }

  m_trips           = (m + vlen - 1) / vlen;
  m_unroll_factor   = m_trips;
  m_trips_loop      = 1;
  peeled_m_trips    = 0;

  if ( (compute_plain_vals_reduce > 0) && (compute_squared_vals_reduce > 0) ) {
    max_m_unrolling = max_m_unrolling/2;
    start_vreg_sum2 = max_m_unrolling;
  }

  if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
    libxsmm_generator_vcvtneps2bf8_avx512_prep_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
  } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
    if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
      libxsmm_generator_vcvtneps2bf16_avx512_prep_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
    }
  } else if ((LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))) {
    libxsmm_generator_vcvt_hf8_tofrom_f32_avx512_prep_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
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
      libxsmm_generator_mateltwise_header_m_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_m_loop );
    }
    /* Initialize accumulators to zero */
    for (_in = 0; _in < split_factor; _in++) {
      for (im = 0; im < m_unroll_factor; im++) {
        if ( compute_plain_vals_reduce > 0 ) {
          if ( flag_reduce_op_add > 0 || flag_reduce_op_absmax > 0) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                     LIBXSMM_X86_INSTR_VXORPS,
                                                     i_micro_kernel_config->vector_name,
                                                     start_vreg_sum + im + _in * m_unroll_factor, start_vreg_sum + im + _in * m_unroll_factor, start_vreg_sum + im + _in * m_unroll_factor );
          } else if ( flag_reduce_op_max > 0 || flag_reduce_op_min > 0 ) {
            libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                (((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0))) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX)) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_in,
                i_gp_reg_mapping->gp_reg_in,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                (im * vlen) * i_micro_kernel_config->datatype_size_in,
                vname_in,
                start_vreg_sum + im + _in * m_unroll_factor, 0, 0, 0 );
            if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
              libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, start_vreg_sum + im + _in * m_unroll_factor, start_vreg_sum + im + _in * m_unroll_factor);
            } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
              libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                  start_vreg_sum + im + _in * m_unroll_factor, start_vreg_sum + im + _in * m_unroll_factor, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
            } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, start_vreg_sum + im + _in * m_unroll_factor, start_vreg_sum + im + _in * m_unroll_factor );
            } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
              libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, start_vreg_sum + im + _in * m_unroll_factor, start_vreg_sum + im + _in * m_unroll_factor);
            }
          }
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                   LIBXSMM_X86_INSTR_VXORPS,
                                                   i_micro_kernel_config->vector_name,
                                                   start_vreg_sum2 + im + _in * m_unroll_factor, start_vreg_sum2 + im + _in * m_unroll_factor, start_vreg_sum2 + im + _in * m_unroll_factor );
        }
      }
    }

    if (n_trips >= 1) {
      if (n_trips > 1) {
        libxsmm_generator_mateltwise_header_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop );
      }

      for (_in = 0; _in < split_factor; _in++) {
        for (im = 0; im < m_unroll_factor; im++) {
          in_use = _in;
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
              (((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0))) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX)) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_in,
              i_gp_reg_mapping->gp_reg_in,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (im * vlen + in_use * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
              vname_in,
              tmp_vreg, 0, 0, 0 );
          if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
            libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
          } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
            libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                tmp_vreg, tmp_vreg, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
          } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
          } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
            libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
          }

          if ( compute_plain_vals_reduce > 0 ) {
            if ((io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (flag_reduce_op_absmax > 0)) {
              libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, absmax_instr,
                  i_micro_kernel_config->vector_name, tmp_vreg, start_vreg_sum + im + _in * m_unroll_factor, start_vreg_sum + im + _in * m_unroll_factor, 0, 0, 0, 3);
            } else {
              if (flag_reduce_op_absmax > 0) {
                libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, absmax_instr, i_micro_kernel_config->vector_name, tmp_vreg, aux_vreg, tmp_vreg );
              }
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                   reduce_instr,
                                                   i_micro_kernel_config->vector_name,
                                                   tmp_vreg, start_vreg_sum + im + _in * m_unroll_factor, start_vreg_sum + im + _in * m_unroll_factor);
            }
          }

          if ( compute_squared_vals_reduce > 0 ) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 reduce_instr_squared,
                                                 i_micro_kernel_config->vector_name,
                                                 tmp_vreg, tmp_vreg, start_vreg_sum2 + im + _in * m_unroll_factor);
          }
        }
      }

      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_in,
          (long long)i_mateltwise_desc->ldi * split_factor * i_micro_kernel_config->datatype_size_in);

      if (n_trips > 1) {
        libxsmm_generator_mateltwise_footer_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, n_trips);
      }
    }

    if (peeled_n_trips > 0) {
      for (_in = 0; _in < peeled_n_trips; _in++) {
        for (im = 0; im < m_unroll_factor; im++) {
          in_use = _in;
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
              (((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0))) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX)) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_in,
              i_gp_reg_mapping->gp_reg_in,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (im * vlen + in_use * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
              vname_in,
              tmp_vreg, 0, 0, 0 );
          if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
            libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
          } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
            libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                tmp_vreg, tmp_vreg, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
          } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
          } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
            libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
          }

          if ( compute_plain_vals_reduce > 0 ) {
            if ((io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (flag_reduce_op_absmax > 0)) {
              libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, absmax_instr,
                  i_micro_kernel_config->vector_name, tmp_vreg, start_vreg_sum + im + _in * m_unroll_factor, start_vreg_sum + im + _in * m_unroll_factor, 0, 0, 0, 3);
            } else {
              if (flag_reduce_op_absmax > 0) {
                libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, absmax_instr, i_micro_kernel_config->vector_name, tmp_vreg, aux_vreg, tmp_vreg );
              }
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                   reduce_instr,
                                                   i_micro_kernel_config->vector_name,
                                                   tmp_vreg, start_vreg_sum + im + _in * m_unroll_factor, start_vreg_sum + im + _in * m_unroll_factor);
            }
          }

          if ( compute_squared_vals_reduce > 0 ) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 reduce_instr_squared,
                                                 i_micro_kernel_config->vector_name,
                                                 tmp_vreg, tmp_vreg, start_vreg_sum2 + im + _in * m_unroll_factor);
          }
        }
      }
    }

    if (n_trips >= 1) {
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_sub_instruction,
          i_gp_reg_mapping->gp_reg_in,
          (long long)i_mateltwise_desc->ldi * split_factor * n_trips * i_micro_kernel_config->datatype_size_in);
    }


    for (_in = 1; _in < split_factor; _in++) {
      for (im = 0; im < m_unroll_factor; im++) {
        if ( compute_plain_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               reduce_instr,
                                               i_micro_kernel_config->vector_name,
                                               start_vreg_sum + im, start_vreg_sum + im + _in * m_unroll_factor, start_vreg_sum + im);
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               reduce_instr,
                                               i_micro_kernel_config->vector_name,
                                               start_vreg_sum2 + im, start_vreg_sum2 + im + _in * m_unroll_factor, start_vreg_sum2 + im);
        }
      }
    }

    /* Store computed results */
    for (im = 0; im < m_unroll_factor; im++) {
      if ( compute_plain_vals_reduce > 0 ) {
        if (reduce_on_output > 0) {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                            (((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT))) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX)) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            im * vlen * i_micro_kernel_config->datatype_size_out,
                                            vname_out,
                                            tmp_vreg, 0, 0, 0 );
          if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
          } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                tmp_vreg, tmp_vreg, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
          } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
          } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
          }

          if ((io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (flag_reduce_op_absmax > 0)) {
            libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, absmax_instr,
                i_micro_kernel_config->vector_name, tmp_vreg, start_vreg_sum + im, start_vreg_sum + im, 0, 0, 0, 3);
          } else {
            if (flag_reduce_op_absmax > 0) {
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, absmax_instr, i_micro_kernel_config->vector_name, tmp_vreg, aux_vreg, tmp_vreg );
            }
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 reduce_instr,
                                                 i_micro_kernel_config->vector_name,
                                                 tmp_vreg, start_vreg_sum + im, start_vreg_sum + im);
          }
        }

        if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          libxsmm_generator_vcvtneps2bf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              start_vreg_sum + im, start_vreg_sum + im,
              30, 31,
              2, 3, 0, 0);
        } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          libxsmm_generator_vcvtf32_to_hf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              start_vreg_sum + im, start_vreg_sum + im, cvt_vreg_aux0, cvt_vreg_aux1, cvt_vreg_aux2, cvt_vreg_aux3, cvt_mask_aux0, cvt_mask_aux1, cvt_mask_aux2);
        } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, start_vreg_sum + im, start_vreg_sum + im, 0,
                                                                  (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
        } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
            libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                start_vreg_sum + im, start_vreg_sum + im,
                tmp_vreg, cvt_vreg_aux0,
                2, 3, 0);
          } else {
           libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, start_vreg_sum + im, start_vreg_sum + im );
          }
        }
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                          (((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX)) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_out,
                                          i_gp_reg_mapping->gp_reg_reduced_elts,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          im * vlen * i_micro_kernel_config->datatype_size_out,
                                          vname_out,
                                          start_vreg_sum + im, 0, 0, 1 );

      }

      if ( compute_squared_vals_reduce > 0 ) {
        if (reduce_on_output > 0) {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                            (((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX)) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            im * vlen * i_micro_kernel_config->datatype_size_out,
                                            vname_out,
                                            tmp_vreg, 0, 0, 0 );
          if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
          } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                tmp_vreg, tmp_vreg, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
          } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
          } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
          }
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               reduce_instr,
                                               i_micro_kernel_config->vector_name,
                                               tmp_vreg, start_vreg_sum2 + im, start_vreg_sum2 + im);
        }

        if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          libxsmm_generator_vcvtneps2bf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              start_vreg_sum2 + im, start_vreg_sum2 + im,
              30, 31,
              2, 3, 0, 0);
        } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          libxsmm_generator_vcvtf32_to_hf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              start_vreg_sum2 + im, start_vreg_sum2 + im, cvt_vreg_aux0, cvt_vreg_aux1, cvt_vreg_aux2, cvt_vreg_aux3, cvt_mask_aux0, cvt_mask_aux1, cvt_mask_aux2);
        } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, start_vreg_sum2 + im, start_vreg_sum2 + im, 0,
                                                                  (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
        } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
            libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                start_vreg_sum2 + im, start_vreg_sum2 + im,
                tmp_vreg, cvt_vreg_aux0,
                2, 3, 0);
          } else {
           libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, start_vreg_sum2 + im, start_vreg_sum2 + im );
          }
        }
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                          (((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX)) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_out,
                                          i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          im * vlen * i_micro_kernel_config->datatype_size_out,
                                          vname_out,
                                          start_vreg_sum2 + im, 0, 0, 1 );
      }
    }

    if (m_trips_loop > 1) {
      /* Adjust input and output pointer */
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_in,
          (long long)m_unroll_factor * vlen * i_micro_kernel_config->datatype_size_in);

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
            i_micro_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_reduced_elts,
            (long long)m_unroll_factor * vlen * i_micro_kernel_config->datatype_size_out);
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
            i_micro_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_reduced_elts_squared,
            (long long)m_unroll_factor * vlen * i_micro_kernel_config->datatype_size_out);
      }
      libxsmm_generator_mateltwise_footer_m_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_m_loop, m_trips_loop);
    }
  }

  if (peeled_m_trips > 0) {
    if (m_trips_loop == 1) {
      /* Adjust input and output pointer */
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_in,
          (long long)m_unroll_factor * vlen * i_micro_kernel_config->datatype_size_in);

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
            i_micro_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_reduced_elts,
            (long long)m_unroll_factor * vlen * i_micro_kernel_config->datatype_size_out);
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
            i_micro_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_reduced_elts_squared,
            (long long)m_unroll_factor * vlen * i_micro_kernel_config->datatype_size_out);
      }
    }

    for (_in = 0; _in < split_factor; _in++) {
      for (im = 0; im < peeled_m_trips; im++) {
        /* Initialize accumulators to zero */
        if ( flag_reduce_op_add > 0 ||  flag_reduce_op_absmax > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                   LIBXSMM_X86_INSTR_VXORPS,
                                                   i_micro_kernel_config->vector_name,
                                                   start_vreg_sum + im + _in * m_unroll_factor, start_vreg_sum + im + _in * m_unroll_factor, start_vreg_sum + im + _in * m_unroll_factor );
        } else if ( flag_reduce_op_max > 0 || flag_reduce_op_min > 0 ) {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
              ( ((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0))) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && !((use_m_masking > 0) && (im == peeled_m_trips - 1)) ) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_in,
              i_gp_reg_mapping->gp_reg_in,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              im * vlen * i_micro_kernel_config->datatype_size_in,
              vname_in,
              start_vreg_sum + im + _in * m_unroll_factor , ((im == peeled_m_trips -1) && (use_m_masking > 0)) ? 1 : 0, ((im == peeled_m_trips -1) && (use_m_masking > 0)) ? mask_reg_in : 0, 0);
          if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
            libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, start_vreg_sum + im + _in * m_unroll_factor, start_vreg_sum + im + _in * m_unroll_factor );
          } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
            libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                start_vreg_sum + im + _in * m_unroll_factor, start_vreg_sum + im + _in * m_unroll_factor, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
          } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, start_vreg_sum + im + _in * m_unroll_factor, start_vreg_sum + im + _in * m_unroll_factor );
          } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
            libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, start_vreg_sum + im + _in * m_unroll_factor, start_vreg_sum + im + _in * m_unroll_factor );
          }
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                   LIBXSMM_X86_INSTR_VXORPS,
                                                   i_micro_kernel_config->vector_name,
                                                   start_vreg_sum2 + im + _in * m_unroll_factor, start_vreg_sum2 + im + _in * m_unroll_factor, start_vreg_sum2 + im + _in * m_unroll_factor );
        }
      }
    }


    if (n_trips >= 1) {
      if (n_trips > 1) {
        libxsmm_generator_mateltwise_header_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop );
      }

      for (_in = 0; _in < split_factor; _in++) {
        for (im = 0; im < peeled_m_trips; im++) {
          in_use = _in;
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
              ( ((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0))) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && !((use_m_masking > 0) && (im == peeled_m_trips - 1)) ) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_in,
              i_gp_reg_mapping->gp_reg_in,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (im * vlen + in_use * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
              vname_in,
              tmp_vreg, ((im == peeled_m_trips -1) && (use_m_masking > 0)) ? 1 : 0, ((im == peeled_m_trips -1) && (use_m_masking > 0)) ? mask_reg_in : 0, 0 );
          if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
            libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
          } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
            libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                tmp_vreg, tmp_vreg, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
          } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
          } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
            libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
          }

          if ( compute_plain_vals_reduce > 0 ) {
            if ((io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (flag_reduce_op_absmax > 0)) {
              libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, absmax_instr,
                  i_micro_kernel_config->vector_name, tmp_vreg, start_vreg_sum + im + _in * m_unroll_factor, start_vreg_sum + im + _in * m_unroll_factor, 0, 0, 0, 3);
            } else {
              if (flag_reduce_op_absmax > 0) {
                libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, absmax_instr, i_micro_kernel_config->vector_name, tmp_vreg, aux_vreg, tmp_vreg );
              }
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                   reduce_instr,
                                                   i_micro_kernel_config->vector_name,
                                                   tmp_vreg, start_vreg_sum + im + _in * m_unroll_factor, start_vreg_sum + im + _in * m_unroll_factor );
            }
          }

          if ( compute_squared_vals_reduce > 0 ) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 reduce_instr_squared,
                                                 i_micro_kernel_config->vector_name,
                                                 tmp_vreg, tmp_vreg, start_vreg_sum2 + im + _in * m_unroll_factor );
          }
        }
      }

      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_in,
          (long long)i_mateltwise_desc->ldi * split_factor * i_micro_kernel_config->datatype_size_in);

      if (n_trips > 1) {
        libxsmm_generator_mateltwise_footer_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, n_trips);
      }
    }

    if (peeled_n_trips > 0) {
      for (_in = 0; _in < peeled_n_trips; _in++) {
        for (im = 0; im < peeled_m_trips; im++) {
          in_use = _in;
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
              ( ((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0))) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && !((use_m_masking > 0) && (im == peeled_m_trips - 1)) ) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_in,
              i_gp_reg_mapping->gp_reg_in,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (im * vlen + in_use * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
              vname_in,
              tmp_vreg, ((im == peeled_m_trips -1) && (use_m_masking > 0)) ? 1 : 0, ((im == peeled_m_trips -1) && (use_m_masking > 0)) ? mask_reg_in : 0, 0 );
          if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
            libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
          } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
            libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                tmp_vreg, tmp_vreg, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
          } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
          } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
            libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
          }

          if ( compute_plain_vals_reduce > 0 ) {
            if ((io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (flag_reduce_op_absmax > 0)) {
              libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, absmax_instr,
                  i_micro_kernel_config->vector_name, tmp_vreg, start_vreg_sum + im + _in * m_unroll_factor, start_vreg_sum + im + _in * m_unroll_factor, 0, 0, 0, 3);
            } else {
              if (flag_reduce_op_absmax > 0) {
                libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, absmax_instr, i_micro_kernel_config->vector_name, tmp_vreg, aux_vreg, tmp_vreg );
              }
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                   reduce_instr,
                                                   i_micro_kernel_config->vector_name,
                                                   tmp_vreg, start_vreg_sum + im + _in * m_unroll_factor, start_vreg_sum + im + _in * m_unroll_factor );
            }
          }

          if ( compute_squared_vals_reduce > 0 ) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 reduce_instr_squared,
                                                 i_micro_kernel_config->vector_name,
                                                 tmp_vreg, tmp_vreg, start_vreg_sum2 + im + _in * m_unroll_factor );
          }
        }
      }
    }

    if (n_trips >= 1) {
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_sub_instruction,
          i_gp_reg_mapping->gp_reg_in,
          (long long)i_mateltwise_desc->ldi * split_factor * n_trips * i_micro_kernel_config->datatype_size_in);
    }

    for (_in = 1; _in < split_factor; _in++) {
      for (im = 0; im < peeled_m_trips; im++) {
        if ( compute_plain_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               reduce_instr,
                                               i_micro_kernel_config->vector_name,
                                               start_vreg_sum + im, start_vreg_sum + im + _in * m_unroll_factor, start_vreg_sum + im);
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               reduce_instr,
                                               i_micro_kernel_config->vector_name,
                                               start_vreg_sum2 + im, start_vreg_sum2 + im + _in * m_unroll_factor, start_vreg_sum2 + im);
        }
      }
    }

    /* Store computed results */
    for (im = 0; im < peeled_m_trips; im++) {
      if ( compute_plain_vals_reduce > 0 ) {
        if (reduce_on_output > 0) {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                            ( ((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT))) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && !((use_m_masking > 0) && (im == peeled_m_trips - 1)) ) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            im * vlen * i_micro_kernel_config->datatype_size_out,
                                            vname_out,
                                            tmp_vreg, ((im == peeled_m_trips -1) && (use_m_masking > 0)) ? 1 : 0, ((im == peeled_m_trips -1) && (use_m_masking > 0)) ? mask_reg_out : 0, 0 );
          if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
          } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                tmp_vreg, tmp_vreg, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
          } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
          } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
          }

          if ((io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (flag_reduce_op_absmax > 0)) {
            libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, absmax_instr,
                i_micro_kernel_config->vector_name, tmp_vreg, start_vreg_sum + im, start_vreg_sum + im, 0, 0, 0, 3);
          } else {
            if (flag_reduce_op_absmax > 0) {
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, absmax_instr, i_micro_kernel_config->vector_name, tmp_vreg, aux_vreg, tmp_vreg );
            }
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 reduce_instr,
                                                 i_micro_kernel_config->vector_name,
                                                 tmp_vreg, start_vreg_sum + im, start_vreg_sum + im);
          }
        }

        if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          libxsmm_generator_vcvtneps2bf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              start_vreg_sum + im, start_vreg_sum + im,
              30, 31,
              2, 3, 0, 0);
        } if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          libxsmm_generator_vcvtf32_to_hf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              start_vreg_sum + im, start_vreg_sum + im, cvt_vreg_aux0, cvt_vreg_aux1, cvt_vreg_aux2, cvt_vreg_aux3, cvt_mask_aux0, cvt_mask_aux1, cvt_mask_aux2);
        } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, start_vreg_sum + im, start_vreg_sum + im, 0,
                                                                  (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
        } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
            libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                start_vreg_sum + im, start_vreg_sum + im,
                tmp_vreg, cvt_vreg_aux0,
                2, 3, 0);
          } else {
           libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, start_vreg_sum + im, start_vreg_sum + im );
          }
        }
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                          ( ((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) )&& (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && !((use_m_masking > 0) && (im == peeled_m_trips - 1)) ) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_out,
                                          i_gp_reg_mapping->gp_reg_reduced_elts,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          im * vlen * i_micro_kernel_config->datatype_size_out,
                                          vname_out,
                                          start_vreg_sum + im, ((im == peeled_m_trips -1) && (use_m_masking > 0)) ? 1 : 0, ((im == peeled_m_trips -1) && (use_m_masking > 0)) ? mask_reg_out : 0, 1 );

      }

      if ( compute_squared_vals_reduce > 0 ) {
        if (reduce_on_output > 0) {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                            ( ((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)))&& (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && !((use_m_masking > 0) && (im == peeled_m_trips - 1)) ) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            im * vlen * i_micro_kernel_config->datatype_size_out,
                                            vname_out,
                                            tmp_vreg, ((im == peeled_m_trips -1) && (use_m_masking > 0)) ? 1 : 0, ((im == peeled_m_trips -1) && (use_m_masking > 0)) ? mask_reg_out : 0, 0 );
          if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
          } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                tmp_vreg, tmp_vreg, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
          } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
          } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
          }
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               reduce_instr,
                                               i_micro_kernel_config->vector_name,
                                               tmp_vreg, start_vreg_sum2 + im, start_vreg_sum2 + im);
        }
        if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          libxsmm_generator_vcvtneps2bf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              start_vreg_sum2 + im, start_vreg_sum2 + im,
              30, 31,
              2, 3, 0, 0);
        } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          libxsmm_generator_vcvtf32_to_hf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              start_vreg_sum2 + im, start_vreg_sum2 + im, cvt_vreg_aux0, cvt_vreg_aux1, cvt_vreg_aux2, cvt_vreg_aux3, cvt_mask_aux0, cvt_mask_aux1, cvt_mask_aux2);
        } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, start_vreg_sum2 + im, start_vreg_sum2 + im, 0,
                                                                  (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
        } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
            libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                start_vreg_sum2 + im, start_vreg_sum2 + im,
                tmp_vreg, cvt_vreg_aux0,
                2, 3, 0);
          } else {
           libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, start_vreg_sum2 + im, start_vreg_sum2 + im );
          }
        }
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                          ( ((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT))) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && !((use_m_masking > 0) && (im == peeled_m_trips - 1)) ) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_out,
                                          i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          im * vlen * i_micro_kernel_config->datatype_size_out,
                                          vname_out,
                                          start_vreg_sum2 + im, ((im == peeled_m_trips -1) && (use_m_masking > 0)) ? 1 : 0, ((im == peeled_m_trips -1) && (use_m_masking > 0)) ? mask_reg_out : 0, 1 );
      }
    }
  }

  if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
    libxsmm_generator_vcvtneps2bf8_avx512_clean_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
  } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
    if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
      libxsmm_generator_vcvtneps2bf16_avx512_clean_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
    }
  } else if ((LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))) {
    libxsmm_generator_vcvt_hf8_tofrom_f32_avx512_clean_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_reduce_rows_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {

  unsigned int i = 0, im, m, n, m_trips, n_trips, n_full_trips, use_m_masking, use_n_masking, mask_in_count, mask_out_count, n_cols_load = 16, compute_squared_vals_reduce, compute_plain_vals_reduce;
  unsigned int reduce_instr = 0, absmax_instr = 0;
  unsigned int aux_vreg = 24;
  unsigned int tmp_vreg = 24;
  unsigned int vlen = 16;
  unsigned int max_m_unrolling = 16;
  unsigned int m_unroll_factor = 16;
  unsigned int peeled_m_trips = 0;
  unsigned int m_trips_loop = 1;
  unsigned int shuf_mask = 0x4e;
  unsigned int cvt_vreg_aux0 = 30, cvt_vreg_aux1 = 31, cvt_vreg_aux2 = 27, cvt_vreg_aux3 = 28;
  unsigned int cvt_mask_aux0 = 6, cvt_mask_aux1 = 5, cvt_mask_aux2 = 4;
  char  vname_in = i_micro_kernel_config->vector_name;
  char  vname_out = i_micro_kernel_config->vector_name;
  char  vname_comp = i_micro_kernel_config->vector_name;
  unsigned int vmove_instruction_in = i_micro_kernel_config->vmove_instruction_in;
  unsigned int vmove_instruction_out = i_micro_kernel_config->vmove_instruction_out;
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

  int bf16_accum = LIBXSMM_X86_GP_REG_RCX;

  libxsmm_generator_reduce_set_lp_vlen_vname_vmove_x86( io_generated_code, i_micro_kernel_config, i_mateltwise_desc, &vlen, &vname_in, &vname_out, &vmove_instruction_in, &vmove_instruction_out );

  if ( (io_generated_code->arch >= LIBXSMM_X86_AVX2) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ) {
    shuf_mask = 0x01;
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
    if (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)) {
      reduce_instr = LIBXSMM_X86_INSTR_VADDPD;
    } else {
      reduce_instr = LIBXSMM_X86_INSTR_VADDPS;
    }
  } else if ( flag_reduce_op_max > 0 || flag_reduce_op_absmax > 0) {
    if (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)) {
      reduce_instr = LIBXSMM_X86_INSTR_VMAXPD;
      if (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) {
        absmax_instr = LIBXSMM_X86_INSTR_VRANGEPD;
      } else {
        absmax_instr = LIBXSMM_X86_INSTR_VPANDQ;
      }
    } else {
      reduce_instr = LIBXSMM_X86_INSTR_VMAXPS;
      if (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) {
        absmax_instr = LIBXSMM_X86_INSTR_VRANGEPS;
      } else {
        absmax_instr = LIBXSMM_X86_INSTR_VPANDQ;
      }
    }
  } else if ( flag_reduce_op_min > 0 ) {
    if (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)) {
      reduce_instr = LIBXSMM_X86_INSTR_VMINPD;
    } else {
      reduce_instr = LIBXSMM_X86_INSTR_VMINPS;
    }
  } else {
    /* This should not happen */
    printf("Only supported reduction OPs are ADD and MAX for this reduce kernel\n");
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  if ((compute_squared_vals_reduce > 0) && !(reduce_instr == LIBXSMM_X86_INSTR_VADDPS || reduce_instr == LIBXSMM_X86_INSTR_VADDPD)) {
    /* This should not happen */
    printf("Support for squares's reduction only when reduction OP is ADD\n");
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in                   = LIBXSMM_X86_GP_REG_R8;
  i_gp_reg_mapping->gp_reg_reduced_elts         = LIBXSMM_X86_GP_REG_R9;
  i_gp_reg_mapping->gp_reg_reduced_elts_squared = LIBXSMM_X86_GP_REG_R10;
  i_gp_reg_mapping->gp_reg_m_loop               = LIBXSMM_X86_GP_REG_R11;
  i_gp_reg_mapping->gp_reg_n_loop               = LIBXSMM_X86_GP_REG_RAX;
  bf16_accum = LIBXSMM_X86_GP_REG_RCX;
  libxsmm_generator_meltw_getval_stack_var( io_generated_code, LIBXSMM_MELTW_STACK_VAR_SCRATCH_PTR, bf16_accum );

  /* load the input pointer and output pointer */
  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      32,
      i_gp_reg_mapping->gp_reg_in,
      0 );

  if ( compute_plain_vals_reduce > 0 ) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
       i_gp_reg_mapping->gp_reg_param_struct,
       LIBXSMM_X86_GP_REG_UNDEF, 0,
       64,
       i_gp_reg_mapping->gp_reg_reduced_elts,
       0 );
    if ( compute_squared_vals_reduce > 0 ) {
      unsigned int result_size = i_mateltwise_desc->n * i_micro_kernel_config->datatype_size_out;
      libxsmm_x86_instruction_alu_reg(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduced_elts, i_gp_reg_mapping->gp_reg_reduced_elts_squared);
      libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_mapping->gp_reg_reduced_elts_squared, result_size);
    }
  } else {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
       i_gp_reg_mapping->gp_reg_param_struct,
       LIBXSMM_X86_GP_REG_UNDEF, 0,
       64,
       i_gp_reg_mapping->gp_reg_reduced_elts_squared,
       0 );
  }

  /* In this case we do not support the algorithm with "on the fly transpose" */
  if ((i_mateltwise_desc->m >= 256) || (flag_reduce_op_absmax > 0) || io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX || (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP))) {
    unsigned int reg_sum = 15, reg_sum_squared = 14;
    unsigned int cur_vreg;
    unsigned int mask_out = 13;
    unsigned int available_vregs = 13;
    unsigned int mask_reg = 0, mask_reg_in = 0;
    unsigned int arch_has_maskregs_and_prec_f64 = 0;
    unsigned int arch_avx512_and_large_m = 0;

    if (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)) {
      vlen = 4;
    } else {
      vlen = 8;
    }

    /* Reconfig if F64 and arch > avx2 */
    if ((io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) &&
        (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP))) {
      reg_sum = 31;
      reg_sum_squared = 30;
      available_vregs = 30;
      arch_has_maskregs_and_prec_f64 = 1;
      if (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) {
        vlen = 8;
      } else {
      }
    } else if ((io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (i_mateltwise_desc->m >= 256 || flag_reduce_op_absmax > 0)) {
      /* Reconfig if large m and arch >= avx512 */
      cvt_vreg_aux0 = 31; cvt_vreg_aux1 = 30; cvt_vreg_aux2 = 29; cvt_vreg_aux3 = 28;
      cvt_mask_aux0 = 7; cvt_mask_aux1 = 6; cvt_mask_aux2 = 5;
      reg_sum = 27;
      reg_sum_squared = 26;
      available_vregs = 26;
      arch_avx512_and_large_m = 1;
      if (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) {
        vlen = 16;
        vname_comp = 'z';
      } else {
        vlen = 8;
        vname_comp = 'y';
      }
    }

    m                 = i_mateltwise_desc->m;
    n                 = i_mateltwise_desc->n;
    use_m_masking     = ( m % vlen == 0 ) ? 0 : 1;
    m_trips           = ( m+vlen-1 )/ vlen;
    m_unroll_factor   = m_trips;
    peeled_m_trips    = 0;
    m_trips_loop      = 1;
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

    if ((use_m_masking > 0) && ( flag_reduce_op_max > 0 || flag_reduce_op_absmax > 0  || flag_reduce_op_min > 0 )) {
      aux_vreg = available_vregs - 1;
      available_vregs--;
      if ((flag_reduce_op_absmax > 0) && (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX)) {
        if (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)) {
          libxsmm_generator_load_vreg_signmask(io_generated_code, vname_comp, LIBXSMM_X86_GP_REG_RAX, aux_vreg, LIBXSMM_DATATYPE_F64);
        } else {
          libxsmm_generator_load_vreg_signmask(io_generated_code, vname_comp, LIBXSMM_X86_GP_REG_RAX, aux_vreg, LIBXSMM_DATATYPE_F32);
        }
      } else {
        if (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)) {
          libxsmm_generator_load_vreg_infinity_double(io_generated_code, vname_comp, LIBXSMM_X86_GP_REG_RAX, aux_vreg, (flag_reduce_op_min > 0) ? 1 : 0);
        } else {
          libxsmm_generator_load_vreg_infinity(io_generated_code, vname_comp, LIBXSMM_X86_GP_REG_RAX, aux_vreg, (flag_reduce_op_min > 0) ? 1 : 0);
        }
      }
    } else if (flag_reduce_op_absmax > 0) {
      aux_vreg = available_vregs - 1;
      available_vregs--;
      if (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)) {
        libxsmm_generator_load_vreg_signmask(io_generated_code, vname_comp, LIBXSMM_X86_GP_REG_RAX, aux_vreg, LIBXSMM_DATATYPE_F64);
      } else {
        libxsmm_generator_load_vreg_signmask(io_generated_code, vname_comp, LIBXSMM_X86_GP_REG_RAX, aux_vreg, LIBXSMM_DATATYPE_F32);
      }
    }

    if (reduce_on_output > 0) {
      tmp_vreg = available_vregs - 1;
      available_vregs--;
    }

    if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
      libxsmm_generator_vcvtneps2bf8_avx512_prep_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
    } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
      if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
        libxsmm_generator_vcvtneps2bf16_avx512_prep_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
      }
    } else if ((LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))) {
      libxsmm_generator_vcvt_hf8_tofrom_f32_avx512_prep_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
    }

    if (arch_avx512_and_large_m > 0) {
      mask_out = 1;
      libxsmm_generator_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, mask_out, vlen - 1, LIBXSMM_DATATYPE_F32);
    } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
      cvt_vreg_aux0 = available_vregs - 1;
      cvt_vreg_aux1 = available_vregs - 2;
      available_vregs -= 2;
      mask_out = 1;
    } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
      mask_out = 1;
    } else {
      if (arch_has_maskregs_and_prec_f64 > 0) {
        mask_out = 1;
        libxsmm_generator_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, mask_out, vlen - 1, (libxsmm_datatype)libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ));
      } else {
        libxsmm_generator_initialize_avx_mask(io_generated_code, mask_out, 1, (libxsmm_datatype)libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ));
      }
    }

    /* Calculate input mask in case we see m_masking */
    if (use_m_masking == 1) {
      if (arch_has_maskregs_and_prec_f64 > 0 || arch_avx512_and_large_m > 0) {
        mask_reg = 2;
        libxsmm_generator_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, mask_reg, vlen - (m % vlen), (arch_has_maskregs_and_prec_f64 > 0) ? LIBXSMM_DATATYPE_F64 : LIBXSMM_DATATYPE_F32);
        mask_reg_in = mask_reg;
      } else {
        const int datatype = libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP);
        mask_reg = available_vregs-1;
        libxsmm_generator_initialize_avx_mask(io_generated_code, mask_reg, m % vlen, (libxsmm_datatype)datatype);
        available_vregs--;
        if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) || LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          mask_reg_in = m % vlen;
        } else {
          mask_reg_in = mask_reg;
        }
      }
    }

    if (n > 1) {
      /* open n loop */
      libxsmm_generator_mateltwise_header_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop );
    }

    /* Initialize accumulators to zero */
    if ( compute_plain_vals_reduce > 0 ) {
      if ( flag_reduce_op_add > 0 || flag_reduce_op_absmax > 0) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VXORPS,
                                                 i_micro_kernel_config->vector_name,
                                                 reg_sum, reg_sum, reg_sum );
      } else if ( flag_reduce_op_max > 0 || flag_reduce_op_min > 0 ) {
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            ( ((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && !((use_m_masking > 0) && ((m_trips == 1))) ) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            vname_in,
            reg_sum, ((use_m_masking == 1) && (m_trips == 1)) ? 1 : 0, ((use_m_masking == 1) && (m_trips == 1)) ? mask_reg_in : 0, 0);

        if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, reg_sum, reg_sum );
        } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              reg_sum, reg_sum, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
        } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, reg_sum, reg_sum );
        } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, reg_sum, reg_sum );
        }

        /* If we have remainder, then we want to blend in -INF for the zero'ed out entries */
        if ((use_m_masking == 1) && (m_trips == 1)) {
          unsigned int blend_vreg_mask_id = (arch_has_maskregs_and_prec_f64 > 0 || arch_avx512_and_large_m > 0) ? 0 : (mask_reg) << 4;
          unsigned int blend_mask_id = (arch_has_maskregs_and_prec_f64 > 0 || arch_avx512_and_large_m > 0) ? mask_reg : 0;
          unsigned int blend_instr = (arch_has_maskregs_and_prec_f64 > 0) ? LIBXSMM_X86_INSTR_VPBLENDMQ :
            ( (arch_avx512_and_large_m > 0) ? LIBXSMM_X86_INSTR_VBLENDMPS : (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)) ? LIBXSMM_X86_INSTR_VBLENDVPD : LIBXSMM_X86_INSTR_VBLENDVPS) ;
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8(io_generated_code, blend_instr, vname_comp, reg_sum, aux_vreg, reg_sum, blend_mask_id, 0, 0, blend_vreg_mask_id);
        }
      }
    }

    if ( compute_squared_vals_reduce > 0 ) {
      libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               LIBXSMM_X86_INSTR_VXORPS,
                                               i_micro_kernel_config->vector_name,
                                               reg_sum_squared, reg_sum_squared, reg_sum_squared );
    }

    if ( m_trips_loop >= 1 ) {
      libxsmm_generator_mateltwise_header_m_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_m_loop );
      for (im = 0; im < m_unroll_factor; im++) {
        cur_vreg = im % available_vregs;
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            ( ((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0))) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX)) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * vlen * i_micro_kernel_config->datatype_size_in,
            vname_in,
            cur_vreg, 0, 0, 0 );
        if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, cur_vreg, cur_vreg );
        } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              cur_vreg, cur_vreg, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
        } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, cur_vreg, cur_vreg );
        } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, cur_vreg, cur_vreg );
        }
        if ( compute_plain_vals_reduce > 0 ) {
          if ((io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (flag_reduce_op_absmax > 0)) {
            libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, absmax_instr,
                i_micro_kernel_config->vector_name, cur_vreg, reg_sum, reg_sum, 0, 0, 0, 3);
          } else {
            if (flag_reduce_op_absmax > 0) {
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, absmax_instr, i_micro_kernel_config->vector_name, cur_vreg, aux_vreg, cur_vreg );
            }
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 reduce_instr,
                                                 i_micro_kernel_config->vector_name,
                                                 cur_vreg, reg_sum, reg_sum );
          }
        }
        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)) ? LIBXSMM_X86_INSTR_VFMADD231PD: LIBXSMM_X86_INSTR_VFMADD231PS,
                                               i_micro_kernel_config->vector_name,
                                               cur_vreg, cur_vreg, reg_sum_squared );
        }
      }
      /* Adjust input pointer */
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_in,
          (long long)m_unroll_factor * vlen * i_micro_kernel_config->datatype_size_in);
      libxsmm_generator_mateltwise_footer_m_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_m_loop, m_trips_loop);
    }

    if (peeled_m_trips > 0) {
      for (im = 0; im < peeled_m_trips; im++) {
        cur_vreg = im % available_vregs;
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            ( ((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0))) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && !((use_m_masking > 0) && (im == (peeled_m_trips-1))) ) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * vlen * i_micro_kernel_config->datatype_size_in,
            vname_in,
            cur_vreg, ((use_m_masking == 1) && (im == (peeled_m_trips-1))) ? 1 : 0, ((use_m_masking == 1) && (im == (peeled_m_trips-1))) ? mask_reg_in : 0, 0 );
        if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, cur_vreg, cur_vreg );
        } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              cur_vreg, cur_vreg, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
        } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, cur_vreg, cur_vreg );
        } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, cur_vreg, cur_vreg );
        }

        /* If we have remainder, then we want to blend in -INF for the zero'ed out entries */
        if ((flag_reduce_op_max > 0 || flag_reduce_op_min > 0) && (use_m_masking == 1) && (im == (peeled_m_trips-1))) {
          unsigned int blend_vreg_mask_id = (arch_has_maskregs_and_prec_f64 > 0 || arch_avx512_and_large_m > 0) ? 0 : (mask_reg) << 4;
          unsigned int blend_mask_id = (arch_has_maskregs_and_prec_f64 > 0 || arch_avx512_and_large_m > 0) ? mask_reg : 0;
          unsigned int blend_instr = (arch_has_maskregs_and_prec_f64 > 0) ? LIBXSMM_X86_INSTR_VPBLENDMQ :
            ( (arch_avx512_and_large_m > 0) ? LIBXSMM_X86_INSTR_VBLENDMPS : (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)) ? LIBXSMM_X86_INSTR_VBLENDVPD : LIBXSMM_X86_INSTR_VBLENDVPS) ;
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8(io_generated_code, blend_instr, vname_comp, cur_vreg, aux_vreg, cur_vreg, blend_mask_id, 0, 0, blend_vreg_mask_id);
        }

        if ((io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (flag_reduce_op_absmax > 0)) {
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, absmax_instr,
              i_micro_kernel_config->vector_name, cur_vreg, reg_sum, reg_sum, 0, 0, 0, 3);
        } else {
          if (flag_reduce_op_absmax > 0) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, absmax_instr, i_micro_kernel_config->vector_name, cur_vreg, aux_vreg, cur_vreg );
          }
          if ( compute_plain_vals_reduce > 0 ) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 reduce_instr,
                                                 i_micro_kernel_config->vector_name,
                                                 cur_vreg, reg_sum, reg_sum );
          }
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)) ? LIBXSMM_X86_INSTR_VFMADD231PD: LIBXSMM_X86_INSTR_VFMADD231PS,
                                               i_micro_kernel_config->vector_name,
                                               cur_vreg, cur_vreg, reg_sum_squared );
        }
      }
    }
    if ( m_trips_loop >= 1 ) {
      /* Adjust input pointer */
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_sub_instruction,
          i_gp_reg_mapping->gp_reg_in,
          (long long)m_unroll_factor * m_trips_loop * vlen * i_micro_kernel_config->datatype_size_in);
    }

    /* Now last horizontal reduction and store of the result... */
    if ( compute_plain_vals_reduce > 0 ) {
      if (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)) {
        libxsmm_generator_hinstrpd_avx_avx512( io_generated_code, reduce_instr, reg_sum, 0, 1);
      } else if (arch_avx512_and_large_m > 0) {
        libxsmm_generator_hinstrps_avx512 ( io_generated_code, reduce_instr, reg_sum, 0, 1);
      } else {
        libxsmm_generator_hinstrps_avx( io_generated_code, reduce_instr, reg_sum, 0, 1);
      }
      if (reduce_on_output > 0) {
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                          vmove_instruction_out,
                                          i_gp_reg_mapping->gp_reg_reduced_elts,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          0,
                                          vname_out,
                                          tmp_vreg, 1, mask_out, 0  );
        if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
        } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              tmp_vreg, tmp_vreg, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
        } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
        } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
        }

        if ((io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (flag_reduce_op_absmax > 0)) {
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, absmax_instr,
              i_micro_kernel_config->vector_name, tmp_vreg, reg_sum, reg_sum, 0, 0, 0, 3);
        } else {
          if (flag_reduce_op_absmax > 0) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, absmax_instr, i_micro_kernel_config->vector_name, tmp_vreg, aux_vreg, tmp_vreg );
          }
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               reduce_instr,
                                               i_micro_kernel_config->vector_name,
                                               tmp_vreg, reg_sum, reg_sum);
        }
      }

      if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
        libxsmm_generator_vcvtneps2bf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
            reg_sum, reg_sum,
            cvt_vreg_aux0, cvt_vreg_aux1,
            cvt_mask_aux0, cvt_mask_aux1, 0, 0);
      } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
        libxsmm_generator_vcvtf32_to_hf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
            reg_sum, reg_sum, cvt_vreg_aux0, cvt_vreg_aux1, cvt_vreg_aux2, cvt_vreg_aux3, cvt_mask_aux0, cvt_mask_aux1, cvt_mask_aux2);
      } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
        libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, reg_sum, reg_sum, 0,
                                                                (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
      } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
        if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
          libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              reg_sum, reg_sum,
              cvt_vreg_aux0, cvt_vreg_aux1,
              cvt_mask_aux0, cvt_mask_aux1, 0);
        } else {
         libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, reg_sum, reg_sum);
        }
      }

      libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                        vmove_instruction_out,
                                        i_gp_reg_mapping->gp_reg_reduced_elts,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        0,
                                        vname_out,
                                        reg_sum, 1, mask_out, 1 );
    }

    if ( compute_squared_vals_reduce > 0 ) {
      if (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)) {
        libxsmm_generator_hinstrpd_avx_avx512( io_generated_code, reduce_instr, reg_sum_squared, 0, 1);
      } else if (arch_avx512_and_large_m > 0) {
        libxsmm_generator_hinstrps_avx512 ( io_generated_code, reduce_instr, reg_sum_squared, 0, 1);
      } else {
        libxsmm_generator_hinstrps_avx( io_generated_code, reduce_instr, reg_sum_squared, 0, 1);
      }
      if (reduce_on_output > 0) {
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                          vmove_instruction_out,
                                          i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          0,
                                          vname_out,
                                          tmp_vreg, 1, mask_out, 0  );
        if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
        } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              tmp_vreg, tmp_vreg, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
        } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
        } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
        }
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             reduce_instr,
                                             i_micro_kernel_config->vector_name,
                                             tmp_vreg, reg_sum_squared, reg_sum_squared);
      }

      if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
        libxsmm_generator_vcvtneps2bf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
            reg_sum_squared, reg_sum_squared,
            cvt_vreg_aux0, cvt_vreg_aux1,
            cvt_mask_aux0, cvt_mask_aux1, 0, 0);
      } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
        libxsmm_generator_vcvtf32_to_hf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
            reg_sum_squared, reg_sum_squared, cvt_vreg_aux0, cvt_vreg_aux1, cvt_vreg_aux2, cvt_vreg_aux3, cvt_mask_aux0, cvt_mask_aux1, cvt_mask_aux2);
      } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
        libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, reg_sum_squared, reg_sum_squared, 0,
                                                                (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
      } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
        if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
          libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              reg_sum_squared, reg_sum_squared,
              cvt_vreg_aux0, cvt_vreg_aux1,
              cvt_mask_aux0, cvt_mask_aux1, 0);
        } else {
         libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, reg_sum_squared, reg_sum_squared);
        }
      }

      libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                        vmove_instruction_out,
                                        i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        0,
                                        vname_out,
                                        reg_sum_squared, 1, mask_out, 1 );
    }

    if (n > 1) {
      /* Adjust input and output pointer */
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_in,
          (long long)i_mateltwise_desc->ldi *  i_micro_kernel_config->datatype_size_in);

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_reduced_elts,
          i_micro_kernel_config->datatype_size_out);
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_reduced_elts_squared,
          i_micro_kernel_config->datatype_size_out);
      }
      /* close n loop */
      libxsmm_generator_mateltwise_footer_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, n);
    }

    if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
      libxsmm_generator_vcvtneps2bf8_avx512_clean_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
    } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
      if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
        libxsmm_generator_vcvtneps2bf16_avx512_clean_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
      }
    } else if ((LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))) {
      libxsmm_generator_vcvt_hf8_tofrom_f32_avx512_clean_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
    }

    return;
  }

  /* We fully unroll in M dimension, calculate mask if there is remainder */
  m                 = i_mateltwise_desc->m;
  n                 = i_mateltwise_desc->n;
  use_m_masking     = ( m % vlen == 0 ) ? 0 : 1;
  use_n_masking     = ( n % vlen == 0 ) ? 0 : 1;
  m_trips           = ( m+vlen-1 )/ vlen;
  n_trips           = ( n+vlen-1 )/ vlen;
  n_full_trips      = ( n % vlen == 0 ) ? n_trips : n_trips-1;

  /* Calculate input mask in case we see m_masking */
  if (use_m_masking == 1) {
    /* Calculate mask reg 1 for input-reading */
    mask_in_count =  vlen - (m % vlen);
    libxsmm_generator_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, 1, mask_in_count, LIBXSMM_DATATYPE_F32);
  }

  /* Calculate output mask in case we see n_masking */
  if (use_n_masking == 1) {
    /* Calculate mask reg 2 for output-writing */
    mask_out_count = vlen - (n % vlen);
    libxsmm_generator_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, 2, mask_out_count, LIBXSMM_DATATYPE_F32);
  }

  if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
    libxsmm_generator_vcvtneps2bf8_avx512_prep_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
  } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
    if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
      libxsmm_generator_vcvtneps2bf16_avx512_prep_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
    }
  } else if ((LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))) {
    libxsmm_generator_vcvt_hf8_tofrom_f32_avx512_prep_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
  }

  /* move blend mask value to GP register and to mask register 7 */
  if (n != 1) {
    if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RAX, 0xf0 );
    } else {
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RAX, 0xff00 );
    }
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVW_GPR_LD, LIBXSMM_X86_GP_REG_RAX, 7 );
    if ((use_m_masking > 0) && ( flag_reduce_op_max > 0 ||  flag_reduce_op_min > 0 )) {
      libxsmm_generator_load_vreg_infinity(io_generated_code, i_micro_kernel_config->vector_name, LIBXSMM_X86_GP_REG_RAX, aux_vreg, (flag_reduce_op_min > 0) ? 1 : 0);
    }
  }

  if (n_full_trips >= 1) {
    if (n_full_trips > 1) {
      /* open n loop */
      libxsmm_generator_mateltwise_header_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop );
    }

    /* We fully unroll M loop here... */
    for (im = 0; im < m_trips; im++) {
      /* load 16 columns of input matrix */
      for (i = 0 ; i < vlen; i++) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            (((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT))) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && !((use_m_masking > 0) && (im == (m_trips-1))) ) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * vlen + i * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
            vname_in,
            i, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );
        if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, i, i );
        } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              i, i, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
        } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, i, i );
        } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, i, i );
        }
        if ((flag_reduce_op_max > 0 || flag_reduce_op_min > 0) && (use_m_masking > 0) && (im == m_trips-1)) {
          libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                       LIBXSMM_X86_INSTR_VBLENDMPS,
                                                       i_micro_kernel_config->vector_name,
                                                       i, aux_vreg, i, use_m_masking, 0 );
        }
      }

      for ( i = vlen; i < 16; i++) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VXORPS,
                                                  i_micro_kernel_config->vector_name,
                                                  i, i, i );

      }


      /* 1st stage */
      /* zmm0/zmm4; 4444 4444 4444 4444 / 0000 0000 0000 0000 -> zmm0: 4444 4444 0000 0000 */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     4, 0, 16, 7, 0 );
        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             4, 0, 17, shuf_mask );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              0, 0, 18 );

        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              4, 4, 19 );


        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     19, 18, 20, 7, 0 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             19, 18, 21, shuf_mask );
      }

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             reduce_instr,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 0 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             reduce_instr,
                                             i_micro_kernel_config->vector_name,
                                             20, 21, 24 );
      }

      /* zmm8/zmm12; cccc cccc cccc cccc / 8888 8888 8888 8888 -> zmm8: cccc cccc 8888 8888 */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                      LIBXSMM_X86_INSTR_VBLENDMPS,
                                                      i_micro_kernel_config->vector_name,
                                                      12, 8, 16, 7, 0 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                              LIBXSMM_X86_INSTR_VSHUFF64X2,
                                              i_micro_kernel_config->vector_name,
                                              12, 8, 17, shuf_mask );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              8, 8, 18 );

        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              12, 12, 19 );


        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                      LIBXSMM_X86_INSTR_VBLENDMPS,
                                                      i_micro_kernel_config->vector_name,
                                                      19, 18, 20, 7, 0 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                              LIBXSMM_X86_INSTR_VSHUFF64X2,
                                              i_micro_kernel_config->vector_name,
                                              19, 18, 21, shuf_mask );
      }

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              reduce_instr,
                                              i_micro_kernel_config->vector_name,
                                              16, 17, 8 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              reduce_instr,
                                              i_micro_kernel_config->vector_name,
                                              20, 21, 28 );
      }

      /* zmm1/zmm5; 5555 5555 5555 5555 / 1111 1111 1111 1111 -> zmm1: 5555 5555 1111 1111 */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     5, 1, 16, 7, 0 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             5, 1, 17, shuf_mask );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              1, 1, 18 );

        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              5, 5, 19 );


        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     19, 18, 20, 7, 0 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             19, 18, 21, shuf_mask );
      }

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             reduce_instr,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 1 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             reduce_instr,
                                             i_micro_kernel_config->vector_name,
                                             20, 21, 25 );
      }

      /* zmm9/zmm13; dddd dddd dddd dddd / 9999 9999 9999 9999 -> zmm9: dddd dddd 9999 9999 */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     13, 9, 16, 7, 0 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             13, 9, 17, shuf_mask );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              9, 9, 18 );

        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              13, 13, 19 );


        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     19, 18, 20, 7, 0 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             19, 18, 21, shuf_mask );
      }

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             reduce_instr,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 9 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             reduce_instr,
                                             i_micro_kernel_config->vector_name,
                                             20, 21, 29 );
      }

      /* zmm2/zmm6; 6666 6666 6666 6666 / 2222 2222 2222 2222 -> zmm2: 6666 6666 2222 2222 */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     6, 2, 16, 7, 0 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             6, 2, 17, shuf_mask );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              2, 2, 18 );

        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              6, 6, 19 );


        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     19, 18, 20, 7, 0 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             19, 18, 21, shuf_mask );

      }

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             reduce_instr,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 2 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             reduce_instr,
                                             i_micro_kernel_config->vector_name,
                                             20, 21, 26 );
      }

      /* zmm10/zmm14; eeee eeee eeee eeee / aaaa aaaa aaaa aaaa -> zmm10: eeee eeee aaaa aaaa */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     14, 10, 16, 7, 0 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             14, 10, 17, shuf_mask );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              10, 10, 18 );

        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              14, 14, 19 );


        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     19, 18, 20, 7, 0 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             19, 18, 21, shuf_mask );
      }

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             reduce_instr,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 10 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             reduce_instr,
                                             i_micro_kernel_config->vector_name,
                                             20, 21, 30 );
      }

      /* zmm3/zmm7; 7777 7777 7777 7777 / 3333 3333 3333 3333  -> zmm3: 7777 7777 3333 3333 */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     7, 3, 16, 7, 0 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             7, 3, 17, shuf_mask );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              3, 3, 18 );

        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              7, 7, 19 );


        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     19, 18, 20, 7, 0 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             19, 18, 21, shuf_mask );
      }

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             reduce_instr,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 3 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             reduce_instr,
                                             i_micro_kernel_config->vector_name,
                                             20, 21, 27 );
      }

      /* zmm11/zmm15; ffff ffff ffff ffff / bbbb bbbb bbbb bbbb  -> zmm11: ffff ffff bbbb bbbb */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     15, 11, 16, 7, 0 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             15, 11, 17, shuf_mask );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              11, 11, 18 );

        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              15, 15, 19 );


        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     19, 18, 20, 7, 0 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             19, 18, 21, shuf_mask );
      }

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             reduce_instr,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 11 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             reduce_instr,
                                             i_micro_kernel_config->vector_name,
                                             20, 21, 31 );
      }

      /* 2nd stage */
      /* zmm0/zmm8; 4444 4444 0000 0000 / cccc cccc 8888 8888  -> zmm0: cccc 8888 4444 0000 */
      if (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) {
        if ( compute_plain_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  8, 0, 16, 0x88 );

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  8, 0, 17, 0xdd );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  reduce_instr,
                                                  i_micro_kernel_config->vector_name,
                                                  16, 17, 0 );
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  28, 24, 20, 0x88 );

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  28, 24, 21, 0xdd );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  reduce_instr,
                                                  i_micro_kernel_config->vector_name,
                                                  20, 21, 24 );
        }

        /* zmm1/zmm9; 5555 5555 1111 1111 / dddd dddd 9999 9999  -> zmm1: dddd 9999 5555 1111 */
        if ( compute_plain_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  9, 1, 16, 0x88 );

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  9, 1, 17, 0xdd );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  reduce_instr,
                                                  i_micro_kernel_config->vector_name,
                                                  16, 17, 1 );
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  29, 25, 20, 0x88 );

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  29, 25, 21, 0xdd );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  reduce_instr,
                                                  i_micro_kernel_config->vector_name,
                                                  20, 21, 25 );
        }

        /* zmm2/zmm10; 6666 6666 2222 2222 / eeee eeee aaaa aaaa  -> zmm2: eeee aaaa 6666 2222 */
        if ( compute_plain_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  10, 2, 16, 0x88 );

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  10, 2, 17, 0xdd );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  reduce_instr,
                                                  i_micro_kernel_config->vector_name,
                                                  16, 17, 2 );
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  30, 26, 20, 0x88 );

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  30, 26, 21, 0xdd );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  reduce_instr,
                                                  i_micro_kernel_config->vector_name,
                                                  20, 21, 26 );
        }

        /* zmm3/zmm11:  7777 7777 3333 3333 / ffff ffff bbbb bbbb  -> zmm3: ffff bbbb 7777 3333 */
        if ( compute_plain_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  11, 3, 16, 0x88 );

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  11, 3, 17, 0xdd );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  reduce_instr,
                                                  i_micro_kernel_config->vector_name,
                                                  16, 17, 3 );
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  31, 27, 20, 0x88 );

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  31, 27, 21, 0xdd );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  reduce_instr,
                                                  i_micro_kernel_config->vector_name,
                                                  20, 21, 27 );
        }
      }
      /* 3rd stage */
      /* zmm0/zmm1; cccc 8888 4444 0000 / dddd 9999 5555 1111  -> zmm0: ddcc 9988 5544 1100 */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 1, 0, 16, 0x44 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 1, 0, 17, 0xee );

        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 reduce_instr,
                                                 i_micro_kernel_config->vector_name,
                                                 16, 17, 0 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 25, 24, 20, 0x44 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 25, 24, 21, 0xee );

        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 reduce_instr,
                                                 i_micro_kernel_config->vector_name,
                                                 20, 21, 24 );
      }

      /* zmm2/zmm3; eeee aaaa 6666 2222 / ffff bbbb 7777 3333  -> zmm2: ffee bbaa 7766 3322 */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 3, 2, 16, 0x44 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 3, 2, 17, 0xee );

        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 reduce_instr,
                                                 i_micro_kernel_config->vector_name,
                                                 16, 17, 2 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 27, 26, 20, 0x44 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 27, 26, 21, 0xee );

        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 reduce_instr,
                                                 i_micro_kernel_config->vector_name,
                                                 20, 21, 26 );
      }

      /* 4th stage */
      /* zmm0/zmm2; ddcc 9988 5544 1100 / ffee bbaa 7766 3322  -> zmm0: fedc ba98 7654 3210 */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 2, 0, 16, 0x88 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 2, 0, 17, 0xdd );

        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 reduce_instr,
                                                 i_micro_kernel_config->vector_name,
                                                 16, 17, 0 );

        /* Update the running reduction result */
        if (im == 0 && reduce_on_output == 0) {
          if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              0,
                                              i_micro_kernel_config->vector_name,
                                              0, 0, 0, 1 );
            libxsmm_generator_vcvtneps2bf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                0, 0,
                3, 4,
                3, 4, 0, 0);
          } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              0,
                                              i_micro_kernel_config->vector_name,
                                              0, 0, 0, 1 );
            libxsmm_generator_vcvtf32_to_hf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                0, 0, cvt_vreg_aux0, cvt_vreg_aux1, cvt_vreg_aux2, cvt_vreg_aux3, cvt_mask_aux0, cvt_mask_aux1, cvt_mask_aux2);
          } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              0,
                                              i_micro_kernel_config->vector_name,
                                              0, 0, 0, 1 );
            libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, 0, 0, 0,
                                                                    (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
          } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                                i_micro_kernel_config->instruction_set,
                                                LIBXSMM_X86_INSTR_VMOVUPS,
                                                bf16_accum,
                                                LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                0,
                                                i_micro_kernel_config->vector_name,
                                                0, 0, 0, 1 );
            if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
               libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                  0, 0,
                  3, 4,
                  3, 4, 0);
            } else {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, 0, 0 );
            }
          }
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            (((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT))) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX)) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            0,
                                            vname_out,
                                            0, 0, 0, 1 );
        } else if (im == 0 && reduce_on_output > 0) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            (((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT))) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX)) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            0,
                                            vname_out,
                                            3, 0, 1, 0 );
          if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, 3, 3 );
          } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                3, 3, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
          } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, 3, 3 );
          } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, 3, 3 );
          }
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, reduce_instr, i_micro_kernel_config->vector_name, 3, 0, 0);

          if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              0,
                                              i_micro_kernel_config->vector_name,
                                              0, 0, 0, 1 );
            libxsmm_generator_vcvtneps2bf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                0, 0,
                3, 4,
                3, 4, 0, 0);
          } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              0,
                                              i_micro_kernel_config->vector_name,
                                              0, 0, 0, 1 );
            libxsmm_generator_vcvtf32_to_hf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                0, 0, cvt_vreg_aux0, cvt_vreg_aux1, cvt_vreg_aux2, cvt_vreg_aux3, cvt_mask_aux0, cvt_mask_aux1, cvt_mask_aux2);
          } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              0,
                                              i_micro_kernel_config->vector_name,
                                              0, 0, 0, 1 );
            libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, 0, 0, 0,
                                                                    (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
          } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              0,
                                              i_micro_kernel_config->vector_name,
                                              0, 0, 0, 1 );
            if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
              libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                  0, 0,
                  3, 4,
                  3, 4, 0);
            } else {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, 0, 0 );
            }
          }
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            (((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT))) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX)) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            0,
                                            vname_out,
                                            0, 0, 0, 1 );

        } else if ( im == (m_trips-1) ) {
          if ((LIBXSMM_DATATYPE_F16  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_HF8  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_BF8  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVUPS,
                bf16_accum,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                0,
                i_micro_kernel_config->vector_name,
                1, 0, 1, 0 );
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                vmove_instruction_out,
                i_gp_reg_mapping->gp_reg_reduced_elts,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                0,
                vname_out,
                1, 0, 1, 0 );
          }
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  reduce_instr,
                                                  i_micro_kernel_config->vector_name,
                                                  1, 0, 0 );
          if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_vcvtneps2bf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                0, 0,
                3, 4,
                3, 4, 0, 0);
          } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_vcvtf32_to_hf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                0, 0, cvt_vreg_aux0, cvt_vreg_aux1, cvt_vreg_aux2, cvt_vreg_aux3, cvt_mask_aux0, cvt_mask_aux1, cvt_mask_aux2);
          } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, 0, 0, 0,
                                                                        (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
          } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
              libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                  0, 0,
                  3, 4,
                  3, 4, 0);
            } else {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, 0, 0 );
            }
          }

          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            (((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT))) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX)) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            0,
                                            vname_out,
                                            0, 0, 0, 1 );
        } else {
          if ((LIBXSMM_DATATYPE_F16  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_HF8  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_BF8  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVUPS,
              bf16_accum,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              0,
              i_micro_kernel_config->vector_name,
              1, 0, 1, 0 );
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                vmove_instruction_out,
                i_gp_reg_mapping->gp_reg_reduced_elts,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                0,
                vname_out,
                1, 0, 1, 0 );
          }
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  reduce_instr,
                                                  i_micro_kernel_config->vector_name,
                                                  1, 0, 0 );
          if ((LIBXSMM_DATATYPE_F16  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_HF8  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_BF8  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            LIBXSMM_X86_INSTR_VMOVUPS,
                                            bf16_accum,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            0,
                                            i_micro_kernel_config->vector_name,
                                            0, 0, 0, 1 );
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              vmove_instruction_out,
                                              i_gp_reg_mapping->gp_reg_reduced_elts,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              0,
                                              vname_out,
                                              0, 0, 0, 1 );
          }
        }
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 26, 24, 20, 0x88 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 26, 24, 21, 0xdd );

        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 reduce_instr,
                                                 i_micro_kernel_config->vector_name,
                                                 20, 21, 24 );

        /* Update the running reduction result */
        if (im == 0 && reduce_on_output == 0) {
          if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              64,
                                              i_micro_kernel_config->vector_name,
                                              24, 0, 0, 1 );
            libxsmm_generator_vcvtneps2bf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                24, 24,
                25, 26,
                3, 4, 0, 0);
          } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              64,
                                              i_micro_kernel_config->vector_name,
                                              24, 0, 0, 1 );
            libxsmm_generator_vcvtf32_to_hf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                24, 24, cvt_vreg_aux0, cvt_vreg_aux1, cvt_vreg_aux2, cvt_vreg_aux3, cvt_mask_aux0, cvt_mask_aux1, cvt_mask_aux2);
          } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              64,
                                              i_micro_kernel_config->vector_name,
                                              24, 0, 0, 1 );
            libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, 24, 24, 0,
                                                                    (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
          } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              64,
                                              i_micro_kernel_config->vector_name,
                                              24, 0, 0, 1 );
            if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
              libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                  24, 24,
                  25, 26,
                  3, 4, 0);
            } else {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, 24, 24 );
            }
          }
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            (((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT))) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX)) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            0,
                                            vname_out,
                                            24, 0, 0, 1 );
        } else if (im == 0 && reduce_on_output > 0) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            (((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX)) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            0,
                                            vname_out,
                                            25, 0, 1, 0 );
          if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, 25, 25 );
          } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                25, 25, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
          } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, 25, 25 );
          } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, 25, 25 );
          }
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, reduce_instr, i_micro_kernel_config->vector_name, 25, 24, 24);

          if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              64,
                                              i_micro_kernel_config->vector_name,
                                              24, 0, 0, 1 );
            libxsmm_generator_vcvtneps2bf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                24, 24,
                25, 26,
                3, 4, 0, 0);
          } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              64,
                                              i_micro_kernel_config->vector_name,
                                              24, 0, 0, 1 );
            libxsmm_generator_vcvtf32_to_hf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                24, 24, cvt_vreg_aux0, cvt_vreg_aux1, cvt_vreg_aux2, cvt_vreg_aux3, cvt_mask_aux0, cvt_mask_aux1, cvt_mask_aux2);
          } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              64,
                                              i_micro_kernel_config->vector_name,
                                              24, 0, 0, 1 );
            libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, 24, 24, 0,
                                                                    (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
          } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              64,
                                              i_micro_kernel_config->vector_name,
                                               24, 0, 0, 1 );
            if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
              libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                   24, 24,
                   25, 26,
                   3, 4, 0);
            } else {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, 24, 24 );
            }
          }
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            (((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT))) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX)) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            0,
                                            vname_out,
                                            24, 0, 0, 1 );

        } else if ( im == (m_trips-1) ) {
          if ((LIBXSMM_DATATYPE_F16  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_BF8  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVUPS,
                bf16_accum,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                64,
                i_micro_kernel_config->vector_name,
                25, 0, 1, 0 );
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                vmove_instruction_out,
                i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                0,
                vname_out,
                25, 0, 1, 0 );
          }
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  reduce_instr,
                                                  i_micro_kernel_config->vector_name,
                                                  25, 24, 24 );
          if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_vcvtneps2bf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                24, 24,
                25, 26,
                3, 4, 0, 0);
          } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_vcvtf32_to_hf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                24, 24, cvt_vreg_aux0, cvt_vreg_aux1, cvt_vreg_aux2, cvt_vreg_aux3, cvt_mask_aux0, cvt_mask_aux1, cvt_mask_aux2);
          } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, 24, 24, 0,
                                                                    (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
          } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
              libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                  24, 24,
                  25, 26,
                  3, 4, 0);
            } else {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, 24, 24 );
            }
          }
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            (((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT))) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX)) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            0,
                                            vname_out,
                                            24, 0, 0, 1 );
        } else {
          if ((LIBXSMM_DATATYPE_F16  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_HF8  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVUPS,
              bf16_accum,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              64,
              i_micro_kernel_config->vector_name,
              25, 0, 1, 0 );
          }else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                vmove_instruction_out,
                i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                0,
                vname_out,
                25, 0, 1, 0 );
          }
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  reduce_instr,
                                                  i_micro_kernel_config->vector_name,
                                                  25, 24, 24 );
          if ((LIBXSMM_DATATYPE_F16  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_HF8  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_BF8  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              64,
                                              i_micro_kernel_config->vector_name,
                                              24, 0, 0, 1 );
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              vmove_instruction_out,
                                              i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              0,
                                              vname_out,
                                              24, 0, 0, 1 );
          }
        }
      }
    }

    if ((n_full_trips >  1) || (n % vlen != 0)) {
      /* Adjust input and output pointer */
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_in,
          (long long)vlen * i_mateltwise_desc->ldi *  i_micro_kernel_config->datatype_size_in);

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_reduced_elts,
          (long long)vlen * i_micro_kernel_config->datatype_size_out);
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_reduced_elts_squared,
          (long long)vlen * i_micro_kernel_config->datatype_size_out);
      }
    }

    if (n_full_trips > 1) {
      /* close n loop */
      libxsmm_generator_mateltwise_footer_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, n_full_trips);
    }
  }

  /* In this case we load only partial number of columns */
  n_cols_load = n % vlen;
  im = 0;
  /* Special case when we reduce as single column */
  if (n == 1) {
    unsigned int reg_sum = 2, reg_sum_squared = 3;
    unsigned int cur_vreg;
    aux_vreg = 3;

    /* Initialize accumulators to zero */
    if ( compute_plain_vals_reduce > 0 ) {
      if ( flag_reduce_op_add > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VXORPS,
                                                 i_micro_kernel_config->vector_name,
                                                 reg_sum, reg_sum, reg_sum );
      } else if ( flag_reduce_op_max > 0 || flag_reduce_op_min > 0 ) {
        if ((use_m_masking > 0) && ( flag_reduce_op_max > 0 || flag_reduce_op_min > 0 )) {
          libxsmm_generator_load_vreg_infinity(io_generated_code, i_micro_kernel_config->vector_name, LIBXSMM_X86_GP_REG_RAX, aux_vreg, (flag_reduce_op_min > 0) ? 1 : 0);
        }
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            vname_in,
            reg_sum, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0);
        if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, reg_sum, reg_sum );
        } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              reg_sum, reg_sum, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
        } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, reg_sum, reg_sum );
        } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, reg_sum, reg_sum );
        }
        if ((flag_reduce_op_max > 0 || flag_reduce_op_min > 0) && (use_m_masking > 0) && (im == m_trips-1)) {
          libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                       LIBXSMM_X86_INSTR_VBLENDMPS,
                                                       i_micro_kernel_config->vector_name,
                                                       reg_sum, aux_vreg, reg_sum, use_m_masking, 0 );
        }
      }
    }

    if ( compute_squared_vals_reduce > 0 ) {
      libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               LIBXSMM_X86_INSTR_VXORPS,
                                               i_micro_kernel_config->vector_name,
                                               reg_sum_squared, reg_sum_squared, reg_sum_squared );
    }

    for (im = 0; im < m_trips; im++) {
      cur_vreg = im % 28 + 4;
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          vmove_instruction_in,
          i_gp_reg_mapping->gp_reg_in,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          im * vlen * i_micro_kernel_config->datatype_size_in,
          vname_in,
          cur_vreg, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );

      if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
        libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, cur_vreg, cur_vreg );
      } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
        libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
            cur_vreg, cur_vreg, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
      } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, cur_vreg, cur_vreg );
      } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
        libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, cur_vreg, cur_vreg );
      }

      if ( (flag_reduce_op_max > 0 || flag_reduce_op_min > 0) && (im == m_trips-1) && (use_m_masking > 0)) {
        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     cur_vreg, aux_vreg, cur_vreg, use_m_masking, 0 );
      }

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             reduce_instr,
                                             i_micro_kernel_config->vector_name,
                                             cur_vreg, reg_sum, reg_sum );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             LIBXSMM_X86_INSTR_VFMADD231PS,
                                             i_micro_kernel_config->vector_name,
                                             cur_vreg, cur_vreg, reg_sum_squared );
      }
    }

    /* Now last horizontal reduction and store of the result... */
    if ( compute_plain_vals_reduce > 0 ) {
      libxsmm_generator_hinstrps_avx512( io_generated_code, reduce_instr, reg_sum, 0, 1);
      if (reduce_on_output > 0) {
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                          vmove_instruction_out,
                                          i_gp_reg_mapping->gp_reg_reduced_elts,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          0,
                                          vname_out,
                                          tmp_vreg, 1, 2, 0  );
        if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
        } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
            tmp_vreg, tmp_vreg, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
        } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
        } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
        }
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             reduce_instr,
                                             i_micro_kernel_config->vector_name,
                                             tmp_vreg, reg_sum, reg_sum);
      }
      if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
        libxsmm_generator_vcvtneps2bf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
            reg_sum, reg_sum,
            0, 1,
            3, 4, 0, 0);
      } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
        libxsmm_generator_vcvtf32_to_hf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
            reg_sum, reg_sum, cvt_vreg_aux0, cvt_vreg_aux1, cvt_vreg_aux2, cvt_vreg_aux3, cvt_mask_aux0, cvt_mask_aux1, cvt_mask_aux2);
      } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
        libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, reg_sum, reg_sum, 0,
                                                                    (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
      } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
        if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
          libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              reg_sum, reg_sum,
              0, 1,
              3, 4, 0);
        } else {
         libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, reg_sum, reg_sum );
        }
      }
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_micro_kernel_config->instruction_set,
                                        vmove_instruction_out,
                                        i_gp_reg_mapping->gp_reg_reduced_elts,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        0,
                                        vname_out,
                                        reg_sum, 2, 0, 1 );
    }

    if ( compute_squared_vals_reduce > 0 ) {
      libxsmm_generator_hinstrps_avx512( io_generated_code, reduce_instr, reg_sum_squared, 0, 1);
      if (reduce_on_output > 0) {
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                          vmove_instruction_out,
                                          i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          0,
                                          vname_out,
                                          tmp_vreg, 1, 2, 0  );
        if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
        } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              tmp_vreg, tmp_vreg, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
        } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
        } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, tmp_vreg, tmp_vreg );
        }
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             reduce_instr,
                                             i_micro_kernel_config->vector_name,
                                             tmp_vreg, reg_sum_squared, reg_sum_squared);
      }
      if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
        libxsmm_generator_vcvtneps2bf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
            reg_sum_squared, reg_sum_squared,
            0, 1,
            3, 4, 0, 0);
      } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
        libxsmm_generator_vcvtf32_to_hf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
            reg_sum_squared, reg_sum_squared, cvt_vreg_aux0, cvt_vreg_aux1, cvt_vreg_aux2, cvt_vreg_aux3, cvt_mask_aux0, cvt_mask_aux1, cvt_mask_aux2);
      } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
        libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, reg_sum_squared, reg_sum_squared, 0,
                                                                    (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
      } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
        if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX)) {
          libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              reg_sum_squared, reg_sum_squared,
              0, 1,
              3, 4, 0);
        } else {
         libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, reg_sum_squared, reg_sum_squared );
        }
      }
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_micro_kernel_config->instruction_set,
                                        vmove_instruction_out,
                                        i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        0,
                                        vname_out,
                                        reg_sum_squared, 2, 0, 1 );
    }
  } else if (n_cols_load != 0) {
    /* We fully unroll M loop here... */
    for (im = 0; im < m_trips; im++) {
      /* load 16 columns of input matrix */
      for (i = 0 ; i < n_cols_load; i++) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            (((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0))) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && !((use_m_masking > 0) && (im == (m_trips-1)))) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * vlen + i * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
            vname_in,
            i, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );
        if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, i, i );
        } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              i, i, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
        } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, i, i );
        } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
          libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, i, i );
        }
        if ((flag_reduce_op_max > 0 || flag_reduce_op_min > 0) && (use_m_masking > 0) && (im == m_trips-1)) {
          libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                       LIBXSMM_X86_INSTR_VBLENDMPS,
                                                       i_micro_kernel_config->vector_name,
                                                       i, aux_vreg, i, use_m_masking, 0 );
        }
      }

      for ( i = n_cols_load; i < vlen; i++) {
        if ((flag_reduce_op_max > 0  || flag_reduce_op_min > 0) && (use_m_masking > 0) && (im == m_trips-1)) {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS, i_micro_kernel_config->vector_name, aux_vreg, i );
        } else {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                   LIBXSMM_X86_INSTR_VXORPS,
                                                   i_micro_kernel_config->vector_name,
                                                   i, i, i );
        }
      }

      /* 1st stage */
      /* zmm0/zmm4; 4444 4444 4444 4444 / 0000 0000 0000 0000 -> zmm0: 4444 4444 0000 0000 */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     4, 0, 16, 7, 0 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             4, 0, 17, shuf_mask );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              0, 0, 18 );

        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              4, 4, 19 );


        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     19, 18, 20, 7, 0 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             19, 18, 21, shuf_mask );
      }

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             reduce_instr,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 0 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             reduce_instr,
                                             i_micro_kernel_config->vector_name,
                                             20, 21, 24 );
      }

      if (n_cols_load > 7) {
        /* zmm8/zmm12; cccc cccc cccc cccc / 8888 8888 8888 8888 -> zmm8: cccc cccc 8888 8888 */
        if ( compute_plain_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                       LIBXSMM_X86_INSTR_VBLENDMPS,
                                                       i_micro_kernel_config->vector_name,
                                                       12, 8, 16, 7, 0 );

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                               LIBXSMM_X86_INSTR_VSHUFF64X2,
                                               i_micro_kernel_config->vector_name,
                                               12, 8, 17, shuf_mask );
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                LIBXSMM_X86_INSTR_VMULPS,
                                                i_micro_kernel_config->vector_name,
                                                8, 8, 18 );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                LIBXSMM_X86_INSTR_VMULPS,
                                                i_micro_kernel_config->vector_name,
                                                12, 12, 19 );


          libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                       LIBXSMM_X86_INSTR_VBLENDMPS,
                                                       i_micro_kernel_config->vector_name,
                                                       19, 18, 20, 7, 0 );

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                               LIBXSMM_X86_INSTR_VSHUFF64X2,
                                               i_micro_kernel_config->vector_name,
                                               19, 18, 21, shuf_mask );
        }

        if ( compute_plain_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               reduce_instr,
                                               i_micro_kernel_config->vector_name,
                                               16, 17, 8 );
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               reduce_instr,
                                               i_micro_kernel_config->vector_name,
                                               20, 21, 28 );
        }
      }

      /* zmm1/zmm5; 5555 5555 5555 5555 / 1111 1111 1111 1111 -> zmm1: 5555 5555 1111 1111 */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     5, 1, 16, 7, 0 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             5, 1, 17, shuf_mask );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              1, 1, 18 );

        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              5, 5, 19 );


        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     19, 18, 20, 7, 0 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             19, 18, 21, shuf_mask );
      }

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             reduce_instr,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 1 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             reduce_instr,
                                             i_micro_kernel_config->vector_name,
                                             20, 21, 25 );
      }

      if (n_cols_load > 8) {
        /* zmm9/zmm13; dddd dddd dddd dddd / 9999 9999 9999 9999 -> zmm9: dddd dddd 9999 9999 */
        if ( compute_plain_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                       LIBXSMM_X86_INSTR_VBLENDMPS,
                                                       i_micro_kernel_config->vector_name,
                                                       13, 9, 16, 7, 0 );

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                               LIBXSMM_X86_INSTR_VSHUFF64X2,
                                               i_micro_kernel_config->vector_name,
                                               13, 9, 17, shuf_mask );
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                LIBXSMM_X86_INSTR_VMULPS,
                                                i_micro_kernel_config->vector_name,
                                                9, 9, 18 );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                LIBXSMM_X86_INSTR_VMULPS,
                                                i_micro_kernel_config->vector_name,
                                                13, 13, 19 );


          libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                       LIBXSMM_X86_INSTR_VBLENDMPS,
                                                       i_micro_kernel_config->vector_name,
                                                       19, 18, 20, 7, 0 );

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                               LIBXSMM_X86_INSTR_VSHUFF64X2,
                                               i_micro_kernel_config->vector_name,
                                               19, 18, 21, shuf_mask );
        }

        if ( compute_plain_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               reduce_instr,
                                               i_micro_kernel_config->vector_name,
                                               16, 17, 9 );
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               reduce_instr,
                                               i_micro_kernel_config->vector_name,
                                               20, 21, 29 );
        }
      }

      /* zmm2/zmm6; 6666 6666 6666 6666 / 2222 2222 2222 2222 -> zmm2: 6666 6666 2222 2222 */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     6, 2, 16, 7, 0 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             6, 2, 17, shuf_mask );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              2, 2, 18 );

        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              6, 6, 19 );


        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     19, 18, 20, 7, 0 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             19, 18, 21, shuf_mask );
      }

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             reduce_instr,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 2 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             reduce_instr,
                                             i_micro_kernel_config->vector_name,
                                             20, 21, 26 );
      }

      if (n_cols_load > 9) {
        /* zmm10/zmm14; eeee eeee eeee eeee / aaaa aaaa aaaa aaaa -> zmm10: eeee eeee aaaa aaaa */
        if ( compute_plain_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                       LIBXSMM_X86_INSTR_VBLENDMPS,
                                                       i_micro_kernel_config->vector_name,
                                                       14, 10, 16, 7, 0 );

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                               LIBXSMM_X86_INSTR_VSHUFF64X2,
                                               i_micro_kernel_config->vector_name,
                                               14, 10, 17, shuf_mask );
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                LIBXSMM_X86_INSTR_VMULPS,
                                                i_micro_kernel_config->vector_name,
                                                10, 10, 18 );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                LIBXSMM_X86_INSTR_VMULPS,
                                                i_micro_kernel_config->vector_name,
                                                14, 14, 19 );


          libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                       LIBXSMM_X86_INSTR_VBLENDMPS,
                                                       i_micro_kernel_config->vector_name,
                                                       19, 18, 20, 7, 0 );

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                               LIBXSMM_X86_INSTR_VSHUFF64X2,
                                               i_micro_kernel_config->vector_name,
                                               19, 18, 21, shuf_mask );
        }

        if ( compute_plain_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               reduce_instr,
                                               i_micro_kernel_config->vector_name,
                                               16, 17, 10 );
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               reduce_instr,
                                               i_micro_kernel_config->vector_name,
                                               20, 21, 30 );
        }
      }

      /* zmm3/zmm7; 7777 7777 7777 7777 / 3333 3333 3333 3333  -> zmm3: 7777 7777 3333 3333 */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     7, 3, 16, 7, 0 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             7, 3, 17, shuf_mask );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              3, 3, 18 );

        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              7, 7, 19 );


        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     19, 18, 20, 7, 0 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             19, 18, 21, shuf_mask );
      }

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             reduce_instr,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 3 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             reduce_instr,
                                             i_micro_kernel_config->vector_name,
                                             20, 21, 27 );
      }

      if (n_cols_load > 10) {
        /* zmm11/zmm15; ffff ffff ffff ffff / bbbb bbbb bbbb bbbb  -> zmm11: ffff ffff bbbb bbbb */
        if ( compute_plain_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                       LIBXSMM_X86_INSTR_VBLENDMPS,
                                                       i_micro_kernel_config->vector_name,
                                                       15, 11, 16, 7, 0 );

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                               LIBXSMM_X86_INSTR_VSHUFF64X2,
                                               i_micro_kernel_config->vector_name,
                                               15, 11, 17, shuf_mask );
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                LIBXSMM_X86_INSTR_VMULPS,
                                                i_micro_kernel_config->vector_name,
                                                11, 11, 18 );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                LIBXSMM_X86_INSTR_VMULPS,
                                                i_micro_kernel_config->vector_name,
                                                15, 15, 19 );


          libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                       LIBXSMM_X86_INSTR_VBLENDMPS,
                                                       i_micro_kernel_config->vector_name,
                                                       19, 18, 20, 7, 0 );

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                               LIBXSMM_X86_INSTR_VSHUFF64X2,
                                               i_micro_kernel_config->vector_name,
                                               19, 18, 21, shuf_mask );
        }

        if ( compute_plain_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               reduce_instr,
                                               i_micro_kernel_config->vector_name,
                                               16, 17, 11 );
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               reduce_instr,
                                               i_micro_kernel_config->vector_name,
                                               20, 21, 31 );
        }
      }

      /* 2nd stage */
      /* zmm0/zmm8; 4444 4444 0000 0000 / cccc cccc 8888 8888  -> zmm0: cccc 8888 4444 0000 */
      if (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) {
        if ( compute_plain_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  8, 0, 16, 0x88 );

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  8, 0, 17, 0xdd );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  reduce_instr,
                                                  i_micro_kernel_config->vector_name,
                                                  16, 17, 0 );
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  28, 24, 20, 0x88 );

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  28, 24, 21, 0xdd );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  reduce_instr,
                                                  i_micro_kernel_config->vector_name,
                                                  20, 21, 24 );
        }

        /* zmm1/zmm9; 5555 5555 1111 1111 / dddd dddd 9999 9999  -> zmm1: dddd 9999 5555 1111 */
        if ( compute_plain_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  9, 1, 16, 0x88 );

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  9, 1, 17, 0xdd );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  reduce_instr,
                                                  i_micro_kernel_config->vector_name,
                                                  16, 17, 1 );
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  29, 25, 20, 0x88 );

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  29, 25, 21, 0xdd );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  reduce_instr,
                                                  i_micro_kernel_config->vector_name,
                                                  20, 21, 25 );
        }

        /* zmm2/zmm10; 6666 6666 2222 2222 / eeee eeee aaaa aaaa  -> zmm2: eeee aaaa 6666 2222 */
        if ( compute_plain_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  10, 2, 16, 0x88 );

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  10, 2, 17, 0xdd );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  reduce_instr,
                                                  i_micro_kernel_config->vector_name,
                                                  16, 17, 2 );
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  30, 26, 20, 0x88 );

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  30, 26, 21, 0xdd );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  reduce_instr,
                                                  i_micro_kernel_config->vector_name,
                                                  20, 21, 26 );
        }

        /* zmm3/zmm11:  7777 7777 3333 3333 / ffff ffff bbbb bbbb  -> zmm3: ffff bbbb 7777 3333 */
        if ( compute_plain_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  11, 3, 16, 0x88 );

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  11, 3, 17, 0xdd );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  reduce_instr,
                                                  i_micro_kernel_config->vector_name,
                                                  16, 17, 3 );
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  31, 27, 20, 0x88 );

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                  i_micro_kernel_config->vector_name,
                                                  31, 27, 21, 0xdd );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  reduce_instr,
                                                  i_micro_kernel_config->vector_name,
                                                  20, 21, 27 );
        }
      }
      /* 3rd stage */
      /* zmm0/zmm1; cccc 8888 4444 0000 / dddd 9999 5555 1111  -> zmm0: ddcc 9988 5544 1100 */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 1, 0, 16, 0x44 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 1, 0, 17, 0xee );

        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 reduce_instr,
                                                 i_micro_kernel_config->vector_name,
                                                 16, 17, 0 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 25, 24, 20, 0x44 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 25, 24, 21, 0xee );

        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 reduce_instr,
                                                 i_micro_kernel_config->vector_name,
                                                 20, 21, 24 );
      }

      /* zmm2/zmm3; eeee aaaa 6666 2222 / ffff bbbb 7777 3333  -> zmm2: ffee bbaa 7766 3322 */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 3, 2, 16, 0x44 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 3, 2, 17, 0xee );

        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 reduce_instr,
                                                 i_micro_kernel_config->vector_name,
                                                 16, 17, 2 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 27, 26, 20, 0x44 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 27, 26, 21, 0xee );

        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 reduce_instr,
                                                 i_micro_kernel_config->vector_name,
                                                 20, 21, 26 );
      }

      /* 4th stage */
      /* zmm0/zmm2; ddcc 9988 5544 1100 / ffee bbaa 7766 3322  -> zmm0: fedc ba98 7654 3210 */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 2, 0, 16, 0x88 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 2, 0, 17, 0xdd );

        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 reduce_instr,
                                                 i_micro_kernel_config->vector_name,
                                                 16, 17, 0 );

        /* Update the running reduction result */
        if (im == 0 && reduce_on_output == 0) {
          if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              0,
                                              i_micro_kernel_config->vector_name,
                                              0, 2, 0, 1 );
            libxsmm_generator_vcvtneps2bf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                0, 0,
                3, 4,
                3, 4, 0, 0);
          } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              0,
                                              i_micro_kernel_config->vector_name,
                                              0, 2, 0, 1 );
            libxsmm_generator_vcvtf32_to_hf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                0, 0, cvt_vreg_aux0, cvt_vreg_aux1, cvt_vreg_aux2, cvt_vreg_aux3, cvt_mask_aux0, cvt_mask_aux1, cvt_mask_aux2);
          } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              0,
                                              i_micro_kernel_config->vector_name,
                                              0, 2, 0, 1 );
             libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, 0, 0, 0,
                                                                    (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
          } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              0,
                                              i_micro_kernel_config->vector_name,
                                              0, 2, 0, 1 );
            if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
              libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                  0, 0,
                  3, 4,
                  3, 4, 0);
            } else {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, 0, 0 );
            }
          }
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            0,
                                            vname_out,
                                            0, 2, 0, 1 );

        } else if (im == 0 && reduce_on_output > 0) {

          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            0,
                                            vname_out,
                                            3, 2, 1, 0 );
          if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, 3, 3 );
          } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                3, 3, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
          } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, 3, 3 );
          } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, 3, 3 );
          }
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, reduce_instr, i_micro_kernel_config->vector_name, 3, 0, 0);

          if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              0,
                                              i_micro_kernel_config->vector_name,
                                              0, 2, 0, 1 );
            libxsmm_generator_vcvtneps2bf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                0, 0,
                3, 4,
                3, 4, 0, 0);
          } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              0,
                                              i_micro_kernel_config->vector_name,
                                              0, 2, 0, 1 );
            libxsmm_generator_vcvtf32_to_hf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                0, 0, cvt_vreg_aux0, cvt_vreg_aux1, cvt_vreg_aux2, cvt_vreg_aux3, cvt_mask_aux0, cvt_mask_aux1, cvt_mask_aux2);
          } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              0,
                                              i_micro_kernel_config->vector_name,
                                              0, 2, 0, 1 );
             libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, 0, 0, 0,
                                                                    (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
          } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              0,
                                              i_micro_kernel_config->vector_name,
                                              0, 2, 0, 1 );
            if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
              libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                  0, 0,
                  3, 4,
                  3, 4, 0);
            } else {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, 0, 0 );
            }
          }
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            0,
                                            vname_out,
                                            0, 2, 0, 1 );

        } else if ( im == (m_trips-1) ) {
          if ((LIBXSMM_DATATYPE_F16  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_HF8  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_BF8  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVUPS,
                bf16_accum,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                0,
                i_micro_kernel_config->vector_name,
                1, 2, 1, 0 );
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                vmove_instruction_out,
                i_gp_reg_mapping->gp_reg_reduced_elts,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                0,
                vname_out,
                1, 2, 1, 0 );
          }
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  reduce_instr,
                                                  i_micro_kernel_config->vector_name,
                                                  1, 0, 0 );
          if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_vcvtneps2bf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                0, 0,
                3, 4,
                3, 4, 0, 0);
          } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_vcvtf32_to_hf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                0, 0, cvt_vreg_aux0, cvt_vreg_aux1, cvt_vreg_aux2, cvt_vreg_aux3, cvt_mask_aux0, cvt_mask_aux1, cvt_mask_aux2);
          } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, 0, 0, 0,
                                                                    (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
          } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
              libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                  0, 0,
                  3, 4,
                  3, 4, 0);
            } else {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, 0, 0 );
            }
          }
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            0,
                                            vname_out,
                                            0, 2, 0, 1 );
        } else {
          if ((LIBXSMM_DATATYPE_F16  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_HF8  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_BF8  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVUPS,
              bf16_accum,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              0,
              i_micro_kernel_config->vector_name,
              1, 2, 1, 0 );
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                vmove_instruction_out,
                i_gp_reg_mapping->gp_reg_reduced_elts,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                0,
                vname_out,
                1, 2, 1, 0 );
          }
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  reduce_instr,
                                                  i_micro_kernel_config->vector_name,
                                                  1, 0, 0 );

          if ((LIBXSMM_DATATYPE_F16  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_HF8  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_BF8  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              0,
                                              i_micro_kernel_config->vector_name,
                                              0, 2, 0, 1 );
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              vmove_instruction_out,
                                              i_gp_reg_mapping->gp_reg_reduced_elts,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              0,
                                              vname_out,
                                              0, 2, 0, 1 );
          }
        }
      }
      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 26, 24, 20, 0x88 );

        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 26, 24, 21, 0xdd );

        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 reduce_instr,
                                                 i_micro_kernel_config->vector_name,
                                                 20, 21, 24 );
        if (im == 0 && reduce_on_output == 0) {
          if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              64,
                                              i_micro_kernel_config->vector_name,
                                              24, 2, 0, 1 );
            libxsmm_generator_vcvtneps2bf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                24, 24,
                25, 26,
                3, 4, 0, 0);
          } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              64,
                                              i_micro_kernel_config->vector_name,
                                              24, 2, 0, 1 );
            libxsmm_generator_vcvtf32_to_hf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                24, 24, cvt_vreg_aux0, cvt_vreg_aux1, cvt_vreg_aux2, cvt_vreg_aux3, cvt_mask_aux0, cvt_mask_aux1, cvt_mask_aux2);
          } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              64,
                                              i_micro_kernel_config->vector_name,
                                              24, 2, 0, 1 );
             libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, 24, 24, 0,
                                                                    (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
          } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              64,
                                              i_micro_kernel_config->vector_name,
                                              24, 2, 0, 1 );
            if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
              libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                   24, 24,
                   25, 26,
                   3, 4, 0);
            } else {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, 24, 24 );
            }
          }
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            0,
                                            vname_out,
                                            24, 2, 0, 1 );

        } else if (im == 0 && reduce_on_output > 0) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            0,
                                            vname_out,
                                            25, 2, 1, 0 );
          if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, 25, 25 );
          } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                25, 25, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
          } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, 25, 25 );
          } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, 25, 25 );
          }
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, reduce_instr, i_micro_kernel_config->vector_name, 25, 24, 24);

          if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              64,
                                              i_micro_kernel_config->vector_name,
                                              24, 2, 0, 1 );
            libxsmm_generator_vcvtneps2bf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                24, 24,
                25, 26,
                3, 4, 0, 0);
          } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              64,
                                              i_micro_kernel_config->vector_name,
                                              24, 2, 0, 1 );
            libxsmm_generator_vcvtf32_to_hf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                24, 24, cvt_vreg_aux0, cvt_vreg_aux1, cvt_vreg_aux2, cvt_vreg_aux3, cvt_mask_aux0, cvt_mask_aux1, cvt_mask_aux2);
          } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              64,
                                              i_micro_kernel_config->vector_name,
                                              24, 2, 0, 1 );
             libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, 24, 24, 0,
                                                                    (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
          } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              64,
                                              i_micro_kernel_config->vector_name,
                                              24, 2, 0, 1 );
            if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
              libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                    24, 24,
                    25, 26,
                    3, 4, 0);
            } else {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, 24, 24 );
            }
          }
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            0,
                                            vname_out,
                                            24, 2, 0, 1 );

        } else if ( im == (m_trips-1) ) {
          if ((LIBXSMM_DATATYPE_F16  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_HF8  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_BF8  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVUPS,
                bf16_accum,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                64,
                i_micro_kernel_config->vector_name,
                25, 2, 1, 0 );
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                vmove_instruction_out,
                i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                0,
                vname_out,
                25, 2, 1, 0 );
          }
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  reduce_instr,
                                                  i_micro_kernel_config->vector_name,
                                                  25, 24, 24 );
          if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_vcvtneps2bf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                24, 24,
                25, 26,
                3, 4, 0, 0);
          } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_generator_vcvtf32_to_hf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                24, 24, cvt_vreg_aux0, cvt_vreg_aux1, cvt_vreg_aux2, cvt_vreg_aux3, cvt_mask_aux0, cvt_mask_aux1, cvt_mask_aux2);
          } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, 24, 24, 0,
                                                                    (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
          } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
            if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
              libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                  24, 24,
                  25, 26,
                  3, 4, 0);
            } else {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, 24, 24 );
            }
          }
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            0,
                                            vname_out,
                                            24, 2, 0, 1 );
        } else {
          if ((LIBXSMM_DATATYPE_F16  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_HF8  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_BF8  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVUPS,
              bf16_accum,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              64,
              i_micro_kernel_config->vector_name,
              25, 2, 1, 0 );
          }else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                vmove_instruction_out,
                i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                0,
                vname_out,
                25, 2, 1, 0 );
          }
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  reduce_instr,
                                                  i_micro_kernel_config->vector_name,
                                                  25, 24, 24 );

          if ((LIBXSMM_DATATYPE_F16  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_HF8  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||
              (LIBXSMM_DATATYPE_BF8  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMOVUPS,
                                              bf16_accum,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              64,
                                              i_micro_kernel_config->vector_name,
                                              24, 2, 0, 1 );
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              vmove_instruction_out,
                                              i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              0,
                                              vname_out,
                                              24, 2, 0, 1 );
          }
        }
      }
    }
  }

  if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
    libxsmm_generator_vcvtneps2bf8_avx512_clean_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
  } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
    if ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) {
      libxsmm_generator_vcvtneps2bf16_avx512_clean_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
    }
  } else if ((LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))) {
    libxsmm_generator_vcvt_hf8_tofrom_f32_avx512_clean_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
  }
}

LIBXSMM_API_INTERN void libxsmm_generator_avx_extract_mask4_from_mask8( libxsmm_generated_code* io_generated_code, unsigned int mask_in, unsigned int mask_out, unsigned int lohi);
LIBXSMM_API_INTERN void libxsmm_generator_avx_extract_mask4_from_mask8( libxsmm_generated_code* io_generated_code, unsigned int mask_in, unsigned int mask_out, unsigned int lohi) {
  if (lohi == 0) {
    libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PD, 'y', mask_in, mask_out );
  } else {
    libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPERM2F128, 'y', mask_in, mask_in, mask_out, 0x1 );
    libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PD, 'y', mask_out, mask_out );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_reduce_cols_index_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {

  unsigned int m, im, use_m_masking, m_trips, max_m_unrolling = 4, m_unroll_factor = 1, m_trips_loop = 0, peeled_m_trips = 0, mask_out_count = 0;
  unsigned int aux_vreg_offset = 16;
  unsigned int idx_tsize = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES) > 0 ) ? 8 : 4;
  unsigned int ind_alu_mov_instr = (idx_tsize == 8) ? LIBXSMM_X86_INSTR_MOVQ : LIBXSMM_X86_INSTR_MOVD;
  int pf_dist = 0, use_nts = 0, pf_instr = LIBXSMM_X86_INSTR_PREFETCHT1, pf_type = 1, load_acc = 1;
  unsigned int vstore_instr = 0;
  unsigned int l_n_code_blocks = 1;
  unsigned int l_code_block_id = 0;
  unsigned int l_m_code_blocks = 0;
  unsigned int l_m_code_block_id = 0;
  unsigned int NO_PF_LABEL_START = 0;
  unsigned int NO_PF_LABEL_START_2 = 1;
  unsigned int END_LABEL = 2;
  unsigned int vlen = 16;
  char  vname_in = i_micro_kernel_config->vector_name;
  char  vname_out = i_micro_kernel_config->vector_name;
  unsigned int vmove_instruction_in = i_micro_kernel_config->vmove_instruction_in;
  unsigned int vmove_instruction_out = i_micro_kernel_config->vmove_instruction_out;
  unsigned int mask_reg = 0, mask_reg_in = 0, mask_reg_out = 0;
  unsigned int cvt_vreg_aux0 = 31, cvt_vreg_aux1 = 30, cvt_vreg_aux2 = 29, cvt_vreg_aux3 = 28;
  unsigned int cvt_mask_aux0 = 7, cvt_mask_aux1 = 6, cvt_mask_aux2 = 5;
  unsigned int l_temp_vreg_argop = 0;
  unsigned int l_mask_argidx32 = 0;
  unsigned int l_mask_argidx64 = 4;
  unsigned int l_argop_mask = 3, l_argop_mask_aux = 0;
  unsigned int l_is_reduce_add = 1;
  unsigned int l_is_reduce_max = 0;
  unsigned int l_is_reduce_min = 0;
  unsigned int l_record_argop = 0;
  unsigned int l_reduce_instr = LIBXSMM_X86_INSTR_VADDPS;
  unsigned int gp_reg_argop = 0;
  unsigned int l_use_stack_vars = 0;
  int l_rbp_offset_inf    = -32;
  int l_rbp_offset_idx = -8;
  char l_vname_argop_bcast = (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 'y' : 'z';
  unsigned int l_bcast_idx_instr = ( idx_tsize == 8 ) ? LIBXSMM_X86_INSTR_VPBROADCASTQ : LIBXSMM_X86_INSTR_VPBROADCASTD ;
  unsigned int l_gpr_bcast_idx_instr = ( idx_tsize == 8 ) ? LIBXSMM_X86_INSTR_VPBROADCASTQ_GPR : LIBXSMM_X86_INSTR_VPBROADCASTD_GPR;
  unsigned int l_argop_cmp_instr = LIBXSMM_X86_INSTR_VCMPPS;
  unsigned int l_argop_blend_instr = (idx_tsize == 8) ? LIBXSMM_X86_INSTR_VPBLENDMQ : LIBXSMM_X86_INSTR_VPBLENDMD;
  unsigned int l_argop_cmp_imm = 0;
  unsigned int l_idx0_vreg_offset = 0, l_idx1_vreg_offset = 0;
#if defined(USE_ENV_TUNING)
  const char *const env_max_m_unroll = getenv("MAX_M_UNROLL_REDUCE_COLS_IDX");
#endif
  const char *const env_pf_dist = getenv("PF_DIST_REDUCE_COLS_IDX");
  const char *const env_pf_type = getenv("PF_TYPE_REDUCE_COLS_IDX");
  const char *const env_nts     = getenv("NTS_REDUCE_COLS_IDX");
  const char *const env_load_acc= getenv("LOAD_ACCS_REDUCE_COLS_IDX");
  libxsmm_jump_label_tracker l_jump_label_tracker;
  libxsmm_reset_jump_label_tracker(&l_jump_label_tracker);

  libxsmm_generator_reduce_set_lp_vlen_vname_vmove_x86( io_generated_code, i_micro_kernel_config, i_mateltwise_desc, &vlen, &vname_in, &vname_out, &vmove_instruction_in, &vmove_instruction_out );

  if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_MAX || i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_MIN) {
    l_is_reduce_add = 0;
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_MAX) {
      l_is_reduce_max = 1;
      l_reduce_instr = LIBXSMM_X86_INSTR_VMAXPS;
      l_argop_cmp_imm = 6;
    }
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_MIN) {
      l_is_reduce_min = 1;
      l_reduce_instr = LIBXSMM_X86_INSTR_VMINPS;
      l_argop_cmp_imm = 9;
    }
    if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_RECORD_ARGOP) > 0) {
      l_record_argop = 1;
      l_use_stack_vars = 1;
    }
    if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_INF_ACC) > 0) {
      l_use_stack_vars = 1;
    }
  }

#if defined(USE_ENV_TUNING)
  if ( 0 == env_max_m_unroll ) {
  } else {
    max_m_unrolling = LIBXSMM_MAX(1,atoi(env_max_m_unroll));
  }
#endif
  if ( 0 == env_pf_dist ) {
  } else {
    pf_dist = atoi(env_pf_dist);
  }
  if ( 0 == env_pf_type ) {
  } else {
    pf_type = atoi(env_pf_type);
  }
  if ( 0 == env_nts ) {
  } else {
    use_nts = atoi(env_nts);
  }
  if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_INIT_ACC) > 0 ) {
    load_acc = 1;
  } else {
    load_acc = 0;
  }
  if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_INF_ACC) > 0) {
    load_acc = 0;
  }
  if ( 0 == env_load_acc ) {
  } else {
    load_acc = atoi(env_load_acc);
  }
  if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_NO_PREFETCH) > 0 ) {
    pf_dist = 0;
  }
  if (pf_dist > 0) {
    l_n_code_blocks++;
  }

  pf_instr      = (pf_type == 2) ? LIBXSMM_X86_INSTR_PREFETCHT1 : LIBXSMM_X86_INSTR_PREFETCHT0 ;
  vstore_instr  = (use_nts == 0) ? vmove_instruction_out : LIBXSMM_X86_INSTR_VMOVNTPS;

  i_gp_reg_mapping->gp_reg_n        = LIBXSMM_X86_GP_REG_R8;
  i_gp_reg_mapping->gp_reg_ind_base = LIBXSMM_X86_GP_REG_R9;
  i_gp_reg_mapping->gp_reg_in_base  = LIBXSMM_X86_GP_REG_R10;
  i_gp_reg_mapping->gp_reg_out      = LIBXSMM_X86_GP_REG_R11;
  i_gp_reg_mapping->gp_reg_m_loop   = LIBXSMM_X86_GP_REG_RAX;
  i_gp_reg_mapping->gp_reg_n_loop   = LIBXSMM_X86_GP_REG_RDX;
  i_gp_reg_mapping->gp_reg_in       = LIBXSMM_X86_GP_REG_RSI;
  i_gp_reg_mapping->gp_reg_in_pf    = LIBXSMM_X86_GP_REG_RCX;
  gp_reg_argop = LIBXSMM_X86_GP_REG_RDI;

  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      48,
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
  libxsmm_x86_instruction_jump_to_label(io_generated_code, LIBXSMM_X86_INSTR_JLE, END_LABEL, &l_jump_label_tracker);

  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      40,
      i_gp_reg_mapping->gp_reg_ind_base,
      0 );

  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      32,
      i_gp_reg_mapping->gp_reg_in_base,
      0 );

  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      64,
      i_gp_reg_mapping->gp_reg_out,
      0 );

  if (l_record_argop > 0) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        72,
        gp_reg_argop,
        0 );
  }

  if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
    vlen = 8;
  }

  m                 = i_mateltwise_desc->m;
  use_m_masking     = ( m % vlen == 0 ) ? 0 : 1;
  m_trips           = (m + vlen - 1) / vlen;
  m_unroll_factor   = m_trips;
  m_trips_loop      = 1;
  peeled_m_trips    = 0;

  if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
    aux_vreg_offset = 8;
    max_m_unrolling = 8;
    if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
      max_m_unrolling = 6;
      cvt_vreg_aux0 = 14;
      cvt_vreg_aux1 = 13;
    }
  } else {
    max_m_unrolling = 14;
  }

  if (use_m_masking == 1) {
    if (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) {
      /* Calculate mask reg 1 for reading/output-writing */
      mask_out_count = vlen - (m % vlen);
      mask_reg = 1;
      libxsmm_generator_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, mask_reg, mask_out_count, LIBXSMM_DATATYPE_F32);
      mask_reg_in = mask_reg;
      mask_reg_out = mask_reg;
      l_mask_argidx32 = mask_reg;
      l_mask_argidx64 = 4;
    } else {
      mask_reg = 15;
      aux_vreg_offset = 7;
      if (max_m_unrolling > 7) {
        max_m_unrolling = 7;
      }
      l_mask_argidx32 = mask_reg;
      l_mask_argidx64 = (LIBXSMM_DATATYPE_F32 != libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ? 12 : 14;
      libxsmm_generator_initialize_avx_mask(io_generated_code, mask_reg, m % vlen, LIBXSMM_DATATYPE_F32);
      mask_reg_in = mask_reg;
      mask_reg_out = mask_reg;
      if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) || LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
        mask_reg_in = m % vlen;
      }
      if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) || LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
        mask_reg_out = m % vlen;
      }
    }
  }

  if (l_record_argop > 0) {
    if (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) {
      l_temp_vreg_argop = 27;
      if (idx_tsize == 8) {
        max_m_unrolling = 6;
      } else {
        max_m_unrolling = 8;
      }
    } else {
      l_bcast_idx_instr = ( idx_tsize == 8 ) ? LIBXSMM_X86_INSTR_VBROADCASTSD : LIBXSMM_X86_INSTR_VBROADCASTSS;
      l_argop_blend_instr = (idx_tsize == 8) ? LIBXSMM_X86_INSTR_VBLENDVPD : LIBXSMM_X86_INSTR_VBLENDVPS;
      l_temp_vreg_argop = 11;
      l_argop_mask = 10;
      if (idx_tsize == 8) {
        max_m_unrolling = 2;
        l_argop_mask_aux = 9;
      } else {
        max_m_unrolling = 3;
      }
    }
  }

  if (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) {
    if ((idx_tsize == 8) && (m % (vlen/2) != 0) && (l_record_argop > 0)) {
      libxsmm_generator_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, l_mask_argidx64, (vlen/2) - (m % (vlen/2)), LIBXSMM_DATATYPE_F64);
    }
  } else {
    if ((idx_tsize == 8) && (m % 4 != 0) && (l_record_argop > 0)) {
      unsigned long long  mask_array[4] = {0, 0, 0, 0};
      unsigned int _i_;
      for (_i_ = 0; _i_ < m % 4 ; _i_++) {
        mask_array[_i_] = 0xFFFFFFFFFFFFFFFF;
      }
      libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) mask_array, "mask_array", 'y', l_mask_argidx64 );
    }
  }

  if (l_use_stack_vars > 0) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBP );
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_RBP);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, 40 );
    if (l_is_reduce_max) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mapping->gp_reg_n_loop, 0xff800000);
    }
    if (l_is_reduce_min) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mapping->gp_reg_n_loop, 0x7f800000);
    }
    libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, l_rbp_offset_inf, i_gp_reg_mapping->gp_reg_n_loop, 1 );
  }

  if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
    libxsmm_generator_vcvtneps2bf8_avx512_prep_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
  } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
    if (io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) {
      libxsmm_generator_vcvtneps2bf16_avx512_prep_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
    }
  } else if ((LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )))  {
    libxsmm_generator_vcvt_hf8_tofrom_f32_avx512_prep_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
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
      libxsmm_generator_mateltwise_header_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_m_loop );
    }
    if ((m_trips_loop == 1) && (peeled_m_trips > 0) && (l_m_code_block_id == l_m_code_blocks - 1)) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_out, (long long)m_unroll_factor * vlen * i_micro_kernel_config->datatype_size_out);
      if (l_record_argop > 0) libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, gp_reg_argop, (long long)m_unroll_factor * vlen * idx_tsize );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in_base, (long long)m_unroll_factor * vlen * i_micro_kernel_config->datatype_size_in);
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
            libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch,
                LIBXSMM_X86_INSTR_VBROADCASTSS, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, l_rbp_offset_inf,
                i_micro_kernel_config->vector_name, im, 0, 1, 0 );
          } else {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS, i_micro_kernel_config->vector_name, 0, im );
          }
        } else {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VXORPS,
                                                 i_micro_kernel_config->vector_name,
                                                 im, im , im);
        }
      } else {
        if ( LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) && (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) ) {
          libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8(io_generated_code,
                                                             LIBXSMM_X86_INSTR_VCVTPH2PS,
                                                             i_micro_kernel_config->vector_name,
                                                             i_gp_reg_mapping->gp_reg_out,
                                                             LIBXSMM_X86_GP_REG_UNDEF,
                                                             0,
                                                             im * vlen  * i_micro_kernel_config->datatype_size_out,
                                                             0,
                                                             LIBXSMM_X86_VEC_REG_UNDEF,
                                                             im,
                                                             ((im == peeled_m_trips-1) && (use_m_masking > 0) && (l_is_peeled_loop > 0)) ? use_m_masking : 0,
                                                             0,
                                                             0);
        } else {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
              vmove_instruction_out,
              i_gp_reg_mapping->gp_reg_out,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              im * vlen  * i_micro_kernel_config->datatype_size_out,
              vname_out,
              im, ((im == peeled_m_trips-1) && (use_m_masking > 0) && (l_is_peeled_loop > 0)) ? 1 : 0, ((im == peeled_m_trips-1) && (use_m_masking > 0) && (l_is_peeled_loop > 0)) ? mask_reg_out : 0, 0 );

          if ((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))) {
            libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, im, im);
          } else if ((LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))) {
            libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                im, im, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
          } else if ((LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))) {
            libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, im, im);
          } else if ((LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))) {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, im, im );
          }
        }
      }
      if (l_record_argop > 0) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                               LIBXSMM_X86_INSTR_VXORPS,
                                               i_micro_kernel_config->vector_name,
                                               im + l_idx0_vreg_offset, im + l_idx0_vreg_offset, im + l_idx0_vreg_offset);
        if (idx_tsize == 8) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VXORPS,
                                                 i_micro_kernel_config->vector_name,
                                                 im + l_idx1_vreg_offset, im + l_idx1_vreg_offset, im + l_idx1_vreg_offset);
        }
      }
    }

    for (l_code_block_id = 0; l_code_block_id < l_n_code_blocks; l_code_block_id++) {
      unsigned int l_emit_pf_code = (pf_dist > 0 && l_code_block_id == 0) ? 1 : 0;
      if (l_emit_pf_code) {
        libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_n_loop, 0);
        libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_n, pf_dist);
        libxsmm_x86_instruction_jump_to_label(io_generated_code, LIBXSMM_X86_INSTR_JLE, (l_m_code_block_id == 0) ? NO_PF_LABEL_START : NO_PF_LABEL_START_2, &l_jump_label_tracker);
        libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_SUBQ, i_gp_reg_mapping->gp_reg_n, pf_dist);
      }
      libxsmm_generator_mateltwise_header_n_dyn_loop(io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, (pf_dist > 0) ? 1 : 0);
      libxsmm_x86_instruction_alu_mem(io_generated_code, ind_alu_mov_instr, i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_n_loop, idx_tsize, 0, i_gp_reg_mapping->gp_reg_in, 0);
      if (l_emit_pf_code) libxsmm_x86_instruction_alu_mem(io_generated_code, ind_alu_mov_instr, i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_n_loop, idx_tsize, pf_dist * idx_tsize, i_gp_reg_mapping->gp_reg_in_pf, 0);

      if (l_record_argop > 0) {
        if (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) {
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8(io_generated_code, l_gpr_bcast_idx_instr, l_vname_argop_bcast, i_gp_reg_mapping->gp_reg_in, LIBXSMM_X86_VEC_REG_UNDEF, l_temp_vreg_argop, 0, 0, 0, 0);
        } else {
          libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, l_rbp_offset_idx, i_gp_reg_mapping->gp_reg_in, 1 );
          libxsmm_x86_instruction_vec_move(io_generated_code, i_micro_kernel_config->instruction_set, l_bcast_idx_instr, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, l_rbp_offset_idx, l_vname_argop_bcast, l_temp_vreg_argop, 0, 0, 0);
        }
      }

      libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_in, (long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in);
      if (l_emit_pf_code) libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_in_pf, (long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in);
      libxsmm_x86_instruction_alu_reg(io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in_base, i_gp_reg_mapping->gp_reg_in);
      if (l_emit_pf_code) libxsmm_x86_instruction_alu_reg(io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in_base, i_gp_reg_mapping->gp_reg_in_pf);

      for (im = 0; im < m_unroll_factor; im++) {
        if ( LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) && (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (l_is_reduce_add > 0) ) {
          libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8(io_generated_code,
                                                             LIBXSMM_X86_INSTR_VCVTPH2PS,
                                                             i_micro_kernel_config->vector_name,
                                                             i_gp_reg_mapping->gp_reg_in,
                                                             LIBXSMM_X86_GP_REG_UNDEF,
                                                             0,
                                                             im * vlen  * i_micro_kernel_config->datatype_size_in,
                                                             0,
                                                             LIBXSMM_X86_VEC_REG_UNDEF,
                                                             im+aux_vreg_offset,
                                                             ((im == peeled_m_trips-1) && (use_m_masking > 0) && (l_is_peeled_loop > 0)) ? use_m_masking : 0,
                                                             0,
                                                             0);
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                        l_reduce_instr,
                                        i_micro_kernel_config->vector_name,
                                        im, im+aux_vreg_offset, im );
        } else if ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) && (l_is_reduce_add > 0) && (!((im == peeled_m_trips-1) && (use_m_masking > 0) && (l_is_peeled_loop > 0)))) {
            libxsmm_x86_instruction_vec_compute_mem_2reg( io_generated_code,
              LIBXSMM_X86_INSTR_VADDPS,
              i_micro_kernel_config->vector_name,
              i_gp_reg_mapping->gp_reg_in,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              im * vlen * i_micro_kernel_config->datatype_size_in, 0,
              im,
              im );
        } else {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
              vmove_instruction_in,
              i_gp_reg_mapping->gp_reg_in,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              im * vlen  * i_micro_kernel_config->datatype_size_in,
              vname_in,
              im+aux_vreg_offset, ((im == peeled_m_trips-1) && (use_m_masking > 0) && (l_is_peeled_loop > 0)) ? use_m_masking : 0, ((im == peeled_m_trips-1) && (use_m_masking > 0) && (l_is_peeled_loop > 0)) ? mask_reg_in : 0, 0 );
          if ( LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) {
            libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, im+aux_vreg_offset, im+aux_vreg_offset);
          } else if ( LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) {
            libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                im+aux_vreg_offset, im+aux_vreg_offset, cvt_vreg_aux0, cvt_vreg_aux1, cvt_mask_aux0, cvt_mask_aux1 );
          } else if ( LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) {
            libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, im+aux_vreg_offset, im+aux_vreg_offset);
          } else if ( LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, im+aux_vreg_offset, im+aux_vreg_offset );
          } else {
            /* Should not happen */
          }

          if (l_record_argop > 0) {
            libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, l_argop_cmp_instr, i_micro_kernel_config->vector_name, im+aux_vreg_offset, im, l_argop_mask, l_argop_cmp_imm );
            if (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) {
              libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code, l_argop_blend_instr, i_micro_kernel_config->vector_name, im + l_idx0_vreg_offset, l_temp_vreg_argop, im + l_idx0_vreg_offset, l_argop_mask, 0 );
              if (idx_tsize == 8) {
                libxsmm_x86_instruction_mask_compute_reg( io_generated_code,LIBXSMM_X86_INSTR_KSHIFTRW, l_argop_mask, LIBXSMM_X86_VEC_REG_UNDEF, l_argop_mask, (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) ? 8 : 4);
                libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code, l_argop_blend_instr, i_micro_kernel_config->vector_name, im + l_idx1_vreg_offset, l_temp_vreg_argop, im + l_idx1_vreg_offset, l_argop_mask, 0 );
              }
            } else {
              if (idx_tsize == 8) {
                libxsmm_generator_avx_extract_mask4_from_mask8( io_generated_code, l_argop_mask, l_argop_mask_aux, 1);
                libxsmm_generator_avx_extract_mask4_from_mask8( io_generated_code, l_argop_mask, l_argop_mask, 0);
              }
              libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, l_argop_blend_instr, i_micro_kernel_config->vector_name, im + l_idx0_vreg_offset, l_temp_vreg_argop, im + l_idx0_vreg_offset, 0, 0, 0, l_argop_mask << 4);
              if (idx_tsize == 8) {
                libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, l_argop_blend_instr, i_micro_kernel_config->vector_name, im + l_idx1_vreg_offset, l_temp_vreg_argop, im + l_idx1_vreg_offset, 0, 0, 0, l_argop_mask_aux << 4);
              }
            }
          }

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                        l_reduce_instr,
                                        i_micro_kernel_config->vector_name,
                                        im, im+aux_vreg_offset, im );
        }

        if (((im * vlen * i_micro_kernel_config->datatype_size_in) % 64 == 0) && (l_emit_pf_code > 0)) {
          libxsmm_x86_instruction_prefetch(io_generated_code,
              pf_instr,
              i_gp_reg_mapping->gp_reg_in_pf,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              im * vlen * i_micro_kernel_config->datatype_size_in);
        }
      }

      libxsmm_generator_mateltwise_footer_n_dyn_loop(io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, i_gp_reg_mapping->gp_reg_n);
      if (l_emit_pf_code) libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_n, pf_dist);
      /* NO_PF_LABEL_START */
      if (l_emit_pf_code) libxsmm_x86_instruction_register_jump_label(io_generated_code, (l_m_code_block_id == 0) ? NO_PF_LABEL_START : NO_PF_LABEL_START_2, &l_jump_label_tracker);
    }

    /* Now store accumulators */
    for (im = 0; im < m_unroll_factor; im++) {
      if ( LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) && (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) ) {
        libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8( io_generated_code,
                                                             LIBXSMM_X86_INSTR_VCVTPS2PH,
                                                             i_micro_kernel_config->vector_name,
                                                             i_gp_reg_mapping->gp_reg_out,
                                                             LIBXSMM_X86_GP_REG_UNDEF,
                                                             0,
                                                             im * vlen  * i_micro_kernel_config->datatype_size_out,
                                                             0,
                                                             LIBXSMM_X86_VEC_REG_UNDEF,
                                                             im,
                                                             ((im == peeled_m_trips-1) && (use_m_masking > 0) && (l_is_peeled_loop > 0)) ? use_m_masking : 0,
                                                             0,
                                                             0 );
      } else {
        if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          libxsmm_generator_vcvtneps2bf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              im, im,
              cvt_vreg_aux0, cvt_vreg_aux1,
              2, 3, 0, 0);
        } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          libxsmm_generator_vcvtf32_to_hf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
              im, im, cvt_vreg_aux0, cvt_vreg_aux1, cvt_vreg_aux2, cvt_vreg_aux3, cvt_mask_aux0, cvt_mask_aux1, cvt_mask_aux2);
        } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          if (io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) {
            libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                im, im,
                cvt_vreg_aux0, cvt_vreg_aux1,
                2, 3, 0);
          } else {
           libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, im, im );
          }
        } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
          libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, im, im, 0,
                                                                (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
        }

        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            vstore_instr,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * vlen  * i_micro_kernel_config->datatype_size_out,
            vname_out,
            im, ((im == peeled_m_trips-1) && (use_m_masking > 0) && (l_is_peeled_loop > 0)) ? use_m_masking : 0, ((im == peeled_m_trips-1) && (use_m_masking > 0) && (l_is_peeled_loop > 0)) ? mask_reg_out : 0, 1 );
      }
    }

    if ( l_record_argop > 0 ) {
      char vname = (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 'y' : 'z';
      if (idx_tsize == 4) {
        for (im = 0; im < m_unroll_factor; im++) {
          unsigned int vreg_id = im + l_idx0_vreg_offset;
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
              LIBXSMM_X86_INSTR_VMOVUPS,
              gp_reg_argop,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              im * vlen * 4,
              vname,
              vreg_id, ((im == peeled_m_trips-1) && (use_m_masking > 0) && (l_is_peeled_loop > 0)) ? use_m_masking : 0, ((im == peeled_m_trips-1) && (use_m_masking > 0) && (l_is_peeled_loop > 0) ) ? l_mask_argidx32 : 0, 1 );
        }
      } else {
        for (im = 0; im < m_unroll_factor; im++) {
          unsigned int use_mask_0 = ((im == peeled_m_trips-1) && (l_is_peeled_loop > 0) && (m % vlen < (vlen/2) && (m % vlen != 0))) ? 1 : 0;
          unsigned int use_mask_1 = ((im == peeled_m_trips-1) && (l_is_peeled_loop > 0) && (m % vlen > (vlen/2) && (m % vlen != 0))) ? 1 : 0;
          unsigned int vreg_id0 = im + l_idx0_vreg_offset;
          unsigned int vreg_id1 = im + l_idx1_vreg_offset;
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
              LIBXSMM_X86_INSTR_VMOVUPD,
              gp_reg_argop,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              im * vlen * 8,
              vname,
              vreg_id0, use_mask_0, (use_mask_0 > 0) ? l_mask_argidx64 : 0, 1 );
          if (use_mask_0 == 0 && !((m % vlen == vlen/2) && (im == peeled_m_trips-1) && (l_is_peeled_loop > 0))) {
            libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                LIBXSMM_X86_INSTR_VMOVUPD,
                gp_reg_argop,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                im * vlen * 8 + vlen * 4,
                vname,
                vreg_id1, use_mask_1, (use_mask_1 > 0) ? l_mask_argidx64 : 0, 1 );
          }
        }
      }
    }

    if (m_trips_loop > 1 && l_m_code_block_id == 0) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_out, (long long)m_unroll_factor * vlen * i_micro_kernel_config->datatype_size_out);
      if (l_record_argop > 0) libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, gp_reg_argop, (long long)m_unroll_factor * vlen * idx_tsize );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in_base, (long long)m_unroll_factor * vlen * i_micro_kernel_config->datatype_size_in);
      libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_m_loop, m_trips_loop);
    }
  }

  if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
    libxsmm_generator_vcvtneps2bf8_avx512_clean_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
  } else if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
    if (io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) {
      libxsmm_generator_vcvtneps2bf16_avx512_clean_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
    }
  } else if ((LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))) {
    libxsmm_generator_vcvt_hf8_tofrom_f32_avx512_clean_stack( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
  }

  if (l_use_stack_vars > 0) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_RSP);
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBP );
  }

  libxsmm_x86_instruction_register_jump_label(io_generated_code, END_LABEL, &l_jump_label_tracker);

#if defined(LIBXSMM_GENERATOR_MATELTWISE_REDUCE_AVX_AVX512_JUMP_LABEL_TRACKER_MALLOC)
  free(p_jump_label_tracker);
#endif
}
