/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include "generator_gemm_avx512_microkernel.h"
#include "generator_x86_instructions.h"
#include "generator_common_x86.h"

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_avx512_kloop_kernel( libxsmm_generated_code*            io_generated_code,
                                                 const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                 const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                 const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                 const unsigned int                 i_m_blocking,
                                                 const unsigned int                 i_n_blocking,
                                                 const unsigned int                 i_k_blocking )
{
  unsigned int l_k = 0;
  unsigned int l_k_pack_factor = 1;
  unsigned int l_m_vector = ( i_m_blocking % i_micro_kernel_config->vector_length  == 0 ) ? i_m_blocking/i_micro_kernel_config->vector_length : (i_m_blocking/i_micro_kernel_config->vector_length)+1;
  unsigned int l_is_Ai8_Bbf16_gemm = ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
  unsigned int l_is_Ai4_Bf16_gemm = (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_INT4_VNNI2) > 0) &&
                                     ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
                                      (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) &&
                                      ((LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))))) ? 1 : 0;
  unsigned int l_is_Ai4_Bi8_gemm = libxsmm_x86_is_Ai4_Bi8_gemm(i_xgemm_desc);
  unsigned int l_is_Ai8_Bbf16_gemm_bf16fma = ((l_is_Ai8_Bbf16_gemm > 0) && (i_micro_kernel_config->vmul_instruction == LIBXSMM_X86_INSTR_VDPBF16PS)) ? 1 : 0;
  unsigned int l_is_i8_uu_ss_gemm = (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) &&
                                    ( ( ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) == 0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) == 0) ) ||
                                      ( ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) >  0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) >  0) ) );

  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A ) {
    l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) );
  }

  if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX ) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) == 0) && ( l_m_vector == 1 ) && (l_is_Ai8_Bbf16_gemm == 0) && (l_is_Ai4_Bf16_gemm == 0) && (l_is_Ai4_Bi8_gemm == 0) && (l_is_i8_uu_ss_gemm == 0) && (LIBXSMM_DATATYPE_BF8 != LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ) {
    libxsmm_generator_gemm_avx512_microkernel_fsdbcst( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
                                                       i_xgemm_desc, i_n_blocking, i_k_blocking );
  } else {
    void (*l_generator_microkernel)(libxsmm_generated_code*, const libxsmm_gp_reg_mapping*, const libxsmm_micro_kernel_config*,
                                    const libxsmm_gemm_descriptor*, const unsigned int, const unsigned int, const int);

    if (l_is_Ai8_Bbf16_gemm > 0 || l_is_Ai4_Bf16_gemm > 0 || ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) > 0) || l_is_Ai4_Bi8_gemm > 0) {
      if (l_is_Ai8_Bbf16_gemm_bf16fma > 0 || l_is_Ai4_Bf16_gemm > 0) {
        l_k_pack_factor = 2;
      } else if (l_is_Ai4_Bi8_gemm > 0) {
        l_k_pack_factor *= 2;
      } else {
        l_k_pack_factor = 1;
      }
      l_generator_microkernel = libxsmm_generator_gemm_avx512_microkernel_nofsdbcst;
    } else if ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ) {
      if ( (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) ) {
        l_generator_microkernel = libxsmm_generator_gemm_avx512_microkernel_m8_bf16_emu_nofsdbcst;
      } else if ( LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
        l_generator_microkernel = libxsmm_generator_gemm_avx512_microkernel_m8_bf8_emu_nofsdbcst;
      } else if ( l_is_i8_uu_ss_gemm != 0 ) {
        l_generator_microkernel = libxsmm_generator_gemm_avx512_microkernel_m8_i8_ss_uu_emu_nofsdbcst;
      } else {
        l_generator_microkernel = libxsmm_generator_gemm_avx512_microkernel_m8_nofsdbcst;
      }
    } else if ( (io_generated_code->arch != LIBXSMM_X86_AVX512_CPX) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) ) {
      l_generator_microkernel = libxsmm_generator_gemm_avx512_microkernel_bf16_emu_nofsdbcst;
    } else if ( LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
      l_generator_microkernel = libxsmm_generator_gemm_avx512_microkernel_bf8_emu_nofsdbcst;
    } else if ( l_is_i8_uu_ss_gemm != 0 ) {
      l_generator_microkernel = libxsmm_generator_gemm_avx512_microkernel_i8_ss_uu_emu_nofsdbcst;
    } else {
      l_generator_microkernel = libxsmm_generator_gemm_avx512_microkernel_nofsdbcst;
    }

    for ( l_k = 0; l_k < i_k_blocking; l_k += l_k_pack_factor) {
      l_generator_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
                              i_xgemm_desc, i_m_blocking, i_n_blocking,
                              ( i_k_blocking == (unsigned int)i_xgemm_desc->k ) ? (int)l_k : -1);
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_loadNinterleave_A_pair_k_i8_to_bf16( libxsmm_generated_code*            io_generated_code,
                                                                                                       const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                                                       const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                                                       const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                                                       unsigned int                       io_A_vreg,
                                                                                                       unsigned int                       i_tmp_vreg,
                                                                                                       unsigned int                       i_interleave_vreg,
                                                                                                       unsigned int                       i_m_blocking,
                                                                                                       unsigned int                       i_m  ) {
  unsigned int l_vec_scf_a;
  unsigned int l_is_Ai8_Bbf16_gemm_bf16fma = (i_micro_kernel_config->vmul_instruction == LIBXSMM_X86_INSTR_VDPBF16PS) ? 1 : 0;
  unsigned int l_i8toi32_cvt_instr = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0) ? LIBXSMM_X86_INSTR_VPMOVZXBD : LIBXSMM_X86_INSTR_VPMOVSXBD;
  unsigned int l_i16toi32_cvt_instr = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0) ? LIBXSMM_X86_INSTR_VPMOVZXWD : LIBXSMM_X86_INSTR_VPMOVSXWD;
  unsigned int l_a_vmove_instruction = ((i_m == i_m_blocking-1) && (i_micro_kernel_config->use_masking_a_c > 0)) ? LIBXSMM_X86_INSTR_VMOVDQU8 : l_i8toi32_cvt_instr;

  if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) > 0) {
    l_vec_scf_a = 1 + i_m;
  } else {
    l_vec_scf_a = 1;
  }
  /* Load "even k"  */
  libxsmm_x86_instruction_vec_move( io_generated_code,
      i_micro_kernel_config->instruction_set,
      l_a_vmove_instruction,
      i_gp_reg_mapping->gp_reg_a,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * i_m,
      i_micro_kernel_config->vector_name,
      io_A_vreg, ( i_m == (i_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 1, 0 );
  if (l_a_vmove_instruction == LIBXSMM_X86_INSTR_VMOVDQU8) {
    libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, l_i8toi32_cvt_instr, i_micro_kernel_config->vector_name, io_A_vreg, io_A_vreg);
  }

  if (l_is_Ai8_Bbf16_gemm_bf16fma > 0) {
    /* Load "odd" k  */
    libxsmm_x86_instruction_vec_move( io_generated_code,
        i_micro_kernel_config->instruction_set,
        l_a_vmove_instruction,
        i_gp_reg_mapping->gp_reg_a,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * i_m + i_xgemm_desc->lda * i_micro_kernel_config->datatype_size_in,
        i_micro_kernel_config->vector_name,
        i_tmp_vreg, ( i_m == (i_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 1, 0 );
    if (l_a_vmove_instruction == LIBXSMM_X86_INSTR_VMOVDQU8) {
      libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, l_i8toi32_cvt_instr, i_micro_kernel_config->vector_name, i_tmp_vreg, i_tmp_vreg);
    }
  }

  libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTDQ2PS, i_micro_kernel_config->vector_name, io_A_vreg, io_A_vreg );
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMULPS, i_micro_kernel_config->vector_name, io_A_vreg, l_vec_scf_a, io_A_vreg);
  if (l_is_Ai8_Bbf16_gemm_bf16fma > 0) {
    libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, io_A_vreg, io_A_vreg );
  } else {
    libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name, io_A_vreg, io_A_vreg, i_tmp_vreg, i_interleave_vreg, 6, 7, 0 );
  }

  if (l_is_Ai8_Bbf16_gemm_bf16fma > 0) {
    libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTDQ2PS, i_micro_kernel_config->vector_name, i_tmp_vreg, i_tmp_vreg );
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMULPS, i_micro_kernel_config->vector_name, i_tmp_vreg, l_vec_scf_a, i_tmp_vreg);
    libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, i_tmp_vreg, i_tmp_vreg);

    /* Interleave "odd" and "even" A vregs */
    libxsmm_x86_instruction_vec_compute_3reg(io_generated_code, LIBXSMM_X86_INSTR_VPERMT2W, i_micro_kernel_config->vector_name, i_tmp_vreg, i_interleave_vreg, io_A_vreg);
  } else {
    libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, l_i16toi32_cvt_instr, i_micro_kernel_config->vector_name, io_A_vreg, io_A_vreg);
    libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, i_micro_kernel_config->vector_name, io_A_vreg, io_A_vreg, 16);
  }
}

LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_process_vreg_A_for_i4i8 ( libxsmm_generated_code*            io_generated_code,
                                                                                  const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                                  const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                                  unsigned int                       i_m,
                                                                                  unsigned int                       i_m_blocking,
                                                                                  unsigned int                       io_A_vreg ) {
  unsigned int l_subtract_zpt = 0, l_vec_zpt_a = 3;
  unsigned int tmp_lo_vreg = io_A_vreg;
  unsigned int tmp_hi_vreg = io_A_vreg + i_m_blocking;
  unsigned int l_vreg_mask_0f = 0;
  unsigned int l_vreg_mask_f0 = 1;
  if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0 || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0) {
    l_subtract_zpt = 1;
    l_vec_zpt_a = 3 + i_m;
  }
  /* Mask higher 4 bits*/
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,  LIBXSMM_X86_INSTR_VPANDD, i_micro_kernel_config->vector_name, io_A_vreg, l_vreg_mask_f0, tmp_hi_vreg);
  /* Shift and Subtract zero pts */
  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSRLD_I, i_micro_kernel_config->vector_name, tmp_hi_vreg, tmp_hi_vreg, 4);
  if (l_subtract_zpt > 0) {
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPSUBB, i_micro_kernel_config->vector_name, l_vec_zpt_a, tmp_hi_vreg, tmp_hi_vreg);
  }
  /* Mask lower 4 bits*/
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,  LIBXSMM_X86_INSTR_VPANDD, i_micro_kernel_config->vector_name, io_A_vreg, l_vreg_mask_0f, tmp_lo_vreg);
  /* Subtract zero pts */
  if (l_subtract_zpt > 0) {
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPSUBB, i_micro_kernel_config->vector_name, l_vec_zpt_a, tmp_lo_vreg, tmp_lo_vreg);
  }
}

LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_process_vreg_A( libxsmm_generated_code*            io_generated_code,
                                                                                  const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                                  const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                                  char                               i_vname_cvt,
                                                                                  unsigned int                       i_is_Ai8_Bf16_gemm,
                                                                                  unsigned int                       i_is_Abf8_Bf16_gemm,
                                                                                  unsigned int                       i_is_Af16_Bf16_gemm,
                                                                                  unsigned int                       i_use_f16_replacement_fma,
                                                                                  unsigned int                       i_use_f32_compute_with_f16_inp,
                                                                                  unsigned int                       i_m,
                                                                                  unsigned int                       i_m_blocking,
                                                                                  unsigned int                       io_A_vreg ) {
  unsigned int l_vec_scf_a = 0, l_vec_zpt_a = 0;
  unsigned int l_is_Ai4_Bf16_gemm = (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_INT4_VNNI2) > 0) &&
                                     ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
                                      (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) &&
                                      ((LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))))) ? 1 : 0;
  unsigned int l_subtract_zpt = 0;
  unsigned int lut_vreg = 0;

  if (i_is_Ai8_Bf16_gemm > 0) {
    if (l_is_Ai4_Bf16_gemm > 0) {
      unsigned int l_use_perm_based_cvt = (io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_SKX || io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CLX  || io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CPX) ? 0 : 1;
      unsigned int tmp_lo_vreg = io_A_vreg;
      unsigned int tmp_hi_vreg = io_A_vreg + i_m_blocking;
      if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) > 0) {
        l_vec_scf_a = 2+i_m;
      } else {
        l_vec_scf_a = 2;
      }
      if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0 || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0) {
        l_subtract_zpt = 1;
        l_vec_zpt_a = l_vec_scf_a + i_m_blocking;
      }
      if (l_use_perm_based_cvt > 0) {
        if (i_use_f32_compute_with_f16_inp > 0 || i_use_f16_replacement_fma > 0) {
          unsigned int l_i8toi32_cvt_instr = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0) ? LIBXSMM_X86_INSTR_VPMOVZXBD : LIBXSMM_X86_INSTR_VPMOVSXBD;
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, l_i8toi32_cvt_instr, i_micro_kernel_config->vector_name, io_A_vreg, tmp_lo_vreg);
          libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSRAD_I, i_micro_kernel_config->vector_name, tmp_lo_vreg, tmp_hi_vreg, 4);
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPERMD, i_micro_kernel_config->vector_name, lut_vreg, tmp_lo_vreg, tmp_lo_vreg);
          if (l_subtract_zpt > 0) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VSUBPS, i_micro_kernel_config->vector_name, l_vec_zpt_a, tmp_lo_vreg, tmp_lo_vreg);
            if (i_use_f16_replacement_fma > 0) {
              libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, tmp_lo_vreg, tmp_lo_vreg, 0,
                  (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, tmp_lo_vreg, tmp_lo_vreg);
            }
          }
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMULPS, i_micro_kernel_config->vector_name, tmp_lo_vreg, l_vec_scf_a, tmp_lo_vreg);
          if (i_use_f16_replacement_fma > 0) {
            libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, tmp_lo_vreg, tmp_lo_vreg, 0,
                (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, tmp_lo_vreg, tmp_lo_vreg);
          }
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPERMD, i_micro_kernel_config->vector_name, lut_vreg, tmp_hi_vreg, tmp_hi_vreg);
          if (l_subtract_zpt > 0) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VSUBPS, i_micro_kernel_config->vector_name, l_vec_zpt_a, tmp_hi_vreg, tmp_hi_vreg);
            if (i_use_f16_replacement_fma > 0) {
              libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, tmp_hi_vreg, tmp_hi_vreg, 0,
                  (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, tmp_hi_vreg, tmp_hi_vreg);
            }
          }
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMULPS, i_micro_kernel_config->vector_name, tmp_hi_vreg, l_vec_scf_a, tmp_hi_vreg);
          if (i_use_f16_replacement_fma > 0) {
            libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, tmp_hi_vreg, tmp_hi_vreg, 0,
                (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, tmp_hi_vreg, tmp_hi_vreg);
          }
        } else {
          unsigned int l_i8toi16_cvt_instr = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0) ? LIBXSMM_X86_INSTR_VPMOVZXBW : LIBXSMM_X86_INSTR_VPMOVSXBW;
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, l_i8toi16_cvt_instr, i_micro_kernel_config->vector_name, io_A_vreg, tmp_lo_vreg);
          libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSRAW_I, i_micro_kernel_config->vector_name, tmp_lo_vreg, tmp_hi_vreg, 4);
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPERMW, i_micro_kernel_config->vector_name, lut_vreg, tmp_lo_vreg, tmp_lo_vreg);
          if (l_subtract_zpt > 0) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VSUBPH, i_micro_kernel_config->vector_name, l_vec_zpt_a, tmp_lo_vreg, tmp_lo_vreg);
          }
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMULPH, i_micro_kernel_config->vector_name, tmp_lo_vreg, l_vec_scf_a, tmp_lo_vreg);
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPERMW, i_micro_kernel_config->vector_name, lut_vreg, tmp_hi_vreg, tmp_hi_vreg);
          if (l_subtract_zpt > 0) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VSUBPH, i_micro_kernel_config->vector_name, l_vec_zpt_a, tmp_hi_vreg, tmp_hi_vreg);
          }
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMULPH, i_micro_kernel_config->vector_name, tmp_hi_vreg, l_vec_scf_a, tmp_hi_vreg);
        }
      } else {
        unsigned int l_vreg_mask_0f = 0;
        unsigned int l_vreg_mask_f0 = 1;
        unsigned int l_i8toi32_cvt_instr = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0) ? LIBXSMM_X86_INSTR_VPMOVZXBD : LIBXSMM_X86_INSTR_VPMOVSXBD;
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,  LIBXSMM_X86_INSTR_VPANDD, 'x', io_A_vreg, l_vreg_mask_f0, tmp_hi_vreg);
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, l_i8toi32_cvt_instr, i_micro_kernel_config->vector_name, tmp_hi_vreg, tmp_hi_vreg);
        libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSRAD_I, i_micro_kernel_config->vector_name, tmp_hi_vreg, tmp_hi_vreg, 4);
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTDQ2PS, i_micro_kernel_config->vector_name, tmp_hi_vreg, tmp_hi_vreg );
        if (l_subtract_zpt > 0) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VSUBPS, i_micro_kernel_config->vector_name, l_vec_zpt_a, tmp_hi_vreg, tmp_hi_vreg);
          if (i_use_f16_replacement_fma > 0) {
            libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, tmp_hi_vreg, tmp_hi_vreg, 0,
                (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, tmp_hi_vreg, tmp_hi_vreg);
          }
        }
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMULPS, i_micro_kernel_config->vector_name, tmp_hi_vreg, l_vec_scf_a, tmp_hi_vreg);
        if (i_use_f16_replacement_fma > 0) {
          libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, tmp_hi_vreg, tmp_hi_vreg, 0,
              (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, tmp_hi_vreg, tmp_hi_vreg);
        }
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,  LIBXSMM_X86_INSTR_VPANDD, 'x', io_A_vreg, l_vreg_mask_0f, tmp_lo_vreg);
        libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLW_I, 'x', tmp_lo_vreg, tmp_lo_vreg, 4);
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, l_i8toi32_cvt_instr, i_micro_kernel_config->vector_name, tmp_lo_vreg, tmp_lo_vreg);
        libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSRAD_I, i_micro_kernel_config->vector_name, tmp_lo_vreg, tmp_lo_vreg, 4);
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTDQ2PS, i_micro_kernel_config->vector_name, tmp_lo_vreg, tmp_lo_vreg );
        if (l_subtract_zpt > 0) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VSUBPS, i_micro_kernel_config->vector_name, l_vec_zpt_a, tmp_lo_vreg, tmp_lo_vreg);
          if (i_use_f16_replacement_fma > 0) {
            libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, tmp_lo_vreg, tmp_lo_vreg, 0,
                (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, tmp_lo_vreg, tmp_lo_vreg);
          }
        }
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMULPS, i_micro_kernel_config->vector_name, tmp_lo_vreg, l_vec_scf_a, tmp_lo_vreg);
        if (i_use_f16_replacement_fma > 0) {
          libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name, tmp_lo_vreg, tmp_lo_vreg, 0,
              (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, tmp_lo_vreg, tmp_lo_vreg);
        }
      }
    } else {
      if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) > 0) {
        l_vec_scf_a = i_m;
      } else {
        l_vec_scf_a = 0;
      }
      if (i_use_f16_replacement_fma > 0) {
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VPMOVSXBD, i_vname_cvt, io_A_vreg, io_A_vreg);
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTDQ2PS, i_vname_cvt, io_A_vreg, io_A_vreg );
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMULPS, i_vname_cvt, io_A_vreg, l_vec_scf_a, io_A_vreg);
        libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_vname_cvt, io_A_vreg, io_A_vreg, 0,
            (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_vname_cvt, io_A_vreg, io_A_vreg);
      } else {
        if (i_use_f32_compute_with_f16_inp > 0) {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VPMOVSXBD, i_vname_cvt, io_A_vreg, io_A_vreg);
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTDQ2PS, i_vname_cvt, io_A_vreg, io_A_vreg );
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMULPS, i_vname_cvt, io_A_vreg, l_vec_scf_a, io_A_vreg);
        } else {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VPMOVSXBW, i_micro_kernel_config->vector_name, io_A_vreg, io_A_vreg);
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTW2PH, i_micro_kernel_config->vector_name, io_A_vreg, io_A_vreg );
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMULPH, i_micro_kernel_config->vector_name, io_A_vreg, l_vec_scf_a, io_A_vreg);
        }
      }
    }
  }

  if (i_is_Af16_Bf16_gemm > 0 && (i_use_f16_replacement_fma > 0 || i_use_f32_compute_with_f16_inp > 0)) {
    libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_vname_cvt, io_A_vreg, io_A_vreg );
  }

  if (i_is_Abf8_Bf16_gemm > 0) {
    if (i_use_f16_replacement_fma > 0 || i_use_f32_compute_with_f16_inp > 0) {
      libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VPMOVSXBW, i_vname_cvt, io_A_vreg, io_A_vreg);
      libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLW_I, i_vname_cvt, io_A_vreg, io_A_vreg, 8 );
      libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_vname_cvt, io_A_vreg, io_A_vreg);
    } else {
      libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VPMOVSXBW, i_micro_kernel_config->vector_name, io_A_vreg, io_A_vreg);
      libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLW_I, i_micro_kernel_config->vector_name, io_A_vreg, io_A_vreg, 8 );
    }
  }

}

LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_nofsdbcst( libxsmm_generated_code*            io_generated_code,
                                                                             const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                             const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                             const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                             const unsigned int                 i_m_blocking,
                                                                             const unsigned int                 i_n_blocking,
                                                                             const int                          i_offset )
{
  /* deriving register blocking from kernel config */
  unsigned int l_m_blocking = ( i_m_blocking % i_micro_kernel_config->vector_length  == 0 ) ? i_m_blocking/i_micro_kernel_config->vector_length : (i_m_blocking/i_micro_kernel_config->vector_length)+1;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  unsigned int l_k = 0;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = i_micro_kernel_config->vector_reg_count - (i_n_blocking * l_m_blocking);
  unsigned int l_vreg_ab_offset = 0;
  /* temp variable for b-offset to handle no-trans/trans B */
  int l_b_offset = 0;
  /* k packing factor for VNNI */
  unsigned int l_k_pack_factor = 1;
  unsigned int l_is_Abf8_Bf16_gemm = ((LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
                                     (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
  unsigned int l_is_Ai8_Bf16_gemm = ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
                                     (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
  unsigned int l_is_Ai4_Bf16_gemm = (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_INT4_VNNI2) > 0) &&
                                     ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
                                      (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) &&
                                      ((LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))))) ? 1 : 0;
  unsigned int l_is_Ai4_Bi8_gemm = libxsmm_x86_is_Ai4_Bi8_gemm(i_xgemm_desc);
  unsigned int l_k_iters = (l_is_Ai4_Bf16_gemm > 0 || l_is_Ai4_Bi8_gemm > 0) ? 2 : 1;
  unsigned int l_is_Af16_Bf16_gemm =((LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
                                     (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
  unsigned int l_is_Ai8_Bbf16_gemm = ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
  unsigned int l_is_Ai8_Bbf16_gemm_bf16fma = ((l_is_Ai8_Bbf16_gemm > 0) && (i_micro_kernel_config->vmul_instruction == LIBXSMM_X86_INSTR_VDPBF16PS)) ? 1 : 0;
  unsigned int l_use_f16_replacement_fma = (((l_is_Ai8_Bf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0 || l_is_Af16_Bf16_gemm > 0) && io_generated_code->arch < LIBXSMM_X86_AVX512_SPR && LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
  unsigned int l_use_f32_compute_with_f16_inp = (((l_is_Ai8_Bf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0 || l_is_Af16_Bf16_gemm > 0) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;

  char vname_cvt = i_micro_kernel_config->vector_name;

  if (l_is_Ai8_Bf16_gemm > 0) {
    if (l_is_Ai4_Bf16_gemm > 0) {
      l_vreg_ab_offset = 3;
      if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) > 0) {
        l_vreg_ab_offset = 2 + l_m_blocking;
      }
    } else {
      l_vreg_ab_offset = 1;
      if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) > 0) {
        l_vreg_ab_offset = l_m_blocking;
      }
    }
    if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0 || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0) {
      l_vreg_ab_offset += l_m_blocking;
    }
  }

  if (l_is_Ai8_Bbf16_gemm > 0) {
    l_vreg_ab_offset = 2;
    if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) > 0) {
      l_vreg_ab_offset = 1 + l_m_blocking;
    }
  }

  if (l_is_Ai4_Bi8_gemm > 0) {
    l_vreg_ab_offset = 3;
    if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0 || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0) {
      l_vreg_ab_offset += l_m_blocking;
    }
  }

#if !defined(NDEBUG)
  if ( (i_n_blocking > 30) || (i_n_blocking < 1) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  if ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX && io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
      if (l_is_Ai8_Bbf16_gemm > 0) {
        if ( ((l_m_blocking*i_n_blocking) + l_m_blocking + 1) > 32 ) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
          return;
        }
      } else if (l_is_Ai4_Bf16_gemm > 0)  {
        int l_m_scf_vregs = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) == 0) ?  3 + l_m_blocking : 2 + 2*l_m_blocking;
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0) {
          l_m_scf_vregs += l_m_blocking;
        }
        if ( ((l_m_blocking*i_n_blocking) + l_m_blocking + 1 + l_m_scf_vregs) > 32 ) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
          return;
        }
      } else {
        if ( ((l_m_blocking*i_n_blocking) + i_n_blocking + 1) > 32 ) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
          return;
        }
      }
      if ( (l_m_blocking < 1) || (l_m_blocking > 8) ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
        return;
      }
  } else {
      if ( (l_m_blocking < 1) || (l_m_blocking > 4) ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
        return;
      }
      if ( (((l_m_blocking*i_n_blocking) + l_m_blocking + 1) > 32) && (i_n_blocking < 7) ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
        return;
      }
  }
#endif

  /* for VNNI we are stepping through to pack ks */
  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A ) {
    l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) );
  }

  /* load column vectors of A upfront */
  for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
    const char *const l_env_a_k_pf_dist = getenv("LIBXSMM_GEMM_K_A_PF_DIST");
    unsigned l_a_k_pf_dist = (l_env_a_k_pf_dist == 0) ? 0 : atoi(l_env_a_k_pf_dist);
    char l_a_vname = (l_is_Ai8_Bf16_gemm == 0 && l_is_Abf8_Bf16_gemm == 0) ? i_micro_kernel_config->vector_name : ((l_use_f32_compute_with_f16_inp) ? (i_micro_kernel_config->vector_name == 'z' ? 'x' : 'x') : ((l_use_f16_replacement_fma > 0) ? (i_micro_kernel_config->vector_name == 'z' ? 'x' : 'x') : (i_micro_kernel_config->vector_name == 'z' ? 'y' : 'x')));
    unsigned int l_a_vmove_instruction = ((l_is_Ai8_Bf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && (l_m != (l_m_blocking - 1)) ) ? LIBXSMM_X86_INSTR_VMOVSD : i_micro_kernel_config->a_vmove_instruction;

    if (l_is_Af16_Bf16_gemm > 0) {
      if ( l_use_f32_compute_with_f16_inp > 0 || l_use_f16_replacement_fma > 0 ) {
        l_a_vname = (i_micro_kernel_config->vector_name == 'z') ? 'y' : 'x';
      }
    }

    if (l_use_f16_replacement_fma > 0) {
      vname_cvt = (l_a_vname == 'y') ? 'z' : ((l_a_vname == 'x') ? ((io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 'y' : 'z') : i_micro_kernel_config->vector_name);
    }

    if (l_is_Ai8_Bbf16_gemm > 0) {
      libxsmm_generator_gemm_avx512_microkernel_loadNinterleave_A_pair_k_i8_to_bf16( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc,
          1+l_m+l_vreg_ab_offset, l_vreg_ab_offset, 0, l_m_blocking, l_m );
    } else {
      if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) {
        unsigned int l_current_mask_reg = (3+l_m)%8;
        char decompress_vname = (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) > 0)) ? (i_micro_kernel_config->vector_name == 'z' ? 'y' : 'x') : i_micro_kernel_config->vector_name;
        /* Load bit mask for current expand operation */
        libxsmm_x86_instruction_mask_move_mem( io_generated_code,
            (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype)) ? LIBXSMM_X86_INSTR_KMOVD_LD : LIBXSMM_X86_INSTR_KMOVW_LD,
            i_gp_reg_mapping->gp_reg_bitmap_a,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (i_micro_kernel_config->vector_length/8) * l_m * l_k_pack_factor,
            l_current_mask_reg );

        libxsmm_x86_instruction_prefetch(io_generated_code,
            LIBXSMM_X86_INSTR_PREFETCHT0,
            i_gp_reg_mapping->gp_reg_bitmap_a,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (i_micro_kernel_config->vector_length/8) * l_m * l_k_pack_factor +  16 * 64);

        /* Expand operation */
        libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8( io_generated_code,
                                                     (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) || LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype)) ? LIBXSMM_X86_INSTR_VPEXPANDW : LIBXSMM_X86_INSTR_VPEXPANDD,
                                                     decompress_vname,
                                                     i_gp_reg_mapping->gp_reg_a,
                                                     i_gp_reg_mapping->gp_reg_decompressed_elts,
                                                     i_micro_kernel_config->datatype_size_in,
                                                     0,
                                                     0,
                                                     LIBXSMM_X86_VEC_REG_UNDEF,
                                                     1+l_m+l_vreg_ab_offset,
                                                     l_current_mask_reg,
                                                     1,
                                                     0);
        if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype)) {
          /* Convert bf16 to f32 */
          libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, 1+l_m+l_vreg_ab_offset, 1+l_m+l_vreg_ab_offset);
        }

        libxsmm_x86_instruction_prefetch(io_generated_code,
            LIBXSMM_X86_INSTR_PREFETCHT0,
            i_gp_reg_mapping->gp_reg_a,
            i_gp_reg_mapping->gp_reg_decompressed_elts, i_micro_kernel_config->datatype_size_in,
            16 * 64);

        /* Move zmm to reg */
        libxsmm_x86_instruction_mask_move( io_generated_code,
          (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype)) ? LIBXSMM_X86_INSTR_KMOVD_GPR_ST : LIBXSMM_X86_INSTR_KMOVW_GPR_ST,
          i_gp_reg_mapping->gp_reg_popcnt,
          l_current_mask_reg );
        /* Popcount */
        libxsmm_x86_instruction_alu_reg( io_generated_code,
            LIBXSMM_X86_INSTR_POPCNT,
            i_gp_reg_mapping->gp_reg_popcnt,
            i_gp_reg_mapping->gp_reg_popcnt);
        /* Adjust count of decompressed elements */
        libxsmm_x86_instruction_alu_reg( io_generated_code,
            LIBXSMM_X86_INSTR_ADDQ,
            i_gp_reg_mapping->gp_reg_popcnt,
            i_gp_reg_mapping->gp_reg_decompressed_elts);
      } else {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            l_a_vmove_instruction,
            i_gp_reg_mapping->gp_reg_a,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
            l_a_vname,
            1+l_m+l_vreg_ab_offset, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 1, 0 );

        if ((((i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor) % 64 == 0) && (l_a_k_pf_dist > 0)) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
              LIBXSMM_X86_INSTR_PREFETCHT0,
              i_gp_reg_mapping->gp_reg_a,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor + l_a_k_pf_dist * i_xgemm_desc->lda * i_micro_kernel_config->datatype_size_in);
        }
      }
    }

    if (l_is_Ai4_Bi8_gemm > 0) {
      /* Process vreg A in case of i4i8 */
      libxsmm_generator_gemm_avx512_microkernel_process_vreg_A_for_i4i8 (io_generated_code, i_micro_kernel_config, i_xgemm_desc,
          l_m, l_m_blocking, 1+l_m+l_vreg_ab_offset);
    } else {
      /* Process vreg A in case of f16/i8 */
      libxsmm_generator_gemm_avx512_microkernel_process_vreg_A( io_generated_code, i_micro_kernel_config, i_xgemm_desc,
          vname_cvt, l_is_Ai8_Bf16_gemm, l_is_Abf8_Bf16_gemm, l_is_Af16_Bf16_gemm, l_use_f16_replacement_fma, l_use_f32_compute_with_f16_inp, l_m, l_m_blocking, 1+l_m+l_vreg_ab_offset);
    }

    /* current A prefetch, next rows for the current column */
    if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2_AHEAD || i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD ) {
      libxsmm_x86_instruction_prefetch( io_generated_code,
          LIBXSMM_X86_INSTR_PREFETCHT1,
          i_gp_reg_mapping->gp_reg_a,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          ((i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor) + (64 * l_m_blocking) );
    }

    /* prefetch a different A matrix provided by the prefetch pointers */
    if ( (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2) || (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C) ) {
      libxsmm_x86_instruction_prefetch( io_generated_code,
          LIBXSMM_X86_INSTR_PREFETCHT1,
          i_gp_reg_mapping->gp_reg_a_prefetch,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor);
    }
  }

  for ( l_k = 0; l_k < l_k_iters; l_k++ ) {
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      char l_b_vname = i_micro_kernel_config->vector_name;
      unsigned int l_b_vmove_instruction = i_micro_kernel_config->b_vmove_instruction;

      if (l_is_Af16_Bf16_gemm > 0 || l_is_Ai8_Bf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0) {
        if ( l_use_f32_compute_with_f16_inp > 0 || l_use_f16_replacement_fma > 0 ) {
          l_b_vname = (i_micro_kernel_config->vector_name == 'z') ? 'y' : 'x';
        }
      }

      if ((l_is_Ai8_Bbf16_gemm > 0) && (l_is_Ai8_Bbf16_gemm_bf16fma == 0)) {
        l_b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTW;
      }

      if ( i_offset != (-1) ) {
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset * i_xgemm_desc->ldb) + (l_n * i_micro_kernel_config->datatype_size_in2) + (i_micro_kernel_config->datatype_size_in2 * l_k * i_xgemm_desc->ldb);
        } else {
          l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset) + (i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in2 + l_k * l_k_pack_factor * i_micro_kernel_config->datatype_size_in2);
        }

        l_b_vname = (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) > 0)) ? (l_b_vname == 'z' ? 'y' : 'x') : l_b_vname;
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            l_b_vmove_instruction,
            i_gp_reg_mapping->gp_reg_b,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            l_b_offset,
            l_b_vname,
            l_vreg_ab_offset, 0, 1, 0 );

        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) > 0 &&  LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype)) {
          /* Convert bf16 to f32 */
          libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, l_vreg_ab_offset, l_vreg_ab_offset);
        }

        if ((l_is_Af16_Bf16_gemm > 0 || l_is_Ai8_Bf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0) && (l_use_f16_replacement_fma > 0 || l_use_f32_compute_with_f16_inp > 0)) {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, vname_cvt, l_vreg_ab_offset, l_vreg_ab_offset );
        }

        if ( l_is_Ai8_Bbf16_gemm > 0 && l_is_Ai8_Bbf16_gemm_bf16fma == 0 ) {
          libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, i_micro_kernel_config->vector_name, l_vreg_ab_offset, l_vreg_ab_offset, 16);
        }

        if (l_n == i_n_blocking - 1) {
          if (i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BL1) {
            if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
              l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset * i_xgemm_desc->ldb);
            } else {
              l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset);
            }
            libxsmm_x86_instruction_prefetch(io_generated_code,
                LIBXSMM_X86_INSTR_PREFETCHT0,
                i_gp_reg_mapping->gp_reg_b,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                l_b_offset + 16 * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2);
          }
        }
      } else {
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          unsigned int l_k_pack_advance = (l_is_Ai8_Bbf16_gemm > 0) ? ((l_is_Ai8_Bbf16_gemm_bf16fma == 0) ? 1 : 2) : l_k_pack_factor;
          l_b_offset = l_n * i_micro_kernel_config->datatype_size_in2 * l_k_pack_advance + (i_micro_kernel_config->datatype_size_in2 * l_k * i_xgemm_desc->ldb) ;
        } else {
          unsigned int l_k_pack_advance = (l_is_Ai8_Bbf16_gemm > 0) ? ((l_is_Ai8_Bbf16_gemm_bf16fma == 0) ? 1 : 2) : l_k_pack_factor;
          l_b_offset = i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in2 + l_k * l_k_pack_advance * i_micro_kernel_config->datatype_size_in2;
        }

        l_b_vname = (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) > 0)) ? (l_b_vname == 'z' ? 'y' : 'x') : l_b_vname;
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            l_b_vmove_instruction,
            i_gp_reg_mapping->gp_reg_b,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            l_b_offset,
            l_b_vname,
            l_vreg_ab_offset, 0, 1, 0 );

        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) > 0 &&  LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype)) {
          /* Convert bf16 to f32 */
          libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, l_vreg_ab_offset, l_vreg_ab_offset);
        }

        if ((l_is_Af16_Bf16_gemm > 0 || l_is_Ai8_Bf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0) && (l_use_f16_replacement_fma > 0 || l_use_f32_compute_with_f16_inp > 0)) {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, vname_cvt, l_vreg_ab_offset, l_vreg_ab_offset );
        }

        if ( l_is_Ai8_Bbf16_gemm > 0 && l_is_Ai8_Bbf16_gemm_bf16fma == 0 ) {
          libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, i_micro_kernel_config->vector_name, l_vreg_ab_offset, l_vreg_ab_offset, 16);
        }

        if (l_n == i_n_blocking - 1) {
          if (i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BL1) {
            libxsmm_x86_instruction_prefetch(io_generated_code,
                LIBXSMM_X86_INSTR_PREFETCHT0,
                i_gp_reg_mapping->gp_reg_b,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                16 * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2);
          }

          /* handle trans B */
          if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
            l_b_offset = i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2 * l_k_iters;
          } else {
            unsigned int l_k_pack_advance = (l_is_Ai8_Bbf16_gemm > 0) ? ((l_is_Ai8_Bbf16_gemm_bf16fma == 0) ? 1 : 2) : l_k_pack_factor;
            l_b_offset = i_micro_kernel_config->datatype_size_in2 * l_k_pack_advance * l_k_iters;
          }

          if (l_k == (l_k_iters-1)) {
            libxsmm_x86_instruction_alu_imm( io_generated_code,
                i_micro_kernel_config->alu_add_instruction,
                i_gp_reg_mapping->gp_reg_b,
                l_b_offset );
          }
        }
      }

      /* In case of batch reduce try to prefetch a few more columns ahead for A... */
      if ((l_n < l_m_blocking)  && ((i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BRGEMM_OOB) > 0) && (LIBXSMM_DATATYPE_I8 != LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype)) && (LIBXSMM_DATATYPE_BF16 != LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype)) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE))) {
        unsigned int pf_a_cols_ahead = 16;
        if (i_xgemm_desc->lda == 1024) {
          pf_a_cols_ahead = 4;
        }
        libxsmm_x86_instruction_prefetch( io_generated_code,
            LIBXSMM_X86_INSTR_PREFETCHT0,
            i_gp_reg_mapping->gp_reg_a,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_n * l_k_pack_factor + pf_a_cols_ahead * i_xgemm_desc->lda * i_micro_kernel_config->datatype_size_in);
      }

      for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
        /* post increment early */
        if ( (l_m == 0) && (l_n == i_n_blocking-1) && (l_k == 0)) {
          unsigned int l_k_pack_advance = (l_is_Ai8_Bbf16_gemm > 0) ? ((l_is_Ai8_Bbf16_gemm_bf16fma == 0) ? 1 : 2): l_k_pack_factor;

          if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) {
            libxsmm_x86_instruction_alu_imm( io_generated_code,
                i_micro_kernel_config->alu_add_instruction,
                i_gp_reg_mapping->gp_reg_bitmap_a,
                (long long)(i_xgemm_desc->lda/8) * l_k_pack_advance);
          } else {
            libxsmm_x86_instruction_alu_imm( io_generated_code,
                i_micro_kernel_config->alu_add_instruction,
                i_gp_reg_mapping->gp_reg_a,
                (long long)i_xgemm_desc->lda * i_micro_kernel_config->datatype_size_in * l_k_pack_advance);
          }

          /* if we prefetch next A into L2, we need to also increment the prefetch pointer */
          if ( (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2) || (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C) ) {
            libxsmm_x86_instruction_alu_imm( io_generated_code,
                i_micro_kernel_config->alu_add_instruction,
                i_gp_reg_mapping->gp_reg_a_prefetch,
                (long long)i_xgemm_desc->lda * i_micro_kernel_config->datatype_size_in * l_k_pack_factor);
          }
        }
        /* issue fma */
        if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
          if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                i_micro_kernel_config->vmul_instruction,
                i_micro_kernel_config->vector_name,
                l_vreg_ab_offset,
                1+l_m+l_vreg_ab_offset + l_k * l_m_blocking,
                l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
          } else if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0 ) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                i_micro_kernel_config->vmul_instruction,
                i_micro_kernel_config->vector_name,
                1+l_m+l_vreg_ab_offset + l_k * l_m_blocking,
                l_vreg_ab_offset,
                l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
          } else {
            /* should not happen */
          }
        } else if (l_use_f16_replacement_fma > 0) {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, vname_cvt, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
              i_micro_kernel_config->vmul_instruction,
              vname_cvt,
              1+l_m+l_vreg_ab_offset + l_k * l_m_blocking,
              l_vreg_ab_offset,
              l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
          libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, vname_cvt, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), 0,
                                                                  (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
        } else {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
              i_micro_kernel_config->vmul_instruction,
              i_micro_kernel_config->vector_name,
              1+l_m+l_vreg_ab_offset + l_k * l_m_blocking,
              l_vreg_ab_offset,
              l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
        }
      }
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_m8_nofsdbcst( libxsmm_generated_code*            io_generated_code,
                                                                                const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                                const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                                const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                                const unsigned int                 i_m_blocking,
                                                                                const unsigned int                 i_n_blocking,
                                                                                const int                          i_offset )
{
  /* deriving register blocking from kernel config */
  unsigned int l_m_blocking = ( i_m_blocking % i_micro_kernel_config->vector_length  == 0 ) ? i_m_blocking/i_micro_kernel_config->vector_length : (i_m_blocking/i_micro_kernel_config->vector_length)+1;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = i_micro_kernel_config->vector_reg_count - (i_n_blocking * l_m_blocking);
  /* temp variable for b-offset to handle no-trans/trans B */
  int l_b_offset = 0;
  /* k packing factor for VNNI */
  unsigned int l_k_pack_factor = 1;
  unsigned int l_vreg_ab_offset = 0;
  unsigned int l_is_Ai8_Bf16_gemm = ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
                                     (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
  unsigned int l_is_Abf8_Bf16_gemm = ((LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
                                     (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
  unsigned int l_is_Af16_Bf16_gemm =((LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
                                     (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
  unsigned int l_use_f16_replacement_fma = (((l_is_Ai8_Bf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0 || l_is_Af16_Bf16_gemm > 0) && io_generated_code->arch < LIBXSMM_X86_AVX512_SPR && LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
  unsigned int l_use_f32_compute_with_f16_inp = (((l_is_Ai8_Bf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0 || l_is_Af16_Bf16_gemm > 0) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
  char vname_cvt = i_micro_kernel_config->vector_name;

  if (l_is_Ai8_Bf16_gemm > 0) {
    l_vreg_ab_offset = 1;
    if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) > 0) {
      l_vreg_ab_offset = l_m_blocking;
    }
  }

  /* for VNNI we are stepping through to pack ks */
  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A ) {
    l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) );
  }

  /* load column vectors of A upfront */
  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    char l_b_vname = i_micro_kernel_config->vector_name;

    if (l_is_Af16_Bf16_gemm > 0 || l_is_Ai8_Bf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0) {
      if ( l_use_f32_compute_with_f16_inp > 0 || l_use_f16_replacement_fma > 0 ) {
        l_b_vname = (i_micro_kernel_config->vector_name == 'z') ? 'y' : 'x';
      }
    }

    if (l_use_f16_replacement_fma > 0) {
      vname_cvt = (l_b_vname == 'y') ? 'z' : ((l_b_vname == 'x') ? 'y' : i_micro_kernel_config->vector_name);
    }

    if ( i_offset != (-1) ) {
      /* handle trans B */
      if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
        l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset * i_xgemm_desc->ldb) + (l_n * i_micro_kernel_config->datatype_size_in2);
      } else {
        l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset) + (i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in2);
      }

      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->b_vmove_instruction,
          i_gp_reg_mapping->gp_reg_b,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          l_b_offset,
          l_b_vname,
          l_vreg_ab_offset+1+l_n, 0, 1, 0 );

      if ((l_is_Af16_Bf16_gemm > 0 || l_is_Ai8_Bf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0) && (l_use_f16_replacement_fma > 0 || l_use_f32_compute_with_f16_inp > 0)) {
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, vname_cvt, l_vreg_ab_offset+1+l_n, l_vreg_ab_offset+1+l_n );
      }

      if (l_n == i_n_blocking - 1) {
        if (i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BL1) {
          if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
            l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset * i_xgemm_desc->ldb);
          } else {
            l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset);
          }
          libxsmm_x86_instruction_prefetch(io_generated_code,
             LIBXSMM_X86_INSTR_PREFETCHT0,
              i_gp_reg_mapping->gp_reg_b,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              l_b_offset + 16 * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2);
        }
      }
    } else {
      /* handle trans B */
      if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
        l_b_offset = l_n * i_micro_kernel_config->datatype_size_in2 * l_k_pack_factor;
      } else {
        l_b_offset = i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in2;
      }

      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->b_vmove_instruction,
          i_gp_reg_mapping->gp_reg_b,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          l_b_offset,
          l_b_vname,
          l_vreg_ab_offset+1+l_n, 0, 1, 0 );

      if ((l_is_Af16_Bf16_gemm > 0 || l_is_Ai8_Bf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0) && (l_use_f16_replacement_fma > 0 || l_use_f32_compute_with_f16_inp > 0)) {
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, vname_cvt, l_vreg_ab_offset+1+l_n, l_vreg_ab_offset+1+l_n );
      }

      if (l_n == i_n_blocking - 1) {
        if (i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BL1) {
          libxsmm_x86_instruction_prefetch(io_generated_code,
              LIBXSMM_X86_INSTR_PREFETCHT0,
              i_gp_reg_mapping->gp_reg_b,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              16 * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2);
        }

        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2;
        } else {
          l_b_offset = i_micro_kernel_config->datatype_size_in2 * l_k_pack_factor;
        }

        libxsmm_x86_instruction_alu_imm( io_generated_code,
            i_micro_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_b,
            l_b_offset );
      }
    }
  }

  for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
    char l_a_vname = (l_is_Ai8_Bf16_gemm == 0 && l_is_Abf8_Bf16_gemm == 0) ? i_micro_kernel_config->vector_name : ((l_use_f32_compute_with_f16_inp) ? (i_micro_kernel_config->vector_name == 'z' ? 'x' : 'x') : ((l_use_f16_replacement_fma > 0) ? (i_micro_kernel_config->vector_name == 'z' ? 'x' : 'x') : (i_micro_kernel_config->vector_name == 'z' ? 'y' : 'x')));
    unsigned int l_a_vmove_instruction = ((l_is_Ai8_Bf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && (l_m != (l_m_blocking - 1)) ) ? LIBXSMM_X86_INSTR_VMOVSD : i_micro_kernel_config->a_vmove_instruction;

    if (l_is_Af16_Bf16_gemm > 0) {
      if ( l_use_f32_compute_with_f16_inp > 0 || l_use_f16_replacement_fma > 0 ) {
        l_a_vname = (i_micro_kernel_config->vector_name == 'z') ? 'y' : 'x';
      }
    }

    libxsmm_x86_instruction_vec_move( io_generated_code,
        i_micro_kernel_config->instruction_set,
        l_a_vmove_instruction,
        i_gp_reg_mapping->gp_reg_a,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
        l_a_vname,
        l_vreg_ab_offset, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 1, 0 );

    /* Process vreg A in case of f16/i8 */
    libxsmm_generator_gemm_avx512_microkernel_process_vreg_A( io_generated_code, i_micro_kernel_config, i_xgemm_desc,
        vname_cvt, l_is_Ai8_Bf16_gemm, l_is_Abf8_Bf16_gemm, l_is_Af16_Bf16_gemm, l_use_f16_replacement_fma, l_use_f32_compute_with_f16_inp, l_m, l_m_blocking, l_vreg_ab_offset);

    /* current A prefetch, next rows for the current column */
    if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2_AHEAD || i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD ) {
      libxsmm_x86_instruction_prefetch( io_generated_code,
          LIBXSMM_X86_INSTR_PREFETCHT1,
          i_gp_reg_mapping->gp_reg_a,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          ((i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor) + (64 * l_m_blocking) );
    }

    /* prefetch a different A matrix provided by the prefetch pointers */
    if ( (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2) || (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C) ) {
      libxsmm_x86_instruction_prefetch( io_generated_code,
          LIBXSMM_X86_INSTR_PREFETCHT1,
          i_gp_reg_mapping->gp_reg_a_prefetch,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor);
    }

    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      /* post increment early */
      if ( (l_n == 0) && (l_m == l_m_blocking-1) ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code,
            i_micro_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_a,
            (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in*l_k_pack_factor );

        /* if we prefetch next A into L2, we need to also increment the prefetch pointer */
        if ( (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2) || (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C) ) {
          libxsmm_x86_instruction_alu_imm( io_generated_code,
              i_micro_kernel_config->alu_add_instruction,
              i_gp_reg_mapping->gp_reg_a_prefetch,
              (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in*l_k_pack_factor );
        }
      }
      /* issue fma */
      if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
              i_micro_kernel_config->vmul_instruction,
              i_micro_kernel_config->vector_name,
              l_vreg_ab_offset+1+l_n,
              l_vreg_ab_offset,
              l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
        } else if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
              i_micro_kernel_config->vmul_instruction,
              i_micro_kernel_config->vector_name,
              l_vreg_ab_offset,
              l_vreg_ab_offset+1+l_n,
              l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
        } else {
          /* should not happen */
        }
      } else if (l_use_f16_replacement_fma > 0) {
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, vname_cvt, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
            i_micro_kernel_config->vmul_instruction,
            vname_cvt,
            l_vreg_ab_offset+1+l_n,
            l_vreg_ab_offset,
            l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
        libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, vname_cvt, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), 0,
                                                                (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
      } else {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
            i_micro_kernel_config->vmul_instruction,
            i_micro_kernel_config->vector_name,
            l_vreg_ab_offset+1+l_n,
            l_vreg_ab_offset,
            l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
      }
    }
  }
}


LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_i8_ss_uu_emu_nofsdbcst( libxsmm_generated_code*            io_generated_code,
                                                                                          const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                                          const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                                          const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                                          const unsigned int                 i_m_blocking,
                                                                                          const unsigned int                 i_n_blocking,
                                                                                          const int                          i_offset )
{
  /* deriving register blocking from kernel config */
  unsigned int l_m_blocking = ( i_m_blocking % i_micro_kernel_config->vector_length  == 0 ) ? i_m_blocking/i_micro_kernel_config->vector_length : (i_m_blocking/i_micro_kernel_config->vector_length)+1;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = i_micro_kernel_config->vector_reg_count - (i_n_blocking * l_m_blocking);
  /* temp variable for b-offset to handle no-trans/trans B */
  int l_b_offset = 0;
  /* we need 2 passes as we unpack the bf16 data on the spot */
  unsigned int l_pass = 0;
  /* k packing factor for VNNI */
  unsigned int l_k_pack_factor = 1;

#if !defined(NDEBUG)
  if ( (i_n_blocking > 30) || (i_n_blocking < 1) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }
  if ( (l_m_blocking < 1) || (l_m_blocking > 4) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
    return;
  }
  if ( (((l_m_blocking*i_n_blocking) + l_m_blocking + 1) > 32) && (i_n_blocking < 7) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
    return;
  }
#endif

  /* for VNNI we are stepping through to pack ks */
  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A ) {
    l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) );
  }

  for ( l_pass = 0; l_pass < 2; ++l_pass ) {
    /* load column vectors of A upfront */
    for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
      if ( l_pass == 0 ) {
        if ( i_micro_kernel_config->use_masking_a_c != 0 ) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->a_vmove_instruction,
            i_gp_reg_mapping->gp_reg_a,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
            i_micro_kernel_config->vector_name,
            1+l_m, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 1, 0 );

          /* we put "1" elements of B matrix into zmm 1+l_m */
          libxsmm_x86_instruction_vec_compute_2reg_mask(io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU16_LD,
                                                        i_micro_kernel_config->vector_name, 1+l_m, 1+l_m, 3, 1);
        } else {
          /* we put "1" elements of B matrix into zmm 1+l_m */
          libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMOVDQU16_LD,
            i_gp_reg_mapping->gp_reg_a,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
            i_micro_kernel_config->vector_name,
            1+l_m, 3, 1, 0 );
        }
      } else {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->a_vmove_instruction,
            i_gp_reg_mapping->gp_reg_a,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
            i_micro_kernel_config->vector_name,
            1+l_m, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 1, 0 );

        /* we put "0" elements of B matrix into zmm 1+l_m */
        libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code, LIBXSMM_X86_INSTR_VPSRLD_I,
                                                      i_micro_kernel_config->vector_name, 1+l_m, 1+l_m, 16);
      }
      /* convert 32bit int to 16bit int */
      libxsmm_x86_instruction_vec_compute_2reg(io_generated_code, LIBXSMM_X86_INSTR_VPMOVUSDW,
                                               'z', 1+l_m, 1+l_m);
      /* convert 8bit ints to 16bit ints */
      if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0 ) {
        libxsmm_x86_instruction_vec_compute_2reg(io_generated_code, LIBXSMM_X86_INSTR_VPMOVZXBW,
                                                 'z', 1+l_m, 1+l_m);
      } else {
        libxsmm_x86_instruction_vec_compute_2reg(io_generated_code, LIBXSMM_X86_INSTR_VPMOVSXBW,
                                                 'z', 1+l_m, 1+l_m);
      }

      if ( l_pass == 1 ) {
        /* current A prefetch, next rows for the current column */
        if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2_AHEAD || i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
              LIBXSMM_X86_INSTR_PREFETCHT1,
              i_gp_reg_mapping->gp_reg_a,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              ((i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor) + (64 * l_m_blocking) );
        }

        /* prefetch a different A matrix provided by the prefetch pointers */
        if ( (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2) || (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
              LIBXSMM_X86_INSTR_PREFETCHT1,
              i_gp_reg_mapping->gp_reg_a_prefetch,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor);
        }
      }
    }

    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      if ( i_offset != (-1) ) {
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset * i_xgemm_desc->ldb) + (l_n * i_micro_kernel_config->datatype_size_in2);
        } else {
          l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset) + (i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in2);
        }

        if ( l_pass == 1 ) {
          l_b_offset += 2;
        }

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VPBROADCASTW,
            i_gp_reg_mapping->gp_reg_b,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            l_b_offset,
            'y',
            0, 0, 1, 0 );

        if ((l_n == i_n_blocking - 1) && (l_pass == 1)) {
          if (i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BL1) {
            if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
              l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset * i_xgemm_desc->ldb);
            } else {
              l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset);
            }
            libxsmm_x86_instruction_prefetch(io_generated_code,
                LIBXSMM_X86_INSTR_PREFETCHT0,
                i_gp_reg_mapping->gp_reg_b,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                l_b_offset + 16 * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2);
          }
        }
      } else {
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = l_n * i_micro_kernel_config->datatype_size_in2 * l_k_pack_factor;
        } else {
          l_b_offset = i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in2;
        }

        if ( l_pass == 1 ) {
          l_b_offset += 2;
        }

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VPBROADCASTW,
            i_gp_reg_mapping->gp_reg_b,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            l_b_offset,
            'y',
            0, 0, 1, 0 );

        if ((l_n == i_n_blocking - 1) && (l_pass == 1)) {
          if (i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BL1) {
            libxsmm_x86_instruction_prefetch(io_generated_code,
                LIBXSMM_X86_INSTR_PREFETCHT0,
                i_gp_reg_mapping->gp_reg_b,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                16 * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2);
          }

          /* handle trans B */
          if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
            l_b_offset = i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2;
          } else {
            l_b_offset = i_micro_kernel_config->datatype_size_in2 * l_k_pack_factor;
          }

          libxsmm_x86_instruction_alu_imm( io_generated_code,
              i_micro_kernel_config->alu_add_instruction,
              i_gp_reg_mapping->gp_reg_b,
              l_b_offset );
        }
      }

      if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0 ) {
        libxsmm_x86_instruction_vec_compute_2reg(io_generated_code, LIBXSMM_X86_INSTR_VPMOVZXBW,
                                                 'z', 0, 0);
      } else {
        libxsmm_x86_instruction_vec_compute_2reg(io_generated_code, LIBXSMM_X86_INSTR_VPMOVSXBW,
                                                 'z', 0, 0);
      }

      for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
        /* post increment early */
        if ( (l_m == 0) && (l_n == i_n_blocking-1) && (l_pass == 1) ) {
          libxsmm_x86_instruction_alu_imm( io_generated_code,
              i_micro_kernel_config->alu_add_instruction,
              i_gp_reg_mapping->gp_reg_a,
              (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in*l_k_pack_factor );

          /* if we prefetch next A into L2, we need to also increment the prefetch pointer */
          if ( (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2) || (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C) ) {
            libxsmm_x86_instruction_alu_imm( io_generated_code,
                i_micro_kernel_config->alu_add_instruction,
                i_gp_reg_mapping->gp_reg_a_prefetch,
                (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in*l_k_pack_factor );
          }
        }
        /* issue fma */
        if ( io_generated_code->arch == LIBXSMM_X86_AVX512_SKX ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                    LIBXSMM_X86_INSTR_VPMADDWD,
                                                    i_micro_kernel_config->vector_name,
                                                    1+l_m,
                                                    0,
                                                    l_vec_reg_acc_start - 1 );
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                    LIBXSMM_X86_INSTR_VPADDD,
                                                    i_micro_kernel_config->vector_name,
                                                    l_vec_reg_acc_start - 1,
                                                    l_vec_reg_acc_start + l_m + (l_m_blocking * l_n),
                                                    l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
        } else {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
              LIBXSMM_X86_INSTR_VPDPWSSD,
              i_micro_kernel_config->vector_name,
              1+l_m,
              0,
              l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
        }
      }
    }
  }
}


LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_m8_i8_ss_uu_emu_nofsdbcst( libxsmm_generated_code*            io_generated_code,
                                                                                             const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                                             const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                                             const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                                             const unsigned int                 i_m_blocking,
                                                                                             const unsigned int                 i_n_blocking,
                                                                                             const int                          i_offset )
{
  /* deriving register blocking from kernel config */
  unsigned int l_m_blocking = ( i_m_blocking % i_micro_kernel_config->vector_length  == 0 ) ? i_m_blocking/i_micro_kernel_config->vector_length : (i_m_blocking/i_micro_kernel_config->vector_length)+1;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = i_micro_kernel_config->vector_reg_count - (i_n_blocking * l_m_blocking);
  /* temp variable for b-offset to handle no-trans/trans B */
  int l_b_offset = 0;
  /* we need 2 passes as we unpack the bf16 data on the spot */
  unsigned int l_pass = 0;
  /* k packing factor for VNNI */
  unsigned int l_k_pack_factor = 1;

#if !defined(NDEBUG)
  if ( (i_n_blocking > 30) || (i_n_blocking < 1) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  if ( io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_SKX ||io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CLX
       || io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CPX) {
      if ( ((l_m_blocking*i_n_blocking) + i_n_blocking + 1) > 32 ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
        return;
      }
      if ( (l_m_blocking < 1) || (l_m_blocking > 8) ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
        return;
      }
  } else {
      if ( (l_m_blocking < 1) || (l_m_blocking > 4) ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
        return;
      }
      if ( (((l_m_blocking*i_n_blocking) + l_m_blocking + 1) > 32) && (i_n_blocking < 7) ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
        return;
      }
  }
#endif

  /* for VNNI we are stepping through to pack ks */
  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A ) {
    l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) );
  }

  for ( l_pass = 0; l_pass < 2; ++l_pass ) {
    /* load column vectors of A upfront */
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      if ( i_offset != (-1) ) {
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset * i_xgemm_desc->ldb) + (l_n * i_micro_kernel_config->datatype_size_in2);
        } else {
          l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset) + (i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in2);
        }

        if ( l_pass == 1 ) {
          l_b_offset += 2;
        }

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VPBROADCASTW,
            i_gp_reg_mapping->gp_reg_b,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            l_b_offset,
            'x',
            1+l_n, 0, 1, 0 );

        if ((l_n == i_n_blocking - 1) && (l_pass == 1)) {
          if (i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BL1) {
            if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
              l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset * i_xgemm_desc->ldb);
            } else {
              l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset);
            }
            libxsmm_x86_instruction_prefetch(io_generated_code,
                LIBXSMM_X86_INSTR_PREFETCHT0,
                i_gp_reg_mapping->gp_reg_b,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                l_b_offset + 16 * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2);
          }
        }
      } else {
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = l_n * i_micro_kernel_config->datatype_size_in2 * l_k_pack_factor;
        } else {
          l_b_offset = i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in2;
        }

        if ( l_pass == 1 ) {
          l_b_offset += 2;
        }

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VPBROADCASTW,
            i_gp_reg_mapping->gp_reg_b,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            l_b_offset,
            'x',
            1+l_n, 0, 1, 0 );

        if ((l_n == i_n_blocking - 1) && (l_pass == 1)) {
          if (i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BL1) {
            libxsmm_x86_instruction_prefetch(io_generated_code,
                LIBXSMM_X86_INSTR_PREFETCHT0,
                i_gp_reg_mapping->gp_reg_b,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                16 * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2);
          }

          /* handle trans B */
          if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
            l_b_offset = i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2;
          } else {
            l_b_offset = i_micro_kernel_config->datatype_size_in2 * l_k_pack_factor;
          }

          libxsmm_x86_instruction_alu_imm( io_generated_code,
              i_micro_kernel_config->alu_add_instruction,
              i_gp_reg_mapping->gp_reg_b,
              l_b_offset );
        }
      }

      if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0 ) {
        libxsmm_x86_instruction_vec_compute_2reg(io_generated_code, LIBXSMM_X86_INSTR_VPMOVZXBW,
                                                 'y', 1+l_n, 1+l_n);
      } else {
        libxsmm_x86_instruction_vec_compute_2reg(io_generated_code, LIBXSMM_X86_INSTR_VPMOVSXBW,
                                                 'y', 1+l_n, 1+l_n);
      }
    }
    for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
      if ( l_pass == 0 ) {
        if ( i_micro_kernel_config->use_masking_a_c != 0 ) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->a_vmove_instruction,
            i_gp_reg_mapping->gp_reg_a,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
            i_micro_kernel_config->vector_name,
            0, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 1, 0 );

          /* we put "1" elements of B matrix into zmm 1+l_m */
          libxsmm_x86_instruction_vec_compute_2reg_mask(io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU16_LD,
                                                        i_micro_kernel_config->vector_name, 0, 0, 3, 1);
        } else {
          /* we put "1" elements of B matrix into zmm 1+l_m */
          libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMOVDQU16_LD,
            i_gp_reg_mapping->gp_reg_a,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
            i_micro_kernel_config->vector_name,
            0, 3, 1, 0 );
        }
      } else {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->a_vmove_instruction,
            i_gp_reg_mapping->gp_reg_a,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
            i_micro_kernel_config->vector_name,
            0, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 1, 0 );

        /* we put "0" elements of B matrix into zmm 1+l_m */
        libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code, LIBXSMM_X86_INSTR_VPSRLD_I,
                                                      i_micro_kernel_config->vector_name, 0, 0, 16);
      }

      /* convert 32bit int to 16bit int */
      libxsmm_x86_instruction_vec_compute_2reg(io_generated_code, LIBXSMM_X86_INSTR_VPMOVUSDW,
                                               'y', 0, 0);
      /* convert 8bit ints to 16bit ints */
      if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0 ) {
        libxsmm_x86_instruction_vec_compute_2reg(io_generated_code, LIBXSMM_X86_INSTR_VPMOVZXBW,
                                                 'y', 0, 0);
      } else {
        libxsmm_x86_instruction_vec_compute_2reg(io_generated_code, LIBXSMM_X86_INSTR_VPMOVSXBW,
                                                 'y', 0, 0);
      }

      if ( l_pass == 1 ) {
        /* current A prefetch, next rows for the current column */
        if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2_AHEAD || i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
              LIBXSMM_X86_INSTR_PREFETCHT1,
              i_gp_reg_mapping->gp_reg_a,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              ((i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor) + (64 * l_m_blocking) );
        }

        /* prefetch a different A matrix provided by the prefetch pointers */
        if ( (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2) || (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
              LIBXSMM_X86_INSTR_PREFETCHT1,
              i_gp_reg_mapping->gp_reg_a_prefetch,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length * l_k_pack_factor) * l_m );
        }
      }

      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        /* post increment early */
        if ( (l_n == 0) && (l_m == l_m_blocking-1) && (l_pass == 1) ) {
          libxsmm_x86_instruction_alu_imm( io_generated_code,
              i_micro_kernel_config->alu_add_instruction,
              i_gp_reg_mapping->gp_reg_a,
              (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in*l_k_pack_factor );

          /* if we prefetch next A into L2, we need to also increment the prefetch pointer */
          if ( (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2) || (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C) ) {
            libxsmm_x86_instruction_alu_imm( io_generated_code,
                i_micro_kernel_config->alu_add_instruction,
                i_gp_reg_mapping->gp_reg_a_prefetch,
                (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in*l_k_pack_factor );
          }
        }
        /* issue fma */
        if ( io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_SKX ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                    LIBXSMM_X86_INSTR_VPMADDWD,
                                                    i_micro_kernel_config->vector_name,
                                                    1+l_n,
                                                    0,
                                                    l_vec_reg_acc_start - 1 );
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                    LIBXSMM_X86_INSTR_VPADDD,
                                                    i_micro_kernel_config->vector_name,
                                                    l_vec_reg_acc_start - 1,
                                                    l_vec_reg_acc_start + l_m + (l_m_blocking * l_n),
                                                    l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
        } else {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
              LIBXSMM_X86_INSTR_VPDPWSSD,
              i_micro_kernel_config->vector_name,
              1+l_n,
              0,
              l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
        }
      }
    }
  }
}


LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_bf16_emu_nofsdbcst( libxsmm_generated_code*            io_generated_code,
                                                                                      const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                                      const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                                      const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                                      const unsigned int                 i_m_blocking,
                                                                                      const unsigned int                 i_n_blocking,
                                                                                      const int                          i_offset )
{
  /* deriving register blocking from kernel config */
  unsigned int l_m_blocking = ( i_m_blocking % i_micro_kernel_config->vector_length  == 0 ) ? i_m_blocking/i_micro_kernel_config->vector_length : (i_m_blocking/i_micro_kernel_config->vector_length)+1;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = i_micro_kernel_config->vector_reg_count - (i_n_blocking * l_m_blocking);
  /* temp variable for b-offset to handle no-trans/trans B */
  int l_b_offset = 0;
  /* we need 2 passes as we unpack the bf16 data on the spot */
  unsigned int l_pass = 0;
  /* k packing factor for VNNI */
  unsigned int l_k_pack_factor = 1;

#if !defined(NDEBUG)
  if ( (i_n_blocking > 30) || (i_n_blocking < 1) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }
  if ( (l_m_blocking < 1) || (l_m_blocking > 4) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
    return;
  }
  if ( (((l_m_blocking*i_n_blocking) + l_m_blocking + 1) > 32) && (i_n_blocking < 7) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
    return;
  }
#endif

  /* for VNNI we are stepping through to pack ks */
  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A ) {
    l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) );
  }

  for ( l_pass = 0; l_pass < 2; ++l_pass ) {
    /* load column vectors of A upfront */
    for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
      if ( l_pass == 0 ) {
        if ( i_micro_kernel_config->use_masking_a_c != 0 ) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->a_vmove_instruction,
            i_gp_reg_mapping->gp_reg_a,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
            i_micro_kernel_config->vector_name,
            1+l_m, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 1, 0 );

          /* we put "1" elements of B matrix into zmm 1+l_m */
          libxsmm_x86_instruction_vec_compute_2reg_mask(io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU16_LD,
                                                        i_micro_kernel_config->vector_name, 1+l_m, 1+l_m, 3, 1);
        } else {
          /* we put "1" elements of B matrix into zmm 1+l_m */
          libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMOVDQU16_LD,
            i_gp_reg_mapping->gp_reg_a,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
            i_micro_kernel_config->vector_name,
            1+l_m, 3, 1, 0 );
        }
      } else {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->a_vmove_instruction,
            i_gp_reg_mapping->gp_reg_a,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
            i_micro_kernel_config->vector_name,
            1+l_m, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 1, 0 );

        /* we put "0" elements of B matrix into zmm 1+l_m */
        libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I,
                                                      i_micro_kernel_config->vector_name, 1+l_m, 1+l_m, 16);
      }

      if ( l_pass == 1 ) {
        /* current A prefetch, next rows for the current column */
        if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2_AHEAD || i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
              LIBXSMM_X86_INSTR_PREFETCHT1,
              i_gp_reg_mapping->gp_reg_a,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              ((i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor) + (64 * l_m_blocking) );
        }

        /* prefetch a different A matrix provided by the prefetch pointers */
        if ( (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2) || (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
              LIBXSMM_X86_INSTR_PREFETCHT1,
              i_gp_reg_mapping->gp_reg_a_prefetch,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor);
        }
      }
    }

    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      if ( i_offset != (-1) ) {
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset * i_xgemm_desc->ldb) + (l_n * i_micro_kernel_config->datatype_size_in2);
        } else {
          l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset) + (i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in2);
        }
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->b_vmove_instruction,
            i_gp_reg_mapping->gp_reg_b,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            l_b_offset,
            i_micro_kernel_config->vector_name,
            0, 0, 1, 0 );

        if ((l_n == i_n_blocking - 1) && (l_pass == 1)) {
          if (i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BL1) {
            if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
              l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset * i_xgemm_desc->ldb);
            } else {
              l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset);
            }
            libxsmm_x86_instruction_prefetch(io_generated_code,
                LIBXSMM_X86_INSTR_PREFETCHT0,
                i_gp_reg_mapping->gp_reg_b,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                l_b_offset + 16 * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2);
          }
        }
      } else {
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = l_n * i_micro_kernel_config->datatype_size_in2 * l_k_pack_factor;
        } else {
          l_b_offset = i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in2;
        }

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->b_vmove_instruction,
            i_gp_reg_mapping->gp_reg_b,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            l_b_offset,
            i_micro_kernel_config->vector_name,
            0, 0, 1, 0 );

        if ((l_n == i_n_blocking - 1) && (l_pass == 1)) {
          if (i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BL1) {
            libxsmm_x86_instruction_prefetch(io_generated_code,
                LIBXSMM_X86_INSTR_PREFETCHT0,
                i_gp_reg_mapping->gp_reg_b,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                16 * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2);
          }

          /* handle trans B */
          if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
            l_b_offset = i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2;
          } else {
            l_b_offset = i_micro_kernel_config->datatype_size_in2 * l_k_pack_factor;
          }

          libxsmm_x86_instruction_alu_imm( io_generated_code,
              i_micro_kernel_config->alu_add_instruction,
              i_gp_reg_mapping->gp_reg_b,
              l_b_offset );
        }
      }

      if ( l_pass == 0 ) {
        /* we put "1" elements of B matrix into zmm0 */
        libxsmm_x86_instruction_vec_compute_2reg_mask(io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU16_LD,
                                                      i_micro_kernel_config->vector_name, 0, 0, 3, 1);
      } else {
        /* we put "0" elements of B matrix into zmm0 */
        libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I,
                                                      i_micro_kernel_config->vector_name, 0, 0, 16);
      }

      for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
        /* post increment early */
        if ( (l_m == 0) && (l_n == i_n_blocking-1) && (l_pass == 1) ) {
          libxsmm_x86_instruction_alu_imm( io_generated_code,
              i_micro_kernel_config->alu_add_instruction,
              i_gp_reg_mapping->gp_reg_a,
              (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in*l_k_pack_factor );

          /* if we prefetch next A into L2, we need to also increment the prefetch pointer */
          if ( (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2) || (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C) ) {
            libxsmm_x86_instruction_alu_imm( io_generated_code,
                i_micro_kernel_config->alu_add_instruction,
                i_gp_reg_mapping->gp_reg_a_prefetch,
                (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in*l_k_pack_factor );
          }
        }
        /* issue fma */
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
            LIBXSMM_X86_INSTR_VFMADD231PS,
            i_micro_kernel_config->vector_name,
            1+l_m,
            0,
            l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
      }
    }
  }
}


LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_m8_bf16_emu_nofsdbcst( libxsmm_generated_code*            io_generated_code,
                                                                                         const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                                         const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                                         const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                                         const unsigned int                 i_m_blocking,
                                                                                         const unsigned int                 i_n_blocking,
                                                                                         const int                          i_offset )
{
  /* deriving register blocking from kernel config */
  unsigned int l_m_blocking = ( i_m_blocking % i_micro_kernel_config->vector_length  == 0 ) ? i_m_blocking/i_micro_kernel_config->vector_length : (i_m_blocking/i_micro_kernel_config->vector_length)+1;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = i_micro_kernel_config->vector_reg_count - (i_n_blocking * l_m_blocking);
  /* temp variable for b-offset to handle no-trans/trans B */
  int l_b_offset = 0;
  /* we need 2 passes as we unpack the bf16 data on the spot */
  unsigned int l_pass = 0;
  /* k packing factor for VNNI */
  unsigned int l_k_pack_factor = 1;

#if !defined(NDEBUG)
  if ( (i_n_blocking > 30) || (i_n_blocking < 1) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  if ( io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_SKX ||io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CLX
       || io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CPX) {
      if ( ((l_m_blocking*i_n_blocking) + i_n_blocking + 1) > 32 ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
        return;
      }
      if ( (l_m_blocking < 1) || (l_m_blocking > 8) ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
        return;
      }
  } else {
      if ( (l_m_blocking < 1) || (l_m_blocking > 4) ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
        return;
      }
      if ( (((l_m_blocking*i_n_blocking) + l_m_blocking + 1) > 32) && (i_n_blocking < 7) ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
        return;
      }
  }
#endif

  /* for VNNI we are stepping through to pack ks */
  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A ) {
    l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) );
  }

  for ( l_pass = 0; l_pass < 2; ++l_pass ) {
    /* load column vectors of A upfront */
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      if ( i_offset != (-1) ) {
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset * i_xgemm_desc->ldb) + (l_n * i_micro_kernel_config->datatype_size_in2);
        } else {
          l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset) + (i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in2);
        }
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->b_vmove_instruction,
            i_gp_reg_mapping->gp_reg_b,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            l_b_offset,
            i_micro_kernel_config->vector_name,
            1+l_n, 0, 1, 0 );

        if ((l_n == i_n_blocking - 1) && (l_pass == 1)) {
          if (i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BL1) {
            if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
              l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset * i_xgemm_desc->ldb);
            } else {
              l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset);
            }
            libxsmm_x86_instruction_prefetch(io_generated_code,
                LIBXSMM_X86_INSTR_PREFETCHT0,
                i_gp_reg_mapping->gp_reg_b,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                l_b_offset + 16 * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2);
          }
        }
      } else {
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = l_n * i_micro_kernel_config->datatype_size_in2 * l_k_pack_factor;
        } else {
          l_b_offset = i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in2;
        }

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->b_vmove_instruction,
            i_gp_reg_mapping->gp_reg_b,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            l_b_offset,
            i_micro_kernel_config->vector_name,
            1+l_n, 0, 1, 0 );

        if ((l_n == i_n_blocking - 1) && (l_pass == 1)) {
          if (i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BL1) {
            libxsmm_x86_instruction_prefetch(io_generated_code,
                LIBXSMM_X86_INSTR_PREFETCHT0,
                i_gp_reg_mapping->gp_reg_b,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                16 * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2);
          }

          /* handle trans B */
          if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
            l_b_offset = i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2;
          } else {
            l_b_offset = i_micro_kernel_config->datatype_size_in2 * l_k_pack_factor;
          }

          libxsmm_x86_instruction_alu_imm( io_generated_code,
              i_micro_kernel_config->alu_add_instruction,
              i_gp_reg_mapping->gp_reg_b,
              l_b_offset );
        }
      }

      if ( l_pass == 0 ) {
        /* we put "1" elements of B matrix into zmm0 */
        libxsmm_x86_instruction_vec_compute_2reg_mask(io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU16_LD,
                                                      i_micro_kernel_config->vector_name, 1+l_n, 1+l_n, 3, 1);
      } else {
        /* we put "0" elements of B matrix into zmm0 */
        libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I,
                                                      i_micro_kernel_config->vector_name, 1+l_n, 1+l_n, 16);
      }
    }
    for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
      if ( l_pass == 0 ) {
        if ( i_micro_kernel_config->use_masking_a_c != 0 ) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->a_vmove_instruction,
            i_gp_reg_mapping->gp_reg_a,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
            i_micro_kernel_config->vector_name,
            0, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 1, 0 );

          /* we put "1" elements of B matrix into zmm 1+l_m */
          libxsmm_x86_instruction_vec_compute_2reg_mask(io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU16_LD,
                                                        i_micro_kernel_config->vector_name, 0, 0, 3, 1);
        } else {
          /* we put "1" elements of B matrix into zmm 1+l_m */
          libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMOVDQU16_LD,
            i_gp_reg_mapping->gp_reg_a,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
            i_micro_kernel_config->vector_name,
            0, 3, 1, 0 );
        }
      } else {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->a_vmove_instruction,
            i_gp_reg_mapping->gp_reg_a,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
            i_micro_kernel_config->vector_name,
            0, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 1, 0 );

        /* we put "0" elements of B matrix into zmm 1+l_m */
        libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I,
                                                      i_micro_kernel_config->vector_name, 0, 0, 16);
      }

      if ( l_pass == 1 ) {
        /* current A prefetch, next rows for the current column */
        if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2_AHEAD || i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
              LIBXSMM_X86_INSTR_PREFETCHT1,
              i_gp_reg_mapping->gp_reg_a,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              ((i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor) + (64 * l_m_blocking) );
        }

        /* prefetch a different A matrix provided by the prefetch pointers */
        if ( (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2) || (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
              LIBXSMM_X86_INSTR_PREFETCHT1,
              i_gp_reg_mapping->gp_reg_a_prefetch,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length * l_k_pack_factor) * l_m );
        }
      }

      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        /* post increment early */
        if ( (l_n == 0) && (l_m == l_m_blocking-1) && (l_pass == 1) ) {
          libxsmm_x86_instruction_alu_imm( io_generated_code,
              i_micro_kernel_config->alu_add_instruction,
              i_gp_reg_mapping->gp_reg_a,
              (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in*l_k_pack_factor );

          /* if we prefetch next A into L2, we need to also increment the prefetch pointer */
          if ( (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2) || (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C) ) {
            libxsmm_x86_instruction_alu_imm( io_generated_code,
                i_micro_kernel_config->alu_add_instruction,
                i_gp_reg_mapping->gp_reg_a_prefetch,
                (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in*l_k_pack_factor );
          }
        }
        /* issue fma */
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
            LIBXSMM_X86_INSTR_VFMADD231PS,
            i_micro_kernel_config->vector_name,
            1+l_n,
            0,
            l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
      }
    }
  }
}


LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_bf8_emu_nofsdbcst( libxsmm_generated_code*            io_generated_code,
                                                                                     const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                                     const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                                     const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                                     const unsigned int                 i_m_blocking,
                                                                                     const unsigned int                 i_n_blocking,
                                                                                     const int                          i_offset )
{
  /* deriving register blocking from kernel config */
  unsigned int l_m_blocking = ( i_m_blocking % i_micro_kernel_config->vector_length  == 0 ) ? i_m_blocking/i_micro_kernel_config->vector_length : (i_m_blocking/i_micro_kernel_config->vector_length)+1;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = i_micro_kernel_config->vector_reg_count - (i_n_blocking * l_m_blocking);
  /* temp variable for b-offset to handle no-trans/trans B */
  int l_b_offset = 0;
  /* we need 4 passes as we unpack the bf8 data on the spot */
  unsigned int l_pass = 0;
  /* A reg offset */
  unsigned int l_a_reg_offset = 2;
  /* B reg offset */
  unsigned int l_b_reg_offset = 0;
  /* permute registers */
  unsigned int l_permute_reg = 1;
  /* k packing factor for VNNI */
  unsigned int l_k_pack_factor = 1;

#if !defined(NDEBUG)
  if ( (i_n_blocking > 30) || (i_n_blocking < 1) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }
  if ( (l_m_blocking < 1) || (l_m_blocking > 4) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
    return;
  }
  if ( (((l_m_blocking*i_n_blocking) + l_m_blocking + 1) > 32) && (i_n_blocking < 7) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
    return;
  }
#endif

  /* for VNNI we are stepping through to pack ks */
  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A ) {
    l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) );
  }

  for ( l_pass = 0; l_pass < 4; ++l_pass ) {
    /* load column vectors of A upfront */
    for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
      if ( i_micro_kernel_config->use_masking_a_c != 0 ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->a_vmove_instruction,
          i_gp_reg_mapping->gp_reg_a,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
          i_micro_kernel_config->vector_name,
          l_a_reg_offset + l_m, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 1, 0 );
      } else {
        /* we put "1" elements of B matrix into zmm 1+l_m */
        libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VMOVDQU8_LD,
          i_gp_reg_mapping->gp_reg_a,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
          i_micro_kernel_config->vector_name,
          l_a_reg_offset + l_m, 0, 1, 0 );
      }

      /* move the right VNNI position into focus */
      if ( l_pass == 0 ) {
        /* we put "0" elements of A matrix into zmm 1+l_m */
        libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I,
                                                      i_micro_kernel_config->vector_name, l_a_reg_offset + l_m, l_a_reg_offset + l_m, 8);
      } else if ( l_pass == 1 ) {
        /* we put "1" elements of A matrix into zmm 1+l_m */
        libxsmm_x86_instruction_vec_compute_2reg_mask(io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU8_LD,
                                                        i_micro_kernel_config->vector_name, l_a_reg_offset + l_m, l_a_reg_offset + l_m, 3, 1);

      } else if ( l_pass == 2 ) {
        /* we put "2" elements of A matrix into zmm 1+l_m */
        libxsmm_x86_instruction_vec_compute_2reg_mask(io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU8_LD,
                                                        i_micro_kernel_config->vector_name, l_a_reg_offset + l_m, l_a_reg_offset + l_m, 4, 1);

        libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code, LIBXSMM_X86_INSTR_VPSRLD_I,
                                                      i_micro_kernel_config->vector_name, l_a_reg_offset + l_m, l_a_reg_offset + l_m, 8);
      } else if ( l_pass == 3 ) {
        /* we put "2" elements of A matrix into zmm 1+l_m */
        libxsmm_x86_instruction_vec_compute_2reg_mask(io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU8_LD,
                                                        i_micro_kernel_config->vector_name, l_a_reg_offset + l_m, l_a_reg_offset + l_m, 5, 1);

        libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code, LIBXSMM_X86_INSTR_VPSRLD_I,
                                                      i_micro_kernel_config->vector_name, l_a_reg_offset + l_m, l_a_reg_offset + l_m, 16);
      }
      /* permute into ymm */
      libxsmm_x86_instruction_vec_compute_3reg(io_generated_code, LIBXSMM_X86_INSTR_VPERMW,
                                               i_micro_kernel_config->vector_name, l_a_reg_offset + l_m, l_permute_reg, l_a_reg_offset + l_m);

      /* convert FP16 into FP32 */
      libxsmm_x86_instruction_vec_compute_2reg(io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS,
                                               i_micro_kernel_config->vector_name, l_a_reg_offset + l_m, l_a_reg_offset + l_m);

#if 0
      if ( l_pass == 3 ) {
        /* current A prefetch, next rows for the current column */
        if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2_AHEAD || i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
              LIBXSMM_X86_INSTR_PREFETCHT1,
              i_gp_reg_mapping->gp_reg_a,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              ((i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor) + (64 * l_m_blocking) );
        }

        /* prefetch a different A matrix provided by the prefetch pointers */
        if ( (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2) || (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
              LIBXSMM_X86_INSTR_PREFETCHT1,
              i_gp_reg_mapping->gp_reg_a_prefetch,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor);
        }
      }
#endif
    }

    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      if ( i_offset != (-1) ) {
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset * i_xgemm_desc->ldb) + (l_n * i_micro_kernel_config->datatype_size_in2) + l_pass;
        } else {
          l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset) + (i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in2) + l_pass;
        }
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->b_vmove_instruction,
            i_gp_reg_mapping->gp_reg_b,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            l_b_offset,
            'y',
            l_b_reg_offset, 0, 1, 0 );

#if 0
        if ((l_n == i_n_blocking - 1) && (l_pass == 3)) {
          if (i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BL1) {
            if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
              l_b_offset = (i_micro_kernel_config->datatype_size_in * i_offset * i_xgemm_desc->ldb);
            } else {
              l_b_offset = (i_micro_kernel_config->datatype_size_in * i_offset);
            }
            libxsmm_x86_instruction_prefetch(io_generated_code,
                LIBXSMM_X86_INSTR_PREFETCHT0,
                i_gp_reg_mapping->gp_reg_b,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                l_b_offset + 16 * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in);
          }
        }
#endif
      } else {
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = (l_n * i_micro_kernel_config->datatype_size_in2 * l_k_pack_factor) + l_pass;
        } else {
          l_b_offset = (i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in2) + l_pass;
        }

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->b_vmove_instruction,
            i_gp_reg_mapping->gp_reg_b,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            l_b_offset,
            'y',
            l_b_reg_offset, 0, 1, 0 );

        if ((l_n == i_n_blocking - 1) && (l_pass == 3)) {
#if 0
          if (i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BL1) {
            libxsmm_x86_instruction_prefetch(io_generated_code,
                LIBXSMM_X86_INSTR_PREFETCHT0,
                i_gp_reg_mapping->gp_reg_b,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                16 * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in);
          }
#endif

          /* handle trans B */
          if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
            l_b_offset = i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2;
          } else {
            l_b_offset = i_micro_kernel_config->datatype_size_in2 * l_k_pack_factor;
          }

          libxsmm_x86_instruction_alu_imm( io_generated_code,
              i_micro_kernel_config->alu_add_instruction,
              i_gp_reg_mapping->gp_reg_b,
              l_b_offset );
        }
      }

      /* make FP16 numbers from the broadcast */
      libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code, LIBXSMM_X86_INSTR_VPSLLW_I,
                                                    (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 'x' : 'y', l_b_reg_offset, l_b_reg_offset, 8);

      /* convert FP16 into FP32 */
      libxsmm_x86_instruction_vec_compute_2reg(io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS,
                                               i_micro_kernel_config->vector_name, l_b_reg_offset, l_b_reg_offset);

      for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
        /* post increment early */
        if ( (l_m == 0) && (l_n == i_n_blocking-1) && (l_pass == 3) ) {
          libxsmm_x86_instruction_alu_imm( io_generated_code,
              i_micro_kernel_config->alu_add_instruction,
              i_gp_reg_mapping->gp_reg_a,
              (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in*l_k_pack_factor );

#if 0
          /* if we prefetch next A into L2, we need to also increment the prefetch pointer */
          if ( (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2) || (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C) ) {
            libxsmm_x86_instruction_alu_imm( io_generated_code,
                i_micro_kernel_config->alu_add_instruction,
                i_gp_reg_mapping->gp_reg_a_prefetch,
                (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in*l_k_pack_factor );
          }
#endif
        }
        /* issue fma */
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
            LIBXSMM_X86_INSTR_VFMADD231PS,
            i_micro_kernel_config->vector_name,
            l_a_reg_offset+l_m,
            l_b_reg_offset,
            l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
      }
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_m8_bf8_emu_nofsdbcst( libxsmm_generated_code*            io_generated_code,
                                                                                        const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                                        const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                                        const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                                        const unsigned int                 i_m_blocking,
                                                                                        const unsigned int                 i_n_blocking,
                                                                                        const int                          i_offset )
{
  /* deriving register blocking from kernel config */
  unsigned int l_m_blocking = ( i_m_blocking % i_micro_kernel_config->vector_length  == 0 ) ? i_m_blocking/i_micro_kernel_config->vector_length : (i_m_blocking/i_micro_kernel_config->vector_length)+1;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = i_micro_kernel_config->vector_reg_count - (i_n_blocking * l_m_blocking);
  /* temp variable for b-offset to handle no-trans/trans B */
  int l_b_offset = 0;
  /* we need 4 passes as we unpack the bf8 data on the spot */
  unsigned int l_pass = 0;
  /* A reg offset */
  unsigned int l_a_reg_offset = 0;
  /* B reg offset */
  unsigned int l_b_reg_offset = 2;
  /* permute registers */
  unsigned int l_permute_reg = 1;
  /* k packing factor for VNNI */
  unsigned int l_k_pack_factor = 1;

#if !defined(NDEBUG)
  if ( (i_n_blocking > 30) || (i_n_blocking < 1) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  if ( io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_SKX ||io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CLX
       || io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CPX) {
      if ( ((l_m_blocking*i_n_blocking) + i_n_blocking + 1) > 32 ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
        return;
      }
      if ( (l_m_blocking < 1) || (l_m_blocking > 8) ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
        return;
      }
  } else {
      if ( (l_m_blocking < 1) || (l_m_blocking > 4) ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
        return;
      }
      if ( (((l_m_blocking*i_n_blocking) + l_m_blocking + 1) > 32) && (i_n_blocking < 7) ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
        return;
      }
  }
#endif

  /* for VNNI we are stepping through to pack ks */
  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A ) {
    l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) );
  }

  for ( l_pass = 0; l_pass < 4; ++l_pass ) {
    /* load column vectors of A upfront */
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      if ( i_offset != (-1) ) {
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset * i_xgemm_desc->ldb) + (l_n * i_micro_kernel_config->datatype_size_in2) + l_pass;
        } else {
          l_b_offset = (i_micro_kernel_config->datatype_size_in2 * i_offset) + (i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in2) + l_pass;
        }
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->b_vmove_instruction,
            i_gp_reg_mapping->gp_reg_b,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            l_b_offset,
            i_micro_kernel_config->vector_name,
            l_b_reg_offset+l_n, 0, 1, 0 );
#if 0
        if ((l_n == i_n_blocking - 1) && (l_pass == 1)) {
          if (i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BL1) {
            if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
              l_b_offset = (i_micro_kernel_config->datatype_size_in * i_offset * i_xgemm_desc->ldb) + l_pass;
            } else {
              l_b_offset = (i_micro_kernel_config->datatype_size_in * i_offset) + l_pass;
            }
            libxsmm_x86_instruction_prefetch(io_generated_code,
                LIBXSMM_X86_INSTR_PREFETCHT0,
                i_gp_reg_mapping->gp_reg_b,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                l_b_offset + 16 * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in);
          }
        }
#endif
      } else {
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = (l_n * i_micro_kernel_config->datatype_size_in2 * l_k_pack_factor) + l_pass;
        } else {
          l_b_offset = (i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in2) + l_pass;
        }

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->b_vmove_instruction,
            i_gp_reg_mapping->gp_reg_b,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            l_b_offset,
            i_micro_kernel_config->vector_name,
            l_b_reg_offset+l_n, 0, 1, 0 );

        if ((l_n == i_n_blocking - 1) && (l_pass == 3)) {
#if 0
          if (i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BL1) {
            libxsmm_x86_instruction_prefetch(io_generated_code,
                LIBXSMM_X86_INSTR_PREFETCHT0,
                i_gp_reg_mapping->gp_reg_b,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                16 * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in);
          }
#endif

          /* handle trans B */
          if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
            l_b_offset = i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2;
          } else {
            l_b_offset = i_micro_kernel_config->datatype_size_in2 * l_k_pack_factor;
          }

          libxsmm_x86_instruction_alu_imm( io_generated_code,
              i_micro_kernel_config->alu_add_instruction,
              i_gp_reg_mapping->gp_reg_b,
              l_b_offset );
        }
      }

      /* make FP16 numbers from the broadcast */
      libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code, LIBXSMM_X86_INSTR_VPSLLW_I,
                                                    (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX ) ? 'x' : 'y', l_b_reg_offset+l_n, l_b_reg_offset+l_n, 8);

      /* convert FP16 into FP32 */
      libxsmm_x86_instruction_vec_compute_2reg(io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS,
                                               i_micro_kernel_config->vector_name, l_b_reg_offset+l_n, l_b_reg_offset+l_n);
    }
    for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
      if ( i_micro_kernel_config->use_masking_a_c != 0 ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->a_vmove_instruction,
          i_gp_reg_mapping->gp_reg_a,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
          i_micro_kernel_config->vector_name,
          l_a_reg_offset, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 1, 0 );
      } else {
        /* we put "1" elements of B matrix into zmm 1+l_m */
        libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VMOVDQU8_LD,
          i_gp_reg_mapping->gp_reg_a,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
          i_micro_kernel_config->vector_name,
          l_a_reg_offset, 0, 1, 0 );
      }

      /* move the right VNNI position into focus */
      if ( l_pass == 0 ) {
        /* we put "0" elements of A matrix into zmm 1+l_m */
        libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I,
                                                      i_micro_kernel_config->vector_name, l_a_reg_offset, l_a_reg_offset, 8);
      } else if ( l_pass == 1 ) {
        /* we put "1" elements of A matrix into zmm 1+l_m */
        libxsmm_x86_instruction_vec_compute_2reg_mask(io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU8_LD,
                                                      i_micro_kernel_config->vector_name, l_a_reg_offset, l_a_reg_offset, 3, 1);
      } else if ( l_pass == 2 ) {
        /* we put "2" elements of A matrix into zmm 1+l_m */
        libxsmm_x86_instruction_vec_compute_2reg_mask(io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU8_LD,
                                                      i_micro_kernel_config->vector_name, l_a_reg_offset, l_a_reg_offset, 4, 1);

        libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code, LIBXSMM_X86_INSTR_VPSRLD_I,
                                                      i_micro_kernel_config->vector_name, l_a_reg_offset, l_a_reg_offset, 8);
      } else if ( l_pass == 3 ) {
        /* we put "2" elements of A matrix into zmm 1+l_m */
        libxsmm_x86_instruction_vec_compute_2reg_mask(io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU8_LD,
                                                      i_micro_kernel_config->vector_name, l_a_reg_offset, l_a_reg_offset, 5, 1);

        libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code, LIBXSMM_X86_INSTR_VPSRLD_I,
                                                      i_micro_kernel_config->vector_name, l_a_reg_offset, l_a_reg_offset, 16);
      }
      /* permute into ymm */
      libxsmm_x86_instruction_vec_compute_3reg(io_generated_code, LIBXSMM_X86_INSTR_VPERMW,
                                               i_micro_kernel_config->vector_name, l_a_reg_offset, l_permute_reg, l_a_reg_offset);

      /* convert FP16 into FP32 */
      libxsmm_x86_instruction_vec_compute_2reg(io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS,
                                               i_micro_kernel_config->vector_name, l_a_reg_offset, l_a_reg_offset);

#if 0
      if ( l_pass == 3 ) {
        /* current A prefetch, next rows for the current column */
        if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2_AHEAD || i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
              LIBXSMM_X86_INSTR_PREFETCHT1,
              i_gp_reg_mapping->gp_reg_a,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              ((i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m) + (64 * l_m_blocking) );
        }

        /* prefetch a different A matrix provided by the prefetch pointers */
        if ( (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2) || (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
              LIBXSMM_X86_INSTR_PREFETCHT1,
              i_gp_reg_mapping->gp_reg_a_prefetch,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m );
        }
      }
#endif

      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        /* post increment early */
        if ( (l_n == 0) && (l_m == l_m_blocking-1) && (l_pass == 3) ) {
          libxsmm_x86_instruction_alu_imm( io_generated_code,
              i_micro_kernel_config->alu_add_instruction,
              i_gp_reg_mapping->gp_reg_a,
              (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in*l_k_pack_factor );

          /* if we prefetch next A into L2, we need to also increment the prefetch pointer */
#if 0
          if ( (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2) || (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C) ) {
            libxsmm_x86_instruction_alu_imm( io_generated_code,
                i_micro_kernel_config->alu_add_instruction,
                i_gp_reg_mapping->gp_reg_a_prefetch,
                (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in*l_k_pack_factor );
          }
#endif
        }
        /* issue fma */
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
            LIBXSMM_X86_INSTR_VFMADD231PS,
            i_micro_kernel_config->vector_name,
            l_b_reg_offset+l_n,
            l_a_reg_offset,
            l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
      }
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_fsdbcst( libxsmm_generated_code*            io_generated_code,
                                                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                           const unsigned int                 i_n_blocking,
                                                                           const unsigned int                 i_k_blocking )
{
  unsigned int  l_n;
  unsigned int  l_k;
  unsigned int  l_n_accs = 0;
  unsigned char l_vec_name_ld_a = i_micro_kernel_config->vector_name;
  unsigned int  l_vreg_ab_offset = 0;
  unsigned int  l_k_pack_factor = 1;
  unsigned int  l_k_iters = i_k_blocking;
  unsigned int l_is_Ai8_Bf16_gemm = ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
                                     (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
  unsigned int l_is_Abf8_Bf16_gemm = ((LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
                                     (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
  unsigned int l_is_Af16_Bf16_gemm =((LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
                                     (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
  unsigned int l_use_f16_replacement_fma = (((l_is_Ai8_Bf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0 || l_is_Af16_Bf16_gemm > 0) && io_generated_code->arch < LIBXSMM_X86_AVX512_SPR && LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
  unsigned int l_use_f32_compute_with_f16_inp = (((l_is_Ai8_Bf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0 || l_is_Af16_Bf16_gemm > 0) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;

  char vname_cvt = i_micro_kernel_config->vector_name;
  unsigned int l_a_vmove_instruction = ((l_is_Ai8_Bf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && (i_micro_kernel_config->use_masking_a_c == 0)) ? LIBXSMM_X86_INSTR_VMOVSD : i_micro_kernel_config->a_vmove_instruction;

  if (l_is_Ai8_Bf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0) {
    l_vec_name_ld_a = (l_use_f32_compute_with_f16_inp > 0) ? 'x' : ( l_use_f16_replacement_fma > 0 ? 'x' : 'y');
    if ( io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
      l_vec_name_ld_a = 'x';
    }
    if (l_is_Ai8_Bf16_gemm > 0) {
      l_vreg_ab_offset = 1;
    }
  }

  if (l_is_Af16_Bf16_gemm > 0) {
    if ( l_use_f32_compute_with_f16_inp > 0 || l_use_f16_replacement_fma > 0 ) {
      l_vec_name_ld_a = (i_micro_kernel_config->vector_name == 'z') ? 'y' : 'x';
    }
  }

  if (l_use_f16_replacement_fma > 0) {
    vname_cvt = (l_vec_name_ld_a == 'y') ? 'z' : ((l_vec_name_ld_a == 'x') ? ((io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 'y' : 'z') : i_micro_kernel_config->vector_name);
  }

#if !defined(NDEBUG)
  if ( i_n_blocking > 30 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }
#endif

  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A ) {
    l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) );
    l_k_iters = i_k_blocking / l_k_pack_factor;
    if ( i_k_blocking % l_k_pack_factor != 0 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_K_BLOCK );
      return;
    }
  }

  /* compute number of n accumulators to hide FMA latencies */
  if (i_n_blocking >= 12) {
    l_n_accs = 1;
  } else if (i_n_blocking >= 6) {
    l_n_accs = 2;
  } else {
    l_n_accs = 4;
  }
  if ( l_n_accs > l_k_iters ) {
    l_n_accs = l_k_iters;
    l_n_accs = (l_n_accs == 0) ? 1 : l_n_accs;
  }

  /* xor additional accumulator, if needed */
  for ( l_k = 1; l_k < l_n_accs; l_k++) {
    for ( l_n = 0; l_n < i_n_blocking; l_n++) {
      libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                           i_micro_kernel_config->vxor_instruction,
                                           i_micro_kernel_config->vector_name,
                                           i_micro_kernel_config->vector_reg_count - (i_n_blocking*(l_k+1)) + l_n,
                                           i_micro_kernel_config->vector_reg_count - (i_n_blocking*(l_k+1)) + l_n,
                                           i_micro_kernel_config->vector_reg_count - (i_n_blocking*(l_k+1)) + l_n );
    }
  }

  /* in case of int8 GEMM on SKX use zmm2 for 16bit 1's */
  if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) &&
         ((io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_SKX) ||(io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CLX)
          || (io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CPX))
      ) {
    short l_all_ones[16] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1};
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
                                                         (const unsigned char *)l_all_ones,
                                                         "my_int16_ones",
                                                         i_micro_kernel_config->vector_name,
                                                         2 );
  } else if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) &&
         (io_generated_code->arch < LIBXSMM_X86_AVX512_CLX) ) {
    short l_all_ones[32] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
                                                         (const unsigned char *)l_all_ones,
                                                         "my_int16_ones",
                                                         i_micro_kernel_config->vector_name,
                                                         2 );
  }
  /* for flat VNNI layout kernel set masking register */
  if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) &&
         ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0) ) {
    unsigned int l_mask[16] = {0xffff0000, 0xffff0000, 0xffff0000, 0xffff0000,
                               0xffff0000, 0xffff0000, 0xffff0000, 0xffff0000,
                               0xffff0000, 0xffff0000, 0xffff0000, 0xffff0000,
                               0xffff0000, 0xffff0000, 0xffff0000, 0xffff0000 };
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
                                                         (const unsigned char *)l_mask,
                                                         "my_bf16_mask",
                                                         i_micro_kernel_config->vector_name,
                                                         3 );
    /* setting vector length for a load to 'y' */
    l_vec_name_ld_a = 'y';
  }

  /* apply k blocking */
  for ( l_k = 0; l_k < l_k_iters; l_k++ ) {
    if ( l_k == 0 ) {
      /* load A */
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        io_generated_code->arch,
                                        l_a_vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_a,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        i_xgemm_desc->lda * l_k * i_micro_kernel_config->datatype_size_in * l_k_pack_factor,
                                        l_vec_name_ld_a,
                                        0 + l_vreg_ab_offset,
                                        i_micro_kernel_config->use_masking_a_c, 1, 0 );


      /* Process vreg A in case of f16/i8 */
      libxsmm_generator_gemm_avx512_microkernel_process_vreg_A( io_generated_code, i_micro_kernel_config, i_xgemm_desc,
          vname_cvt, l_is_Ai8_Bf16_gemm, l_is_Abf8_Bf16_gemm, l_is_Af16_Bf16_gemm, l_use_f16_replacement_fma, l_use_f32_compute_with_f16_inp, 0, 1, l_vreg_ab_offset);

      /* current A prefetch, next rows for the current column */
      if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2_AHEAD ||
           i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD) {
        libxsmm_x86_instruction_prefetch( io_generated_code,
                                          i_micro_kernel_config->prefetch_instruction,
                                          i_gp_reg_mapping->gp_reg_a,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (i_xgemm_desc->lda * l_k + i_micro_kernel_config->datatype_size_in * l_k_pack_factor) + 64 );
      }
      if ( l_k_iters > 1 ) {
        /* second A load in first iteration, in case of large blockings -> hiding L1 latencies */
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          io_generated_code->arch,
                                          l_a_vmove_instruction,
                                          i_gp_reg_mapping->gp_reg_a,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          i_xgemm_desc->lda * (l_k+1) * i_micro_kernel_config->datatype_size_in * l_k_pack_factor,
                                          l_vec_name_ld_a,
                                          1 + l_vreg_ab_offset,
                                          i_micro_kernel_config->use_masking_a_c, 1, 0 );

        /* Process vreg A in case of f16/i8 */
        libxsmm_generator_gemm_avx512_microkernel_process_vreg_A( io_generated_code, i_micro_kernel_config, i_xgemm_desc,
            vname_cvt, l_is_Ai8_Bf16_gemm, l_is_Abf8_Bf16_gemm, l_is_Af16_Bf16_gemm, l_use_f16_replacement_fma, l_use_f32_compute_with_f16_inp, 0, 1, 1 + l_vreg_ab_offset);

        /* current A prefetch, next rows for the current column */
        if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2_AHEAD ||
             i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            i_micro_kernel_config->prefetch_instruction,
                                            i_gp_reg_mapping->gp_reg_a,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            (i_xgemm_desc->lda * (l_k+1) * i_micro_kernel_config->datatype_size_in * l_k_pack_factor) + 64 );
        }
      }
    } else if ( l_k < (l_k_iters - 1) ) {
      /* pipelined load of A, one k iteration ahead */
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        io_generated_code->arch,
                                        l_a_vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_a,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        i_xgemm_desc->lda * (l_k+1) * i_micro_kernel_config->datatype_size_in * l_k_pack_factor,
                                        l_vec_name_ld_a,
                                        (l_k+1)%2 + l_vreg_ab_offset,
                                        i_micro_kernel_config->use_masking_a_c, 1, 0 );

      /* Process vreg A in case of f16/i8 */
      libxsmm_generator_gemm_avx512_microkernel_process_vreg_A( io_generated_code, i_micro_kernel_config, i_xgemm_desc,
          vname_cvt, l_is_Ai8_Bf16_gemm, l_is_Abf8_Bf16_gemm, l_is_Af16_Bf16_gemm, l_use_f16_replacement_fma, l_use_f32_compute_with_f16_inp, 0, 1, (l_k+1)%2 + l_vreg_ab_offset);

      /* current A prefetch, next rows for the current column */
      if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2_AHEAD          ||
           i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD    ) {
        libxsmm_x86_instruction_prefetch( io_generated_code,
                                          i_micro_kernel_config->prefetch_instruction,
                                          i_gp_reg_mapping->gp_reg_a,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (i_xgemm_desc->lda * (l_k+1) * i_micro_kernel_config->datatype_size_in * l_k_pack_factor) + 64 );
      }
    }

    /* next A prefetch "same" rows in "same" column, but in a different matrix */
    if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2 ||
         i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C) {
      libxsmm_x86_instruction_prefetch( io_generated_code,
                                        i_micro_kernel_config->prefetch_instruction,
                                        i_gp_reg_mapping->gp_reg_a_prefetch,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        (i_xgemm_desc->lda * l_k * i_micro_kernel_config->datatype_size_in * l_k_pack_factor) );
      if ( l_k == (l_k_iters - 1) ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code,
                                         i_micro_kernel_config->alu_add_instruction,
                                         i_gp_reg_mapping->gp_reg_a_prefetch,
                                         (long long)i_k_blocking * i_micro_kernel_config->datatype_size_in * i_xgemm_desc->lda );
      }
    }

    /* in last k-iteration: advance pointers */
    if ( l_k == (l_k_iters - 1) ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                       i_micro_kernel_config->alu_add_instruction,
                                       i_gp_reg_mapping->gp_reg_a,
                                       (long long)i_k_blocking * i_micro_kernel_config->datatype_size_in * i_xgemm_desc->lda );
    }

    /* in case of bfloat16 "prepare" A matrix in registers zmm l_k%2 and zmm3 using FP32 numbers */
    if ( ((LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) &&
         (io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX)) &&
         ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0)                             ) {
      /* we put "0" elements of A matrix into zmm3 */
      libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code,
          LIBXSMM_X86_INSTR_VPSLLD_I,
          i_micro_kernel_config->vector_name,
          l_k%2,
          3,
          16);

      /* we put "1" elements of A matrix into l_k%2 zmm*/
      libxsmm_x86_instruction_vec_compute_2reg_mask(io_generated_code,
          LIBXSMM_X86_INSTR_VMOVDQU16_LD,
          i_micro_kernel_config->vector_name,
          l_k%2,
          l_k%2,
          3,
          1);
    }

    if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) &&
         ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0) ) {
      /* convert 16 bit values into 32 bit (integer convert) */
      libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
            LIBXSMM_X86_INSTR_VPMOVSXWD,
            i_micro_kernel_config->vector_name,
            l_k%2, l_k%2 );
      libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code,
            LIBXSMM_X86_INSTR_VPSLLD_I,
            i_micro_kernel_config->vector_name,
            l_k%2, l_k%2, 16 );
    }

    /* compute vectorwidth (A) * column broadcast (B) */
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0 ) {
      for ( l_n = 0; l_n < i_n_blocking; l_n++) {
        /* determining base, idx and scale values */
        unsigned int l_b_reg = i_gp_reg_mapping->gp_reg_b;
        unsigned int l_b_idx = LIBXSMM_X86_GP_REG_UNDEF;
        unsigned int l_scale = 0;
        unsigned int l_disp = (l_k*i_micro_kernel_config->datatype_size_in2*l_k_pack_factor)+(l_n*i_xgemm_desc->ldb*i_micro_kernel_config->datatype_size_in2);

        if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) || l_is_Ai8_Bf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0 || LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
          if (((l_is_Ai8_Bf16_gemm > 0) || (l_is_Abf8_Bf16_gemm > 0) || (l_is_Af16_Bf16_gemm > 0)) && ((l_use_f16_replacement_fma > 0) || (l_use_f32_compute_with_f16_inp > 0))) {
            char l_b_vname = (i_micro_kernel_config->vector_name == 'z') ? 'y' : 'x';

            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              io_generated_code->arch,
                                              LIBXSMM_X86_INSTR_VPBROADCASTW,
                                              l_b_reg,
                                              l_b_idx, l_scale,
                                              l_disp,
                                              l_b_vname,
                                              3, 0, 1, 0 );

            if ((l_use_f16_replacement_fma > 0) || (l_use_f32_compute_with_f16_inp > 0)) {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, vname_cvt, 3, 3 );
            }

            if (l_use_f16_replacement_fma > 0) {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, vname_cvt, i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n, i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                  i_micro_kernel_config->vmul_instruction,
                  vname_cvt,
                  3,
                  l_k%2 + l_vreg_ab_offset,
                  i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );
              libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, vname_cvt, i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n, i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n, 0,
                                                                    (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
            } else {
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                  i_micro_kernel_config->vmul_instruction,
                  vname_cvt,
                  3,
                  l_k%2 + l_vreg_ab_offset,
                  i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );
            }
          } else {
            libxsmm_x86_instruction_vec_compute_mem_2reg( io_generated_code,
                                                          i_micro_kernel_config->vmul_instruction,
                                                          i_micro_kernel_config->vector_name,
                                                          l_b_reg,
                                                          l_b_idx,
                                                          l_scale,
                                                          l_disp,
                                                          1,
                                                          l_k%2 + l_vreg_ab_offset,
                                                          i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );
          }
        } else if (LIBXSMM_DATATYPE_I16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
          if ( io_generated_code->arch == LIBXSMM_X86_AVX512_SKX ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              io_generated_code->arch,
                                              LIBXSMM_X86_INSTR_VPBROADCASTD,
                                              l_b_reg,
                                              l_b_idx, l_scale,
                                              l_disp,
                                              i_micro_kernel_config->vector_name,
                                              3, 0, 1, 0 );
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VPMADDWD,
                                              i_micro_kernel_config->vector_name,
                                              l_k%2,
                                              3,
                                              3 );
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VPADDD,
                                              i_micro_kernel_config->vector_name,
                                              3,
                                              i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n,
                                              i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );
          } else if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_CLX ) || ( io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CLX ) ||
                        ( io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CPX )|| ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT )
                    ) {
            libxsmm_x86_instruction_vec_compute_mem_2reg( io_generated_code,
                                                          LIBXSMM_X86_INSTR_VPDPWSSD,
                                                          i_micro_kernel_config->vector_name,
                                                          l_b_reg,
                                                          l_b_idx,
                                                          l_scale,
                                                          l_disp,
                                                          1,
                                                          l_k%2,
                                                          i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );
          } else {
            /* should not happen */
          }
        } else if (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
          if ( (io_generated_code->arch < LIBXSMM_X86_AVX512_CLX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CLX)
              && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX)
            ) {
            /* let's broadcast B into zmm3 */
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              io_generated_code->arch,
                                              LIBXSMM_X86_INSTR_VPBROADCASTD,
                                              l_b_reg,
                                              l_b_idx, l_scale,
                                              l_disp,
                                              i_micro_kernel_config->vector_name,
                                              3, 0, 1, 0 );

            /* 8 bit mix-sign Mul */
            if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0  ) {
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                       LIBXSMM_X86_INSTR_VPMADDUBSW,
                                                       i_micro_kernel_config->vector_name,
                                                       3,
                                                       l_k%2,
                                                       3 );
            } else if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0 ) {
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                       LIBXSMM_X86_INSTR_VPMADDUBSW,
                                                       i_micro_kernel_config->vector_name,
                                                       l_k%2,
                                                       3,
                                                       3 );
            } else {
              LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
              return;
            }

            /* 16 bit mul with 1 */
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VPMADDWD,
                                              i_micro_kernel_config->vector_name,
                                              2,
                                              3,
                                              3 );

            /* add to accumulator */
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VPADDD,
                                              i_micro_kernel_config->vector_name,
                                              3,
                                              i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n,
                                              i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );
          } else if ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) {
            if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0 ) {
              libxsmm_x86_instruction_vec_compute_mem_2reg( io_generated_code,
                                                            LIBXSMM_X86_INSTR_VPDPBUSD,
                                                            i_micro_kernel_config->vector_name,
                                                            l_b_reg,
                                                            l_b_idx,
                                                            l_scale,
                                                            l_disp,
                                                            1,
                                                            l_k%2,
                                                            i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );
            } else if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0 ) {
              libxsmm_x86_instruction_vec_move( io_generated_code,
                                                io_generated_code->arch,
                                                LIBXSMM_X86_INSTR_VPBROADCASTD,
                                                l_b_reg,
                                                l_b_idx, l_scale,
                                                l_disp,
                                                i_micro_kernel_config->vector_name,
                                                3, 0, 1, 0 );

              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                LIBXSMM_X86_INSTR_VPDPBUSD,
                                                i_micro_kernel_config->vector_name,
                                                l_k%2,
                                                3,
                                                i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );

            } else {
              LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
              return;
            }
          } else {
            LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
            return;
          }
        } else if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) {
          if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) {
            /* broadcast pair of B matrix values into zmm2 */
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              io_generated_code->arch,
                                              LIBXSMM_X86_INSTR_VPBROADCASTW,
                                              l_b_reg,
                                              l_b_idx, l_scale,
                                              l_disp,
                                              i_micro_kernel_config->vector_name,
                                              2, 0, 1, 0 );

            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                  LIBXSMM_X86_INSTR_VPANDD, i_micro_kernel_config->vector_name,
                  2, 3, 2 );

            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                LIBXSMM_X86_INSTR_VFMADD231PS,
                                                i_micro_kernel_config->vector_name,
                                                l_k%2,
                                                2,
                                                i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );
          } else {
            if ( io_generated_code->arch < LIBXSMM_X86_AVX512_CPX && io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX ) {
              /* broadcast pair of B matrix values into zmm2 */
              libxsmm_x86_instruction_vec_move( io_generated_code,
                                                io_generated_code->arch,
                                                LIBXSMM_X86_INSTR_VBROADCASTSS,
                                                l_b_reg,
                                                l_b_idx, l_scale,
                                                l_disp,
                                                i_micro_kernel_config->vector_name,
                                                2, 0, 1, 0 );

              /* we put "1" elements of B matrix into zmm2 */
              libxsmm_x86_instruction_vec_compute_2reg_mask(io_generated_code,
                  LIBXSMM_X86_INSTR_VMOVDQU16_LD,
                  i_micro_kernel_config->vector_name,
                  2,
                  2,
                  3, 1);

              /* perform fma operations for multiplying "1" elements of A and B */
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                LIBXSMM_X86_INSTR_VFMADD231PS,
                                                i_micro_kernel_config->vector_name,
                                                l_k%2,
                                                2,
                                                i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );

              /* broadcast pair of B matrix values into zmm2 */
              libxsmm_x86_instruction_vec_move( io_generated_code,
                                                io_generated_code->arch,
                                                LIBXSMM_X86_INSTR_VBROADCASTSS,
                                                l_b_reg,
                                                l_b_idx, l_scale,
                                                l_disp,
                                                i_micro_kernel_config->vector_name,
                                                2, 0, 1, 0 );

              /* we put "0" elements of B matrix into zmm2 */
              libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code,
                  LIBXSMM_X86_INSTR_VPSLLD_I,
                  i_micro_kernel_config->vector_name,
                  2,
                  2,
                  16);

              /* perform fma operations for multiplying "0" elements of A and B */
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                LIBXSMM_X86_INSTR_VFMADD231PS,
                                                i_micro_kernel_config->vector_name,
                                                3,
                                                2,
                                                i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );
            } else {
              libxsmm_x86_instruction_vec_compute_mem_2reg( io_generated_code,
                                                            LIBXSMM_X86_INSTR_VDPBF16PS,
                                                            i_micro_kernel_config->vector_name,
                                                            l_b_reg,
                                                            l_b_idx,
                                                            l_scale,
                                                            l_disp,
                                                            1,
                                                            l_k%2,
                                                            i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );
            }
          }
        } else {
          /* should not happen */
        }
      }
    } else {
      for ( l_n = 0; l_n < i_n_blocking; l_n++) {
        /* determining base, idx and scale values */
        unsigned int l_b_reg = i_gp_reg_mapping->gp_reg_b;
        unsigned int l_b_idx = LIBXSMM_X86_GP_REG_UNDEF;
        unsigned int l_scale = 0;
        unsigned int l_disp = (l_k*i_xgemm_desc->ldb*i_micro_kernel_config->datatype_size_in2*l_k_pack_factor) + (l_n*i_micro_kernel_config->datatype_size_in2);

        if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) || l_is_Ai8_Bf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0 || LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
          if (((l_is_Ai8_Bf16_gemm > 0) || (l_is_Abf8_Bf16_gemm > 0) || (l_is_Af16_Bf16_gemm > 0)) && ((l_use_f16_replacement_fma > 0) || (l_use_f32_compute_with_f16_inp > 0))) {
            char l_b_vname = (i_micro_kernel_config->vector_name == 'z') ? 'y' : 'x';
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              io_generated_code->arch,
                                              LIBXSMM_X86_INSTR_VPBROADCASTW,
                                              l_b_reg,
                                              l_b_idx, l_scale,
                                              l_disp,
                                              l_b_vname,
                                              3, 0, 1, 0 );

            if ((l_use_f16_replacement_fma > 0) || (l_use_f32_compute_with_f16_inp > 0)) {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, vname_cvt, 3, 3 );
            }

            if (l_use_f16_replacement_fma > 0) {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, vname_cvt, i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n, i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                  i_micro_kernel_config->vmul_instruction,
                  vname_cvt,
                  3,
                  l_k%2 + l_vreg_ab_offset,
                  i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );
              libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, vname_cvt, i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n, i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n, 0,
                                                                      (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
            } else {
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                  i_micro_kernel_config->vmul_instruction,
                  vname_cvt,
                  3,
                  l_k%2 + l_vreg_ab_offset,
                  i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );
            }
          } else {
            libxsmm_x86_instruction_vec_compute_mem_2reg( io_generated_code,
                                                          i_micro_kernel_config->vmul_instruction,
                                                          i_micro_kernel_config->vector_name,
                                                          l_b_reg,
                                                          l_b_idx,
                                                          l_scale,
                                                          l_disp,
                                                          1,
                                                          l_k%2 + l_vreg_ab_offset,
                                                          i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );
          }
        } else if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) &&
                    ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0) ) {
          /* broadcast pair of B matrix values into zmm2 */
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            io_generated_code->arch,
                                            LIBXSMM_X86_INSTR_VPBROADCASTW,
                                            l_b_reg,
                                            l_b_idx, l_scale,
                                            l_disp,
                                            i_micro_kernel_config->vector_name,
                                            2, 0, 1, 0 );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                LIBXSMM_X86_INSTR_VPANDD, i_micro_kernel_config->vector_name,
                2, 3, 2 );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                              LIBXSMM_X86_INSTR_VFMADD231PS,
                                              i_micro_kernel_config->vector_name,
                                              l_k%2,
                                              2,
                                              i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );
        } else {
          /* should not happen */
        }
      }
    }
  }

  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0 ) {
    /* advance pointers of B only when we are not fully unrolling K and taking care of intermediate advances */
    if ( i_k_blocking < (unsigned int)i_xgemm_desc->k ) {
      /* advance pointers of B */
      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                       i_micro_kernel_config->alu_add_instruction,
                                       i_gp_reg_mapping->gp_reg_b,
                                       (long long)i_k_blocking * i_micro_kernel_config->datatype_size_in2 );
    }
  } else {
    /* advance pointers of B only when we are not fully unrolling K and taking care of intermediate advances */
    if ( i_k_blocking < (unsigned int)i_xgemm_desc->k ) {
      /* advance B ptr by K rows */
      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                       i_micro_kernel_config->alu_add_instruction,
                                       i_gp_reg_mapping->gp_reg_b,
                                       (long long)i_k_blocking * i_micro_kernel_config->datatype_size_in2 * i_xgemm_desc->ldb );
    }
  }

  /* add additional accumulators, if needed */
  for ( l_k = 1; l_k < l_n_accs; l_k++) {
    for ( l_n = 0; l_n < i_n_blocking; l_n++) {
      if ( (LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) ||
           (LIBXSMM_DATATYPE_F64  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) ||
           (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) ||
           (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) ||
           (l_is_Ai8_Bf16_gemm > 0) || (l_is_Abf8_Bf16_gemm > 0)) {
        char vname_add = (l_use_f16_replacement_fma > 0) ? vname_cvt : i_micro_kernel_config->vector_name;
        if (l_use_f16_replacement_fma > 0) {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, vname_cvt, i_micro_kernel_config->vector_reg_count - (i_n_blocking*(l_k+1)) + l_n, i_micro_kernel_config->vector_reg_count - (i_n_blocking*(l_k+1)) + l_n );
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, vname_cvt, i_micro_kernel_config->vector_reg_count - i_n_blocking + l_n, i_micro_kernel_config->vector_reg_count - i_n_blocking + l_n );
        }
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             i_micro_kernel_config->vadd_instruction,
                                             vname_add,
                                             i_micro_kernel_config->vector_reg_count - (i_n_blocking*(l_k+1)) + l_n,
                                             i_micro_kernel_config->vector_reg_count - i_n_blocking + l_n,
                                             i_micro_kernel_config->vector_reg_count - i_n_blocking + l_n );
        if (l_use_f16_replacement_fma > 0) {
          libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, vname_cvt, i_micro_kernel_config->vector_reg_count - i_n_blocking + l_n, i_micro_kernel_config->vector_reg_count - i_n_blocking + l_n, 0,
                                                                  (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
        }

      } else if ( (LIBXSMM_DATATYPE_I16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) ||
                  (LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ))    ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                             LIBXSMM_X86_INSTR_VPADDD,
                                             i_micro_kernel_config->vector_name,
                                             i_micro_kernel_config->vector_reg_count - (i_n_blocking*(l_k+1)) + l_n,
                                             i_micro_kernel_config->vector_reg_count - i_n_blocking + l_n,
                                             i_micro_kernel_config->vector_reg_count - i_n_blocking + l_n );
      } else {
        /* should not happen */
      }
    }
  }
}
