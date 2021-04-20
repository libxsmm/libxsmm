/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
*               Intel Corporation - All rights reserved                       *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Breuer (Univ. Jena), Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include "generator_aarch64_instructions.h"
#include "generator_gemm_common_aarch64.h"
#include "generator_common.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_init_micro_kernel_config_aarch64( libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                              const unsigned int             i_arch,
                                                              const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  memset(io_micro_kernel_config, 0, sizeof(*io_micro_kernel_config)); /* avoid warning "maybe used uninitialized" */
  if ( i_arch  == LIBXSMM_AARCH64_V81 ) {
    io_micro_kernel_config->instruction_set = LIBXSMM_AARCH64_V81;
    io_micro_kernel_config->vector_reg_count = 32;
    io_micro_kernel_config->use_masking_a_c = 0;
    io_micro_kernel_config->vector_name = 'v';
    if ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 2;
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->datatype_size_out = 8;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_STP_I_OFF;
      io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_STNP_I_OFF;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
    } else if ( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 4;
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->datatype_size_out = 4;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_STP_I_OFF;
      io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_STNP_I_OFF;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
    } else {
      /* should not happend */
    }
  } else if ( i_arch  == LIBXSMM_AARCH64_A64FX ) {
    io_micro_kernel_config->instruction_set = LIBXSMM_AARCH64_A64FX;
    io_micro_kernel_config->vector_reg_count = 32;
    io_micro_kernel_config->use_masking_a_c = 0;
    io_micro_kernel_config->vector_name = 'v';
    if ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 8;
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->datatype_size_out = 8;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_STP_I_OFF;
      io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_STNP_I_OFF;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
    } else if ( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 16;
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->datatype_size_out = 4;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_STP_I_OFF;
      io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_STNP_I_OFF;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
    } else {
      /* should not happend */
    }
  } else {
    /* that should no happen */
  }
}

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_aarch64_get_max_n_blocking( const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                                                const libxsmm_gemm_descriptor*      i_xgemm_desc,
                                                                const unsigned int                  i_arch ) {
  LIBXSMM_UNUSED( i_micro_kernel_config );
  LIBXSMM_UNUSED( i_xgemm_desc );

  if ( i_arch == LIBXSMM_AARCH64_V81 ) {
    return 30;
  } else if ( i_arch == LIBXSMM_AARCH64_A64FX ) {
    return 30;
  } else {
    return 0;
  }
}

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_aarch64_get_initial_m_blocking( libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                                    const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                                    const unsigned int              i_arch ) {
  unsigned int l_m_blocking = 0;

  if ( ( i_arch == LIBXSMM_AARCH64_V81 ) && ( LIBXSMM_GEMM_PRECISION_F32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) ) {
    /* Remark switching ti OUT datatype check here to cover BF16 in, Fp32/Int32 out kernel with the same logic */
    /* @TODO check if there is a better blocking strategy */
    if ( i_xgemm_desc->m >= 16 ) {
      l_m_blocking = 16;
    } else {
      l_m_blocking = i_xgemm_desc->m;
      /* in case we don't have a full vector length, we use masking */
      if (l_m_blocking == 15) {  /* for 15 we would need 5 M registers :-( 4-4-4-2-1 */
        l_m_blocking = 12;
      }
    }
  } else if ( ( i_arch == LIBXSMM_AARCH64_V81 ) && ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    /* @TODO check if there is a better blocking strategy */
    if ( i_xgemm_desc->m >= 8 ) {
      l_m_blocking = 8;
    } else {
      l_m_blocking = i_xgemm_desc->m;
    }
  } else if ( ( i_arch == LIBXSMM_AARCH64_A64FX ) && ( LIBXSMM_GEMM_PRECISION_F32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) ) {
    /* Remark switching ti OUT datatype check here to cover BF16 in, Fp32/Int32 out kernel with the same logic */
    /* @TODO check if there is a better blocking strategy */
    if ( i_xgemm_desc->m >= 64 ) {
      l_m_blocking = 64;
    } else {
      l_m_blocking = i_xgemm_desc->m;
    }
  } else if ( ( i_arch == LIBXSMM_AARCH64_A64FX ) && ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    /* @TODO check if there is a better blocking strategy */
    if ( i_xgemm_desc->m >= 32 ) {
      l_m_blocking = 32;
    } else {
      l_m_blocking = i_xgemm_desc->m;
    }
  } else {
    /* we should never end up here, if we do let the user know */
    /*LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
    return 0;*/
  }

  libxsmm_generator_gemm_init_micro_kernel_config_aarch64( io_micro_kernel_config, i_arch, i_xgemm_desc );

  return l_m_blocking;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_aarch64_update_m_blocking( libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                               const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                               const unsigned int             i_arch,
                                                               const unsigned int             i_current_m_blocking ) {
  unsigned int l_m_blocking = 0;

  if ( ( i_arch == LIBXSMM_AARCH64_V81 ) && ( LIBXSMM_GEMM_PRECISION_F32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) ) {
    /* Remark switching ti OUT datatype check here to cover BF16 in, Fp32 out kernel with the same logic */
    if (i_current_m_blocking == 16) {
      l_m_blocking = i_xgemm_desc->m % 16;
      if (l_m_blocking == 15) { /* for 15 we would need 5 M registers 4-4-4-2-1 */
        l_m_blocking = 12;
      }
    } else if ( i_current_m_blocking == 12 && i_xgemm_desc->m != 12 ) {
      l_m_blocking = i_xgemm_desc->m % 4;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( ( i_arch == LIBXSMM_AARCH64_V81 ) && ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    if (i_current_m_blocking == 8) {
      l_m_blocking = i_xgemm_desc->m % 8;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( ( i_arch == LIBXSMM_AARCH64_A64FX ) && ( LIBXSMM_GEMM_PRECISION_F32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) ) {
    /* Remark switching ti OUT datatype check here to cover BF16 in, Fp32 out kernel with the same logic */
    if (i_current_m_blocking == 64 ) {
      l_m_blocking = i_xgemm_desc->m % 64;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( ( i_arch == LIBXSMM_AARCH64_A64FX ) && ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    if (i_current_m_blocking == 32) {
      l_m_blocking = i_xgemm_desc->m % 32;
    } else {
      /* we are done with m_blocking */
    }
  } else {
    /* we should never end up here, if we do let the user know */
    /*LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
    return 0;*/
  }

  libxsmm_generator_gemm_init_micro_kernel_config_aarch64( io_micro_kernel_config, i_arch, i_xgemm_desc );

  return l_m_blocking;
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_setup_n_blocking( libxsmm_generated_code*        io_generated_code,
                                                      libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                      const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                      const unsigned int             i_arch,
                                                      unsigned int*                  o_n_N,
                                                      unsigned int*                  o_n_n) {
  unsigned int max_n_blocking = libxsmm_generator_gemm_aarch64_get_max_n_blocking( io_micro_kernel_config, i_xgemm_desc, i_arch );
  const unsigned int init_m_blocking = libxsmm_generator_gemm_aarch64_get_initial_m_blocking( io_micro_kernel_config, i_xgemm_desc, i_arch );
  unsigned int init_m_blocks = 0;

  /* check for valid values */
  if ( max_n_blocking == 0 || io_micro_kernel_config->vector_length == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  init_m_blocks = LIBXSMM_UPDIV(init_m_blocking, io_micro_kernel_config->vector_length);

  /* increment m register blocking in case of 2 remainder registers */
  if ( init_m_blocking % io_micro_kernel_config->vector_length == 3 ) {
    init_m_blocks++;
  }
  while ((init_m_blocks * max_n_blocking + init_m_blocks + 1) > io_micro_kernel_config->vector_reg_count) {
    max_n_blocking--;
  }

  libxsmm_compute_equalized_blocking( i_xgemm_desc->n, max_n_blocking, &(o_n_N[0]), &(o_n_n[0]), &(o_n_N[1]), &(o_n_n[1]) );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_setup_k_strides( libxsmm_generated_code*            io_generated_code,
                                                     const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                     const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                     const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                     const unsigned int                 i_m_blocking,
                                                     const unsigned int                 i_n_blocking ) {
  /* temp variable for b-offset to handle no-trans/trans B */
  int l_b_offset = 0;
  /* register blocking counter in n */
  unsigned int l_n = 0;

  /* preload offset of B */
  if ( io_generated_code->arch == LIBXSMM_AARCH64_V81 ) {
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
      l_b_offset = i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in;
    } else {
      l_b_offset = i_micro_kernel_config->datatype_size_in;
    }
  } else if ( io_generated_code->arch == LIBXSMM_AARCH64_A64FX ) {
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
      l_b_offset = (i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in) - (i_micro_kernel_config->datatype_size_in*i_n_blocking);
    } else {
      l_b_offset = i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in;
      libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_help_2, (unsigned long long)l_b_offset );

      l_b_offset = i_xgemm_desc->ldb - i_n_blocking;
      l_b_offset *= i_micro_kernel_config->datatype_size_in;;
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_help_1, (unsigned long long)l_b_offset );

  /* preload offset of A */
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_help_0,
                                           (unsigned long long)((unsigned long long)(i_xgemm_desc->lda - i_m_blocking) * i_micro_kernel_config->datatype_size_in) );

  /* load b offsets */
  if ( io_generated_code->arch != LIBXSMM_AARCH64_A64FX ) {
    if ( i_n_blocking < 7 ) {
      for ( l_n = 1; l_n < i_n_blocking; l_n++ ) {
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = l_n * i_micro_kernel_config->datatype_size_in;
        } else {
          l_b_offset = i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in;
        }
        libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 + (l_n - 1), (unsigned long long)l_b_offset );
      }
    }
  }
}

