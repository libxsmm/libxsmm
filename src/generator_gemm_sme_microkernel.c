/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
*               Intel Corporation - All rights reserved                       *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Stefan Remke (Univ. Jena)
******************************************************************************/
#include "generator_aarch64_instructions.h"
#include "generator_common_aarch64.h"
#include "generator_gemm_common_aarch64.h"
#include "generator_gemm_aarch64.h"
#include "generator_common.h"
#include "generator_gemm_sme_microkernel.h"

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_microkernel_sme( libxsmm_generated_code*            io_generated_code,
                                                     const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                     const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                     const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                     const unsigned int                 i_m_blocking,
                                                     const unsigned int                 i_n_blocking ){
  /* load A and B */
  libxsmm_aarch64_instruction_sme_mov( io_generated_code,
                                       LIBXSMM_AARCH64_INSTR_SVE2_LD1W_2,
                                       0,
                                       i_gp_reg_mapping->gp_reg_a,
                                       0,
                                       LIBXSMM_AARCH64_SVE_REG_P8);

  libxsmm_aarch64_instruction_sme_mov( io_generated_code,
                                       LIBXSMM_AARCH64_INSTR_SVE2_LD1W_2,
                                       2,
                                       ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0) ? i_gp_reg_mapping->gp_reg_b : i_gp_reg_mapping->gp_reg_reduce_count,
                                       0,
                                       LIBXSMM_AARCH64_SVE_REG_P9);

  /* update pointer */
  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                        i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                        0, LIBXSMM_AARCH64_SHIFTMODE_LSL );

  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                       ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0) ? i_gp_reg_mapping->gp_reg_b : i_gp_reg_mapping->gp_reg_reduce_count,
                                                       i_gp_reg_mapping->gp_reg_help_1,
                                                       ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0) ? i_gp_reg_mapping->gp_reg_b : i_gp_reg_mapping->gp_reg_reduce_count,
                                                       0, LIBXSMM_AARCH64_SHIFTMODE_LSL );


  /* compute fmopa */
  libxsmm_aarch64_instruction_sme_compute( io_generated_code,
                                           LIBXSMM_AARCH64_INSTR_SME_FMOPA_SP,
                                           0,
                                           LIBXSMM_AARCH64_SVE_REG_Z2,
                                           LIBXSMM_AARCH64_SVE_REG_Z0,
                                           LIBXSMM_AARCH64_SVE_REG_P1,
                                           LIBXSMM_AARCH64_SVE_REG_P0 );
  if( i_m_blocking > 16 ){
    libxsmm_aarch64_instruction_sme_compute( io_generated_code,
                                            LIBXSMM_AARCH64_INSTR_SME_FMOPA_SP,
                                            1,
                                            LIBXSMM_AARCH64_SVE_REG_Z2,
                                            LIBXSMM_AARCH64_SVE_REG_Z1,
                                            LIBXSMM_AARCH64_SVE_REG_P1,
                                            LIBXSMM_AARCH64_SVE_REG_P2 );
  }
  if( i_n_blocking > 16 ){
    libxsmm_aarch64_instruction_sme_compute( io_generated_code,
                                            LIBXSMM_AARCH64_INSTR_SME_FMOPA_SP,
                                            2,
                                            LIBXSMM_AARCH64_SVE_REG_Z3,
                                            LIBXSMM_AARCH64_SVE_REG_Z0,
                                            LIBXSMM_AARCH64_SVE_REG_P3,
                                            LIBXSMM_AARCH64_SVE_REG_P0 );
  }
  if( i_m_blocking > 16 && i_n_blocking > 16){
    libxsmm_aarch64_instruction_sme_compute( io_generated_code,
                                            LIBXSMM_AARCH64_INSTR_SME_FMOPA_SP,
                                            3,
                                            LIBXSMM_AARCH64_SVE_REG_Z3,
                                            LIBXSMM_AARCH64_SVE_REG_Z1,
                                            LIBXSMM_AARCH64_SVE_REG_P3,
                                            LIBXSMM_AARCH64_SVE_REG_P2 );
  }

}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_microkernel_sme_64x16( libxsmm_generated_code*            io_generated_code,
                                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                           const unsigned int                 i_m_blocking,
                                                           const unsigned int                 i_n_blocking ){
   /* load A and B */
  libxsmm_aarch64_instruction_sme_mov( io_generated_code,
                                       LIBXSMM_AARCH64_INSTR_SVE2_LD1W_4,
                                       0,
                                       i_gp_reg_mapping->gp_reg_a,
                                       0,
                                       LIBXSMM_AARCH64_SVE_REG_P8);
   libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                         LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF,
                                         ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0) ? i_gp_reg_mapping->gp_reg_b : i_gp_reg_mapping->gp_reg_reduce_count,
                                         0,
                                         0,
                                         LIBXSMM_AARCH64_SVE_REG_Z4,
                                         LIBXSMM_AARCH64_SVE_REG_P4 ) ;

  /* update pointer */
  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                        i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                        0, LIBXSMM_AARCH64_SHIFTMODE_LSL );

  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                       ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0) ? i_gp_reg_mapping->gp_reg_b : i_gp_reg_mapping->gp_reg_reduce_count,
                                                       i_gp_reg_mapping->gp_reg_help_1,
                                                       ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0) ? i_gp_reg_mapping->gp_reg_b : i_gp_reg_mapping->gp_reg_reduce_count,
                                                       0, LIBXSMM_AARCH64_SHIFTMODE_LSL );

  /* compute fmopa */
  libxsmm_aarch64_instruction_sme_compute( io_generated_code,
                                           LIBXSMM_AARCH64_INSTR_SME_FMOPA_SP,
                                           0,
                                           LIBXSMM_AARCH64_SVE_REG_Z4,
                                           LIBXSMM_AARCH64_SVE_REG_Z0,
                                           LIBXSMM_AARCH64_SVE_REG_P4,
                                           LIBXSMM_AARCH64_SVE_REG_P0 );
  if( i_m_blocking > 16 ){
    libxsmm_aarch64_instruction_sme_compute( io_generated_code,
                                             LIBXSMM_AARCH64_INSTR_SME_FMOPA_SP,
                                             1,
                                             LIBXSMM_AARCH64_SVE_REG_Z4,
                                             LIBXSMM_AARCH64_SVE_REG_Z1,
                                             LIBXSMM_AARCH64_SVE_REG_P4,
                                             LIBXSMM_AARCH64_SVE_REG_P1 );
  }
  if( i_m_blocking > 32 ){
    libxsmm_aarch64_instruction_sme_compute( io_generated_code,
                                            LIBXSMM_AARCH64_INSTR_SME_FMOPA_SP,
                                            2,
                                            LIBXSMM_AARCH64_SVE_REG_Z4,
                                            LIBXSMM_AARCH64_SVE_REG_Z2,
                                            LIBXSMM_AARCH64_SVE_REG_P4,
                                            LIBXSMM_AARCH64_SVE_REG_P2 );
  }
  if( i_m_blocking > 48 ){
    libxsmm_aarch64_instruction_sme_compute( io_generated_code,
                                            LIBXSMM_AARCH64_INSTR_SME_FMOPA_SP,
                                            3,
                                            LIBXSMM_AARCH64_SVE_REG_Z4,
                                            LIBXSMM_AARCH64_SVE_REG_Z3,
                                            LIBXSMM_AARCH64_SVE_REG_P4,
                                            LIBXSMM_AARCH64_SVE_REG_P3 );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_microkernel_sme_16x64( libxsmm_generated_code*            io_generated_code,
                                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                           const unsigned int                 i_m_blocking,
                                                           const unsigned int                 i_n_blocking ){
  /* load A and B */
  libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                        LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF,
                                        i_gp_reg_mapping->gp_reg_a,
                                        0,
                                        0,
                                        LIBXSMM_AARCH64_SVE_REG_Z4,
                                        LIBXSMM_AARCH64_SVE_REG_P4 );
  libxsmm_aarch64_instruction_sme_mov( io_generated_code,
                                       LIBXSMM_AARCH64_INSTR_SVE2_LD1W_4,
                                       0,
                                       ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0) ? i_gp_reg_mapping->gp_reg_b : i_gp_reg_mapping->gp_reg_reduce_count,
                                       0,
                                       LIBXSMM_AARCH64_SVE_REG_P8);
  /* update pointer */
  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                       i_gp_reg_mapping->gp_reg_a,
                                                       i_gp_reg_mapping->gp_reg_help_0,
                                                       i_gp_reg_mapping->gp_reg_a,
                                                        0, LIBXSMM_AARCH64_SHIFTMODE_LSL );

  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                       ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0) ? i_gp_reg_mapping->gp_reg_b : i_gp_reg_mapping->gp_reg_reduce_count,
                                                       i_gp_reg_mapping->gp_reg_help_1,
                                                       ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0) ? i_gp_reg_mapping->gp_reg_b : i_gp_reg_mapping->gp_reg_reduce_count,
                                                       0, LIBXSMM_AARCH64_SHIFTMODE_LSL );

  /* compute fmopa */
  libxsmm_aarch64_instruction_sme_compute( io_generated_code,
                                           LIBXSMM_AARCH64_INSTR_SME_FMOPA_SP,
                                           0,
                                           LIBXSMM_AARCH64_SVE_REG_Z0,
                                           LIBXSMM_AARCH64_SVE_REG_Z4,
                                           LIBXSMM_AARCH64_SVE_REG_P0,
                                           LIBXSMM_AARCH64_SVE_REG_P4 );
  if( i_n_blocking > 16 ){
    libxsmm_aarch64_instruction_sme_compute( io_generated_code,
                                             LIBXSMM_AARCH64_INSTR_SME_FMOPA_SP,
                                             1,
                                             LIBXSMM_AARCH64_SVE_REG_Z1,
                                             LIBXSMM_AARCH64_SVE_REG_Z4,
                                             LIBXSMM_AARCH64_SVE_REG_P1,
                                             LIBXSMM_AARCH64_SVE_REG_P4 );
  }
  if( i_n_blocking > 32 ){
    libxsmm_aarch64_instruction_sme_compute( io_generated_code,
                                            LIBXSMM_AARCH64_INSTR_SME_FMOPA_SP,
                                            2,
                                            LIBXSMM_AARCH64_SVE_REG_Z2,
                                            LIBXSMM_AARCH64_SVE_REG_Z4,
                                            LIBXSMM_AARCH64_SVE_REG_P2,
                                            LIBXSMM_AARCH64_SVE_REG_P4 );
  }
  if( i_n_blocking > 48 ){
    libxsmm_aarch64_instruction_sme_compute( io_generated_code,
                                            LIBXSMM_AARCH64_INSTR_SME_FMOPA_SP,
                                            3,
                                            LIBXSMM_AARCH64_SVE_REG_Z3,
                                            LIBXSMM_AARCH64_SVE_REG_Z4,
                                            LIBXSMM_AARCH64_SVE_REG_P3,
                                            LIBXSMM_AARCH64_SVE_REG_P4 );
  }
}
