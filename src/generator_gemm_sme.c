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
#include "generator_gemm_sme.h"
#include "generator_mateltwise_aarch64.h"
#include "generator_mateltwise_transform_common.h"
#include "generator_gemm_sme_microkernel.h"
#include "generator_common.h"


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_kloop_sme_het( libxsmm_generated_code*            io_generated_code,
                                                   libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                   const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                   const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                   const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                   const unsigned int                 i_m_blocking,
                                                   const unsigned int                 i_n_blocking,
                                                   const unsigned int                 i_blocking_scheme, /* 0 = 32x32 , 1 = 64x16, 2 = 16x64  */
                                                   const unsigned int                 i_trans_size ){
  void (*l_generator_microkernel)( libxsmm_generated_code*, const libxsmm_gp_reg_mapping*, const libxsmm_micro_kernel_config*, const libxsmm_gemm_descriptor*,
                                   const unsigned int, const unsigned int );
  if( i_blocking_scheme == 0){
    l_generator_microkernel = libxsmm_generator_gemm_aarch64_microkernel_sme;
  } else if( i_blocking_scheme == 1){
    l_generator_microkernel = libxsmm_generator_gemm_aarch64_microkernel_sme_64x16;
  } else {
    l_generator_microkernel = libxsmm_generator_gemm_aarch64_microkernel_sme_16x64;
  }

  /* advance A and B */
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code,
                                             i_gp_reg_mapping->gp_reg_help_0,
                                             ((long long)i_xgemm_desc->lda ) * i_micro_kernel_config->datatype_size_in );

  if( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ){
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code,
                                               i_gp_reg_mapping->gp_reg_help_1,
                                               ((long long)i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in ));
  } else {
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code,
                                               i_gp_reg_mapping->gp_reg_help_1,
                                               (long long ) i_trans_size * i_micro_kernel_config->datatype_size_in);

  }

  /* set predication registers*/
  if( i_blocking_scheme == 0){
    /* load */
    libxsmm_generator_set_pn_register_aarch64_sve2( io_generated_code,
                                                    LIBXSMM_AARCH64_SVE_REG_P8,
                                                    i_m_blocking * i_micro_kernel_config->datatype_size_in,
                                                    i_gp_reg_mapping->gp_reg_help_2,
                                                    0 );
    libxsmm_generator_set_pn_register_aarch64_sve2( io_generated_code,
                                                    LIBXSMM_AARCH64_SVE_REG_P9,
                                                    i_n_blocking * i_micro_kernel_config->datatype_size_in,
                                                    i_gp_reg_mapping->gp_reg_help_2,
                                                    0 );
    if( i_m_blocking == 32  ){
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                    LIBXSMM_AARCH64_SVE_REG_P0,
                                                    -1,
                                                    i_gp_reg_mapping->gp_reg_help_2 );
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                    LIBXSMM_AARCH64_SVE_REG_P2,
                                                    -1,
                                                    i_gp_reg_mapping->gp_reg_help_2 );
    } else if( i_m_blocking > 16 ) {
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                    LIBXSMM_AARCH64_SVE_REG_P0,
                                                    -1,
                                                    i_gp_reg_mapping->gp_reg_help_2 );
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                    LIBXSMM_AARCH64_SVE_REG_P2,
                                                    (i_m_blocking-16)*i_micro_kernel_config->datatype_size_in,
                                                    i_gp_reg_mapping->gp_reg_help_2 );
    } else {
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                    LIBXSMM_AARCH64_SVE_REG_P0,
                                                    (i_m_blocking)*i_micro_kernel_config->datatype_size_in,
                                                    i_gp_reg_mapping->gp_reg_help_2 );
    }
    if( i_n_blocking == 32  ){
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                    LIBXSMM_AARCH64_SVE_REG_P1,
                                                    -1,
                                                    i_gp_reg_mapping->gp_reg_help_2 );
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                    LIBXSMM_AARCH64_SVE_REG_P3,
                                                    -1,
                                                    i_gp_reg_mapping->gp_reg_help_2 );
    } else if( i_n_blocking > 16 ) {
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                    LIBXSMM_AARCH64_SVE_REG_P1,
                                                    -1,
                                                    i_gp_reg_mapping->gp_reg_help_2 );
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                    LIBXSMM_AARCH64_SVE_REG_P3,
                                                    (i_n_blocking-16)*i_micro_kernel_config->datatype_size_in,
                                                    i_gp_reg_mapping->gp_reg_help_2 );
    } else {
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                    LIBXSMM_AARCH64_SVE_REG_P1,
                                                    (i_n_blocking)*i_micro_kernel_config->datatype_size_in,
                                                    i_gp_reg_mapping->gp_reg_help_2 );
    }
  } else if( i_blocking_scheme == 1){
    /* load */
    libxsmm_generator_set_pn_register_aarch64_sve2( io_generated_code,
                                                    LIBXSMM_AARCH64_SVE_REG_P8,
                                                    i_m_blocking * i_micro_kernel_config->datatype_size_in,
                                                    i_gp_reg_mapping->gp_reg_help_2,
                                                    1 );
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                  LIBXSMM_AARCH64_SVE_REG_P4,
                                                  i_n_blocking * i_micro_kernel_config->datatype_size_in,
                                                  i_gp_reg_mapping->gp_reg_help_2 );
    /* compute */
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                  LIBXSMM_AARCH64_SVE_REG_P0,
                                                  (i_m_blocking > 16 ) ? -1 : (int)i_m_blocking * (int)i_micro_kernel_config->datatype_size_in,
                                                  i_gp_reg_mapping->gp_reg_help_2 );
    if( i_m_blocking > 16 ){
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                  LIBXSMM_AARCH64_SVE_REG_P1,
                                                  (i_m_blocking > 32 ) ? -1 : (int)(i_m_blocking - 16)* (int)i_micro_kernel_config->datatype_size_in,
                                                  i_gp_reg_mapping->gp_reg_help_2 );
    }
    if( i_m_blocking > 32 ){
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                  LIBXSMM_AARCH64_SVE_REG_P2,
                                                  (i_m_blocking > 48 ) ? -1 : (int)(i_m_blocking - 32)* (int)i_micro_kernel_config->datatype_size_in,
                                                  i_gp_reg_mapping->gp_reg_help_2 );
    }
    if( i_m_blocking > 48 ){
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                  LIBXSMM_AARCH64_SVE_REG_P3,
                                                  (i_m_blocking == 64 ) ? -1 : (int)(i_m_blocking - 48)* (int)i_micro_kernel_config->datatype_size_in,
                                                  i_gp_reg_mapping->gp_reg_help_2 );
    }
  } else if( i_blocking_scheme == 2){
    /* load */
    libxsmm_generator_set_pn_register_aarch64_sve2( io_generated_code,
                                                    LIBXSMM_AARCH64_SVE_REG_P8,
                                                    i_n_blocking * i_micro_kernel_config->datatype_size_in,
                                                    i_gp_reg_mapping->gp_reg_help_2,
                                                    1 );
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                  LIBXSMM_AARCH64_SVE_REG_P4,
                                                  i_m_blocking * i_micro_kernel_config->datatype_size_in,
                                                  i_gp_reg_mapping->gp_reg_help_2 );
    /* compute */
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                  LIBXSMM_AARCH64_SVE_REG_P0,
                                                  (i_n_blocking > 16 ) ? -1 : (int)i_n_blocking * (int)i_micro_kernel_config->datatype_size_in,
                                                  i_gp_reg_mapping->gp_reg_help_2 );
    if( i_n_blocking > 16 ){
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                  LIBXSMM_AARCH64_SVE_REG_P1,
                                                  (i_n_blocking > 32 ) ? -1 : (int)(i_n_blocking - 16)* (int)i_micro_kernel_config->datatype_size_in,
                                                  i_gp_reg_mapping->gp_reg_help_2 );
    }
    if( i_n_blocking > 32 ){
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                  LIBXSMM_AARCH64_SVE_REG_P2,
                                                  (i_n_blocking > 48 ) ? -1 : (int)(i_n_blocking - 32)* (int)i_micro_kernel_config->datatype_size_in,
                                                  i_gp_reg_mapping->gp_reg_help_2 );
    }
    if( i_n_blocking > 48 ){
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                  LIBXSMM_AARCH64_SVE_REG_P3,
                                                  (i_n_blocking == 64 ) ? -1 : (int)(i_n_blocking - 48)* (int)i_micro_kernel_config->datatype_size_in,
                                                  i_gp_reg_mapping->gp_reg_help_2 );
    }
  } else {
    /* this should not happen! */
  }

  /* Handle BRGEMM */
  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0 ) {

    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_help_4, 0 );
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_help_5, 0 );

    /* open BR loop */
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_help_6, LIBXSMM_AARCH64_GP_REG_XZR, 0,
                                          i_gp_reg_mapping->gp_reg_reduce_loop );
    libxsmm_aarch64_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  }

  /* set k loop counter */
  libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_kloop, (unsigned int)i_xgemm_desc->k );

  /* apply microkernel */
  l_generator_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, i_m_blocking, i_n_blocking);

  libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_kloop, 1 );

  /* reset A pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                 LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 i_gp_reg_mapping->gp_reg_a,
                                                 i_gp_reg_mapping->gp_reg_help_0,
                                                 i_gp_reg_mapping->gp_reg_a,
                                                 (long long)i_xgemm_desc->k * i_xgemm_desc->lda * i_micro_kernel_config->datatype_size_in );

  /* reset B pointer */
  if( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ){
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_b,
                                                   i_gp_reg_mapping->gp_reg_help_1,
                                                   i_gp_reg_mapping->gp_reg_b,
                                                   (long long)i_xgemm_desc->k * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in );
  } else {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_reduce_count,
                                                   i_gp_reg_mapping->gp_reg_help_1,
                                                   i_gp_reg_mapping->gp_reg_reduce_count,
                                                   (long long) i_xgemm_desc->k * i_trans_size * i_micro_kernel_config->datatype_size_in);
  }

  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0 ) {
          /* increment forward counting BRGEMM count */
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_help_2, i_xgemm_desc->c1 );
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_scf, i_xgemm_desc->c2 );
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, i_gp_reg_mapping->gp_reg_help_2,
                                                          i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_a, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    if( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0  ) {
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, i_gp_reg_mapping->gp_reg_scf,
                                                          i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_b, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    } else {
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, i_gp_reg_mapping->gp_reg_scf,
                                                          i_gp_reg_mapping->gp_reg_reduce_count, i_gp_reg_mapping->gp_reg_reduce_count, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    }
    
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, i_gp_reg_mapping->gp_reg_help_2,
                                                          i_gp_reg_mapping->gp_reg_help_4, i_gp_reg_mapping->gp_reg_help_4, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, i_gp_reg_mapping->gp_reg_scf,
                                                          i_gp_reg_mapping->gp_reg_help_5, i_gp_reg_mapping->gp_reg_help_5, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    /* close BRGEMM loop */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_reduce_loop, 1 );

    /* restore A and B register */
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR, i_gp_reg_mapping->gp_reg_a,
                                                          i_gp_reg_mapping->gp_reg_help_4, i_gp_reg_mapping->gp_reg_a, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    if( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0  ) {
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR, i_gp_reg_mapping->gp_reg_b,
                                                            i_gp_reg_mapping->gp_reg_help_5, i_gp_reg_mapping->gp_reg_b, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    } else {
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR, i_gp_reg_mapping->gp_reg_reduce_count,
                                                          i_gp_reg_mapping->gp_reg_help_5, i_gp_reg_mapping->gp_reg_reduce_count, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    }
  }

}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_kernel_sme_het_blocking( libxsmm_generated_code*        io_generated_code,
                                                           const libxsmm_gemm_descriptor* i_xgemm_desc ){
  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;
  libxsmm_gemm_descriptor*          l_xgemm_desc_opa;
  libxsmm_gemm_descriptor           l_new_xgemm_desc_opa;

  unsigned int l_m_scheme[2]     = {0,0};  /* [0] = 32 type blocks, [1] = 16 type blocks */
  unsigned int l_n_scheme[2]     = {0,0};  /* [0] = 64 type blocks, [1] = rest type blocks */
  unsigned int l_m_rest = 0;
  unsigned int l_n_rest = 0;
  unsigned int l_m_2 = 0;
  unsigned int l_trans_loop = 0;
  unsigned int l_trans_rest = 0;
  unsigned int l_perfect_blocking_m = 64;
  unsigned int l_perfect_m_count =0;
  unsigned int l_rest_m = 0;
  unsigned int l_m_blocking = 0;
  libxsmm_generator_gemm_aarch64_setup_blocking_sme( i_xgemm_desc,
                                                     l_m_scheme,
                                                     l_n_scheme);
  libxsmm_generator_gemm_init_micro_kernel_config_aarch64( &l_micro_kernel_config,
                                                           io_generated_code->arch,
                                                           i_xgemm_desc );


  /* define gp register mapping */
  libxsmm_reset_aarch64_gp_reg_mapping( &l_gp_reg_mapping );

  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_AARCH64_GP_REG_X0;
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_AARCH64_GP_REG_X0;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_AARCH64_GP_REG_X1;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_AARCH64_GP_REG_X2;
  l_gp_reg_mapping.gp_reg_reduce_count = LIBXSMM_AARCH64_GP_REG_X3;
  l_gp_reg_mapping.gp_reg_a_offset = LIBXSMM_AARCH64_GP_REG_X4;
  l_gp_reg_mapping.gp_reg_b_offset = LIBXSMM_AARCH64_GP_REG_X5;
  l_gp_reg_mapping.gp_reg_mloop = LIBXSMM_AARCH64_GP_REG_X6;
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_AARCH64_GP_REG_X7;
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_AARCH64_GP_REG_X8;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_AARCH64_GP_REG_X9;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_AARCH64_GP_REG_X10;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_AARCH64_GP_REG_X11;
  l_gp_reg_mapping.gp_reg_scf    = LIBXSMM_AARCH64_GP_REG_X13;
  l_gp_reg_mapping.gp_reg_help_3 = LIBXSMM_AARCH64_GP_REG_X17;
  l_gp_reg_mapping.gp_reg_help_4 = LIBXSMM_AARCH64_GP_REG_X20;
  l_gp_reg_mapping.gp_reg_help_5 = LIBXSMM_AARCH64_GP_REG_X21;
  l_gp_reg_mapping.gp_reg_reduce_loop = LIBXSMM_AARCH64_GP_REG_X22;
  l_gp_reg_mapping.gp_reg_help_6 = LIBXSMM_AARCH64_GP_REG_X23;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  libxsmm_aarch64_instruction_open_stream( io_generated_code, 0xe0f );

  l_new_xgemm_desc_opa = *i_xgemm_desc;
  l_xgemm_desc_opa = (libxsmm_gemm_descriptor*) &l_new_xgemm_desc_opa;

  if( l_m_scheme[1] > 0 ){
    l_m_rest = l_xgemm_desc_opa->m - l_m_scheme[0]*32;
  }
  if( l_n_scheme[1] > 0){
    l_n_rest = l_xgemm_desc_opa->n - l_n_scheme[0]*64;
  }

  if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI & l_xgemm_desc_opa->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI) ||
       ((LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI & l_xgemm_desc_opa->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI) ) {
    int l_offset_ptr_a = (int)sizeof(libxsmm_matrix_op_arg);
    int l_offset_ptr_b = (int)(sizeof(libxsmm_matrix_op_arg) + sizeof(libxsmm_matrix_arg));
    int l_offset_ptr_c = (int)(sizeof(libxsmm_matrix_op_arg) + 2*sizeof(libxsmm_matrix_arg));
    /* RDI holds the pointer to the struct, so lets first move this one into R15 */
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_AND_SR,
                                                         l_gp_reg_mapping.gp_reg_param_struct, l_gp_reg_mapping.gp_reg_param_struct, l_gp_reg_mapping.gp_reg_help_1,
                                                         0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    /* A pointer */
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                          l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, l_offset_ptr_a, l_gp_reg_mapping.gp_reg_a );
    /* B pointer */
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                          l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, l_offset_ptr_b, l_gp_reg_mapping.gp_reg_b );
    /* C pointer */
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                          l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, l_offset_ptr_c, l_gp_reg_mapping.gp_reg_c );
  }

  if ((l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 1) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                            l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 16, l_gp_reg_mapping.gp_reg_help_6 );
  }

  /* setting up the stack frame */
  libxsmm_generator_gemm_setup_stack_frame_aarch64( io_generated_code, i_xgemm_desc, &l_gp_reg_mapping, &l_micro_kernel_config);

  libxsmm_aarch64_instruction_sm( io_generated_code,
                                  LIBXSMM_AARCH64_INSTR_SME_SMSTART);

  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  if( l_n_scheme[0] > 0 ){

    /* open N loop */
    libxsmm_generator_loop_header_aarch64( io_generated_code,
                                          &l_loop_label_tracker,
                                          l_gp_reg_mapping.gp_reg_nloop,
                                          l_n_scheme[0]*64);


    if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0 ){
      /* save address of stackpointer */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    LIBXSMM_AARCH64_GP_REG_XSP,
                                                    l_gp_reg_mapping.gp_reg_help_1,
                                                    l_gp_reg_mapping.gp_reg_help_1,
                                                    0);
      /* allocate memory on stack for transposed B */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                    LIBXSMM_AARCH64_GP_REG_XSP,
                                                    l_gp_reg_mapping.gp_reg_help_1,
                                                    LIBXSMM_AARCH64_GP_REG_XSP,
                                                    64 * l_micro_kernel_config.datatype_size_in * l_xgemm_desc_opa->k );

      /* store address of stack pointer to transposed B register ( x3 )*/
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    LIBXSMM_AARCH64_GP_REG_XSP,
                                                    l_gp_reg_mapping.gp_reg_help_1,
                                                    l_gp_reg_mapping.gp_reg_reduce_count ,
                                                    0);
      /* store address into x26 */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    LIBXSMM_AARCH64_GP_REG_XSP,
                                                    l_gp_reg_mapping.gp_reg_help_1,
                                                    LIBXSMM_AARCH64_GP_REG_X26,
                                                    0);

      l_trans_loop = l_xgemm_desc_opa->k / 16;
      l_trans_rest = l_xgemm_desc_opa->k % 16;

      if( l_trans_loop > 0 ){
        libxsmm_generator_loop_header_aarch64(io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_help_2, l_trans_loop);

        libxsmm_generator_sme_transpose_64( io_generated_code,
                                            l_gp_reg_mapping.gp_reg_b,
                                            l_xgemm_desc_opa->ldb,
                                            0,
                                            64,
                                            l_gp_reg_mapping.gp_reg_help_0 );

        /* advance pointer B */
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                      LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                      l_gp_reg_mapping.gp_reg_b,
                                                      l_gp_reg_mapping.gp_reg_help_0,
                                                      l_gp_reg_mapping.gp_reg_b ,
                                                      (long long )16*l_micro_kernel_config.datatype_size_in );
        libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_help_2, 1 );
      }
      if( l_trans_rest > 0 ){
        libxsmm_generator_sme_transpose_64( io_generated_code,
                                            l_gp_reg_mapping.gp_reg_b,
                                            l_xgemm_desc_opa->ldb,
                                            l_trans_rest,
                                            64,
                                            l_gp_reg_mapping.gp_reg_help_0 );
      }
      /* reset address of stackpointer */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    l_gp_reg_mapping.gp_reg_help_1,
                                                    l_gp_reg_mapping.gp_reg_help_1,
                                                    LIBXSMM_AARCH64_GP_REG_XSP,
                                                    0);
      /* reset b pointer */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                    l_gp_reg_mapping.gp_reg_b,
                                                    l_gp_reg_mapping.gp_reg_help_0,
                                                    l_gp_reg_mapping.gp_reg_b ,
                                                    (l_xgemm_desc_opa->k - l_trans_rest) * l_micro_kernel_config.datatype_size_in );
    } else {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    l_gp_reg_mapping.gp_reg_b,
                                                    l_gp_reg_mapping.gp_reg_help_1,
                                                    LIBXSMM_AARCH64_GP_REG_X26,
                                                    0);

    }
    if( l_m_scheme[0] > 0){
      /* open M loop */
      libxsmm_generator_loop_header_aarch64( io_generated_code,
                                            &l_loop_label_tracker,
                                            l_gp_reg_mapping.gp_reg_mloop,
                                            l_m_scheme[0] * 32 );
      /* save pointer of x2 */
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                            LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                            LIBXSMM_AARCH64_GP_REG_XZR,
                                                            l_gp_reg_mapping.gp_reg_c,
                                                            l_gp_reg_mapping.gp_reg_help_3,
                                                            0,
                                                            LIBXSMM_AARCH64_SHIFTMODE_LSL );

      /* set help register x9 */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    l_gp_reg_mapping.gp_reg_c,
                                                    LIBXSMM_AARCH64_GP_REG_X11,
                                                    l_gp_reg_mapping.gp_reg_help_0,
                                                    (long long) l_xgemm_desc_opa->ldc * 16*l_micro_kernel_config.datatype_size_out );

      /* load C */
      libxsmm_generator_load_32x32_aarch64_sme( io_generated_code,
                                                          l_gp_reg_mapping.gp_reg_c,
                                                          l_gp_reg_mapping.gp_reg_help_0,
                                                          32,
                                                          32,
                                                          l_xgemm_desc_opa->ldc );

      /* compute outer product */
      libxsmm_generator_gemm_aarch64_kloop_sme_het( io_generated_code,
                                                    &l_loop_label_tracker,
                                                    &l_gp_reg_mapping,
                                                    &l_micro_kernel_config,
                                                    l_xgemm_desc_opa,
                                                    32,
                                                    32,
                                                    0,
                                                    64 );

      /* restore pointer of x2 */
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                            LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                            LIBXSMM_AARCH64_GP_REG_XZR,
                                                            l_gp_reg_mapping.gp_reg_help_3,
                                                            l_gp_reg_mapping.gp_reg_c,
                                                            0,
                                                            LIBXSMM_AARCH64_SHIFTMODE_LSL );
      /* set help register x9 */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    l_gp_reg_mapping.gp_reg_c,
                                                    LIBXSMM_AARCH64_GP_REG_X11,
                                                    l_gp_reg_mapping.gp_reg_help_0,
                                                    (long long) l_xgemm_desc_opa->ldc * 16*l_micro_kernel_config.datatype_size_out );

      libxsmm_generator_store_32x32_aarch64_sme( io_generated_code,
                                                 l_gp_reg_mapping.gp_reg_c,
                                                 l_gp_reg_mapping.gp_reg_help_0,
                                                 32,
                                                 32,
                                                 l_xgemm_desc_opa->ldc,
                                                 1 );

      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                          LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                          LIBXSMM_AARCH64_GP_REG_XZR,
                                                          l_gp_reg_mapping.gp_reg_help_3,
                                                          l_gp_reg_mapping.gp_reg_c,
                                                          0,
                                                          LIBXSMM_AARCH64_SHIFTMODE_LSL );

      /* advance C pointer */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    l_gp_reg_mapping.gp_reg_c,
                                                    l_gp_reg_mapping.gp_reg_help_2,
                                                    l_gp_reg_mapping.gp_reg_c,
                                                    (long long)32 * l_micro_kernel_config.datatype_size_in * l_xgemm_desc_opa->ldc );

      /* advance B pointer */
      if( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ){
        /* reset b pointer */
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                      LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                      l_gp_reg_mapping.gp_reg_b,
                                                      l_gp_reg_mapping.gp_reg_help_1,
                                                      l_gp_reg_mapping.gp_reg_b,
                                                      (long long) 32 * l_micro_kernel_config.datatype_size_in );
      } else {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                      LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                      l_gp_reg_mapping.gp_reg_reduce_count,
                                                      l_gp_reg_mapping.gp_reg_help_1,
                                                      l_gp_reg_mapping.gp_reg_reduce_count,
                                                      (long long)32 * l_micro_kernel_config.datatype_size_in );
      }

      /* right side of 32x64 kernel */
      /* save pointer of x2 to x17 */
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                            LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                            LIBXSMM_AARCH64_GP_REG_XZR,
                                                            l_gp_reg_mapping.gp_reg_c,
                                                            l_gp_reg_mapping.gp_reg_help_3,
                                                            0,
                                                            LIBXSMM_AARCH64_SHIFTMODE_LSL );

      /* set help register x9 */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    l_gp_reg_mapping.gp_reg_c,
                                                    LIBXSMM_AARCH64_GP_REG_X11,
                                                    l_gp_reg_mapping.gp_reg_help_0,
                                                    l_xgemm_desc_opa->ldc * 16*l_micro_kernel_config.datatype_size_out );

      /* load C */
      libxsmm_generator_load_32x32_aarch64_sme( io_generated_code,
                                                          l_gp_reg_mapping.gp_reg_c,
                                                          l_gp_reg_mapping.gp_reg_help_0,
                                                          32,
                                                          32,
                                                          l_xgemm_desc_opa->ldc );

      /* compute outer product */
      libxsmm_generator_gemm_aarch64_kloop_sme_het( io_generated_code,
                                                    &l_loop_label_tracker,
                                                    &l_gp_reg_mapping,
                                                    &l_micro_kernel_config,
                                                    l_xgemm_desc_opa,
                                                    32,
                                                    32,
                                                    0,
                                                    64 );

      /* restore pointer of x2 */
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                            LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                            LIBXSMM_AARCH64_GP_REG_XZR,
                                                            l_gp_reg_mapping.gp_reg_help_3,
                                                            l_gp_reg_mapping.gp_reg_c,
                                                            0,
                                                            LIBXSMM_AARCH64_SHIFTMODE_LSL );
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                        LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                        l_gp_reg_mapping.gp_reg_c,
                                                        LIBXSMM_AARCH64_GP_REG_X11,
                                                        l_gp_reg_mapping.gp_reg_help_0,
                                                        l_xgemm_desc_opa->ldc * 16*l_micro_kernel_config.datatype_size_out );

      libxsmm_generator_store_32x32_aarch64_sme( io_generated_code,
                                                      l_gp_reg_mapping.gp_reg_c,
                                                      l_gp_reg_mapping.gp_reg_help_0,
                                                      32,
                                                      32,
                                                      l_xgemm_desc_opa->ldc,
                                                      1 );

      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                          LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                          LIBXSMM_AARCH64_GP_REG_XZR,
                                                          l_gp_reg_mapping.gp_reg_help_3,
                                                          l_gp_reg_mapping.gp_reg_c,
                                                          0,
                                                          LIBXSMM_AARCH64_SHIFTMODE_LSL );

      /* reset C to next m block */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                    l_gp_reg_mapping.gp_reg_c,
                                                    LIBXSMM_AARCH64_GP_REG_X11,
                                                    l_gp_reg_mapping.gp_reg_c,
                                                    (long long) l_xgemm_desc_opa->ldc * l_micro_kernel_config.datatype_size_in * 32 - 32*l_micro_kernel_config.datatype_size_in );

      /* reset B */
      if((l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0){
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                        LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                        LIBXSMM_AARCH64_GP_REG_X26,
                                                        l_gp_reg_mapping.gp_reg_help_1,
                                                        l_gp_reg_mapping.gp_reg_reduce_count,
                                                        0);
      } else {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                        LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                        LIBXSMM_AARCH64_GP_REG_X26,
                                                        l_gp_reg_mapping.gp_reg_help_1,
                                                        l_gp_reg_mapping.gp_reg_b,
                                                        0);
      }
      /* Adjust A for next m block*/
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    l_gp_reg_mapping.gp_reg_a,
                                                    LIBXSMM_AARCH64_GP_REG_X11,
                                                    l_gp_reg_mapping.gp_reg_a,
                                                    (long long) 32*l_micro_kernel_config.datatype_size_in );

      /* close M loop */
        libxsmm_generator_loop_footer_aarch64( io_generated_code,
                                              &l_loop_label_tracker,
                                              l_gp_reg_mapping.gp_reg_mloop,
                                              32 );
    }

    /* rest of M */
    if( l_m_scheme[1] > 0 && l_m_rest < 17 ){
      /* save pointer of x2 to x17 */
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                          LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                          LIBXSMM_AARCH64_GP_REG_XZR,
                                                          l_gp_reg_mapping.gp_reg_c,
                                                          l_gp_reg_mapping.gp_reg_help_3,
                                                          0,
                                                          LIBXSMM_AARCH64_SHIFTMODE_LSL );
      /* load C */
      libxsmm_generated_load_16x64_aarch64_sme( io_generated_code,
                                                l_gp_reg_mapping.gp_reg_c,
                                                l_m_rest,
                                                64,
                                                l_xgemm_desc_opa->ldc );

      libxsmm_generator_gemm_aarch64_kloop_sme_het( io_generated_code,
                                                    &l_loop_label_tracker,
                                                    &l_gp_reg_mapping,
                                                    &l_micro_kernel_config,
                                                    l_xgemm_desc_opa,
                                                    l_m_rest,
                                                    64,
                                                    2,
                                                    64 );

      /* reset C pointer */
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                          LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                          LIBXSMM_AARCH64_GP_REG_XZR,
                                                          l_gp_reg_mapping.gp_reg_help_3,
                                                          l_gp_reg_mapping.gp_reg_c,
                                                          0,
                                                          LIBXSMM_AARCH64_SHIFTMODE_LSL );

      /* store C */
      libxsmm_generated_store_16x64_aarch64_sme( io_generated_code,
                                                l_gp_reg_mapping.gp_reg_c,
                                                l_m_rest,
                                                64,
                                                l_xgemm_desc_opa->ldc);
      /* reset C pointer */
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                          LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                          LIBXSMM_AARCH64_GP_REG_XZR,
                                                          l_gp_reg_mapping.gp_reg_help_3,
                                                          l_gp_reg_mapping.gp_reg_c,
                                                          0,
                                                          LIBXSMM_AARCH64_SHIFTMODE_LSL );

    } else if( l_m_scheme[1] > 0 && l_m_rest > 16){
      /* save pointer of x2 to x17 */
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                            LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                            LIBXSMM_AARCH64_GP_REG_XZR,
                                                            l_gp_reg_mapping.gp_reg_c,
                                                            l_gp_reg_mapping.gp_reg_help_3,
                                                            0,
                                                            LIBXSMM_AARCH64_SHIFTMODE_LSL );

      /* set help register x9 */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    l_gp_reg_mapping.gp_reg_c,
                                                    LIBXSMM_AARCH64_GP_REG_X11,
                                                    l_gp_reg_mapping.gp_reg_help_0,
                                                    (long long) l_xgemm_desc_opa->ldc * 16*l_micro_kernel_config.datatype_size_out );

      /* load C */
      libxsmm_generator_load_32x32_aarch64_sme( io_generated_code,
                                                          l_gp_reg_mapping.gp_reg_c,
                                                          l_gp_reg_mapping.gp_reg_help_0,
                                                          l_m_rest,
                                                          32,
                                                          l_xgemm_desc_opa->ldc );

      /* compute outer product */
      libxsmm_generator_gemm_aarch64_kloop_sme_het( io_generated_code,
                                                    &l_loop_label_tracker,
                                                    &l_gp_reg_mapping,
                                                    &l_micro_kernel_config,
                                                    l_xgemm_desc_opa,
                                                    l_m_rest,
                                                    32,
                                                    0,
                                                    64 );

      /* restore pointer of x2 */
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                            LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                            LIBXSMM_AARCH64_GP_REG_XZR,
                                                            l_gp_reg_mapping.gp_reg_help_3,
                                                            l_gp_reg_mapping.gp_reg_c,
                                                            0,
                                                            LIBXSMM_AARCH64_SHIFTMODE_LSL );
      /* set help register x9 */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    l_gp_reg_mapping.gp_reg_c,
                                                    LIBXSMM_AARCH64_GP_REG_X11,
                                                    l_gp_reg_mapping.gp_reg_help_0,
                                                    (long long) l_xgemm_desc_opa->ldc * 16*l_micro_kernel_config.datatype_size_out );

      libxsmm_generator_store_32x32_aarch64_sme( io_generated_code,
                                                      l_gp_reg_mapping.gp_reg_c,
                                                      l_gp_reg_mapping.gp_reg_help_0,
                                                      l_m_rest,
                                                      32,
                                                      l_xgemm_desc_opa->ldc,
                                                      1 );

      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                          LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                          LIBXSMM_AARCH64_GP_REG_XZR,
                                                          l_gp_reg_mapping.gp_reg_help_3,
                                                          l_gp_reg_mapping.gp_reg_c,
                                                          0,
                                                          LIBXSMM_AARCH64_SHIFTMODE_LSL );

      /* advance C pointer */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    l_gp_reg_mapping.gp_reg_c,
                                                    l_gp_reg_mapping.gp_reg_help_2,
                                                    l_gp_reg_mapping.gp_reg_c,
                                                    (long long)32 * l_micro_kernel_config.datatype_size_out * l_xgemm_desc_opa->ldc );

      /* advance B pointer */
      if( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ){
        /* reset b pointer */
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                      LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                      l_gp_reg_mapping.gp_reg_b,
                                                      l_gp_reg_mapping.gp_reg_help_1,
                                                      l_gp_reg_mapping.gp_reg_b,
                                                      (long long) 32 * l_micro_kernel_config.datatype_size_in );
      } else {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                      LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                      l_gp_reg_mapping.gp_reg_reduce_count,
                                                      l_gp_reg_mapping.gp_reg_help_1,
                                                      l_gp_reg_mapping.gp_reg_reduce_count,
                                                      (long long)32 * l_micro_kernel_config.datatype_size_in );
      }

      /* right side of 32x64 kernel */
      /* save pointer of x2 to x17 */
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                            LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                            LIBXSMM_AARCH64_GP_REG_XZR,
                                                            l_gp_reg_mapping.gp_reg_c,
                                                            l_gp_reg_mapping.gp_reg_help_3,
                                                            0,
                                                            LIBXSMM_AARCH64_SHIFTMODE_LSL );

      /* set help register x9 */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    l_gp_reg_mapping.gp_reg_c,
                                                    LIBXSMM_AARCH64_GP_REG_X11,
                                                    l_gp_reg_mapping.gp_reg_help_0,
                                                    (long long) l_xgemm_desc_opa->ldc * 16*l_micro_kernel_config.datatype_size_in );

      /* load C */
      libxsmm_generator_load_32x32_aarch64_sme( io_generated_code,
                                                          l_gp_reg_mapping.gp_reg_c,
                                                          l_gp_reg_mapping.gp_reg_help_0,
                                                          l_m_rest,
                                                          32,
                                                          l_xgemm_desc_opa->ldc );

      /* compute outer product */
      libxsmm_generator_gemm_aarch64_kloop_sme_het( io_generated_code,
                                                    &l_loop_label_tracker,
                                                    &l_gp_reg_mapping,
                                                    &l_micro_kernel_config,
                                                    l_xgemm_desc_opa,
                                                    l_m_rest,
                                                    32,
                                                    0,
                                                    64 );

      /* restore pointer of x2 */
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                            LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                            LIBXSMM_AARCH64_GP_REG_XZR,
                                                            l_gp_reg_mapping.gp_reg_help_3,
                                                            l_gp_reg_mapping.gp_reg_c,
                                                            0,
                                                            LIBXSMM_AARCH64_SHIFTMODE_LSL );
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                        LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                        l_gp_reg_mapping.gp_reg_c,
                                                        LIBXSMM_AARCH64_GP_REG_X11,
                                                        l_gp_reg_mapping.gp_reg_help_0,
                                                        (long long) l_xgemm_desc_opa->ldc * 16*l_micro_kernel_config.datatype_size_out );

      libxsmm_generator_store_32x32_aarch64_sme( io_generated_code,
                                                      l_gp_reg_mapping.gp_reg_c,
                                                      l_gp_reg_mapping.gp_reg_help_0,
                                                      l_m_rest,
                                                      32,
                                                      l_xgemm_desc_opa->ldc,
                                                      1 );

      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                          LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                          LIBXSMM_AARCH64_GP_REG_XZR,
                                                          l_gp_reg_mapping.gp_reg_help_3,
                                                          l_gp_reg_mapping.gp_reg_c,
                                                          0,
                                                          LIBXSMM_AARCH64_SHIFTMODE_LSL );

      /* reset B */
      if((l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0){
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                        LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                        LIBXSMM_AARCH64_GP_REG_X26,
                                                        l_gp_reg_mapping.gp_reg_help_1,
                                                        l_gp_reg_mapping.gp_reg_reduce_count,
                                                        0);
      } else {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                        LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                        LIBXSMM_AARCH64_GP_REG_X26,
                                                        l_gp_reg_mapping.gp_reg_help_1,
                                                        l_gp_reg_mapping.gp_reg_b,
                                                        0);
      }

      /* reset C */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                    l_gp_reg_mapping.gp_reg_c,
                                                    LIBXSMM_AARCH64_GP_REG_X11,
                                                    l_gp_reg_mapping.gp_reg_c,
                                                    (long long) l_xgemm_desc_opa->ldc * l_micro_kernel_config.datatype_size_out * 32 );
    }
    /* reset A */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                  LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                  l_gp_reg_mapping.gp_reg_a,
                                                  l_gp_reg_mapping.gp_reg_help_0,
                                                  l_gp_reg_mapping.gp_reg_a,
                                                  (long long) l_m_scheme[0] * 32 * l_micro_kernel_config.datatype_size_in );

    /* advance B */
    if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0 ){
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    l_gp_reg_mapping.gp_reg_b,
                                                    l_gp_reg_mapping.gp_reg_help_0,
                                                    l_gp_reg_mapping.gp_reg_b,
                                                    (long long)  l_xgemm_desc_opa->ldb * 64 * l_micro_kernel_config.datatype_size_in );
    } else {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    l_gp_reg_mapping.gp_reg_b,
                                                    l_gp_reg_mapping.gp_reg_help_0,
                                                    l_gp_reg_mapping.gp_reg_b,
                                                    (long long) 64 * l_micro_kernel_config.datatype_size_in );
    }
    /* advance C */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                  LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                  l_gp_reg_mapping.gp_reg_c,
                                                  l_gp_reg_mapping.gp_reg_help_0,
                                                  l_gp_reg_mapping.gp_reg_c,
                                                  (long long)  (l_xgemm_desc_opa->ldc * 64 * l_micro_kernel_config.datatype_size_out) - (l_m_scheme[0] * 32 * l_micro_kernel_config.datatype_size_out) );

    /* close N loop */
    libxsmm_generator_loop_footer_aarch64( io_generated_code,
                                            &l_loop_label_tracker,
                                            l_gp_reg_mapping.gp_reg_nloop,
                                            64 );
  }

  if( l_n_rest > 0 && l_n_rest < 17){
    if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0 ){
        /* save address of stackpointer */
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       LIBXSMM_AARCH64_GP_REG_XSP,
                                                       l_gp_reg_mapping.gp_reg_help_1,
                                                       l_gp_reg_mapping.gp_reg_help_1,
                                                       0);

        /* allocate memory on stack for transposed B */
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                       LIBXSMM_AARCH64_GP_REG_XSP,
                                                       l_gp_reg_mapping.gp_reg_help_1,
                                                       LIBXSMM_AARCH64_GP_REG_XSP,
                                                       (long long) 16 * l_xgemm_desc_opa->k * l_micro_kernel_config.datatype_size_in );
        /* store address of stack pointer to transposed B register ( x3 )*/
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       LIBXSMM_AARCH64_GP_REG_XSP,
                                                       l_gp_reg_mapping.gp_reg_help_1,
                                                       l_gp_reg_mapping.gp_reg_reduce_count,
                                                      0);
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       l_gp_reg_mapping.gp_reg_reduce_count,
                                                       l_gp_reg_mapping.gp_reg_help_1,
                                                       LIBXSMM_AARCH64_GP_REG_XSP,
                                                       0);

        l_trans_loop = l_xgemm_desc_opa->k / 32;
        l_trans_rest = l_xgemm_desc_opa->k % 32;

        if( l_trans_loop > 0 ){
          libxsmm_generator_loop_header_aarch64(io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_help_2, l_trans_loop);

          libxsmm_generator_transpose_sme( io_generated_code,
                                           l_gp_reg_mapping.gp_reg_b,
                                           l_xgemm_desc_opa->ldb,
                                           32,
                                           l_n_rest,
                                           l_gp_reg_mapping.gp_reg_help_0 );
          /* advance pointer B */
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                         l_gp_reg_mapping.gp_reg_b,
                                                         l_gp_reg_mapping.gp_reg_help_1,
                                                         l_gp_reg_mapping.gp_reg_b,
                                                         32*l_micro_kernel_config.datatype_size_in );
          libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_help_2, 1 );
        }
        if(l_trans_rest > 0 ){
          libxsmm_generator_transpose_sme( io_generated_code,
                                            l_gp_reg_mapping.gp_reg_b,
                                            l_xgemm_desc_opa->ldb,
                                            l_trans_rest,
                                            l_n_rest,
                                            l_gp_reg_mapping.gp_reg_help_0 );
          /* advance pointer B */
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                         l_gp_reg_mapping.gp_reg_b,
                                                         l_gp_reg_mapping.gp_reg_help_1,
                                                         l_gp_reg_mapping.gp_reg_b,
                                                         l_trans_rest*l_micro_kernel_config.datatype_size_in );
        }
        /* reset address of stackpointer */
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       l_gp_reg_mapping.gp_reg_help_1,
                                                       l_gp_reg_mapping.gp_reg_help_1,
                                                       LIBXSMM_AARCH64_GP_REG_XSP,
                                                       0);
        /* reset b pointer */
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                       l_gp_reg_mapping.gp_reg_b,
                                                       l_gp_reg_mapping.gp_reg_help_1,
                                                       l_gp_reg_mapping.gp_reg_b ,
                                                       (long long) l_xgemm_desc_opa->k * l_micro_kernel_config.datatype_size_in );
    }

    l_perfect_blocking_m = 64;
    l_perfect_m_count = l_xgemm_desc_opa->m / 64;
    l_rest_m = l_xgemm_desc_opa->m % 64;
    for( l_m_2 = 0; l_m_2 < 2; l_m_2++ ) {
      if( (l_rest_m == 0 && l_m_2 == 1) || (l_perfect_m_count == 0 && l_m_2 == 0)){
        continue;
      }
      l_m_blocking = (l_m_2 == 0) ? l_perfect_blocking_m : l_rest_m;
      /* open M loop */
      libxsmm_generator_loop_header_aarch64( io_generated_code, &l_loop_label_tracker,
                                             l_gp_reg_mapping.gp_reg_mloop,
                                             (l_m_2 == 0) ? (l_perfect_blocking_m * l_perfect_m_count) : l_rest_m );

      /* save pointer of x2 */
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                           LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                           LIBXSMM_AARCH64_GP_REG_XZR,
                                                           l_gp_reg_mapping.gp_reg_c,
                                                           l_gp_reg_mapping.gp_reg_help_3,
                                                           0,
                                                           LIBXSMM_AARCH64_SHIFTMODE_LSL );

      /* set x9 register */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    l_gp_reg_mapping.gp_reg_c,
                                                    LIBXSMM_AARCH64_GP_REG_X11,
                                                    l_gp_reg_mapping.gp_reg_help_0,
                                                    (long long) l_xgemm_desc_opa->ldc * 16 * l_micro_kernel_config.datatype_size_out  );


      /* load block of C */
      libxsmm_generator_load_64x16_aarch64_sme( io_generated_code,
                                                l_gp_reg_mapping.gp_reg_c,
                                                l_m_blocking,
                                                l_n_rest,
                                                l_xgemm_desc_opa->ldc );
      /* compute outer product */
      libxsmm_generator_gemm_aarch64_kloop_sme_het( io_generated_code,
                                                    &l_loop_label_tracker,
                                                    &l_gp_reg_mapping,
                                                    &l_micro_kernel_config,
                                                    l_xgemm_desc_opa,
                                                    l_m_blocking,
                                                    l_n_rest,
                                                    1,
                                                    16 );

      /* restore pointer of x2 */
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                           LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                           LIBXSMM_AARCH64_GP_REG_XZR,
                                                           l_gp_reg_mapping.gp_reg_help_3,
                                                           l_gp_reg_mapping.gp_reg_c,
                                                           0,
                                                           LIBXSMM_AARCH64_SHIFTMODE_LSL );
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    l_gp_reg_mapping.gp_reg_c,
                                                    LIBXSMM_AARCH64_GP_REG_X11,
                                                    l_gp_reg_mapping.gp_reg_help_0,
                                                    (long long) l_xgemm_desc_opa->ldc * 16 * l_micro_kernel_config.datatype_size_out  );

      /* store block of C */
      libxsmm_generated_store_64x16_aarch64_sme( io_generated_code,
                                                  l_gp_reg_mapping.gp_reg_c,
                                                  l_m_blocking,
                                                  l_n_rest,
                                                  l_xgemm_desc_opa->ldc );

      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                           LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                           LIBXSMM_AARCH64_GP_REG_XZR,
                                                           l_gp_reg_mapping.gp_reg_help_3,
                                                           l_gp_reg_mapping.gp_reg_c,
                                                           0,
                                                           LIBXSMM_AARCH64_SHIFTMODE_LSL );
      /* advance C pointer */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     l_gp_reg_mapping.gp_reg_c,
                                                     l_gp_reg_mapping.gp_reg_help_2,
                                                     l_gp_reg_mapping.gp_reg_c,
                                                     (long long)l_m_blocking*l_micro_kernel_config.datatype_size_out );

      /* advance A pointer */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     l_gp_reg_mapping.gp_reg_a,
                                                     l_gp_reg_mapping.gp_reg_help_0,
                                                     l_gp_reg_mapping.gp_reg_a,
                                                     (long long)l_m_blocking*l_micro_kernel_config.datatype_size_out );

      /* close M loop */
      libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker,
                                             l_gp_reg_mapping.gp_reg_mloop, l_m_blocking );
    }
  } else if( l_n_rest > 16 && l_n_rest < 33){

      if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0 ){
        /* save address of stackpointer */
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       LIBXSMM_AARCH64_GP_REG_XSP,
                                                       l_gp_reg_mapping.gp_reg_help_1,
                                                       l_gp_reg_mapping.gp_reg_help_1,
                                                      0);

        /* allocate memory on stack for transposed B */
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                       LIBXSMM_AARCH64_GP_REG_XSP,
                                                       l_gp_reg_mapping.gp_reg_help_1,
                                                       LIBXSMM_AARCH64_GP_REG_XSP,
                                                       32 * l_xgemm_desc_opa->k * l_micro_kernel_config.datatype_size_out);
        /* store address of stack pointer to transposed B register ( x3 )*/
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       LIBXSMM_AARCH64_GP_REG_XSP,
                                                       l_gp_reg_mapping.gp_reg_help_1,
                                                       l_gp_reg_mapping.gp_reg_reduce_count,
                                                       0);
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       l_gp_reg_mapping.gp_reg_reduce_count,
                                                       l_gp_reg_mapping.gp_reg_help_1,
                                                       LIBXSMM_AARCH64_GP_REG_XSP,
                                                       0);

        l_trans_loop = l_xgemm_desc_opa->k / 16 ;
        l_trans_rest = l_xgemm_desc_opa->k % 16 ;

        if( l_trans_loop > 0 ){
        libxsmm_generator_loop_header_aarch64(io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_help_2, l_trans_loop);
          libxsmm_generator_transpose_sme( io_generated_code,
                                           l_gp_reg_mapping.gp_reg_b,
                                           l_xgemm_desc_opa->ldb,
                                           16,
                                           l_n_rest,
                                           l_gp_reg_mapping.gp_reg_help_0 );
          /* advance pointer B */
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                         l_gp_reg_mapping.gp_reg_b,
                                                         l_gp_reg_mapping.gp_reg_help_1,
                                                         l_gp_reg_mapping.gp_reg_b ,
                                                         16*l_micro_kernel_config.datatype_size_in );
        libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_help_2, 1 );
        }
        if(l_trans_rest > 0 ){
          libxsmm_generator_transpose_sme( io_generated_code,
                                            l_gp_reg_mapping.gp_reg_b,
                                            l_xgemm_desc_opa->ldb,
                                            l_trans_rest,
                                            l_n_rest,
                                            l_gp_reg_mapping.gp_reg_help_0 );
          /* advance pointer B */
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                         l_gp_reg_mapping.gp_reg_b,
                                                         l_gp_reg_mapping.gp_reg_help_1,
                                                         l_gp_reg_mapping.gp_reg_b,
                                                         l_trans_rest*l_micro_kernel_config.datatype_size_in);
        }
        /* reset address of stackpointer */
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       l_gp_reg_mapping.gp_reg_help_1,
                                                       l_gp_reg_mapping.gp_reg_help_1,
                                                       LIBXSMM_AARCH64_GP_REG_XSP,
                                                       0);
        /* reset b pointer */
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                          l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_b ,
                                                          l_xgemm_desc_opa->k * l_micro_kernel_config.datatype_size_in );
      }
    l_perfect_blocking_m = 32;
    l_perfect_m_count = l_xgemm_desc_opa->m / 32;
    l_rest_m = l_xgemm_desc_opa->m % 32;
     /* apply m_blocking */
    for( l_m_2 = 0; l_m_2 < 2; l_m_2++ ) {
      if( (l_rest_m == 0 && l_m_2 == 1) || (l_perfect_m_count == 0 && l_m_2 == 0)){
        continue;
      }
      l_m_blocking = (l_m_2 == 0) ? l_perfect_blocking_m : l_rest_m;
      /* open M loop */
      libxsmm_generator_loop_header_aarch64( io_generated_code,
                                             &l_loop_label_tracker,
                                             l_gp_reg_mapping.gp_reg_mloop,
                                             (l_m_2 == 0) ? (l_perfect_blocking_m * l_perfect_m_count) : l_rest_m );

      /* save pointer of x2 */
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                           LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                           LIBXSMM_AARCH64_GP_REG_XZR,
                                                           l_gp_reg_mapping.gp_reg_c,
                                                           l_gp_reg_mapping.gp_reg_help_3,
                                                           0,
                                                           LIBXSMM_AARCH64_SHIFTMODE_LSL );

      /* set x9 register */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    l_gp_reg_mapping.gp_reg_c,
                                                    LIBXSMM_AARCH64_GP_REG_X11,
                                                    l_gp_reg_mapping.gp_reg_help_0,
                                                    (long long) l_xgemm_desc_opa->ldc * 16*l_micro_kernel_config.datatype_size_out  );


      /* load block of C */
      libxsmm_generator_load_32x32_aarch64_sme( io_generated_code,
                                                    l_gp_reg_mapping.gp_reg_c,
                                                    l_gp_reg_mapping.gp_reg_help_0,
                                                    l_m_blocking,
                                                    l_n_rest,
                                                    l_xgemm_desc_opa->ldc );


      /* compute outer product */
      libxsmm_generator_gemm_aarch64_kloop_sme_het( io_generated_code,
                                                    &l_loop_label_tracker,
                                                    &l_gp_reg_mapping,
                                                    &l_micro_kernel_config,
                                                    l_xgemm_desc_opa,
                                                    l_m_blocking,
                                                    l_n_rest,
                                                    0,
                                                    32 );

      /* restore pointer of x2 */
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                           LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                           LIBXSMM_AARCH64_GP_REG_XZR,
                                                           l_gp_reg_mapping.gp_reg_help_3,
                                                           l_gp_reg_mapping.gp_reg_c,
                                                           0,
                                                           LIBXSMM_AARCH64_SHIFTMODE_LSL );
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    l_gp_reg_mapping.gp_reg_c,
                                                    LIBXSMM_AARCH64_GP_REG_X11,
                                                    l_gp_reg_mapping.gp_reg_help_0,
                                                    (long long) l_xgemm_desc_opa->ldc * 16 * l_micro_kernel_config.datatype_size_out );

      /* store block of C */
      libxsmm_generator_store_32x32_aarch64_sme( io_generated_code,
                                                      l_gp_reg_mapping.gp_reg_c,
                                                      l_gp_reg_mapping.gp_reg_help_0,
                                                      l_m_blocking,
                                                      l_n_rest,
                                                      l_xgemm_desc_opa->ldc,
                                                      1 );

      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                           LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                           LIBXSMM_AARCH64_GP_REG_XZR,
                                                           l_gp_reg_mapping.gp_reg_help_3,
                                                           l_gp_reg_mapping.gp_reg_c,
                                                           0,
                                                           LIBXSMM_AARCH64_SHIFTMODE_LSL );
      /* advance C pointer */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     l_gp_reg_mapping.gp_reg_c,
                                                     l_gp_reg_mapping.gp_reg_help_2,
                                                     l_gp_reg_mapping.gp_reg_c,
                                                     (long long)l_m_blocking*l_micro_kernel_config.datatype_size_out );

      /* advance A pointer */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     l_gp_reg_mapping.gp_reg_a,
                                                     l_gp_reg_mapping.gp_reg_help_0,
                                                     l_gp_reg_mapping.gp_reg_a,
                                                     (long long)l_m_blocking*l_micro_kernel_config.datatype_size_in );

      /* close M loop */
      libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker,
                                             l_gp_reg_mapping.gp_reg_mloop, l_m_blocking );
    }
  } else if( l_n_rest > 32 && l_n_rest <= 48 && l_xgemm_desc_opa->m > 16 ){
    if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0 ){
      /* save address of stackpointer */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    LIBXSMM_AARCH64_GP_REG_XSP,
                                                    l_gp_reg_mapping.gp_reg_help_1,
                                                    l_gp_reg_mapping.gp_reg_help_1,
                                                    0);
      /* allocate memory on stack for transposed B */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                    LIBXSMM_AARCH64_GP_REG_XSP,
                                                    l_gp_reg_mapping.gp_reg_help_1,
                                                    LIBXSMM_AARCH64_GP_REG_XSP,
                                                    (long long) 64 * l_micro_kernel_config.datatype_size_in * l_xgemm_desc_opa->k );

      /* store address of stack pointer to transposed B register ( x3 )*/
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    LIBXSMM_AARCH64_GP_REG_XSP,
                                                    l_gp_reg_mapping.gp_reg_help_1,
                                                    l_gp_reg_mapping.gp_reg_reduce_count ,
                                                    0);
      /* store address into x26 */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    LIBXSMM_AARCH64_GP_REG_XSP,
                                                    l_gp_reg_mapping.gp_reg_help_1,
                                                    LIBXSMM_AARCH64_GP_REG_X26,
                                                    0);
      l_trans_loop = l_xgemm_desc_opa->k / 16;
      l_trans_rest = l_xgemm_desc_opa->k % 16;
      if( l_trans_loop > 0 ){
        libxsmm_generator_loop_header_aarch64(io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_help_2, l_trans_loop);

        libxsmm_generator_sme_transpose_64( io_generated_code,
                                            l_gp_reg_mapping.gp_reg_b,
                                            l_xgemm_desc_opa->ldb,
                                            0,
                                            l_n_rest,
                                            l_gp_reg_mapping.gp_reg_help_0 );

        /* advance pointer B */
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                      LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                      l_gp_reg_mapping.gp_reg_b,
                                                      l_gp_reg_mapping.gp_reg_help_0,
                                                      l_gp_reg_mapping.gp_reg_b ,
                                                      16*l_micro_kernel_config.datatype_size_in );
        libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_help_2, 1 );
      }
      if( l_trans_rest > 0 ){
        libxsmm_generator_sme_transpose_64( io_generated_code,
                                            l_gp_reg_mapping.gp_reg_b,
                                            l_xgemm_desc_opa->ldb,
                                            l_trans_rest,
                                            l_n_rest,
                                            l_gp_reg_mapping.gp_reg_help_0 );
      }
      /* reset address of stackpointer */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    l_gp_reg_mapping.gp_reg_help_1,
                                                    l_gp_reg_mapping.gp_reg_help_1,
                                                    LIBXSMM_AARCH64_GP_REG_XSP,
                                                    0);
      /* reset b pointer */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                    l_gp_reg_mapping.gp_reg_b,
                                                    l_gp_reg_mapping.gp_reg_help_0,
                                                    l_gp_reg_mapping.gp_reg_b ,
                                                    (l_xgemm_desc_opa->k - l_trans_rest) * l_micro_kernel_config.datatype_size_in );
    }

    l_perfect_blocking_m = 32;
    l_perfect_m_count = l_xgemm_desc_opa->m / 32;
    l_rest_m = l_xgemm_desc_opa->m % 32;
     /* apply m_blocking */
    for( l_m_2 = 0; l_m_2 < 2; l_m_2++ ) {
      if( (l_rest_m == 0 && l_m_2 == 1) || (l_perfect_m_count == 0 && l_m_2 == 0) || (l_m_2 == 1 && l_xgemm_desc_opa->m % 32 <= 16 )){
        continue;
      }
      l_m_blocking = (l_m_2 == 0) ? l_perfect_blocking_m : l_rest_m;
      /* open M loop */
      libxsmm_generator_loop_header_aarch64( io_generated_code,
                                             &l_loop_label_tracker,
                                             l_gp_reg_mapping.gp_reg_mloop,
                                             (l_m_2 == 0) ? (l_perfect_blocking_m * l_perfect_m_count) : l_rest_m );

      /* save pointer of x2 */
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                           LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                           LIBXSMM_AARCH64_GP_REG_XZR,
                                                           l_gp_reg_mapping.gp_reg_c,
                                                           l_gp_reg_mapping.gp_reg_help_3,
                                                           0,
                                                           LIBXSMM_AARCH64_SHIFTMODE_LSL );

      /* set x9 register */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    l_gp_reg_mapping.gp_reg_c,
                                                    LIBXSMM_AARCH64_GP_REG_X11,
                                                    l_gp_reg_mapping.gp_reg_help_0,
                                                    (long long) l_xgemm_desc_opa->ldc * 16*l_micro_kernel_config.datatype_size_out  );


      /* load block of C */
      libxsmm_generator_load_32x32_aarch64_sme( io_generated_code,
                                                    l_gp_reg_mapping.gp_reg_c,
                                                    l_gp_reg_mapping.gp_reg_help_0,
                                                    l_m_blocking,
                                                    l_n_rest,
                                                    l_xgemm_desc_opa->ldc );


      /* compute outer product */
      libxsmm_generator_gemm_aarch64_kloop_sme_het( io_generated_code,
                                                    &l_loop_label_tracker,
                                                    &l_gp_reg_mapping,
                                                    &l_micro_kernel_config,
                                                    l_xgemm_desc_opa,
                                                    l_m_blocking,
                                                    l_n_rest,
                                                    0,
                                                    64 );

      /* restore pointer of x2 */
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                           LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                           LIBXSMM_AARCH64_GP_REG_XZR,
                                                           l_gp_reg_mapping.gp_reg_help_3,
                                                           l_gp_reg_mapping.gp_reg_c,
                                                           0,
                                                           LIBXSMM_AARCH64_SHIFTMODE_LSL );
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    l_gp_reg_mapping.gp_reg_c,
                                                    LIBXSMM_AARCH64_GP_REG_X11,
                                                    l_gp_reg_mapping.gp_reg_help_0,
                                                    (long long) l_xgemm_desc_opa->ldc * 16 * l_micro_kernel_config.datatype_size_out );

      /* store block of C */
      libxsmm_generator_store_32x32_aarch64_sme( io_generated_code,
                                                      l_gp_reg_mapping.gp_reg_c,
                                                      l_gp_reg_mapping.gp_reg_help_0,
                                                      l_m_blocking,
                                                      l_n_rest,
                                                      l_xgemm_desc_opa->ldc,
                                                      1 );

      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                           LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                           LIBXSMM_AARCH64_GP_REG_XZR,
                                                           l_gp_reg_mapping.gp_reg_help_3,
                                                           l_gp_reg_mapping.gp_reg_c,
                                                           0,
                                                           LIBXSMM_AARCH64_SHIFTMODE_LSL );
      /* advance C pointer */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     l_gp_reg_mapping.gp_reg_c,
                                                     l_gp_reg_mapping.gp_reg_help_2,
                                                     l_gp_reg_mapping.gp_reg_c,
                                                     (long long)l_m_blocking*l_micro_kernel_config.datatype_size_out );

      /* advance A pointer */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     l_gp_reg_mapping.gp_reg_a,
                                                     l_gp_reg_mapping.gp_reg_help_0,
                                                     l_gp_reg_mapping.gp_reg_a,
                                                     (long long)l_m_blocking*l_micro_kernel_config.datatype_size_in );

      /* close M loop */
      libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker,
                                             l_gp_reg_mapping.gp_reg_mloop, l_m_blocking );
    }
    if( l_xgemm_desc_opa->m % 32 > 0 && l_xgemm_desc_opa->m % 32 <= 16 ){
      /* save pointer of x2 */
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                           LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                           LIBXSMM_AARCH64_GP_REG_XZR,
                                                           l_gp_reg_mapping.gp_reg_c,
                                                           l_gp_reg_mapping.gp_reg_help_3,
                                                           0,
                                                           LIBXSMM_AARCH64_SHIFTMODE_LSL );

      /* load block of C */
      libxsmm_generated_load_16x64_aarch64_sme( io_generated_code,
                                                l_gp_reg_mapping.gp_reg_c,
                                                l_xgemm_desc_opa->m % 32,
                                                l_n_rest,
                                                l_xgemm_desc_opa->ldc );

      /* compute outer product */
      libxsmm_generator_gemm_aarch64_kloop_sme_het( io_generated_code,
                                                    &l_loop_label_tracker,
                                                    &l_gp_reg_mapping,
                                                    &l_micro_kernel_config,
                                                    l_xgemm_desc_opa,
                                                    l_xgemm_desc_opa->m % 32,
                                                    l_n_rest,
                                                    2,
                                                    64 );

      /* restore pointer of x2 */
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                           LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                           LIBXSMM_AARCH64_GP_REG_XZR,
                                                           l_gp_reg_mapping.gp_reg_help_3,
                                                           l_gp_reg_mapping.gp_reg_c,
                                                           0,
                                                           LIBXSMM_AARCH64_SHIFTMODE_LSL );

      /* store block of C */
      libxsmm_generated_store_16x64_aarch64_sme( io_generated_code,
                                                 l_gp_reg_mapping.gp_reg_c,
                                                 l_xgemm_desc_opa->m % 32,
                                                 l_n_rest,
                                                 l_xgemm_desc_opa->ldc );

      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                           LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                           LIBXSMM_AARCH64_GP_REG_XZR,
                                                           l_gp_reg_mapping.gp_reg_help_3,
                                                           l_gp_reg_mapping.gp_reg_c,
                                                           0,
                                                           LIBXSMM_AARCH64_SHIFTMODE_LSL );
    }
    /* restore A pointer */
    if(  l_xgemm_desc_opa->m % 32 > 0 && l_xgemm_desc_opa->m % 32 <= 16 ){
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   l_gp_reg_mapping.gp_reg_a,
                                                   l_gp_reg_mapping.gp_reg_help_0,
                                                   l_gp_reg_mapping.gp_reg_a,
                                                   (l_perfect_m_count > 0 ) ? (long long)l_perfect_blocking_m*l_perfect_m_count*l_micro_kernel_config.datatype_size_in  : (long long) l_xgemm_desc_opa->m*l_micro_kernel_config.datatype_size_in );
    } else {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                    l_gp_reg_mapping.gp_reg_a,
                                                    l_gp_reg_mapping.gp_reg_help_0,
                                                    l_gp_reg_mapping.gp_reg_a,
                                                    (long long)l_xgemm_desc_opa->m * l_micro_kernel_config.datatype_size_in );
    }
    /* advance B pointer */
      if( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ){
        /* reset b pointer */
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                      LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                      l_gp_reg_mapping.gp_reg_b,
                                                      l_gp_reg_mapping.gp_reg_help_1,
                                                      l_gp_reg_mapping.gp_reg_b,
                                                      (long long) 32 * l_micro_kernel_config.datatype_size_in );
      } else {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                      LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                      l_gp_reg_mapping.gp_reg_reduce_count,
                                                      l_gp_reg_mapping.gp_reg_help_1,
                                                      l_gp_reg_mapping.gp_reg_reduce_count,
                                                      (long long)32 * l_micro_kernel_config.datatype_size_in );
      }
    /* advance C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   l_gp_reg_mapping.gp_reg_c,
                                                   l_gp_reg_mapping.gp_reg_help_2,
                                                   l_gp_reg_mapping.gp_reg_c,
                                                   (long long) (32 * l_micro_kernel_config.datatype_size_out* l_xgemm_desc_opa->ldc));
    if(  l_xgemm_desc_opa->m % 32 > 0 && l_xgemm_desc_opa->m % 32 <= 16 ){
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   l_gp_reg_mapping.gp_reg_c,
                                                   l_gp_reg_mapping.gp_reg_help_2,
                                                   l_gp_reg_mapping.gp_reg_c,
                                                   (long long) l_perfect_m_count*32*l_micro_kernel_config.datatype_size_out );
    } else {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   l_gp_reg_mapping.gp_reg_c,
                                                   l_gp_reg_mapping.gp_reg_help_2,
                                                   l_gp_reg_mapping.gp_reg_c,
                                                   (long long) l_xgemm_desc_opa->m * l_micro_kernel_config.datatype_size_out );
    }
    l_perfect_blocking_m = 64;
    l_perfect_m_count = l_xgemm_desc_opa->m / 64;
    l_rest_m = l_xgemm_desc_opa->m % 64;
    if(l_xgemm_desc_opa->m % 32 <= 16 && l_xgemm_desc_opa->m % 32 != 0 ){
      if( l_xgemm_desc_opa->m % 64 > 32 ){
        l_rest_m = 32;
      } else {
        l_rest_m = 0;
      }
    }
    for( l_m_2 = 0; l_m_2 < 2; l_m_2++ ) {
      if( (l_rest_m == 0 && l_m_2 == 1) || (l_perfect_m_count == 0 && l_m_2 == 0)){
        continue;
      }
      l_m_blocking = (l_m_2 == 0) ? l_perfect_blocking_m : l_rest_m;
      /* open M loop */
      libxsmm_generator_loop_header_aarch64( io_generated_code, &l_loop_label_tracker,
                                             l_gp_reg_mapping.gp_reg_mloop,
                                             (l_m_2 == 0) ? (l_perfect_blocking_m * l_perfect_m_count) : l_rest_m );

      /* save pointer of x2 */
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                           LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                           LIBXSMM_AARCH64_GP_REG_XZR,
                                                           l_gp_reg_mapping.gp_reg_c,
                                                           l_gp_reg_mapping.gp_reg_help_3,
                                                           0,
                                                           LIBXSMM_AARCH64_SHIFTMODE_LSL );

      /* set x9 register */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    l_gp_reg_mapping.gp_reg_c,
                                                    LIBXSMM_AARCH64_GP_REG_X11,
                                                    l_gp_reg_mapping.gp_reg_help_0,
                                                    (long long) l_xgemm_desc_opa->ldc * 16 * l_micro_kernel_config.datatype_size_out  );


      /* load block of C */
      libxsmm_generator_load_64x16_aarch64_sme( io_generated_code,
                                                l_gp_reg_mapping.gp_reg_c,
                                                l_m_blocking,
                                                l_n_rest-32,
                                                l_xgemm_desc_opa->ldc );
      /* compute outer product */
      libxsmm_generator_gemm_aarch64_kloop_sme_het( io_generated_code,
                                                    &l_loop_label_tracker,
                                                    &l_gp_reg_mapping,
                                                    &l_micro_kernel_config,
                                                    l_xgemm_desc_opa,
                                                    l_m_blocking,
                                                    l_n_rest-32,
                                                    1,
                                                    64 );

      /* restore pointer of x2 */
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                           LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                           LIBXSMM_AARCH64_GP_REG_XZR,
                                                           l_gp_reg_mapping.gp_reg_help_3,
                                                           l_gp_reg_mapping.gp_reg_c,
                                                           0,
                                                           LIBXSMM_AARCH64_SHIFTMODE_LSL );
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    l_gp_reg_mapping.gp_reg_c,
                                                    LIBXSMM_AARCH64_GP_REG_X11,
                                                    l_gp_reg_mapping.gp_reg_help_0,
                                                    (long long) l_xgemm_desc_opa->ldc * 16 * l_micro_kernel_config.datatype_size_out  );

      /* store block of C */
      libxsmm_generated_store_64x16_aarch64_sme( io_generated_code,
                                                  l_gp_reg_mapping.gp_reg_c,
                                                  l_m_blocking,
                                                  l_n_rest-32,
                                                  l_xgemm_desc_opa->ldc );

      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                           LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                           LIBXSMM_AARCH64_GP_REG_XZR,
                                                           l_gp_reg_mapping.gp_reg_help_3,
                                                           l_gp_reg_mapping.gp_reg_c,
                                                           0,
                                                           LIBXSMM_AARCH64_SHIFTMODE_LSL );
      /* advance C pointer */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     l_gp_reg_mapping.gp_reg_c,
                                                     l_gp_reg_mapping.gp_reg_help_2,
                                                     l_gp_reg_mapping.gp_reg_c,
                                                     (long long)l_m_blocking*l_micro_kernel_config.datatype_size_out );

      /* advance A pointer */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     l_gp_reg_mapping.gp_reg_a,
                                                     l_gp_reg_mapping.gp_reg_help_0,
                                                     l_gp_reg_mapping.gp_reg_a,
                                                     (long long)l_m_blocking*l_micro_kernel_config.datatype_size_out );

      /* close M loop */
      libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker,
                                             l_gp_reg_mapping.gp_reg_mloop, l_m_blocking );
    }

  } else if( l_n_rest > 32 ){

    if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0 ){
      /* save address of stackpointer */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    LIBXSMM_AARCH64_GP_REG_XSP,
                                                    l_gp_reg_mapping.gp_reg_help_1,
                                                    l_gp_reg_mapping.gp_reg_help_1,
                                                    0);
      /* allocate memory on stack for transposed B */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                    LIBXSMM_AARCH64_GP_REG_XSP,
                                                    l_gp_reg_mapping.gp_reg_help_1,
                                                    LIBXSMM_AARCH64_GP_REG_XSP,
                                                    (long long) 64 * l_micro_kernel_config.datatype_size_in * l_xgemm_desc_opa->k );

      /* store address of stack pointer to transposed B register ( x3 )*/
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    LIBXSMM_AARCH64_GP_REG_XSP,
                                                    l_gp_reg_mapping.gp_reg_help_1,
                                                    l_gp_reg_mapping.gp_reg_reduce_count ,
                                                    0);
      /* store address into x26 */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    LIBXSMM_AARCH64_GP_REG_XSP,
                                                    l_gp_reg_mapping.gp_reg_help_1,
                                                    LIBXSMM_AARCH64_GP_REG_X26,
                                                    0);

      l_trans_loop = l_xgemm_desc_opa->k / 16;
      l_trans_rest = l_xgemm_desc_opa->k % 16;
      if( l_trans_loop > 0 ){
        libxsmm_generator_loop_header_aarch64(io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_help_2, l_trans_loop);

        libxsmm_generator_sme_transpose_64( io_generated_code,
                                            l_gp_reg_mapping.gp_reg_b,
                                            l_xgemm_desc_opa->ldb,
                                            0,
                                            l_n_rest,
                                            l_gp_reg_mapping.gp_reg_help_0 );

        /* advance pointer B */
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                      LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                      l_gp_reg_mapping.gp_reg_b,
                                                      l_gp_reg_mapping.gp_reg_help_0,
                                                      l_gp_reg_mapping.gp_reg_b ,
                                                      16*l_micro_kernel_config.datatype_size_in );
        libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_help_2, 1 );
      }
      if( l_trans_rest > 0 ){
        libxsmm_generator_sme_transpose_64( io_generated_code,
                                            l_gp_reg_mapping.gp_reg_b,
                                            l_xgemm_desc_opa->ldb,
                                            l_trans_rest,
                                            l_n_rest,
                                            l_gp_reg_mapping.gp_reg_help_0 );
      }
      /* reset address of stackpointer */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    l_gp_reg_mapping.gp_reg_help_1,
                                                    l_gp_reg_mapping.gp_reg_help_1,
                                                    LIBXSMM_AARCH64_GP_REG_XSP,
                                                    0);
      /* reset b pointer */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                    l_gp_reg_mapping.gp_reg_b,
                                                    l_gp_reg_mapping.gp_reg_help_0,
                                                    l_gp_reg_mapping.gp_reg_b ,
                                                    (l_xgemm_desc_opa->k - l_trans_rest) * l_micro_kernel_config.datatype_size_in );
    }
    l_perfect_blocking_m = 16;
    l_perfect_m_count = l_xgemm_desc_opa->m / 16;
    l_rest_m = l_xgemm_desc_opa->m % 16;
     /* apply m_blocking */
    for( l_m_2 = 0; l_m_2 < 2; l_m_2++ ) {
      if( (l_rest_m == 0 && l_m_2 == 1) || (l_perfect_m_count == 0 && l_m_2 == 0)){
        continue;
      }
      l_m_blocking = (l_m_2 == 0) ? l_perfect_blocking_m : l_rest_m;
      /* open M loop */
      libxsmm_generator_loop_header_aarch64( io_generated_code, &l_loop_label_tracker,
                                             l_gp_reg_mapping.gp_reg_mloop, (l_m_2 == 0) ? (l_perfect_blocking_m * l_perfect_m_count) : l_rest_m );

      /* save pointer of x2 */
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                           LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                           LIBXSMM_AARCH64_GP_REG_XZR,
                                                           l_gp_reg_mapping.gp_reg_c,
                                                           l_gp_reg_mapping.gp_reg_help_3,
                                                           0,
                                                           LIBXSMM_AARCH64_SHIFTMODE_LSL );

      /* load block of C */
      libxsmm_generated_load_16x64_aarch64_sme( io_generated_code,
                                                l_gp_reg_mapping.gp_reg_c,
                                                l_m_blocking,
                                                l_n_rest,
                                                l_xgemm_desc_opa->ldc );

      /* compute outer product */
      libxsmm_generator_gemm_aarch64_kloop_sme_het( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config,
                                                    l_xgemm_desc_opa, l_m_blocking, l_n_rest, 2, 64 );

      /* restore pointer of x2 */
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                           LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                           LIBXSMM_AARCH64_GP_REG_XZR,
                                                           l_gp_reg_mapping.gp_reg_help_3,
                                                           l_gp_reg_mapping.gp_reg_c,
                                                           0,
                                                           LIBXSMM_AARCH64_SHIFTMODE_LSL );

      /* store block of C */
      libxsmm_generated_store_16x64_aarch64_sme( io_generated_code,
                                                 l_gp_reg_mapping.gp_reg_c,
                                                 l_m_blocking,
                                                 l_n_rest,
                                                 l_xgemm_desc_opa->ldc );

      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                           LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                           LIBXSMM_AARCH64_GP_REG_XZR,
                                                           l_gp_reg_mapping.gp_reg_help_3,
                                                           l_gp_reg_mapping.gp_reg_c,
                                                           0,
                                                           LIBXSMM_AARCH64_SHIFTMODE_LSL );
      /* advance C pointer */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_c,
                                                     (long long)l_m_blocking*l_micro_kernel_config.datatype_size_in );

      /* advance A pointer */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a,
                                                     (long long)l_m_blocking*l_micro_kernel_config.datatype_size_in );

      /* close M loop */
      libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker,
                                             l_gp_reg_mapping.gp_reg_mloop, l_m_blocking );

    }
  }

  libxsmm_aarch64_instruction_sm( io_generated_code,
                                  LIBXSMM_AARCH64_INSTR_SME_SMSTOP);

  /* at the end this is called*/
  libxsmm_generator_gemm_destroy_stack_frame_aarch64( io_generated_code );

  /* close asm */
  libxsmm_aarch64_instruction_close_stream( io_generated_code, 0xe0f );

}
