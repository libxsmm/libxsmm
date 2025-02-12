/******************************************************************************
* Copyright (c), 2025 IBM Corporation - All rights reserved.                  *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Will Trojak (IBM Corp.)
******************************************************************************/

#include "generator_gemm_s390x.h"

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_s390x_vxrs_m_loop( libxsmm_generated_code        *io_generated_code,
                                               const libxsmm_gemm_descriptor *i_xgemm_desc,
                                               libxsmm_s390x_reg             *io_reg_tracker,
                                               libxsmm_loop_label_tracker    *io_loop_labels ) {

  /* m loop values */
  unsigned int l_m_iters = i_xgemm_desc->m ;
  unsigned int l_m_loop;

  if ( l_m_iters > 1 ) {
    l_m_loop = libxsmm_s390x_reg_get( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR );
    libxsmm_s390x_instr_gpr_set_value( io_generated_code, l_m_loop, l_m_iters );
    libxsmm_s390x_instr_register_jump_label( io_generated_code, io_loop_labels );
  }

  libxsmm_s390x_instr_nop(io_generated_code);
  libxsmm_s390x_instr_nop(io_generated_code);

  if ( l_m_iters > 1 ) {
    libxsmm_s390x_instr_branch_count_jump_label( io_generated_code, l_m_loop, io_loop_labels );
    libxsmm_s390x_reg_free( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR, l_m_loop );
  }

}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_s390x_vxrs_kernel( libxsmm_generated_code        *io_generated_code,
                                               const libxsmm_gemm_descriptor *i_xgemm_desc,
                                               libxsmm_s390x_reg             *io_reg_tracker ) {
  /* Reset loop labels as this is a new kernel */
  libxsmm_loop_label_tracker l_loop_labels;
  libxsmm_reset_loop_label_tracker( &l_loop_labels );

  /* Start stream based on ABI */
  libxsmm_s390x_instr_open_stack( io_generated_code );

  /* Unpack the args from the LIBXSMM matrix arg struct, place them in arg GPR */
  libxsmm_s390x_instr_unpack_args( io_generated_code, io_reg_tracker );

  /* GPRs holding pointers to A, B, and C */
  /*unsigned int i_a = LIBXSMM_S390X_GPR_ARG0;
  unsigned int i_b = LIBXSMM_S390X_GPR_ARG1;
  unsigned int i_c = LIBXSMM_S390X_GPR_ARG2;*/

  /* n loop values */
  unsigned int l_n_iters = i_xgemm_desc->n ;
  unsigned int l_n_loop;

  if ( l_n_iters > 1 ) {
    l_n_loop = libxsmm_s390x_reg_get( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR );
    libxsmm_s390x_instr_gpr_set_value( io_generated_code, l_n_loop, l_n_iters );
    libxsmm_s390x_instr_register_jump_label( io_generated_code, &l_loop_labels );
  }

  libxsmm_s390x_instr_nop(io_generated_code);
  libxsmm_s390x_instr_nop(io_generated_code);

  if ( l_n_iters > 1 ) {
    libxsmm_s390x_instr_branch_count_jump_label( io_generated_code, l_n_loop, &l_loop_labels );
    libxsmm_s390x_reg_free( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR, l_n_loop );
  }

  /* Collapse and return */
  libxsmm_s390x_instr_collapse_stack( io_generated_code );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_s390x_kernel( libxsmm_generated_code        *io_generated_code,
                                          const libxsmm_gemm_descriptor *i_xgemm_desc ) {

  libxsmm_s390x_reg l_reg_tracker = libxsmm_s390x_reg_init( io_generated_code );

  libxsmm_generator_gemm_s390x_vxrs_kernel( io_generated_code,
                                            i_xgemm_desc,
                                            &l_reg_tracker );

  libxsmm_s390x_reg_destroy( io_generated_code, &l_reg_tracker );
  return;
}
