/******************************************************************************
* Copyright (c) 2024, IBM Corporation                                         *
* - All rights reserved.                                                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Will Trojak (IBM Corp.)
******************************************************************************/

#include "generator_gemm_vsx_microkernel.h"

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_vsx_mk_bytes( libxsmm_datatype const datatype ) {
  unsigned int bytes = 0;

  switch ( LIBXSMM_GEMM_GETENUM_A_PREC( (const unsigned char *)(&datatype) ) ) {
    case LIBXSMM_DATATYPE_F32: {
      bytes = 4;
    } break;
    case LIBXSMM_DATATYPE_F64: {
      bytes = 8;
    } break;
    case LIBXSMM_DATATYPE_F16: {
      bytes = 2;
    } break;
    case LIBXSMM_DATATYPE_BF16: {
      bytes = 2;
    } break;
    default: {
      fprintf(stderr, "libxsmm_generator_vsx_mk_bytes: unsupported datatype\n");
      exit(-1);
    }
  }

  return bytes;
}

LIBXSMM_API_INTERN
void libxsmm_generator_vsx_microkernel( libxsmm_generated_code *io_generated,
                                        libxsmm_gemm_descriptor const *i_xgemm_desc,
                                        unsigned char a_ptr_gpr,
                                        unsigned char b_ptr_gpr,
                                        unsigned char c_ptr_gpr,
                                        unsigned int  m_block,
                                        unsigned int  n_block,
                                        unsigned int  k_block ) {

  /* local registers to use for a pointer */
  /*
  unsigned char a_l_gpr = LIBXSMM_PPCLE_GPR_R1;
  unsigned char b_l_gpr = LIBXSMM_PPCLE_GPR_R2;
  unsigned char c_l_gpr = LIBXSMM_PPCLE_GPR_R3;

  unsigned int stride_a = i_xgemm_desc->lda * l_bytes_per_val;
  unsigned int stride_b = i_xgemm_desc->ldb * l_bytes_per_val;
  unsigned int stride_c = i_xgemm_desc->ldc * l_bytes_per_val;
  */


}
