/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_packed_gemm_avx_avx2_avx512.h"
#include "generator_packed_gemm_aarch64.h"

LIBXSMM_API void libxsmm_generator_packed_gemm( libxsmm_generated_code*         io_generated_code,
                                                const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                const unsigned int              i_packed_width ) {
  if ( !(
         ((LIBXSMM_GEMM_GETENUM_A_PREC(    i_xgemm_desc->datatype ) == LIBXSMM_DATATYPE_F64)  && (LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ) == LIBXSMM_DATATYPE_F64)  &&
          (LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype ) == LIBXSMM_DATATYPE_F64)  && (LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) == LIBXSMM_DATATYPE_F64)     )  ||
         ((LIBXSMM_GEMM_GETENUM_A_PREC(    i_xgemm_desc->datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ) == LIBXSMM_DATATYPE_F32)  &&
          (LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) == LIBXSMM_DATATYPE_F32)     ) ) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
    return;
  }

  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
    return;
  }

  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_B) == LIBXSMM_GEMM_FLAG_VNNI_B ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
    return;
  }

  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_C) == LIBXSMM_GEMM_FLAG_VNNI_C ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
    return;
  }

  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) == LIBXSMM_GEMM_FLAG_TRANS_A ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA_TRANS );
    return;
  }

  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == LIBXSMM_GEMM_FLAG_TRANS_B ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB_TRANS );
    return;
  }

  if ( i_xgemm_desc->lda < i_xgemm_desc->m ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
    return;
  }

  if ( i_xgemm_desc->ldb < i_xgemm_desc->k ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB );
    return;
  }

  if ( i_xgemm_desc->ldc < i_xgemm_desc->m ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDC );
    return;
  }

  if ( (io_generated_code->arch >= LIBXSMM_X86_AVX) &&
       (io_generated_code->arch <= LIBXSMM_X86_ALLFEAT) ) {
    libxsmm_generator_packed_gemm_avx_avx2_avx512( io_generated_code,
                                                   i_xgemm_desc,
                                                   i_packed_width );
  } else if ( (io_generated_code->arch >= LIBXSMM_AARCH64_V81) &&
              (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT) ) {
    libxsmm_generator_packed_gemm_aarch64( io_generated_code,
                                           i_xgemm_desc,
                                           i_packed_width );
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
    return;
  }
}
