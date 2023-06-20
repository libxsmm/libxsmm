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
#include "generator_common.h"
#include "generator_packed_spgemm_csr_bsparse.h"
#include "generator_packed_spgemm_csr_asparse.h"
#include "generator_packed_spgemm_csc_bsparse.h"
#include "generator_packed_spgemm_bcsc_bsparse.h"
#include "generator_packed_spgemm_csc_csparse.h"


LIBXSMM_API
void libxsmm_generator_packed_spgemm_csr_kernel( libxsmm_generated_code*        io_generated_code,
                                                 const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                 const unsigned int*            i_row_idx,
                                                 const unsigned int*            i_column_idx,
                                                 const void*                    i_values,
                                                 const unsigned int             i_packed_width ) {
  /* A matrix is sparse */
  if ( (i_xgemm_desc->lda == 0) && (i_xgemm_desc->ldb > 0) && (i_xgemm_desc->ldc > 0) ) {
    /* check LDB */
    if ( i_xgemm_desc->ldb < i_xgemm_desc->n ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB );
      return;
    }
    /* check LDC */
    if ( i_xgemm_desc->ldc < i_xgemm_desc->n ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDC );
      return;
    }
    libxsmm_generator_packed_spgemm_csr_asparse( io_generated_code, i_xgemm_desc, i_row_idx, i_column_idx, i_values, i_packed_width );
  /* B matrix is sparse */
  } else if ( (i_xgemm_desc->lda > 0) && (i_xgemm_desc->ldb == 0) && (i_xgemm_desc->ldc > 0) ) {
    /* check LDA */
    if ( i_xgemm_desc->lda < i_xgemm_desc->k ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
      return;
    }
    /* check LDC */
    /* coverity[copy_paste_error] */
    if ( i_xgemm_desc->ldc < i_xgemm_desc->n ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDC );
      return;
    }
    libxsmm_generator_packed_spgemm_csr_bsparse( io_generated_code, i_xgemm_desc, i_row_idx, i_column_idx, i_values, i_packed_width );
  } else {
    /* something bad happened... */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_SPGEMM_GEN );
    return;
  }
}

LIBXSMM_API
void libxsmm_generator_packed_spgemm_csc_kernel( libxsmm_generated_code*        io_generated_code,
                                                 const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                 const unsigned int*            i_row_idx,
                                                 const unsigned int*            i_column_idx,
                                                 const void*                    i_values,
                                                 const unsigned int             i_packed_width ) {
  /* B matrix is sparse */
  if ( (i_xgemm_desc->lda > 0) && (i_xgemm_desc->ldb == 0) && (i_xgemm_desc->ldc > 0) ) {
    /* check LDA */
    if ( i_xgemm_desc->lda < i_xgemm_desc->k ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
      return;
    }
    /* check LDC */
    if ( i_xgemm_desc->ldc < i_xgemm_desc->n ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDC );
      return;
    }
    libxsmm_generator_packed_spgemm_csc_bsparse( io_generated_code, i_xgemm_desc, i_row_idx, i_column_idx, i_values, i_packed_width );
  /* C matrix is sparse */
  } else if ( (i_xgemm_desc->lda > 0) && (i_xgemm_desc->ldb > 0) && (i_xgemm_desc->ldc == 0) ) {
#if 0
    /* check LDA */
    if ( i_xgemm_desc->lda < i_xgemm_desc->k ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
      return;
    }
#endif
    /* check LDB */
    if ( i_xgemm_desc->ldb < i_xgemm_desc->n ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB );
      return;
    }
    libxsmm_generator_packed_spgemm_csc_csparse( io_generated_code, i_xgemm_desc, i_row_idx, i_column_idx, i_values, i_packed_width );
  } else {
    /* something bad happened... */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_SPGEMM_GEN );
    return;
  }
}

LIBXSMM_API
void libxsmm_generator_packed_spgemm_bcsc_kernel( libxsmm_generated_code*        io_generated_code,
                                                  const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                  const unsigned int             i_packed_width,
                                                  const unsigned int             i_bk,
                                                  const unsigned int             i_bn ) {
  /* B matrix is sparse */
  if ( (i_xgemm_desc->lda > 0) && (i_xgemm_desc->ldc > 0) ) {
    /* check LDA */
    if ( i_xgemm_desc->lda < i_xgemm_desc->k ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
      return;
    }
    /* check LDC */
    if ( i_xgemm_desc->ldc < i_xgemm_desc->n ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDC );
      return;
    }
    libxsmm_generator_packed_spgemm_bcsc_bsparse( io_generated_code, i_xgemm_desc, i_packed_width, i_bk, i_bn );
  } else {
    /* something bad happened... */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_SPGEMM_GEN );
    return;
  }
}

