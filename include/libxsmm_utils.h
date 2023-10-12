/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_UTILS_H
#define LIBXSMM_UTILS_H

/**
 * Any intrinsics interface (libxsmm_intrinsics_x86.h) shall be explicitly
 * included, i.e., separate from libxsmm_utils.h.
*/
#include "utils/libxsmm_lpflt_quant.h"
#include "utils/libxsmm_barrier.h"
#include "utils/libxsmm_timer.h"
#include "utils/libxsmm_math.h"
#include "utils/libxsmm_mhd.h"

#if defined(__BLAS) && (1 == __BLAS)
# if defined(__OPENBLAS)
    LIBXSMM_EXTERN void openblas_set_num_threads(int num_threads);
#   define LIBXSMM_BLAS_INIT openblas_set_num_threads(1);
# endif
#endif
#if !defined(LIBXSMM_BLAS_INIT)
# define LIBXSMM_BLAS_INIT
#endif

/** Call libxsmm_gemm_print using LIBXSMM's GEMM-flags. */
#define LIBXSMM_GEMM_PRINT(OSTREAM, PRECISION, FLAGS, M, N, K, DALPHA, A, LDA, B, LDB, DBETA, C, LDC) \
  LIBXSMM_GEMM_PRINT2(OSTREAM, PRECISION, PRECISION, FLAGS, M, N, K, DALPHA, A, LDA, B, LDB, DBETA, C, LDC)
#define LIBXSMM_GEMM_PRINT2(OSTREAM, IPREC, OPREC, FLAGS, M, N, K, DALPHA, A, LDA, B, LDB, DBETA, C, LDC) \
  libxsmm_gemm_dprint2(OSTREAM, (libxsmm_datatype)(IPREC), (libxsmm_datatype)(OPREC), \
    /* Use 'n' (instead of 'N') avoids warning about "no macro replacement within a character constant". */ \
    (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_A & (FLAGS)) ? 'n' : 't'), \
    (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_B & (FLAGS)) ? 'n' : 't'), \
    M, N, K, DALPHA, A, LDA, B, LDB, DBETA, C, LDC)

/**
 * Utility function, which either prints information about the GEMM call
 * or dumps (FILE/ostream=0) all input and output data into MHD files.
 * The Meta Image Format (MHD) is suitable for visual inspection using,
 * e.g., ITK-SNAP or ParaView.
 */
LIBXSMM_API void libxsmm_gemm_print(void* ostream,
  libxsmm_datatype precision, const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
  const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc);
LIBXSMM_API void libxsmm_gemm_print2(void* ostream,
  libxsmm_datatype iprec, libxsmm_datatype oprec, const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
  const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc);

#endif /*LIBXSMM_UTILS_H*/
