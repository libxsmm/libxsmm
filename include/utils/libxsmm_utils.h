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
#include "libxsmm_lpflt_quant.h"
#include "libxsmm_barrier.h"
#include "libxsmm_timer.h"
#include "libxsmm_math.h"
#include "libxsmm_mhd.h"

#if defined(__BLAS) && (1 == __BLAS)
# if defined(__OPENBLAS)
    LIBXSMM_EXTERN void openblas_set_num_threads(int num_threads);
#   define LIBXSMM_BLAS_INIT openblas_set_num_threads(1);
# endif
#endif
#if !defined(LIBXSMM_BLAS_INIT)
# define LIBXSMM_BLAS_INIT
#endif

/** Consolidate CBLAS transpose requests into a set of flags. */
#define LIBXSMM_GEMM_CFLAGS(TRANSA, TRANSB) /* check for N/n rather than T/t since C/c is also valid! */ \
   ((CblasNoTrans == (TRANSA) ? LIBXSMM_GEMM_FLAG_NONE : LIBXSMM_GEMM_FLAG_TRANS_A) \
  | (CblasNoTrans == (TRANSB) ? LIBXSMM_GEMM_FLAG_NONE : LIBXSMM_GEMM_FLAG_TRANS_B))

/** Consolidate transpose requests into a set of flags. */
#define LIBXSMM_GEMM_VNNI_FLAGS(TRANSA, TRANSB, VNNIA, VNNIB) /* check for N/n rather than T/t since C/c is also valid! */ \
   ((('n' == (TRANSA) || *"N" == (TRANSA)) ? LIBXSMM_GEMM_FLAG_NONE : LIBXSMM_GEMM_FLAG_TRANS_A) \
  | (('n' == (TRANSB) || *"N" == (TRANSB)) ? LIBXSMM_GEMM_FLAG_NONE : LIBXSMM_GEMM_FLAG_TRANS_B) \
  | (('n' == (VNNIA) || *"N" == (VNNIA)) ? LIBXSMM_GEMM_FLAG_NONE : LIBXSMM_GEMM_FLAG_VNNI_A) \
  | (('n' == (VNNIB) || *"N" == (VNNIB)) ? LIBXSMM_GEMM_FLAG_NONE : LIBXSMM_GEMM_FLAG_VNNI_B))

/** Inlinable GEMM exercising the compiler's code generation (macro template). TODO: only NN is supported and SP/DP matrices. */
#define LIBXSMM_INLINE_XGEMM(ITYPE, OTYPE, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) do { \
  /* Use 'n' (instead of 'N') avoids warning about "no macro replacement within a character constant". */ \
  const char libxsmm_inline_xgemm_transa_ = (char)(NULL != ((void*)(TRANSA)) ? (*(const char*)(TRANSA)) : \
    (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & LIBXSMM_FLAGS) ? 'n' : 't')); \
  const char libxsmm_inline_xgemm_transb_ = (char)(NULL != ((void*)(TRANSB)) ? (*(const char*)(TRANSB)) : \
    (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & LIBXSMM_FLAGS) ? 'n' : 't')); \
  const libxsmm_blasint libxsmm_inline_xgemm_m_ = *(const libxsmm_blasint*)(M); /* must be specified */ \
  const libxsmm_blasint libxsmm_inline_xgemm_k_ = (NULL != ((void*)(K)) ? (*(const libxsmm_blasint*)(K)) : libxsmm_inline_xgemm_m_); \
  const libxsmm_blasint libxsmm_inline_xgemm_n_ = (NULL != ((void*)(N)) ? (*(const libxsmm_blasint*)(N)) : libxsmm_inline_xgemm_k_); \
  const libxsmm_blasint libxsmm_inline_xgemm_lda_ = (NULL != ((void*)(LDA)) ? (*(const libxsmm_blasint*)(LDA)) : \
    (('n' == libxsmm_inline_xgemm_transa_ || *"N" == libxsmm_inline_xgemm_transa_) ? libxsmm_inline_xgemm_m_ : libxsmm_inline_xgemm_k_)); \
  const libxsmm_blasint libxsmm_inline_xgemm_ldb_ = (NULL != ((void*)(LDB)) ? (*(const libxsmm_blasint*)(LDB)) : \
    (('n' == libxsmm_inline_xgemm_transb_ || *"N" == libxsmm_inline_xgemm_transb_) ? libxsmm_inline_xgemm_k_ : libxsmm_inline_xgemm_n_)); \
  const libxsmm_blasint libxsmm_inline_xgemm_ldc_ = (NULL != ((void*)(LDC)) ? (*(const libxsmm_blasint*)(LDC)) : libxsmm_inline_xgemm_m_); \
  const OTYPE libxsmm_inline_xgemm_alpha_ = (NULL != ((void*)(ALPHA)) ? (*(const OTYPE*)(ALPHA)) : ((OTYPE)LIBXSMM_ALPHA)); \
  const OTYPE libxsmm_inline_xgemm_beta_  = (NULL != ((void*)(BETA))  ? (*(const OTYPE*)(BETA))  : ((OTYPE)LIBXSMM_BETA)); \
  libxsmm_blasint libxsmm_inline_xgemm_ni_, libxsmm_inline_xgemm_mi_ = 0, libxsmm_inline_xgemm_ki_; /* loop induction variables */ \
  LIBXSMM_ASSERT('n' == libxsmm_inline_xgemm_transa_ || *"N" == libxsmm_inline_xgemm_transa_); \
  LIBXSMM_ASSERT('n' == libxsmm_inline_xgemm_transb_ || *"N" == libxsmm_inline_xgemm_transb_); \
  LIBXSMM_PRAGMA_SIMD \
  for (libxsmm_inline_xgemm_mi_ = 0; libxsmm_inline_xgemm_mi_ < libxsmm_inline_xgemm_m_; ++libxsmm_inline_xgemm_mi_) { \
    LIBXSMM_PRAGMA_LOOP_COUNT(1, LIBXSMM_CONFIG_MAX_DIM, LIBXSMM_CONFIG_AVG_DIM) \
    for (libxsmm_inline_xgemm_ki_ = 0; libxsmm_inline_xgemm_ki_ < libxsmm_inline_xgemm_k_; ++libxsmm_inline_xgemm_ki_) { \
      LIBXSMM_PRAGMA_UNROLL \
      for (libxsmm_inline_xgemm_ni_ = 0; libxsmm_inline_xgemm_ni_ < libxsmm_inline_xgemm_n_; ++libxsmm_inline_xgemm_ni_) { \
        ((OTYPE*)(C))[libxsmm_inline_xgemm_ni_*libxsmm_inline_xgemm_ldc_+libxsmm_inline_xgemm_mi_] \
          = ((const ITYPE*)(B))[libxsmm_inline_xgemm_ni_*libxsmm_inline_xgemm_ldb_+libxsmm_inline_xgemm_ki_] * \
           (((const ITYPE*)(A))[libxsmm_inline_xgemm_ki_*libxsmm_inline_xgemm_lda_+libxsmm_inline_xgemm_mi_] * libxsmm_inline_xgemm_alpha_) \
          + ((const OTYPE*)(C))[libxsmm_inline_xgemm_ni_*libxsmm_inline_xgemm_ldc_+libxsmm_inline_xgemm_mi_] * libxsmm_inline_xgemm_beta_; \
      } \
    } \
  } \
} while(0)

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
