/******************************************************************************
** Copyright (c) 2017-2018, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_BGEMM_H
#define LIBXSMM_BGEMM_H

#include "libxsmm_typedefs.h"


/** Denotes the BGEMM data order. */
typedef enum libxsmm_bgemm_order {
  LIBXSMM_BGEMM_ORDER_JIK = 0,
  LIBXSMM_BGEMM_ORDER_IJK = 1,
  LIBXSMM_BGEMM_ORDER_JKI = 2,
  LIBXSMM_BGEMM_ORDER_IKJ = 3,
  LIBXSMM_BGEMM_ORDER_KJI = 4,
  LIBXSMM_BGEMM_ORDER_KIJ = 5
} libxsmm_bgemm_order;

/** Describes the Block-GEMM (BGEMM) operation. */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_bgemm_handle libxsmm_bgemm_handle;


LIBXSMM_API libxsmm_bgemm_handle* libxsmm_bgemm_handle_create(
  /** Number of threads used to run BGEMM. */
  /*unsigned*/ int nthreads, libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  /** If the block-size (BM, BN, or BK) is not given, a suitable value is chosen internally. */
  const libxsmm_blasint* bm, const libxsmm_blasint* bn, const libxsmm_blasint* bk,
  /** If b_m1, b_n1, b_k1, or b_k2 is not supplied, the respective value defaults to one. */
  const libxsmm_blasint* b_m1, const libxsmm_blasint* b_n1, const libxsmm_blasint* b_k1, const libxsmm_blasint* b_k2,
  /** If alpha is not supplied (NULL), then LIBXSMM_ALPHA is used instead. */ const void* alpha,
  /** If beta is not supplied (NULL), then LIBXSMM_BETA is used instead. */   const void*  beta,
  /** See libxsmm_gemm_flags (LIBXSMM_FLAGS is used if NULL is given). */ const int* gemm_flags,
  /** See libxsmm_gemm_prefetch_type; a strategy chosen automatically if NULL is given. */
  const libxsmm_gemm_prefetch_type* prefetch,
  /** See libxsmm_bgemm_order; an order is chosen automatically if NULL is given. */
  const libxsmm_bgemm_order* order);

LIBXSMM_API void libxsmm_bgemm_handle_destroy(const libxsmm_bgemm_handle* handle);

/** Copy-in functions for A, B, and C matrices. A leading dimension for the source buffer is optional and can be NULL. */
LIBXSMM_API int libxsmm_bgemm_copyin_a(const libxsmm_bgemm_handle* handle, const void* src, const libxsmm_blasint* ld, void* dst);
LIBXSMM_API int libxsmm_bgemm_copyin_b(const libxsmm_bgemm_handle* handle, const void* src, const libxsmm_blasint* ld, void* dst);
LIBXSMM_API int libxsmm_bgemm_copyin_c(const libxsmm_bgemm_handle* handle, const void* src, const libxsmm_blasint* ld, void* dst);
/** Copy-out function for the C-matrix. A leading dimension for the destination buffer is optional and can be NULL. */
LIBXSMM_API int libxsmm_bgemm_copyout_c(const libxsmm_bgemm_handle* handle, const void* src, const libxsmm_blasint* ld, void* dst);

/** Convert function required to reorganize elements in delta for BWD and UPD passes of RNN and LSTM */
LIBXSMM_API int libxsmm_bgemm_convert_b_to_a(const libxsmm_bgemm_handle* handle, const void* src, const libxsmm_blasint* ld, void* dst);

/**
* Fine grain parallelized block-GEMM (BGEMM), which uses a block structure
* layout for the A and B matrices. The implementation is parallelized
* among M, N, and K using fine-grained on-demand locks when writing C.
*/
LIBXSMM_API void libxsmm_bgemm_st(const libxsmm_bgemm_handle* handle, const void* a, const void* b, void* c,
  /*unsigned*/int start_thread, /*unsigned*/int tid);

/**
 * Implementation of libxsmm_bgemm, which is parallelized with OpenMP
 * and uses an OpenMP or custom barrier implementation. The function
 * allows to run multiple GEMMs, which is specified by 'count' (RNNs).
 * This function requires to link against libxsmmext.
 */
LIBXSMM_APIEXT void libxsmm_bgemm_omp(const libxsmm_bgemm_handle* handle,
  const void* a, const void* b, void* c, /*unsigned*/int count);

#endif /*LIBXSMM_BGEMM_H*/

