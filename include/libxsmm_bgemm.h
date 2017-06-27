/******************************************************************************
** Copyright (c) 2017, Intel Corporation                                     **
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

#include <libxsmm_frontend.h>
#include <libxsmm_sync.h>


#include <libxsmm_intrinsics_x86.h>
typedef struct{ volatile int var[16]; } lock_var;
#define LOCK_T lock_var
#define LOCK_INIT(x) { (x)->var[0] = 0; }
#define LOCK_SET(x) { do { \
        while ((x)->var[0] ==1); \
        _mm_pause(); \
      } while(__sync_lock_test_and_set(&((x)->var[0]), 1) != 0); \
}
#define LOCK_CHECK(x) { while ((x)->var[0] ==0); _mm_pause(); }
#define LOCK_UNSET(x) { __sync_lock_release(&((x)->var[0])); }


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
typedef struct LIBXSMM_RETARGETABLE libxsmm_bgemm_handle libxsmm_bgemm_handle;


LIBXSMM_API libxsmm_bgemm_handle* libxsmm_bgemm_handle_create(
  libxsmm_gemm_precision precision, char transa, char transb,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint bm, libxsmm_blasint bn, libxsmm_blasint bk,
  const void* alpha, const void* beta,
  const libxsmm_bgemm_order* order);

LIBXSMM_API void libxsmm_bgemm_handle_destroy(const libxsmm_bgemm_handle* handle);

LIBXSMM_API int libxsmm_bgemm_copyin_a(const libxsmm_bgemm_handle* handle, const void* src, const libxsmm_blasint* ld, void* dst);
LIBXSMM_API int libxsmm_bgemm_copyin_b(const libxsmm_bgemm_handle* handle, const void* src, const libxsmm_blasint* ld, void* dst);
LIBXSMM_API int libxsmm_bgemm_copyin_c(const libxsmm_bgemm_handle* handle, const void* src, const libxsmm_blasint* ld, void* dst);

/**
 * Fine grain parallelized version(s) of BGEMM
 * - Requires block structure layout for A,B matrices
 * - Parallelized across all three dims - M, N, K
 * - Uses fine-grain on-demand locks for write to C and fast barrier
 * - Allows for calling multiple GEMMs, specified by 'count'
 */
LIBXSMM_API void libxsmm_bgemm(const libxsmm_bgemm_handle* handle,
  const void* a, const void* b, void* c, int tid, int nthreads);

LIBXSMM_API void libxsmm_bgemm_omp(const libxsmm_bgemm_handle* handle,
  const void* a, const void* b, void* c);

#endif /*LIBXSMM_BGEMM_H*/

