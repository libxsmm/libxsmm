/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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
/* Kunal Banerjee (Intel Corp.), Dheevatsa Mudigere (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_BGEMM_H
#define LIBXSMM_BGEMM_H

#include <libxsmm.h>

#define _USE_LIBXSMM_PREFETCH
#define BG_type 1 /* 1-float, 2-double */

/*#define OMP_LOCK*/
#if defined(OMP_LOCK)
#define LOCK_T omp_lock_t
#define LOCK_INIT(x) {omp_init_lock(x);}
#define LOCK_SET(x) {omp_set_lock(x);}
#define LOCK_UNSET(x) {omp_unset_lock(x);}
#else
typedef struct{volatile int var[16];}lock_var;
#define LOCK_T lock_var
#define LOCK_INIT(x) {(x)->var[0] = 0;}
#define LOCK_SET(x) {do { \
        while ((x)->var[0] ==1); \
        _mm_pause(); \
      } while(__sync_lock_test_and_set(&((x)->var[0]), 1) != 0); \
}
#define LOCK_CHECK(x) {while ((x)->var[0] ==0); _mm_pause();}
#define LOCK_UNSET(x) {__sync_lock_release(&((x)->var[0]));}
#endif

#if (BG_type==2)
  typedef double real;
#else
  typedef float real;
#endif


typedef struct LIBXSMM_RETARGETABLE libxsmm_blkgemm_handle {
  int m, n, k;
  int bm, bn, bk;
  int mb, nb, kb;
  /* The following have been added by KB */
  int b_m1, b_n1, b_k1, b_k2;
  int _ORDER;
  int C_pre_init;
  LOCK_T* _wlock;
  libxsmm_xmmfunction _l_kernel;
#if defined(_USE_LIBXSMM_PREFETCH)
  libxsmm_xmmfunction _l_kernel_pf;
#endif
  libxsmm_barrier* bar;
} libxsmm_blkgemm_handle;


LIBXSMM_API void libxsmm_blkgemm_handle_alloc(libxsmm_blkgemm_handle* handle, int M, int MB, int N, int NB);

LIBXSMM_API void libxsmm_blksgemm_init_a( libxsmm_blkgemm_handle* handle,
                                             real* libxsmm_mat_dst,
                                             real* colmaj_mat_src );

LIBXSMM_API void libxsmm_blksgemm_init_b( libxsmm_blkgemm_handle* handle,
                                             real* libxsmm_mat_dst,
                                             real* colmaj_mat_src );

LIBXSMM_API void libxsmm_blksgemm_init_c( libxsmm_blkgemm_handle* handle,
                                             real* libxsmm_mat_dst,
                                             real* colmaj_mat_src );

LIBXSMM_API void libxsmm_blksgemm_check_c( libxsmm_blkgemm_handle* handle,
                                              real* libxsmm_mat_dst,
                                              real* colmaj_mat_src );

LIBXSMM_API void libxsmm_blksgemm_exec( const libxsmm_blkgemm_handle* handle,
                                           const char transA,
                                           const char transB,
                                           const real* alpha,
                                           const real* a,
                                           const real* b,
                                           const real* beta,
                                           real* c );

/**
 * Fine grain parallelized version(s) of BGEMM
 * - Requires block structure layout for A,B matrices
 * - Parallelized across all three dims - M, N, K
 * - Uses fine-grain on-demand locks for write to C and fast barrier
 * - Allows for calling multiple GEMMs, specified by 'count'
 */
LIBXSMM_API void libxsmm_bgemm(
  int _M, int _N, int _K,
  int B_M, int B_N, int B_K,
  const real* Ap, const real* Bp, real* Cp,
  int tid, int nthrds,
  const libxsmm_blkgemm_handle* handle);

#endif /*LIBXSMM_BGEMM_H*/

