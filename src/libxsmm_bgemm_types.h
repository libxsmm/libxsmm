/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
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
#ifndef LIBXSMM_BGEMM_TYPES_H
#define LIBXSMM_BGEMM_TYPES_H

#include "libxsmm_gemm.h"

#if !defined(LIBXSMM_BGEMM_CHECKS) && !defined(NDEBUG)
# define LIBXSMM_BGEMM_CHECKS
#endif


LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE libxsmm_bgemm_lock {
  char pad[LIBXSMM_CACHELINE];
  volatile LIBXSMM_ATOMIC_LOCKTYPE state;
} libxsmm_bgemm_lock;


LIBXSMM_EXTERN_C struct LIBXSMM_RETARGETABLE libxsmm_bgemm_handle {
  union { double d; float s; int w; } alpha, beta;
  libxsmm_gemm_precision iprec, oprec;
  libxsmm_xmmfunction kernel_pf;
  libxsmm_xmmfunction kernel;
  libxsmm_barrier* barrier;
  libxsmm_bgemm_lock* locks;
  libxsmm_bgemm_order order;
  libxsmm_blasint m, n, k, bm, bn, bk;
  libxsmm_blasint b_m1, b_n1, b_k1, b_k2;
  libxsmm_blasint mb, nb, kb;
  void* buffer;
  int nthreads;
};

#endif /*LIBXSMM_BGEMM_TYPES_H*/

