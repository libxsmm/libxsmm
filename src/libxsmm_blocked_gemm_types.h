/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_BLOCKED_GEMM_TYPES_H
#define LIBXSMM_BLOCKED_GEMM_TYPES_H

#include "libxsmm_gemm.h"

#if !defined(LIBXSMM_BLOCKED_GEMM_CHECKS) && !defined(NDEBUG)
# define LIBXSMM_BLOCKED_GEMM_CHECKS
#endif


LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE libxsmm_blocked_gemm_lock {
  char pad[LIBXSMM_CACHELINE];
  volatile LIBXSMM_ATOMIC_LOCKTYPE state;
} libxsmm_blocked_gemm_lock;


LIBXSMM_EXTERN_C struct LIBXSMM_RETARGETABLE libxsmm_blocked_gemm_handle {
  union { double d; float s; int w; } alpha, beta;
  libxsmm_gemm_precision iprec, oprec;
  libxsmm_xmmfunction kernel_pf;
  libxsmm_xmmfunction kernel;
  libxsmm_barrier* barrier;
  libxsmm_blocked_gemm_lock* locks;
  libxsmm_blocked_gemm_order order;
  libxsmm_blasint m, n, k, bm, bn, bk;
  libxsmm_blasint b_m1, b_n1, b_k1, b_k2;
  libxsmm_blasint mb, nb, kb;
  void* buffer;
  int nthreads;
};

#endif /*LIBXSMM_BLOCKED_GEMM_TYPES_H*/

