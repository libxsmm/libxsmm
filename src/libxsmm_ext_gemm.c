/******************************************************************************
** Copyright (c) 2016, Intel Corporation                                     **
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
#include "libxsmm_gemm.h"
#include "libxsmm_sync.h"
#include "libxsmm_ext.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#if !defined(NDEBUG)
# include <stdio.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


#if defined(LIBXSMM_BUILD) && defined(LIBXSMM_RTLD_NEXT) && !defined(__STATIC)

/* overrides the regular libxsmm_gemm_init in case of LD_PRELOADing libxsmmext */
LIBXSMM_API_DEFINITION int libxsmm_gemm_init(int archid, int prefetch)
{
  int result = EXIT_SUCCESS;
  /* internal pre-initialization step */
  libxsmm_gemm_configure(archid, prefetch);
#if !defined(__BLAS) || (0 != __BLAS)
  result = (0 != libxsmm_original_sgemm && 0 != libxsmm_original_dgemm) ? EXIT_SUCCESS : EXIT_FAILURE;
#endif
#if !defined(NDEBUG) /* library code is expected to be mute */
  if (EXIT_SUCCESS != result) {
    static LIBXSMM_TLS int error_blas = 0;
    if (0 == error_blas) {
      fprintf(stderr, "LIBXSMM: application must be linked against a LAPACK/BLAS implementation!\n");
      error_blas = 1;
    }
  }
#endif
  return result;
}

#endif /*defined(LIBXSMM_BUILD) && defined(LIBXSMM_RTLD_NEXT) && !defined(__STATIC)*/


LIBXSMM_API_DEFINITION void libxsmm_sgemm_omp(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  const int tm = libxsmm_gemm_tile[1/*SP*/][0/*M*/];
  const int tn = libxsmm_gemm_tile[1/*SP*/][1/*N*/];
  const int tk = libxsmm_gemm_tile[1/*SP*/][2/*K*/];
  LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
  assert(0 < tm && 0 < tn && 0 < tk && 0 < libxsmm_nt);
  LIBXSMM_INIT
#if defined(_OPENMP)
  if (0 != libxsmm_mp) { /* enable OpenMP support */
    if (0 == LIBXSMM_MOD2(libxsmm_mp, 2)) { /* even: enable internal parallelization */
# if defined(LIBXSMM_EXT_TASKS)
      if (0 == libxsmm_tasks)
# endif
      {
        LIBXSMM_TILED_XGEMM(LIBXSMM_EXT_FOR_PARALLEL, LIBXSMM_NOOP, LIBXSMM_EXT_SINGLE,
          LIBXSMM_GEMM_COLLAPSE, LIBXSMM_EXT_FOR_LOOP, LIBXSMM_EXT_FOR_KERNEL, LIBXSMM_EXT_FOR_SYNC,
          LIBXSMM_EXT_MIN_NTASKS, LIBXSMM_EXT_OVERHEAD, libxsmm_nt,
          float, flags | LIBXSMM_GEMM_FLAG_F32PREC, tm, tn, tk, *m, *n, *k,
          0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA),
          a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
          0 != beta ? *beta : ((float)LIBXSMM_BETA),
          c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
      }
# if defined(LIBXSMM_EXT_TASKS)
      else {
        LIBXSMM_TILED_XGEMM(LIBXSMM_EXT_TSK_PARALLEL, LIBXSMM_EXT_SINGLE, LIBXSMM_NOOP,
          LIBXSMM_GEMM_COLLAPSE, LIBXSMM_EXT_TSK_LOOP, LIBXSMM_EXT_TSK_KERNEL_VARS, LIBXSMM_NOOP,
          LIBXSMM_EXT_MIN_NTASKS, LIBXSMM_EXT_OVERHEAD, libxsmm_nt,
          float, flags | LIBXSMM_GEMM_FLAG_F32PREC, tm, tn, tk, *m, *n, *k,
          0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA),
          a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
          0 != beta ? *beta : ((float)LIBXSMM_BETA),
          c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
      }
# endif
    }
    else { /* odd: prepare for external parallelization */
# if defined(LIBXSMM_EXT_TASKS)
      if (0 == libxsmm_tasks)
# endif
      {
        LIBXSMM_TILED_XGEMM(LIBXSMM_NOOP, LIBXSMM_NOOP, LIBXSMM_EXT_SINGLE,
          LIBXSMM_GEMM_COLLAPSE, LIBXSMM_EXT_FOR_LOOP, LIBXSMM_EXT_FOR_KERNEL, LIBXSMM_EXT_FOR_SYNC,
          LIBXSMM_EXT_MIN_NTASKS, LIBXSMM_EXT_OVERHEAD, libxsmm_nt,
          float, flags | LIBXSMM_GEMM_FLAG_F32PREC, tm, tn, tk, *m, *n, *k,
          0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA),
          a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
          0 != beta ? *beta : ((float)LIBXSMM_BETA),
          c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
      }
# if defined(LIBXSMM_EXT_TASKS)
      else {
        LIBXSMM_TILED_XGEMM(LIBXSMM_NOOP, LIBXSMM_EXT_SINGLE, LIBXSMM_NOOP,
          LIBXSMM_GEMM_COLLAPSE, LIBXSMM_EXT_TSK_LOOP, LIBXSMM_EXT_TSK_KERNEL_VARS, LIBXSMM_EXT_TSK_SYNC,
          LIBXSMM_EXT_MIN_NTASKS, LIBXSMM_EXT_OVERHEAD, libxsmm_nt,
          float, flags | LIBXSMM_GEMM_FLAG_F32PREC, tm, tn, tk, *m, *n, *k,
          0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA),
          a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
          0 != beta ? *beta : ((float)LIBXSMM_BETA),
          c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
      }
# endif
    }
  }
  else
#endif
  {
    LIBXSMM_TILED_XGEMM(LIBXSMM_NOOP, LIBXSMM_NOOP, LIBXSMM_NOOP,
      LIBXSMM_GEMM_COLLAPSE, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP,
      LIBXSMM_MIN_NTASKS, LIBXSMM_OVERHEAD, libxsmm_nt,
      float, flags | LIBXSMM_GEMM_FLAG_F32PREC, tm, tn, tk, *m, *n, *k,
      0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA),
      a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
      0 != beta ? *beta : ((float)LIBXSMM_BETA),
      c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
  }
}


LIBXSMM_API_DEFINITION void libxsmm_dgemm_omp(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  const int tm = libxsmm_gemm_tile[0/*DP*/][0/*M*/];
  const int tn = libxsmm_gemm_tile[0/*DP*/][1/*N*/];
  const int tk = libxsmm_gemm_tile[0/*DP*/][2/*K*/];
  LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
  LIBXSMM_INIT
#if defined(_OPENMP)
  if (0 != libxsmm_mp) { /* enable OpenMP support */
    if (0 == LIBXSMM_MOD2(libxsmm_mp, 2)) { /* even: enable internal parallelization */
# if defined(LIBXSMM_EXT_TASKS)
      if (0 == libxsmm_tasks)
# endif
      {
        LIBXSMM_TILED_XGEMM(LIBXSMM_EXT_FOR_PARALLEL, LIBXSMM_NOOP, LIBXSMM_EXT_SINGLE,
          LIBXSMM_GEMM_COLLAPSE, LIBXSMM_EXT_FOR_LOOP, LIBXSMM_EXT_FOR_KERNEL, LIBXSMM_EXT_FOR_SYNC,
          LIBXSMM_EXT_MIN_NTASKS, LIBXSMM_EXT_OVERHEAD, libxsmm_nt,
          double, flags, tm, tn, tk, *m, *n, *k,
          0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA),
          a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
          0 != beta ? *beta : ((double)LIBXSMM_BETA),
          c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
      }
# if defined(LIBXSMM_EXT_TASKS)
      else {
        LIBXSMM_TILED_XGEMM(LIBXSMM_EXT_TSK_PARALLEL, LIBXSMM_EXT_SINGLE, LIBXSMM_NOOP,
          LIBXSMM_GEMM_COLLAPSE, LIBXSMM_EXT_TSK_LOOP, LIBXSMM_EXT_TSK_KERNEL_VARS, LIBXSMM_NOOP,
          LIBXSMM_EXT_MIN_NTASKS, LIBXSMM_EXT_OVERHEAD, libxsmm_nt,
          double, flags, tm, tn, tk, *m, *n, *k,
          0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA),
          a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
          0 != beta ? *beta : ((double)LIBXSMM_BETA),
          c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
      }
# endif
    }
    else { /* odd: prepare for external parallelization */
# if defined(LIBXSMM_EXT_TASKS)
      if (0 == libxsmm_tasks)
# endif
      {
        LIBXSMM_TILED_XGEMM(LIBXSMM_NOOP, LIBXSMM_NOOP, LIBXSMM_EXT_SINGLE,
          LIBXSMM_GEMM_COLLAPSE, LIBXSMM_EXT_FOR_LOOP, LIBXSMM_EXT_FOR_KERNEL, LIBXSMM_EXT_FOR_SYNC,
          LIBXSMM_EXT_MIN_NTASKS, LIBXSMM_EXT_OVERHEAD, libxsmm_nt,
          double, flags, tm, tn, tk, *m, *n, *k,
          0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA),
          a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
          0 != beta ? *beta : ((double)LIBXSMM_BETA),
          c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
      }
# if defined(LIBXSMM_EXT_TASKS)
      else {
        LIBXSMM_TILED_XGEMM(LIBXSMM_NOOP, LIBXSMM_EXT_SINGLE, LIBXSMM_NOOP,
          LIBXSMM_GEMM_COLLAPSE, LIBXSMM_EXT_TSK_LOOP, LIBXSMM_EXT_TSK_KERNEL_VARS, LIBXSMM_EXT_TSK_SYNC,
          LIBXSMM_EXT_MIN_NTASKS, LIBXSMM_EXT_OVERHEAD, libxsmm_nt,
          double, flags, tm, tn, tk, *m, *n, *k,
          0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA),
          a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
          0 != beta ? *beta : ((double)LIBXSMM_BETA),
          c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
      }
# endif
    }
  }
  else
#endif
  {
    LIBXSMM_TILED_XGEMM(LIBXSMM_NOOP, LIBXSMM_NOOP, LIBXSMM_NOOP,
      LIBXSMM_GEMM_COLLAPSE, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP,
      LIBXSMM_MIN_NTASKS, LIBXSMM_OVERHEAD, libxsmm_nt,
      double, flags, tm, tn, tk, *m, *n, *k,
      0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA),
      a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
      0 != beta ? *beta : ((double)LIBXSMM_BETA),
      c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
  }
}


#if defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT)

LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE LIBXSMM_GEMM_WEAK void LIBXSMM_FSYMBOL(sgemm)(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  assert(LIBXSMM_FSYMBOL(sgemm) != libxsmm_original_sgemm);
  libxsmm_sgemm_omp(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE LIBXSMM_GEMM_WEAK void LIBXSMM_FSYMBOL(dgemm)(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  assert(LIBXSMM_FSYMBOL(dgemm) != libxsmm_original_dgemm);
  libxsmm_dgemm_omp(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#endif

