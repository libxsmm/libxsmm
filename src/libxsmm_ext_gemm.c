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
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include "libxsmm_gemm.h"
#include "libxsmm_ext.h"


#if defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT)

#if defined(LIBXSMM_GEMM_WRAP_STATIC) || defined(LIBXSMM_GEMM_WRAP_DYNAMIC)
LIBXSMM_API_DEFINITION LIBXSMM_GEMM_WEAK libxsmm_sgemm_function libxsmm_original_sgemm(const void* caller)
{
  static LIBXSMM_TLS libxsmm_sgemm_function original = 0;
  LIBXSMM_GEMM_WRAPPER(float, original, caller);
  assert(0 != original);
  return original;
}


LIBXSMM_API_DEFINITION LIBXSMM_GEMM_WEAK libxsmm_dgemm_function libxsmm_original_dgemm(const void* caller)
{
  static LIBXSMM_TLS libxsmm_dgemm_function original = 0;
  LIBXSMM_GEMM_WRAPPER(double, original, caller);
  assert(0 != original);
  return original;
}
#endif


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_FSYMBOL(__wrap_sgemm)(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  libxsmm_sgemm_omp(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_FSYMBOL(__wrap_dgemm)(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  libxsmm_dgemm_omp(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#endif /*defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT)*/


LIBXSMM_API_DEFINITION void libxsmm_sgemm_omp(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_GEMM_DESCRIPTOR_DIM_TYPE tm, tn, tk;
  LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb);
  LIBXSMM_INIT
  tm = libxsmm_gemm_tile[1/*SP*/][0/*M*/];
  tn = libxsmm_gemm_tile[1/*SP*/][1/*N*/];
  tk = libxsmm_gemm_tile[1/*SP*/][2/*K*/];
  assert(0 < tm && 0 < tn && 0 < tk && 0 < libxsmm_nt);
#if defined(_OPENMP)
  if (0 != libxsmm_mt) { /* enable OpenMP support */
    if (0 == LIBXSMM_MOD2(libxsmm_mt, 2)) { /* even: enable internal parallelization */
# if defined(LIBXSMM_EXT_TASKS)
      if (0 == libxsmm_tasks)
# endif
      {
        LIBXSMM_TILED_XGEMM(LIBXSMM_EXT_FOR_PARALLEL, LIBXSMM_NOOP, LIBXSMM_EXT_FOR_SINGLE,
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
        LIBXSMM_TILED_XGEMM(LIBXSMM_NOOP, LIBXSMM_NOOP, LIBXSMM_EXT_FOR_SINGLE,
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
#if defined(LIBXSMM_GEMM_TILED)
    libxsmm_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#else
    LIBXSMM_TILED_XGEMM(LIBXSMM_NOOP, LIBXSMM_NOOP, LIBXSMM_NOOP,
      LIBXSMM_GEMM_COLLAPSE, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP,
      LIBXSMM_MIN_NTASKS, LIBXSMM_OVERHEAD, libxsmm_nt,
      float, flags | LIBXSMM_GEMM_FLAG_F32PREC, tm, tn, tk, *m, *n, *k,
      0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA),
      a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
      0 != beta ? *beta : ((float)LIBXSMM_BETA),
      c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
#endif
  }
}


LIBXSMM_API_DEFINITION void libxsmm_dgemm_omp(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_GEMM_DESCRIPTOR_DIM_TYPE tm, tn, tk;
  LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb);
  LIBXSMM_INIT
  tm = libxsmm_gemm_tile[0/*DP*/][0/*M*/];
  tn = libxsmm_gemm_tile[0/*DP*/][1/*N*/];
  tk = libxsmm_gemm_tile[0/*DP*/][2/*K*/];
  assert(0 < tm && 0 < tn && 0 < tk && 0 < libxsmm_nt);
#if defined(_OPENMP)
  if (0 != libxsmm_mt) { /* enable OpenMP support */
    if (0 == LIBXSMM_MOD2(libxsmm_mt, 2)) { /* even: enable internal parallelization */
# if defined(LIBXSMM_EXT_TASKS)
      if (0 == libxsmm_tasks)
# endif
      {
        LIBXSMM_TILED_XGEMM(LIBXSMM_EXT_FOR_PARALLEL, LIBXSMM_NOOP, LIBXSMM_EXT_FOR_SINGLE,
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
        LIBXSMM_TILED_XGEMM(LIBXSMM_NOOP, LIBXSMM_NOOP, LIBXSMM_EXT_FOR_SINGLE,
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
#if defined(LIBXSMM_GEMM_TILED)
    libxsmm_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#else
    LIBXSMM_TILED_XGEMM(LIBXSMM_NOOP, LIBXSMM_NOOP, LIBXSMM_NOOP,
      LIBXSMM_GEMM_COLLAPSE, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP,
      LIBXSMM_MIN_NTASKS, LIBXSMM_OVERHEAD, libxsmm_nt,
      double, flags, tm, tn, tk, *m, *n, *k,
      0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA),
      a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
      0 != beta ? *beta : ((double)LIBXSMM_BETA),
      c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
#endif
  }
}


#if defined(LIBXSMM_BUILD)

/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_sgemm_omp)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*,
  const float*, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_sgemm_omp)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  libxsmm_sgemm_omp(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_dgemm_omp)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*,
  const double*, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_dgemm_omp)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  libxsmm_dgemm_omp(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#endif /*defined(LIBXSMM_BUILD)*/

