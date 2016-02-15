/******************************************************************************
** Copyright (c) 2015-2016, Intel Corporation                                **
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
#include "libxsmm_gemm_wrap.h"
#include "libxsmm_gemm.h"


#if defined(LIBXSMM_GEMM_WRAP)
#if !defined(__STATIC) /*avoid remark about external function definition with no prior declaration*/
# if defined(LIBXSMM_OFFLOAD_TARGET)
#   pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
# endif
# include <stdlib.h>
# include <dlfcn.h>
# if defined(LIBXSMM_OFFLOAD_TARGET)
#   pragma offload_attribute(pop)
# endif

LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_GEMM_WRAP_SGEMM(
  const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*,
  const float*, const libxsmm_blasint* ldb,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_GEMM_WRAP_DGEMM(
  const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*,
  const double*, const libxsmm_blasint* ldb,
  const double*, double*, const libxsmm_blasint*);


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int libxsmm_gemm_init(
  libxsmm_sgemm_function sgemm_function, libxsmm_dgemm_function dgemm_function)
{
  if (NULL == sgemm_function) {
    union { const void* pv; libxsmm_sgemm_function pf; } internal = { NULL };
    internal.pv = dlsym(RTLD_NEXT, LIBXSMM_STRINGIFY(LIBXSMM_FSYMBOL(sgemm)));
    if (NULL != internal.pv) {
      libxsmm_internal_sgemm = internal.pf;
    }
  }
  else {
    libxsmm_internal_sgemm = sgemm_function;
  }

  if (NULL == dgemm_function) {
    union { const void* pv; libxsmm_dgemm_function pf; } internal = { NULL };
    internal.pv = dlsym(RTLD_NEXT, LIBXSMM_STRINGIFY(LIBXSMM_FSYMBOL(dgemm)));
    if (NULL != internal.pv) {
      libxsmm_internal_dgemm = internal.pf;
    }
  }
  else {
    libxsmm_internal_dgemm = dgemm_function;
  }

  return (NULL != libxsmm_internal_sgemm
       && NULL != libxsmm_internal_dgemm)
    ? EXIT_SUCCESS
    : EXIT_FAILURE;
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int libxsmm_gemm_finalize(void)
{
  return EXIT_SUCCESS;
}
#endif /*!defined(__STATIC)*/


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_GEMM_WRAP_SGEMM(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
  assert(LIBXSMM_GEMM_WRAP_SGEMM != libxsmm_internal_sgemm);
  LIBXSMM_XGEMM(float, libxsmm_blasint, libxsmm_internal_sgemm, flags, *m, *n, *k,
    0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA),
    a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
    0 != beta ? *beta : ((float)LIBXSMM_BETA),
    c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_GEMM_WRAP_DGEMM(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
  assert(LIBXSMM_GEMM_WRAP_DGEMM != libxsmm_internal_dgemm);
  LIBXSMM_XGEMM(double, libxsmm_blasint, libxsmm_internal_dgemm, flags, *m, *n, *k,
    0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA),
    a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
    0 != beta ? *beta : ((double)LIBXSMM_BETA),
    c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
}
#endif /*defined(LIBXSMM_GEMM_WRAP)*/

