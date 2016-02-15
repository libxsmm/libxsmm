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
#ifndef LIBXSMM_GEMM_WRAP_H
#define LIBXSMM_GEMM_WRAP_H

#include <libxsmm.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#if !defined(NDEBUG)
# include <assert.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#define LIBXSMM_GEMM_DECLARE_FLAGS(FLAGS, TRANSA, TRANSB, M, N, K, A, B, C) \
  int FLAGS = (0 != (TRANSA) \
    ? (('N' == *(TRANSA) || 'n' == *(TRANSA)) ? (LIBXSMM_FLAGS & ~LIBXSMM_GEMM_FLAG_TRANS_A) \
                                              : (LIBXSMM_FLAGS |  LIBXSMM_GEMM_FLAG_TRANS_A)) \
    : LIBXSMM_FLAGS); \
  FLAGS = (0 != (TRANSB) \
    ? (('N' == *(TRANSB) || 'n' == *(TRANSB)) ? ((FLAGS) & ~LIBXSMM_GEMM_FLAG_TRANS_B) \
                                              : ((FLAGS) |  LIBXSMM_GEMM_FLAG_TRANS_B)) \
    : (FLAGS)); \
  assert(0 != (M) && 0 != (N) && 0 != (K) && 0 != (A) && 0 != (B) && 0 != (C))

#if !defined(LIBXSMM_GEMM_WRAP) && defined(__GNUC__) && !defined(_WIN32) && !(defined(__APPLE__) && defined(__MACH__) && \
  LIBXSMM_VERSION3(6, 1, 0) >= LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)) && !defined(__CYGWIN__)
# if defined(__STATIC) /* -Wl,--wrap=xgemm_ */
#   define LIBXSMM_GEMM_WRAP
#   define LIBXSMM_GEMM_WRAP_SGEMM LIBXSMM_FSYMBOL(__wrap_sgemm)
#   define LIBXSMM_GEMM_WRAP_DGEMM LIBXSMM_FSYMBOL(__wrap_dgemm)
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE LIBXSMM_ATTRIBUTE(weak) void LIBXSMM_FSYMBOL(__real_sgemm)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*, const float* b, const libxsmm_blasint*,
  const float* beta, float*, const libxsmm_blasint*);
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE LIBXSMM_ATTRIBUTE(weak) void LIBXSMM_FSYMBOL(__real_dgemm)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*, const double* b, const libxsmm_blasint*,
  const double* beta, double*, const libxsmm_blasint*);
/* mute warning about external function definition with no prior declaration */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_GEMM_WRAP_SGEMM(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*, const float* b, const libxsmm_blasint*,
  const float* beta, float*, const libxsmm_blasint*);
/* mute warning about external function definition with no prior declaration */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_GEMM_WRAP_DGEMM(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*, const double* b, const libxsmm_blasint*,
  const double* beta, double*, const libxsmm_blasint*);
# elif !defined(__CYGWIN__) /* LD_PRELOAD */
#   define LIBXSMM_GEMM_WRAP
#   define LIBXSMM_GEMM_WEAK LIBXSMM_ATTRIBUTE(weak)
#   define LIBXSMM_GEMM_WRAP_SGEMM LIBXSMM_FSYMBOL(sgemm)
#   define LIBXSMM_GEMM_WRAP_DGEMM LIBXSMM_FSYMBOL(dgemm)
# endif
#endif /*defined(LIBXSMM_GEMM_WRAP)*/

#if !defined(LIBXSMM_GEMM_WEAK)
# define LIBXSMM_GEMM_WEAK
#endif

#endif /*LIBXSMM_GEMM_WRAP_H*/

