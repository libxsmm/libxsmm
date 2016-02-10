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
#ifndef LIBXSMM_GEMM_DIFF_H
#define LIBXSMM_GEMM_DIFF_H

#include <libxsmm.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <libxsmm_generator.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


/** Generic implementations only relying on high-level constructs. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE unsigned int libxsmm_gemm_diff(const libxsmm_gemm_descriptor* a, const libxsmm_gemm_descriptor* b);

/** Implementations using specific instruction set extensions. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE unsigned int libxsmm_gemm_diff_sse(const libxsmm_gemm_descriptor* a, const libxsmm_gemm_descriptor* b);
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE unsigned int libxsmm_gemm_diff_avx(const libxsmm_gemm_descriptor* a, const libxsmm_gemm_descriptor* b);
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE unsigned int libxsmm_gemm_diff_avx2(const libxsmm_gemm_descriptor* a, const libxsmm_gemm_descriptor* b);

#if defined(__MIC__)
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE unsigned int libxsmm_gemm_diff_imci(const libxsmm_gemm_descriptor* a, const libxsmm_gemm_descriptor* b);
#endif


#if defined(LIBXSMM_BUILD) && !defined(LIBXSMM_GEMM_DIFF_NOINLINE)
# include "libxsmm_gemm_diff.c"
#endif

#endif /*LIBXSMM_GEMM_DIFF_H*/

