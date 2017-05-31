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

/** Enable masked load of reference (A) descriptor */
#if !defined(LIBXSMM_GEMM_DIFF_MASK_A)
# define LIBXSMM_GEMM_DIFF_MASK_A
#endif
#if !defined(LIBXSMM_GEMM_DIFF_ZERO_PADDED)
/*# define LIBXSMM_GEMM_DIFF_ZERO_PADDED*/
#endif
/** Enable generic implementation (1: LIBXSMM, 2: memcmp) */
#if !defined(LIBXSMM_GEMM_DIFF_SW) /*&& defined(__MIC__)*/
# define LIBXSMM_GEMM_DIFF_SW 1
#endif

#if defined(LIBXSMM_BUILD) && !defined(LIBXSMM_GEMM_DIFF_NOINLINE)
# define LIBXSMM_GEMM_DIFF_API LIBXSMM_INLINE LIBXSMM_RETARGETABLE
# define LIBXSMM_GEMM_DIFF_API_DEFINITION LIBXSMM_GEMM_DIFF_API LIBXSMM_ATTRIBUTE_UNUSED
#else
# define LIBXSMM_GEMM_DIFF_API LIBXSMM_API
# define LIBXSMM_GEMM_DIFF_API_DEFINITION LIBXSMM_API_DEFINITION
#endif


/** Function type representing the gemm_diff functionality. */
typedef LIBXSMM_RETARGETABLE unsigned int (*libxsmm_gemm_diff_function)(
  const libxsmm_gemm_descriptor*, const libxsmm_gemm_descriptor*);
/** Function type representing the gemm_diffn functionality. */
typedef LIBXSMM_RETARGETABLE unsigned int (*libxsmm_gemm_diffn_function)(
  const libxsmm_gemm_descriptor*, const void*, unsigned int, unsigned int, int);


/** Initialize GEMM/DIFF module; not thread-safe. */
LIBXSMM_GEMM_DIFF_API void libxsmm_gemm_diff_init(int target_arch);
LIBXSMM_GEMM_DIFF_API void libxsmm_gemm_diff_finalize(void);

/** Dispatched implementation which may (or may not) use a SIMD extension. */
LIBXSMM_GEMM_DIFF_API unsigned int libxsmm_gemm_diff(
  const libxsmm_gemm_descriptor* reference, const libxsmm_gemm_descriptor* desc);

/** Generic implementation which is only relying on high-level constructs. */
LIBXSMM_GEMM_DIFF_API unsigned int libxsmm_gemm_diff_sw(
  const libxsmm_gemm_descriptor* reference, const libxsmm_gemm_descriptor* desc);
/** Collection of implementations which are using specific instruction set extensions. */
LIBXSMM_GEMM_DIFF_API unsigned int libxsmm_gemm_diff_sse(
  const libxsmm_gemm_descriptor* reference, const libxsmm_gemm_descriptor* desc);
LIBXSMM_GEMM_DIFF_API unsigned int libxsmm_gemm_diff_avx(
  const libxsmm_gemm_descriptor* reference, const libxsmm_gemm_descriptor* desc);
LIBXSMM_GEMM_DIFF_API unsigned int libxsmm_gemm_diff_avx2(
  const libxsmm_gemm_descriptor* reference, const libxsmm_gemm_descriptor* desc);
LIBXSMM_GEMM_DIFF_API unsigned int libxsmm_gemm_diff_imci(
  const libxsmm_gemm_descriptor* reference, const libxsmm_gemm_descriptor* desc);

/**
 * Compare a number of descriptors (array) against a reference descriptor.
 * Returns the index of the first match (or ndesc in case of no match).
 */
LIBXSMM_GEMM_DIFF_API unsigned int libxsmm_gemm_diffn(const libxsmm_gemm_descriptor* reference,
  /** Array of descriptors with ndesc elements. */
  const void* descs, unsigned int hint, unsigned int ndescs,
  /** Number of bytes until the next descriptor is reached (stride). */
  int nbytes);

/** Generic implementation of libxsmm_gemm_diffn which is only relying on high-level constructs. */
LIBXSMM_GEMM_DIFF_API unsigned int libxsmm_gemm_diffn_sw(const libxsmm_gemm_descriptor* reference,
  const void* descs, unsigned int hint, unsigned int ndescs, int nbytes);
/** Collection of implementations which are using specific instruction set extensions. */
LIBXSMM_GEMM_DIFF_API unsigned int libxsmm_gemm_diffn_avx(const libxsmm_gemm_descriptor* reference,
  const void* descs, unsigned int hint, unsigned int ndescs, int nbytes);
LIBXSMM_GEMM_DIFF_API unsigned int libxsmm_gemm_diffn_avx2(const libxsmm_gemm_descriptor* reference,
  const void* descs, unsigned int hint, unsigned int ndescs, int nbytes);
LIBXSMM_GEMM_DIFF_API unsigned int libxsmm_gemm_diffn_avx512(const libxsmm_gemm_descriptor* reference,
  const void* descs, unsigned int hint, unsigned int ndescs, int nbytes);
LIBXSMM_GEMM_DIFF_API unsigned int libxsmm_gemm_diffn_imci(const libxsmm_gemm_descriptor* reference,
  const void* descs, unsigned int hint, unsigned int ndescs, int nbytes);

#if defined(LIBXSMM_BUILD) && !defined(LIBXSMM_GEMM_DIFF_NOINLINE)
# include "libxsmm_gemm_diff.c"
#endif

#endif /*LIBXSMM_GEMM_DIFF_H*/

