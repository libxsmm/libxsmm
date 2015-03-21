/******************************************************************************
** Copyright (c) 2013-2015, Intel Corporation                                **
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
#ifndef LIBXSMM_MACROS_H
#define LIBXSMM_MACROS_H

#define LIBXSMM_STRINGIFY(SYMBOL) #SYMBOL
#define LIBXSMM_CONCATENATE(A, B) A##B
#define LIBXSMM_FSYMBOL(SYMBOL) LIBXSMM_CONCATENATE(SYMBOL, _)

#define LIBXSMM_BLASPREC(PREFIX, REAL, FUNCTION) LIBXSMM_BLASPREC_##REAL(PREFIX, FUNCTION)
#define LIBXSMM_BLASPREC_double(PREFIX, FUNCTION) PREFIX##d##FUNCTION
#define LIBXSMM_BLASPREC_float(PREFIX, FUNCTION) PREFIX##s##FUNCTION

#if defined(__cplusplus)
# define LIBXSMM_EXTERN_C extern "C"
# define LIBXSMM_INLINE inline
#else
# define LIBXSMM_EXTERN_C
# if (199901L <= __STDC_VERSION__)
#   define LIBXSMM_PRAGMA(DIRECTIVE) _Pragma(LIBXSMM_STRINGIFY(DIRECTIVE))
#   define LIBXSMM_RESTRICT restrict
#   define LIBXSMM_INLINE inline
# else
#   define LIBXSMM_INLINE static
# endif /*C99*/
#endif /*__cplusplus*/

#if !defined(LIBXSMM_RESTRICT)
# if ((defined(__GNUC__) && !defined(__CYGWIN32__)) || defined(__INTEL_COMPILER)) && !defined(_WIN32)
#   define LIBXSMM_RESTRICT __restrict__
# elif defined(_MSC_VER) || defined(__INTEL_COMPILER)
#   define LIBXSMM_RESTRICT __restrict
# else
#   define LIBXSMM_RESTRICT
# endif
#endif /*LIBXSMM_RESTRICT*/

#if !defined(LIBXSMM_PRAGMA)
# if defined(__INTEL_COMPILER) || defined(_MSC_VER)
#   define LIBXSMM_PRAGMA(DIRECTIVE) __pragma(DIRECTIVE)
# else
#   define LIBXSMM_PRAGMA(DIRECTIVE)
# endif
#endif /*LIBXSMM_PRAGMA*/

#if (201307 <= _OPENMP) // V4.0
# define LIBXSMM_PRAGMA_SIMD_REDUCTION(EXPRESSION) LIBXSMM_PRAGMA(omp simd reduction(EXPRESSION))
# define LIBXSMM_PRAGMA_SIMD_COLLAPSE(N) LIBXSMM_PRAGMA(omp simd collapse(N))
#elif defined(__INTEL_COMPILER)
# define LIBXSMM_PRAGMA_SIMD_REDUCTION(EXPRESSION) LIBXSMM_PRAGMA(simd reduction(EXPRESSION))
# define LIBXSMM_PRAGMA_SIMD_COLLAPSE(N) LIBXSMM_PRAGMA(simd collapse(N))
#else
# define LIBXSMM_PRAGMA_SIMD_REDUCTION(EXPRESSION)
# define LIBXSMM_PRAGMA_SIMD_COLLAPSE(N)
#endif

#if defined(__INTEL_COMPILER)
# if defined(LIBMICSMM_USE_LOOPHINTS)
#   pragma loop_count min(1), max(LIBMICSMM_MAX_RESULT_SIZE), avg(23*23)
# endif
#   if (defined(LIBMICSMM_USE_LIBXSMM) && defined(__LIBXSMM) && (0 < (LIBXSMM_ALIGNED_STORES))) || defined(LIBMICSMM_USE_XALIGN)
#   pragma simd aligned(c:1)
#   endif
#elif (201307 <= _OPENMP) // V4.0
#   if (defined(LIBMICSMM_USE_LIBXSMM) && defined(__LIBXSMM) && (0 < (LIBXSMM_ALIGNED_STORES))) || defined(LIBMICSMM_USE_XALIGN)
#   pragma omp simd aligned(c:1)
#   endif
#endif

#define LIBXSMM_MIN(A, B) ((A) < (B) ? (A) : (B))
#define LIBXSMM_MAX(A, B) ((A) < (B) ? (B) : (A))

#if defined(_WIN32) && !defined(__GNUC__)
# define LIBXSMM_ATTRIBUTE(A) __declspec(A)
# define LIBXSMM_ALIGNED(DECL, N) LIBXSMM_ATTRIBUTE(align(N)) DECL
#elif defined(__GNUC__)
# define LIBXSMM_ATTRIBUTE(A) __attribute__((A))
# define LIBXSMM_ALIGNED(DECL, N) DECL LIBXSMM_ATTRIBUTE(aligned(N))
#endif

#if defined(__INTEL_OFFLOAD) && (!defined(_WIN32) || (1400 <= __INTEL_COMPILER))
# define LIBXSMM_OFFLOAD 1
# define LIBXSMM_TARGET(A) LIBXSMM_ATTRIBUTE(target(A))
#else
/*# define LIBXSMM_OFFLOAD 0*/
# define LIBXSMM_TARGET(A)
#endif

#if defined(__INTEL_COMPILER)
# define LIBXSMM_ASSUME_ALIGNED(A, N) __assume_aligned(A, N)
# define LIBXSMM_ASSUME(EXPRESSION) __assume(EXPRESSION)
#else
# define LIBXSMM_ASSUME_ALIGNED(A, N)
# if defined(_MSC_VER)
#   define LIBXSMM_ASSUME(EXPRESSION) __assume(EXPRESSION)
# elif (40500 <= (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__))
#   define LIBXSMM_ASSUME(EXPRESSION) do { if (!(EXPRESSION)) __builtin_unreachable(); } while(0)
# endif
#endif
#define LIBXSMM_ALIGN_VALUE(DST_TYPE, SRC_TYPE, VALUE, ALIGNMENT) ((DST_TYPE)((-( \
  -((intptr_t)(VALUE) * ((intptr_t)sizeof(SRC_TYPE))) & \
  -((intptr_t)(LIBXSMM_MAX(ALIGNMENT, 1))))) / sizeof(SRC_TYPE)))
#define LIBXSMM_ALIGN(TYPE, PTR, ALIGNMENT) LIBXSMM_ALIGN_VALUE(TYPE, char, PTR, ALIGNMENT)

#if defined(LIBXSTREAM_OFFLOAD)
# pragma offload_attribute(push,target(mic))
# include <stdint.h>
# pragma offload_attribute(pop)
#else
# include <stdint.h>
#endif

#endif /*LIBXSMM_MACROS_H*/
