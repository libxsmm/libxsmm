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
#ifndef LIBXSMM_INTRINSICS_X86_H
#define LIBXSMM_INTRINSICS_X86_H

#include "libxsmm_cpuid.h"

/** Macro evaluates to LIBXSMM_ATTRIBUTE_TARGET_xxx (see below). */
#define LIBXSMM_ATTRIBUTE_TARGET(TARGET) LIBXSMM_CONCATENATE(LIBXSMM_ATTRIBUTE_TARGET_, TARGET)

#if !defined(LIBXSMM_INTRINSICS_STATIC) && /* GCC 4.4 (target-attribute) */ \
  (defined(__GNUC__) && (LIBXSMM_VERSION3(4, 4, 0) > LIBXSMM_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)) && !defined(__clang__)) || \
  (defined(__clang__) && LIBXSMM_VERSION3(3, 7, 0) > LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)) || \
  (defined(__APPLE__) && defined(__MACH__) && !defined(__INTEL_COMPILER))
# define LIBXSMM_INTRINSICS_STATIC
#endif

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif

/** PGI's intrinsic headers do not compile, __SSE4_x__/__AVX__ etc. are never defined (-tp=haswell, etc.) */
#if !defined(LIBXSMM_INTRINSICS_NONE) && defined(__PGI)
# define LIBXSMM_INTRINSICS_NONE
#endif

#if defined(__MIC__) && !defined(LIBXSMM_INTRINSICS_NONE)
# define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_IMCI
# define LIBXSMM_INTRINSICS(TARGET)
# define LIBXSMM_INTRINSICS_INCLUDE
#elif !defined(LIBXSMM_INTRINSICS_NONE) /*!defined(__MIC__)*/
# if    defined(__AVX512F__)  && defined(__AVX512CD__) \
   &&   defined(__AVX512DQ__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512VNNI__) \
   &&   defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__) \
   && !(defined(__APPLE__) && defined(__MACH__)) \
   && (!defined(__clang__) || ((LIBXSMM_VERSION3(3, 9, 0) <= LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)) \
   || (LIBXSMM_VERSION3(0, 0, 0) == LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))))
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_ICL
# elif    defined(__AVX512F__)  && defined(__AVX512CD__) \
   &&   defined(__AVX512DQ__) && defined(__AVX512BW__) && defined(__AVX512VL__) \
   &&   defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__) \
   && !(defined(__APPLE__) && defined(__MACH__)) \
   && (!defined(__clang__) || ((LIBXSMM_VERSION3(3, 9, 0) <= LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)) \
   || (LIBXSMM_VERSION3(0, 0, 0) == LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))))
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_CORE
# elif  defined(__AVX512F__) && defined(__AVX512CD__) \
   &&   defined(__AVX512PF__) && defined(__AVX512ER__) \
   &&   defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__) \
   && !(defined(__APPLE__) && defined(__MACH__)) \
   && (!defined(__clang__) || ((LIBXSMM_VERSION3(3, 5, 0) <= LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)) \
   || (LIBXSMM_VERSION3(0, 0, 0) == LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))))
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_MIC
# elif  defined(__AVX512F__) && defined(__AVX512CD__) \
   &&   defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__) \
   && !(defined(__APPLE__) && defined(__MACH__)) \
   && (!defined(__clang__) || ((LIBXSMM_VERSION3(3, 5, 0) <= LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)) \
   || (LIBXSMM_VERSION3(0, 0, 0) == LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))))
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512
# elif defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__)
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_AVX2
# elif defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__)
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_AVX
# elif defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__)
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_SSE4
# elif defined(__SSE3__)
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_SSE3
# elif defined(__x86_64__) || defined(_WIN32) || defined(_WIN64)
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_GENERIC
# endif
# if defined(LIBXSMM_STATIC_TARGET_ARCH) && !defined(LIBXSMM_INTRINSICS_STATIC)
#   if defined(__INTEL_COMPILER)
      /* TODO: compiler version check for LIBXSMM_MAX_STATIC_TARGET_ARCH */
#     if 1500 <= (__INTEL_COMPILER)
#       define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_CORE
#     elif 1400 <= (__INTEL_COMPILER)
#       define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_MIC
#     else
#       define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX2
#     endif
#     define LIBXSMM_INTRINSICS(TARGET)/*no need for target flags*/
#     define LIBXSMM_INTRINSICS_INCLUDE
#     include <immintrin.h>
#   elif defined(_CRAYC) && defined(__GNUC__)
      /* TODO: version check e.g., LIBXSMM_VERSION2(11, 5) <= LIBXSMM_VERSION2(_RELEASE, _RELEASE_MINOR) */
#     define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX
#     define LIBXSMM_INTRINSICS(TARGET)/*no need for target flags*/
#     define LIBXSMM_INTRINSICS_INCLUDE
#     include <immintrin.h>
#   elif defined(_MSC_VER) && !defined(__clang__)
      /* TODO: compiler version check for LIBXSMM_MAX_STATIC_TARGET_ARCH */
#     define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX2
#     define LIBXSMM_INTRINSICS(TARGET)/*no need for target flags*/
#     define LIBXSMM_INTRINSICS_INCLUDE
#     include <immintrin.h>
#   elif (LIBXSMM_VERSION3(5, 1, 0) <= LIBXSMM_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)) && !defined(__PGI)
      /* AVX-512 pseudo intrinsics are missing e.g., reductions */
#     if !defined(LIBXSMM_INTRINSICS_AVX512_NOREDUCTIONS)
#       define LIBXSMM_INTRINSICS_AVX512_NOREDUCTIONS
#     endif
#     if !defined(__CYGWIN__)
#       define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_CORE
#     else /* Error: invalid register for .seh_savexmm */
#       define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX2
#     endif
#     define LIBXSMM_INTRINSICS_INCLUDE
#     include <immintrin.h>
#   elif (LIBXSMM_VERSION3(4, 9, 0) <= LIBXSMM_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)) && !defined(__PGI)
      /* AVX-512 pseudo intrinsics are missing e.g., reductions */
#     if !defined(LIBXSMM_INTRINSICS_AVX512_NOREDUCTIONS)
#       define LIBXSMM_INTRINSICS_AVX512_NOREDUCTIONS
#     endif
#     if !defined(__CYGWIN__)
#       define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_MIC
#     else /* Error: invalid register for .seh_savexmm */
#       define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX2
#     endif
#     define LIBXSMM_INTRINSICS_INCLUDE
#     include <immintrin.h>
#   else /* GCC/legacy incl. Clang */
#     if defined(__clang__) && !(defined(__APPLE__) && defined(__MACH__))
#       if (LIBXSMM_VERSION3(0, 0, 0) == LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)) /* devel */ || \
           (LIBXSMM_VERSION3(7, 0, 0) <= LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)) /* TODO */
          /* no limitations */
#       elif (LIBXSMM_VERSION3(4, 0, 0) <= LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))
#         if !defined(LIBXSMM_INTRINSICS_STATIC) && (LIBXSMM_STATIC_TARGET_ARCH < LIBXSMM_X86_AVX2/*workaround*/)
#           define LIBXSMM_INTRINSICS_STATIC
#         endif
#       elif !defined(LIBXSMM_INTRINSICS_STATIC)
#         define LIBXSMM_INTRINSICS_STATIC
#       endif
#       if !defined(__CYGWIN__)
#         define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_MIC
#       else /* Error: invalid register for .seh_savexmm */
#         define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX2
#       endif
#     elif (defined(__GNUC__) && LIBXSMM_VERSION3(4, 7, 0) <= LIBXSMM_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)) || defined(__clang__)
#       define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX2
#       if !defined(LIBXSMM_INTRINSICS_STATIC) && (LIBXSMM_STATIC_TARGET_ARCH < LIBXSMM_X86_AVX2/*workaround*/)
#         define LIBXSMM_INTRINSICS_STATIC
#       endif
#     elif (defined(__GNUC__) && LIBXSMM_VERSION3(4, 4, 0) <= LIBXSMM_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))
#       define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX
#       if !defined(LIBXSMM_INTRINSICS_STATIC) && (LIBXSMM_STATIC_TARGET_ARCH < LIBXSMM_X86_AVX2/*workaround*/)
#         define LIBXSMM_INTRINSICS_STATIC
#       endif
#     else /* fall-back */
#       define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_STATIC_TARGET_ARCH
#       if !defined(LIBXSMM_INTRINSICS_STATIC) && (LIBXSMM_STATIC_TARGET_ARCH < LIBXSMM_X86_AVX2/*workaround*/)
#         define LIBXSMM_INTRINSICS_STATIC
#       endif
#       if !defined(LIBXSMM_INTRINSICS_NONE) && !defined(__PGI)
#         define LIBXSMM_INTRINSICS_NONE
#       endif
#     endif
#     if !defined(LIBXSMM_INTRINSICS_INCLUDE) && !defined(__PGI)
#       define LIBXSMM_INTRINSICS_INCLUDE
#     endif
#     if defined(LIBXSMM_INTRINSICS_INCLUDE) && !defined(LIBXSMM_INTRINSICS_NONE)
#       if !defined(__SSE3__)
#         define __SSE3__ 1
#       endif
#       if !defined(__SSSE3__)
#         define __SSSE3__ 1
#       endif
#       if !defined(__SSE4_1__)
#         define __SSE4_1__ 1
#       endif
#       if !defined(__SSE4_2__)
#         define __SSE4_2__ 1
#       endif
#       if !defined(__AVX__)
#         define __AVX__ 1
#       endif
#       if !defined(__AVX2__)
#         define __AVX2__ 1
#       endif
#       if !defined(__FMA__)
#         define __FMA__ 1
#       endif
#       if !defined(__AVX512F__)
#         define __AVX512F__ 1
#       endif
#       if !defined(__AVX512CD__)
#         define __AVX512CD__ 1
#       endif
#       if !defined(__AVX512PF__)
#         define __AVX512PF__ 1
#       endif
#       if !defined(__AVX512ER__)
#         define __AVX512ER__ 1
#       endif
#       if !defined(__AVX512DQ__)
#         define __AVX512DQ__ 1
#       endif
#       if !defined(__AVX512BW__)
#         define __AVX512BW__ 1
#       endif
#       if !defined(__AVX512VL__)
#         define __AVX512VL__ 1
#       endif
#       if !defined(__AVX512VNNI__)
#         define __AVX512VNNI__ 1
#       endif
#       if defined(__GNUC__) && !defined(__clang__)
#         pragma GCC push_options
#         if (LIBXSMM_X86_AVX < LIBXSMM_MAX_STATIC_TARGET_ARCH)
#           pragma GCC target("avx2,fma")
#         else
#           pragma GCC target("avx")
#         endif
#       endif
#       include <immintrin.h>
#       if defined(__GNUC__) && !defined(__clang__)
#         pragma GCC pop_options
#       endif
#       if (LIBXSMM_X86_SSE3 > (LIBXSMM_STATIC_TARGET_ARCH))
#         undef __SSE3__
#       endif
#       if (LIBXSMM_X86_SSE4 > (LIBXSMM_STATIC_TARGET_ARCH))
#         undef __SSSE3__
#         undef __SSE4_1__
#         undef __SSE4_2__
#       endif
#       if (LIBXSMM_X86_AVX > (LIBXSMM_STATIC_TARGET_ARCH))
#         undef __AVX__
#       endif
#       if (LIBXSMM_X86_AVX2 > (LIBXSMM_STATIC_TARGET_ARCH))
#         undef __AVX2__
#         undef __FMA__
#       endif
#       if (LIBXSMM_X86_AVX512 > (LIBXSMM_STATIC_TARGET_ARCH))
#         undef __AVX512F__
#         undef __AVX512CD__
#       endif
#       if (LIBXSMM_X86_AVX512_MIC > (LIBXSMM_STATIC_TARGET_ARCH))
#         undef __AVX512F__
#         undef __AVX512CD__
#         undef __AVX512PF__
#         undef __AVX512ER__
#       endif
#       if (LIBXSMM_X86_AVX512_CORE > (LIBXSMM_STATIC_TARGET_ARCH))
#         undef __AVX512F__
#         undef __AVX512CD__
#         undef __AVX512DQ__
#         undef __AVX512BW__
#         undef __AVX512VL__
#       endif
#       if (LIBXSMM_X86_AVX512_ICL > (LIBXSMM_STATIC_TARGET_ARCH))
#         undef __AVX512VNNI__
#       endif
#     endif /*defined(LIBXSMM_INTRINSICS_INCLUDE)*/
#   endif /* GCC/legacy incl. Clang */
#   if !defined(LIBXSMM_MAX_STATIC_TARGET_ARCH)
#     error "LIBXSMM_MAX_STATIC_TARGET_ARCH not defined!"
#   endif
#   if !defined(LIBXSMM_INTRINSICS)
#     if (LIBXSMM_MAX_STATIC_TARGET_ARCH > LIBXSMM_STATIC_TARGET_ARCH)
#       define LIBXSMM_INTRINSICS(TARGET) LIBXSMM_ATTRIBUTE(LIBXSMM_ATTRIBUTE_TARGET(TARGET))
        /* LIBXSMM_ATTRIBUTE_TARGET_xxx is required to literally match the CPUID (libxsmm_cpuid.h)! */
#       define LIBXSMM_ATTRIBUTE_TARGET_1002 target("sse2") /* LIBXSMM_X86_GENERIC (64-bit ABI) */
#       if (LIBXSMM_X86_SSE3 <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
#         define LIBXSMM_ATTRIBUTE_TARGET_1003 target("sse3")
#       else
#         define LIBXSMM_ATTRIBUTE_TARGET_1003 LIBXSMM_ATTRIBUTE_TARGET_1002
#       endif
#       if (LIBXSMM_X86_SSE4 <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
#         define LIBXSMM_ATTRIBUTE_TARGET_1004 target("sse4.1,sse4.2")
#       else
#         define LIBXSMM_ATTRIBUTE_TARGET_1004 LIBXSMM_ATTRIBUTE_TARGET_1003
#       endif
#       if (LIBXSMM_X86_AVX <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
#         define LIBXSMM_ATTRIBUTE_TARGET_1005 target("avx")
#       else
#         define LIBXSMM_ATTRIBUTE_TARGET_1005 LIBXSMM_ATTRIBUTE_TARGET_1004
#       endif
#       if (LIBXSMM_X86_AVX2 <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
#         define LIBXSMM_ATTRIBUTE_TARGET_1006 target("avx2,fma")
#       else
#         define LIBXSMM_ATTRIBUTE_TARGET_1006 LIBXSMM_ATTRIBUTE_TARGET_1005
#       endif
#       if (LIBXSMM_X86_AVX512 <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
#         define LIBXSMM_ATTRIBUTE_TARGET_1007 target("avx2,fma,avx512f,avx512cd")
#       else
#         define LIBXSMM_ATTRIBUTE_TARGET_1007 LIBXSMM_ATTRIBUTE_TARGET_1006
#       endif
#       if (LIBXSMM_X86_AVX512_MIC <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
#         define LIBXSMM_ATTRIBUTE_TARGET_1010 target("avx2,fma,avx512f,avx512cd,avx512pf,avx512er")
#       else /* LIBXSMM_X86_AVX512 */
#         define LIBXSMM_ATTRIBUTE_TARGET_1010 LIBXSMM_ATTRIBUTE_TARGET_1007
#       endif
#       if (LIBXSMM_X86_AVX512_KNM <= LIBXSMM_MAX_STATIC_TARGET_ARCH) /* TODO: add compiler flags */
#         define LIBXSMM_ATTRIBUTE_TARGET_1011 target("avx2,fma,avx512f,avx512cd,avx512pf,avx512er")
#       else /* LIBXSMM_X86_AVX512_MIC */
#         define LIBXSMM_ATTRIBUTE_TARGET_1011 LIBXSMM_ATTRIBUTE_TARGET_1010
#       endif
#       if (LIBXSMM_X86_AVX512_CORE <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
#         define LIBXSMM_ATTRIBUTE_TARGET_1020 target("avx2,fma,avx512f,avx512cd,avx512dq,avx512bw,avx512vl")
#       else /* LIBXSMM_X86_AVX512 */
#         define LIBXSMM_ATTRIBUTE_TARGET_1020 LIBXSMM_ATTRIBUTE_TARGET_1007
#       endif
#       if (LIBXSMM_X86_AVX512_ICL <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
#         define LIBXSMM_ATTRIBUTE_TARGET_1022 target("avx2,fma,avx512f,avx512cd,avx512dq,avx512bw,avx512vl,avx512vnni")
#       else /* LIBXSMM_X86_AVX512_CORE */
#         define LIBXSMM_ATTRIBUTE_TARGET_1022 LIBXSMM_ATTRIBUTE_TARGET_1020
#       endif
#     else
#       define LIBXSMM_INTRINSICS(TARGET)/*no need for target flags*/
#     endif
#   endif /*!defined(LIBXSMM_INTRINSICS)*/
# elif defined(LIBXSMM_STATIC_TARGET_ARCH) && !defined(LIBXSMM_INTRINSICS_INCLUDE)
#   define LIBXSMM_INTRINSICS_INCLUDE
#   include <immintrin.h>
# endif /*defined(LIBXSMM_STATIC_TARGET_ARCH)*/
#endif /*!defined(LIBXSMM_INTRINSICS_NONE)*/

#if !defined(LIBXSMM_STATIC_TARGET_ARCH)
# define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_TARGET_ARCH_GENERIC
#endif

#if !defined(LIBXSMM_MAX_STATIC_TARGET_ARCH)
# define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_STATIC_TARGET_ARCH
#endif

/** Include basic x86 intrinsics such as __rdtsc. */
#if defined(LIBXSMM_INTRINSICS_INCLUDE)
# if defined(_WIN32)
#   include <intrin.h>
# else
#   include <x86intrin.h>
# endif
# include <xmmintrin.h>
# if defined(__SSE3__)
#   include <pmmintrin.h>
# endif
#endif

#if !defined(LIBXSMM_INTRINSICS)
# if !defined(LIBXSMM_INTRINSICS_NONE)
#   define LIBXSMM_INTRINSICS_NONE
# endif
# define LIBXSMM_INTRINSICS(TARGET)
#endif

#if !defined(LIBXSMM_INTRINSICS_NONE)
# if defined(_WIN32)
#   include <malloc.h>
# else
#   include <mm_malloc.h>
# endif
#endif

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

/** Intrinsic-specific fix-ups */
#if defined(__clang__)
# define LIBXSMM_INTRINSICS_LDDQU_SI128(A) _mm_loadu_si128(A)
#else
# define LIBXSMM_INTRINSICS_LDDQU_SI128(A) _mm_lddqu_si128(A)
#endif
#if defined(__clang__) && ( \
      (LIBXSMM_VERSION3(3, 9, 0)  > LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__) && \
       LIBXSMM_VERSION3(0, 0, 0) != LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)) \
   || (LIBXSMM_VERSION3(7, 3, 0)  > LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__) && \
       defined(__APPLE__) && defined(__MACH__)))
/* prototypes with incorrect signature: _mm512_load_ps takes DP*, _mm512_load_pd takes SP* (checked with v3.8.1) */
# define LIBXSMM_INTRINSICS_MM512_LOAD_PS(A) _mm512_load_ps((const double*)(A))
# define LIBXSMM_INTRINSICS_MM512_LOAD_PD(A) _mm512_load_pd((const float*)(A))
/* Clang misses _mm512_stream_p? (checked with v3.8.1). */
# define LIBXSMM_INTRINSICS_MM512_STREAM_SI512(A, B) _mm512_store_si512(A, B)
# define LIBXSMM_INTRINSICS_MM512_STREAM_PS(A, B) _mm512_store_ps(A, B)
# define LIBXSMM_INTRINSICS_MM512_STREAM_PD(A, B) _mm512_store_pd(A, B)
#else
# define LIBXSMM_INTRINSICS_MM512_LOAD_PS(A) _mm512_load_ps((const float*)(A))
# define LIBXSMM_INTRINSICS_MM512_LOAD_PD(A) _mm512_load_pd((const double*)(A))
# define LIBXSMM_INTRINSICS_MM512_STREAM_SI512(A, B) _mm512_stream_si512((__m512i*)(A), B)
# define LIBXSMM_INTRINSICS_MM512_STREAM_PS(A, B) _mm512_stream_ps(A, B)
# define LIBXSMM_INTRINSICS_MM512_STREAM_PD(A, B) _mm512_stream_pd(A, B)
#endif
#if defined(__INTEL_COMPILER)
# define LIBXSMM_INTRINSICS_MM512_SET_EPI16(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, \
                                  A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28, A29, A30, A31) \
                           _mm512_set_epi16(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, \
                                  A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28, A29, A30, A31)
# define LIBXSMM_INTRINSICS_MM512_MASK_I32GATHER_EPI32(A, B, C, D, E) _mm512_mask_i32gather_epi32(A, B, C, D, E)
# define LIBXSMM_INTRINSICS_MM512_EXTRACTI64x4_EPI64(A, B) _mm512_extracti64x4_epi64(A, B)
# define LIBXSMM_INTRINSICS_MM512_PERMUTEVAR_EPI32(A, B) _mm512_permutexvar_epi32(A, B)
# define LIBXSMM_INTRINSICS_MM512_ABS_PS(A) _mm512_abs_ps(A)
# define LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32() _mm512_undefined_epi32()
# define LIBXSMM_INTRINSICS_MM512_UNDEFINED() _mm512_undefined()
# define LIBXSMM_INTRINSICS_MM_UNDEFINED_PD() _mm_undefined_pd()
#else
# define LIBXSMM_INTRINSICS_MM512_SET_EPI16(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, \
                                  A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28, A29, A30, A31) \
                           _mm512_set_epi32((A0) |  ((A1) << 16),  (A2) |  ((A3) << 16),  (A4) |  ((A5) << 16),  (A6) |  ((A7) << 16), \
                                            (A8) |  ((A9) << 16), (A10) | ((A11) << 16), (A12) | ((A13) << 16), (A14) | ((A15) << 16), \
                                           (A16) | ((A17) << 16), (A18) | ((A19) << 16), (A20) | ((A21) << 16), (A22) | ((A23) << 16), \
                                           (A24) | ((A25) << 16), (A26) | ((A27) << 16), (A28) | ((A29) << 16), (A30) | ((A31) << 16))
# define LIBXSMM_INTRINSICS_MM512_MASK_I32GATHER_EPI32(A, B, C, D, E) _mm512_castps_si512(_mm512_mask_i32gather_ps( \
                           _mm512_castsi512_ps(A), B, C, (const float*)(D), E))
# define LIBXSMM_INTRINSICS_MM512_EXTRACTI64x4_EPI64(A, B) _mm256_castpd_si256(_mm512_extractf64x4_pd(_mm512_castsi512_pd(A), B))
# define LIBXSMM_INTRINSICS_MM512_PERMUTEVAR_EPI32(A, B) _mm512_permutexvar_epi32(A, B)
# define LIBXSMM_INTRINSICS_MM512_ABS_PS(A) _mm512_castsi512_ps(_mm512_and_epi32( \
                           _mm512_castps_si512(A), _mm512_set1_epi32(0x7FFFFFFF)))
# define LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32() _mm512_set1_epi32(0)
# define LIBXSMM_INTRINSICS_MM512_UNDEFINED() _mm512_set1_ps(0)
# define LIBXSMM_INTRINSICS_MM_UNDEFINED_PD() _mm_set1_pd(0)
#endif

LIBXSMM_API_INLINE int LIBXSMM_INTRINSICS_BITSCANFWD32_SW(unsigned int n) {
  unsigned int i, r = 0; if (0 != n) for (i = 1; 0 == (n & i); i <<= 1) { ++r; } return r;
}
LIBXSMM_API_INLINE int LIBXSMM_INTRINSICS_BITSCANFWD64_SW(unsigned long long n) {
  unsigned int i, r = 0; if (0 != n) for (i = 1; 0 == (n & i); i <<= 1) { ++r; } return r;
}
#define LIBXSMM_INTRINSICS_BITSCANBWD32_SW(N) LIBXSMM_LOG2_32((unsigned int)(N))
#define LIBXSMM_INTRINSICS_BITSCANBWD64_SW(N) LIBXSMM_LOG2_64((unsigned long long)(N))

#if defined(_WIN32) && !defined(LIBXSMM_INTRINSICS_NONE)
  LIBXSMM_API_INLINE unsigned int LIBXSMM_INTRINSICS_BITSCANFWD32(unsigned int n) {
    unsigned long r = 0; _BitScanForward(&r, n); return (0 != n) * r;
  }
  LIBXSMM_API_INLINE unsigned int LIBXSMM_INTRINSICS_BITSCANBWD32(unsigned int n) {
    unsigned long r = 0; _BitScanReverse(&r, n); return r;
  }
# if defined(_WIN64)
    LIBXSMM_API_INLINE unsigned int LIBXSMM_INTRINSICS_BITSCANFWD64(unsigned long long n) {
      unsigned long r = 0; _BitScanForward64(&r, n); return (0 != n) * r;
    }
    LIBXSMM_API_INLINE unsigned int LIBXSMM_INTRINSICS_BITSCANBWD64(unsigned long long n) {
      unsigned long r = 0; _BitScanReverse64(&r, n); return r;
    }
# else
#   define LIBXSMM_INTRINSICS_BITSCANFWD64 LIBXSMM_INTRINSICS_BITSCANFWD64_SW
#   define LIBXSMM_INTRINSICS_BITSCANBWD64 LIBXSMM_INTRINSICS_BITSCANBWD64_SW
# endif
#elif defined(__GNUC__) && !defined(LIBXSMM_INTRINSICS_NONE)
# define LIBXSMM_INTRINSICS_BITSCANFWD32(N) ((0 != (N)) * __builtin_ctz(N))
# define LIBXSMM_INTRINSICS_BITSCANFWD64(N) ((0 != (N)) * __builtin_ctzll(N))
# define LIBXSMM_INTRINSICS_BITSCANBWD32(N) ((0 != (N)) * (31 - __builtin_clz(N)))
# define LIBXSMM_INTRINSICS_BITSCANBWD64(N) ((0 != (N)) * (63 - __builtin_clzll(N)))
#else /* fall-back implementation */
# define LIBXSMM_INTRINSICS_BITSCANFWD32 LIBXSMM_INTRINSICS_BITSCANFWD32_SW
# define LIBXSMM_INTRINSICS_BITSCANFWD64 LIBXSMM_INTRINSICS_BITSCANFWD64_SW
# define LIBXSMM_INTRINSICS_BITSCANBWD32 LIBXSMM_INTRINSICS_BITSCANBWD32_SW
# define LIBXSMM_INTRINSICS_BITSCANBWD64 LIBXSMM_INTRINSICS_BITSCANBWD64_SW
#endif

#if !defined(LIBXSMM_INTRINSICS_KNC) && !defined(LIBXSMM_INTRINSICS_NONE) && defined(__MIC__)
# define LIBXSMM_INTRINSICS_KNC
#endif

/** LIBXSMM_INTRINSICS_X86 is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXSMM_INTRINSICS_X86) && !defined(LIBXSMM_INTRINSICS_NONE) && \
    !defined(LIBXSMM_INTRINSICS_STATIC) && (LIBXSMM_X86_GENERIC <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
# define LIBXSMM_INTRINSICS_X86
#endif

/** LIBXSMM_INTRINSICS_SSE3 is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXSMM_INTRINSICS_SSE3) && !defined(LIBXSMM_INTRINSICS_NONE) && defined(LIBXSMM_INTRINSICS_X86) && \
    !defined(LIBXSMM_INTRINSICS_STATIC) && (LIBXSMM_X86_SSE3 <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
# define LIBXSMM_INTRINSICS_SSE3
#endif

/** LIBXSMM_INTRINSICS_SSE4 is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXSMM_INTRINSICS_SSE4) && !defined(LIBXSMM_INTRINSICS_NONE) && defined(LIBXSMM_INTRINSICS_SSE3) && \
    !defined(LIBXSMM_INTRINSICS_STATIC) && (LIBXSMM_X86_SSE4 <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
# define LIBXSMM_INTRINSICS_SSE4
#endif

/** LIBXSMM_INTRINSICS_AVX is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXSMM_INTRINSICS_AVX) && !defined(LIBXSMM_INTRINSICS_NONE) && defined(LIBXSMM_INTRINSICS_SSE4) && \
    !defined(LIBXSMM_INTRINSICS_STATIC) && (LIBXSMM_X86_AVX <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
# define LIBXSMM_INTRINSICS_AVX
#endif

/** LIBXSMM_INTRINSICS_AVX2 is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXSMM_INTRINSICS_AVX2) && !defined(LIBXSMM_INTRINSICS_NONE) && defined(LIBXSMM_INTRINSICS_AVX) && \
    !defined(LIBXSMM_INTRINSICS_STATIC) && (LIBXSMM_X86_AVX2 <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
# define LIBXSMM_INTRINSICS_AVX2
#endif

/** LIBXSMM_INTRINSICS_AVX512 is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXSMM_INTRINSICS_AVX512) && !defined(LIBXSMM_INTRINSICS_NONE) && defined(LIBXSMM_INTRINSICS_AVX2) && \
    !defined(LIBXSMM_INTRINSICS_STATIC) && (LIBXSMM_X86_AVX512 <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
# define LIBXSMM_INTRINSICS_AVX512
#endif

/** LIBXSMM_INTRINSICS_AVX512_MIC is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXSMM_INTRINSICS_AVX512_MIC) && !defined(LIBXSMM_INTRINSICS_NONE) && defined(LIBXSMM_INTRINSICS_AVX512) && \
    !defined(LIBXSMM_INTRINSICS_STATIC) && (LIBXSMM_X86_AVX512_MIC <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
# define LIBXSMM_INTRINSICS_AVX512_MIC
#endif

/** LIBXSMM_INTRINSICS_AVX512_KNM is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXSMM_INTRINSICS_AVX512_KNM) && !defined(LIBXSMM_INTRINSICS_NONE) && defined(LIBXSMM_INTRINSICS_AVX512_MIC) && \
    !defined(LIBXSMM_INTRINSICS_STATIC) && (LIBXSMM_X86_AVX512_KNM <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
# define LIBXSMM_INTRINSICS_AVX512_KNM
#endif

/** LIBXSMM_INTRINSICS_AVX512_CORE is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXSMM_INTRINSICS_AVX512_CORE) && !defined(LIBXSMM_INTRINSICS_NONE) && defined(LIBXSMM_INTRINSICS_AVX512) && \
    !defined(LIBXSMM_INTRINSICS_STATIC) && (LIBXSMM_X86_AVX512_CORE <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
# define LIBXSMM_INTRINSICS_AVX512_CORE
#endif

/** LIBXSMM_INTRINSICS_AVX512_ICL is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXSMM_INTRINSICS_AVX512_ICL) && !defined(LIBXSMM_INTRINSICS_NONE) && defined(LIBXSMM_INTRINSICS_AVX512_CORE) && \
    !defined(LIBXSMM_INTRINSICS_STATIC) && (LIBXSMM_X86_AVX512_ICL <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
# define LIBXSMM_INTRINSICS_AVX512_ICL
#endif

#endif /*LIBXSMM_INTRINSICS_X86_H*/

