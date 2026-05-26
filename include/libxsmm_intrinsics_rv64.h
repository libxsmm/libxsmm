/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_UTILS_INTRINSICS_X86_H
#define LIBXSMM_UTILS_INTRINSICS_X86_H

#include "libxsmm_cpuid.h"

/** Macro evaluates to LIBXSMM_ATTRIBUTE_TARGET_xxx (see below). */
#define LIBXSMM_ATTRIBUTE_TARGET(TARGET) LIBXSMM_CONCATENATE(LIBXSMM_ATTRIBUTE_TARGET_, TARGET)

#if !defined(LIBXSMM_INTRINSICS_NONE) && !defined(LIBXSMM_PLATFORM_X86)
# define LIBXSMM_INTRINSICS_NONE
#endif
#if /*no intrinsics: tested with 17.x and 18.x*/(defined(__PGI) && \
    LIBXSMM_VERSION2(19, 0) > LIBXSMM_VERSION2(__PGIC__, __PGIC_MINOR__)) \
 || /*legacy*/(defined(_CRAYC) && !defined(__GNUC__))
# if !defined(LIBXSMM_INTRINSICS_NONE) && !defined(LIBXSMM_INTRINSICS_STATIC)
#   define LIBXSMM_INTRINSICS_NONE
# endif
#elif !defined(LIBXSMM_INTRINSICS_STATIC) && !defined(LIBXSMM_INTRINSICS_NONE) && ( \
      (defined(__GNUC__) && !defined(__clang__) && !defined(LIBXSMM_INTEL_COMPILER) && !defined(_CRAYC) && \
        LIBXSMM_VERSION2(4, 4) > LIBXSMM_VERSION2(__GNUC__, __GNUC_MINOR__)) /* GCC 4.4 (target-attribute) */ \
   || (defined(__clang__) && LIBXSMM_VERSION2(3, 7) > LIBXSMM_VERSION2(__clang_major__, __clang_minor__)) \
   || (defined(__APPLE__) && defined(__MACH__) && !defined(LIBXSMM_INTEL_COMPILER) && defined(__clang__) && \
        LIBXSMM_VERSION2(9, 0) > LIBXSMM_VERSION2(__clang_major__, __clang_minor__)))
# define LIBXSMM_INTRINSICS_STATIC
#endif


/** https://github.com/intel/Immintrin-debug */
#if !defined(LIBXSMM_INTRINSICS_DEBUG) && 0
# define LIBXSMM_INTRINSICS_DEBUG
/* workarounds removed after LIBXSMM 1.16.1-1.16.1-1268 */
# include "immintrin_dbg.h"
#endif
#if !defined(LIBXSMM_INTRINSICS_NONE)
# if    defined(__AVX512F__)  && defined(__AVX512CD__) \
   &&   defined(__AVX512DQ__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512VNNI__) && defined(__AVX512BF16__) \
   &&   defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__) \
   && (!defined(__GNUC__)  || defined(__clang__) || defined(LIBXSMM_INTEL_COMPILER) || defined(_CRAYC) /* TODO: check GCC, Clang, etc. */ \
                           || (LIBXSMM_VERSION2(10, 0) <= LIBXSMM_VERSION2(__GNUC__, __GNUC_MINOR__))) \
   && (!defined(__clang__) || (LIBXSMM_VERSION2( 9, 0) <= LIBXSMM_VERSION2(__clang_major__, __clang_minor__))) \
   && (!defined(__APPLE__) || !defined(__MACH__) || LIBXSMM_VERSION2(99, 0) <= LIBXSMM_VERSION2(__clang_major__, __clang_minor__))
#   if !defined(LIBXSMM_STATIC_TARGET_ARCH)
#     define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_CPX
#   endif
#   define LIBXSMM_INTRINSICS_INCLUDE
# elif  defined(__AVX512F__)  && defined(__AVX512CD__) \
   &&   defined(__AVX512DQ__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512VNNI__) \
   &&   defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__) \
   && (!defined(__GNUC__)  || defined(__clang__) || defined(LIBXSMM_INTEL_COMPILER) || defined(_CRAYC) \
                           || (LIBXSMM_VERSION2(8, 0) <= LIBXSMM_VERSION2(__GNUC__, __GNUC_MINOR__))) \
   && (!defined(__clang__) || (LIBXSMM_VERSION2(8, 1) <= LIBXSMM_VERSION2(__clang_major__, __clang_minor__))) \
   && (!defined(__APPLE__) || !defined(__MACH__) || LIBXSMM_VERSION2(10, 0) <= LIBXSMM_VERSION2(__clang_major__, __clang_minor__))
#   if !defined(LIBXSMM_STATIC_TARGET_ARCH)
#     define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_CLX
#   endif
#   define LIBXSMM_INTRINSICS_INCLUDE
# elif  defined(__AVX512F__)  && defined(__AVX512CD__) \
   &&   defined(__AVX512DQ__) && defined(__AVX512BW__) && defined(__AVX512VL__) \
   &&   defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__) \
   && (!defined(__GNUC__)  || defined(__clang__) || defined(LIBXSMM_INTEL_COMPILER) || defined(_CRAYC) \
                           || (LIBXSMM_VERSION2(8, 0) <= LIBXSMM_VERSION2(__GNUC__, __GNUC_MINOR__))) \
   && (!defined(__clang__) || (LIBXSMM_VERSION2(8, 1) <= LIBXSMM_VERSION2(__clang_major__, __clang_minor__))) \
   && (!defined(__APPLE__) || !defined(__MACH__) || LIBXSMM_VERSION2(9, 0) <= LIBXSMM_VERSION2(__clang_major__, __clang_minor__))
#   if !defined(LIBXSMM_STATIC_TARGET_ARCH)
#     define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_SKX
#   endif
#   define LIBXSMM_INTRINSICS_INCLUDE
# elif defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__)
#   if !defined(LIBXSMM_STATIC_TARGET_ARCH)
#     define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_AVX2
#   endif
#   define LIBXSMM_INTRINSICS_INCLUDE
# elif defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__)
#   if !defined(LIBXSMM_STATIC_TARGET_ARCH)
#     define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_AVX
#   endif
#   define LIBXSMM_INTRINSICS_INCLUDE
# elif defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__)
#   if !defined(LIBXSMM_STATIC_TARGET_ARCH)
#     define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_SSE42
#   endif
#   define LIBXSMM_INTRINSICS_INCLUDE
# elif defined(__SSE3__)
#   if !defined(LIBXSMM_STATIC_TARGET_ARCH)
#     define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_SSE3
#   endif
#   define LIBXSMM_INTRINSICS_INCLUDE
# elif defined(LIBXSMM_PLATFORM_X86)
#   if !defined(LIBXSMM_STATIC_TARGET_ARCH)
#     define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_GENERIC
#   endif
#   if defined(__GNUC__)
#     define LIBXSMM_INTRINSICS_INCLUDE
#   endif
# endif
# if defined(LIBXSMM_STATIC_TARGET_ARCH) && !defined(LIBXSMM_INTRINSICS_STATIC)
#   if defined(__INTEL_COMPILER)
#     if !defined(LIBXSMM_MAX_STATIC_TARGET_ARCH)
        /* TODO: compiler version check for LIBXSMM_MAX_STATIC_TARGET_ARCH */
#       if 1904 <= (LIBXSMM_INTEL_COMPILER) && !defined(_WIN32)
#         define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_CPX
#       elif 1801 <= (LIBXSMM_INTEL_COMPILER)
#         define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_CLX
#       elif 1500 <= (LIBXSMM_INTEL_COMPILER)
#         define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_SKX
#       else
#         define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX2
#       endif
#     endif
#     define LIBXSMM_INTRINSICS(TARGET)/*no need for target flags*/
#     define LIBXSMM_INTRINSICS_INCLUDE
#   elif defined(_CRAYC) && defined(__GNUC__)
      /* TODO: version check, e.g., LIBXSMM_VERSION2(11, 5) <= LIBXSMM_VERSION2(_RELEASE, _RELEASE_MINOR) */
#     if !defined(LIBXSMM_MAX_STATIC_TARGET_ARCH)
#       define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX
#     endif
#     define LIBXSMM_INTRINSICS(TARGET)/*no need for target flags*/
#     define LIBXSMM_INTRINSICS_INCLUDE
#   elif defined(_MSC_VER) && !defined(__clang__)
      /* TODO: compiler version check for LIBXSMM_MAX_STATIC_TARGET_ARCH */
#     if !defined(LIBXSMM_MAX_STATIC_TARGET_ARCH)
#       define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX2
#     endif
#     define LIBXSMM_INTRINSICS(TARGET)/*no need for target flags*/
#     define LIBXSMM_INTRINSICS_INCLUDE
#   elif (!defined(__GNUC__)  || LIBXSMM_VERSION2(4, 9) <= LIBXSMM_VERSION2(__GNUC__, __GNUC_MINOR__)) \
      && (!defined(__clang__) || LIBXSMM_VERSION2(4, 0) <= LIBXSMM_VERSION2(__clang_major__, __clang_minor__)) \
      && (!defined(__APPLE__) || !defined(__MACH__)) && !defined(__PGI) && !defined(_MSC_VER)
#     if !defined(LIBXSMM_MAX_STATIC_TARGET_ARCH)
#       if defined(__CYGWIN__) && !defined(LIBXSMM_INTRINSICS_DEBUG) /* Cygwin: invalid register for .seh_savexmm */
#         define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX2
#       elif (defined(__clang__) && LIBXSMM_VERSION2(10, 0) <= LIBXSMM_VERSION2(__clang_major__, __clang_minor__))
#         define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_CPX
#       elif (defined(__GNUC__)  && LIBXSMM_VERSION2(10, 0) <= LIBXSMM_VERSION2(__GNUC__, __GNUC_MINOR__)) \
          || (defined(__clang__) && LIBXSMM_VERSION2( 9, 0) <= LIBXSMM_VERSION2(__clang_major__, __clang_minor__) && !defined(__cray__))
#         define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_CPX
#       elif (defined(__GNUC__)  && LIBXSMM_VERSION2(8, 0) <= LIBXSMM_VERSION2(__GNUC__, __GNUC_MINOR__)) \
          || (defined(__clang__) && LIBXSMM_VERSION2(8, 1) <= LIBXSMM_VERSION2(__clang_major__, __clang_minor__))
#         define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_CLX
#       elif (defined(__GNUC__)  && LIBXSMM_VERSION2(8, 0) <= LIBXSMM_VERSION2(__GNUC__, __GNUC_MINOR__)) \
          || (defined(__clang__) && LIBXSMM_VERSION2(8, 1) <= LIBXSMM_VERSION2(__clang_major__, __clang_minor__))
#         define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_SKX
#       else
#         define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX2
#       endif
#     endif
#     define LIBXSMM_INTRINSICS_INCLUDE
#   else /* GCC/legacy incl. Clang */
#     if defined(__clang__) && !(defined(__APPLE__) && defined(__MACH__)) && !defined(_WIN32)
#       if (LIBXSMM_VERSION2(7, 0) <= LIBXSMM_VERSION2(__clang_major__, __clang_minor__)) /* TODO */
          /* no limitations */
#       elif (LIBXSMM_VERSION2(4, 0) <= LIBXSMM_VERSION2(__clang_major__, __clang_minor__))
#         if !defined(LIBXSMM_INTRINSICS_STATIC) && (LIBXSMM_STATIC_TARGET_ARCH < LIBXSMM_X86_AVX2/*workaround*/)
#           define LIBXSMM_INTRINSICS_STATIC
#         endif
#       elif !defined(LIBXSMM_INTRINSICS_STATIC)
#         define LIBXSMM_INTRINSICS_STATIC
#       endif
#       if !defined(LIBXSMM_MAX_STATIC_TARGET_ARCH)
#         if defined(__CYGWIN__) && !defined(LIBXSMM_INTRINSICS_DEBUG) /* Cygwin: invalid register for .seh_savexmm */
#           define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX2
#         elif LIBXSMM_VERSION2(10, 0) <= LIBXSMM_VERSION2(__clang_major__, __clang_minor__)
#           define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_CPX
#         elif LIBXSMM_VERSION2( 9, 0) <= LIBXSMM_VERSION2(__clang_major__, __clang_minor__) && !defined(__cray__)
#           define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_CPX
#         elif LIBXSMM_VERSION2( 8, 1) <= LIBXSMM_VERSION2(__clang_major__, __clang_minor__)
#           define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_CLX
#         else
#           define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_SKX
#         endif
#       endif
#     else /* fallback */
#       if !defined(LIBXSMM_MAX_STATIC_TARGET_ARCH)
#         define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_STATIC_TARGET_ARCH
#       endif
#       if !defined(LIBXSMM_INTRINSICS_STATIC) && (LIBXSMM_STATIC_TARGET_ARCH < LIBXSMM_X86_AVX2/*workaround*/)
#         define LIBXSMM_INTRINSICS_STATIC
#       endif
#     endif
#     if !defined(LIBXSMM_INTRINSICS_INCLUDE) && (!defined(__PGI) || LIBXSMM_VERSION2(19, 0) <= LIBXSMM_VERSION2(__PGIC__, __PGIC_MINOR__))
#       define LIBXSMM_INTRINSICS_INCLUDE
#     endif
#   endif /* GCC/legacy incl. Clang */
#   if !defined(LIBXSMM_MAX_STATIC_TARGET_ARCH)
#     error "LIBXSMM_MAX_STATIC_TARGET_ARCH not defined!"
#   endif
#   if defined(LIBXSMM_TARGET_ARCH) && (LIBXSMM_TARGET_ARCH < LIBXSMM_MAX_STATIC_TARGET_ARCH)
#     undef LIBXSMM_MAX_STATIC_TARGET_ARCH
#     define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_TARGET_ARCH
#   endif
#   if defined(LIBXSMM_INTRINSICS_INCLUDE) && !defined(LIBXSMM_INTRINSICS_NONE) && !defined(LIBXSMM_INTRINSICS_DEBUG)
#     include <immintrin.h>
#   endif /*defined(LIBXSMM_INTRINSICS_INCLUDE)*/
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
#       if (LIBXSMM_X86_SSE42 <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
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
#       if (LIBXSMM_X86_AVX512_SKX <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
#         define LIBXSMM_ATTRIBUTE_TARGET_1101 target("avx2,fma,avx512f,avx512cd,avx512dq,avx512bw,avx512vl")
#       else /* LIBXSMM_X86_AVX2 */
#         define LIBXSMM_ATTRIBUTE_TARGET_1101 LIBXSMM_ATTRIBUTE_TARGET_1006
#       endif
#       if (LIBXSMM_X86_AVX512_CLX <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
#         define LIBXSMM_ATTRIBUTE_TARGET_1102 target("avx2,fma,avx512f,avx512cd,avx512dq,avx512bw,avx512vl,avx512vnni")
#       else /* LIBXSMM_X86_AVX512_SKX */
#         define LIBXSMM_ATTRIBUTE_TARGET_1102 LIBXSMM_ATTRIBUTE_TARGET_1101
#       endif
#       if (LIBXSMM_X86_AVX512_CPX <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
#         define LIBXSMM_ATTRIBUTE_TARGET_1103 target("avx2,fma,avx512f,avx512cd,avx512dq,avx512bw,avx512vl,avx512vnni,avx512bf16")
#       else /* LIBXSMM_X86_AVX512_CLX */
#         define LIBXSMM_ATTRIBUTE_TARGET_1103 LIBXSMM_ATTRIBUTE_TARGET_1102
#       endif
#     else
#       define LIBXSMM_INTRINSICS(TARGET)/*no need for target flags*/
#     endif
#   elif !defined(LIBXSMM_INTRINSICS_TARGET)
#     define LIBXSMM_INTRINSICS_TARGET
#   endif /*!defined(LIBXSMM_INTRINSICS)*/
# endif /*defined(LIBXSMM_STATIC_TARGET_ARCH)*/
#endif /*!defined(LIBXSMM_INTRINSICS_NONE)*/

#if !defined(LIBXSMM_STATIC_TARGET_ARCH)
# if !defined(LIBXSMM_INTRINSICS_NONE) && !defined(LIBXSMM_INTRINSICS_STATIC)
#   define LIBXSMM_INTRINSICS_NONE
# endif
# define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_TARGET_ARCH_GENERIC
#endif

#if !defined(LIBXSMM_MAX_STATIC_TARGET_ARCH)
# define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_STATIC_TARGET_ARCH
#elif (LIBXSMM_MAX_STATIC_TARGET_ARCH < LIBXSMM_STATIC_TARGET_ARCH)
# undef LIBXSMM_MAX_STATIC_TARGET_ARCH
# define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_STATIC_TARGET_ARCH
#endif

#if !defined(LIBXSMM_INTRINSICS)
# define LIBXSMM_INTRINSICS(TARGET)
#endif

/** Include basic x86 intrinsics such as __rdtsc. */
#if defined(LIBXSMM_INTRINSICS_INCLUDE) && !defined(LIBXSMM_INTRINSICS_DEBUG)
# if defined(_WIN32)
#   include <intrin.h>
# elif defined(LIBXSMM_INTEL_COMPILER) || defined(_CRAYC) || defined(__clang__) || defined(__PGI)
#   include <x86intrin.h>
# elif defined(__GNUC__) && (LIBXSMM_VERSION2(4, 4) <= LIBXSMM_VERSION2(__GNUC__, __GNUC_MINOR__))
#   include <x86intrin.h>
# endif
# include <xmmintrin.h>
# if defined(__SSE3__)
#   include <pmmintrin.h>
# endif
#endif

#if !defined(LIBXSMM_INTRINSICS_NONE)
# if defined(_WIN32)
#   include <malloc.h>
# else
#   include <mm_malloc.h>
# endif
#endif

/**
 * Intrinsic-specific fix-ups
 */
# define LIBXSMM_INTRINSICS_LOADU_SI128(A) _mm_loadu_si128(A)
#if !defined(LIBXSMM_INTEL_COMPILER) && defined(__clang__) && ( \
      (LIBXSMM_VERSION2(3, 9) > LIBXSMM_VERSION2(__clang_major__, __clang_minor__)) \
   || (LIBXSMM_VERSION2(7, 3) > LIBXSMM_VERSION2(__clang_major__, __clang_minor__) && \
       defined(__APPLE__) && defined(__MACH__)))
/* prototypes with incorrect signature: _mm512_load_ps takes DP*, _mm512_load_pd takes SP* (checked with v3.8.1) */
# define LIBXSMM_INTRINSICS_MM512_LOAD_PS(A) _mm512_loadu_ps((const double*)(A))
# define LIBXSMM_INTRINSICS_MM512_LOAD_PD(A) _mm512_loadu_pd((const float*)(A))
/* Clang misses _mm512_stream_p? (checked with v3.8.1). */
# define LIBXSMM_INTRINSICS_MM512_STREAM_SI512(A, B) _mm512_store_si512(A, B)
# define LIBXSMM_INTRINSICS_MM512_STREAM_PS(A, B) _mm512_storeu_ps(A, B)
# define LIBXSMM_INTRINSICS_MM512_STREAM_PD(A, B) _mm512_store_pd(A, B)
#else
# define LIBXSMM_INTRINSICS_MM512_LOAD_PS(A) _mm512_loadu_ps((const float*)(A))
# define LIBXSMM_INTRINSICS_MM512_LOAD_PD(A) _mm512_loadu_pd((const double*)(A))
# define LIBXSMM_INTRINSICS_MM512_STREAM_SI512(A, B) _mm512_stream_si512((__m512i*)(A), (B))
# define LIBXSMM_INTRINSICS_MM512_STREAM_PS(A, B) _mm512_stream_ps(A, B)
# define LIBXSMM_INTRINSICS_MM512_STREAM_PD(A, B) _mm512_stream_pd(A, B)
#endif
#if !defined(LIBXSMM_INTEL_COMPILER) || (defined(__clang__) && ( \
      (LIBXSMM_VERSION2(8, 0) > LIBXSMM_VERSION2(__clang_major__, __clang_minor__)))) \
   || (defined(__APPLE__) && defined(__MACH__)) || defined(__GNUC__)
# define LIBXSMM_INTRINSICS_MM256_STORE_EPI32(A, B) _mm256_storeu_si256((__m256i*)(A), B)
#else
# define LIBXSMM_INTRINSICS_MM256_STORE_EPI32(A, B) _mm256_storeu_epi32(A, B)
#endif
#if defined(LIBXSMM_INTEL_COMPILER)
# if 1600 <= (LIBXSMM_INTEL_COMPILER)
#   define LIBXSMM_INTRINSICS_MM512_SET_EPI16(E31, E30, E29, E28, E27, E26, E25, E24, E23, E22, E21, E20, E19, E18, E17, E16, \
                                                        E15, E14, E13, E12, E11, E10, E9, E8, E7, E6, E5, E4, E3, E2, E1, E0) \
                             _mm512_set_epi16(E31, E30, E29, E28, E27, E26, E25, E24, E23, E22, E21, E20, E19, E18, E17, E16, \
                                                        E15, E14, E13, E12, E11, E10, E9, E8, E7, E6, E5, E4, E3, E2, E1, E0)
# else
#   define LIBXSMM_INTRINSICS_MM512_SET_EPI16(E31, E30, E29, E28, E27, E26, E25, E24, E23, E22, E21, E20, E19, E18, E17, E16, \
                                                        E15, E14, E13, E12, E11, E10, E9, E8, E7, E6, E5, E4, E3, E2, E1, E0) \
         _mm512_castps_si512(_mm512_set_epi16(E31, E30, E29, E28, E27, E26, E25, E24, E23, E22, E21, E20, E19, E18, E17, E16, \
                                                        E15, E14, E13, E12, E11, E10, E9, E8, E7, E6, E5, E4, E3, E2, E1, E0))
# endif
#else
# define LIBXSMM_INTRINSICS_MM512_SET_EPI16(E31, E30, E29, E28, E27, E26, E25, E24, E23, E22, E21, E20, E19, E18, E17, E16, \
                                                      E15, E14, E13, E12, E11, E10, E9, E8, E7, E6, E5, E4, E3, E2, E1, E0) \
               _mm512_set_epi32(((E31) << 16) | (E30), ((E29) << 16) | (E28), ((E27) << 16) | (E26), ((E25) << 16) | (E24), \
                                ((E23) << 16) | (E22), ((E21) << 16) | (E20), ((E19) << 16) | (E18), ((E17) << 16) | (E16), \
                                ((E15) << 16) | (E14), ((E13) << 16) | (E12), ((E11) << 16) | (E10),  ((E9) << 16) |  (E8), \
                                 ((E7) << 16) |  (E6),  ((E5) << 16) |  (E4),  ((E3) << 16) |  (E2),  ((E1) << 16) |  (E0))
#endif
#if (defined(LIBXSMM_INTEL_COMPILER) \
  || (defined(__GNUC__) && LIBXSMM_VERSION2(7, 0) <= LIBXSMM_VERSION2(__GNUC__, __GNUC_MINOR__)) \
  || (defined(__clang__) && (!defined(__APPLE__) || !defined(__MACH__)) \
      && LIBXSMM_VERSION2(4, 0) <= LIBXSMM_VERSION2(__clang_major__, __clang_minor__))) \
  && defined(NDEBUG) /* avoid warning "maybe-uninitialized" due to undefined value init */
# define LIBXSMM_INTRINSICS_MM512_MASK_I32GATHER_EPI32(A, B, C, D, E) _mm512_mask_i32gather_epi32(A, B, C, D, E)
# define LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(A, B) _mm512_extracti64x4_epi64(A, B)
# define LIBXSMM_INTRINSICS_MM512_ABS_PS(A) _mm512_abs_ps(A)
# define LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32() _mm512_undefined_epi32()
# define LIBXSMM_INTRINSICS_MM512_UNDEFINED() _mm512_undefined()
# define LIBXSMM_INTRINSICS_MM256_UNDEFINED_SI256() _mm256_undefined_si256()
# define LIBXSMM_INTRINSICS_MM_UNDEFINED_SI128() _mm_undefined_si128()
# define LIBXSMM_INTRINSICS_MM_UNDEFINED_PD() _mm_undefined_pd()
#else
# define LIBXSMM_INTRINSICS_MM512_MASK_I32GATHER_EPI32(A, B, C, D, E) _mm512_castps_si512(_mm512_mask_i32gather_ps( \
                           _mm512_castsi512_ps(A), B, C, (const float*)(D), E))
# define LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(A, B) _mm256_castpd_si256(_mm512_extractf64x4_pd(_mm512_castsi512_pd(A), B))
# define LIBXSMM_INTRINSICS_MM512_ABS_PS(A) _mm512_castsi512_ps(_mm512_and_epi32( \
                           _mm512_castps_si512(A), _mm512_set1_epi32(0x7FFFFFFF)))
# define LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32() _mm512_set1_epi32(0)
# define LIBXSMM_INTRINSICS_MM512_UNDEFINED() _mm512_set1_ps(0)
# define LIBXSMM_INTRINSICS_MM256_UNDEFINED_SI256() _mm256_set1_epi32(0)
# define LIBXSMM_INTRINSICS_MM_UNDEFINED_SI128() _mm_set1_epi32(0)
# define LIBXSMM_INTRINSICS_MM_UNDEFINED_PD() _mm_set1_pd(0)
#endif
#if (defined(LIBXSMM_INTEL_COMPILER) && (1800 <= (LIBXSMM_INTEL_COMPILER))) \
  || (!defined(LIBXSMM_INTEL_COMPILER) && defined(__GNUC__) \
      && LIBXSMM_VERSION2(7, 0) <= LIBXSMM_VERSION2(__GNUC__, __GNUC_MINOR__)) \
  || ((!defined(__APPLE__) || !defined(__MACH__)) && defined(__clang__) \
      && LIBXSMM_VERSION2(8, 0) <= LIBXSMM_VERSION2(__clang_major__, __clang_minor__))
# define LIBXSMM_INTRINSICS_MM512_STORE_MASK(DST_PTR, SRC, NBITS) \
    LIBXSMM_CONCATENATE(_store_mask, NBITS)((LIBXSMM_CONCATENATE(__mmask, NBITS)*)(DST_PTR), SRC)
# define LIBXSMM_INTRINSICS_MM512_LOAD_MASK(SRC_PTR, NBITS) \
    LIBXSMM_CONCATENATE(_load_mask, NBITS)((/*const*/ LIBXSMM_CONCATENATE(__mmask, NBITS)*)(SRC_PTR))
# define LIBXSMM_INTRINSICS_MM512_CVTU32_MASK(A, NBITS) LIBXSMM_CONCATENATE(_cvtu32_mask, NBITS)((unsigned int)(A))
#elif defined(LIBXSMM_INTEL_COMPILER)
# define LIBXSMM_INTRINSICS_MM512_STORE_MASK(DST_PTR, SRC, NBITS) \
    (*(LIBXSMM_CONCATENATE(__mmask, NBITS)*)(DST_PTR) = (LIBXSMM_CONCATENATE(__mmask, NBITS))(SRC))
# define LIBXSMM_INTRINSICS_MM512_LOAD_MASK(SRC_PTR, NBITS) \
    ((LIBXSMM_CONCATENATE(__mmask, NBITS))_mm512_mask2int(*(const __mmask16*)(SRC_PTR)))
# define LIBXSMM_INTRINSICS_MM512_CVTU32_MASK(A, NBITS) LIBXSMM_CONCATENATE(LIBXSMM_INTRINSICS_MM512_CVTU32_MASK_, NBITS)(A)
# define LIBXSMM_INTRINSICS_MM512_CVTU32_MASK_32(A) ((__mmask32)(A))
# define LIBXSMM_INTRINSICS_MM512_CVTU32_MASK_16(A) _mm512_int2mask((int)(A))
# define LIBXSMM_INTRINSICS_MM512_CVTU32_MASK_8(A) ((__mmask8)(A))
#else
# define LIBXSMM_INTRINSICS_MM512_STORE_MASK(DST_PTR, SRC, NBITS) \
    (*(LIBXSMM_CONCATENATE(__mmask, NBITS)*)(DST_PTR) = (LIBXSMM_CONCATENATE(__mmask, NBITS))(SRC))
# define LIBXSMM_INTRINSICS_MM512_LOAD_MASK(SRC_PTR, NBITS) (*(const LIBXSMM_CONCATENATE(__mmask, NBITS)*)(SRC_PTR))
# define LIBXSMM_INTRINSICS_MM512_CVTU32_MASK(A, NBITS) ((LIBXSMM_CONCATENATE(__mmask, NBITS))(A))
#endif
#define LIBXSMM_INTRINSICS_MM512_STORE_MASK64(DST_PTR, SRC) LIBXSMM_INTRINSICS_MM512_STORE_MASK(DST_PTR, SRC, 64)
#define LIBXSMM_INTRINSICS_MM512_STORE_MASK32(DST_PTR, SRC) LIBXSMM_INTRINSICS_MM512_STORE_MASK(DST_PTR, SRC, 32)
#define LIBXSMM_INTRINSICS_MM512_STORE_MASK16(DST_PTR, SRC) LIBXSMM_INTRINSICS_MM512_STORE_MASK(DST_PTR, SRC, 16)
#define LIBXSMM_INTRINSICS_MM512_STORE_MASK8(DST_PTR, SRC) LIBXSMM_INTRINSICS_MM512_STORE_MASK(DST_PTR, SRC, 8)
#define LIBXSMM_INTRINSICS_MM512_LOAD_MASK64(SRC_PTR) LIBXSMM_INTRINSICS_MM512_LOAD_MASK(SRC_PTR, 64)
#define LIBXSMM_INTRINSICS_MM512_LOAD_MASK32(SRC_PTR) LIBXSMM_INTRINSICS_MM512_LOAD_MASK(SRC_PTR, 32)
#define LIBXSMM_INTRINSICS_MM512_LOAD_MASK16(SRC_PTR) LIBXSMM_INTRINSICS_MM512_LOAD_MASK(SRC_PTR, 16)
#define LIBXSMM_INTRINSICS_MM512_LOAD_MASK8(SRC_PTR) LIBXSMM_INTRINSICS_MM512_LOAD_MASK(SRC_PTR, 8)
#define LIBXSMM_INTRINSICS_MM512_CVTU32_MASK32(A) LIBXSMM_INTRINSICS_MM512_CVTU32_MASK(A, 32)
#define LIBXSMM_INTRINSICS_MM512_CVTU32_MASK16(A) LIBXSMM_INTRINSICS_MM512_CVTU32_MASK(A, 16)
#define LIBXSMM_INTRINSICS_MM512_CVTU32_MASK8(A) LIBXSMM_INTRINSICS_MM512_CVTU32_MASK(A, 8)

/**
 * Pseudo intrinsics for portability
 */
LIBXSMM_API_INLINE int LIBXSMM_INTRINSICS_BITSCANFWD32_SW(unsigned int n) {
  unsigned int i, r = 0; if (0 != n) for (i = 1; 0 == (n & i); i <<= 1) { ++r; } return r;
}
LIBXSMM_API_INLINE int LIBXSMM_INTRINSICS_BITSCANFWD64_SW(unsigned long long n) {
  unsigned int i, r = 0; if (0 != n) for (i = 1; 0 == (n & i); i <<= 1) { ++r; } return r;
}

/** Binary Logarithm (based on Stackoverflow's NBITSx macro). */
#define LIBXSMM_INTRINSICS_BITSCANBWD_SW02(N) (0 != ((N) & 0x2/*0b10*/) ? 1 : 0)
#define LIBXSMM_INTRINSICS_BITSCANBWD_SW04(N) (0 != ((N) & 0xC/*0b1100*/) ? (2 | LIBXSMM_INTRINSICS_BITSCANBWD_SW02((N) >> 2)) : LIBXSMM_INTRINSICS_BITSCANBWD_SW02(N))
#define LIBXSMM_INTRINSICS_BITSCANBWD_SW08(N) (0 != ((N) & 0xF0/*0b11110000*/) ? (4 | LIBXSMM_INTRINSICS_BITSCANBWD_SW04((N) >> 4)) : LIBXSMM_INTRINSICS_BITSCANBWD_SW04(N))
#define LIBXSMM_INTRINSICS_BITSCANBWD_SW16(N) (0 != ((N) & 0xFF00) ? (8 | LIBXSMM_INTRINSICS_BITSCANBWD_SW08((N) >> 8)) : LIBXSMM_INTRINSICS_BITSCANBWD_SW08(N))
#define LIBXSMM_INTRINSICS_BITSCANBWD_SW32(N) (0 != ((N) & 0xFFFF0000) ? (16 | LIBXSMM_INTRINSICS_BITSCANBWD_SW16((N) >> 16)) : LIBXSMM_INTRINSICS_BITSCANBWD_SW16(N))
#define LIBXSMM_INTRINSICS_BITSCANBWD_SW64(N) (0 != ((N) & 0xFFFFFFFF00000000) ? (32 | LIBXSMM_INTRINSICS_BITSCANBWD_SW32((N) >> 32)) : LIBXSMM_INTRINSICS_BITSCANBWD_SW32(N))
#define LIBXSMM_INTRINSICS_BITSCANBWD32_SW(N) LIBXSMM_INTRINSICS_BITSCANBWD_SW32((unsigned int)(N))
#define LIBXSMM_INTRINSICS_BITSCANBWD64_SW(N) LIBXSMM_INTRINSICS_BITSCANBWD_SW64((unsigned long long)(N))

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
# define LIBXSMM_INTRINSICS_BITSCANFWD64 LIBXSMM_INTRINSICS_BITSCANFWD64_SW
# define LIBXSMM_INTRINSICS_BITSCANBWD64 LIBXSMM_INTRINSICS_BITSCANBWD64_SW
# endif
#elif defined(__GNUC__) && !defined(LIBXSMM_INTRINSICS_NONE)
# define LIBXSMM_INTRINSICS_BITSCANFWD32(N) ((0 != (N)) * __builtin_ctz(N))
# define LIBXSMM_INTRINSICS_BITSCANFWD64(N) ((0 != (N)) * __builtin_ctzll(N))
# define LIBXSMM_INTRINSICS_BITSCANBWD32(N) ((0 != (N)) * (31 - __builtin_clz(N)))
# define LIBXSMM_INTRINSICS_BITSCANBWD64(N) ((0 != (N)) * (63 - __builtin_clzll(N)))
#else /* fallback implementation */
# define LIBXSMM_INTRINSICS_BITSCANFWD32 LIBXSMM_INTRINSICS_BITSCANFWD32_SW
# define LIBXSMM_INTRINSICS_BITSCANFWD64 LIBXSMM_INTRINSICS_BITSCANFWD64_SW
# define LIBXSMM_INTRINSICS_BITSCANBWD32 LIBXSMM_INTRINSICS_BITSCANBWD32_SW
# define LIBXSMM_INTRINSICS_BITSCANBWD64 LIBXSMM_INTRINSICS_BITSCANBWD64_SW
#endif

/** LIBXSMM_NBITS determines the minimum number of bits needed to represent N. */
#define LIBXSMM_NBITS(N) (LIBXSMM_INTRINSICS_BITSCANBWD64(N) + LIBXSMM_MIN(1, N))
#define LIBXSMM_ISQRT2(N) ((unsigned int)((1ULL << (LIBXSMM_NBITS(N) >> 1)) /*+ LIBXSMM_MIN(1, N)*/))
/** LIBXSMM_ILOG2 definition matches ceil(log2(N)). */
LIBXSMM_API_INLINE unsigned int LIBXSMM_ILOG2(unsigned long long n) {
  unsigned int result = 0; if (1 < n) {
    const unsigned int m = LIBXSMM_INTRINSICS_BITSCANBWD64(n);
    result = m + ((unsigned int)LIBXSMM_INTRINSICS_BITSCANBWD64(n - 1) == m);
  } return result;
}

/**
 * Target attribution
 */
/** LIBXSMM_INTRINSICS_X86 is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXSMM_INTRINSICS_X86) && !defined(LIBXSMM_INTRINSICS_NONE) && (LIBXSMM_X86_GENERIC <= LIBXSMM_STATIC_TARGET_ARCH || \
   (!defined(LIBXSMM_INTRINSICS_STATIC) && LIBXSMM_X86_GENERIC <= LIBXSMM_MAX_STATIC_TARGET_ARCH))
# define LIBXSMM_INTRINSICS_X86
#endif
/** LIBXSMM_INTRINSICS_SSE3 is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXSMM_INTRINSICS_SSE3) && !defined(LIBXSMM_INTRINSICS_NONE) && (LIBXSMM_X86_SSE3 <= LIBXSMM_STATIC_TARGET_ARCH || \
   (!defined(LIBXSMM_INTRINSICS_STATIC) && LIBXSMM_X86_SSE3 <= LIBXSMM_MAX_STATIC_TARGET_ARCH))
# define LIBXSMM_INTRINSICS_SSE3
#endif
/** LIBXSMM_INTRINSICS_SSE42 is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXSMM_INTRINSICS_SSE42) && !defined(LIBXSMM_INTRINSICS_NONE) && (LIBXSMM_X86_SSE42 <= LIBXSMM_STATIC_TARGET_ARCH || \
   (!defined(LIBXSMM_INTRINSICS_STATIC) && LIBXSMM_X86_SSE42 <= LIBXSMM_MAX_STATIC_TARGET_ARCH))
# define LIBXSMM_INTRINSICS_SSE42
#endif
/** LIBXSMM_INTRINSICS_AVX is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXSMM_INTRINSICS_AVX) && !defined(LIBXSMM_INTRINSICS_NONE) && (LIBXSMM_X86_AVX <= LIBXSMM_STATIC_TARGET_ARCH || \
   (!defined(LIBXSMM_INTRINSICS_STATIC) && LIBXSMM_X86_AVX <= LIBXSMM_MAX_STATIC_TARGET_ARCH))
# define LIBXSMM_INTRINSICS_AVX
#endif
/** LIBXSMM_INTRINSICS_AVX2 is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXSMM_INTRINSICS_AVX2) && !defined(LIBXSMM_INTRINSICS_NONE) && (LIBXSMM_X86_AVX2 <= LIBXSMM_STATIC_TARGET_ARCH || \
   (!defined(LIBXSMM_INTRINSICS_STATIC) && LIBXSMM_X86_AVX2 <= LIBXSMM_MAX_STATIC_TARGET_ARCH))
# define LIBXSMM_INTRINSICS_AVX2
#endif
/** LIBXSMM_INTRINSICS_AVX512_SKX is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXSMM_INTRINSICS_AVX512_SKX) && !defined(LIBXSMM_INTRINSICS_NONE) && (LIBXSMM_X86_AVX512_SKX <= LIBXSMM_STATIC_TARGET_ARCH || \
   (!defined(LIBXSMM_INTRINSICS_STATIC) && LIBXSMM_X86_AVX512_SKX <= LIBXSMM_MAX_STATIC_TARGET_ARCH))
# define LIBXSMM_INTRINSICS_AVX512_SKX
#endif
/** LIBXSMM_INTRINSICS_AVX512_CLX is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXSMM_INTRINSICS_AVX512_CLX) && !defined(LIBXSMM_INTRINSICS_NONE) && (LIBXSMM_X86_AVX512_CLX <= LIBXSMM_STATIC_TARGET_ARCH || \
   (!defined(LIBXSMM_INTRINSICS_STATIC) && LIBXSMM_X86_AVX512_CLX <= LIBXSMM_MAX_STATIC_TARGET_ARCH))
# define LIBXSMM_INTRINSICS_AVX512_CLX
#endif
/** LIBXSMM_INTRINSICS_AVX512_CPX is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXSMM_INTRINSICS_AVX512_CPX) && !defined(LIBXSMM_INTRINSICS_NONE) && defined(LIBXSMM_X86_AVX512_CPX) && \
    !defined(LIBXSMM_INTRINSICS_STATIC) && (LIBXSMM_X86_AVX512_CPX <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
# define LIBXSMM_INTRINSICS_AVX512_CPX
#endif

/** 2048-bit state for xoshiro128+ RNG (state/symbols needed even if AVX-512 is not used) */
#define LIBXSMM_INTRINSICS_MM512_RNG_STATE(INDEX) (*(__m512i*)LIBXSMM_CONCATENATE(libxsmm_intrinsics_mm512_rng_state, INDEX))
LIBXSMM_APIVAR_PUBLIC(unsigned int libxsmm_intrinsics_mm512_rng_state0[16]);
LIBXSMM_APIVAR_PUBLIC(unsigned int libxsmm_intrinsics_mm512_rng_state1[16]);
LIBXSMM_APIVAR_PUBLIC(unsigned int libxsmm_intrinsics_mm512_rng_state2[16]);
LIBXSMM_APIVAR_PUBLIC(unsigned int libxsmm_intrinsics_mm512_rng_state3[16]);

/**
 * Pseudo intrinsics (AVX-512)
 */
#if defined(LIBXSMM_INTRINSICS_AVX512_SKX) /*__AVX512F__*/
#define MAX_COEFF (16)
#define INIT_COEFF(c, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15) \
  (c[0] = c0; c[1] = c1; c[2] = c2; c[3] = c3; c[4] = c4; c[5] = c5; c[6] = c6; c[7] = c7;\
   c[8] = c8; c[9] = c9; c[10] = c10; c[11] = c11; c[12] = c12; c[13] = c13; c[14] = c14; c[15] = c15;)

void get_gelu_fwd_minimax3(float *xin, float *xout, uint32_t *consts, uint32_t* coeffc0, uint32_t *coeffc1, uint32_t *coeffc2)
{
  uint32_t *threas  = consts;
  uint32_t *absmask = consts + 1;
  uint32_t *scale   = consts + 2;
  uint32_t *shifter = consts + 3;
  uint32_t *half    = consts + 4;
  asm volatile (
      "vsetivli t3, 16, e32, m2, tu, mu\n"
      "lw x28, 0(%[threas])\n"
      "lw x29, 0(%[absmask])\n"
      "lw x30, 0(%[scale])\n"
      "lw x31, 0(%[shifter])\n"
      "lw x5,  0(%[half])\n"
      "vmv.v.x v2, x28\n"           // threas
      "vmv.v.x v4, x29\n"           // absmask
      "vmv.v.x v6, x30\n"           // scale
      "vmv.v.x v8, x31\n"           // shifter
      "vmv.v.x v10, x5\n"           // half
      "vle32.v v12, 0(%[coeffc0])\n"// Load coefficient c2
      "vle32.v v14, 0(%[coeffc1])\n"// Load coefficient c1
      "vle32.v v16, 0(%[coeffc2])\n"// Load coefficient c0
      "vle32.v v0, 0(%[xin])\n"     // load x
      "vfabs.v v18, v0\n"           // Get abs in
      "vfabs.v v20, v2\n"           // Get abs thres
      "vfmin.vv v22, v18, v20\n"    // Round = xr
      "vfsgnj.vv v24, v0, v22\n"    // Inject sign from x
      "vand.vv v26, v4, v24\n"      // Add   = xa
      "vmv.v.v v28, v26\n"          // Copy xa
      "vfmadd.vv v28, v6, v8\n"     // fmadd = index
      "vand.vi v30, v28, 0xf\n"     // get index bits
      "vrgather.vv v18, v12, v30\n" // Permute c0
      "vrgather.vv v20, v14, v30\n" // Permute c1
      "vrgather.vv v22, v16, v30\n" // Permute c2
      "vfmadd.vv v18, v26, v20\n"   // Get poly step 1
      "vfmadd.vv v18, v26, v22\n"   // Get poly step 2
      "vfmadd.vv v18, v24, v10\n"   // Get poly step 3
      "vfmul.vv v18, v0, v18\n"     // Mulitply input
      "vse32.v v18, 0(%[xout])\n"   // Store output
      :
      : [threas]"r"(threas), [absmask]"r"(absmask), [scale]"r"(scale), [shifter]"r"(shifter), [half]"r"(half), [coeffc2]"r"(coeffc2), [coeffc1]"r"(coeffc1), [coeffc0]"r"(coeffc0), [xin]"r"(xin), [xout]"r"(xout)
      :"t3", "x5", "x28", "x29", "x30", "x31", "x6"
        );
}

#if defined(LIBXSMM_INTRINSICS_RVV) /*__AVX512DQ__ needed*/
LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_RISCV_RVV) char* LIBXSMM_INTRINSICS_RVV_GELU_FWD_PS_MINIMAX3(uint32_t *xin, uint32_t *xout) {
  uint32_t * coeffc0 = malloc(sizeof(uint32_t) * MAX_COEFF);
  uint32_t * coeffc1 = malloc(sizeof(uint32_t) * MAX_COEFF);
  uint32_t * coeffc2 = malloc(sizeof(uint32_t) * MAX_COEFF);

  *test = 0;

  INIT_COEFF(0xbd877b85u, 0xbd7d9780u, 0xbd4cb70eu, 0xbd08a1e9u, 0xbc808857u, 0xb9476fd2u, 0x3c36f765u, 0x3c924160u, 0x3ca7b1fcu, 0x3ca5732cu, 0x3c95af63u, 0x3c8079f7u, 0x3c55fa4fu, 0x3c2fa86bu, 0x3c0fbb00u, 0x3bec178cu);
  INIT_COEFF(0xb7c7fb58u, 0xbacb9740u, 0xbc3e4b3au, 0xbd0d292au, 0xbd8bc5d0u, 0xbdd9978fu, 0xbe0f92d3u, 0xbe27b66du, 0xbe328ce7u, 0xbe3125bfu, 0xbe26dc9du, 0xbe17a056u, 0xbe06bdebu, 0xbdecc593u, 0xbdcf57aau, 0xbdb5ea3au);
  INIT_COEFF(0x3ecc4231u, 0x3ecc541cu, 0x3ecd6c48u, 0x3ed174c3u, 0x3ed9bd5du, 0x3ee5acd5u, 0x3ef2aeddu, 0x3efd5384u, 0x3f016724u, 0x3f00f778u, 0x3efb389eu, 0x3ef0464du, 0x3ee3014fu, 0x3ed50a78u, 0x3ec779dbu, 0x3ebae363u);

  get_gelu_fwd_minimax3(xin, xout, coffc0, coffc1, coffc2);

  free(coffc0);
  free(coffc1);
  free(coffc2);

  return xout;
}

#undef MAX_COEFF
#undef INIT_COEFF

#endif /*defined(LIBXSMM_INTRINSICS_RVV)*/

#endif /*__AVX512F__*/

#endif /*LIBXSMM_UTILS_INTRINSICS_X86_H*/
