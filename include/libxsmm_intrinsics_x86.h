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
 * Pseudo intrinsics (AVX-2)
 */
#if defined(LIBXSMM_INTRINSICS_AVX2) /*__AVX2__*/
# if defined(__GNUC__) && !defined(__clang__) && !defined(LIBXSMM_INTEL_COMPILER) && !defined(_CRAYC) && 0
LIBXSMM_PRAGMA_OPTIMIZE_OFF /* avoid ICE in case of symbols (-g) */
# endif
/** Generate random number in the interval [0, 1); thread save, state needs to be managed by user.
 *  this is based on xoshiro128+ 1.0, e.g. http://prng.di.unimi.it/xoshiro128plus.c */
LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX2) __m256i LIBXSMM_INTRINSICS_MM256_RNG_XOSHIRO128P_EXTSTATE_EPI32(unsigned int* stateptr) {
  __m256i state_0 = _mm256_loadu_si256( (const __m256i*)stateptr      );
  __m256i state_1 = _mm256_loadu_si256( (const __m256i*)(stateptr+16) );
  __m256i state_2 = _mm256_loadu_si256( (const __m256i*)(stateptr+32) );
  __m256i state_3 = _mm256_loadu_si256( (const __m256i*)(stateptr+48) );
  const __m256i result = _mm256_add_epi32(state_0, state_3);
  const __m256i s = _mm256_slli_epi32(state_1, 9);
  __m256i t;
  state_2 = _mm256_xor_si256(state_2, state_0);
  state_3 = _mm256_xor_si256(state_3, state_1);
  state_1 = _mm256_xor_si256(state_1, state_2);
  state_0 = _mm256_xor_si256(state_0, state_3);
  state_2 = _mm256_xor_si256(state_2, s);
  _mm256_storeu_si256( (__m256i*)stateptr   , state_0 );
  _mm256_storeu_si256( (__m256i*)(stateptr+16), state_1 );
  _mm256_storeu_si256( (__m256i*)(stateptr+32), state_2 );
  t = _mm256_slli_epi32(state_3, 11);
  state_3 = _mm256_or_si256(t, _mm256_srli_epi32(state_3, 32 - 11));
  _mm256_storeu_si256( (__m256i*)(stateptr+48), state_3 );
  return result;
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX2) __m256 LIBXSMM_INTRINSICS_MM256_RNG_EXTSTATE_PS(unsigned int* stateptr) {
  const __m256i rng_mantissa = _mm256_srli_epi32( LIBXSMM_INTRINSICS_MM256_RNG_XOSHIRO128P_EXTSTATE_EPI32(stateptr), 9 );
  const __m256 one = _mm256_set1_ps(1.0f);
  return _mm256_sub_ps(_mm256_castsi256_ps(_mm256_or_si256(_mm256_set1_epi32(0x3f800000), rng_mantissa)), one);
}
# if defined(__GNUC__) && !defined(__clang__) && !defined(LIBXSMM_INTEL_COMPILER) && !defined(_CRAYC) && 0
LIBXSMM_PRAGMA_OPTIMIZE_ON
# endif
#endif /*__AVX2__*/

/**
 * Pseudo intrinsics (AVX-512)
 */
#if defined(LIBXSMM_INTRINSICS_AVX512_SKX) /*__AVX512F__*/
# define LIBXSMM_INTRINSICS_MM512_QUANTIZE_NEAR_PS_EPI16( A, B ) _mm512_cvtepi32_epi16(_mm512_cvt_roundps_epi32( \
    _mm512_mul_ps(LIBXSMM_INTRINSICS_MM512_LOAD_PS(A), B), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_SKX) __m512i LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(__m512 a) {
  const __m512i vnaninf = _mm512_set1_epi32(0x7f800000), vrneadd = _mm512_set1_epi32(0x00007fff);
  const __m512i vfixup = _mm512_set1_epi32(0x00000001), vfixupmask = _mm512_set1_epi32(0x00010000);
  const __m512i mm512_roundbf16rne_a_ = _mm512_castps_si512(a);
  const __mmask16 mm512_roundbf16rne_mask1_ = _mm512_cmp_epi32_mask(_mm512_and_epi32(mm512_roundbf16rne_a_, vnaninf), vnaninf, _MM_CMPINT_NE);
  const __mmask16 mm512_roundbf16rne_mask2_ = _mm512_cmp_epi32_mask(_mm512_and_epi32(mm512_roundbf16rne_a_, vfixupmask), vfixupmask, _MM_CMPINT_EQ);
  return _mm512_mask_add_epi32(mm512_roundbf16rne_a_, mm512_roundbf16rne_mask1_, mm512_roundbf16rne_a_, _mm512_mask_add_epi32(vrneadd, mm512_roundbf16rne_mask2_, vrneadd, vfixup));
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_SKX) __m256i LIBXSMM_INTRINSICS_MM512_CVT_FP32_BF16(__m512 a) {
  return _mm512_cvtepi32_epi16(_mm512_srai_epi32(LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(a), 16));
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_SKX) __m512i LIBXSMM_INTRINSICS_MM512_CVT2_FP32_BF16(__m512 a, __m512 b) {
  const __m256i aa = _mm512_cvtepi32_epi16(_mm512_srai_epi32(LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(b), 16));
  const __m256i bb = _mm512_cvtepi32_epi16(_mm512_srai_epi32(LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(a), 16));
  return _mm512_inserti64x4(_mm512_inserti64x4(_mm512_setzero_si512(), aa, 0), bb, 1);
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_SKX) __m512 LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(__m256i a) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(a),16));
}

/** SVML-intrinsics */
LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_SKX) __m512 LIBXSMM_INTRINSICS_MM512_TANH_PS_RATIONAL_78(__m512 x) {
  const  __m512 c0        = _mm512_set1_ps(2027025.0f);
  const  __m512 c1        = _mm512_set1_ps(270270.0f);
  const  __m512 c2        = _mm512_set1_ps(6930.0f);
  const  __m512 c3        = _mm512_set1_ps(36.0f);
  const  __m512 c1_d      = _mm512_set1_ps(945945.0f);
  const  __m512 c2_d      = _mm512_set1_ps(51975.0f);
  const  __m512 c3_d      = _mm512_set1_ps(630.0f);
  const  __m512 hi_bound  = _mm512_set1_ps(4.97f);
  const  __m512 lo_bound  = _mm512_set1_ps(-4.97f);
  const  __m512 ones      = _mm512_set1_ps(1.0f);
  const  __m512 neg_ones  = _mm512_set1_ps(-1.0f);

  const __m512 x2         = _mm512_mul_ps( x, x );
  const __m512 t1_nom     = _mm512_fmadd_ps( c3, x2, c2 );
  const __m512 t2_nom     = _mm512_fmadd_ps( t1_nom, x2, c1 );
  const __m512 t3_nom     = _mm512_fmadd_ps( t2_nom, x2, c0 );
  const __m512 nom        = _mm512_mul_ps( t3_nom, x );
  const __m512 t1_denom   = _mm512_add_ps( x2, c3_d );
  const __m512 t2_denom   = _mm512_fmadd_ps( t1_denom, x2, c2_d );
  const __m512 t3_denom   = _mm512_fmadd_ps( t2_denom, x2, c1_d );
  const __m512 denom      = _mm512_fmadd_ps( t3_denom, x2, c0 );
  const __m512 denom_rcp  = _mm512_rcp14_ps( denom );
  const __mmask16 mask_hi = _mm512_cmp_ps_mask( x, hi_bound, _CMP_GT_OQ);
  const __mmask16 mask_lo = _mm512_cmp_ps_mask( x, lo_bound, _CMP_LT_OQ);
  __m512 result           = _mm512_mul_ps( nom, denom_rcp );
  result                  = _mm512_mask_blend_ps(mask_hi, result, ones);
  result                  = _mm512_mask_blend_ps(mask_lo, result, neg_ones);

  return result;
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_SKX) __m512 LIBXSMM_INTRINSICS_MM512_TANH_PS_RATIONAL_32(__m512 x) {
  const  __m512 c1        = _mm512_set1_ps((float)(1.0/27.0));
  const  __m512 c2        = _mm512_set1_ps((float)(1.0/3));
  const  __m512 hi_bound  = _mm512_set1_ps(3.2f);
  const  __m512 lo_bound  = _mm512_set1_ps(-3.2f);
  const  __m512 ones      = _mm512_set1_ps(1.0f);
  const  __m512 neg_ones  = _mm512_set1_ps(-1.0f);

  const __m512 x2         = _mm512_mul_ps( x, x );
  const __m512 t1_nom     = _mm512_fmadd_ps( x2, c1, ones);
  const __m512 nom        = _mm512_mul_ps( t1_nom, x );
  const __m512 denom      = _mm512_fmadd_ps( x2, c2, ones);
  const __m512 denom_rcp  = _mm512_rcp14_ps( denom );
  const __mmask16 mask_hi = _mm512_cmp_ps_mask( x, hi_bound, _CMP_GT_OQ);
  const __mmask16 mask_lo = _mm512_cmp_ps_mask( x, lo_bound, _CMP_LT_OQ);
  __m512 result           = _mm512_mul_ps(nom, denom_rcp);
  result                  = _mm512_mask_blend_ps(mask_hi, result, ones);
  result                  = _mm512_mask_blend_ps(mask_lo, result, neg_ones);

  return result;
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_SKX) __m512 LIBXSMM_INTRINSICS_MM512_TANH_PS_EXP2(__m512 _x) {
  const __m512 twice_log2_e = _mm512_set1_ps((float)(1.442695*2));
  const __m512 half       = _mm512_set1_ps(0.5f);
  const __m512 c2         = _mm512_set1_ps(0.240226507f);
  const __m512 c1         = _mm512_set1_ps(0.452920674f);
  const __m512 c0         = _mm512_set1_ps(0.713483036f);
  const __m512 ones       = _mm512_set1_ps(1.0f);
  const __m512 minus_twos = _mm512_set1_ps(-2.0f);

  const __m512 x          = _mm512_fmadd_ps(_x, twice_log2_e, half);
#if 1
  const __m512 y          = _mm512_sub_ps(x, _mm512_roundscale_round_ps(x, 1, _MM_FROUND_CUR_DIRECTION));
#else
  const __m512 y          = _mm512_reduce_ps(x, 1);
#endif
  const __m512 t1         = _mm512_fmadd_ps( y, c2, c1);
  const __m512 two_to_y   = _mm512_fmadd_ps( y, t1, c0);
  const __m512 exp        = _mm512_scalef_ps( two_to_y, x );
  const __m512 denom_rcp  = _mm512_rcp14_ps( _mm512_add_ps( exp, ones) );
  __m512 result     = _mm512_fmadd_ps( denom_rcp, minus_twos, ones);

 return result;
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_SKX) __m512 LIBXSMM_INTRINSICS_MM512_TANH_PS_EXP3(__m512 _x) {
  const __m512 twice_log2_e = _mm512_set1_ps((float)(1.442695*2));
  const __m512 half       = _mm512_set1_ps(0.5f);
  const __m512 c3         = _mm512_set1_ps(0.05550410866f);
  const __m512 c2         = _mm512_set1_ps(0.15697034396f);
  const __m512 c1         = _mm512_set1_ps(0.49454875509f);
  const __m512 c0         = _mm512_set1_ps(0.70654502287f);
  const __m512 ones       = _mm512_set1_ps(1.0f);
  const __m512 minus_twos = _mm512_set1_ps(-2.0f);

  const __m512 x          = _mm512_fmadd_ps(_x, twice_log2_e, half);
#if 1
  const __m512 y          = _mm512_sub_ps(x, _mm512_roundscale_round_ps(x, 1, _MM_FROUND_CUR_DIRECTION));
#else
  const __m512 y          = _mm512_reduce_ps(x, 1);
#endif
  const __m512 t1         = _mm512_fmadd_ps( y, c3, c2);
  const __m512 t2         = _mm512_fmadd_ps( y, t1, c1);
  const __m512 two_to_y   = _mm512_fmadd_ps( y, t2, c0);
  const __m512 exp        = _mm512_scalef_ps( two_to_y, x );
  const __m512 denom_rcp  = _mm512_rcp14_ps( _mm512_add_ps( exp, ones) );
  __m512 result     = _mm512_fmadd_ps( denom_rcp, minus_twos, ones);

  return result;
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_SKX) __m512 LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2(__m512 x) {
  __m512 result, func_p0, func_p1, func_p2;
  const __m512i sign_mask = _mm512_set1_epi32( 0x80000000 );
  const __m512i sign_filter = _mm512_set1_epi32( 0x7FFFFFFF );
  const __m512i lut_low = _mm512_set1_epi32( 246 );
  const __m512i lut_high = _mm512_set1_epi32( 261 );
  const __m512 tanh_p0_2_reg = _mm512_set_ps( 0.40555000f,  0.11892800f, -0.00972979f, -0.02740300f, -0.0169851f, -0.00776152f, -0.00305889f,
                                             -0.00116259f, -0.00041726f, -8.53233e-6f,  1.0000000f,  0.99999800f,  0.99975400f,  0.99268200f,
                                              0.93645300f,  0.73833900f);
  const __m512 tanh_p1_2_reg = _mm512_set_ps( 0.495602f, 0.88152f, 1.125700000f, 1.17021000f, 1.1289000000f, 1.07929000f, 1.0432300f, 1.023010f,
                                              1.011620f, 1.00164f, 1.56828e-14f, 4.49924e-7f, 0.0000646924f, 0.00260405f, 0.0311608f, 0.168736f);
  const __m512 tanh_p2_2_reg = _mm512_set_ps(-0.108238f, -0.2384280f, -0.354418000f, -0.38240300f, -0.34135700f, -0.274509000f, -0.20524900f, -0.1511960f,
                                             -0.107635f, -0.0466868f, -3.60822e-16f, -2.05971e-8f, -4.24538e-6f, -0.000231709f, -0.00386434f, -0.0277702f);

  const __m512i signs   = _mm512_and_epi32(_mm512_castps_si512(x), sign_mask);
  const __m512i abs_arg = _mm512_and_epi32(_mm512_castps_si512(x), sign_filter);
  __m512i indices       = _mm512_srli_epi32(abs_arg, 22);
  indices               = _mm512_max_epi32(indices, lut_low);
  indices               = _mm512_min_epi32(indices, lut_high);

  func_p0               = _mm512_permutexvar_ps(indices, tanh_p0_2_reg);
  func_p1               = _mm512_permutexvar_ps(indices, tanh_p1_2_reg);
  func_p2               = _mm512_permutexvar_ps(indices, tanh_p2_2_reg);

  result                = _mm512_fmadd_ps(_mm512_castsi512_ps(abs_arg), func_p2, func_p1);
  result                = _mm512_fmadd_ps(_mm512_castsi512_ps(abs_arg), result, func_p0);
  result                = _mm512_castsi512_ps(_mm512_xor_epi32(_mm512_castps_si512(result), signs));

  return result;
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_SKX) __m512 LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX3(__m512 x) {
  __m512 result, func_p0, func_p1, func_p2, func_p3;
  const __m512i sign_mask = _mm512_set1_epi32( 0x80000000 );
  const __m512i sign_filter = _mm512_set1_epi32( 0x7FFFFFFF );
  const __m512i lut_low = _mm512_set1_epi32( 246 );
  const __m512i lut_high = _mm512_set1_epi32( 261 );

  const __m512 tanh_p0_3_reg = _mm512_setr_ps( 0.466283000f,  0.82850600f,  0.97437500f,  0.99882600f,  0.9999860f,  1.0000000f, -1.50006e-08f, -7.98169e-06f,
                                              -4.53753e-05f, -0.00023755f, -0.00125285f, -0.00572314f, -0.0227717f, -0.0629089f, -0.084234300f,  0.071199800f);
  const __m512 tanh_p1_3_reg = _mm512_setr_ps( 0.500617f, 0.124369f, 0.0137214f, 0.000464124f, 4.02465e-06f, 0.00000f, 1.00001f, 1.00028f, 1.00112f, 1.00414f,
                                               1.015570f, 1.050950f, 1.1478500f, 1.310130000f, 1.378950000f, 1.07407f);
  const __m512 tanh_p2_3_reg = _mm512_setr_ps(-0.16133200f, -0.0305526f, -0.00245909f, -6.12647e-05f, -3.76127e-07f,  0.000000f, -0.000245872f, -0.00341151f,
                                              -0.00971505f, -0.0256817f, -0.06869110f, -0.162433000f, -0.346828000f, -0.566516f, -0.640214000f, -0.44011900f);
  const __m512 tanh_p3_3_reg = _mm512_setr_ps( 0.0177393f,  0.00253432f,  0.000147303f,  2.69963e-06f, 1.16764e-08f, 0.0000000f, -0.330125f, -0.3176210f,
                                              -0.3017760f, -0.27358000f, -0.219375000f, -0.136197000f, -0.01868680f, 0.0808901f,  0.107095f,  0.0631459f);

  const __m512i signs   = _mm512_and_epi32(_mm512_castps_si512(x), sign_mask);
  const __m512i abs_arg = _mm512_and_epi32(_mm512_castps_si512(x), sign_filter);
  __m512i indices       = _mm512_srli_epi32(abs_arg, 22);
  indices               = _mm512_max_epi32(indices, lut_low);
  indices               = _mm512_min_epi32(indices, lut_high);

  func_p0               = _mm512_permutexvar_ps(indices, tanh_p0_3_reg);
  func_p1               = _mm512_permutexvar_ps(indices, tanh_p1_3_reg);
  func_p2               = _mm512_permutexvar_ps(indices, tanh_p2_3_reg);
  func_p3               = _mm512_permutexvar_ps(indices, tanh_p3_3_reg);

  result                = _mm512_fmadd_ps(_mm512_castsi512_ps(abs_arg), func_p3, func_p2);
  result                = _mm512_fmadd_ps(_mm512_castsi512_ps(abs_arg), result, func_p1);
  result                = _mm512_fmadd_ps(_mm512_castsi512_ps(abs_arg), result, func_p0);
  result                = _mm512_castsi512_ps(_mm512_xor_epi32(_mm512_castps_si512(result), signs));

  return result;
}

#if defined(LIBXSMM_INTRINSICS_AVX512_SKX) /*__AVX512DQ__ needed*/
LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_SKX) __m512 LIBXSMM_INTRINSICS_MM512_GELU_FWD_PS_MINIMAX3(__m512 x) {
  const  __m512 thres   = _mm512_castsi512_ps(_mm512_set1_epi32(0x40879fff));
  const  __m512 absmask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
  const  __m512 scale   = _mm512_castsi512_ps(_mm512_set1_epi32(0x406a0ea1));
  const  __m512 shifter = _mm512_castsi512_ps(_mm512_set1_epi32(0x4b400000));
  const  __m512 half    = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f000000));
  const  __m512 _c2     = _mm512_castsi512_ps(_mm512_setr_epi32(0xbd877b85u, 0xbd7d9780u, 0xbd4cb70eu, 0xbd08a1e9u, 0xbc808857u, 0xb9476fd2u, 0x3c36f765u, 0x3c924160u,
                                                                0x3ca7b1fcu, 0x3ca5732cu, 0x3c95af63u, 0x3c8079f7u, 0x3c55fa4fu, 0x3c2fa86bu, 0x3c0fbb00u, 0x3bec178cu));
  const  __m512 _c1     = _mm512_castsi512_ps(_mm512_setr_epi32(0xb7c7fb58u, 0xbacb9740u, 0xbc3e4b3au, 0xbd0d292au, 0xbd8bc5d0u, 0xbdd9978fu, 0xbe0f92d3u, 0xbe27b66du,
                                                                0xbe328ce7u, 0xbe3125bfu, 0xbe26dc9du, 0xbe17a056u, 0xbe06bdebu, 0xbdecc593u, 0xbdcf57aau, 0xbdb5ea3au));
  const  __m512 _c0     = _mm512_castsi512_ps(_mm512_setr_epi32(0x3ecc4231u, 0x3ecc541cu, 0x3ecd6c48u, 0x3ed174c3u, 0x3ed9bd5du, 0x3ee5acd5u, 0x3ef2aeddu, 0x3efd5384u,
                                                                0x3f016724u, 0x3f00f778u, 0x3efb389eu, 0x3ef0464du, 0x3ee3014fu, 0x3ed50a78u, 0x3ec779dbu, 0x3ebae363u));
  __m512 result;
  __m512 xr    = _mm512_range_round_ps(x, thres, 2, _MM_FROUND_NO_EXC);
  __m512 xa    = _mm512_and_ps(xr, absmask);
  __m512 index = _mm512_fmadd_ps(xa, scale, shifter);
  __m512 c2    = _mm512_permutexvar_ps(_mm512_castps_si512(index), _c2);
  __m512 c1    = _mm512_permutexvar_ps(_mm512_castps_si512(index), _c1);
  __m512 c0    = _mm512_permutexvar_ps(_mm512_castps_si512(index), _c0);
  __m512 poly  = _mm512_fmadd_ps(c2, xa, c1);
  poly         = _mm512_fmadd_ps(poly, xa, c0);
  result       = _mm512_mul_ps(x, _mm512_fmadd_ps(poly, xr, half));

  return result;
}
#endif /*defined(LIBXSMM_INTRINSICS_AVX512_SKX)*/

#if defined(LIBXSMM_INTRINSICS_AVX512_SKX) /*__AVX512DQ__ needed*/
LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_SKX) __m512 LIBXSMM_INTRINSICS_MM512_GELU_BWD_PS_MINIMAX3(__m512 x) {
  const  __m512 thres   = _mm512_castsi512_ps(_mm512_set1_epi32(0x408f5fff));
  const  __m512 absmask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
  const  __m512 scale   = _mm512_castsi512_ps(_mm512_set1_epi32(0x405d67c9));
  const  __m512 shifter = _mm512_castsi512_ps(_mm512_set1_epi32(0x4b400000));
  const  __m512 half    = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f000000));
  const  __m512 _c2     = _mm512_castsi512_ps(_mm512_setr_epi32(0xbe87047bu, 0xbe6eb875u, 0xbe2210c1u, 0xbd81727fu, 0x3cb9625cu, 0x3da2cbe8u, 0x3dd1d4d1u, 0x3dca0bd0u,
                                                                0x3da47dd0u, 0x3d6f1bd3u, 0x3d216381u, 0x3cd2618cu, 0x3c89f6e6u, 0x3c3ca672u, 0x3c08ed08u, 0x3bd26a14u));
  const  __m512 _c1     = _mm512_castsi512_ps(_mm512_setr_epi32(0xb930e738u, 0xbc4b28bau, 0xbda4212fu, 0xbe5feb0eu, 0xbec8b0e5u, 0xbf09e61bu, 0xbf1c403fu, 0xbf185954u,
                                                                0xbf03e1eeu, 0xbed08a61u, 0xbe9b4508u, 0xbe61788bu, 0xbe257770u, 0xbdfc542au, 0xbdca014eu, 0xbda8d7e9u));
  const  __m512 _c0     = _mm512_castsi512_ps(_mm512_setr_epi32(0x3f4c4245u, 0x3f4c927bu, 0x3f5085f8u, 0x3f5d7bdau, 0x3f73ea12u, 0x3f86142fu, 0x3f8d3df4u, 0x3f8b4b0fu,
                                                                0x3f8022c8u, 0x3f5e5423u, 0x3f39ceb5u, 0x3f199bedu, 0x3f00bee0u, 0x3ede1737u, 0x3ec59b86u, 0x3eb4454cu));
  __m512 result;
  __m512 xr    = _mm512_range_round_ps(x, thres, 2, _MM_FROUND_NO_EXC);
  __m512 xa    = _mm512_and_ps(xr, absmask);
  __m512 index = _mm512_fmadd_ps(xa, scale, shifter);
  __m512 c2    = _mm512_permutexvar_ps(_mm512_castps_si512(index), _c2);
  __m512 c1    = _mm512_permutexvar_ps(_mm512_castps_si512(index), _c1);
  __m512 c0    = _mm512_permutexvar_ps(_mm512_castps_si512(index), _c0);
  __m512 poly  = _mm512_fmadd_ps(c2, xa, c1);
  poly         = _mm512_fmadd_ps(poly, xa, c0);
  result       = _mm512_fmadd_ps(poly, xr, half);

  return result;
}
#endif /*defined(LIBXSMM_INTRINSICS_AVX512_SKX)*/

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_SKX) __m512 LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_FWD(__m512 x) {
  const  __m512 c1     = _mm512_set1_ps( (float)0.79788);
  const  __m512 c2     = _mm512_set1_ps( (float)0.03568);
  const  __m512 c_half = _mm512_set1_ps( (float)0.5);

  __m512 x_half   = _mm512_mul_ps( x, c_half );
  __m512 x_sq   = _mm512_mul_ps( x, x );
  __m512 poly_x1 = _mm512_mul_ps(x, _mm512_fmadd_ps( x_sq, c2, c1));
  __m512 tanh_poly_x = LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2(poly_x1);
  __m512 output = _mm512_fmadd_ps(tanh_poly_x, x_half, x_half);

  return output;
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_SKX) __m512 LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_BWD(__m512 x) {
  const  __m512 c1     = _mm512_set1_ps( (float)0.79788);
  const  __m512 c2     = _mm512_set1_ps( (float)0.03568);
  const  __m512 c3     = _mm512_set1_ps( (float)0.05352);
  const  __m512 c4     = _mm512_set1_ps( (float)0.39894);
  const  __m512 c_half = _mm512_set1_ps( (float)0.5);
  const  __m512 c_ones = _mm512_set1_ps( (float)1.0);
  const  __m512 c_minus_1 = _mm512_set1_ps( (float)-1.0);

  __m512 x_sq   = _mm512_mul_ps( x, x );
  __m512 poly_x1 = _mm512_mul_ps(x, _mm512_fmadd_ps( x_sq, c2, c1));
  __m512 poly_x2 = _mm512_mul_ps(x, _mm512_fmadd_ps( x_sq, c3, c4));

  __m512 tanh_poly_x = LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2(poly_x1);
  __m512 out1 = _mm512_add_ps(c_ones, tanh_poly_x);
  __m512 out2 = _mm512_add_ps(c_half, poly_x2);
  __m512 out3 = _mm512_fmsub_ps(poly_x2, tanh_poly_x, out2);
  __m512 out4 = _mm512_mul_ps(c_minus_1, out3);
  __m512 output = _mm512_mul_ps(out1, out4);

  return output;
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_SKX) __m512 LIBXSMM_INTRINSICS_MM512_EXP_PS_2DTS(__m512 in) {
  const __m512 log2_e   = _mm512_set1_ps(1.442695f);
  const __m512 half     = _mm512_set1_ps(0.5f);
  const __m512 c2       = _mm512_set1_ps(0.240226507f);
  const __m512 c1       = _mm512_set1_ps(0.452920674f);
  const __m512 c0       = _mm512_set1_ps(0.713483036f);

  const __m512 x        = _mm512_fmadd_ps(in, log2_e, half);
#if 1
  const __m512 y        = _mm512_sub_ps(x, _mm512_roundscale_round_ps(x, 1, _MM_FROUND_CUR_DIRECTION));
#else
  const __m512 y        = _mm512_reduce_ps(x, 1);
#endif
  const __m512 t1       = _mm512_fmadd_ps( y, c2, c1);
  const __m512 two_to_y = _mm512_fmadd_ps( y, t1, c0);
  const __m512 exp      = _mm512_scalef_ps( two_to_y, x );

  return exp;
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_SKX) __m512 LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS(__m512 in) {
  const __m512 log2_e   = _mm512_set1_ps(1.442695f);
  const __m512 half     = _mm512_set1_ps(0.5f);
  const __m512 c3       = _mm512_set1_ps(0.05550410866f);
  const __m512 c2       = _mm512_set1_ps(0.15697034396f);
  const __m512 c1       = _mm512_set1_ps(0.49454875509f);
  const __m512 c0       = _mm512_set1_ps(0.70654502287f);

  const __m512 x        = _mm512_fmadd_ps(in, log2_e, half);
#if 1
  const __m512 y        = _mm512_sub_ps(x, _mm512_roundscale_round_ps(x, 1, _MM_FROUND_CUR_DIRECTION));
#else
  const __m512 y        = _mm512_reduce_ps(x, 1);
#endif
  const __m512 t1       = _mm512_fmadd_ps( y, c3, c2);
  const __m512 t2       = _mm512_fmadd_ps( y, t1, c1);
  const __m512 two_to_y = _mm512_fmadd_ps( y, t2, c0);
  const __m512 exp      = _mm512_scalef_ps( two_to_y, x );

  return exp;
}

# if defined(__GNUC__) && !defined(__clang__) && !defined(LIBXSMM_INTEL_COMPILER) && !defined(_CRAYC) && 0
LIBXSMM_PRAGMA_OPTIMIZE_OFF /* avoid ICE in case of symbols (-g) */
# endif
/** Generate random number in the interval [0, 1); not thread-safe.
 *  this is based on xoshiro128+ 1.0, e.g. http://prng.di.unimi.it/xoshiro128plus.c */
LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_SKX) __m512i LIBXSMM_INTRINSICS_MM512_RNG_XOSHIRO128P_EPI32(void) {
  const __m512i result = _mm512_add_epi32(LIBXSMM_INTRINSICS_MM512_RNG_STATE(0), LIBXSMM_INTRINSICS_MM512_RNG_STATE(3));
  const __m512i s = _mm512_slli_epi32(LIBXSMM_INTRINSICS_MM512_RNG_STATE(1), 9);
  __m512i t;
  LIBXSMM_INTRINSICS_MM512_RNG_STATE(2) = _mm512_xor_epi32(LIBXSMM_INTRINSICS_MM512_RNG_STATE(2), LIBXSMM_INTRINSICS_MM512_RNG_STATE(0));
  LIBXSMM_INTRINSICS_MM512_RNG_STATE(3) = _mm512_xor_epi32(LIBXSMM_INTRINSICS_MM512_RNG_STATE(3), LIBXSMM_INTRINSICS_MM512_RNG_STATE(1));
  LIBXSMM_INTRINSICS_MM512_RNG_STATE(1) = _mm512_xor_epi32(LIBXSMM_INTRINSICS_MM512_RNG_STATE(1), LIBXSMM_INTRINSICS_MM512_RNG_STATE(2));
  LIBXSMM_INTRINSICS_MM512_RNG_STATE(0) = _mm512_xor_epi32(LIBXSMM_INTRINSICS_MM512_RNG_STATE(0), LIBXSMM_INTRINSICS_MM512_RNG_STATE(3));
  LIBXSMM_INTRINSICS_MM512_RNG_STATE(2) = _mm512_xor_epi32(LIBXSMM_INTRINSICS_MM512_RNG_STATE(2), s);
  t = _mm512_slli_epi32(LIBXSMM_INTRINSICS_MM512_RNG_STATE(3), 11);
  LIBXSMM_INTRINSICS_MM512_RNG_STATE(3) = _mm512_or_epi32(t, _mm512_srli_epi32(LIBXSMM_INTRINSICS_MM512_RNG_STATE(3), 32 - 11));
  return result;
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_SKX) __m512 LIBXSMM_INTRINSICS_MM512_RNG_PS(void) {
  const __m512i rng_mantissa = _mm512_srli_epi32( LIBXSMM_INTRINSICS_MM512_RNG_XOSHIRO128P_EPI32(), 9 );
  const __m512 one = _mm512_set1_ps(1.0f);
  return _mm512_sub_ps(_mm512_castsi512_ps(_mm512_or_epi32(_mm512_set1_epi32(0x3f800000), rng_mantissa)), one);
}

/** Generate random number in the interval [0, 1); thread save, state needs to be managed by user.
 *  this is based on xoshiro128+ 1.0, e.g. http://prng.di.unimi.it/xoshiro128plus.c */
LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_SKX) __m512i LIBXSMM_INTRINSICS_MM512_RNG_XOSHIRO128P_EXTSTATE_EPI32(unsigned int* stateptr) {
  __m512i state_0 = _mm512_loadu_si512( stateptr    );
  __m512i state_1 = _mm512_loadu_si512( stateptr+16 );
  __m512i state_2 = _mm512_loadu_si512( stateptr+32 );
  __m512i state_3 = _mm512_loadu_si512( stateptr+48 );
  const __m512i result = _mm512_add_epi32(state_0, state_3);
  const __m512i s = _mm512_slli_epi32(state_1, 9);
  __m512i t;
  state_2 = _mm512_xor_epi32(state_2, state_0);
  state_3 = _mm512_xor_epi32(state_3, state_1);
  state_1 = _mm512_xor_epi32(state_1, state_2);
  state_0 = _mm512_xor_epi32(state_0, state_3);
  state_2 = _mm512_xor_epi32(state_2, s);
  _mm512_storeu_si512( stateptr   , state_0 );
  _mm512_storeu_si512( stateptr+16, state_1 );
  _mm512_storeu_si512( stateptr+32, state_2 );
  t = _mm512_slli_epi32(state_3, 11);
  state_3 = _mm512_or_epi32(t, _mm512_srli_epi32(state_3, 32 - 11));
  _mm512_storeu_si512( stateptr+48, state_3 );
  return result;
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_SKX) __m512 LIBXSMM_INTRINSICS_MM512_RNG_EXTSTATE_PS(unsigned int* stateptr) {
  const __m512i rng_mantissa = _mm512_srli_epi32( LIBXSMM_INTRINSICS_MM512_RNG_XOSHIRO128P_EXTSTATE_EPI32(stateptr), 9 );
  const __m512 one = _mm512_set1_ps(1.0f);
  return _mm512_sub_ps(_mm512_castsi512_ps(_mm512_or_epi32(_mm512_set1_epi32(0x3f800000), rng_mantissa)), one);
}
# if defined(__GNUC__) && !defined(__clang__) && !defined(LIBXSMM_INTEL_COMPILER) && !defined(_CRAYC) && 0
LIBXSMM_PRAGMA_OPTIMIZE_ON
# endif
#endif /*__AVX512F__*/


#endif /*LIBXSMM_UTILS_INTRINSICS_X86_H*/
