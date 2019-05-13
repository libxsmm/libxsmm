/******************************************************************************
** Copyright (c) 2016-2019, Intel Corporation                                **
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

#if /*no intrinsics: tested with 17.x and 18.x*/defined(__PGI) || /*legacy*/(defined(_CRAYC) && !defined(__GNUC__))
# if !defined(LIBXSMM_INTRINSICS_NONE) && !defined(LIBXSMM_INTRINSICS_STATIC)
#   define LIBXSMM_INTRINSICS_NONE
# endif
#elif !defined(LIBXSMM_INTRINSICS_STATIC) && !defined(LIBXSMM_INTRINSICS_NONE) && /* GCC 4.4 (target-attribute) */ \
    (defined(__GNUC__) && !defined(__clang__) && !defined(LIBXSMM_INTEL_COMPILER) && !defined(_CRAYC) && \
     LIBXSMM_VERSION3(4, 4, 0) > LIBXSMM_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)) \
 || (defined(__clang__) && LIBXSMM_VERSION3(3, 7, 0) > LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)) \
 || (defined(__APPLE__) && defined(__MACH__) && !defined(LIBXSMM_INTEL_COMPILER) && defined(__clang__) && \
     LIBXSMM_VERSION3(9, 0, 0) > LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))
# define LIBXSMM_INTRINSICS_STATIC
#endif

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif

#if defined(__MIC__) && !defined(LIBXSMM_INTRINSICS_NONE)
# define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_TARGET_ARCH_GENERIC
# define LIBXSMM_INTRINSICS(TARGET)
# define LIBXSMM_INTRINSICS_INCLUDE
#elif !defined(LIBXSMM_INTRINSICS_NONE) /*!defined(__MIC__)*/
# if    defined(__AVX512F__)  && defined(__AVX512CD__) \
   &&   defined(__AVX512DQ__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512VNNI__) \
   &&   defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__) \
   && (!defined(__GNUC__)  || defined(__clang__) || defined(__INTEL_COMPILER) || defined(_CRAYC) \
                           || (LIBXSMM_VERSION3(6, 0, 0) <= LIBXSMM_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))) \
   && (!defined(__clang__) || (LIBXSMM_VERSION3(4, 0, 0) <= LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__) \
                           || (LIBXSMM_VERSION3(0, 0, 0) == LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)))) \
   && (!defined(__APPLE__) || !defined(__MACH__) || LIBXSMM_VERSION3(8, 1, 0) <= LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_CLX
#   define LIBXSMM_INTRINSICS_INCLUDE
# elif  defined(__AVX512F__)  && defined(__AVX512CD__) \
   &&   defined(__AVX512DQ__) && defined(__AVX512BW__) && defined(__AVX512VL__) \
   &&   defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__) \
   && (!defined(__GNUC__)  || defined(__clang__) || defined(__INTEL_COMPILER) || defined(_CRAYC) \
                           || (LIBXSMM_VERSION3(6, 0, 0) <= LIBXSMM_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))) \
   && (!defined(__clang__) || (LIBXSMM_VERSION3(4, 0, 0) <= LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__) \
                           || (LIBXSMM_VERSION3(0, 0, 0) == LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)))) \
   && (!defined(__APPLE__) || !defined(__MACH__) || LIBXSMM_VERSION3(8, 1, 0) <= LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_CORE
#   define LIBXSMM_INTRINSICS_INCLUDE
# elif  defined(__AVX512F__) && defined(__AVX512CD__) \
   &&   defined(__AVX512PF__) && defined(__AVX512ER__) \
   &&   defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__) \
   && (!defined(__GNUC__)  || defined(__clang__) || defined(__INTEL_COMPILER) || defined(_CRAYC) \
                           || (LIBXSMM_VERSION3(6, 0, 0) <= LIBXSMM_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))) \
   && (!defined(__clang__) || (LIBXSMM_VERSION3(4, 0, 0) <= LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__) \
                           || (LIBXSMM_VERSION3(0, 0, 0) == LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)))) \
   && (!defined(__APPLE__) || !defined(__MACH__) || LIBXSMM_VERSION3(8, 1, 0) <= LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_MIC
#   define LIBXSMM_INTRINSICS_INCLUDE
# elif  defined(__AVX512F__) && defined(__AVX512CD__) \
   &&   defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__) \
   && (!defined(__GNUC__)  || defined(__clang__) || defined(__INTEL_COMPILER) || defined(_CRAYC) \
                           || (LIBXSMM_VERSION3(6, 0, 0) <= LIBXSMM_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))) \
   && (!defined(__clang__) || (LIBXSMM_VERSION3(4, 0, 0) <= LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__) \
                           || (LIBXSMM_VERSION3(0, 0, 0) == LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)))) \
   && (!defined(__APPLE__) || !defined(__MACH__) || LIBXSMM_VERSION3(8, 1, 0) <= LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512
#   define LIBXSMM_INTRINSICS_INCLUDE
# elif defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__)
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_AVX2
#   define LIBXSMM_INTRINSICS_INCLUDE
# elif defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__)
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_AVX
#   define LIBXSMM_INTRINSICS_INCLUDE
# elif defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__)
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_SSE4
#   define LIBXSMM_INTRINSICS_INCLUDE
# elif defined(__SSE3__)
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_SSE3
#   define LIBXSMM_INTRINSICS_INCLUDE
# elif defined(__x86_64__) || defined(_WIN32) || defined(_WIN64)
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_GENERIC
# endif
# if defined(LIBXSMM_STATIC_TARGET_ARCH) && !defined(LIBXSMM_INTRINSICS_STATIC)
#   if defined(__INTEL_COMPILER)
      /* TODO: compiler version check for LIBXSMM_MAX_STATIC_TARGET_ARCH */
#     if 1500 <= (LIBXSMM_INTEL_COMPILER)
#       define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_CORE
#     elif 1400 <= (LIBXSMM_INTEL_COMPILER)
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
#   elif (defined(__GNUC__) && LIBXSMM_VERSION3(4, 9, 0) <= LIBXSMM_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)) \
      && !defined(__PGI)
#     if LIBXSMM_X86_AVX2 < LIBXSMM_STATIC_TARGET_ARCH && !defined(__CYGWIN__)
#       define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_STATIC_TARGET_ARCH
#     else /* Cygwin: invalid register for .seh_savexmm */
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
#         if (LIBXSMM_MAX_STATIC_TARGET_ARCH < LIBXSMM_STATIC_TARGET_ARCH)
#           undef LIBXSMM_STATIC_TARGET_ARCH /* account for compiler issues */
#           define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_MAX_STATIC_TARGET_ARCH
#         endif
#       else /* Error: invalid register for .seh_savexmm */
#         define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX2
#       endif
#     else /* fall-back */
#       define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_STATIC_TARGET_ARCH
#       if !defined(LIBXSMM_INTRINSICS_STATIC) && (LIBXSMM_STATIC_TARGET_ARCH < LIBXSMM_X86_AVX2/*workaround*/)
#         define LIBXSMM_INTRINSICS_STATIC
#       endif
#     endif
#     if !defined(LIBXSMM_INTRINSICS_INCLUDE) && !defined(__PGI)
#       define LIBXSMM_INTRINSICS_INCLUDE
#     endif
#     if defined(LIBXSMM_INTRINSICS_INCLUDE) && !defined(LIBXSMM_INTRINSICS_NONE) && !defined(LIBXSMM_INTRINSICS_STATIC)
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
#       if !defined(__AVX5124VNNIW__)
#         define __AVX5124VNNIW__ 1
#       endif
#       if !defined(__AVX5124FMAPS__)
#         define __AVX5124FMAPS__ 1
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
#       if !defined(__AVX512BF16__)
#         define __AVX512BF16__ 1
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
#         undef __AVX512PF__
#         undef __AVX512ER__
#       endif
#       if (LIBXSMM_X86_AVX512_KNM > (LIBXSMM_STATIC_TARGET_ARCH))
#         undef __AVX5124VNNIW__
#         undef __AVX5124FMAPS__
#       endif
#       if (LIBXSMM_X86_AVX512_CORE > (LIBXSMM_STATIC_TARGET_ARCH))
#         undef __AVX512DQ__
#         undef __AVX512BW__
#         undef __AVX512VL__
#       endif
#       if (LIBXSMM_X86_AVX512_CLX > (LIBXSMM_STATIC_TARGET_ARCH))
#         undef __AVX512VNNI__
#       endif
#       if (LIBXSMM_X86_AVX512_CPX > (LIBXSMM_STATIC_TARGET_ARCH))
#         undef __AVX512BF16__
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
#       if (LIBXSMM_X86_AVX512_KNM <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
#         define LIBXSMM_ATTRIBUTE_TARGET_1011 target("avx2,fma,avx512f,avx512cd,avx512pf,avx512er,avx5124vnniw,avx5124fmaps")
#       else /* LIBXSMM_X86_AVX512_MIC */
#         define LIBXSMM_ATTRIBUTE_TARGET_1011 LIBXSMM_ATTRIBUTE_TARGET_1010
#       endif
#       if (LIBXSMM_X86_AVX512_CORE <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
#         define LIBXSMM_ATTRIBUTE_TARGET_1020 target("avx2,fma,avx512f,avx512cd,avx512dq,avx512bw,avx512vl")
#       else /* LIBXSMM_X86_AVX512 */
#         define LIBXSMM_ATTRIBUTE_TARGET_1020 LIBXSMM_ATTRIBUTE_TARGET_1007
#       endif
#       if (LIBXSMM_X86_AVX512_CLX <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
#         define LIBXSMM_ATTRIBUTE_TARGET_1021 target("avx2,fma,avx512f,avx512cd,avx512dq,avx512bw,avx512vl,avx512vnni")
#       else /* LIBXSMM_X86_AVX512_CORE */
#         define LIBXSMM_ATTRIBUTE_TARGET_1021 LIBXSMM_ATTRIBUTE_TARGET_1020
#       endif
#       if (LIBXSMM_X86_AVX512_CPX <= LIBXSMM_MAX_STATIC_TARGET_ARCH) /* TODO: verify compiler flag */
#         define LIBXSMM_ATTRIBUTE_TARGET_1022 target("avx2,fma,avx512f,avx512cd,avx512dq,avx512bw,avx512vl,avx512vnni,avx512bf16")
#       else /* LIBXSMM_X86_AVX512_CORE */
#         define LIBXSMM_ATTRIBUTE_TARGET_1022 LIBXSMM_ATTRIBUTE_TARGET_1021
#       endif
#     else
#       define LIBXSMM_INTRINSICS(TARGET)/*no need for target flags*/
#     endif
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
#endif

#if !defined(LIBXSMM_INTRINSICS)
# define LIBXSMM_INTRINSICS(TARGET)
#endif

/** Include basic x86 intrinsics such as __rdtsc. */
#if defined(LIBXSMM_INTRINSICS_INCLUDE)
# if defined(_WIN32)
#   include <intrin.h>
# elif defined(LIBXSMM_INTEL_COMPILER) || defined(_CRAYC) || defined(__clang__) || defined(__PGI)
#   include <x86intrin.h>
# elif defined(__GNUC__) && (LIBXSMM_VERSION3(4, 4, 0) <= LIBXSMM_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))
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
# define LIBXSMM_INTRINSICS_MM512_STREAM_SI512(A, B) _mm512_store_si512((A), (B))
# define LIBXSMM_INTRINSICS_MM512_STREAM_PS(A, B) _mm512_store_ps((A), (B))
# define LIBXSMM_INTRINSICS_MM512_STREAM_PD(A, B) _mm512_store_pd(A, B)
#else
# define LIBXSMM_INTRINSICS_MM512_LOAD_PS(A) _mm512_load_ps((const float*)(A))
# define LIBXSMM_INTRINSICS_MM512_LOAD_PD(A) _mm512_load_pd((const double*)(A))
# define LIBXSMM_INTRINSICS_MM512_STREAM_SI512(A, B) _mm512_stream_si512((__m512i*)(A), (B))
# define LIBXSMM_INTRINSICS_MM512_STREAM_PS(A, B) _mm512_stream_ps((A), (B))
# define LIBXSMM_INTRINSICS_MM512_STREAM_PD(A, B) _mm512_stream_pd((A), (B))
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
# define LIBXSMM_INTRINSICS_MM512_MASK_I32GATHER_EPI32(A, B, C, D, E) _mm512_mask_i32gather_epi32(A, B, C, D, E)
# define LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(A, B) _mm512_extracti64x4_epi64(A, B)
# define LIBXSMM_INTRINSICS_MM512_ABS_PS(A) _mm512_abs_ps(A)
# define LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32() _mm512_undefined_epi32()
# define LIBXSMM_INTRINSICS_MM512_UNDEFINED() _mm512_undefined()
# define LIBXSMM_INTRINSICS_MM_UNDEFINED_PD() _mm_undefined_pd()
#else
# define LIBXSMM_INTRINSICS_MM512_SET_EPI16(E31, E30, E29, E28, E27, E26, E25, E24, E23, E22, E21, E20, E19, E18, E17, E16, \
                                                      E15, E14, E13, E12, E11, E10, E9, E8, E7, E6, E5, E4, E3, E2, E1, E0) \
               _mm512_set_epi32(((E31) << 16) | (E30), ((E29) << 16) | (E28), ((E27) << 16) | (E26), ((E25) << 16) | (E24), \
                                ((E23) << 16) | (E22), ((E21) << 16) | (E20), ((E19) << 16) | (E18), ((E17) << 16) | (E16), \
                                ((E15) << 16) | (E14), ((E13) << 16) | (E12), ((E11) << 16) | (E10),  ((E9) << 16) |  (E8), \
                                 ((E7) << 16) |  (E6),  ((E5) << 16) |  (E4),  ((E3) << 16) |  (E2),  ((E1) << 16) |  (E0))
# define LIBXSMM_INTRINSICS_MM512_MASK_I32GATHER_EPI32(A, B, C, D, E) _mm512_castps_si512(_mm512_mask_i32gather_ps( \
                           _mm512_castsi512_ps(A), B, C, (const float*)(D), E))
# define LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(A, B) _mm256_castpd_si256(_mm512_extractf64x4_pd(_mm512_castsi512_pd(A), B))
# define LIBXSMM_INTRINSICS_MM512_ABS_PS(A) _mm512_castsi512_ps(_mm512_and_epi32( \
                           _mm512_castps_si512(A), _mm512_set1_epi32(0x7FFFFFFF)))
# define LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32() _mm512_set1_epi32(0)
# define LIBXSMM_INTRINSICS_MM512_UNDEFINED() _mm512_set1_ps(0)
# define LIBXSMM_INTRINSICS_MM_UNDEFINED_PD() _mm_set1_pd(0)
#endif
#if (defined(LIBXSMM_INTEL_COMPILER) && (1800 <= (LIBXSMM_INTEL_COMPILER))) || (defined(__GNUC__) \
      && LIBXSMM_VERSION3(6, 0, 0) <= LIBXSMM_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)) \
  || ((!defined(__APPLE__) || !defined(__MACH__)) && defined(__clang__) \
      && LIBXSMM_VERSION3(8, 0, 0) <= LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))
# define LIBXSMM_INTRINSICS_MM512_LOAD_MASK16(SRC_PTR) _load_mask16((/*const*/ __mmask16*)(SRC_PTR))
# define LIBXSMM_INTRINSICS_MM512_STORE_MASK16(DST_PTR, SRC) _store_mask16((__mmask16*)(DST_PTR), SRC);
#else
# define LIBXSMM_INTRINSICS_MM512_LOAD_MASK16(SRC_PTR) ((__mmask16)_mm512_mask2int(*((__mmask16*)(SRC_PTR))))
# define LIBXSMM_INTRINSICS_MM512_STORE_MASK16(DST_PTR, SRC) (*(unsigned short*)(DST_PTR) = (unsigned short)(SRC))
#endif

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
#if !defined(LIBXSMM_INTRINSICS_KNC) && !defined(LIBXSMM_INTRINSICS_NONE) && defined(__MIC__)
# define LIBXSMM_INTRINSICS_KNC
#endif
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
/** LIBXSMM_INTRINSICS_SSE4 is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXSMM_INTRINSICS_SSE4) && !defined(LIBXSMM_INTRINSICS_NONE) && (LIBXSMM_X86_SSE4 <= LIBXSMM_STATIC_TARGET_ARCH || \
   (!defined(LIBXSMM_INTRINSICS_STATIC) && LIBXSMM_X86_SSE4 <= LIBXSMM_MAX_STATIC_TARGET_ARCH))
# define LIBXSMM_INTRINSICS_SSE4
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
/** LIBXSMM_INTRINSICS_AVX512 is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXSMM_INTRINSICS_AVX512) && !defined(LIBXSMM_INTRINSICS_NONE) && (LIBXSMM_X86_AVX512 <= LIBXSMM_STATIC_TARGET_ARCH || \
   (!defined(LIBXSMM_INTRINSICS_STATIC) && LIBXSMM_X86_AVX512 <= LIBXSMM_MAX_STATIC_TARGET_ARCH))
# define LIBXSMM_INTRINSICS_AVX512
#endif
/** LIBXSMM_INTRINSICS_AVX512_MIC is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXSMM_INTRINSICS_AVX512_MIC) && !defined(LIBXSMM_INTRINSICS_NONE) && (LIBXSMM_X86_AVX512_MIC <= LIBXSMM_STATIC_TARGET_ARCH || \
   (!defined(LIBXSMM_INTRINSICS_STATIC) && LIBXSMM_X86_AVX512_MIC <= LIBXSMM_MAX_STATIC_TARGET_ARCH))
# define LIBXSMM_INTRINSICS_AVX512_MIC
#endif
/** LIBXSMM_INTRINSICS_AVX512_KNM is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXSMM_INTRINSICS_AVX512_KNM) && !defined(LIBXSMM_INTRINSICS_NONE) && (LIBXSMM_X86_AVX512_KNM <= LIBXSMM_STATIC_TARGET_ARCH || \
   (!defined(LIBXSMM_INTRINSICS_STATIC) && LIBXSMM_X86_AVX512_KNM <= LIBXSMM_MAX_STATIC_TARGET_ARCH))
# define LIBXSMM_INTRINSICS_AVX512_KNM
#endif
/** LIBXSMM_INTRINSICS_AVX512_CORE is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXSMM_INTRINSICS_AVX512_CORE) && !defined(LIBXSMM_INTRINSICS_NONE) && (LIBXSMM_X86_AVX512_CORE <= LIBXSMM_STATIC_TARGET_ARCH || \
   (!defined(LIBXSMM_INTRINSICS_STATIC) && LIBXSMM_X86_AVX512_CORE <= LIBXSMM_MAX_STATIC_TARGET_ARCH))
# define LIBXSMM_INTRINSICS_AVX512_CORE
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

/**
 * Pseudo intrinsics (AVX-512)
 */
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
# define LIBXSMM_INTRINSICS_MM512_QUANTIZE_NEAR_PS_EPI16( A, B ) _mm512_cvtepi32_epi16(_mm512_cvt_roundps_epi32( \
    _mm512_mul_ps(LIBXSMM_INTRINSICS_MM512_LOAD_PS(A), B), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512) __m512i LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(__m512 a) {
  const __m512i vnaninf = _mm512_set1_epi32(0x7f800000), vrneadd = _mm512_set1_epi32(0x00007fff);
  const __m512i vfixup = _mm512_set1_epi32(0x00000001), vfixupmask = _mm512_set1_epi32(0x00010000);
  const __m512i mm512_roundbf16rne_a_ = _mm512_castps_si512(a);
  const __mmask16 mm512_roundbf16rne_mask1_ = _mm512_cmp_epi32_mask(_mm512_and_epi32(mm512_roundbf16rne_a_, vnaninf), vnaninf, _MM_CMPINT_NE);
  const __mmask16 mm512_roundbf16rne_mask2_ = _mm512_cmp_epi32_mask(_mm512_and_epi32(mm512_roundbf16rne_a_, vfixupmask), vfixupmask, _MM_CMPINT_EQ);
  return _mm512_mask_add_epi32(mm512_roundbf16rne_a_, mm512_roundbf16rne_mask1_, mm512_roundbf16rne_a_, _mm512_mask_add_epi32(vrneadd, mm512_roundbf16rne_mask2_, vrneadd, vfixup));
}

/** SVML-intrinsics */
LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512) __m512 LIBXSMM_INTRINSICS_MM512_TANH_PS_RATIONAL_78( __m512 x ) {
  const  __m512 c0        = _mm512_set1_ps( (float)2027025.0 );
  const  __m512 c1        = _mm512_set1_ps( (float)270270.0 );
  const  __m512 c2        = _mm512_set1_ps( (float)6930.0 );
  const  __m512 c3        = _mm512_set1_ps( (float)36.0 );
  const  __m512 c1_d      = _mm512_set1_ps( (float)945945.0 );
  const  __m512 c2_d      = _mm512_set1_ps( (float)51975.0 );
  const  __m512 c3_d      = _mm512_set1_ps( (float)630.0 );
  const  __m512 hi_bound  = _mm512_set1_ps( (float)4.97 );
  const  __m512 lo_bound  = _mm512_set1_ps( (float)-4.97 );
  const  __m512 ones      = _mm512_set1_ps( (float)1.0 );
  const  __m512 neg_ones  = _mm512_set1_ps( (float)-1.0 );

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

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512) __m512 LIBXSMM_INTRINSICS_MM512_TANH_PS_RATIONAL_32( __m512 x ) {
  const  __m512 c1        = _mm512_set1_ps( (float)(1.0/27.0));
  const  __m512 c2        = _mm512_set1_ps( (float)(1.0/3));
  const  __m512 hi_bound  = _mm512_set1_ps( (float)3.2 );
  const  __m512 lo_bound  = _mm512_set1_ps( (float)-3.2 );
  const  __m512 ones      = _mm512_set1_ps( (float)1.0 );
  const  __m512 neg_ones  = _mm512_set1_ps( (float)-1.0 );

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

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512) __m512 LIBXSMM_INTRINSICS_MM512_TANH_PS_EXP2( __m512 _x ) {
  const __m512 twice_log2_e = _mm512_set1_ps(1.442695f*2 );
  const __m512 half       = _mm512_set1_ps( 0.5f );
  const __m512 c2         = _mm512_set1_ps( 0.240226507f );
  const __m512 c1         = _mm512_set1_ps( 0.452920674f );
  const __m512 c0         = _mm512_set1_ps( 0.713483036f );
  const __m512 ones       = _mm512_set1_ps( 1.0 );
  const __m512 minus_twos = _mm512_set1_ps( -2.0f );

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

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512) __m512 LIBXSMM_INTRINSICS_MM512_TANH_PS_EXP3( __m512 _x ) {
  const __m512 twice_log2_e = _mm512_set1_ps(1.442695f*2 );
  const __m512 half       = _mm512_set1_ps( 0.5f );
  const __m512 c3         = _mm512_set1_ps( 0.05550410866f );
  const __m512 c2         = _mm512_set1_ps( 0.15697034396f );
  const __m512 c1         = _mm512_set1_ps( 0.49454875509f );
  const __m512 c0         = _mm512_set1_ps( 0.70654502287f );
  const __m512 ones       = _mm512_set1_ps( 1.0 );
  const __m512 minus_twos = _mm512_set1_ps( -2.0f );

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

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512) __m512 LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2( __m512 x ) {
  __m512 result, func_p0, func_p1, func_p2;
  const __m512i sign_mask = _mm512_set1_epi32( 0x80000000 );
  const __m512i sign_filter = _mm512_set1_epi32( 0x7FFFFFFF );
  const __m512i lut_low = _mm512_set1_epi32( 246 );
  const __m512i lut_high = _mm512_set1_epi32( 261 );
  const __m512 tanh_p0_2_reg = _mm512_set_ps( 0.40555,  0.118928, -0.00972979, -0.027403, -0.0169851, -0.00776152, -0.00305889, -0.00116259,  -0.00041726, -8.53233e-6,  1.0,  0.999998,   0.999754,    0.992682,    0.936453,    0.738339);
  const __m512 tanh_p1_2_reg = _mm512_set_ps( 0.495602, 0.88152,  1.1257,    1.17021,       1.1289,    1.07929,   1.04323,  1.02301, 1.01162, 1.00164, 1.56828e-14, 4.49924e-7, 0.0000646924, 0.00260405, 0.0311608, 0.168736);
  const __m512 tanh_p2_2_reg = _mm512_set_ps( -0.108238,  -0.238428,    -0.354418,   -0.382403,   -0.341357,    -0.274509,   -0.205249,  -0.151196,  -0.107635, -0.0466868, -3.60822e-16, -2.05971e-8, -4.24538e-6, -0.000231709, -0.00386434, -0.0277702);

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

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512) __m512 LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX3( __m512 x ) {
  __m512 result, func_p0, func_p1, func_p2, func_p3;
  const __m512i sign_mask = _mm512_set1_epi32( 0x80000000 );
  const __m512i sign_filter = _mm512_set1_epi32( 0x7FFFFFFF );
  const __m512i lut_low = _mm512_set1_epi32( 246 );
  const __m512i lut_high = _mm512_set1_epi32( 261 );

  const __m512 tanh_p0_3_reg = _mm512_setr_ps(0.466283, 0.828506, 0.974375, 0.998826, 0.999986, 1.0, -1.50006e-08, -7.98169e-06, -4.53753e-05, -0.00023755, -0.00125285, -0.00572314, -0.0227717, -0.0629089, -0.0842343, 0.0711998);
  const __m512 tanh_p1_3_reg = _mm512_setr_ps(0.500617, 0.124369, 0.0137214, 0.000464124, 4.02465e-06, 0.0, 1.00001, 1.00028, 1.00112, 1.00414, 1.01557, 1.05095, 1.14785, 1.31013, 1.37895, 1.07407);
  const __m512 tanh_p2_3_reg = _mm512_setr_ps(-0.161332, -0.0305526, -0.00245909, -6.12647e-05, -3.76127e-07, 0.0, -0.000245872, -0.00341151, -0.00971505, -0.0256817, -0.0686911, -0.162433, -0.346828, -0.566516, -0.640214, -0.440119);
  const __m512 tanh_p3_3_reg = _mm512_setr_ps(0.0177393, 0.00253432, 0.000147303, 2.69963e-06, 1.16764e-08, 0.0, -0.330125, -0.317621, -0.301776, -0.27358, -0.219375, -0.136197, -0.0186868, 0.0808901, 0.107095, 0.0631459);

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

#if defined(LIBXSMM_INTEL_COMPILER)
# define LIBXSMM_INTRINSICS_MM512_TANH_PS(A) _mm512_tanh_ps(A)
#else
# include <math.h>
LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512) __m512 LIBXSMM_INTRINSICS_MM512_TANH_PS(__m512 a) {
  float a16[16]; int i;
  _mm512_store_ps(a16, a);
  for (i = 0; i < 16; ++i) a16[i] = LIBXSMM_TANHF(a16[i]);
  return _mm512_loadu_ps(a16);
}
#endif /* SVML */

/** 2048-bit state for RNG */
LIBXSMM_APIVAR(__m512i libxsmm_intrinsics_mm512_rng_state0);
LIBXSMM_APIVAR(__m512i libxsmm_intrinsics_mm512_rng_state1);
LIBXSMM_APIVAR(__m512i libxsmm_intrinsics_mm512_rng_state2);
LIBXSMM_APIVAR(__m512i libxsmm_intrinsics_mm512_rng_state3);

/** Generate random number in the interval [0, 1); not thread-safe. */
# if defined(__GNUC__) && !defined(__clang__) && !defined(LIBXSMM_INTEL_COMPILER) && !defined(_CRAYC)
LIBXSMM_PRAGMA_OPTIMIZE_OFF /* avoid ICE in case of symbols (-g) */
# endif
LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512) __m512 LIBXSMM_INTRINSICS_MM512_RNG_PS(void) {
  const __m512i rng_mantissa = _mm512_srli_epi32(_mm512_add_epi32(libxsmm_intrinsics_mm512_rng_state0, libxsmm_intrinsics_mm512_rng_state3), 9);
  const __m512i s = _mm512_slli_epi32(libxsmm_intrinsics_mm512_rng_state1, 9);
  const __m512 one = _mm512_set1_ps(1.0f);
  __m512i t;
  libxsmm_intrinsics_mm512_rng_state2 = _mm512_xor_epi32(libxsmm_intrinsics_mm512_rng_state2, libxsmm_intrinsics_mm512_rng_state0);
  libxsmm_intrinsics_mm512_rng_state3 = _mm512_xor_epi32(libxsmm_intrinsics_mm512_rng_state3, libxsmm_intrinsics_mm512_rng_state1);
  libxsmm_intrinsics_mm512_rng_state1 = _mm512_xor_epi32(libxsmm_intrinsics_mm512_rng_state1, libxsmm_intrinsics_mm512_rng_state2);
  libxsmm_intrinsics_mm512_rng_state0 = _mm512_xor_epi32(libxsmm_intrinsics_mm512_rng_state0, libxsmm_intrinsics_mm512_rng_state3);
  libxsmm_intrinsics_mm512_rng_state2 = _mm512_xor_epi32(libxsmm_intrinsics_mm512_rng_state2, s);
  t = _mm512_slli_epi32(libxsmm_intrinsics_mm512_rng_state3, 11);
  libxsmm_intrinsics_mm512_rng_state3 = _mm512_or_epi32(t, _mm512_srli_epi32(libxsmm_intrinsics_mm512_rng_state3, 32 - 11));
  return _mm512_sub_ps(_mm512_castsi512_ps(_mm512_or_epi32(_mm512_set1_epi32(0x3f800000), rng_mantissa)), one);
}
# if defined(__GNUC__) && !defined(__clang__) && !defined(LIBXSMM_INTEL_COMPILER) && !defined(_CRAYC)
LIBXSMM_PRAGMA_OPTIMIZE_ON
# endif
#endif /*__AVX512F__*/

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#endif /*LIBXSMM_INTRINSICS_X86_H*/

