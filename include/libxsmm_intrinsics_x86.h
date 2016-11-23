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
#ifndef LIBXSMM_INTRINSICS_X86_H
#define LIBXSMM_INTRINSICS_X86_H

#include "libxsmm_cpuid.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif

#if defined(__MIC__)
# define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_IMCI
# define LIBXSMM_INTRINSICS
#else
# if  defined(__AVX512F__)  && defined(__AVX512CD__) \
   && defined(__AVX512DQ__) && defined(__AVX512BW__) && defined(__AVX512VL__)
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_CORE
# elif defined(__AVX512F__) && defined(__AVX512CD__) \
   && defined(__AVX512PF__) && defined(__AVX512ER__)
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_MIC
# elif defined(__AVX512F__) && defined(__AVX512CD__)
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512
# elif defined(__AVX2__)
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_AVX2
# elif defined(__AVX__)
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_AVX
# elif defined(__SSE4_2__)
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_SSE4_2
# elif defined(__SSE4_1__)
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_SSE4_1
# elif defined(__SSE3__)
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_SSE3
# elif defined(__x86_64__)
#   define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_X86_GENERIC
# endif
# if defined(__INTEL_COMPILER)
    /* TODO: compiler version check for LIBXSMM_MAX_STATIC_TARGET_ARCH */
#   if 1300 <= (__INTEL_COMPILER)
#     define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_CORE
#   else
#     define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX2
#   endif
#   define LIBXSMM_INTRINSICS/*no need for target flags*/
#   include <immintrin.h>
# elif defined(_CRAYC) && defined(__GNUC__)
    /* TODO: version check e.g., LIBXSMM_VERSION2(11, 5) <= LIBXSMM_VERSION2(_RELEASE, _RELEASE_MINOR) */
#   define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX
#   define LIBXSMM_INTRINSICS/*no need for target flags*/
#   include <immintrin.h>
# elif defined(_MSC_VER)
    /* TODO: compiler version check for LIBXSMM_MAX_STATIC_TARGET_ARCH */
#   define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX2
#   define LIBXSMM_INTRINSICS/*no need for target flags*/
#   include <immintrin.h>
# else
#   if !defined(__SSE3__)
#     define __SSE3__ 1
#   endif
#   if !defined(__SSSE3__)
#     define LIBXSMM_UNDEF_SSSE
#     define __SSSE3__ 1
#   endif
#   if !defined(__SSE4_1__)
#     define __SSE4_1__ 1
#   endif
#   if !defined(__SSE4_2__)
#     define __SSE4_2__ 1
#   endif
#   if !defined(__AVX__)
#     define __AVX__ 1
#   endif
#   if defined(__clang__)
#     if (LIBXSMM_X86_AVX2 > LIBXSMM_STATIC_TARGET_ARCH)
#       if defined(__APPLE__) && defined(__MACH__)
#         define LIBXSMM_INTRINSICS LIBXSMM_ATTRIBUTE(target("sse3,sse4.1,sse4.2,avx,avx2"))
#         define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX2
#       else
#         if (LIBXSMM_VERSION3(3, 8, 0) <= LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))
#           define LIBXSMM_INTRINSICS LIBXSMM_ATTRIBUTE(target("sse3,sse4.1,sse4.2,avx,avx2,avx512f,avx512cd,avx512pf,avx512er,avx512dq,avx512bw,avx512vl"))
#           define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_CORE
#         elif (LIBXSMM_VERSION3(3, 4, 0) <= LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))
#           define LIBXSMM_INTRINSICS LIBXSMM_ATTRIBUTE(target("sse3,sse4.1,sse4.2,avx,avx2,avx512f,avx512cd,avx512pf,avx512er"))
#           define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX512_MIC
#         else
#           define LIBXSMM_INTRINSICS LIBXSMM_ATTRIBUTE(target("sse3,sse4.1,sse4.2,avx,avx2"))
#           define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX2
#         endif
#       endif
#       if (LIBXSMM_X86_AVX512_MIC <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
#         if !defined(__AVX512F__)
#           define __AVX512F__ 1
#         endif
#         if !defined(__AVX512CD__)
#           define __AVX512CD__ 1
#         endif
#         if !defined(__AVX512PF__)
#           define __AVX512PF__ 1
#         endif
#         if !defined(__AVX512ER__)
#           define __AVX512ER__ 1
#         endif
#       endif
#       if (LIBXSMM_X86_AVX512_CORE <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
#         if !defined(__AVX512DQ__)
#           define __AVX512DQ__ 1
#         endif
#         if !defined(__AVX512BW__)
#           define __AVX512BW__ 1
#         endif
#         if !defined(__AVX512VL__)
#           define __AVX512VL__ 1
#         endif
#       endif
#       if !defined(__AVX2__)
#         define __AVX2__ 1
#       endif
#     else
#       define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_STATIC_TARGET_ARCH
#       define LIBXSMM_INTRINSICS/*no need for target flags*/
#     endif
#     include <immintrin.h>
#   elif defined(__GNUC__) && (LIBXSMM_VERSION3(4, 4, 0) <= LIBXSMM_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))
#     if  (LIBXSMM_VERSION3(4, 9, 0) <= LIBXSMM_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)) && 0
      /* TODO: AVX-512 in GCC appears to be incomplete (missing at _mm512_mask_reduce_or_epi32, and some pseudo intrinsics) */
#     elif defined(__GNUC__) && (LIBXSMM_VERSION3(4, 7, 0) <= LIBXSMM_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))
#       if (LIBXSMM_X86_AVX2 > LIBXSMM_STATIC_TARGET_ARCH)
#         define LIBXSMM_INTRINSICS LIBXSMM_ATTRIBUTE(target("sse3,sse4.1,sse4.2,avx,avx2"))
#         define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX2
#         if !defined(__AVX2__)
#           define __AVX2__ 1
#         endif
#         pragma GCC push_options
#         pragma GCC target("sse3,sse4.1,sse4.2,avx,avx2")
#         include <immintrin.h>
#         pragma GCC pop_options
#       else
#         define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_STATIC_TARGET_ARCH
#         define LIBXSMM_INTRINSICS/*no need for target flags*/
#         include <immintrin.h>
#       endif
#     elif (LIBXSMM_X86_AVX > LIBXSMM_STATIC_TARGET_ARCH)
#       define LIBXSMM_INTRINSICS LIBXSMM_ATTRIBUTE(target("sse3,sse4.1,sse4.2,avx"))
#       define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_X86_AVX
#       if !defined(__AVX__)
#         define __AVX__ 1
#       endif
#       pragma GCC push_options
#       pragma GCC target("sse3,sse4.1,sse4.2,avx")
#       include <immintrin.h>
#       pragma GCC pop_options
#     endif
#   endif
#   if !defined(LIBXSMM_STATIC_TARGET_ARCH) || (LIBXSMM_X86_SSE3 > (LIBXSMM_STATIC_TARGET_ARCH))
#     undef __SSE3__
#   endif
#   if defined(LIBXSMM_UNDEF_SSSE)
#     undef LIBXSMM_UNDEF_SSSE
#     undef __SSSE3__
#   endif
#   if !defined(LIBXSMM_STATIC_TARGET_ARCH) || (LIBXSMM_X86_SSE4_1 > (LIBXSMM_STATIC_TARGET_ARCH))
#     undef __SSE4_1__
#   endif
#   if !defined(LIBXSMM_STATIC_TARGET_ARCH) || (LIBXSMM_X86_SSE4_2 > (LIBXSMM_STATIC_TARGET_ARCH))
#     undef __SSE4_2__
#   endif
#   if !defined(LIBXSMM_STATIC_TARGET_ARCH) || (LIBXSMM_X86_AVX > (LIBXSMM_STATIC_TARGET_ARCH))
#     undef __AVX__
#   endif
#   if !defined(LIBXSMM_STATIC_TARGET_ARCH) || (LIBXSMM_X86_AVX2 > (LIBXSMM_STATIC_TARGET_ARCH))
#     undef __AVX2__
#   endif
#   if !defined(LIBXSMM_STATIC_TARGET_ARCH) || (LIBXSMM_X86_AVX512 > (LIBXSMM_STATIC_TARGET_ARCH))
#     undef __AVX512F__
#     undef __AVX512CD__
#   endif
#   if !defined(LIBXSMM_STATIC_TARGET_ARCH) || (LIBXSMM_X86_AVX512_MIC > (LIBXSMM_STATIC_TARGET_ARCH))
#     undef __AVX512F__
#     undef __AVX512CD__
#     undef __AVX512PF__
#     undef __AVX512ER__
#   endif
#   if !defined(LIBXSMM_STATIC_TARGET_ARCH) || (LIBXSMM_X86_AVX512_CORE > (LIBXSMM_STATIC_TARGET_ARCH))
#     undef __AVX512F__
#     undef __AVX512CD__
#     undef __AVX512DQ__
#     undef __AVX512BW__
#     undef __AVX512VL__
#   endif
# endif
#endif

#if !defined(LIBXSMM_STATIC_TARGET_ARCH)
# define LIBXSMM_STATIC_TARGET_ARCH LIBXSMM_TARGET_ARCH_GENERIC
#endif

#if !defined(LIBXSMM_MAX_STATIC_TARGET_ARCH)
# define LIBXSMM_MAX_STATIC_TARGET_ARCH LIBXSMM_STATIC_TARGET_ARCH
#endif

/** Include basic x86 intrinsics such as __rdtsc. */
#if defined(LIBXSMM_INTRINSICS)
# if defined(_WIN32)
#   include <intrin.h>
# else
#   include <x86intrin.h>
# endif
# include <xmmintrin.h>
# if defined(__SSE3__)
#   include <pmmintrin.h>
# endif
#else
# if !defined(LIBXSMM_INTRINSICS_NONE)
#   define LIBXSMM_INTRINSICS_NONE
# endif
# define LIBXSMM_INTRINSICS
#endif

#if defined(_WIN32)
# include <malloc.h>
#else
# include <mm_malloc.h>
#endif

/** Intrinsic-specifc fixups */
#if defined(__clang__)
# define LIBXSMM_INTRINSICS_LDDQU_SI128 _mm_loadu_si128
#else
# define LIBXSMM_INTRINSICS_LDDQU_SI128 _mm_lddqu_si128
#endif

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#endif /*LIBXSMM_INTRINSICS_X86_H*/

