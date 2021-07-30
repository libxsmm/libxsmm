/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_CPUID_H
#define LIBXSMM_CPUID_H

#include "libxsmm_macros.h"

/**
 * Enumerates the available target architectures and instruction
 * set extensions as returned by libxsmm_get_target_archid().
 * LIBXSMM_X86_ALLFEAT: pseudo-value enabling all features
 * used anywhere in LIBXSMM (never set as an architecture,
 * used as an upper bound in comparisons to distinct x86).
 */
#define LIBXSMM_TARGET_ARCH_UNKNOWN   0
#define LIBXSMM_TARGET_ARCH_GENERIC   1
#define LIBXSMM_X86_GENERIC           1002
#define LIBXSMM_X86_SSE3              1003
#define LIBXSMM_X86_SSE42             1004
#define LIBXSMM_X86_AVX               1005
#define LIBXSMM_X86_AVX2              1006
#define LIBXSMM_X86_AVX512_VL256      1007
#define LIBXSMM_X86_AVX512            1010
#define LIBXSMM_X86_AVX512_MIC        1011 /* KNL */
#define LIBXSMM_X86_AVX512_KNM        1012
#define LIBXSMM_X86_AVX512_CORE       1020 /* SKX */
#define LIBXSMM_X86_AVX512_VL256_CLX  1021
#define LIBXSMM_X86_AVX512_CLX        1022
#define LIBXSMM_X86_AVX512_VL256_CPX  1023
#define LIBXSMM_X86_AVX512_CPX        1024
#define LIBXSMM_X86_AVX512_SPR        1025
#define LIBXSMM_X86_ALLFEAT           1999
#define LIBXSMM_AARCH64_V81           2001 /* Baseline */
#define LIBXSMM_AARCH64_V82           2002 /* A64FX minus SVE */
#define LIBXSMM_AARCH64_A64FX         2100 /* SVE */
#define LIBXSMM_AARCH64_ALLFEAT       2999

#if defined(LIBXSMM_PLATFORM_X86)
/** Zero-initialized structure; assumes conservative properties. */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_cpuid_info {
  int constant_tsc; /** Timer stamp counter is monotonic. */
  int has_context;  /** Context switches are permitted. */
} libxsmm_cpuid_info;
#else
typedef int libxsmm_cpuid_info;
#endif

/** Returns the target architecture and instruction set extensions. */
#if defined(__cplusplus) /* note: stay compatible with TF */
LIBXSMM_API int libxsmm_cpuid_x86(libxsmm_cpuid_info* info = NULL);
LIBXSMM_API int libxsmm_cpuid_arm(libxsmm_cpuid_info* info = NULL);
#else
LIBXSMM_API int libxsmm_cpuid_x86(libxsmm_cpuid_info* info);
LIBXSMM_API int libxsmm_cpuid_arm(libxsmm_cpuid_info* info);
#endif

/**
 * Similar to libxsmm_cpuid_x86, but conceptually not x86-specific.
 * The actual code path (as used by LIBXSMM) is determined by
 * libxsmm_[get|set]_target_archid/libxsmm_[get|set]_target_arch.
 */
LIBXSMM_API int libxsmm_cpuid(void);

/** Names the CPU architecture given by CPUID. */
LIBXSMM_API const char* libxsmm_cpuid_name(int id);

/** SIMD vector length (VLEN) in 32-bit elements. */
LIBXSMM_API int libxsmm_cpuid_vlen32(int id);

#endif /*LIBXSMM_CPUID_H*/

