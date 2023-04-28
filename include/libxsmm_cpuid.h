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
#ifndef LIBXSMM_CPUID_H
#define LIBXSMM_CPUID_H

#include "libxsmm_typedefs.h"

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
#define LIBXSMM_X86_AVX2_ADL          1007
#define LIBXSMM_X86_AVX2_SRF          1008
#define LIBXSMM_X86_AVX512_VL128      1041
#define LIBXSMM_X86_AVX512_VL256      1051
#define LIBXSMM_X86_AVX512_VL256_CLX  1052
#define LIBXSMM_X86_AVX512_VL256_CPX  1053
#define LIBXSMM_X86_AVX512            1101
#define LIBXSMM_X86_AVX512_MIC        1102 /* KNL */
#define LIBXSMM_X86_AVX512_KNM        1103
#define LIBXSMM_X86_AVX512_CORE       1104 /* SKX */
#define LIBXSMM_X86_AVX512_CLX        1105
#define LIBXSMM_X86_AVX512_CPX        1106
#define LIBXSMM_X86_AVX512_SPR        1107
#define LIBXSMM_X86_AVX512_GNR        1108
#define LIBXSMM_X86_ALLFEAT           1999
#define LIBXSMM_AARCH64_V81           2001 /* Baseline */
#define LIBXSMM_AARCH64_V82           2002 /* A64FX minus SVE */
#define LIBXSMM_AARCH64_APPL_M1       2101 /* Apple M1 */
#define LIBXSMM_AARCH64_SVE128        2201 /* SVE 128 */
#define LIBXSMM_AARCH64_SVE256        2301 /* SVE 256 */
#define LIBXSMM_AARCH64_NEOV1         2302 /* Neoverse V1, Graviton 3 */
#define LIBXSMM_AARCH64_SVE512        2401 /* SVE 512 */
#define LIBXSMM_AARCH64_A64FX         2402 /* A64FX */
#define LIBXSMM_AARCH64_ALLFEAT       2999

 /** Zero-initialized structure; assumes conservative properties. */
LIBXSMM_EXTERN_C typedef struct libxsmm_cpuid_info {
  char model[1024]; /** CPU-name (OS-specific implementation). */
#if defined(LIBXSMM_PLATFORM_X86)
  int constant_tsc; /** Timer stamp counter is monotonic. */
  int has_context;  /** Context switches are permitted. */
#endif
} libxsmm_cpuid_info;

/** Returns the target architecture and instruction set extensions. */
#if defined(__cplusplus) /* note: stay compatible with TF */
LIBXSMM_API int libxsmm_cpuid_x86(libxsmm_cpuid_info* info = NULL);
LIBXSMM_API int libxsmm_cpuid_arm(libxsmm_cpuid_info* info = NULL);
#else
LIBXSMM_API int libxsmm_cpuid_x86(libxsmm_cpuid_info* info);
LIBXSMM_API int libxsmm_cpuid_arm(libxsmm_cpuid_info* info);
#endif

/**
 * TODO: limited lifetime API until we have a fully-fledged ARM CPU flags test.
 */
LIBXSMM_API unsigned int libxsmm_cpuid_arm_mmla_gemm_pack_b_to_vnnit_on_stack(void);

/**
 * TODO: limited lifetime API until we have a fully-fledged ARM CPU flags test.
 * Might be needed to overwrite BFMMLA with BFDOT for performance study.
 */
LIBXSMM_API int libxsmm_cpuid_arm_use_bfdot(void);
LIBXSMM_API int libxsmm_cpuid_x86_amx_gemm_enforce_mx1_tile_blocking(void);

/**
 * return the VNNI/Dot-product/Matmul blocking for a specific
 * architecture and datatype */
LIBXSMM_API int libxsmm_cpuid_dot_pack_factor(libxsmm_datatype datatype);

/**
 * Similar to libxsmm_cpuid_x86, but conceptually not arch-specific.
 * The actual code path (as used by LIBXSMM) is determined by
 * libxsmm_[get|set]_target_archid/libxsmm_[get|set]_target_arch.
 */
#if defined(__cplusplus)
LIBXSMM_API int libxsmm_cpuid(libxsmm_cpuid_info* info = NULL);
#else
LIBXSMM_API int libxsmm_cpuid(libxsmm_cpuid_info* info);
#endif

/**
 * Names the CPU architecture given by CPUID.
 * Do not use libxsmm_cpuid() to match the current CPU!
 * Use libxsmm_get_target_archid() instead.
 */
LIBXSMM_API const char* libxsmm_cpuid_name(int id);

/**
 * SIMD vector length (VLEN) in 32-bit elements.
 * Do not use libxsmm_cpuid() to match the current CPU!
 * Use libxsmm_get_target_archid() instead.
 */
LIBXSMM_API int libxsmm_cpuid_vlen32(int id);

/**
 * SIMD vector length (VLEN) measured in Bytes.
 * Do not use libxsmm_cpuid() to match the current CPU!
 * Use libxsmm_get_target_archid() instead.
 */
#define libxsmm_cpuid_vlen(ID) (4 * libxsmm_cpuid_vlen32(ID))

#endif /*LIBXSMM_CPUID_H*/
