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
#define LIBXSMM_X86_AVX512_VL128_SKX  1041
#define LIBXSMM_X86_AVX512_VL256_SKX  1051
#define LIBXSMM_X86_AVX512_VL256_CLX  1052
#define LIBXSMM_X86_AVX512_VL256_CPX  1053
#define LIBXSMM_X86_AVX512_SKX        1101
#define LIBXSMM_X86_AVX512_CLX        1102
#define LIBXSMM_X86_AVX512_CPX        1103
#define LIBXSMM_X86_AVX512_SPR        1104
#define LIBXSMM_X86_AVX512_GNR        1105
#define LIBXSMM_X86_ALLFEAT           1999
#define LIBXSMM_AARCH64_V81           2001 /* Baseline */
#define LIBXSMM_AARCH64_V82           2002 /* A64FX minus SVE */
#define LIBXSMM_AARCH64_APPL_M1       2101 /* Apple M1 */
#define LIBXSMM_AARCH64_SVE128        2201 /* SVE 128 */
#define LIBXSMM_AARCH64_NEOV2         2202 /* Neoverse V2, NVIDIA Grace, Graviton 4 */
#define LIBXSMM_AARCH64_SVE256        2301 /* SVE 256 */
#define LIBXSMM_AARCH64_NEOV1         2302 /* Neoverse V1, Graviton 3 */
#define LIBXSMM_AARCH64_SVE512        2401 /* SVE 512 */
#define LIBXSMM_AARCH64_A64FX         2402 /* A64FX */
#define LIBXSMM_AARCH64_ALLFEAT       2999

 /** Zero-initialized structure; assumes conservative properties. */
LIBXSMM_EXTERN_C typedef struct libxsmm_cpuid_info {
  char model[1024]; /** CPU-name (OS-specific implementation). */
  int constant_tsc; /** Timer stamp counter is monotonic. */
#if defined(LIBXSMM_PLATFORM_X86)
  int has_context;  /** Context switches are permitted. */
#endif
} libxsmm_cpuid_info;

/** Returns the target architecture and instruction set extensions. */
LIBXSMM_API int libxsmm_cpuid_x86(libxsmm_cpuid_info* LIBXSMM_ARGDEF(info, NULL));
LIBXSMM_API int libxsmm_cpuid_arm(libxsmm_cpuid_info* LIBXSMM_ARGDEF(info, NULL));

/**
 * TODO: limited lifetime API until we have a fully-fledged ARM CPU flags test.
 */
LIBXSMM_API unsigned int libxsmm_cpuid_arm_mmla_gemm_pack_b_to_vnnit_on_stack(void);

/**
 * TODO: limited lifetime API until we have a fully-fledged ARM CPU flags test.
 * Might be needed to overwrite BFMMLA with BFDOT for performance study.
 */
LIBXSMM_API int libxsmm_cpuid_arm_use_bfdot(void);
LIBXSMM_API int libxsmm_cpuid_x86_use_high_prec_eltwise_approx(void);
LIBXSMM_API int libxsmm_cpuid_x86_amx_gemm_enforce_mx1_tile_blocking(void);
LIBXSMM_API unsigned int libxsmm_cpuid_x86_srf_gemm_set_n_max_blocking(void);
LIBXSMM_API int libxsmm_cpuid_arm_use_i8dot(void);
LIBXSMM_API int libxsmm_cpuid_x86_bf8_gemm_via_stack(void);

/**
 * return the VNNI/Dot-product/Matmul blocking for a specific
 * architecture and datatype */
LIBXSMM_API int libxsmm_cpuid_dot_pack_factor(libxsmm_datatype datatype);

/**
 * Similar to libxsmm_cpuid_x86, but conceptually not arch-specific.
 * The actual code path (as used by LIBXSMM) is determined by
 * libxsmm_[get|set]_target_archid/libxsmm_[get|set]_target_arch.
 */
LIBXSMM_API int libxsmm_cpuid(libxsmm_cpuid_info* LIBXSMM_ARGDEF(info, NULL));

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
