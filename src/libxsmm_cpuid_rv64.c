/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Siddharth Rai, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm_cpuid.h>
#include <libxsmm_generator.h>
#include <libxsmm_sync.h>
#include "libxsmm_main.h"

#include <signal.h>
#include <setjmp.h>

#define VLMAX (65536)

LIBXSMM_API int libxsmm_cpuid_rv64(libxsmm_cpuid_info* info)
{
  int mvl;
  libxsmm_cpuid_info cpuid_info;
  size_t cpuinfo_model_size = sizeof(cpuid_info.model);
#ifdef LIBXSMM_PLATFORM_RV64
  int rvl = VLMAX;
  __asm__(".option arch, +zve64x\n\t""vsetvli %0, %1, e8, m1, ta, ma\n": "=r"(mvl): "r" (rvl));
#else
  mvl = 0;
#endif

  libxsmm_cpuid_model(cpuid_info.model, &cpuinfo_model_size);
  LIBXSMM_ASSERT(0 != cpuinfo_model_size || '\0' == *cpuid_info.model);
  cpuid_info.constant_tsc = 1;

  /* Get MVL in bits */
  switch (mvl * 8){
    case 128:
      mvl = LIBXSMM_RV64_MVL128;
      break;

    case 256:
      mvl = LIBXSMM_RV64_MVL256;
      break;

    default:
      mvl = LIBXSMM_RV64_MVL128;
      break;
  }

  if (NULL != info) memcpy(info, &cpuid_info, sizeof(cpuid_info));

  return mvl;
}

LIBXSMM_API unsigned int libxsmm_gemm_prefetch_reuse_a(void){
  const char *const env_gemm_prefetch_reuse_a = getenv("LIBXSMM_GEMM_PREFETCH_REUSE_A");
  unsigned int result = (env_gemm_prefetch_reuse_a == 0) ? 0 : atoi(env_gemm_prefetch_reuse_a);
  return result;
}

LIBXSMM_API unsigned int libxsmm_gemm_prefetch_reuse_b(void){
  const char *const env_gemm_prefetch_reuse_b = getenv("LIBXSMM_GEMM_PREFETCH_REUSE_B");
  unsigned int result = (env_gemm_prefetch_reuse_b == 0) ? 0 : atoi(env_gemm_prefetch_reuse_b);
  return result;
}

LIBXSMM_API unsigned int libxsmm_gemm_prefetch_reuse_c(void){
  const char *const env_gemm_prefetch_reuse_c = getenv("LIBXSMM_GEMM_PREFETCH_REUSE_C");
  unsigned int result = (env_gemm_prefetch_reuse_c == 0) ? 0 : atoi(env_gemm_prefetch_reuse_c);
  return result;
}

LIBXSMM_API unsigned int libxsmm_gemm_prefetch_a(void){
  const char *const env_gemm_prefetch_a = getenv("LIBXSMM_GEMM_PREFETCH_A");
  unsigned int result = (env_gemm_prefetch_a == 0) ? 0 : atoi(env_gemm_prefetch_a);
  return result;
}

LIBXSMM_API unsigned int libxsmm_gemm_prefetch_b(void){
  const char *const env_gemm_prefetch_b = getenv("LIBXSMM_GEMM_PREFETCH_B");
  unsigned int result = (env_gemm_prefetch_b == 0) ? 0 : atoi(env_gemm_prefetch_b);
  return result;
}

LIBXSMM_API unsigned int libxsmm_gemm_m_prefetch_stride(void){
  const char *const env_gemm_m_prefetch_stride = getenv("LIBXSMM_GEMM_M_PREFETCH_STRIDE");
  unsigned int result = (env_gemm_m_prefetch_stride == 0) ? 0 : atoi(env_gemm_m_prefetch_stride);
  return result;
}

#undef VLMAX
