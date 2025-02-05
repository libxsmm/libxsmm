/******************************************************************************
* Copyright (c) 2024, IBM Corporation - All rights reserved.                  *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Will Trojak (IBM Corp.)
******************************************************************************/
#include <libxsmm_cpuid.h>
#include <libxsmm_generator.h>
#include <libxsmm_sync.h>
#include "libxsmm_main.h"

#if !defined(LIBXSMM_CPUID_PPC_BASELINE) && 0
# define LIBXSMM_CPUID_PPC_BASELINE LIBXSMM_PPC64LE_FPF
#endif

LIBXSMM_API int libxsmm_cpuid_ppc(libxsmm_cpuid_info* info)
{
  static int result = LIBXSMM_TARGET_ARCH_UNKNOWN;
#if defined(LIBXSMM_PLATFORM_PPC64LE)
  libxsmm_cpuid_info cpuid_info;
  //size_t model_size = 0;
  if (NULL != info)
  {
    size_t cpuinfo_model_size = sizeof(cpuid_info.model);
    libxsmm_cpuid_model(cpuid_info.model, &cpuinfo_model_size);
    LIBXSMM_ASSERT(0 != cpuinfo_model_size || '\0' == *cpuid_info.model);
    //model_size = cpuinfo_model_size;
    cpuid_info.constant_tsc = 1;
  }
  if (LIBXSMM_TARGET_ARCH_UNKNOWN == result) { /* avoid re-detecting features */
# if defined(LIBXSMM_CPUID_PPC_BASELINE)
    result = LIBXSMM_CPUID_PPC_BASELINE;
# else
    if ( __builtin_cpu_supports("mma") ) {
      result = LIBXSMM_PPC64LE_MMA;
    } else if ( __builtin_cpu_supports("vsx") ) {
      result = LIBXSMM_PPC64LE_VSX;
    } else {
      result = LIBXSMM_PPC64LE_FPF;
    }
# endif
  }

  if (NULL != info) memcpy(info, &cpuid_info, sizeof(cpuid_info));
#else
# if !defined(NDEBUG)
  static int error_once = 0;
  if (0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM WARNING: libxsmm_cpuid_ppc called on non-PPC platform!\n");
  }
# endif
  if (NULL != info) memset(info, 0, sizeof(*info));
#endif
  return result;
}
