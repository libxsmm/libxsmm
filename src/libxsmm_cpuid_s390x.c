/******************************************************************************
* Copyright (c), 2025 IBM Corporation - All rights reserved.                  *
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

#if !defined(LIBXSMM_CPUID_S390X_BASELINE) && 0
# define LIBXSMM_CPUID_S390X_BASELINE LIBXSMM_S390X_Z15
#endif

LIBXSMM_API int libxsmm_cpuid_s390x(libxsmm_cpuid_info* info)
{
  static int result = LIBXSMM_TARGET_ARCH_UNKNOWN;
#if defined(LIBXSMM_PLATFORM_S390X)
  libxsmm_cpuid_info cpuid_info;
  //size_t model_size = 0;
  if (NULL != info)
  {
    size_t cpuinfo_model_size = sizeof(cpuid_info.model);
    libxsmm_cpuid_model(cpuid_info.model, &cpuinfo_model_size);
    LIBXSMM_ASSERT(0 != cpuinfo_model_size || '\0' == *cpuid_info.model);
    cpuid_info.constant_tsc = 1;
  }
  if (LIBXSMM_TARGET_ARCH_UNKNOWN == result) { /* avoid re-detecting features */
# if defined(LIBXSMM_CPUID_S390X_BASELINE)
    result = LIBXSMM_CPUID_S390X_BASELINE;
# elif defined(__ARCH__) && ( __ARCH__ == 13 )
    result = LIBXSMM_S390X_Z15
# elif defined(__ARCH__) && ( __ARCH__ == 14 )
    result = LIBXSMM_S390X_Z16
# endif
  }

  if (NULL != info) memcpy(info, &cpuid_info, sizeof(cpuid_info));
#else
# if !defined(NDEBUG)
  static int error_once = 0;
  if (0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM WARNING: libxsmm_cpuid_s390x called on non-s390x platform!\n");
  }
# endif
  if (NULL != info) memset(info, 0, sizeof(*info));
#endif
  return result;
}
