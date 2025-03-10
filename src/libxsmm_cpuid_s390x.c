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
# define LIBXSMM_CPUID_S390X_BASELINE LIBXSMM_S390X_ARCH11
#endif

#if (defined(__zarch__) && 0 != (__zarch__)) || \
    (defined(__s390x__) && 0 != (__s390x__))
# include <sys/auxv.h>
# if defined(HWCAP_S390_VX)
#  define LIBXSMM_S390X_HWCAP_VX HWCAP_S390_VX
# else
#  define LIBXSMM_S390X_HWCAP_VX 0
# endif
# if defined(HWCAP_S390_VXRS_EXT)
#  define LIBXSMM_S390X_HWCAP_VXEX HWCAP_S390_VXRS_EXT
# else
#  define LIBXSMM_S390X_HWCAP_VXEX 0
# endif

# if defined(HWCAP_S390_VXRS_EXT2)
#  define LIBXSMM_S390X_HWCAP_VXEX2 HWCAP_S390_VXRS_EXT2
# else
#  define LIBXSMM_S390X_HWCAP_VXEX2 0
# endif
# if defined(HWCAP_S390_NNPA)
#  define LIBXSMM_S390X_HWCAP_NNPA HWCAP_S390_NNPA
# else
#  define LIBXSMM_S390X_HWCAP_NNPA 0
# endif
#else
# define LIBXSMM_S390X_HWCAP_VX 0
# define LIBXSMM_S390X_HWCAP_VXEX 0
# define LIBXSMM_S390X_HWCAP_VXEX2 0
# define LIBXSMM_S390X_HWCAP_NNPA 0
#endif

LIBXSMM_API int libxsmm_cpuid_s390x(libxsmm_cpuid_info* info)
{
  static int result = LIBXSMM_TARGET_ARCH_UNKNOWN;
#if defined(LIBXSMM_PLATFORM_S390X)
  libxsmm_cpuid_info cpuid_info;
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
# else
    /* Querry this way as hardware and linux kernel support is needed */
    if ( getauxval(AT_HWCAP) & LIBXSMM_S390X_HWCAP_NNPA ) {
      result = LIBXSMM_S390X_ARCH14;
    } else if ( getauxval(AT_HWCAP) & LIBXSMM_S390X_HWCAP_VXEX2 ) {
      result = LIBXSMM_S390X_ARCH13;
    } else if ( getauxval(AT_HWCAP) & LIBXSMM_S390X_HWCAP_VXEX ) {
      result = LIBXSMM_S390X_ARCH12;
    } else if ( getauxval(AT_HWCAP) & LIBXSMM_S390X_HWCAP_VX ) {
      result = LIBXSMM_S390X_ARCH11;
    } else {
      fprintf(stderr, "LIBXSMM WARNING: s390x arch facilities not supported\n");
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
    fprintf(stderr, "LIBXSMM WARNING: libxsmm_cpuid_s390x called on non-s390x platform!\n");
  }
# endif
  if (NULL != info) memset(info, 0, sizeof(*info));
#endif
  return result;
}
