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


LIBXSMM_API_INTERN
unsigned int libxsmm_cpuid_s390x_stfle( unsigned long     *i_fle,
                                        const unsigned int i_max_len ) {
  i_fle[0] = i_max_len - 1;
  for ( unsigned int i = 1 ; i < i_max_len + 1 ; ++i ) {
    i_fle
      [i] = 0;
  }
#if defined(__zarch__) || defined(__s390x__)
  __asm__ volatile ( "lg 0,%[len]\n"
                     "stfle %[fac]\n"
                     "stg 0,%[len]\n"
                     : [fac] "=QS"(*(unsigned long(*)[i_max_len])&i_fle[1]),
                       [len] "+RT"(i_fle[0])
                     :
                     : "%r0", "cc"
                     );
#endif
  return (unsigned int)(i_fle[0] + 1);
}


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
# elif defined(__zarch__) || defined(__s390x__)
    unsigned int l_max_len = 8;
    unsigned long l_fle[l_max_len + 1];
    unsigned int l_fle_len = libxsmm_cpuid_s390x_stfle( l_fle, l_max_len );

    if ( l_fle_len >= 3 ) {
      /* Test for Neural-network-processing-assist */
      if ( ( ( l_fle[3] & ( 0x01UL << 26 ) ) >> 26 ) == 1 ) {
        result = LIBXSMM_S390X_ARCH14;
      /* Test for Vector-enhancements facility 2 */
      } else if ( ( ( l_fle[3] & ( 0x01UL << 43 ) ) >> 43 ) == 1 ) {
        result = LIBXSMM_S390X_ARCH13;
      /* Test for Vector-enhancements facility 1 */
      } else if ( ( ( l_fle[3] & ( 0x01UL << 56 ) ) >> 56 ) == 1 ) {
        result = LIBXSMM_S390X_ARCH12;
       /* Test for Vector facility for z/Architecture */
      } else if (  ( ( l_fle[3] & ( 0x01UL << 62 ) ) >> 62 ) == 1 ) {
        result = LIBXSMM_S390X_ARCH11;
      } else {
        fprintf(stderr, "LIBXSMM WARNING: s390x arch facilities not supported\n");
        for ( unsigned int i = 0; i < l_fle_len ; ++i) {
          fprintf(stderr, "LIBXSMM WARNING: S390X FLE[%d] = 0x%016lx\n", i, l_fle[i+1]);
        }
      }
    } else {
      fprintf(stderr, "LIBXSMM WARNING: s390x arch facilities not supported\n");
      for ( unsigned int i = 0; i < l_fle_len ; ++i) {
        fprintf(stderr, "LIBXSMM WARNING: S390X FLE[%d] = 0x%016lx\n", i, l_fle[i+1]);
      }
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
