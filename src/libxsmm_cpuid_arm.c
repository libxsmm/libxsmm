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
#include <libxsmm_cpuid.h>
#include <libxsmm_generator.h>
#include <libxsmm_memory.h>

#if defined(LIBXSMM_PLATFORM_AARCH64)
# include "libxsmm_main.h"
# include <asm/hwcap.h>
# if (defined(LIBXSMM_BUILD) && (1 < (LIBXSMM_BUILD))) /* GLIBC */
#   include <sys/auxv.h>
# endif
#else
# include <libxsmm_sync.h>
#endif


LIBXSMM_API int libxsmm_cpuid_arm(libxsmm_cpuid_info* info)
{
  static int result = LIBXSMM_TARGET_ARCH_UNKNOWN;
#if defined(LIBXSMM_PLATFORM_X86)
# if !defined(NDEBUG)
  static int error_once = 0;
  if (0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM WARNING: libxsmm_cpuid_arm called on x86 platform!\n");
  }
# endif
  if (NULL != info) LIBXSMM_MEMZERO127(info);
#else
  if (NULL != info) LIBXSMM_MEMZERO127(info);
# if defined(LIBXSMM_PLATFORM_AARCH64)
  result = LIBXSMM_AARCH64_V81;
  {
#   if defined(LIBXSMM_INTERCEPT_DYNAMIC)
    typedef unsigned long (*getcap_fn)(unsigned long);
    static getcap_fn getcap = NULL;
    if (NULL == getcap) {
#     if defined(RTLD_DEFAULT)
      void *const handle = RTLD_DEFAULT;
#     else
      void *const handle = dlopen(NULL, RTLD_LOCAL);
#     endif
      dlerror();
      getcap = (getcap_fn)dlsym(handle, "getauxval");
      if (NULL != dlerror()) getcap = NULL;
#     if !defined(RTLD_DEFAULT)
      if (NULL != handle) result = dlclose(handle);
#     endif
    }
    if (NULL != getcap) {
      const unsigned long capabilities = getcap(AT_HWCAP);
#     if defined(HWCAP_DCPOP)
      if (HWCAP_DCPOP & capabilities) {
#       if defined(HWCAP_SVE)
        if (HWCAP_SVE & capabilities) {
          result = LIBXSMM_AARCH64_A64FX;
        }
        else
#       endif
        {
          result = LIBXSMM_AARCH64_V82;
        }
      } /* HWCAP_DCPOP */
#     endif
    }
    else {
    }
#   endif
  }
# endif
#endif
  return result;
}
