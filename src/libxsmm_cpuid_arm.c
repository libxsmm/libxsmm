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
#include <libxsmm_sync.h>

#if defined(_MSC_VER)
# define LIBXSMM_CPUID_ARM_ENC16(OP0, OP1, CRN, CRM, OP2) ( \
    (((OP0) & 1) << 14) | \
    (((OP1) & 7) << 11) | \
    (((CRN) & 15) << 7) | \
    (((CRM) & 15) << 3) | \
    (((OP2) & 7) << 0))
# define ID_AA64ISAR1_EL1 LIBXSMM_CPUID_ARM_ENC16(0b11, 0b000, 0b0000, 0b0110, 0b001)
# define ID_AA64PFR0_EL1  LIBXSMM_CPUID_ARM_ENC16(0b11, 0b000, 0b0000, 0b0100, 0b000)
# define LIBXSMM_CPUID_ARM_MRS(RESULT, ID) RESULT = _ReadStatusReg(ID)
#else
# define LIBXSMM_CPUID_ARM_MRS(RESULT, ID) __asm__ __volatile__( \
    "mrs %0," LIBXSMM_STRINGIFY(ID) : "=r"(RESULT))
#endif


LIBXSMM_API int libxsmm_cpuid_arm(libxsmm_cpuid_info* info)
{
  static int result = LIBXSMM_TARGET_ARCH_UNKNOWN;
#if defined(LIBXSMM_PLATFORM_AARCH64)
  /* avoid redetecting features */
  if (LIBXSMM_TARGET_ARCH_UNKNOWN == result) {
    result = LIBXSMM_AARCH64_V81;
    { uint64_t capability; /* 64-bit value */
      LIBXSMM_CPUID_ARM_MRS(capability, ID_AA64ISAR1_EL1);
      if (0xF & capability) { /* DPB */
        LIBXSMM_CPUID_ARM_MRS(capability, ID_AA64PFR0_EL1);
        if (0xF & (capability >> 32)) { /* SVE */
          result = LIBXSMM_AARCH64_A64FX;
        }
        else {
          result = LIBXSMM_AARCH64_V82;
        }
      }
    }
    if (NULL != info) LIBXSMM_MEMZERO127(info);
  }
#else
# if !defined(NDEBUG)
  static int error_once = 0;
  if (0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM WARNING: libxsmm_cpuid_arm called on non-ARM platform!\n");
  }
# endif
  if (NULL != info) LIBXSMM_MEMZERO127(info);
#endif
  return result;
}
