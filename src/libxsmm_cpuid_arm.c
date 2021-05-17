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

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <signal.h>
#include <setjmp.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

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


#if defined(LIBXSMM_PLATFORM_AARCH64)
LIBXSMM_APIVAR_DEFINE(jmp_buf internal_cpuid_arm_jmp_buf);

LIBXSMM_API_INTERN void internal_cpuid_arm_sigill(int /*signum*/);
LIBXSMM_API_INTERN void internal_cpuid_arm_sigill(int signum)
{
  void (*const handler)(int) = signal(signum, internal_cpuid_arm_sigill);
  LIBXSMM_ASSERT(SIGILL == signum);
  if (SIG_ERR != handler) longjmp(internal_cpuid_arm_jmp_buf, 1);
}
#endif


LIBXSMM_API int libxsmm_cpuid_arm(libxsmm_cpuid_info* info)
{
  static int result = LIBXSMM_TARGET_ARCH_UNKNOWN;
#if defined(LIBXSMM_PLATFORM_AARCH64)
#if defined(__APPLE__) && defined(__arm64__)
  result = LIBXSMM_AARCH64_V81;
#else
  if (LIBXSMM_TARGET_ARCH_UNKNOWN == result) { /* avoid redetecting features */
    void (*const handler)(int) = signal(SIGILL, internal_cpuid_arm_sigill);
    result = LIBXSMM_AARCH64_V81;
    if (SIG_ERR != handler) {
      uint64_t capability; /* 64-bit value */
      if (0 == setjmp(internal_cpuid_arm_jmp_buf)) {
        LIBXSMM_CPUID_ARM_MRS(capability, ID_AA64ISAR1_EL1);
        if (0xF & capability) { /* DPB */
          result = LIBXSMM_AARCH64_V82;
          if (0 == setjmp(internal_cpuid_arm_jmp_buf)) {
            LIBXSMM_CPUID_ARM_MRS(capability, ID_AA64PFR0_EL1);
            if (0xF & (capability >> 32)) { /* SVE */
              result = LIBXSMM_AARCH64_A64FX;
            }
          }
        }
      }
      /* restore original state */
      signal(SIGILL, handler);
    }
    if (NULL != info) LIBXSMM_MEMZERO127(info);
  }
#endif
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
