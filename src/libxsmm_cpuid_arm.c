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
#include <libxsmm_cpuid.h>
#include <libxsmm_generator.h>
#include <libxsmm_memory.h>
#include <libxsmm_sync.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <signal.h>
#include <setjmp.h>
#if !defined(_WIN32) && !defined(__linux__)
# include <sys/sysctl.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXSMM_CPUID_ARM_BASELINE)
# if defined(__APPLE__) && defined(__arm64__) && 1
#   define LIBXSMM_CPUID_ARM_BASELINE LIBXSMM_AARCH64_APPL_M1
# elif 1
#   define LIBXSMM_CPUID_ARM_BASELINE LIBXSMM_AARCH64_V81
# endif
#endif

#if !defined(LIBXSMM_CPUID_ARM_CNTB_FALLBACK) && 1
# define LIBXSMM_CPUID_ARM_CNTB_FALLBACK
#endif

#if defined(LIBXSMM_PLATFORM_AARCH64)
# if defined(_MSC_VER)
#   define LIBXSMM_CPUID_ARM_ENC16(OP0, OP1, CRN, CRM, OP2) ( \
      (((OP0) & 1) << 14) | \
      (((OP1) & 7) << 11) | \
      (((CRN) & 15) << 7) | \
      (((CRM) & 15) << 3) | \
      (((OP2) & 7) << 0))
#   define ID_AA64ISAR1_EL1 LIBXSMM_CPUID_ARM_ENC16(0b11, 0b000, 0b0000, 0b0110, 0b001)
#   define ID_AA64PFR0_EL1  LIBXSMM_CPUID_ARM_ENC16(0b11, 0b000, 0b0000, 0b0100, 0b000)
#   define MIDR_EL1         LIBXSMM_CPUID_ARM_ENC16(0b11, 0b000, 0b0000, 0b0000, 0b000)
#   define LIBXSMM_CPUID_ARM_MRS(RESULT, ID) RESULT = _ReadStatusReg(ID)
# else
#   define LIBXSMM_CPUID_ARM_MRS(RESULT, ID) __asm__ __volatile__( \
      "mrs %0," LIBXSMM_STRINGIFY(ID) : "=r"(RESULT))
# endif
LIBXSMM_APIVAR_DEFINE(jmp_buf internal_cpuid_arm_jmp_buf);
LIBXSMM_API_INTERN void internal_cpuid_arm_sigill(int /*signum*/);
LIBXSMM_API_INTERN void internal_cpuid_arm_sigill(int signum) {
  void (*const handler)(int) = signal(signum, internal_cpuid_arm_sigill);
  LIBXSMM_ASSERT(SIGILL == signum);
  if (SIG_ERR != handler) longjmp(internal_cpuid_arm_jmp_buf, 1);
}
LIBXSMM_API_INTERN int libxsmm_cpuid_arm_svcntb(void);
LIBXSMM_API_INTERN int libxsmm_cpuid_arm_svcntb(void) {
  int result = 0;
  if (0 == setjmp(internal_cpuid_arm_jmp_buf)) {
# if (defined(__has_builtin) && __has_builtin(__builtin_sve_svcntb)) && 0
    const uint64_t vlen_bytes = __builtin_sve_svcntb();
    if (0 < vlen_bytes && 256 >= vlen_bytes) result = (int)vlen_bytes;
# elif !defined(_MSC_VER) /* TODO: improve condition */
    register uint64_t vlen_bytes __asm__("x0") = 0;
    __asm__ __volatile__(".byte 0xe0, 0xe3, 0x20, 0x04" /*cntb %0*/ : "=r"(vlen_bytes));
    if (0 < vlen_bytes && 256 >= vlen_bytes) result = (int)vlen_bytes;
# endif
  }
  return result;
}
LIBXSMM_API_INTERN char libxsmm_cpuid_arm_vendor(char vendor[], size_t* vendor_size);
LIBXSMM_API_INTERN char libxsmm_cpuid_arm_vendor(char vendor[], size_t* vendor_size) {
  uint64_t result = 0;
  if (0 == setjmp(internal_cpuid_arm_jmp_buf)) {
    LIBXSMM_CPUID_ARM_MRS(result, MIDR_EL1);
  }
  if (NULL != vendor_size && 0 != *vendor_size) {
    if (NULL != vendor) {
# if !defined(_WIN32) && !defined(__linux__)
      if (0 != sysctlbyname("machdep.cpu.brand_string", vendor, vendor_size, NULL, 0) || 0 == *vendor_size) {
        /*const*/ int mib[] = {CTL_HW, HW_MODEL};
        if (0 != sysctl(mib, sizeof(mib) / sizeof(*mib), vendor, vendor_size, NULL, 0) || 0 == *vendor_size) {
          *vendor_size = 0;
          *vendor = '\0';
        }
      }
# else
      *vendor_size = 0;
      *vendor = '\0';
# endif
    }
    else { /* error */
      *vendor_size = 0;
      result = 0;
    }
  }
  return (char)(0xFF & (result >> 24));
}
#endif


LIBXSMM_API int libxsmm_cpuid_arm(libxsmm_cpuid_info* info)
{
  static int result = LIBXSMM_TARGET_ARCH_UNKNOWN;
#if defined(LIBXSMM_PLATFORM_AARCH64)
  if (NULL != info) LIBXSMM_MEMZERO127(info);
  if (LIBXSMM_TARGET_ARCH_UNKNOWN == result) { /* avoid redetecting features */
    void (*const handler)(int) = signal(SIGILL, internal_cpuid_arm_sigill);
# if defined(LIBXSMM_CPUID_ARM_BASELINE)
    result = LIBXSMM_CPUID_ARM_BASELINE;
# else
    result = LIBXSMM_AARCH64_V81;
# endif
    if (SIG_ERR != handler) {
      char vendor_string[1024];
      size_t vendor_size = sizeof(vendor_string);
      const char vendor = libxsmm_cpuid_arm_vendor(vendor_string, &vendor_size);
      uint64_t id_aa64isar1_el1 = 0;
      if (0 == setjmp(internal_cpuid_arm_jmp_buf)) {
        LIBXSMM_CPUID_ARM_MRS(id_aa64isar1_el1, ID_AA64ISAR1_EL1);
      }
      if (LIBXSMM_AARCH64_V82 <= result
        || /* DPB */ 0 != (0xF & id_aa64isar1_el1))
      {
        uint64_t id_aa64pfr0_el1 = 0;
        int no_access = 0; /* try libxsmm_cpuid_arm_svcntb */
        if (LIBXSMM_AARCH64_V82 > result) result = LIBXSMM_AARCH64_V82;
        if (0 == setjmp(internal_cpuid_arm_jmp_buf)) {
          LIBXSMM_CPUID_ARM_MRS(id_aa64pfr0_el1, ID_AA64PFR0_EL1);
        }
        else no_access = 1;
        if (0 != (0xF & (id_aa64pfr0_el1 >> 32)) || 0 != no_access) { /* SVE */
          const int vlen_bytes = libxsmm_cpuid_arm_svcntb();
          switch (vlen_bytes) {
            case 16: { /* SVE 128-bit */
              if (LIBXSMM_AARCH64_SVE128 > result) result = LIBXSMM_AARCH64_SVE128;
            } break;
            case 32: { /* SVE 256-bit */
              const int sve256 = (1 == (0xF & (id_aa64isar1_el1 >> 44))
                ? LIBXSMM_AARCH64_NEOV1 /* BF16 */
                : LIBXSMM_AARCH64_SVE256);
              if (sve256 > result) result = sve256;
            } break;
# if defined(LIBXSMM_CPUID_ARM_CNTB_FALLBACK)
            case 0: /* fallback (hack) */
# endif
            case 64: { /* SVE 512-bit */
              if ('F' == vendor) { /* Fujitsu */
                if (LIBXSMM_AARCH64_A64FX > result) {
# if defined(LIBXSMM_CPUID_ARM_CNTB_FALLBACK)
                  if (0 != libxsmm_verbosity && 0 == vlen_bytes) { /* library code is expected to be mute */
                    fprintf(stderr, "LIBXSMM WARNING: assuming SVE 512-bit vector length!\n");
                  }
# endif
                  result = LIBXSMM_AARCH64_A64FX;
                }
              }
              else
# if defined(LIBXSMM_CPUID_ARM_CNTB_FALLBACK)
              if (64 == vlen_bytes)
# endif
              {
                LIBXSMM_ASSERT(0 == no_access);
                if (LIBXSMM_AARCH64_SVE512 > result) result = LIBXSMM_AARCH64_SVE512;
              }
            } break;
            default: if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
              if (0 != no_access || 0 == vlen_bytes) {
                fprintf(stderr, "LIBXSMM WARNING: cannot determine SVE vector length!\n");
              }
              else {
                fprintf(stderr, "LIBXSMM WARNING: unexpected SVE %i-bit vector length!\n",
                  vlen_bytes * 8);
              }
            }
          }
        }
      }
# if defined(__APPLE__) && defined(__arm64__)
      else if (0 != vendor_size) { /* determine CPU based on vendor-string (everything else failed) */
        if (LIBXSMM_AARCH64_APPL_M1 > result && 0 == strncmp("Apple M1", vendor_string, vendor_size)) {
          result = LIBXSMM_AARCH64_APPL_M1;
        }
      }
# endif
      /* restore original state */
      signal(SIGILL, handler);
    }
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
