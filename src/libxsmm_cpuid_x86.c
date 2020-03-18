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
#include <libxsmm_intrinsics_x86.h>
#include <libxsmm_generator.h>
#include <libxsmm_memory.h>

#if defined(LIBXSMM_PLATFORM_SUPPORTED)
/* XGETBV: receive results (EAX, EDX) for eXtended Control Register (XCR). */
/* CPUID, receive results (EAX, EBX, ECX, EDX) for requested FUNCTION/SUBFN. */
#if defined(_MSC_VER) /*defined(_WIN32) && !defined(__GNUC__)*/
#   define LIBXSMM_XGETBV(XCR, EAX, EDX) { \
      unsigned long long libxsmm_xgetbv_ = _xgetbv(XCR); \
      EAX = (int)libxsmm_xgetbv_; \
      EDX = (int)(libxsmm_xgetbv_ >> 32); \
    }
#   define LIBXSMM_CPUID_X86(FUNCTION, SUBFN, EAX, EBX, ECX, EDX) { \
      int libxsmm_cpuid_x86_[/*4*/] = { 0, 0, 0, 0 }; \
      __cpuidex(libxsmm_cpuid_x86_, FUNCTION, SUBFN); \
      EAX = (unsigned int)libxsmm_cpuid_x86_[0]; \
      EBX = (unsigned int)libxsmm_cpuid_x86_[1]; \
      ECX = (unsigned int)libxsmm_cpuid_x86_[2]; \
      EDX = (unsigned int)libxsmm_cpuid_x86_[3]; \
    }
# elif defined(__GNUC__) || !defined(_CRAYC)
#   if (64 > (LIBXSMM_BITS))
      LIBXSMM_EXTERN LIBXSMM_RETARGETABLE int __get_cpuid( /* prototype */
        unsigned int, unsigned int*, unsigned int*, unsigned int*, unsigned int*);
#     define LIBXSMM_XGETBV(XCR, EAX, EDX) EAX = (EDX) = 0xFFFFFFFF
#     define LIBXSMM_CPUID_X86(FUNCTION, SUBFN, EAX, EBX, ECX, EDX) \
        EAX = (EBX) = (EDX) = 0; ECX = (SUBFN); \
        __get_cpuid(FUNCTION, &(EAX), &(EBX), &(ECX), &(EDX))
#   else /* 64-bit */
#     define LIBXSMM_XGETBV(XCR, EAX, EDX) __asm__ __volatile__( \
        ".byte 0x0f, 0x01, 0xd0" /*xgetbv*/ : "=a"(EAX), "=d"(EDX) : "c"(XCR) \
      )
#     define LIBXSMM_CPUID_X86(FUNCTION, SUBFN, EAX, EBX, ECX, EDX) \
        __asm__ __volatile__ (".byte 0x0f, 0xa2" /*cpuid*/ \
        : "=a"(EAX), "=b"(EBX), "=c"(ECX), "=d"(EDX) \
        : "a"(FUNCTION), "b"(0), "c"(SUBFN), "d"(0) \
      )
#   endif
# else /* legacy Cray Compiler */
#   define LIBXSMM_XGETBV(XCR, EAX, EDX) EAX = (EDX) = 0
#   define LIBXSMM_CPUID_X86(FUNCTION, SUBFN, EAX, EBX, ECX, EDX) EAX = (EBX) = (ECX) = (EDX) = 0
# endif
#endif

#define LIBXSMM_CPUID_CHECK(VALUE, CHECK) ((CHECK) == ((CHECK) & (VALUE)))


LIBXSMM_API int libxsmm_cpuid_x86(libxsmm_cpuid_x86_info* info)
{
  static int result = LIBXSMM_TARGET_ARCH_UNKNOWN;
#if !defined(LIBXSMM_PLATFORM_SUPPORTED)
  if (NULL != info) LIBXSMM_MEMZERO127(info);
#else
  unsigned int eax, ebx, ecx, edx;
  LIBXSMM_CPUID_X86(0, 0/*ecx*/, eax, ebx, ecx, edx);
  if (1 <= eax) { /* CPUID max. leaf */
    if (LIBXSMM_TARGET_ARCH_UNKNOWN == result) { /* detect CPU-feature only once */
      int feature_cpu = LIBXSMM_X86_GENERIC, feature_os = LIBXSMM_X86_GENERIC;
      unsigned int maxleaf = eax;
      LIBXSMM_CPUID_X86(1, 0/*ecx*/, eax, ebx, ecx, edx);
      /* Check for CRC32 (this is not a proper test for SSE 4.2 as a whole!) */
      if (LIBXSMM_CPUID_CHECK(ecx, 0x00100000)) {
        if (LIBXSMM_CPUID_CHECK(ecx, 0x10000000)) { /* AVX(0x10000000) */
          if (LIBXSMM_CPUID_CHECK(ecx, 0x00001000)) { /* FMA(0x00001000) */
            unsigned int ecx2;
            LIBXSMM_CPUID_X86(7, 0/*ecx*/, eax, ebx, ecx2, edx);
            /* AVX512F(0x00010000), AVX512CD(0x10000000) */
            if (LIBXSMM_CPUID_CHECK(ebx, 0x10010000)) { /* Common */
              /* AVX512DQ(0x00020000), AVX512BW(0x40000000), AVX512VL(0x80000000) */
              if (LIBXSMM_CPUID_CHECK(ebx, 0xC0020000)) { /* AVX512-Core */
                if (LIBXSMM_CPUID_CHECK(ecx2, 0x00000800)) { /* VNNI */
# if 0 /* no check required yet */
                  unsigned int ecx3;
                  LIBXSMM_CPUID_X86(7, 1/*ecx*/, eax, ebx, ecx3, edx);
# else
                  LIBXSMM_CPUID_X86(7, 1/*ecx*/, eax, ebx, ecx2, edx);
# endif
                  if (LIBXSMM_CPUID_CHECK(eax, 0x00000020)) { /* BF16 */
                    feature_cpu = LIBXSMM_X86_AVX512_CPX;
                  }
                  else feature_cpu = LIBXSMM_X86_AVX512_CLX; /* CLX */
                }
                else feature_cpu = LIBXSMM_X86_AVX512_CORE; /* SKX */
              }
              /* AVX512PF(0x04000000), AVX512ER(0x08000000) */
              else if (LIBXSMM_CPUID_CHECK(ebx, 0x0C000000)) { /* AVX512-MIC */
                if (LIBXSMM_CPUID_CHECK(edx, 0x0000000C)) { /* KNM */
                  feature_cpu = LIBXSMM_X86_AVX512_KNM;
                }
                else feature_cpu = LIBXSMM_X86_AVX512_MIC; /* KNL */
              }
              else feature_cpu = LIBXSMM_X86_AVX512; /* AVX512-Common */
            }
            else feature_cpu = LIBXSMM_X86_AVX2;
          }
          else feature_cpu = LIBXSMM_X86_AVX;
        }
        else feature_cpu = LIBXSMM_X86_SSE4;
      }
# if !defined(LIBXSMM_INTRINSICS_DEBUG)
      LIBXSMM_ASSERT_MSG(LIBXSMM_STATIC_TARGET_ARCH <= LIBXSMM_MAX(LIBXSMM_X86_SSE3, feature_cpu),
        /* TODO: confirm SSE3 */"missed detecting ISA extensions");
      /* coverity[dead_error_line] */
      if (LIBXSMM_STATIC_TARGET_ARCH > feature_cpu) feature_cpu = LIBXSMM_STATIC_TARGET_ARCH;
# endif
      /* XSAVE/XGETBV(0x04000000), OSXSAVE(0x08000000) */
      if (LIBXSMM_CPUID_CHECK(ecx, 0x0C000000)) { /* OS SSE support */
        feature_os = LIBXSMM_MIN(LIBXSMM_X86_SSE4, feature_cpu);
        if (LIBXSMM_X86_AVX <= feature_cpu) {
          LIBXSMM_XGETBV(0, eax, edx);
          if (LIBXSMM_CPUID_CHECK(eax, 0x00000006)) { /* OS XSAVE 256-bit */
            feature_os = LIBXSMM_MIN(LIBXSMM_X86_AVX2, feature_cpu);
            if (LIBXSMM_X86_AVX512 <= feature_cpu && 7 <= maxleaf
              && LIBXSMM_CPUID_CHECK(eax, 0x000000E0)) /* OS XSAVE 512-bit */
            {
              feature_os = feature_cpu; /* unlimited */
            }
          }
        }
      }
      else feature_os = LIBXSMM_TARGET_ARCH_GENERIC;
      if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
        const int target_vlen32 = libxsmm_cpuid_vlen32(feature_cpu);
        const char *const compiler_support = (libxsmm_cpuid_vlen32(LIBXSMM_MAX_STATIC_TARGET_ARCH) < target_vlen32
          ? "" : (((2 <= libxsmm_verbosity || 0 > libxsmm_verbosity) && LIBXSMM_MAX_STATIC_TARGET_ARCH < feature_cpu)
            ? "highly " : NULL));
        int warnings = 0;
# if !defined(NDEBUG) && defined(__OPTIMIZE__)
        fprintf(stderr, "LIBXSMM WARNING: library is optimized without -DNDEBUG and contains debug code!\n");
        ++warnings;
# endif
        if (NULL != compiler_support) {
          const char *const name = libxsmm_cpuid_name( /* exclude MIC when running on Core processors */
            (LIBXSMM_X86_AVX512 <= LIBXSMM_MAX_STATIC_TARGET_ARCH && LIBXSMM_X86_AVX512_CORE <= feature_cpu)
              ? LIBXSMM_X86_AVX2 : LIBXSMM_MAX_STATIC_TARGET_ARCH);
          fprintf(stderr, "LIBXSMM WARNING: %soptimized non-JIT code paths are limited to \"%s\"!\n", compiler_support, name);
          ++warnings;
        }
        if (LIBXSMM_STATIC_TARGET_ARCH < feature_cpu && feature_os < feature_cpu) {
          fprintf(stderr, "LIBXSMM WARNING: detected CPU features are not permitted by the OS!\n");
          ++warnings;
        }
        if (0 != warnings) fprintf(stderr, "\n");
      }
# if 0 /* permitted features */
      result = LIBXSMM_MIN(feature_cpu, feature_os);
# else /* opportunistic */
      result = feature_cpu;
# endif
    }
    if (NULL != info) {
      LIBXSMM_CPUID_X86(0x80000007, 0/*ecx*/, eax, ebx, ecx, edx);
      info->constant_tsc = LIBXSMM_CPUID_CHECK(edx, 0x00000100);
    }
  }
  else {
    if (NULL != info) LIBXSMM_MEMZERO127(info);
    result = LIBXSMM_X86_GENERIC;
  }
#endif
  return result;
}


LIBXSMM_API int libxsmm_cpuid(void)
{
  return libxsmm_cpuid_x86(NULL/*info*/);
}


LIBXSMM_API const char* libxsmm_cpuid_name(int id)
{
  const char* target_arch = NULL;
  switch (id) {
    case LIBXSMM_X86_AVX512_CPX: {
      target_arch = "cpx";
    } break;
    case LIBXSMM_X86_AVX512_CLX: {
      target_arch = "clx";
    } break;
    case LIBXSMM_X86_AVX512_CORE: {
      target_arch = "skx";
    } break;
    case LIBXSMM_X86_AVX512_KNM: {
      target_arch = "knm";
    } break;
    case LIBXSMM_X86_AVX512_MIC: {
      target_arch = "knl";
    } break;
    case LIBXSMM_X86_AVX512: {
      /* TODO: rework BE to use target ID instead of set of strings (target_arch = "avx3") */
      target_arch = "hsw";
    } break;
    case LIBXSMM_X86_AVX2: {
      target_arch = "hsw";
    } break;
    case LIBXSMM_X86_AVX: {
      target_arch = "snb";
    } break;
    case LIBXSMM_X86_SSE4: {
      /* TODO: rework BE to use target ID instead of set of strings (target_arch = "sse4") */
      target_arch = "wsm";
    } break;
    case LIBXSMM_X86_SSE3: {
      /* WSM includes SSE4, but BE relies on SSE3 only,
       * hence we enter "wsm" path starting with SSE3.
       */
      target_arch = "wsm";
    } break;
    case LIBXSMM_TARGET_ARCH_GENERIC: {
      target_arch = "generic";
    } break;
    default: if (LIBXSMM_X86_GENERIC <= id) {
      target_arch = "x86";
    }
    else {
      target_arch = "unknown";
    }
  }

  LIBXSMM_ASSERT(NULL != target_arch);
  return target_arch;
}


LIBXSMM_API int libxsmm_cpuid_vlen32(int id)
{
  int result;
  if (LIBXSMM_X86_AVX512 <= id) {
    result = 16;
  }
  else if (LIBXSMM_X86_AVX <= id) {
    result = 8;
  }
  else if (LIBXSMM_X86_SSE3 <= id) {
    result = 4;
  }
  else { /* scalar */
    result = 1;
  }
  return result;
}

