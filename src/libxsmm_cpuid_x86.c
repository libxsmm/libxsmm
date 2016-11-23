/******************************************************************************
** Copyright (c) 2016, Intel Corporation                                     **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm_intrinsics_x86.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <assert.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

/** Execute CPUID, and receive results (EAX, EBX, ECX, EDX) for requested FUNCTION. */
#if defined(__GNUC__) || defined(__PGI)
# if (4294967295U < (__SIZE_MAX__)) || defined(_CRAYC)
#   define LIBXSMM_CPUID_X86(FUNCTION, EAX, EBX, ECX, EDX) __asm__ __volatile__ ( \
      ".byte 0x0f, 0xa2" /*cpuid*/ : "=a"(EAX), "=b"(EBX), "=c"(ECX), "=d"(EDX) : "a"(FUNCTION), "c"(0) \
    )
# else
LIBXSMM_EXTERN LIBXSMM_RETARGETABLE int __get_cpuid(unsigned int, unsigned int*, unsigned int*, unsigned int*, unsigned int*);
#   define LIBXSMM_CPUID_X86(FUNCTION, EAX, EBX, ECX, EDX) __get_cpuid(FUNCTION, &(EAX), &(EBX), &(ECX), &(EDX))
# endif
#elif !defined(_CRAYC)
# define LIBXSMM_CPUID_X86(FUNCTION, EAX, EBX, ECX, EDX) { \
    int libxsmm_cpuid_x86_[4]; \
    __cpuid(libxsmm_cpuid_x86_, FUNCTION); \
    EAX = (unsigned int)libxsmm_cpuid_x86_[0]; \
    EBX = (unsigned int)libxsmm_cpuid_x86_[1]; \
    ECX = (unsigned int)libxsmm_cpuid_x86_[2]; \
    EDX = (unsigned int)libxsmm_cpuid_x86_[3]; \
  }
#else
# define LIBXSMM_CPUID_X86(FUNCTION, EAX, EBX, ECX, EDX) LIBXSMM_X86_AVX
#endif

/** Execute the XGETBV (x86), and receive results (EAX, EDX) for req. eXtended Control Register (XCR). */
#if defined(__GNUC__) || defined(__PGI)
# define LIBXSMM_XGETBV(XCR, EAX, EDX) __asm__ __volatile__( \
    ".byte 0x0f, 0x01, 0xd0" /*xgetbv*/ : "=a"(EAX), "=d"(EDX) : "c"(XCR) \
  )
#elif !defined(_CRAYC)
# define LIBXSMM_XGETBV(XCR, EAX, EDX) { \
    unsigned long long libxsmm_xgetbv_ = _xgetbv(XCR); \
    EAX = (int)libxsmm_xgetbv_; \
    EDX = (int)(libxsmm_xgetbv_ >> 32); \
  }
#else
# define LIBXSMM_XGETBV(XCR, EAX, EDX)
#endif


LIBXSMM_API_DEFINITION int libxsmm_cpuid_x86(void)
{
  int target_arch = LIBXSMM_STATIC_TARGET_ARCH;
  unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;

  LIBXSMM_CPUID_X86(0, eax, ebx, ecx, edx);
  if (1 <= eax) { /* CPUID */
    LIBXSMM_CPUID_X86(1, eax, ebx, ecx, edx);

    /* XSAVE/XGETBV(0x04000000), OSXSAVE(0x08000000) */
    if (0x0C000000 == (0x0C000000 & ecx)) {
      /* Check for CRC32 (this is not a proper test for SSE 4.2 as a whole!) */
      if (0x00100000 == (0x00100000 & ecx)) {
        target_arch = LIBXSMM_X86_SSE4_2;
      }
      LIBXSMM_XGETBV(0, eax, edx);

      if (0x00000006 == (0x00000006 & eax)) { /* OS XSAVE 256-bit */
        if (0x000000E0 == (0x000000E0 & eax)) { /* OS XSAVE 512-bit */
          LIBXSMM_CPUID_X86(7, eax, ebx, ecx, edx);

          /* AVX512F(0x00010000), AVX512CD(0x10000000) */
          if (0x10010000 == (0x10010000 & ebx)) { /* Common */
            /* AVX512DQ(0x00020000), AVX512BW(0x40000000), AVX512VL(0x80000000) */
            if (0xC0020000 == (0xC0020000 & ebx)) { /* SKX (Core) */
              target_arch = LIBXSMM_X86_AVX512_CORE;
            }
            /* AVX512PF(0x04000000), AVX512ER(0x08000000) */
            else if (0x0C000000 == (0x0C000000 & ebx)) { /* KNL (MIC) */
              target_arch = LIBXSMM_X86_AVX512_MIC;
            }
            else { /* Common */
              target_arch = LIBXSMM_X86_AVX512;
            }
          }
        }
        else if (0x10000000 == (0x10000000 & ecx)) { /* AVX(0x10000000) */
          if (0x00001000 == (0x00001000 & ecx)) { /* FMA(0x00001000) */
            target_arch = LIBXSMM_X86_AVX2;
          }
          else {
            target_arch = LIBXSMM_X86_AVX;
          }
        }
      }
    }
  }

#if defined(LIBXSMM_STATIC_TARGET_ARCH)
  /* check if procedure obviously failed to detect the highest available instruction set extension */
  assert(LIBXSMM_STATIC_TARGET_ARCH <= target_arch);
#endif

  return LIBXSMM_MAX(target_arch, LIBXSMM_STATIC_TARGET_ARCH);
}

