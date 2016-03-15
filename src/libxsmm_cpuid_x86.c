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
#include "libxsmm_cpuid_x86.h"
#include "libxsmm_intrinsics_x86.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#if !defined(NDEBUG)
# include <assert.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int libxsmm_cpuid_x86(const char** archid)
{
  int target_arch = LIBXSMM_STATIC_TARGET_ARCH;
  unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
  if (archid) *archid = "x86";

  LIBXSMM_CPUID_X86(0, eax, ebx, ecx, edx);
  if (1 <= eax) { /* CPUID */
    LIBXSMM_CPUID_X86(1, eax, ebx, ecx, edx);

    /* XSAVE/XGETBV(0x04000000), OSXSAVE(0x08000000) */
    if (0x0C000000 == (0x0C000000 & ecx)) {
      /* Check for CRC32 (this is not a proper test for SSE 4.2 as a whole!) */
      if (0x00100000 == (0x00100000 & ecx)) {
        target_arch = LIBXSMM_MAX(LIBXSMM_X86_SSE4_2, target_arch);
      }
      LIBXSMM_XGETBV(0, eax, edx);

      if (0x00000006 == (0x00000006 & eax)) { /* OS XSAVE 256-bit */
        if (0x000000E0 == (0x000000E0 & eax)) { /* OS XSAVE 512-bit */
          LIBXSMM_CPUID_X86(7, eax, ebx, ecx, edx);

          /* AVX512F(0x00010000), AVX512CD(0x10000000), AVX512PF(0x04000000),
             AVX512ER(0x08000000) */
          if (0x1C010000 == (0x1C010000 & ebx)) {
            target_arch = LIBXSMM_MAX(LIBXSMM_X86_AVX512, target_arch);
            if (archid) *archid = "knl";
          }
          /* AVX512F(0x00010000), AVX512CD(0x10000000), AVX512DQ(0x00020000),
             AVX512BW(0x40000000), AVX512VL(0x80000000) */
          else if (0xD0030000 == (0xD0030000 & ebx)) {
            target_arch = LIBXSMM_MAX(LIBXSMM_X86_AVX512, target_arch);
            if (archid) *archid = "skx";
          }
        }
        else if (0x10000000 == (0x10000000 & ecx)) { /* AVX(0x10000000) */
          if (0x00001000 == (0x00001000 & ecx)) { /* FMA(0x00001000) */
            target_arch = LIBXSMM_MAX(LIBXSMM_X86_AVX2, target_arch);
            if (archid) *archid = "hsw";
          }
          else {
            target_arch = LIBXSMM_MAX(LIBXSMM_X86_AVX, target_arch);
            if (archid) *archid = "snb";
          }
        }
      }
    }
  }

#if defined(LIBXSMM_STATIC_TARGET_ARCH)
  /* check if procedure obviously failed to detect the highest available instruction set extension */
  assert(LIBXSMM_STATIC_TARGET_ARCH <= target_arch);
#endif

  return target_arch;
}

