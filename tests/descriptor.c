/******************************************************************************
** Copyright (c) 2015-2016, Intel Corporation                                **
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
#include <libxsmm_source.h>
#include <stdlib.h>
#if defined(_DEBUG)
# include <stdio.h>
#endif


/**
 * This test case is NOT an example of how to use LIBXSMM
 * since INTERNAL functions are tested which are not part
 * of the LIBXSMM API.
 */
int main(void)
{
  const int cpuid_archid = libxsmm_cpuid();
  const int m = 64, n = 239, k = 64, lda = 64, ldb = 240, ldc = 240;
  union { libxsmm_gemm_descriptor descriptor; char simd[LIBXSMM_ALIGNMENT]; } a, b;
  unsigned int i, result = EXIT_SUCCESS;

  LIBXSMM_GEMM_DESCRIPTOR(a.descriptor, LIBXSMM_ALIGNMENT, LIBXSMM_FLAGS,
    LIBXSMM_LD(m, n), LIBXSMM_LD(n, m), k,
    LIBXSMM_LD(lda, ldb), LIBXSMM_LD(ldb, lda), ldc,
    1, 0.0, LIBXSMM_PREFETCH_NONE);
  LIBXSMM_GEMM_DESCRIPTOR(b.descriptor, LIBXSMM_ALIGNMENT, LIBXSMM_FLAGS,
    LIBXSMM_LD(m, n), LIBXSMM_LD(n, m), k,
    LIBXSMM_LD(lda, ldb), LIBXSMM_LD(ldb, lda), ldc,
    1.0, 0, LIBXSMM_PREFETCH_BL2_VIA_C);

#if defined(LIBXSMM_GEMM_DIFF_MASK_A)
  for (i = LIBXSMM_GEMM_DESCRIPTOR_SIZE; i < sizeof(a.simd); ++i) {
    a.simd[i] = 'a'; b.simd[i] = 'b';
  }
#else
  for (i = LIBXSMM_GEMM_DESCRIPTOR_SIZE; i < sizeof(a.simd); ++i) {
    a.simd[i] = b.simd[i] = 0;
  }
#endif
  {
#if defined(LIBXSMM_GENERATOR_SMALLDESC)
    union { libxsmm_gemm_descriptor desc; char padding[16]; } descs[8];
#else
    union { libxsmm_gemm_descriptor desc; char padding[32]; } descs[8];
#endif
    descs[0].desc = b.descriptor; descs[1].desc = a.descriptor;
    descs[2].desc = a.descriptor; descs[3].desc = a.descriptor;
    descs[4].desc = a.descriptor; descs[5].desc = a.descriptor;
    descs[6].desc = b.descriptor; descs[7].desc = a.descriptor;

    /* DIFF Testing
     */
    if (0 == libxsmm_gemm_diff_sw(&a.descriptor, &b.descriptor)) {
#if defined(_DEBUG)
      fprintf(stderr, "using generic code path\n");
#endif
      result = 1;
    }
    else if (0 == libxsmm_gemm_diff_sw(&b.descriptor, &a.descriptor)) {
#if defined(_DEBUG)
      fprintf(stderr, "using generic code path\n");
#endif
      result = 2;
    }
    else if (0 != libxsmm_gemm_diff_sw(&a.descriptor, &a.descriptor)) {
#if defined(_DEBUG)
      fprintf(stderr, "using generic code path\n");
#endif
      result = 3;
    }
    else if (0 != libxsmm_gemm_diff_sw(&b.descriptor, &b.descriptor)) {
#if defined(_DEBUG)
      fprintf(stderr, "using generic code path\n");
#endif
      result = 4;
    }
    if (EXIT_SUCCESS == result && LIBXSMM_X86_AVX <= cpuid_archid) {
      if (0 == libxsmm_gemm_diff_avx(&a.descriptor, &b.descriptor)) {
#if defined(_DEBUG)
        fprintf(stderr, "using AVX code path\n");
#endif
        result = 9;
      }
      else if (0 == libxsmm_gemm_diff_avx(&b.descriptor, &a.descriptor)) {
#if defined(_DEBUG)
        fprintf(stderr, "using AVX code path\n");
#endif
        result = 10;
      }
      else if (0 != libxsmm_gemm_diff_avx(&a.descriptor, &a.descriptor)) {
#if defined(_DEBUG)
        fprintf(stderr, "using AVX code path\n");
#endif
        result = 11;
      }
      else if (0 != libxsmm_gemm_diff_avx(&b.descriptor, &b.descriptor)) {
#if defined(_DEBUG)
        fprintf(stderr, "using AVX code path\n");
#endif
        result = 12;
      }
    }
    if (EXIT_SUCCESS == result && LIBXSMM_X86_AVX2 <= cpuid_archid) {
      if (0 == libxsmm_gemm_diff_avx2(&a.descriptor, &b.descriptor)) {
#if defined(_DEBUG)
        fprintf(stderr, "using AVX2 code path\n");
#endif
        result = 13;
      }
      else if (0 == libxsmm_gemm_diff_avx2(&b.descriptor, &a.descriptor)) {
#if defined(_DEBUG)
        fprintf(stderr, "using AVX2 code path\n");
#endif
        result = 14;
      }
      else if (0 != libxsmm_gemm_diff_avx2(&a.descriptor, &a.descriptor)) {
#if defined(_DEBUG)
        fprintf(stderr, "using AVX2 code path\n");
#endif
        result = 15;
      }
      else if (0 != libxsmm_gemm_diff_avx2(&b.descriptor, &b.descriptor)) {
#if defined(_DEBUG)
        fprintf(stderr, "using AVX2 code path\n");
#endif
        result = 16;
      }
    }
    if (EXIT_SUCCESS == result) {
      if (0 == libxsmm_gemm_diff(&a.descriptor, &b.descriptor)) {
#if defined(_DEBUG)
        fprintf(stderr, "using dispatched code path\n");
#endif
        result = 17;
      }
      else if (0 == libxsmm_gemm_diff(&b.descriptor, &a.descriptor)) {
#if defined(_DEBUG)
        fprintf(stderr, "using dispatched code path\n");
#endif
        result = 18;
      }
      else if (0 != libxsmm_gemm_diff(&a.descriptor, &a.descriptor)) {
#if defined(_DEBUG)
        fprintf(stderr, "using dispatched code path\n");
#endif
        result = 19;
      }
      else if (0 != libxsmm_gemm_diff(&b.descriptor, &b.descriptor)) {
#if defined(_DEBUG)
        fprintf(stderr, "using dispatched code path\n");
#endif
        result = 20;
      }
    }
    /* DIFFN Testing
     */
    if (EXIT_SUCCESS == result) {
      if (1 != libxsmm_gemm_diffn_sw(&a.descriptor, &descs[0].desc, 0/*hint*/,
        sizeof(descs) / sizeof(*descs), sizeof(*descs)))
      {
#if defined(_DEBUG)
        fprintf(stderr, "using generic diffn-search\n");
#endif
        result = 21;
      }
      else if (6 != libxsmm_gemm_diffn_sw(&b.descriptor, &descs[0].desc, 2/*hint*/,
        sizeof(descs) / sizeof(*descs), sizeof(*descs)))
      {
#if defined(_DEBUG)
        fprintf(stderr, "using generic diffn-search\n");
#endif
        result = 22;
      }
    }
    if (EXIT_SUCCESS == result && LIBXSMM_X86_AVX <= cpuid_archid) {
      if (1 != libxsmm_gemm_diffn_avx(&a.descriptor, &descs[0].desc, 0/*hint*/,
        sizeof(descs) / sizeof(*descs), sizeof(*descs)))
      {
#if defined(_DEBUG)
        fprintf(stderr, "using AVX-based diffn-search\n");
#endif
        result = 23;
      }
      else if (6 != libxsmm_gemm_diffn_avx(&b.descriptor, &descs[0].desc, 2/*hint*/,
        sizeof(descs) / sizeof(*descs), sizeof(*descs)))
      {
#if defined(_DEBUG)
        fprintf(stderr, "using AVX-based diffn-search\n");
#endif
        result = 24;
      }
    }
    if (EXIT_SUCCESS == result && LIBXSMM_X86_AVX2 <= cpuid_archid) {
      if (1 != libxsmm_gemm_diffn_avx2(&a.descriptor, &descs[0].desc, 0/*hint*/,
        sizeof(descs) / sizeof(*descs), sizeof(*descs)))
      {
#if defined(_DEBUG)
        fprintf(stderr, "using AVX2-based diffn-search\n");
#endif
        result = 25;
      }
      else if (6 != libxsmm_gemm_diffn_avx2(&b.descriptor, &descs[0].desc, 2/*hint*/,
        sizeof(descs) / sizeof(*descs), sizeof(*descs)))
      {
#if defined(_DEBUG)
        fprintf(stderr, "using AVX2-based diffn-search\n");
#endif
        result = 26;
      }
    }
    if (EXIT_SUCCESS == result && LIBXSMM_X86_AVX512_MIC/*incl. LIBXSMM_X86_AVX512_CORE*/ <= cpuid_archid) {
      if (1 != libxsmm_gemm_diffn_avx512(&a.descriptor, &descs[0].desc, 0/*hint*/,
        sizeof(descs) / sizeof(*descs), sizeof(*descs)))
      {
#if defined(_DEBUG)
        fprintf(stderr, "using AVX512-based diffn-search\n");
#endif
        result = 27;
      }
      else if (6 != libxsmm_gemm_diffn_avx512(&b.descriptor, &descs[0].desc, 2/*hint*/,
        sizeof(descs) / sizeof(*descs), sizeof(*descs)))
      {
#if defined(_DEBUG)
        fprintf(stderr, "using AVX512-based diffn-search\n");
#endif
        result = 28;
      }
    }
    if (EXIT_SUCCESS == result) {
      if (1 != libxsmm_gemm_diffn(&a.descriptor, &descs[0].desc, 0/*hint*/,
        sizeof(descs) / sizeof(*descs), sizeof(*descs)))
      {
#if defined(_DEBUG)
        fprintf(stderr, "using dispatched diffn-search\n");
#endif
        result = 29;
      }
      else if (6 != libxsmm_gemm_diffn(&b.descriptor, &descs[0].desc, 2/*hint*/,
        sizeof(descs) / sizeof(*descs), sizeof(*descs)))
      {
#if defined(_DEBUG)
        fprintf(stderr, "using dispatched diffn-search\n");
#endif
        result = 30;
      }
    }
    /* Offload
     */
#if defined(LIBXSMM_OFFLOAD_TARGET)
#   pragma offload target(LIBXSMM_OFFLOAD_TARGET)
#endif
    if (EXIT_SUCCESS == result) {
#if defined(LIBXSMM_GENERATOR_SMALLDESC)
      union { libxsmm_gemm_descriptor desc; char padding[16]; } local[8];
#else
      union { libxsmm_gemm_descriptor desc; char padding[32]; } local[8];
#endif
      local[0].desc = b.descriptor; local[1].desc = a.descriptor;
      local[2].desc = a.descriptor; local[3].desc = a.descriptor;
      local[4].desc = a.descriptor; local[5].desc = a.descriptor;
      local[6].desc = b.descriptor; local[7].desc = a.descriptor;

      if (0 == libxsmm_gemm_diff_imci(&a.descriptor, &b.descriptor)) {
#if defined(_DEBUG)
        fprintf(stderr, "using IMCI code path\n");
#endif
        result = 31;
      }
      else if (0 == libxsmm_gemm_diff_imci(&b.descriptor, &a.descriptor)) {
#if defined(_DEBUG)
        fprintf(stderr, "using IMCI code path\n");
#endif
        result = 32;
      }
      else if (0 != libxsmm_gemm_diff_imci(&a.descriptor, &a.descriptor)) {
#if defined(_DEBUG)
        fprintf(stderr, "using IMCI code path\n");
#endif
        result = 33;
      }
      else if (0 != libxsmm_gemm_diff_imci(&b.descriptor, &b.descriptor)) {
#if defined(_DEBUG)
        fprintf(stderr, "using IMCI code path\n");
#endif
        result = 34;
      }
      else if (1 != libxsmm_gemm_diffn_imci(&a.descriptor, &local[0].desc, 0/*hint*/,
        sizeof(local) / sizeof(*local), sizeof(*local)))
      {
#if defined(_DEBUG)
        fprintf(stderr, "using IMCI-based diffn-search\n");
#endif
        result = 35;
      }
      else if (6 != libxsmm_gemm_diffn_imci(&b.descriptor, &local[0].desc, 2/*hint*/,
        sizeof(local) / sizeof(*local), sizeof(*local)))
      {
#if defined(_DEBUG)
        fprintf(stderr, "using IMCI-based diffn-search\n");
#endif
        result = 36;
      }
    }
  }

  return result;
}

