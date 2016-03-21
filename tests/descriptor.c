#include <libxsmm_gemm_diff.h>
#include <libxsmm_cpuid_x86.h>
#include <stdlib.h>
#include <stdio.h>


/**
 * This test case is NOT an example of how to use LIBXSMM
 * since INTERNAL functions are tested which are not part
 * of the LIBXSMM API.
 */
int main()
{
  const char* archid;
  const int cpuid = libxsmm_cpuid_x86(&archid);
  const int m = 64, n = 239, k = 64, lda = 64, ldb = 240, ldc = 240;
  union { libxsmm_gemm_descriptor descriptor; char simd[LIBXSMM_ALIGNMENT]; } a, b;
  unsigned int i;

  LIBXSMM_GEMM_DESCRIPTOR(a.descriptor, LIBXSMM_ALIGNMENT, LIBXSMM_FLAGS,
    LIBXSMM_LD(m, n), LIBXSMM_LD(n, m), k,
    LIBXSMM_LD(lda, ldb), LIBXSMM_LD(ldb, lda), ldc,
    LIBXSMM_ALPHA, LIBXSMM_BETA,
    LIBXSMM_PREFETCH_NONE);
  LIBXSMM_GEMM_DESCRIPTOR(b.descriptor, LIBXSMM_ALIGNMENT, LIBXSMM_FLAGS,
    LIBXSMM_LD(m, n), LIBXSMM_LD(n, m), k,
    LIBXSMM_LD(lda, ldb), LIBXSMM_LD(ldb, lda), ldc,
    LIBXSMM_ALPHA, LIBXSMM_BETA,
    LIBXSMM_PREFETCH_BL2_VIA_C);

#if defined(LIBXSMM_GEMM_DIFF_MASK_A)
  for (i = LIBXSMM_GEMM_DESCRIPTOR_SIZE; i < sizeof(a.simd); ++i) {
    a.simd[i] = 'a'; b.simd[i] = 'b';
  }
#else
  for (i = LIBXSMM_GEMM_DESCRIPTOR_SIZE; i < sizeof(a.simd); ++i) {
    a.simd[i] = b.simd[i] = 0;
  }
#endif

  if (0 == libxsmm_gemm_diff_sw(&a.descriptor, &b.descriptor)) {
    fprintf(stderr, "using generic code path\n");
    return 1;
  }
  else if (0 == libxsmm_gemm_diff_sw(&b.descriptor, &a.descriptor)) {
    fprintf(stderr, "using generic code path\n");
    return 2;
  }
  else if (0 != libxsmm_gemm_diff_sw(&a.descriptor, &a.descriptor)) {
    fprintf(stderr, "using generic code path\n");
    return 3;
  }
  else if (0 != libxsmm_gemm_diff_sw(&b.descriptor, &b.descriptor)) {
    fprintf(stderr, "using generic code path\n");
    return 4;
  }
  if (LIBXSMM_X86_AVX <= cpuid) {
    if (0 == libxsmm_gemm_diff_avx(&a.descriptor, &b.descriptor)) {
      fprintf(stderr, "using AVX code path\n");
      return 9;
    }
    else if (0 == libxsmm_gemm_diff_avx(&b.descriptor, &a.descriptor)) {
      fprintf(stderr, "using AVX code path\n");
      return 10;
    }
    else if (0 != libxsmm_gemm_diff_avx(&a.descriptor, &a.descriptor)) {
      fprintf(stderr, "using AVX code path\n");
      return 11;
    }
    else if (0 != libxsmm_gemm_diff_avx(&b.descriptor, &b.descriptor)) {
      fprintf(stderr, "using AVX code path\n");
      return 12;
    }
  }
  if (LIBXSMM_X86_AVX2 <= cpuid) {
    if (0 == libxsmm_gemm_diff_avx2(&a.descriptor, &b.descriptor)) {
      fprintf(stderr, "using AVX2 code path\n");
      return 13;
    }
    else if (0 == libxsmm_gemm_diff_avx2(&b.descriptor, &a.descriptor)) {
      fprintf(stderr, "using AVX2 code path\n");
      return 14;
    }
    else if (0 != libxsmm_gemm_diff_avx2(&a.descriptor, &a.descriptor)) {
      fprintf(stderr, "using AVX2 code path\n");
      return 15;
    }
    else if (0 != libxsmm_gemm_diff_avx2(&b.descriptor, &b.descriptor)) {
      fprintf(stderr, "using AVX2 code path\n");
      return 16;
    }
  }
  if (0 == libxsmm_gemm_diff(&a.descriptor, &b.descriptor)) {
    fprintf(stderr, "using dispatched code path\n");
    return 17;
  }
  else if (0 == libxsmm_gemm_diff(&b.descriptor, &a.descriptor)) {
    fprintf(stderr, "using dispatched code path\n");
    return 18;
  }
  else if (0 != libxsmm_gemm_diff(&a.descriptor, &a.descriptor)) {
    fprintf(stderr, "using dispatched code path\n");
    return 19;
  }
  else if (0 != libxsmm_gemm_diff(&b.descriptor, &b.descriptor)) {
    fprintf(stderr, "using dispatched code path\n");
    return 20;
  }

  { /* testing diff-search */
    union { libxsmm_gemm_descriptor desc; char padding[32]; } descs[8];
    descs[0].desc = b.descriptor; descs[1].desc = a.descriptor;
    descs[2].desc = a.descriptor; descs[3].desc = a.descriptor;
    descs[4].desc = a.descriptor; descs[5].desc = a.descriptor;
    descs[6].desc = b.descriptor; descs[7].desc = a.descriptor;

    if (1 != libxsmm_gemm_diffn_sw(&a.descriptor, &descs[0].desc, 0/*hint*/,
      sizeof(descs) / sizeof(*descs), sizeof(*descs)))
    {
      fprintf(stderr, "using generic diff-search\n");
      return 21;
    }
    else if (6 != libxsmm_gemm_diffn_sw(&b.descriptor, &descs[0].desc, 2/*hint*/,
      sizeof(descs) / sizeof(*descs), sizeof(*descs)))
    {
      fprintf(stderr, "using generic diff-search\n");
      return 22;
    }
    if (LIBXSMM_X86_AVX <= cpuid) {
      if (1 != libxsmm_gemm_diffn_avx(&a.descriptor, &descs[0].desc, 0/*hint*/,
        sizeof(descs) / sizeof(*descs), sizeof(*descs)))
      {
        fprintf(stderr, "using AVX-based diff-search\n");
        return 23;
      }
      else if (6 != libxsmm_gemm_diffn_avx(&b.descriptor, &descs[0].desc, 2/*hint*/,
        sizeof(descs) / sizeof(*descs), sizeof(*descs)))
      {
        fprintf(stderr, "using AVX-based diff-search\n");
        return 24;
      }
    }
    if (LIBXSMM_X86_AVX2 <= cpuid) {
      if (1 != libxsmm_gemm_diffn_avx2(&a.descriptor, &descs[0].desc, 0/*hint*/,
        sizeof(descs) / sizeof(*descs), sizeof(*descs)))
      {
        fprintf(stderr, "using AVX2-based diff-search\n");
        return 25;
      }
      else if (6 != libxsmm_gemm_diffn_avx2(&b.descriptor, &descs[0].desc, 2/*hint*/,
        sizeof(descs) / sizeof(*descs), sizeof(*descs)))
      {
        fprintf(stderr, "using AVX2-based diff-search\n");
        return 26;
      }
    }
    if (LIBXSMM_X86_AVX512 <= cpuid) {
      if (1 != libxsmm_gemm_diffn_avx512(&a.descriptor, &descs[0].desc, 0/*hint*/,
        sizeof(descs) / sizeof(*descs), sizeof(*descs)))
      {
        fprintf(stderr, "using AVX512-based diff-search\n");
        return 27;
      }
      else if (6 != libxsmm_gemm_diffn_avx512(&b.descriptor, &descs[0].desc, 2/*hint*/,
        sizeof(descs) / sizeof(*descs), sizeof(*descs)))
      {
        fprintf(stderr, "using AVX512-based diff-search\n");
        return 28;
      }
    }
    if (1 != libxsmm_gemm_diffn(&a.descriptor, &descs[0].desc, 0/*hint*/,
      sizeof(descs) / sizeof(*descs), sizeof(*descs)))
    {
      fprintf(stderr, "using dispatched diff-search\n");
      return 29;
    }
    else if (6 != libxsmm_gemm_diffn(&b.descriptor, &descs[0].desc, 2/*hint*/,
      sizeof(descs) / sizeof(*descs), sizeof(*descs)))
    {
      fprintf(stderr, "using dispatched diff-search\n");
      return 30;
    }
  }

  return EXIT_SUCCESS;
}

