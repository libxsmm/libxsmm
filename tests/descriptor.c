#include <libxsmm_gemm_diff.h>
#include <libxsmm_cpuid.h>
#include <stdlib.h>
#include <stdio.h>


/**
 * This test case is NOT an example of how to use LIBXSMM
 * since INTERNAL functions are tested which are not part
 * of the LIBXSMM API.
 */
int main()
{
  int is_static, has_crc32;
  const char *const cpuid = libxsmm_cpuid(&is_static, &has_crc32);
  const int m = 64, n = 239, k = 64, lda = 64, ldb = 240, ldc = 240;
  union { libxsmm_gemm_descriptor descriptor; char simd[32]; } a, b;
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

  if (0 == libxsmm_gemm_diff(&a.descriptor, &b.descriptor)) {
    fprintf(stderr, "using static code path\n");
    return 1;
  }
  else if (0 != libxsmm_gemm_diff(&a.descriptor, &a.descriptor)) {
    fprintf(stderr, "using static code path\n");
    return 2;
  }
  else if (0 == libxsmm_gemm_diff(&b.descriptor, &a.descriptor)) {
    fprintf(stderr, "using static code path\n");
    return 3;
  }
  if (0 != has_crc32/*sse4.2*/ || 0 != cpuid) {
    if (0 == libxsmm_gemm_diff_sse(&a.descriptor, &b.descriptor)) {
      fprintf(stderr, "using SSE code path\n");
      return 4;
    }
    else if (0 != libxsmm_gemm_diff_sse(&a.descriptor, &a.descriptor)) {
      fprintf(stderr, "using SSE code path\n");
      return 5;
    }
    else if (0 == libxsmm_gemm_diff_sse(&b.descriptor, &a.descriptor)) {
      fprintf(stderr, "using SSE code path\n");
      return 6;
    }
  }
  if (0 != cpuid) {
    if (0 == libxsmm_gemm_diff_avx(&a.descriptor, &b.descriptor)) {
      fprintf(stderr, "using AVX code path\n");
      return 7;
    }
    else if (0 != libxsmm_gemm_diff_avx(&a.descriptor, &a.descriptor)) {
      fprintf(stderr, "using AVX code path\n");
      return 8;
    }
    else if (0 == libxsmm_gemm_diff_avx(&b.descriptor, &a.descriptor)) {
      fprintf(stderr, "using AVX code path\n");
      return 9;
    }
    if (/*snb*/'b' != cpuid[2]) {
      if (0 == libxsmm_gemm_diff_avx2(&a.descriptor, &b.descriptor)) {
      fprintf(stderr, "using AVX2 code path\n");
        return 10;
      }
      else if (0 != libxsmm_gemm_diff_avx2(&a.descriptor, &a.descriptor)) {
      fprintf(stderr, "using AVX2 code path\n");
        return 11;
      }
      else if (0 == libxsmm_gemm_diff_avx2(&b.descriptor, &a.descriptor)) {
      fprintf(stderr, "using AVX2 code path\n");
        return 12;
      }
    }
  }

  return EXIT_SUCCESS;
}

