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
  int is_static, has_crc32, i;
  const char *const cpuid = libxsmm_cpuid(&is_static, &has_crc32);
  const libxsmm_gemm_diff_function diff = 0 != cpuid
    ? (/*snb*/'b' != cpuid[2] ? libxsmm_gemm_diff_avx2 : libxsmm_gemm_diff_avx)
    : (0 != has_crc32/*sse4.2*/ ? libxsmm_gemm_diff_sse : libxsmm_gemm_diff);
  const int m = 64, n = 239, k = 64, lda = 64, ldb = 240, ldc = 240;
  union { libxsmm_gemm_descriptor descriptor; char simd[32]; } a;
  libxsmm_gemm_descriptor b;

  LIBXSMM_GEMM_DESCRIPTOR(a.descriptor, LIBXSMM_ALIGNMENT, LIBXSMM_FLAGS,
    LIBXSMM_LD(m, n), LIBXSMM_LD(n, m), k,
    LIBXSMM_LD(lda, ldb), LIBXSMM_LD(ldb, lda), ldc,
    LIBXSMM_ALPHA, LIBXSMM_BETA,
    LIBXSMM_PREFETCH_NONE);

#if !defined(LIBXSMM_GEMM_DIFF_MASK_A)
  for (i = LIBXSMM_GEMM_DESCRIPTOR_SIZE; i < sizeof(a.simd); ++i) a.simd[i] = 0;
#endif

  LIBXSMM_GEMM_DESCRIPTOR(b, LIBXSMM_ALIGNMENT, LIBXSMM_FLAGS,
    LIBXSMM_LD(m, n), LIBXSMM_LD(n, m), k,
    LIBXSMM_LD(lda, ldb), LIBXSMM_LD(ldb, lda), ldc,
    LIBXSMM_ALPHA, LIBXSMM_BETA,
    LIBXSMM_PREFETCH_BL2_VIA_C);

  if (0 == libxsmm_gemm_diff(&a.descriptor, &b)) {
    fprintf(stderr, "using static code path\n");
    return 1;
  }
  else if (0 != libxsmm_gemm_diff(&a.descriptor, &a.descriptor)) {
    fprintf(stderr, "using static code path\n");
    return 2;
  }
  else if (0 == libxsmm_gemm_diff(&b, &a.descriptor)) {
    fprintf(stderr, "using static code path\n");
    return 3;
  }
  else if (0 == diff(&a.descriptor, &b)) {
    fprintf(stderr, "using %s code path\n", cpuid
      ? cpuid : (0 != has_crc32 ? "SSE"
      : (0 != is_static ? "static" : "non-AVX")));
    return 4;
  }
  else if (0 != diff(&a.descriptor, &a.descriptor)) {
    fprintf(stderr, "using %s code path\n", cpuid
      ? cpuid : (0 != has_crc32 ? "SSE"
      : (0 != is_static ? "static" : "non-AVX")));
    return 5;
  }
  else if (0 == diff(&b, &a.descriptor)) {
    fprintf(stderr, "using %s code path\n", cpuid
      ? cpuid : (0 != has_crc32 ? "SSE"
      : (0 != is_static ? "static" : "non-AVX")));
    return 6;
  }
  else {
    return EXIT_SUCCESS;
  }
}

