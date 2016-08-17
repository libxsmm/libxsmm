#include <libxsmm_source.h>
#include <stdlib.h>
#include <stdio.h>


LIBXSMM_EXTERN libxsmm_dmmfunction dmmdispatch(int m, int n, int k);


int main(void)
{
  const int m = LIBXSMM_MAX_M, n = LIBXSMM_MAX_N, k = LIBXSMM_MAX_K;
  const libxsmm_dmmfunction fa = libxsmm_dmmdispatch(m, n, k,
    NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/, NULL/*alpha*/, NULL/*beta*/,
    NULL/*flags*/, NULL/*prefetch*/);
  const libxsmm_dmmfunction fb = dmmdispatch(m, n, k);
#if defined(_DEBUG)
  if (fa != fb) {
    union { libxsmm_xmmfunction xmm; void* pmm; } a, b;
    a.xmm.dmm = fa; b.xmm.dmm = fb;
    fprintf(stderr, "Error: %p != %p\n", a.pmm, b.pmm);
  }
#endif
  return fa == fb ? EXIT_SUCCESS : EXIT_FAILURE;
}

