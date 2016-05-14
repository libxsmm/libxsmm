#include <libxsmm.h>
#include <stdlib.h>


int main()
{
  const int archid = libxsmm_get_target_arch();

  /* official runtime check for JIT availability */
  if (LIBXSMM_X86_AVX <= archid) { /* available */
#if 0 == LIBXSMM_JIT
    /* runtime check should have been negative */
    return EXIT_FAILURE;
#endif
    libxsmm_set_target_archid("0"); /* disable JIT */
    /* likely returns NULL, however should not crash */
    libxsmm_smmdispatch(LIBXSMM_MAX_M, LIBXSMM_MAX_N, LIBXSMM_MAX_K,
      NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/,
      NULL/*alpha*/, NULL/*beta*/, NULL/*flags*/, NULL/*prefetch*/);
  }
#if 0 != LIBXSMM_JIT /* JIT is built-in (enabled at compile-time) */
  else { /* JIT is not available at runtime */
    /* bypass CPUID flags and setup to something supported with JIT */
    libxsmm_set_target_arch(LIBXSMM_X86_AVX);

    if (0 == libxsmm_dmmdispatch(LIBXSMM_MAX_M, LIBXSMM_MAX_N, LIBXSMM_MAX_K,
      NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/,
      NULL/*alpha*/, NULL/*beta*/, NULL/*flags*/, NULL/*prefetch*/))
    {
      /* requested function should have been JITted */
      return EXIT_FAILURE;
    }
  }
#endif

  return EXIT_SUCCESS;
}

