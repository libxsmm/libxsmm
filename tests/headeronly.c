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
#include <libxsmm_source.h>

/* must match definitions in headeronly_aux.c */
#if !defined(ITYPE)
# define ITYPE double
#endif
#if !defined(OTYPE)
# define OTYPE ITYPE
#endif


LIBXSMM_EXTERN_C LIBXSMM_MMFUNCTION_TYPE2(ITYPE, OTYPE) mmdispatch(int m, int n, int k);


int main(void)
{
  const int m = LIBXSMM_MAX_M, n = LIBXSMM_MAX_N, k = LIBXSMM_MAX_K;
  const LIBXSMM_MMFUNCTION_TYPE2(ITYPE, OTYPE) fa = LIBXSMM_MMDISPATCH_SYMBOL2(ITYPE, OTYPE)(m, n, k,
    NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/, NULL/*alpha*/, NULL/*beta*/,
    NULL/*flags*/, NULL/*prefetch*/);
  const LIBXSMM_MMFUNCTION_TYPE2(ITYPE, OTYPE) fb = mmdispatch(m, n, k);
  int result = EXIT_SUCCESS;

  if (fa == fb) { /* test unregistering and freeing kernel */
    union {
      LIBXSMM_MMFUNCTION_TYPE2(ITYPE, OTYPE) f;
      const void* p;
    } kernel;
    kernel.f = fa;
    libxsmm_release_kernel(kernel.p);
  }
  else {
    libxsmm_registry_info registry_info;
    result = libxsmm_get_registry_info(&registry_info);
    if (EXIT_SUCCESS == result && 2 != registry_info.size) {
      result = EXIT_FAILURE;
    }
  }
  return result;
}

