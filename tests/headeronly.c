/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
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


LIBXSMM_EXTERN_C libxsmm_gemmfunction mmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k);


int main(void)
{
  const libxsmm_blasint m = LIBXSMM_MAX_M, n = LIBXSMM_MAX_N, k = LIBXSMM_MAX_K;
  const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(
    m, n, k, m/*lda*/, k/*ldb*/, m/*ldc*/,
    LIBXSMM_DATATYPE(ITYPE), LIBXSMM_DATATYPE(ITYPE),
    LIBXSMM_DATATYPE(OTYPE), LIBXSMM_DATATYPE(OTYPE));
  const libxsmm_gemmfunction fa = libxsmm_dispatch_gemm_v2(gemm_shape,
    LIBXSMM_GEMM_FLAG_NONE, (libxsmm_bitfield)LIBXSMM_PREFETCH);
  const libxsmm_gemmfunction fb = mmdispatch(m, n, k);
  int result = EXIT_SUCCESS;

  if (fa == fb) { /* test unregistering and freeing kernel */
    union {
      libxsmm_gemmfunction f;
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
