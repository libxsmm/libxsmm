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

/* must match definitions in headeronly.c */
#if !defined(ITYPE)
# define ITYPE double
#endif
#if !defined(OTYPE)
# define OTYPE ITYPE
#endif


LIBXSMM_EXTERN_C libxsmm_gemmfunction mmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k);
LIBXSMM_EXTERN_C libxsmm_gemmfunction mmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k)
{
  libxsmm_gemmfunction result;
#if defined(__cplusplus) /* C++ by chance: test libxsmm_mmfunction<> wrapper */
  const libxsmm_mmfunction<ITYPE, OTYPE, LIBXSMM_PREFETCH> mmfunction(m, n, k);
  result = mmfunction.kernel();
#else
  const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(m, n, k, m/*lda*/, k/*ldb*/, m/*ldc*/,
    LIBXSMM_DATATYPE(ITYPE), LIBXSMM_DATATYPE(ITYPE), LIBXSMM_DATATYPE(OTYPE), LIBXSMM_DATATYPE(OTYPE));
  result = libxsmm_dispatch_gemm_v2(gemm_shape, LIBXSMM_GEMM_FLAG_NONE,
    (libxsmm_bitfield)LIBXSMM_PREFETCH);
#endif
  return result;
}
