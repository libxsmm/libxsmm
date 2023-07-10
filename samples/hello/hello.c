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

int main(int argc, char* argv[]) {
  const double beta = 1;
  const char transa = 'N', transb = 'N';
  const libxsmm_blasint batchsize = 1000, m = 13, n = 5, k = 7;
  double* const a = malloc(sizeof(double) * batchsize * m * k);
  double* const b = malloc(sizeof(double) * batchsize * k * n);
  double* const c = malloc(sizeof(double) * m * n);
  libxsmm_blasint ki, i, j;
  const int flags_trans = LIBXSMM_GEMM_FLAGS(transa, transb);
  const int flags_ab = (LIBXSMM_NEQ(0, beta) ? 0 : LIBXSMM_GEMM_FLAG_BETA_0);
  /* determine matrix shape and precision */
  const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(m, n, k, m /*lda*/, k /*ldb*/, m /*ldc*/, LIBXSMM_DATATYPE_F64,
    LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64);
  /* generate and dispatch a matrix multiplication kernel */
  const libxsmm_gemmfunction kernel = libxsmm_dispatch_gemm_v2(
    gemm_shape, (libxsmm_bitfield)(flags_trans | flags_ab), (libxsmm_bitfield)LIBXSMM_GEMM_PREFETCH_NONE);
  libxsmm_gemm_param gemm_param; /* collect call-arguments into single structure */
  assert(NULL != kernel && NULL != a && NULL != b && NULL != c);
  LIBXSMM_UNUSED(argc);
  LIBXSMM_UNUSED(argv);
  for (i = 0; i < batchsize; ++i) { /* initialize input */
    for (ki = 0; ki < k; ++ki) {
      for (j = 0; j < m; ++j) a[i * j * ki] = ((double)1) / ((i + j + ki) % 25);
      for (j = 0; j < n; ++j) b[i * j * ki] = ((double)7) / ((i + j + ki) % 75);
    }
  }
  memset(c, 0, sizeof(double) * m * n);
  /* kernel multiplies and accumulates matrices: C += Ai * Bi */
  gemm_param.c.primary = c;
  for (i = 0; i < batchsize; ++i) {
    gemm_param.a.primary = (double*)(a + i * m * k);
    gemm_param.b.primary = (double*)(b + i * k * n);
    kernel(&gemm_param);
  }
  free(a), free(b), free(c);
  return 0;
}
