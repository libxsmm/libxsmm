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
  const size_t t = sizeof(double);
  int flags = LIBXSMM_GEMM_FLAG_NONE, batchsize = 1000, m = 13, n = 5, k = 7, ki, i, j;
  double *a = malloc(t * batchsize * m * k), *b = malloc(t * batchsize * k * n);
  double *c = malloc(t * m * n), alpha = 1, beta = 1;
  /* generates and dispatches a matrix multiplication kernel */
  libxsmm_dmmfunction kernel = libxsmm_dmmdispatch(
    m, n, k, NULL /*lda*/, NULL /*ldb*/, NULL /*ldc*/, &alpha, &beta, &flags, NULL /*prefetch*/);
  assert(NULL != kernel && NULL != a && NULL != b && NULL != c);
  LIBXSMM_UNUSED(argc); LIBXSMM_UNUSED(argv);
  for (i = 0; i < batchsize; ++i) { /* initialize input */
    for (ki = 0; ki < k; ++ki) {
      for (j = 0; j < m; ++j) a[i * j * ki] = ((double)1) / ((i + j + ki) % 25);
      for (j = 0; j < n; ++j) b[i * j * ki] = ((double)7) / ((i + j + ki) % 75);
    }
  }
  memset(c, 0, t * m * n);
  /* kernel multiplies and accumulates matrices: C += Ai * Bi */
  for (i = 0; i < batchsize; ++i) kernel(a + i * m * k, b + i * k * n, c);
  free(a), free(b), free(c);
  return 0;
}
