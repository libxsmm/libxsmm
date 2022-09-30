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
#include <vector>

int main(int argc, char* argv[]) {
  typedef double T;
  int batchsize = 1000, m = 13, n = 5, k = 7;
  std::vector<T> a(batchsize * m * k), b(batchsize * k * n), c(m * n, 0);
  /* C/C++ and Fortran interfaces are available */
  typedef libxsmm_mmfunction<T> kernel_type;
  /* generates and dispatches a matrix multiplication kernel (C++ functor) */
  kernel_type kernel(LIBXSMM_GEMM_FLAG_NONE, m, n, k, 1.0 /*alpha*/, 1.0 /*beta*/);
  LIBXSMM_UNUSED(argc); LIBXSMM_UNUSED(argv);
  assert(kernel);
  for (int i = 0; i < batchsize; ++i) { /* initialize input */
    for (int ki = 0; ki < k; ++ki) {
      for (int j = 0; j < m; ++j) a[i * j * ki] = static_cast<T>(1) / ((i + j + ki) % 25);
      for (int j = 0; j < n; ++j) b[i * j * ki] = static_cast<T>(7) / ((i + j + ki) % 75);
    }
  }
  /* kernel multiplies and accumulates matrices: C += Ai * Bi */
  for (int i = 0; i < batchsize; ++i) kernel(&a[i * m * k], &b[i * k * n], &c[0]);
}
