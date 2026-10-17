/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxsmm.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


LIBXSMM_INLINE int packedcsr_run(int fp64, unsigned int width, unsigned int n,
  const unsigned int* rowptr, int beta0)
{
  const unsigned int ldc = n + 1, colidx[2] = { 0, 0 };
  const libxsmm_datatype type = fp64 ? LIBXSMM_DATATYPE_F64 : LIBXSMM_DATATYPE_F32;
  const libxsmm_gemm_shape shape = libxsmm_create_gemm_shape(5, n, 1, 0, n, ldc, type, type, type, type);
  float af[2] = { 3, 3 }, bf[29*32], cf[5*30*32];
  double ad[2] = { 3, 3 }, bd[29*32], cd[5*30*32];
  libxsmm_gemm_param param;
  libxsmm_xmmfunction kernel;
  unsigned int i, initial;
  int result = EXIT_SUCCESS;

  for (i = 0; i < n * width; ++i) {
    bf[i] = (float)(i + 1);
    bd[i] = (double)(i + 1);
  }
  kernel.gemm = libxsmm_create_packed_spgemm_csr(shape,
    beta0 ? LIBXSMM_GEMM_FLAG_BETA_0 : LIBXSMM_GEMM_FLAG_NONE,
    LIBXSMM_GEMM_PREFETCH_NONE, width, rowptr, colidx, fp64 ? (const void*)ad : (const void*)af);
  if (NULL == kernel.gemm) {
    fprintf(stderr, "ERROR: packed CSR kernel could not be dispatched!\n");
    return EXIT_FAILURE;
  }
  memset(&param, 0, sizeof(param));
  param.a.primary = fp64 ? (void*)ad : (void*)af;
  param.b.primary = fp64 ? (void*)bd : (void*)bf;
  param.c.primary = fp64 ? (void*)cd : (void*)cf;

  for (initial = 0; initial < 2 && EXIT_SUCCESS == result; ++initial) {
    const double old = initial ? NAN : 19.0;
    for (i = 0; i < 5 * ldc * width; ++i) {
      cf[i] = (float)old;
      cd[i] = old;
    }
    kernel.gemm(&param);
    for (i = 0; i < 5 * ldc * width; ++i) {
      const unsigned int row = i / (ldc * width), col = (i / width) % ldc, lane = i % width;
      const double actual = fp64 ? cd[i] : (double)cf[i];
      double expected = old;
      if (col < n) {
        expected = rowptr[row + 1] > rowptr[row] ? 3.0 * (col * width + lane + 1) : 0.0;
        if (0 == beta0) expected += old;
      }
      if (LIBXSMM_ISNAN(expected) ? !LIBXSMM_ISNAN(actual) : LIBXSMM_NEQ(expected, actual)) {
        fprintf(stderr, "ERROR: packed CSR fp64=%i width=%u n=%u beta0=%i initial=%u at (%u,%u,%u): %g != %g\n",
          fp64, width, n, beta0, initial, row, col, lane, expected, actual);
        result = EXIT_FAILURE;
        break;
      }
    }
  }
  libxsmm_release_kernel(kernel.ptr_const);
  return result;
}


int main(void)
{
  /* Empty first, middle and last rows, followed by an entirely empty matrix. */
  const unsigned int rowptr[2][6] = { { 0, 0, 1, 1, 2, 2 }, { 0, 0, 0, 0, 0, 0 } };
  const unsigned int widths[] = { 1, 4, 8, 16, 17, 32 }, ns[] = { 1, 29 };
  const int archid = libxsmm_get_target_archid();
  unsigned int w, n, empty;
  int fp64, beta0;

  /* Exercise the x86 packed CSR emitter only where it is available. */
  if (LIBXSMM_X86_AVX > archid || LIBXSMM_X86_ALLFEAT < archid) return EXIT_SUCCESS;
  for (fp64 = 0; fp64 < 2; ++fp64) {
    for (w = 0; w < sizeof(widths) / sizeof(*widths); ++w) {
      for (n = 0; n < sizeof(ns) / sizeof(*ns); ++n) {
        for (empty = 0; empty < 2; ++empty) {
          for (beta0 = 0; beta0 < 2; ++beta0) {
            if (EXIT_SUCCESS != packedcsr_run(fp64, widths[w], ns[n], rowptr[empty], beta0)) return EXIT_FAILURE;
          }
        }
      }
    }
  }
  return EXIT_SUCCESS;
}
