/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm_source.h>

/* The A-in-registers kernels reach B and C through 32-bit offsets, so shapes whose operands
 * exceed 2 GiB must be rejected at code-generation time rather than emitting code that reads
 * and writes far outside of the operands (see libxsmm/libxsmm#805). The oversized shapes are
 * only dispatched, never executed, hence no large buffers are needed. */

#if !defined(SPGEMMBIG_NNZ_PER_ROW)
# define SPGEMMBIG_NNZ_PER_ROW 4
#endif
#define SPGEMMBIG_M 150
#define SPGEMMBIG_K 126


LIBXSMM_INLINE void spgemmbig_csr(unsigned int* rowptr, unsigned int* colidx, double* values)
{
  unsigned int i, j, nnz = 0;
  for (i = 0; i < SPGEMMBIG_M; ++i) {
    rowptr[i] = nnz;
    for (j = 0; j < SPGEMMBIG_NNZ_PER_ROW; ++j) {
      colidx[nnz] = (i + j) % SPGEMMBIG_K;
      values[nnz] = 1.0 + (i % 3);
      ++nnz;
    }
  }
  rowptr[SPGEMMBIG_M] = nnz;
}


LIBXSMM_INLINE libxsmm_gemmfunction spgemmbig_dispatch(libxsmm_blasint n, libxsmm_blasint ldb, libxsmm_blasint ldc,
  const unsigned int* rowptr, const unsigned int* colidx, const double* values)
{
  const libxsmm_gemm_shape shape = libxsmm_create_gemm_shape(SPGEMMBIG_M, n, SPGEMMBIG_K, 0/*lda*/, ldb, ldc,
    LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64);
  return libxsmm_create_spgemm_csr_areg(shape, LIBXSMM_GEMM_FLAGS('N', 'N'), LIBXSMM_GEMM_PREFETCH_NONE,
    n, rowptr, colidx, values);
}


LIBXSMM_INLINE void spgemmbig_release(libxsmm_gemmfunction kernel)
{
  void* fp = NULL;
  LIBXSMM_VALUE_ASSIGN(fp, kernel);
  libxsmm_release_kernel(fp);
}


int main(void)
{
  const libxsmm_blasint oversized_b = (libxsmm_blasint)(0x80000000ULL / (8 * SPGEMMBIG_K) + 1);
  const libxsmm_blasint oversized_c = (libxsmm_blasint)(0x80000000ULL / (8 * SPGEMMBIG_M) + 1);
  unsigned int rowptr[SPGEMMBIG_M+1], colidx[SPGEMMBIG_M*SPGEMMBIG_NNZ_PER_ROW];
  double values[SPGEMMBIG_M*SPGEMMBIG_NNZ_PER_ROW];
  int result = EXIT_SUCCESS;
  libxsmm_gemmfunction kernel;
  libxsmm_blasint i, j, n;

  spgemmbig_csr(rowptr, colidx, values);
  n = LIBXSMM_UPDIV(libxsmm_cpuid_vlen(libxsmm_get_target_archid()), 8);

  { /* an in-range shape must still be JIT-ed and produce correct results */
    const libxsmm_blasint ld = 2 * n;
    double *const b = (double*)malloc(sizeof(double) * SPGEMMBIG_K * ld);
    double *const c = (double*)malloc(sizeof(double) * SPGEMMBIG_M * ld);
    if (NULL == b || NULL == c) {
      free(b); free(c);
      return EXIT_FAILURE;
    }
    for (i = 0; i < SPGEMMBIG_K * ld; ++i) b[i] = 1.0;
    memset(c, 0, sizeof(double) * SPGEMMBIG_M * ld);
    kernel = spgemmbig_dispatch(n, ld, ld, rowptr, colidx, values);
    if (NULL != kernel) {
      libxsmm_gemm_param param;
      memset(&param, 0, sizeof(param));
      param.b.primary = b;
      param.c.primary = c;
      kernel(&param);
      for (i = 0; i < SPGEMMBIG_M && EXIT_SUCCESS == result; ++i) {
        const double expect = SPGEMMBIG_NNZ_PER_ROW * (1.0 + (i % 3));
        for (j = 0; j < n; ++j) {
          if (LIBXSMM_NEQ(expect, c[(size_t)i * ld + j])) {
            fprintf(stderr, "ERROR: mismatch at (%i,%i): %f != %f\n",
              (int)i, (int)j, expect, c[(size_t)i * ld + j]);
            result = EXIT_FAILURE;
            break;
          }
        }
      }
      spgemmbig_release(kernel);
    }
    else {
      fprintf(stderr, "ERROR: in-range shape could not be dispatched!\n");
      result = EXIT_FAILURE;
    }
    free(b); free(c);
  }

  if (EXIT_SUCCESS == result) { /* C beyond 2 GiB */
    kernel = spgemmbig_dispatch(n, n, oversized_c, rowptr, colidx, values);
    if (NULL != kernel) {
      fprintf(stderr, "ERROR: ldc=%i was not rejected!\n", (int)oversized_c);
      spgemmbig_release(kernel);
      result = EXIT_FAILURE;
    }
  }

  if (EXIT_SUCCESS == result) { /* B beyond 2 GiB */
    kernel = spgemmbig_dispatch(n, oversized_b, n, rowptr, colidx, values);
    if (NULL != kernel) {
      fprintf(stderr, "ERROR: ldb=%i was not rejected!\n", (int)oversized_b);
      spgemmbig_release(kernel);
      result = EXIT_FAILURE;
    }
  }

  return result;
}
