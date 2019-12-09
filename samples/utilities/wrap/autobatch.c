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
#include <libxsmm.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#if !defined(ITYPE)
# define ITYPE double
#endif
#if !defined(GEMM)
# if defined(WRAP)
#   define GEMM LIBXSMM_BLAS_SYMBOL(ITYPE, gemm)
# else /* prototype for LIBXSMM's wrapped GEMM; this way auto-batch can be tested as if GEMM calls are intercepted */
#   define GEMM LIBXSMM_FSYMBOL(LIBXSMM_CONCATENATE(__wrap_, LIBXSMM_TPREFIX(ITYPE, gemm)))
# endif
#endif
#if !defined(CALL_BEGIN_END)
# define CALL_BEGIN_END
#endif


void GEMM(const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const ITYPE*, const ITYPE*, const libxsmm_blasint*, const ITYPE*, const libxsmm_blasint*,
  const ITYPE*, ITYPE*, const libxsmm_blasint*);


int main(int argc, char* argv[])
{
  const libxsmm_blasint maxn = 1 < argc ? atoi(argv[1]) : 23;
  const libxsmm_blasint maxv = LIBXSMM_MIN(2 < argc ? atoi(argv[2]) : 2, maxn);
  const libxsmm_blasint size = 3 < argc ? atoi(argv[3]) : 1000;

  const libxsmm_blasint m = ((rand() % maxv) + 1) * maxn / maxv;
  const libxsmm_blasint n = ((rand() % maxv) + 1) * maxn / maxv;
  const libxsmm_blasint k = ((rand() % maxv) + 1) * maxn / maxv;
  const libxsmm_blasint lda = m, ldb = k, ldc = m;
  const ITYPE alpha = 1.0, beta = 0.0;
  const char transa = 'N', transb = 'N';
#if defined(CALL_BEGIN_END)
  const int flags = LIBXSMM_GEMM_FLAGS(transa, transb)
# if 0
    | LIBXSMM_MMBATCH_FLAG_SEQUENTIAL
# endif
# if 1
    | LIBXSMM_MMBATCH_FLAG_STATISTIC
# endif
  ;
#endif

  ITYPE *a = 0, *b = 0, *c = 0;
  int result = EXIT_SUCCESS, i;

  libxsmm_init();

  a = (ITYPE*)malloc((size_t)maxn * (size_t)maxn * sizeof(ITYPE));
  b = (ITYPE*)malloc((size_t)maxn * (size_t)maxn * sizeof(ITYPE));
  c = (ITYPE*)malloc((size_t)maxn * (size_t)maxn * sizeof(ITYPE));
  if (0 == a || 0 == b || 0 == c) result = EXIT_FAILURE;

  if (EXIT_SUCCESS == result) {
    LIBXSMM_MATINIT_OMP(ITYPE, 42, a, maxn, maxn, maxn, 1.0);
    LIBXSMM_MATINIT_OMP(ITYPE, 24, b, maxn, maxn, maxn, 1.0);
    LIBXSMM_MATINIT_OMP(ITYPE, 0, c, maxn, maxn, maxn, 1.0);

#if defined(_OPENMP)
#   pragma omp parallel private(i)
#endif
    {
#if defined(CALL_BEGIN_END)
# if defined(_OPENMP)
#     pragma omp single nowait
# endif /* enable batch-recording of the specified matrix multiplication */
      libxsmm_mmbatch_begin(LIBXSMM_GEMM_PRECISION(ITYPE),
        &flags, &m, &n, &k, &lda, &ldb, &ldc, &alpha, &beta);
#endif
#if defined(_OPENMP)
#     pragma omp for
#endif
      for (i = 0; i < size; ++i) {
        const libxsmm_blasint mi = ((rand() % maxv) + 1) * maxn / maxv;
        const libxsmm_blasint ni = ((rand() % maxv) + 1) * maxn / maxv;
        const libxsmm_blasint ki = ((rand() % maxv) + 1) * maxn / maxv;
        const libxsmm_blasint ilda = mi, ildb = ki, ildc = mi;
        assert(0 < mi && 0 < ni && 0 < ki && mi <= ilda && ki <= ildb && mi <= ildc);
        GEMM(&transa, &transb, &mi, &ni, &ki, &alpha, a, &ilda, b, &ildb, &beta, c, &ildc);
      }
#if defined(CALL_BEGIN_END)
# if defined(_OPENMP)
#     pragma omp single nowait
# endif /* disable/flush multiplication batch */
      libxsmm_mmbatch_end();
#endif
    }
  }

  libxsmm_finalize();
  free(a);
  free(b);
  free(c);

  return result;
}

