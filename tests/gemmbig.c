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

/* Shapes beyond the intended small-GEMM range must not crash: the generator either JITs them
 * or bails out such that the reference kernel is dispatched (see libxsmm/libxsmm#992). */

#if !defined(GEMMBIG_BATCH)
# define GEMMBIG_BATCH 2
#endif
#if !defined(GEMMBIG_EPSILON)
# define GEMMBIG_EPSILON 0.05
#endif


LIBXSMM_INLINE float gemmbig_tof32(libxsmm_bfloat16 value)
{
  union { float f; unsigned int u; } cvt;
  cvt.u = ((unsigned int)value) << 16;
  return cvt.f;
}


LIBXSMM_INLINE libxsmm_bfloat16 gemmbig_tobf16(float value)
{
  union { float f; unsigned int u; } cvt;
  cvt.f = value;
  return (libxsmm_bfloat16)((cvt.u + 0x8000) >> 16);
}


LIBXSMM_INLINE void gemmbig_fill(libxsmm_bfloat16* data, size_t size, unsigned int seed)
{
  size_t i;
  for (i = 0; i < size; ++i) {
    seed = seed * 1103515245u + 12345u;
    data[i] = gemmbig_tobf16(((float)((seed >> 16) & 0xFF) / 255.0f) - 0.5f);
  }
}


LIBXSMM_INLINE int gemmbig_run(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k)
{
  const libxsmm_blasint lda = m, ldb = k, ldc = m;
  const size_t size_a = (size_t)lda * k * GEMMBIG_BATCH;
  const size_t size_b = (size_t)ldb * n * GEMMBIG_BATCH;
  const size_t size_c = (size_t)ldc * n;
  libxsmm_bfloat16 *const a = (libxsmm_bfloat16*)malloc(sizeof(libxsmm_bfloat16) * size_a);
  libxsmm_bfloat16 *const b = (libxsmm_bfloat16*)malloc(sizeof(libxsmm_bfloat16) * size_b);
  libxsmm_bfloat16 *const c = (libxsmm_bfloat16*)malloc(sizeof(libxsmm_bfloat16) * size_c);
  libxsmm_gemm_batch_reduce_config brconfig;
  libxsmm_gemm_shape shape;
  libxsmm_gemmfunction kernel;
  libxsmm_gemm_param param;
  unsigned long long batch = GEMMBIG_BATCH;
  int result = EXIT_SUCCESS;
  libxsmm_blasint i, j, l;

  if (NULL == a || NULL == b || NULL == c) {
    free(a); free(b); free(c);
    return EXIT_FAILURE;
  }
  gemmbig_fill(a, size_a, 1u);
  gemmbig_fill(b, size_b, 2u);
  memset(c, 0, sizeof(libxsmm_bfloat16) * size_c);

  shape = libxsmm_create_gemm_shape(m, n, k, lda, ldb, ldc,
    LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32);
  memset(&brconfig, 0, sizeof(brconfig));
  brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
  brconfig.br_stride_a_hint = (libxsmm_blasint)(sizeof(libxsmm_bfloat16) * lda * k);
  brconfig.br_stride_b_hint = (libxsmm_blasint)(sizeof(libxsmm_bfloat16) * ldb * n);
  kernel = libxsmm_dispatch_brgemm(shape,
    LIBXSMM_GEMM_FLAGS('N', 'N') | LIBXSMM_GEMM_FLAG_BETA_0, LIBXSMM_GEMM_PREFETCH_NONE, brconfig);

  if (NULL != kernel) {
    memset(&param, 0, sizeof(param));
    param.a.primary = a;
    param.b.primary = b;
    param.c.primary = c;
    param.op.tertiary = &batch;
    kernel(&param);

    for (j = 0; j < n && EXIT_SUCCESS == result; ++j) {
      for (i = 0; i < m; ++i) {
        float expect = 0;
        libxsmm_blasint r;
        for (r = 0; r < GEMMBIG_BATCH; ++r) {
          const libxsmm_bfloat16 *const ar = a + (size_t)r * lda * k;
          const libxsmm_bfloat16 *const br = b + (size_t)r * ldb * n;
          for (l = 0; l < k; ++l) {
            expect += gemmbig_tof32(ar[(size_t)l * lda + i]) * gemmbig_tof32(br[(size_t)j * ldb + l]);
          }
        }
        if (GEMMBIG_EPSILON < LIBXSMM_ABS(expect - gemmbig_tof32(c[(size_t)j * ldc + i]))
          * (1.0 / LIBXSMM_MAX(LIBXSMM_ABS(expect), 1.0)))
        {
          fprintf(stderr, "ERROR: m=%i n=%i k=%i mismatch at (%i,%i): %f != %f\n",
            (int)m, (int)n, (int)k, (int)i, (int)j, expect, gemmbig_tof32(c[(size_t)j * ldc + i]));
          result = EXIT_FAILURE;
          break;
        }
      }
    }
  }
  else { /* neither a JIT-ed nor a reference kernel: dispatch must never silently fail */
    fprintf(stderr, "ERROR: m=%i n=%i k=%i could not be dispatched!\n", (int)m, (int)n, (int)k);
    result = EXIT_FAILURE;
  }

  free(a); free(b); free(c);
  return result;
}


int main(void)
{
  /* shapes taken from libxsmm/libxsmm#992 (scaled down in N to bound the test runtime) */
  static const libxsmm_blasint shapes[][3] = {
    {    64, 32,  64 },
    {  2048, 16, 512 },
    {  4096, 16, 512 },
    {  8192,  8, 768 }
  };
  const int nshapes = (int)(sizeof(shapes) / sizeof(*shapes));
  int result = EXIT_SUCCESS, i;

  for (i = 0; i < nshapes && EXIT_SUCCESS == result; ++i) {
    result = gemmbig_run(shapes[i][0], shapes[i][1], shapes[i][2]);
  }

  return result;
}
