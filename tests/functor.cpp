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

#if defined(_DEBUG)
# define FPRINTF(STREAM, ...) fprintf(STREAM, __VA_ARGS__)
#else
# define FPRINTF(STREAM, ...)
#endif


template<typename T>
void init_buf(T* buf, int size) {
  for (int i = 0; i < size; ++i) {
    buf[i] = (T)(1.0f * rand() / RAND_MAX);
  }
}


int main()
{
  float alpha = 1, beta = 0;
  int result = EXIT_SUCCESS;
  int dim = 32;

  libxsmm_mmfunction<libxsmm_bfloat16> kernel(LIBXSMM_GEMM_FLAG_NONE, dim, dim, dim, alpha, beta);
  if (kernel) { /* AVX-512 is LIBXSMM's prerequisite for Bfloat16 (no further emulation) */
    libxsmm_bfloat16 *const abf16 = new libxsmm_bfloat16[dim*dim];
    libxsmm_bfloat16 *const bbf16 = new libxsmm_bfloat16[dim*dim];
    float *const a = new float[dim*dim];
    float *const b = new float[dim*dim];
    float *const c = new float[dim*dim];

    init_buf(a, dim * dim);
    init_buf(b, dim * dim);
    libxsmm_rne_convert_fp32_bf16(a, abf16, dim * dim);
    libxsmm_rne_convert_fp32_bf16(b, bbf16, dim * dim);

    kernel(abf16, bbf16, c);
    FPRINTF(stderr, "c[0]=%f, c[1]=%f\n", c[0], c[1]);

    libxsmm_bsgemm(NULL, NULL, &dim, &dim, &dim, &alpha, abf16, &dim, bbf16, &dim, &beta, c, &dim);
    FPRINTF(stderr, "c[0]=%f, c[1]=%f\n", c[0], c[1]);

    delete[] abf16;
    delete[] bbf16;
    delete[] a;
    delete[] b;
    delete[] c;
  }
  else if (LIBXSMM_X86_AVX512 <= libxsmm_get_target_archid()) {
    FPRINTF(stderr, "failed to generate kernel\n");
    result = EXIT_FAILURE;
  }

  return result;
}

