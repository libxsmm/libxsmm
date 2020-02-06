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
#include <libxsmm_macros.h>

#if !defined(ELEM_TYPE)
# define ELEM_TYPE short
#endif

#define VLA_IJK_DECL(DIM, TYPE, ARRAY, DATA, S1, S2) LIBXSMM_VLA_DECL(  DIM, TYPE, ARRAY, DATA, S1, S2)
#define VLA_IJK_INDX(DIM, ARRAY, I0, I1, I2, S1, S2) LIBXSMM_VLA_ACCESS(DIM, ARRAY, I0, I1, I2, S1, S2)

#define VLA_IKJ_DECL(DIM, TYPE, ARRAY, DATA, S1, S2) LIBXSMM_VLA_DECL(  DIM, TYPE, ARRAY, DATA, S2, S1)
#define VLA_IKJ_INDX(DIM, ARRAY, I0, I1, I2, S1, S2) LIBXSMM_VLA_ACCESS(DIM, ARRAY, I0, I2, I1, S2, S1)


int main(/*int argc, char* argv[]*/)
{
  int ni = 9, nj = 7, nk = 3, i, j, k, linear = 0, result = EXIT_SUCCESS;
  ELEM_TYPE *const input = (ELEM_TYPE*)malloc(sizeof(ELEM_TYPE) * ni * nj * nk);
  LIBXSMM_VLA_DECL(1, const ELEM_TYPE, in1, input);
  VLA_IJK_DECL(3, const ELEM_TYPE, jk3, input, nj, nk);
  VLA_IKJ_DECL(3, const ELEM_TYPE, kj3, input, nj, nk);

  LIBXSMM_ASSERT(NULL != input);
  for (i = 0; i < (ni * nj * nk); ++i) input[i] = (ELEM_TYPE)i;
  for (i = 0; i < ni && EXIT_SUCCESS == result; ++i) {
    for (j = 0; j < nj; ++j) {
      for (k = 0; k < nk; ++k) {
        const ELEM_TYPE gold0 = input[linear];
        const ELEM_TYPE test0 = VLA_IJK_INDX(3, jk3, i, j, k, nj, nk);
        const ELEM_TYPE gold1 = VLA_IJK_INDX(3, kj3, i, k, j, nk, nj);
        const ELEM_TYPE test1 = VLA_IKJ_INDX(3, kj3, i, j, k, nj, nk);
        if (gold0 != LIBXSMM_VLA_ACCESS(1, in1, linear) ||
            gold0 != test0 || gold1 != test1)
        {
          result = EXIT_FAILURE;
          j = nj; break;
        }
        ++linear;
      }
    }
  }

  free(input);
  return result;
}

