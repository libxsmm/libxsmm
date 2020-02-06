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


int main(void)
{
  const int defaults[] = { LIBXSMM_GEMM_FLAG_NONE,
    LIBXSMM_GEMM_FLAG_TRANS_A,  LIBXSMM_GEMM_FLAG_TRANS_B,
    LIBXSMM_GEMM_FLAG_TRANS_A | LIBXSMM_GEMM_FLAG_TRANS_B
  };
  const char trans[] = "NnTtCcX";
  const int ndefaults = sizeof(defaults) / sizeof(*defaults), ntrans = sizeof(trans);
  int result = EXIT_SUCCESS;
  int i, j = -1, k = -1, flags = 0;

  for (i = 0; i < ndefaults && EXIT_SUCCESS == result; ++i) {
    flags = LIBXSMM_GEMM_PFLAGS(0, 0, defaults[i]);
    if (defaults[i] != flags) { result = EXIT_FAILURE; break; }
    for (j = 0; j < ntrans && EXIT_SUCCESS == result; ++j) {
      flags = LIBXSMM_GEMM_PFLAGS(trans + j, 0, defaults[i]);
      if (0 != (LIBXSMM_GEMM_FLAG_TRANS_A & flags) && ('N' == trans[j] || 'n' == trans[j])) { result = EXIT_FAILURE; break; }
      if (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & flags) && ('T' == trans[j] || 't' == trans[j])) { result = EXIT_FAILURE; break; }
      if (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & flags) && ('C' == trans[j] || 'c' == trans[j])) { result = EXIT_FAILURE; break; }
      for (k = 0; k < ntrans; ++k) {
        flags = LIBXSMM_GEMM_PFLAGS(0, trans + k, defaults[i]);
        if (0 != (LIBXSMM_GEMM_FLAG_TRANS_B & flags) && ('N' == trans[k] || 'n' == trans[k])) { result = EXIT_FAILURE; break; }
        if (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & flags) && ('T' == trans[k] || 't' == trans[k])) { result = EXIT_FAILURE; break; }
        if (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & flags) && ('C' == trans[k] || 'c' == trans[k])) { result = EXIT_FAILURE; break; }
        flags = LIBXSMM_GEMM_PFLAGS(trans + j, trans + k, defaults[i]);
        if (0 != (LIBXSMM_GEMM_FLAG_TRANS_A & flags) && ('N' == trans[j] || 'n' == trans[j])) { result = EXIT_FAILURE; break; }
        if (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & flags) && ('T' == trans[j] || 't' == trans[j])) { result = EXIT_FAILURE; break; }
        if (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & flags) && ('C' == trans[j] || 'c' == trans[j])) { result = EXIT_FAILURE; break; }
        if (0 != (LIBXSMM_GEMM_FLAG_TRANS_B & flags) && ('N' == trans[k] || 'n' == trans[k])) { result = EXIT_FAILURE; break; }
        if (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & flags) && ('T' == trans[k] || 't' == trans[k])) { result = EXIT_FAILURE; break; }
        if (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & flags) && ('C' == trans[k] || 'c' == trans[k])) { result = EXIT_FAILURE; break; }
      }
    }
  }

#if defined(_DEBUG)
  if (EXIT_SUCCESS != result) {
    fprintf(stderr, "%c%c -> %i\n", 0 <= j ? trans[j] : '0', 0 <= k ? trans[k] : '0', flags);
  }
#endif

  return result;
}

