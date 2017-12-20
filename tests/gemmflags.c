/******************************************************************************
** Copyright (c) 2017-2018, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm_source.h>
#include <stdlib.h>
#if defined(_DEBUG)
# include <stdio.h>
#endif


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

