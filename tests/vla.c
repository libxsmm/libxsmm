/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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
#include <libxsmm_macros.h>
#include <stdlib.h>
#include <assert.h>

#define ELEMENT_TYPE short


int main(/*int argc, char* argv[]*/)
{
  int ni = 9, nj = 7, nk = 3, i, j, k, linear = 0, result = EXIT_SUCCESS;
  ELEMENT_TYPE* in1 = (ELEMENT_TYPE*)malloc(ni * nj * nk * sizeof(ELEMENT_TYPE));
  LIBXSMM_VLA_DECL(3, const ELEMENT_TYPE, in3, in1, nj, nk);

  assert(0 != in1);
  for (i = 0; i < (ni * nj * nk); ++i) in1[i] = (ELEMENT_TYPE)i;
  for (i = 0; i < ni; ++i) {
    for (j = 0; j < nj; ++j) {
      for (k = 0; k < nk; ++k) {
        if (in1[linear] != LIBXSMM_VLA_ACCESS(3, in3, i, j, k, nj, nk)) {
          result = EXIT_FAILURE;
          i = ni; j = nj;
          break;
        }
        ++linear;
      }
    }
  }

  free(in1);
  return result;
}
