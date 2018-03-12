/******************************************************************************
** Copyright (c) 2015-2018, Intel Corporation                                **
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

/* must match definitions in headeronly_aux.c */
#if !defined(REAL_TYPE)
# define REAL_TYPE double
#endif
#if !defined(REAL_OUT)
# define REAL_OUT REAL_TYPE
#endif


LIBXSMM_EXTERN LIBXSMM_MMFUNCTION_TYPE2(REAL_TYPE, REAL_OUT) mmdispatch(int m, int n, int k);


int main(void)
{
  const int m = LIBXSMM_MAX_M, n = LIBXSMM_MAX_N, k = LIBXSMM_MAX_K;
  const LIBXSMM_MMFUNCTION_TYPE2(REAL_TYPE, REAL_OUT) fa = LIBXSMM_MMDISPATCH_SYMBOL2(REAL_TYPE, REAL_OUT)(m, n, k,
    NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/, NULL/*alpha*/, NULL/*beta*/,
    NULL/*flags*/, NULL/*prefetch*/);
  const LIBXSMM_MMFUNCTION_TYPE2(REAL_TYPE, REAL_OUT) fb = mmdispatch(m, n, k);
  int result = EXIT_SUCCESS;
  if (fa != fb) {
    result = EXIT_FAILURE;
  }
  return result;
}

