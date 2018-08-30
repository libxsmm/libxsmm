/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
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
/* Kunal Banerjee (Intel Corp.)
******************************************************************************/

LIBXSMM_VLA_DECL(4, LIBXSMM_BGEMM_TEMPLATE_TYPE, real_dst, (LIBXSMM_BGEMM_TEMPLATE_TYPE*)dst, handle->kb, handle->bn, handle->bk);
LIBXSMM_VLA_DECL(4, const LIBXSMM_BGEMM_TEMPLATE_TYPE, real_src, (const LIBXSMM_BGEMM_TEMPLATE_TYPE*)src, handle->nb, handle->bk, handle->bn);
libxsmm_blasint kb, nb, bk, bn;
libxsmm_blasint ii, jj, job, jobT;

if (handle->n == handle->k && handle->bn == handle->bk) {
  for (kb = 0; kb < handle->kb; ++kb) {
    for (nb = 0; nb < handle->nb; ++nb) {
      for (bk = 0; bk < handle->bk; ++bk) {
        for (bn = 0; bn < handle->bn; ++bn) {
          LIBXSMM_VLA_ACCESS(4, real_dst, nb, kb, bn, bk, handle->kb, handle->bn, handle->bk) =
            LIBXSMM_VLA_ACCESS(4, real_src, kb, nb, bk, bn, handle->nb, handle->bk, handle->bn);
        }
      }
    }
  }
} else {
  for (kb = 0; kb < handle->kb; ++kb) {
    for (nb = 0; nb < handle->nb; ++nb) {
      for (bk = 0; bk < handle->bk; ++bk) {
        for (bn = 0; bn < handle->bn; ++bn) {
          job = (kb*handle->bk + bk)*handle->n + (nb*handle->bn + bn);
          ii = job / handle->k;
          jj = job % handle->k;
          jobT = jj*handle->n + ii;
          LIBXSMM_VLA_ACCESS(4, real_dst, (jobT/handle->k)/handle->bn, (jobT%handle->k)/handle->bk, (jobT/handle->k)%handle->bn, (jobT%handle->k)%handle->bk, handle->kb, handle->bn, handle->bk) =
            LIBXSMM_VLA_ACCESS(4, real_src, kb, nb, bk, bn, handle->nb, handle->bk, handle->bn);
        }
      }
    }
  }
}
