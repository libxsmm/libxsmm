/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Kunal Banerjee (Intel Corp.)
******************************************************************************/

LIBXSMM_VLA_DECL(4, LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE, real_dst, (LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE*)dst, handle->kb, handle->bn, handle->bk);
LIBXSMM_VLA_DECL(4, const LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE, real_src, (const LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE*)src, handle->nb, handle->bk, handle->bn);
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
