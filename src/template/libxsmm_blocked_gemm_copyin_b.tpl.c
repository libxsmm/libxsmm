/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Hans Pabst (Intel Corp.)
******************************************************************************/

LIBXSMM_VLA_DECL(4, LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE, real_dst, (LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE*)dst, handle->kb, handle->bn, handle->bk);
LIBXSMM_VLA_DECL(2, const LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE, real_src, (const LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE*)src, handle->k);
libxsmm_blasint kb, nb, bk, bn;

for (nb = 0; nb < handle->nb; ++nb) {
  for (kb = 0; kb < handle->kb; ++kb) {
    for (bn = 0; bn < handle->bn; ++bn) {
      for (bk = 0; bk < handle->bk; ++bk) {
        LIBXSMM_VLA_ACCESS(4, real_dst, nb, kb, bn, bk, handle->kb, handle->bn, handle->bk) =
        LIBXSMM_VLA_ACCESS(2, real_src, nb * handle->bn + bn, kb * handle->bk + bk, handle->k);
      }
    }
  }
}

