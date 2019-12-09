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

LIBXSMM_VLA_DECL(4, LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE, real_dst, (LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE*)dst, handle->kb, handle->bk, handle->bm);
LIBXSMM_VLA_DECL(2, const LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE, real_src, (const LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE*)src, handle->m);
libxsmm_blasint mb, kb, bm, bk;

for (mb = 0; mb < handle->mb; ++mb) {
  for (kb = 0; kb < handle->kb; ++kb) {
    for (bk = 0; bk < handle->bk; ++bk) {
      for (bm = 0; bm < handle->bm; ++bm) {
        LIBXSMM_VLA_ACCESS(4, real_dst, mb, kb, bk, bm, handle->kb, handle->bk, handle->bm) =
        LIBXSMM_VLA_ACCESS(2, real_src, kb * handle->bk + bk, mb * handle->bm + bm, handle->m);
      }
    }
  }
}

