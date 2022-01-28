/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Evangelos Georganas, Hans Pabst (Intel Corp.)
******************************************************************************/

int i1, i2, i3, i4, i5, i6;
int lpb, bfm, W, H, fmb, N, C;
/* low precision formatting */
if ( tensor->layout->num_dims == 6 ) {
  lpb = tensor->layout->dim_size[0];
  bfm = tensor->layout->dim_size[1];
  W = tensor->layout->dim_size[2];
  H = tensor->layout->dim_size[3];
  fmb = tensor->layout->dim_size[4];
  N = tensor->layout->dim_size[5];
} else {
  lpb = 1;
  bfm = tensor->layout->dim_size[0];
  W = tensor->layout->dim_size[1];
  H = tensor->layout->dim_size[2];
  fmb = tensor->layout->dim_size[3];
  N = tensor->layout->dim_size[4];
}
C = fmb * bfm * lpb;

/* printf(" layout act copy out  N %i fmb %i H %i W %i bfm %i lpb %i \n", N, fmb, H, W, bfm, lpb); */
{
  LIBXSMM_VLA_DECL(6, const element_type, handle_data_1, (const element_type*)tensor->data, fmb, H, W, bfm, lpb);
  LIBXSMM_VLA_DECL(4, element_type, user_data, (element_type*)data, C, H, W);

  for (i1 = 0; i1 < N; ++i1) {
    for (i2 = 0; i2 < fmb; ++i2) {
      for (i3 = 0; i3 < H; ++i3) {
        for (i4 = 0; i4 < W; ++i4) {
          for (i5 = 0; i5 < bfm; ++i5) {
            for (i6 = 0; i6 < lpb; ++i6) {
              LIBXSMM_VLA_ACCESS(4, user_data, i1, ((size_t)i2*bfm*lpb) + ((size_t)i5*lpb) + i6, i3, i4, C, H, W) =
                LIBXSMM_VLA_ACCESS(6, handle_data_1, i1, i2, i3, i4, i5, i6, fmb, H, W, bfm, lpb);
            }
          }
        }
      }
    }
  }
}
