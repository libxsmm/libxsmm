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

/* @TODO: use for-loops to potentially leverage NUMA in the future */
int i1, i2, i3, i4, i5, i6, i7;
int lpb = 0;
int bofm = 0;
int bifm = 0;
int S = 0;
int R = 0;
int ifmb = 0;
int ofmb = 0;
/* low precision formatting */
if ( tensor->layout->num_dims == 7 ) {
  lpb = tensor->layout->dim_size[0];
  bofm = tensor->layout->dim_size[1];
  bifm = tensor->layout->dim_size[2];
  S = tensor->layout->dim_size[3];
  R = tensor->layout->dim_size[4];
  ifmb = tensor->layout->dim_size[5];
  ofmb = tensor->layout->dim_size[6];
} else if ( tensor->layout->num_dims == 6 ) {
  lpb = 1;
  bofm = tensor->layout->dim_size[0];
  bifm = tensor->layout->dim_size[1];
  S = tensor->layout->dim_size[2];
  R = tensor->layout->dim_size[3];
  ifmb = tensor->layout->dim_size[4];
  ofmb = tensor->layout->dim_size[5];
} else {
  /* should not happen, @TODO throw ERR */
}

{
  LIBXSMM_VLA_DECL(4, element_type, user_data, (element_type*)data, ifmb * bifm * lpb, R, S);
  LIBXSMM_VLA_DECL(7, const element_type, handle_data_1, (const element_type*)tensor->data, ifmb, R, S, bifm, bofm, lpb);

  for (i1 = 0; i1 < ofmb; ++i1) {
    for (i2 = 0; i2 < ifmb; ++i2) {
      for (i3 = 0; i3 < R; ++i3) {
        for (i4 = 0; i4 < S; ++i4) {
          for (i5 = 0; i5 < bifm; ++i5) {
            for (i6 = 0; i6 < bofm; ++i6) {
              for (i7 = 0; i7 < lpb; ++i7) {
                LIBXSMM_VLA_ACCESS(4, user_data, i1 * bofm + i6, ((size_t)i2*bifm*lpb) + ((size_t)i5*lpb) + i7, i3, i4, ifmb * bifm * lpb, R, S) =
                LIBXSMM_VLA_ACCESS(7, handle_data_1, i1, i2, i3, i4, i5, i6, i7, ifmb, R, S, bifm, bofm, lpb);
              }
            }
          }
        }
      }
    }
  }
}

