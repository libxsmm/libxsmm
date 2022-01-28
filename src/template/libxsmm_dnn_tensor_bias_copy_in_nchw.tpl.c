/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/

/* use for-loops to potentially leverage NUMA in the future */
int i1, i2, i3;
#if defined(LIBXSMM_DNN_COPY_LOW_PRECISION)
int lpb = tensor->layout->dim_size[0];
int bfm = tensor->layout->dim_size[1];
int fmb = tensor->layout->dim_size[2];
#else
int lpb = 1;
int bfm = tensor->layout->dim_size[0];
int fmb = tensor->layout->dim_size[1];
#endif

const element_type* user_data = (const element_type*)data;
LIBXSMM_VLA_DECL(3, element_type, handle_data, (element_type*)tensor->data, bfm, lpb);

for (i1 = 0; i1 < fmb; ++i1) {
  for (i2 = 0; i2 < bfm; ++i2) {
    for (i3 = 0; i3 < lpb; ++i3) {
      LIBXSMM_VLA_ACCESS(3, handle_data, i1, i2, i3, bfm, lpb) = user_data[(i1*bfm*lpb) + (i2*lpb) + i3];
    }
  }
}

