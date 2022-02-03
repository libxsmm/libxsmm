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

/* this is crappy as it requires a complicated if... */
if (handle->desc.pad_h_in > 0 || handle->desc.pad_w_in > 0) {
  for ( ij = 0; ij < handle->ifhp; ij++ ) {
    for ( ii = 0; ii < handle->ifwp; ii++ ) {
      if ( (ij < handle->desc.pad_h_in) || (ij >= (handle->desc.H+handle->desc.pad_h_in)) ||
           (ii < handle->desc.pad_w_in) || (ii >= (handle->desc.W+handle->desc.pad_w_in)) ) {
        for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
          LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1lpblock, ij, ii, ifm2, handle->blocksifm*handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock) = (element_input_type)0;
        }
      }
    }
  }
}

