/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Sasikanth Avancha, Dhiraj Kalamkar (Intel Corp.)
******************************************************************************/


#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "libxsmm.h"

 void check_physical_pad(const char *s, float *tensor, int nImg, int nBfm, int fh, int fw, int ifm, int iph, int ipw );
 void check_physical_pad(const char *s, libxsmm_bfloat16 *tensor, int nImg, int nBfm, int fh, int fw, int ifm, int iph, int ipw );
 void MeanOfLayer(char *s, float *array, int size);
 void MeanOfLayer(char *s, double *array, int size);
 void MeanOfLayer(char *s, int *array, int size);
 void MeanOfLayer(char *s, libxsmm_bfloat16 *array, int size);
