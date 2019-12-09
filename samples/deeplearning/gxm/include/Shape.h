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

#define MAX_DIMS 8

typedef struct
{
  int ndims; // Number of dimensions in tensor
  int dims[MAX_DIMS]; //Logical dimensions: for activations assume N,FM,H,W; for weight tensor assume OFM,IFM,KH,KW
} Shape;

