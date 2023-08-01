/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Dhiraj Kalamkar (Intel Corp.)
******************************************************************************/
#include "utils.h"
thread_local struct drand48_data rand_buf;

void set_random_seed(int seed)
{
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    srand48_r(seed+tid, &rand_buf);
  }
}
