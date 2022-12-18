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

#define LIBXSMM_PARALLEL_KERNEL_TEST
#if defined(_OPENMP)
# include <omp.h>
#endif
#include "gemm_kernel.c"
#undef LIBXSMM_PARALLEL_KERNEL_TEST

