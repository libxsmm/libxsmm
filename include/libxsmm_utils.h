/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_UTILS_H
#define LIBXSMM_UTILS_H

/**
 * Any intrinsics interface (libxsmm_intrinsics_x86.h) shall be explicitly
 * included, i.e., separate from libxsmm_utils.h.
*/
#include "utils/libxsmm_lpflt_quant.h"
#include "utils/libxsmm_barrier.h"
#include "utils/libxsmm_timer.h"
#include "utils/libxsmm_math.h"

#if defined(__BLAS) && (1 == __BLAS)
# if defined(__OPENBLAS)
    LIBXSMM_EXTERN void openblas_set_num_threads(int num_threads);
#   define LIBXSMM_BLAS_INIT openblas_set_num_threads(1);
# endif
#endif
#if !defined(LIBXSMM_BLAS_INIT)
# define LIBXSMM_BLAS_INIT
#endif

#endif /*LIBXSMM_UTILS_H*/
