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
#ifndef LIBXSMM_EXT_GEMM_H
#define LIBXSMM_EXT_GEMM_H

#include <libxsmm.h>

#if defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT) && defined(LIBXSMM_BLAS_WRAP_DYNAMIC)
LIBXSMM_APIEXT libxsmm_dgemm_batch_strided_function libxsmmext_original_dgemm_batch_strided(void);
LIBXSMM_APIEXT libxsmm_sgemm_batch_strided_function libxsmmext_original_sgemm_batch_strided(void);
LIBXSMM_APIEXT libxsmm_dgemm_batch_function libxsmmext_original_dgemm_batch(void);
LIBXSMM_APIEXT libxsmm_sgemm_batch_function libxsmmext_original_sgemm_batch(void);
LIBXSMM_APIEXT libxsmm_dgemm_function libxsmmext_original_dgemm(void);
LIBXSMM_APIEXT libxsmm_sgemm_function libxsmmext_original_sgemm(void);
LIBXSMM_APIEXT libxsmm_dgemv_function libxsmmext_original_dgemv(void);
LIBXSMM_APIEXT libxsmm_sgemv_function libxsmmext_original_sgemv(void);
#endif

#endif /*LIBXSMM_EXT_GEMM_H*/
