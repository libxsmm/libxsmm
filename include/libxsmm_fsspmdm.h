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
#ifndef LIBXSMM_FSSPMDM_H
#define LIBXSMM_FSSPMDM_H

#include "libxsmm_typedefs.h"

#define libxsmm_dfsspmdm libxsmm_fsspmdm
#define libxsmm_sfsspmdm libxsmm_fsspmdm

/** Opaque type for Fixed-size Sparse Matrix x Dense Matrix (FsSpMDM). */
LIBXSMM_EXTERN_C typedef struct libxsmm_fsspmdm libxsmm_fsspmdm;

/**
 * Create a handle used for subsequent execution (libxsmm_fsspmdm_execute),
 * and optionally benchmark alternative kernels (if timer_tick is given).
 */
LIBXSMM_API libxsmm_fsspmdm* libxsmm_fsspmdm_create(libxsmm_datatype datatype,
  libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint K, libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  const void* alpha, const void* beta, const void* a_dense, int LIBXSMM_ARGDEF(c_is_nt, 0),
  libxsmm_timer_tickint LIBXSMM_ARGDEF((*timer_tick)(void), NULL));
LIBXSMM_API libxsmm_dfsspmdm* libxsmm_dfsspmdm_create(
  libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint K, libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  double alpha, double beta, const double* a_dense, int LIBXSMM_ARGDEF(c_is_nt, 0),
  libxsmm_timer_tickint LIBXSMM_ARGDEF((*timer_tick)(void), NULL));
LIBXSMM_API libxsmm_sfsspmdm* libxsmm_sfsspmdm_create(
  libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint K, libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  float alpha, float beta, const float* a_dense, int LIBXSMM_ARGDEF(c_is_nt, 0),
  libxsmm_timer_tickint LIBXSMM_ARGDEF((*timer_tick)(void), NULL));

LIBXSMM_API void libxsmm_fsspmdm_execute(const libxsmm_fsspmdm* handle, const void* B, void* C);
LIBXSMM_API void libxsmm_dfsspmdm_execute(const libxsmm_dfsspmdm* handle, const double* B, double* C);
LIBXSMM_API void libxsmm_sfsspmdm_execute(const libxsmm_sfsspmdm* handle, const float* B, float* C);

LIBXSMM_API void libxsmm_fsspmdm_destroy(libxsmm_fsspmdm* handle);
LIBXSMM_API void libxsmm_dfsspmdm_destroy(libxsmm_dfsspmdm* handle);
LIBXSMM_API void libxsmm_sfsspmdm_destroy(libxsmm_sfsspmdm* handle);

#endif /*LIBXSMM_FSSPMDM_H*/
