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

/** Opaque types for fsspmdm */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dfsspmdm libxsmm_dfsspmdm;
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_sfsspmdm libxsmm_sfsspmdm;

LIBXSMM_API libxsmm_dfsspmdm* libxsmm_dfsspmdm_create( libxsmm_blasint M,   libxsmm_blasint   N, libxsmm_blasint   K,
                                                       libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
                                                       const double alpha, const double beta, libxsmm_blasint c_is_nt,
                                                       const double* a_dense );

LIBXSMM_API void libxsmm_dfsspmdm_execute( const libxsmm_dfsspmdm* handle, const double* B, double* C );

LIBXSMM_API void libxsmm_dfsspmdm_destroy( libxsmm_dfsspmdm* handle );

LIBXSMM_API libxsmm_sfsspmdm* libxsmm_sfsspmdm_create( libxsmm_blasint M,   libxsmm_blasint   N, libxsmm_blasint   K,
                                                       libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
                                                       const float alpha, const float beta, libxsmm_blasint c_is_nt,
                                                       const float* a_dense );

LIBXSMM_API void libxsmm_sfsspmdm_execute( const libxsmm_sfsspmdm* handle, const float* B, float* C );

LIBXSMM_API void libxsmm_sfsspmdm_destroy( libxsmm_sfsspmdm* handle );

#endif /*LIBXSMM_FSSPMDM_H*/

