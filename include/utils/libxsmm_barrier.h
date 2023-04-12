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
#ifndef LIBXSMM_UTILS_BARRIER_H
#define LIBXSMM_UTILS_BARRIER_H

#include "../libxsmm_macros.h"


/** Opaque type which represents a barrier. */
LIBXSMM_EXTERN_C typedef struct libxsmm_barrier libxsmm_barrier;

/** Create barrier from one of the threads. */
LIBXSMM_API libxsmm_barrier* libxsmm_barrier_create(int ncores, int nthreads_per_core);
/** Initialize the barrier from each thread of the team. */
LIBXSMM_API void libxsmm_barrier_init(libxsmm_barrier* barrier, int tid);
/** Wait for the entire team to arrive. */
LIBXSMM_API void libxsmm_barrier_wait(libxsmm_barrier* barrier, int tid);
/** Destroy the resources associated with this barrier. */
LIBXSMM_API void libxsmm_barrier_destroy(const libxsmm_barrier* barrier);

#endif /*LIBXSMM_UTILS_BARRIER_H*/
