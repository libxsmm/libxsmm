/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Kunal Banerjee (Intel Corp.), Dheevatsa Mudigere (Intel Corp.)
   Alexander Heinecke (Intel Corp.), Hans Pabst (Intel Corp.)
******************************************************************************/
#include "libxsmm_blocked_gemm_types.h"
#include "libxsmm_main.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_APIEXT void libxsmm_blocked_gemm_omp(const libxsmm_blocked_gemm_handle* handle,
  const void* a, const void* b, void* c, /*unsigned*/int count)
{
  static int error_once = 0;
  if (0 != handle && 0 != a && 0 != b && 0 != c && 0 < count) {
#if defined(_OPENMP)
#   pragma omp parallel num_threads(handle->nthreads)
#endif /*defined(_OPENMP)*/
    {
      int i;
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      for (i = 0; i < count; ++i) {
        libxsmm_blocked_gemm_st(handle, a, b, c, 0/*start_thread*/, tid);
      }
    }
  }
  else if (0 != libxsmm_get_verbosity() /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: invalid arguments for libxsmm_blocked_gemm_omp!\n");
  }
}

