/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/*  Hans Pabst, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm_sync.h>
#include "libxsmm_main.h"

#if !defined(LIBXSMM_SYNC_FUTEX) && defined(__linux__) && defined(__USE_GNU)
# define LIBXSMM_SYNC_FUTEX
#endif

#include <stdint.h>
#if defined(_WIN32)
# include <process.h>
#else
# if defined(LIBXSMM_SYNC_FUTEX) && defined(__linux__) && defined(__USE_GNU)
#   include <linux/futex.h>
# endif
# include <time.h>
#endif

#if !defined(LIBXSMM_SYNC_RWLOCK_BITS)
# if defined(__MINGW32__)
#   define LIBXSMM_SYNC_RWLOCK_BITS 32
# else
#   define LIBXSMM_SYNC_RWLOCK_BITS 16
# endif
#endif


LIBXSMM_API unsigned int libxsmm_get_pid(void)
{
#if defined(_WIN32)
  return (unsigned int)_getpid();
#else
  return (unsigned int)getpid();
#endif
}


LIBXSMM_API_INTERN unsigned int internal_get_tid(void);
LIBXSMM_API_INTERN unsigned int internal_get_tid(void)
{
  const unsigned int nthreads = LIBXSMM_ATOMIC_ADD_FETCH(&libxsmm_thread_count, 1, LIBXSMM_ATOMIC_RELAXED);
#if !defined(NDEBUG)
  static int error_once = 0;
  if (LIBXSMM_NTHREADS_MAX < nthreads
    && 0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: maximum number of threads is exhausted!\n");
  }
#endif
  LIBXSMM_ASSERT(LIBXSMM_ISPOT(LIBXSMM_NTHREADS_MAX));
  return LIBXSMM_MOD2(nthreads - 1, LIBXSMM_NTHREADS_MAX);
}


LIBXSMM_API unsigned int libxsmm_get_tid(void)
{
#if (0 != LIBXSMM_SYNC)
# if defined(_OPENMP) && defined(LIBXSMM_SYNC_OMP)
  return (unsigned int)omp_get_thread_num();
# else
  static LIBXSMM_TLS unsigned int tid = 0xFFFFFFFF;
  if (0xFFFFFFFF == tid) tid = internal_get_tid();
  return tid;
# endif
#else
  return 0;
#endif
}
