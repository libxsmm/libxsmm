/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Kunal Banerjee (Intel Corp.), Dheevatsa Mudigere (Intel Corp.)
   Alexander Heinecke (Intel Corp.), Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm_bgemm.h>
#include "libxsmm_main.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdio.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXSMM_BGEMM_BARRIER)
/*# define LIBXSMM_BGEMM_BARRIER*/
#endif


LIBXSMM_API_DEFINITION void libxsmm_bgemm_omp(const libxsmm_bgemm_handle* handle,
  const void* a, const void* b, void* c, /*unsigned*/int count)
{
  static int error_once = 0;
  if (0 < count) {
    if (0 != a && 0 != b && 0 != c) {
#if !defined(_OPENMP)
      const int nthreads = 1;
#else
      const int nthreads = omp_get_max_threads();
# if defined(LIBXSMM_BGEMM_BARRIER)
      libxsmm_barrier* barrier = 0;
      /* make an informed guess about the number of threads per core */
      if (224 <= nthreads
#   if !defined(__MIC__)
        && LIBXSMM_X86_AVX512_MIC <= libxsmm_target_archid
        && LIBXSMM_X86_AVX512_CORE > libxsmm_target_archid
#   endif
        )
      {
        barrier = libxsmm_barrier_create(nthreads / 4, 4);
      }
      else
      {
        barrier = libxsmm_barrier_create(nthreads / 2, 2);
      }
# endif /*defined(LIBXSMM_BGEMM_BARRIER)*/
#     pragma omp parallel
#endif /*defined(_OPENMP)*/
      {
        int tid = 0, i;
#if defined(_OPENMP)
        tid = omp_get_thread_num();
#endif
        assert(tid < nthreads);
#if defined(LIBXSMM_BGEMM_BARRIER)
        libxsmm_barrier_init(barrier, tid);
#endif
        for (i = 0; i < count; ++i) {
          libxsmm_bgemm(handle, a, b, c, tid, nthreads);
#if defined(LIBXSMM_BGEMM_BARRIER)
          libxsmm_barrier_wait(barrier, tid);
#elif defined(_OPENMP)
#         pragma omp barrier
#endif
        }
      }
#if defined(LIBXSMM_BGEMM_BARRIER)
      libxsmm_barrier_release(barrier);
#endif
    }
    else if (0 != libxsmm_verbosity /* library code is expected to be mute */
          && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: BGEMM matrix-operands cannot be NULL!\n");
    }
  }
  else if (0 > count && 0 != libxsmm_verbosity /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: BGEMM count-argument cannot be negative!\n");
  }
}

